"""Integration and unit tests for the podcast pipeline."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import pytest
from typer.testing import CliRunner

from podcast_agent.cli.app import app
from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import (
    ClaimAssessment,
    EpisodeOutput,
    EpisodePlan,
    EpisodeSegment,
    EpisodeScript,
    GroundingReport,
    GroundingStatus,
    RenderManifest,
    RenderSegment,
    SegmentRepairResult,
    ScriptClaim,
)
from podcast_agent.tts import TTSClient


BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory. The historian argues that the harbor served as a relay for scholars, not soldiers.

Chapter 2: Signals
Inside the city archive, the team finds coded letters describing the observatory's mirrors and the way they coordinated messages across valleys. The engineer explains that the mirror network only worked because local guilds kept the routes maintained.

Chapter 3: Fracture
The discovery creates tension among the sponsors, who want the observatory framed as a military outpost. The historian pushes back and points to logs showing that teachers, navigators, and astronomers shared the site.

Chapter 4: Legacy
By the end of the journey, the team understands that the observatory connected communities through knowledge exchange. The final journal entries argue that its legacy was cooperation, memory, and scientific practice.
"""


class FakeTTSClient(TTSClient):
    """Simple deterministic TTS stub for pipeline tests."""

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del instructions
        return f"{voice or 'default'}::{audio_format or 'mp3'}::{text}".encode("utf-8")


class DelayedTTSClient(TTSClient):
    """TTS stub that finishes segments out of order."""

    def __init__(self, delays: dict[str, float]) -> None:
        super().__init__()
        self.delays = delays

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del voice, audio_format, instructions
        time.sleep(self.delays[text])
        return text.encode("utf-8")


class RetryingTTSClient(TTSClient):
    """TTS stub with per-text transient or permanent failures."""

    def __init__(self, failures: dict[str, int]) -> None:
        super().__init__()
        self.failures = failures
        self.call_counts: dict[str, int] = {}

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del voice, audio_format, instructions
        self.call_counts[text] = self.call_counts.get(text, 0) + 1
        if self.call_counts[text] <= self.failures.get(text, 0):
            raise RuntimeError(f"planned failure for {text}")
        return text.encode("utf-8")


class ConcurrencyTrackingTTSClient(TTSClient):
    """TTS stub that records peak concurrent synthesize calls."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del text, voice, audio_format, instructions
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
            return b"audio"
        finally:
            with self._lock:
                self.active_calls -= 1


def _build_render_manifest(episode_id: str, segment_texts: list[str]) -> RenderManifest:
    return RenderManifest(
        episode_id=episode_id,
        title=f"Episode {episode_id}",
        narrator="Narrator",
        segments=[
            RenderSegment(
                segment_id=f"{episode_id}-segment-{index}",
                speaker="Narrator",
                text=text,
                ssml=f"<speak>{text}</speak>",
                grounded_claim_ids=[f"claim-{index}"],
            )
            for index, text in enumerate(segment_texts, start=1)
        ],
    )


def _build_audio_orchestrator(
    tmp_path: Path,
    tts: TTSClient,
    *,
    audio_parallelism: int = 4,
    audio_retry_attempts: int = 2,
) -> PipelineOrchestrator:
    settings = Settings()
    settings = settings.model_copy(
        update={
            "pipeline": settings.pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "audio_parallelism": audio_parallelism,
                    "audio_retry_attempts": audio_retry_attempts,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        tts=tts,
        settings=settings,
    )
    orchestrator.current_book_id = "observatory-book"
    orchestrator.run_id = "audio-run"
    orchestrator.run_logger.bind_run(orchestrator.run_id)
    return orchestrator


def _write_episode_output_artifact(path: Path, manifest: RenderManifest) -> None:
    payload = {
        "plan": {
            "episode_id": manifest.episode_id,
            "sequence": 1,
            "title": manifest.title,
            "synopsis": "Original episode output.",
            "chapter_ids": [],
            "themes": [],
        },
        "script": {
            "episode_id": manifest.episode_id,
            "title": manifest.title,
            "narrator": manifest.narrator,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "beat_id": f"{segment.segment_id}-beat",
                    "heading": "Heading",
                    "narration": segment.text,
                    "claims": [
                        {
                            "claim_id": claim_id,
                            "text": segment.text,
                            "evidence_chunk_ids": ["chunk-1"],
                        }
                        for claim_id in segment.grounded_claim_ids
                    ],
                    "citations": ["chunk-1"],
                }
                for segment in manifest.segments
            ],
        },
        "report": {
            "episode_id": manifest.episode_id,
            "overall_status": "pass",
            "claim_assessments": [
                {
                    "claim_id": claim_id,
                    "status": "grounded",
                    "reason": "Grounded in source material.",
                    "evidence_chunk_ids": ["chunk-1"],
                }
                for segment in manifest.segments
                for claim_id in segment.grounded_claim_ids
            ],
        },
        "manifest": manifest.model_dump(mode="json"),
        "audio_manifest": None,
        "repair_attempts": [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_pipeline_creates_multi_chapter_episode_and_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=1)

    assert result["series_plan"]["episodes"]
    assert len(result["series_plan"]["episodes"]) == 1
    assert len(result["series_plan"]["episodes"][0]["chapter_ids"]) == 4
    assert result["episodes"][0]["manifest"]["segments"]
    episode_output_path = (
        tmp_path
        / "runs"
        / orchestrator.run_id
        / "observatory-book"
        / result["episodes"][0]["plan"]["episode_id"]
        / "episode_output.json"
    )
    assert episode_output_path.exists()
    assert not episode_output_path.with_name("script.json").exists()
    assert not episode_output_path.with_name("grounding_report.json").exists()
    assert not episode_output_path.with_name("render_manifest.json").exists()


def test_index_book_persists_chunks_and_embeddings(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    repository = InMemoryRepository()
    orchestrator = PipelineOrchestrator(repository=repository, llm=HeuristicLLMClient())

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion)

    assert structure.book_id in repository.structures
    assert structure.book_id in repository.embeddings
    assert len(repository.structures[structure.book_id].chunks) == len(repository.embeddings[structure.book_id])


def test_index_book_can_end_at_matching_chapter_title(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    repository = InMemoryRepository()
    settings = Settings()
    settings = settings.model_copy(
        update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})}
    )
    orchestrator = PipelineOrchestrator(
        repository=repository,
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion, end_chapter="chapter 2: signals")

    assert len(structure.chapters) == 2
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2]
    assert all(chunk.chapter_number in {1, 2} for chunk in structure.chunks)
    assert len(repository.embeddings[structure.book_id]) == len(structure.chunks)
    run_log = (tmp_path / "runs" / orchestrator.run_id / "run.log").read_text(encoding="utf-8")
    assert '"event_type": "structuring_chapter_started"' in run_log
    assert '"chapter_number": 3' not in run_log


def test_index_book_can_start_from_matching_chapter_title(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(
        update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})}
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion, start_chapter="chapter 2: signals")

    assert [chapter.title for chapter in structure.chapters] == [
        "Chapter 2: Signals",
        "Chapter 3: Fracture",
        "Chapter 4: Legacy",
    ]
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2, 3]


def test_index_book_can_apply_inclusive_chapter_range(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(
        update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})}
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(
        ingestion,
        start_chapter="chapter 2: signals",
        end_chapter="chapter 3: fracture",
    )

    assert [chapter.title for chapter in structure.chapters] == [
        "Chapter 2: Signals",
        "Chapter 3: Fracture",
    ]
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2]


def test_index_book_raises_for_unknown_start_chapter(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")

    with pytest.raises(ValueError, match="Unable to find start chapter"):
        orchestrator.index_book(ingestion, start_chapter="Chapter 9: Missing")


def test_index_book_raises_for_unknown_end_chapter(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")

    with pytest.raises(ValueError, match="Unable to find end chapter"):
        orchestrator.index_book(ingestion, end_chapter="Chapter 9: Missing")


def test_index_book_raises_when_end_precedes_start(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")

    with pytest.raises(ValueError, match="appears before start chapter"):
        orchestrator.index_book(
            ingestion,
            start_chapter="Chapter 3: Fracture",
            end_chapter="Chapter 2: Signals",
        )


def test_pipeline_defaults_raise_parallelism() -> None:
    settings = Settings()

    assert settings.pipeline.episode_parallelism == 3
    assert settings.pipeline.audio_parallelism == 8
    assert settings.pipeline.audio_retry_attempts == 2
    assert settings.pipeline.beat_parallelism == 4
    assert settings.pipeline.beat_write_timeout_seconds == 120.0
    assert settings.pipeline.grounding_parallelism == 3
    assert settings.pipeline.structuring_parallelism == 3
    assert settings.pipeline.max_episode_minutes == 360
    assert settings.pipeline.max_structuring_llm_chapter_words == 42232


def test_ingest_book_reads_pdf_source_and_persists_artifact(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.read_source_text",
        lambda path: "[Page 1]\nChapter 1: Arrival\nThe observatory appears.",
    )

    ingestion = orchestrator.ingest_book(pdf_path, title="Observatory Book", author="A. Writer")

    assert ingestion.source_type.value == "pdf"
    assert ingestion.raw_text.startswith("[Page 1]")
    artifact_path = tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "ingestion.json"
    assert artifact_path.exists()
    artifact_payload = artifact_path.read_text(encoding="utf-8")
    assert '"source_type": "pdf"' in artifact_payload
    assert '"raw_text": "[Page 1]\\nChapter 1: Arrival\\nThe observatory appears."' in artifact_payload


def test_validation_schema_is_claim_level(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion)
    _, plan = orchestrator.plan_episodes(structure, episode_count=2)
    script = orchestrator.write_episode(structure.book_id, plan.episodes[0])
    report = orchestrator.validate_episode(structure.book_id, script)

    assert report.claim_assessments
    assert all(assessment.claim_id for assessment in report.claim_assessments)


def test_books_under_source_word_floor_collapse_to_single_episode() -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(
        Path("examples/river_of_hours.txt"),
        title="River of Hours",
        author="Sample Author",
        episode_count=1,
    )

    assert len(result["series_plan"]["episodes"]) == 1
    assert len(result["series_plan"]["episodes"][0]["chapter_ids"]) == 12


def test_river_of_hours_text_indexes_same_chapter_count(tmp_path: Path) -> None:
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    text_path = Path("tests/fixtures/river_of_hours.txt")
    ingestion = orchestrator.ingest_book(text_path, title="River of Hours", author="Sample Author")
    structure = orchestrator.index_book(ingestion)

    assert ingestion.source_type.value == "text"
    assert len(structure.chapters) == 12
    assert structure.chapters[0].title == "Chapter 1: On Beginnings"
    assert structure.chapters[-1].title == "Chapter 12: A Practical Metaphysics"


def test_pipeline_can_synthesize_audio_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        tts=FakeTTSClient(),
    )

    result = orchestrator.run_pipeline(
        book_path,
        title="Observatory Book",
        author="A. Writer",
        episode_count=1,
        synthesize_audio=True,
    )

    audio_manifest = result["episodes"][0]["audio_manifest"]
    assert audio_manifest is not None
    assert audio_manifest["segments"]
    assert audio_manifest["audio_path"].endswith(".mp3")
    assert Path(audio_manifest["audio_path"]).exists()


def test_synthesize_audio_preserves_manifest_order_when_parallel(tmp_path: Path) -> None:
    orchestrator = _build_audio_orchestrator(
        tmp_path,
        DelayedTTSClient({"first": 0.03, "second": 0.01, "third": 0.02}),
        audio_parallelism=3,
    )
    manifest = _build_render_manifest("episode-1", ["first", "second", "third"])

    audio_manifest = orchestrator.synthesize_audio(manifest)

    assert [segment["text"] for segment in audio_manifest.model_dump(mode="json")["segments"]] == [
        "first",
        "second",
        "third",
    ]
    assert Path(audio_manifest.audio_path).read_bytes() == b"firstsecondthird"


def test_synthesize_audio_retries_failed_segment_then_succeeds(tmp_path: Path) -> None:
    tts = RetryingTTSClient({"retry-me": 1})
    orchestrator = _build_audio_orchestrator(tmp_path, tts, audio_parallelism=2, audio_retry_attempts=2)
    manifest = _build_render_manifest("episode-1", ["retry-me", "steady"])

    audio_manifest = orchestrator.synthesize_audio(manifest)

    assert tts.call_counts["retry-me"] == 2
    assert tts.call_counts["steady"] == 1
    assert Path(audio_manifest.audio_path).read_bytes() == b"retry-mesteady"


def test_synthesize_audio_fails_episode_after_retries_and_skips_artifact(tmp_path: Path) -> None:
    tts = RetryingTTSClient({"always-fail": 3})
    orchestrator = _build_audio_orchestrator(tmp_path, tts, audio_parallelism=2, audio_retry_attempts=2)
    manifest = _build_render_manifest("episode-1", ["always-fail", "steady"])
    expected_audio_path = tmp_path / "runs" / "audio-run" / "observatory-book" / "episode-1" / "episode-1.mp3"

    try:
        orchestrator.synthesize_audio(manifest)
    except RuntimeError as exc:
        assert "episode-1-segment-1" in str(exc)
    else:
        raise AssertionError("Expected audio synthesis to fail after exhausting retries")

    assert tts.call_counts["always-fail"] == 3
    assert not expected_audio_path.exists()


def test_synthesize_audio_uses_global_parallelism_cap_across_episodes(tmp_path: Path) -> None:
    tts = ConcurrencyTrackingTTSClient()
    orchestrator = _build_audio_orchestrator(tmp_path, tts, audio_parallelism=2)
    manifests = [
        _build_render_manifest("episode-1", ["a1", "a2", "a3"]),
        _build_render_manifest("episode-2", ["b1", "b2", "b3"]),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(orchestrator.synthesize_audio, manifest) for manifest in manifests]
        for future in futures:
            future.result()

    assert tts.max_active_calls == 2


def test_regenerate_audio_from_episode_output_artifact(tmp_path: Path) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["first line", "second line"])
    _write_episode_output_artifact(source_path, manifest)

    result = orchestrator.regenerate_audio_from_artifact(source_path)

    assert result.manifest is not None
    assert result.manifest.model_dump(mode="json") == manifest.model_dump(mode="json")
    assert result.audio_manifest is not None
    assert Path(result.audio_manifest.audio_path).exists()
    persisted_output = (
        tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "episode-1" / "episode_output.json"
    )
    assert persisted_output.exists()
    persisted_payload = json.loads(persisted_output.read_text(encoding="utf-8"))
    assert persisted_payload["manifest"]["episode_id"] == "episode-1"
    assert persisted_payload["audio_manifest"]["audio_path"].endswith("episode-1.mp3")


def test_regenerate_audio_from_standalone_manifest_artifact(tmp_path: Path) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "external-manifests" / "render_manifest.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["alpha", "beta"])
    source_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = orchestrator.regenerate_audio_from_artifact(source_path, book_id="observatory-book")

    assert result.audio_manifest is not None
    assert Path(result.audio_manifest.audio_path).read_bytes() == b"ballad::mp3::alphaballad::mp3::beta"
    assert result.plan.synopsis == "Audio regenerated directly from an existing render manifest."


def test_regenerate_audio_requires_valid_manifest_payload(tmp_path: Path) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({"manifest": {"episode_id": "episode-1"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a valid render manifest"):
        orchestrator.regenerate_audio_from_artifact(source_path)


def test_regenerate_audio_does_not_invoke_upstream_pipeline_stages(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["first", "second"])
    _write_episode_output_artifact(source_path, manifest)

    for method_name in ("ingest_book", "index_book", "plan_episodes", "write_episode", "validate_episode", "repair_episode"):
        monkeypatch.setattr(
            orchestrator,
            method_name,
            lambda *args, _method_name=method_name, **kwargs: pytest.fail(f"{_method_name} should not be called"),
        )

    result = orchestrator.regenerate_audio_from_artifact(source_path)

    assert result.audio_manifest is not None


def test_repair_episode_merges_only_failed_segments_and_logs_diffs(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    original_script = EpisodeScript(
        episode_id="episode-1",
        title="Episode 1",
        narrator="Narrator",
        segments=[
            EpisodeSegment(
                segment_id="episode-1-segment-1",
                beat_id="episode-1-beat-1",
                heading="Opening",
                narration="Original failed narration",
                claims=[ScriptClaim(claim_id="claim-1", text="Claim one", evidence_chunk_ids=["chunk-1"])],
                citations=["chunk-1"],
            ),
            EpisodeSegment(
                segment_id="episode-1-segment-2",
                beat_id="episode-1-beat-2",
                heading="Stable",
                narration="Untouched narration",
                claims=[ScriptClaim(claim_id="claim-1", text="Claim two", evidence_chunk_ids=["chunk-2"])],
                citations=["chunk-2"],
            ),
        ],
    )
    failed_report = GroundingReport(
        episode_id="episode-1",
        overall_status="fail",
        claim_assessments=[
                ClaimAssessment(
                    claim_id="claim-1",
                    status=GroundingStatus.UNSUPPORTED,
                    reason="Unsupported claim.",
                    evidence_chunk_ids=["chunk-1"],
                ),
                ClaimAssessment(
                    claim_id="claim-1",
                    status=GroundingStatus.GROUNDED,
                    reason="Grounded claim.",
                    evidence_chunk_ids=["chunk-2"],
                ),
        ],
    )
    repaired_segment = original_script.segments[0].model_copy(
        update={
            "narration": "Repaired narration",
            "claims": [ScriptClaim(claim_id="claim-1", text="Repaired claim", evidence_chunk_ids=["chunk-1"])],
        }
    )

    monkeypatch.setattr(
        orchestrator.repair_agent,
        "repair",
        lambda script, report, attempt: SegmentRepairResult(
            episode_id=script.episode_id,
            attempt=attempt,
            repaired_segment_ids=[repaired_segment.segment_id],
            repaired_segments=[repaired_segment],
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_episode",
        lambda book_id, script: GroundingReport(
            episode_id=script.episode_id,
            overall_status="pass",
            claim_assessments=[
                ClaimAssessment(
                    claim_id="claim-1",
                    status=GroundingStatus.GROUNDED,
                    reason="Grounded after repair.",
                    evidence_chunk_ids=["chunk-1"],
                ),
                ClaimAssessment(
                    claim_id="claim-1",
                    status=GroundingStatus.GROUNDED,
                    reason="Still grounded.",
                    evidence_chunk_ids=["chunk-2"],
                ),
            ],
        ),
    )

    repair = orchestrator.repair_episode("observatory-book", original_script, failed_report)

    assert repair.repaired_segment_ids == ["episode-1-segment-1"]
    assert repair.script.segments[0].narration == "Repaired narration"
    assert repair.script.segments[1].narration == "Untouched narration"
    repair_artifact = tmp_path / "runs" / "audio-run" / "observatory-book" / "episode-1" / "repair_attempt_1.json"
    assert repair_artifact.exists()
    run_log = (tmp_path / "runs" / "audio-run" / "run.log").read_text(encoding="utf-8")
    assert '"event_type": "repair_payload_diagnostics"' in run_log
    assert '"event_type": "repair_segment_diff"' in run_log
    assert '"event_type": "repair_merge_summary"' in run_log


def test_run_pipeline_can_end_at_matching_chapter_title(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(
        book_path,
        title="Observatory Book",
        author="A. Writer",
        end_chapter="chapter 2: signals",
        episode_count=1,
    )

    planned_chapters = result["series_plan"]["episodes"][0]["chapter_ids"]
    assert len(planned_chapters) == 2
    structure_path = tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "structure.json"
    assert structure_path.exists()
    structure_payload = structure_path.read_text(encoding="utf-8")
    assert '"chapter_number": 3' not in structure_payload


def test_run_pipeline_can_start_from_matching_chapter_title(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(
        book_path,
        title="Observatory Book",
        author="A. Writer",
        start_chapter="chapter 2: signals",
        episode_count=1,
    )

    planned_chapters = result["series_plan"]["episodes"][0]["chapter_ids"]
    assert len(planned_chapters) == 3
    structure_path = tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "structure.json"
    assert structure_path.exists()
    structure_payload = structure_path.read_text(encoding="utf-8")
    assert '"title": "Chapter 2: Signals"' in structure_payload
    assert '"title": "Chapter 1: Arrival"' not in structure_payload


def test_run_pipeline_can_apply_inclusive_chapter_range(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(
        book_path,
        title="Observatory Book",
        author="A. Writer",
        start_chapter="chapter 2: signals",
        end_chapter="chapter 3: fracture",
        episode_count=1,
    )

    planned_chapters = result["series_plan"]["episodes"][0]["chapter_ids"]
    assert len(planned_chapters) == 2
    structure_path = tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "structure.json"
    assert structure_path.exists()
    structure_payload = structure_path.read_text(encoding="utf-8")
    assert '"title": "Chapter 2: Signals"' in structure_payload
    assert '"title": "Chapter 3: Fracture"' in structure_payload
    assert '"title": "Chapter 4: Legacy"' not in structure_payload


def test_run_pipeline_processes_episodes_in_parallel_when_enabled(tmp_path: Path, monkeypatch) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(
        update={
            "pipeline": settings.pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "minimum_source_words_per_episode": 50,
                    "episode_parallelism": 2,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )
    barrier = threading.Barrier(2)
    thread_names: list[str] = []
    lock = threading.Lock()

    def fake_run_episode_plan(book_id: str, episode_plan: EpisodePlan, *, synthesize_audio: bool) -> EpisodeOutput:
        del book_id, synthesize_audio
        with lock:
            thread_names.append(threading.current_thread().name)
        barrier.wait(timeout=1)
        script = EpisodeScript(
            episode_id=episode_plan.episode_id,
            title=episode_plan.title,
            narrator="Narrator",
            segments=[],
        )
        report = GroundingReport(
            episode_id=episode_plan.episode_id,
            overall_status="pass",
            claim_assessments=[],
        )
        return EpisodeOutput(plan=episode_plan, script=script, report=report)

    monkeypatch.setattr(orchestrator, "_run_episode_plan", fake_run_episode_plan)

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=2)

    assert len(result["series_plan"]["episodes"]) == 2
    assert result["episodes"][0]["plan"]["sequence"] == 1
    assert result["episodes"][1]["plan"]["sequence"] == 2
    assert len(set(thread_names)) == 2


def test_run_pipeline_requires_episode_count_cli(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["run-pipeline", str(book_path), "--title", "Observatory Book", "--author", "A. Writer"],
    )

    assert result.exit_code != 0
    assert "Missing option '--episode-count'" in result.output


def test_render_audio_from_manifest_cli_uses_saved_episode_output(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["cli first", "cli second"])
    _write_episode_output_artifact(source_path, manifest)

    monkeypatch.setattr("podcast_agent.cli.app._build_orchestrator", lambda database_url: orchestrator)

    result = CliRunner().invoke(app, ["render-audio-from-manifest", str(source_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["audio_manifest"]["audio_path"].endswith("episode-1.mp3")
    assert payload["manifest"]["episode_id"] == "episode-1"
