"""Integration and unit tests for the podcast pipeline."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
import time
import wave

import pytest
from typer.testing import CliRunner

from podcast_agent.cli.app import app
from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import EpisodePreparation, PipelineOrchestrator
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
    SpokenEpisodeNarration,
    SpokenNarrationChunk,
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


def _build_wav_bytes(*, framerate: int = 16000, frame_count: int = 200) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(framerate)
        wav_handle.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


class WavTTSClient(TTSClient):
    """TTS stub that returns valid WAV bytes per segment."""

    def __init__(self, sample_rates: dict[str, int] | None = None) -> None:
        super().__init__()
        self.sample_rates = sample_rates or {}

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del voice, audio_format, instructions
        framerate = self.sample_rates.get(text, 16000)
        return _build_wav_bytes(framerate=framerate)


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


class WavConcurrencyTrackingTTSClient(TTSClient):
    """WAV-producing TTS stub that records peak concurrent synthesize calls."""

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
            return _build_wav_bytes()
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
    tts_provider: str | None = None,
    tts_audio_format: str | None = None,
    tts_kokoro_parallelism: int | None = None,
    tts_kokoro_chunk_min_words: int | None = None,
    tts_kokoro_chunk_max_words: int | None = None,
) -> PipelineOrchestrator:
    settings = Settings()
    tts_updates: dict[str, str | int] = {}
    if tts_provider is not None:
        tts_updates["provider"] = tts_provider
    if tts_audio_format is not None:
        tts_updates["audio_format"] = tts_audio_format
    if tts_kokoro_parallelism is not None:
        tts_updates["kokoro_parallelism"] = tts_kokoro_parallelism
    if tts_kokoro_chunk_min_words is not None:
        tts_updates["kokoro_chunk_min_words"] = tts_kokoro_chunk_min_words
    if tts_kokoro_chunk_max_words is not None:
        tts_updates["kokoro_chunk_max_words"] = tts_kokoro_chunk_max_words
    settings = settings.model_copy(
        update={
            "pipeline": settings.pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "audio_parallelism": audio_parallelism,
                    "audio_retry_attempts": audio_retry_attempts,
                }
            ),
            "tts": settings.tts.model_copy(update=tts_updates) if tts_updates else settings.tts,
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
    _write_spoken_script_artifact(path.parent, manifest)


def _write_spoken_script_artifact(episode_dir: Path, manifest: RenderManifest) -> None:
    chunks = [
        SpokenNarrationChunk(
            chunk_id=segment.segment_id,
            text=segment.text,
            word_count=len(segment.text.split()),
        )
        for segment in manifest.segments
    ]
    spoken_script = SpokenEpisodeNarration(
        episode_id=manifest.episode_id,
        title=manifest.title,
        narrator=manifest.narrator,
        narration=" ".join(segment.text for segment in manifest.segments),
        chunks=chunks,
    )
    (episode_dir / "spoken_script.json").write_text(
        spoken_script.model_dump_json(indent=2),
        encoding="utf-8",
    )


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
    assert episode_output_path.with_name("factual_script.json").exists()
    assert episode_output_path.with_name("spoken_script.json").exists()
    assert episode_output_path.with_name("spoken_delivery.json").exists()
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
    assert settings.pipeline.batch_parallelism == 3
    assert settings.pipeline.audio_parallelism == 6
    assert settings.pipeline.audio_retry_attempts == 2
    assert settings.pipeline.beat_parallelism == 6
    assert settings.pipeline.beat_write_retry_attempts == 2
    assert settings.pipeline.beat_write_timeout_seconds == 180.0
    assert settings.pipeline.grounding_parallelism == 5
    assert settings.pipeline.structuring_parallelism == 6
    assert settings.pipeline.max_episode_minutes == 360
    assert settings.pipeline.max_structuring_llm_chapter_words == 75000
    assert settings.pipeline.min_episode_source_ratio == 0.3
    assert settings.spoken_delivery.enabled is True
    assert settings.spoken_delivery.timeout_seconds == 1200.0
    assert settings.spoken_delivery.chunk_min_words == 700
    assert settings.spoken_delivery.chunk_max_words == 900
    assert settings.tts.kokoro_parallelism == 2
    assert settings.tts.kokoro_worker_threads == 4
    assert settings.tts.kokoro_chunk_min_words == 550
    assert settings.tts.kokoro_chunk_max_words == 600


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


def test_synthesize_audio_kokoro_converts_wav_to_mp3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tts = WavTTSClient()
    orchestrator = _build_audio_orchestrator(
        tmp_path,
        tts,
        tts_provider="kokoro",
        tts_audio_format="mp3",
    )
    manifest = _build_render_manifest("episode-1", ["alpha", "beta"])

    def fake_convert(_self: PipelineOrchestrator, wav_bytes: bytes) -> bytes:
        assert wav_bytes[:4] == b"RIFF"
        return b"mp3-bytes"

    monkeypatch.setattr(PipelineOrchestrator, "_convert_wav_to_mp3_bytes", fake_convert)

    audio_manifest = orchestrator.synthesize_audio(manifest)

    assert audio_manifest.audio_path.endswith(".mp3")
    assert Path(audio_manifest.audio_path).read_bytes() == b"mp3-bytes"


def test_synthesize_audio_kokoro_logs_merge_diagnostics_on_failure(tmp_path: Path) -> None:
    tts = WavTTSClient({"alpha": 16000, "beta": 22050})
    orchestrator = _build_audio_orchestrator(
        tmp_path,
        tts,
        tts_provider="kokoro",
        tts_audio_format="mp3",
    )
    manifest = _build_render_manifest("episode-1", ["alpha", "beta"])

    with pytest.raises(RuntimeError):
        orchestrator.synthesize_audio(manifest)

    run_log_path = tmp_path / "runs" / "audio-run" / "run.log"
    events = [json.loads(line) for line in run_log_path.read_text(encoding="utf-8").splitlines()]
    merge_events = [event for event in events if event["event_type"] == "audio_merge_failed"]
    assert merge_events
    payload = merge_events[-1]["payload"]
    assert payload["episode_id"] == manifest.episode_id
    assert payload["provider"] == "kokoro"
    assert payload["diagnostics"]


def test_synthesize_audio_kokoro_caps_parallelism_at_two_workers(tmp_path: Path) -> None:
    tts = WavConcurrencyTrackingTTSClient()
    orchestrator = _build_audio_orchestrator(
        tmp_path,
        tts,
        audio_parallelism=4,
        tts_provider="kokoro",
        tts_audio_format="wav",
        tts_kokoro_parallelism=2,
    )
    manifest = _build_render_manifest("episode-1", ["first", "second", "third", "fourth"])

    orchestrator.synthesize_audio(manifest)

    assert tts.max_active_calls == 2


def test_synthesize_audio_kokoro_rechunks_to_configured_max_words(tmp_path: Path) -> None:
    tts = WavTTSClient()
    orchestrator = _build_audio_orchestrator(
        tmp_path,
        tts,
        tts_provider="kokoro",
        tts_audio_format="wav",
        tts_kokoro_chunk_max_words=450,
    )
    long_text = "word " * 1201
    manifest = _build_render_manifest("episode-1", [long_text])

    audio_manifest = orchestrator.synthesize_audio(manifest)

    assert len(audio_manifest.segments) == 3
    word_counts = [len(segment.text.split()) for segment in audio_manifest.segments]
    assert word_counts == [450, 450, 301]
    with wave.open(str(Path(audio_manifest.audio_path))) as wav_handle:
        assert wav_handle.getnframes() == 200 * 3
        assert wav_handle.getframerate() == 16000
        assert wav_handle.getnchannels() == 1


def test_orchestrator_uses_kokoro_specific_spoken_delivery_chunk_bounds() -> None:
    settings = Settings().model_copy(
        update={
            "tts": Settings().tts.model_copy(update={"provider": "kokoro"}),
        }
    )

    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    assert orchestrator.spoken_delivery_agent.chunk_min_words == settings.tts.kokoro_chunk_min_words
    assert orchestrator.spoken_delivery_agent.chunk_max_words == settings.tts.kokoro_chunk_max_words


def test_orchestrator_keeps_default_spoken_delivery_chunk_bounds_for_non_kokoro() -> None:
    settings = Settings()

    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    assert orchestrator.spoken_delivery_agent.chunk_min_words == settings.spoken_delivery.chunk_min_words
    assert orchestrator.spoken_delivery_agent.chunk_max_words == settings.spoken_delivery.chunk_max_words


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
    assert [segment.text for segment in result.manifest.segments] == ["first line", "second line"]
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
    _write_spoken_script_artifact(source_path.parent, manifest)

    result = orchestrator.regenerate_audio_from_artifact(source_path, book_id="observatory-book")

    assert result.audio_manifest is not None
    assert Path(result.audio_manifest.audio_path).read_bytes() == b"ballad::mp3::alphaballad::mp3::beta"


def test_regenerate_audio_requires_valid_manifest_payload(tmp_path: Path) -> None:
    orchestrator = _build_audio_orchestrator(tmp_path, FakeTTSClient())
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({"manifest": {"episode_id": "episode-1"}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="spoken_script.json not found"):
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

    def fake_prepare_episode_plan(book_id: str, episode_plan: EpisodePlan) -> EpisodePreparation:
        del book_id
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
        return EpisodePreparation(
            plan=episode_plan,
            script=script,
            report=report,
            repair_attempts=[],
        )

    def fake_finalize_episode_plan(
        book_id: str,
        preparation: EpisodePreparation,
        *,
        synthesize_audio: bool,
        allow_downstream: bool,
        previous_spoken_script: SpokenEpisodeNarration | None,
        next_plan: EpisodePlan | None,
        next_script: EpisodeScript | None,
    ) -> tuple[EpisodeOutput, SpokenEpisodeNarration | None]:
        del book_id, synthesize_audio, allow_downstream, previous_spoken_script, next_plan, next_script
        return EpisodeOutput(
            plan=preparation.plan,
            script=preparation.script,
            report=preparation.report,
        ), None

    monkeypatch.setattr(orchestrator, "_prepare_episode_plan", fake_prepare_episode_plan)
    monkeypatch.setattr(orchestrator, "_finalize_episode_plan", fake_finalize_episode_plan)

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=2)

    assert len(result["series_plan"]["episodes"]) == 2
    assert result["episodes"][0]["plan"]["sequence"] == 1
    assert result["episodes"][1]["plan"]["sequence"] == 2
    assert len(set(thread_names)) == 2


def test_run_pipeline_defers_spoken_delivery_until_grounding_barrier(tmp_path: Path, monkeypatch) -> None:
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
    prepared_count = 0
    prepared_lock = threading.Lock()
    all_prepared = threading.Event()

    def fake_prepare_episode_plan(book_id: str, episode_plan: EpisodePlan) -> EpisodePreparation:
        nonlocal prepared_count
        del book_id
        with prepared_lock:
            prepared_count += 1
            if prepared_count == 2:
                all_prepared.set()
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
        return EpisodePreparation(
            plan=episode_plan,
            script=script,
            report=report,
            repair_attempts=[],
        )

    def fake_spoken_delivery_episode(book_id: str, script: EpisodeScript):
        del book_id, script
        assert all_prepared.is_set()
        return None, None, None

    def fake_finalize_episode_plan(
        book_id: str,
        preparation: EpisodePreparation,
        *,
        synthesize_audio: bool,
        allow_downstream: bool,
        previous_spoken_script: SpokenEpisodeNarration | None,
        next_plan: EpisodePlan | None,
        next_script: EpisodeScript | None,
    ) -> tuple[EpisodeOutput, SpokenEpisodeNarration | None]:
        del synthesize_audio, previous_spoken_script, next_plan, next_script
        if allow_downstream:
            orchestrator.spoken_delivery_episode(book_id, preparation.script)
        return EpisodeOutput(
            plan=preparation.plan,
            script=preparation.script,
            report=preparation.report,
        ), None

    monkeypatch.setattr(orchestrator, "_prepare_episode_plan", fake_prepare_episode_plan)
    monkeypatch.setattr(orchestrator, "_finalize_episode_plan", fake_finalize_episode_plan)
    monkeypatch.setattr(orchestrator, "spoken_delivery_episode", fake_spoken_delivery_episode)

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=2)

    assert len(result["episodes"]) == 2


def test_framing_failure_skips_manifest(tmp_path: Path, monkeypatch) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "minimum_source_words_per_episode": 50,
                    "episode_parallelism": 1,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    def fail_framing(*args, **kwargs):
        raise RuntimeError("framing failed")

    monkeypatch.setattr(orchestrator, "_build_episode_framing", fail_framing)

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=1)

    assert result["episodes"][0]["manifest"] is None


def test_run_pipeline_skips_downstream_when_any_episode_fails_grounding(tmp_path: Path, monkeypatch) -> None:
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

    def fake_prepare_episode_plan(book_id: str, episode_plan: EpisodePlan) -> EpisodePreparation:
        del book_id
        script = EpisodeScript(
            episode_id=episode_plan.episode_id,
            title=episode_plan.title,
            narrator="Narrator",
            segments=[],
        )
        report = GroundingReport(
            episode_id=episode_plan.episode_id,
            overall_status="fail" if episode_plan.sequence == 2 else "pass",
            claim_assessments=[],
        )
        return EpisodePreparation(
            plan=episode_plan,
            script=script,
            report=report,
            repair_attempts=[],
        )

    def forbidden_spoken_delivery_episode(book_id: str, script: EpisodeScript):
        del book_id, script
        raise AssertionError("spoken delivery should be skipped when grounding fails")

    def forbidden_render_manifest(script: EpisodeScript, report: GroundingReport, spoken_script=None):
        del script, report, spoken_script
        raise AssertionError("render manifest should be skipped when grounding fails")

    monkeypatch.setattr(orchestrator, "_prepare_episode_plan", fake_prepare_episode_plan)
    monkeypatch.setattr(orchestrator, "spoken_delivery_episode", forbidden_spoken_delivery_episode)
    monkeypatch.setattr(orchestrator, "render_manifest", forbidden_render_manifest)

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer", episode_count=2)

    assert len(result["episodes"]) == 2
    for episode in result["episodes"]:
        assert episode["spoken_script"] is None
        assert episode["spoken_delivery"] is None
        assert episode["manifest"] is None
        assert episode["audio_manifest"] is None


def test_grounding_rolls_forward_with_limited_parallelism(tmp_path: Path, monkeypatch) -> None:
    settings = Settings()
    settings = settings.model_copy(
        update={
            "pipeline": settings.pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
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
    episode_plans = [
        EpisodePlan(
            episode_id=f"episode-{index}",
            sequence=index,
            title=f"Episode {index}",
            chapter_ids=[],
            beats=[],
        )
        for index in range(1, 4)
    ]
    start_times: dict[str, float] = {}
    finish_times: dict[str, float] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def fake_prepare_episode_plan(book_id: str, episode_plan: EpisodePlan) -> EpisodePreparation:
        del book_id
        start = time.monotonic()
        with lock:
            start_times[episode_plan.episode_id] = start
        if episode_plan.sequence in (1, 2):
            barrier.wait(timeout=1)
            time.sleep(0.05 if episode_plan.sequence == 1 else 0.2)
        else:
            time.sleep(0.01)
        finish = time.monotonic()
        with lock:
            finish_times[episode_plan.episode_id] = finish
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
        return EpisodePreparation(
            plan=episode_plan,
            script=script,
            report=report,
            repair_attempts=[],
        )

    def fake_finalize_episode_plan(
        book_id: str,
        preparation: EpisodePreparation,
        *,
        synthesize_audio: bool,
        allow_downstream: bool,
        previous_spoken_script: SpokenEpisodeNarration | None,
        next_plan: EpisodePlan | None,
        next_script: EpisodeScript | None,
    ) -> tuple[EpisodeOutput, SpokenEpisodeNarration | None]:
        del book_id, synthesize_audio, allow_downstream, previous_spoken_script, next_plan, next_script
        return EpisodeOutput(
            plan=preparation.plan,
            script=preparation.script,
            report=preparation.report,
        ), None

    monkeypatch.setattr(orchestrator, "_prepare_episode_plan", fake_prepare_episode_plan)
    monkeypatch.setattr(orchestrator, "_finalize_episode_plan", fake_finalize_episode_plan)

    orchestrator._process_episode_plans("test-book", episode_plans, synthesize_audio=False)

    assert start_times["episode-3"] < finish_times["episode-2"]


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
    seen_llm_provider: list[str | None] = []

    monkeypatch.setattr(
        "podcast_agent.cli.app._build_orchestrator",
        lambda database_url, **kwargs: (seen_llm_provider.append(kwargs.get("llm_provider")) or orchestrator),
    )

    result = CliRunner().invoke(
        app, ["render-audio-from-manifest", str(source_path), "--llm-provider", "anthropic"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["audio_manifest"]["audio_path"].endswith("episode-1.mp3")
    assert payload["manifest"]["episode_id"] == "episode-1"
    assert seen_llm_provider == ["anthropic"]


def test_spoken_delivery_cli_writes_sibling_artifacts(tmp_path: Path, monkeypatch) -> None:
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=Settings().model_copy(
            update={"pipeline": Settings().pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})}
        ),
    )
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["First fact. Second fact.", "Third fact follows."])
    _write_episode_output_artifact(source_path, manifest)

    monkeypatch.setattr(
        "podcast_agent.cli.app._build_orchestrator",
        lambda database_url, **kwargs: orchestrator,
    )

    result = CliRunner().invoke(app, ["spoken-delivery", str(source_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["spoken_script"]["episode_id"] == "episode-1"
    assert (source_path.parent / "spoken_script.json").exists()
    assert (source_path.parent / "spoken_delivery.json").exists()
    assert (source_path.parent / "spoken_delivery_arc_plan.json").exists()


def test_spoken_delivery_cli_rejects_failed_grounding_episode_output(tmp_path: Path, monkeypatch) -> None:
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=Settings().model_copy(
            update={"pipeline": Settings().pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})}
        ),
    )
    source_path = tmp_path / "source-run" / "observatory-book" / "episode-1" / "episode_output.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_render_manifest("episode-1", ["cli first", "cli second"])
    _write_episode_output_artifact(source_path, manifest)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["report"]["overall_status"] = "fail"
    source_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "podcast_agent.cli.app._build_orchestrator",
        lambda database_url, **kwargs: orchestrator,
    )

    result = CliRunner().invoke(app, ["spoken-delivery", str(source_path)])

    assert result.exit_code != 0
    assert "grounding did not pass" in result.output
    assert not (source_path.parent / "spoken_script.json").exists()
    assert not (source_path.parent / "spoken_delivery.json").exists()
