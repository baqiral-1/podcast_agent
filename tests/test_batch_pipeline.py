"""Batch pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import (
    BatchBookSpec,
    BatchRunManifest,
    EpisodeBeat,
    EpisodeFraming,
    EpisodeOutput,
    EpisodePlan,
    EpisodeScript,
    GroundingReport,
    RewriteMetrics,
    SeriesPlan,
    SpokenDeliveryEpisodeResult,
    SpokenEpisodeNarration,
    SpokenNarrationChunk,
)

BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea.

Chapter 2: Signals
Inside the city archive, the team finds coded letters.
"""


def test_run_batch_writes_shared_run_dir(tmp_path: Path) -> None:
    book_one = tmp_path / "book-one.txt"
    book_two = tmp_path / "book-two.txt"
    book_one.write_text(BOOK_TEXT, encoding="utf-8")
    book_two.write_text(BOOK_TEXT, encoding="utf-8")

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 2,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    manifest = BatchRunManifest(
        run_id="batch-test",
        with_audio=False,
        books=[
            BatchBookSpec(
                source_path=str(book_one),
                title="Batch Book One",
                author="Author One",
                episode_count=1,
            ),
            BatchBookSpec(
                source_path=str(book_two),
                title="Batch Book Two",
                author="Author Two",
                episode_count=1,
            ),
        ],
    )

    result = orchestrator.run_batch(manifest)

    run_dir = tmp_path / "runs" / "batch-test"
    assert result["run_id"] == "batch-test"
    assert run_dir.exists()
    assert (run_dir / "run.log").exists()
    for book in result["books"]:
        book_id = book["book_id"]
        assert (run_dir / book_id / "ingestion.json").exists()
        assert book["episodes"]


def test_run_batch_logs_stage_parallelism_overrides(tmp_path: Path) -> None:
    book_one = tmp_path / "book-one.txt"
    book_two = tmp_path / "book-two.txt"
    book_one.write_text(BOOK_TEXT, encoding="utf-8")
    book_two.write_text(BOOK_TEXT, encoding="utf-8")

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 2,
                    "analysis_parallelism": 1,
                    "spoken_delivery_parallelism": 1,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    manifest = BatchRunManifest(
        run_id="batch-parallelism",
        with_audio=False,
        books=[
            BatchBookSpec(
                source_path=str(book_one),
                title="Batch Book One",
                author="Author One",
                episode_count=1,
            ),
            BatchBookSpec(
                source_path=str(book_two),
                title="Batch Book Two",
                author="Author Two",
                episode_count=1,
            ),
        ],
    )

    orchestrator.run_batch(manifest)

    run_log = (tmp_path / "runs" / "batch-parallelism" / "run.log").read_text(encoding="utf-8")
    assert '"batch_started"' in run_log
    assert '"analysis_parallelism": 1' in run_log
    assert '"spoken_delivery_parallelism": 1' in run_log


def test_run_batch_log_events_include_book_identifiers(tmp_path: Path) -> None:
    book_one = tmp_path / "book-one.txt"
    book_two = tmp_path / "book-two.txt"
    book_one.write_text(BOOK_TEXT, encoding="utf-8")
    book_two.write_text(BOOK_TEXT, encoding="utf-8")

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 2,
                    "analysis_parallelism": 2,
                    "spoken_delivery_parallelism": 2,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    manifest = BatchRunManifest(
        run_id="batch-book-identifiers",
        with_audio=False,
        books=[
            BatchBookSpec(
                source_path=str(book_one),
                title="Batch Book One",
                author="Author One",
                episode_count=1,
            ),
            BatchBookSpec(
                source_path=str(book_two),
                title="Batch Book Two",
                author="Author Two",
                episode_count=1,
            ),
        ],
    )

    orchestrator.run_batch(manifest)

    expected_book_ids = {"batch-book-one", "batch-book-two"}
    expected_book_titles = {"Batch Book One", "Batch Book Two"}
    lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "batch-book-identifiers" / "run.log").read_text(encoding="utf-8").splitlines()
    ]

    for line in lines:
        payload = line["payload"]
        assert "book_id" in payload
        assert "book_title" in payload

    batch_stage_events = {
        "batch_started",
        "batch_stage_start",
        "batch_stage_end",
        "batch_completed",
    }
    for line in lines:
        if line["event_type"] not in batch_stage_events:
            continue
        payload = line["payload"]
        assert payload["book_id"] is None
        assert payload["book_title"] is None
        assert set(payload["book_ids"]) == expected_book_ids
        assert set(payload["book_titles"]) == expected_book_titles

    per_book_events = {
        "beat_write_started",
        "beat_write_completed",
        "writing_diagnostics",
        "planning_diagnostics",
        "analysis_diagnostics",
        "validation_diagnostics",
        "llm_request",
        "artifact_written",
    }
    for line in lines:
        if line["event_type"] not in per_book_events:
            continue
        payload = line["payload"]
        assert payload["book_id"] in expected_book_ids
        assert payload["book_title"] in expected_book_titles


def test_run_batch_fails_before_structuring_on_invalid_chapter_range(tmp_path: Path, monkeypatch) -> None:
    book_one = tmp_path / "book-one.txt"
    book_two = tmp_path / "book-two.txt"
    book_one.write_text(BOOK_TEXT, encoding="utf-8")
    book_two.write_text(BOOK_TEXT, encoding="utf-8")

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 2,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    def fail_if_structuring_called(*_args, **_kwargs):
        pytest.fail("Structuring should not run when chapter validation fails.")

    monkeypatch.setattr(orchestrator.structuring_agent, "structure", fail_if_structuring_called)

    manifest = BatchRunManifest(
        run_id="batch-invalid-chapter",
        with_audio=False,
        books=[
            BatchBookSpec(
                source_path=str(book_one),
                title="Batch Book One",
                author="Author One",
                episode_count=1,
                start_chapter="Chapter 9: Missing",
            ),
            BatchBookSpec(
                source_path=str(book_two),
                title="Batch Book Two",
                author="Author Two",
                episode_count=1,
            ),
        ],
    )

    with pytest.raises(ValueError, match="Unable to find start chapter"):
        orchestrator.run_batch(manifest)


def test_batch_book_spec_spoken_only_requires_artifact_path() -> None:
    with pytest.raises(ValueError, match="artifact_path is required"):
        BatchBookSpec.model_validate(
            {
                "title": "Spoken Only",
                "spoken-delivery-only": True,
            }
        )


def test_batch_book_spec_accepts_spoken_only_aliases() -> None:
    spec = BatchBookSpec.model_validate(
        {
            "title": "Spoken Only",
            "spoken-delivery-only": True,
            "artifact-path": "/tmp/source-book",
        }
    )

    assert spec.spoken_delivery_only is True
    assert spec.artifact_path == "/tmp/source-book"
    assert spec.source_path is None
    assert spec.episode_count is None


def test_run_batch_supports_mixed_spoken_only_books(tmp_path: Path, monkeypatch) -> None:
    full_book = tmp_path / "full-book.txt"
    full_book.write_text(BOOK_TEXT, encoding="utf-8")

    source_book_root = tmp_path / "source-run" / "spoken-only-book"
    source_book_root.mkdir(parents=True)
    (source_book_root / "ingestion.json").write_text(
        json.dumps({"book_id": "source-book", "title": "Source Book"}),
        encoding="utf-8",
    )
    (source_book_root / "analysis.json").write_text(
        json.dumps({"book_id": "source-book", "themes": ["theme"]}),
        encoding="utf-8",
    )
    (source_book_root / "series_plan.json").write_text(
        json.dumps({"book_id": "source-book", "episodes": []}),
        encoding="utf-8",
    )

    source_episode_dir = source_book_root / "episode-1"
    source_episode_dir.mkdir(parents=True)
    source_output = EpisodeOutput(
        plan=EpisodePlan(
            episode_id="episode-1",
            sequence=1,
            title="Episode 1",
            chapter_ids=[],
            chunk_ids=[],
            themes=[],
            beats=[
                EpisodeBeat(
                    beat_id="beat-1",
                    title="Beat 1",
                    chunk_ids=[],
                )
            ],
        ),
        script=EpisodeScript(
            episode_id="episode-1",
            title="Episode 1",
            narrator="Narrator",
            segments=[],
        ),
        report=GroundingReport(
            episode_id="episode-1",
            overall_status="pass",
            claim_assessments=[],
        ),
    )
    (source_episode_dir / "episode_output.json").write_text(
        source_output.model_dump_json(indent=2),
        encoding="utf-8",
    )

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 1,
                    "analysis_parallelism": 1,
                    "spoken_delivery_parallelism": 1,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingest_calls: list[str] = []
    original_ingest = orchestrator.ingest_book

    def counting_ingest(*args, **kwargs):
        result = original_ingest(*args, **kwargs)
        ingest_calls.append(result.book_id)
        return result

    monkeypatch.setattr(orchestrator, "ingest_book", counting_ingest)

    def fake_spoken_delivery_episode(book_id: str, script: EpisodeScript):
        del book_id
        spoken_script = SpokenEpisodeNarration(
            episode_id=script.episode_id,
            title=script.title,
            narrator=script.narrator,
            narration="Spoken narration body.",
            chunks=[
                SpokenNarrationChunk(
                    chunk_id=f"{script.episode_id}-chunk-1",
                    text="Spoken narration body.",
                    word_count=3,
                )
            ],
        )
        spoken_delivery = SpokenDeliveryEpisodeResult(
            episode_id=script.episode_id,
            mode="full",
            metrics=RewriteMetrics(
                source_word_count=3,
                spoken_word_count=3,
                expansion_ratio=1.0,
                source_sentence_count=1,
                spoken_sentence_count=1,
                source_average_sentence_length=3.0,
                spoken_average_sentence_length=3.0,
                source_paragraph_count=1,
                spoken_paragraph_count=1,
            ),
            chunk_count=1,
        )
        return spoken_script, spoken_delivery, None

    monkeypatch.setattr(orchestrator, "spoken_delivery_episode", fake_spoken_delivery_episode)
    monkeypatch.setattr(
        orchestrator,
        "_build_episode_framing",
        lambda plan, **kwargs: EpisodeFraming(
            recap=f"Previously on {plan.title}.",
            next_overview="Next episode preview.",
        ),
    )

    manifest = BatchRunManifest(
        run_id="mixed-mode-batch",
        with_audio=False,
        books=[
            BatchBookSpec(
                source_path=str(full_book),
                title="Full Book",
                author="Author One",
                episode_count=1,
            ),
            BatchBookSpec.model_validate(
                {
                    "title": "Spoken Only Book",
                    "spoken-delivery-only": True,
                    "artifact-path": str(source_book_root),
                }
            ),
        ],
    )

    result = orchestrator.run_batch(manifest)

    assert ingest_calls == ["full-book"]
    by_book_id = {book["book_id"]: book for book in result["books"]}

    full_book_payload = by_book_id["full-book"]
    assert full_book_payload["ingestion"]["title"] == "Full Book"
    assert full_book_payload["episodes"]

    spoken_only_payload = by_book_id["spoken-only-book"]
    assert spoken_only_payload["ingestion"]["book_id"] == "source-book"
    assert spoken_only_payload["analysis"]["book_id"] == "source-book"
    assert spoken_only_payload["series_plan"]["book_id"] == "source-book"
    assert len(spoken_only_payload["episodes"]) == 1
    assert spoken_only_payload["episodes"][0]["spoken_delivery"] is not None

    run_dir = tmp_path / "runs" / "mixed-mode-batch"
    assert (run_dir / "spoken-only-book" / "ingestion.json").exists()
    assert (run_dir / "spoken-only-book" / "analysis.json").exists()
    assert (run_dir / "spoken-only-book" / "series_plan.json").exists()
    assert (run_dir / "spoken-only-book" / "episode-1" / "episode_output.json").exists()


def test_run_batch_spoken_only_fallback_to_factual_script(tmp_path: Path, monkeypatch) -> None:
    source_book_root = tmp_path / "source-run" / "spoken-only-book"
    source_book_root.mkdir(parents=True)
    (source_book_root / "ingestion.json").write_text(
        json.dumps({"book_id": "source-book", "title": "Source Book"}),
        encoding="utf-8",
    )
    (source_book_root / "analysis.json").write_text(
        json.dumps({"book_id": "source-book", "themes": ["theme"]}),
        encoding="utf-8",
    )
    (source_book_root / "series_plan.json").write_text(
        SeriesPlan(
            book_id="source-book",
            format="single_narrator",
            strategy_summary="Summary",
            episodes=[
                EpisodePlan(
                    episode_id="episode-1",
                    sequence=1,
                    title="Episode 1",
                    chapter_ids=[],
                    chunk_ids=[],
                    themes=[],
                    beats=[EpisodeBeat(beat_id="beat-1", title="Beat 1", chunk_ids=[])],
                )
            ],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    source_episode_dir = source_book_root / "episode-1"
    source_episode_dir.mkdir(parents=True)
    (source_episode_dir / "factual_script.json").write_text(
        EpisodeScript(
            episode_id="episode-1",
            title="Episode 1",
            narrator="Narrator",
            segments=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={
                    "artifact_root": tmp_path / "runs",
                    "episode_parallelism": 1,
                    "batch_parallelism": 1,
                    "analysis_parallelism": 1,
                    "spoken_delivery_parallelism": 1,
                }
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    def fake_spoken_delivery_episode(book_id: str, script: EpisodeScript):
        del book_id
        spoken_script = SpokenEpisodeNarration(
            episode_id=script.episode_id,
            title=script.title,
            narrator=script.narrator,
            narration="Spoken narration body.",
            chunks=[
                SpokenNarrationChunk(
                    chunk_id=f"{script.episode_id}-chunk-1",
                    text="Spoken narration body.",
                    word_count=3,
                )
            ],
        )
        spoken_delivery = SpokenDeliveryEpisodeResult(
            episode_id=script.episode_id,
            mode="full",
            metrics=RewriteMetrics(
                source_word_count=3,
                spoken_word_count=3,
                expansion_ratio=1.0,
                source_sentence_count=1,
                spoken_sentence_count=1,
                source_average_sentence_length=3.0,
                spoken_average_sentence_length=3.0,
                source_paragraph_count=1,
                spoken_paragraph_count=1,
            ),
            chunk_count=1,
        )
        return spoken_script, spoken_delivery, None

    monkeypatch.setattr(orchestrator, "spoken_delivery_episode", fake_spoken_delivery_episode)
    monkeypatch.setattr(
        orchestrator,
        "_build_episode_framing",
        lambda plan, **kwargs: EpisodeFraming(
            recap=f"Previously on {plan.title}.",
            next_overview="Next episode preview.",
        ),
    )

    manifest = BatchRunManifest(
        run_id="spoken-only-factual-fallback",
        with_audio=False,
        books=[
            BatchBookSpec.model_validate(
                {
                    "title": "Spoken Only Book",
                    "spoken-delivery-only": True,
                    "artifact-path": str(source_book_root),
                }
            ),
        ],
    )

    result = orchestrator.run_batch(manifest)
    spoken_only_payload = result["books"][0]
    assert spoken_only_payload["book_id"] == "spoken-only-book"
    assert len(spoken_only_payload["episodes"]) == 1
    assert spoken_only_payload["episodes"][0]["spoken_delivery"] is not None


def test_run_batch_spoken_only_factual_script_requires_series_plan(tmp_path: Path) -> None:
    source_book_root = tmp_path / "source-run" / "spoken-only-book"
    source_episode_dir = source_book_root / "episode-1"
    source_episode_dir.mkdir(parents=True)
    (source_episode_dir / "factual_script.json").write_text(
        EpisodeScript(
            episode_id="episode-1",
            title="Episode 1",
            narrator="Narrator",
            segments=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={"artifact_root": tmp_path / "runs"}
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )
    manifest = BatchRunManifest(
        run_id="spoken-only-missing-series-plan",
        with_audio=False,
        books=[
            BatchBookSpec.model_validate(
                {
                    "title": "Spoken Only Book",
                    "spoken-delivery-only": True,
                    "artifact-path": str(source_book_root),
                }
            ),
        ],
    )

    with pytest.raises(ValueError, match="requires series_plan.json"):
        orchestrator.run_batch(manifest)


def test_run_batch_spoken_only_factual_script_episode_id_must_match_series_plan(tmp_path: Path) -> None:
    source_book_root = tmp_path / "source-run" / "spoken-only-book"
    source_episode_dir = source_book_root / "episode-1"
    source_episode_dir.mkdir(parents=True)
    (source_episode_dir / "factual_script.json").write_text(
        EpisodeScript(
            episode_id="episode-1",
            title="Episode 1",
            narrator="Narrator",
            segments=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (source_book_root / "series_plan.json").write_text(
        SeriesPlan(
            book_id="source-book",
            format="single_narrator",
            strategy_summary="Summary",
            episodes=[
                EpisodePlan(
                    episode_id="episode-2",
                    sequence=1,
                    title="Episode 2",
                    chapter_ids=[],
                    chunk_ids=[],
                    themes=[],
                    beats=[EpisodeBeat(beat_id="beat-2", title="Beat 2", chunk_ids=[])],
                )
            ],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    settings = Settings().model_copy(
        update={
            "pipeline": Settings().pipeline.model_copy(
                update={"artifact_root": tmp_path / "runs"}
            )
        }
    )
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )
    manifest = BatchRunManifest(
        run_id="spoken-only-mismatched-series-plan",
        with_audio=False,
        books=[
            BatchBookSpec.model_validate(
                {
                    "title": "Spoken Only Book",
                    "spoken-delivery-only": True,
                    "artifact-path": str(source_book_root),
                }
            ),
        ],
    )

    with pytest.raises(ValueError, match="Missing episode_id 'episode-1'"):
        orchestrator.run_batch(manifest)
