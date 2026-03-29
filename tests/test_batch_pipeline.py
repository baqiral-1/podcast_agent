"""Batch pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import BatchBookSpec, BatchRunManifest

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
