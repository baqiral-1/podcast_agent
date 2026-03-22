"""Batch pipeline tests."""

from __future__ import annotations

from pathlib import Path

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
