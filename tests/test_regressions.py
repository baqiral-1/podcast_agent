"""Regression tests for reviewed runtime issues."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_agent.cli.app import _build_orchestrator
from podcast_agent.config import PipelineConfig, Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.db.repository import _format_pgvector
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.llm.base import LLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import GroundingStatus


BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory.

Chapter 2: Signals
Inside the city archive, the team finds coded letters describing the observatory's mirrors and the way they coordinated messages across valleys.
"""


class FailingValidationLLM(LLMClient):
    """LLM stub that never produces a passing grounding report."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "grounding_report":
            script = payload["script"]
            assessments = []
            for segment in script["segments"]:
                for claim in segment["claims"]:
                    assessments.append(
                        {
                            "claim_id": claim["claim_id"],
                            "status": GroundingStatus.UNSUPPORTED,
                            "reason": "Synthetic regression failure.",
                            "evidence_chunk_ids": [],
                        }
                    )
            return response_model.model_validate(
                {
                    "episode_id": script["episode_id"],
                    "overall_status": "fail",
                    "claim_assessments": assessments,
                }
            )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


def test_build_orchestrator_uses_in_memory_when_database_url_missing(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    orchestrator = _build_orchestrator(None)

    assert isinstance(orchestrator.repository, InMemoryRepository)
    assert Settings().database.dsn is None


def test_pgvector_serialization_uses_vector_literal() -> None:
    assert _format_pgvector([0.1, -0.2, 0.0]) == "[0.1,-0.2,0.0]"


def test_failed_grounding_skips_render_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
    )

    result = orchestrator.run_pipeline(book_path, title="Broken Grounding", author="A. Writer")

    assert result["episodes"]
    assert result["episodes"][0]["report"]["overall_status"] == "fail"
    assert result["episodes"][0]["manifest"] is None


def test_each_run_uses_book_title_and_minute_timestamp_in_artifact_subdirectory(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(
        pipeline=PipelineConfig(artifact_root=tmp_path / "runs"),
    )

    first = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
        settings=settings,
    )
    second = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
        settings=settings,
    )

    first.run_pipeline(book_path, title="Run One", author="A. Writer")
    second.run_pipeline(book_path, title="Run Two", author="A. Writer")

    run_dirs = sorted(path.name for path in (tmp_path / "runs").iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert run_dirs[0].startswith("run-one-")
    assert run_dirs[1].startswith("run-two-")


def test_run_log_captures_llm_prompts_and_artifact_writes(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    orchestrator.run_pipeline(book_path, title="Logged Book", author="A. Writer")

    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]

    assert any(line["event_type"] == "llm_request" for line in lines)
    assert any(line["event_type"] == "artifact_written" for line in lines)
    assert any(line["event_type"] == "stage_start" for line in lines)


def test_incremental_structuring_logs_one_request_per_chapter(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Chapter Logged Book", author="A. Writer")
    orchestrator.index_book(ingestion)

    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    chapter_requests = [
        line for line in lines if line["event_type"] == "llm_request" and line["payload"]["schema_name"] == "structured_chapter"
    ]

    assert len(chapter_requests) == 2
