from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import (
    ActorMetadata,
    BookRecord,
    EventPrimitive,
    PrimitiveSubstrate,
    ReadingPrimitive,
)
from podcast_agent.utils.actor_metadata import collect_actor_ids_for_primitives


def _settings_with_artifact_root(tmp_path: Path) -> Settings:
    settings = Settings()
    return settings.model_copy(
        update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path})}
    )


def test_collect_actor_ids_for_primitives_uses_unified_actor_ids() -> None:
    event = EventPrimitive(
        id="evt_1",
        substrate=PrimitiveSubstrate.EVENTS,
        title="A coup lands",
        core_passage_ids=["passage_1"],
        actor_ids=["mossadeq", "shah"],
        event_type="coup",
        what_happened="A coup removes the government.",
    )
    reading = ReadingPrimitive(
        id="rdg_1",
        substrate=PrimitiveSubstrate.READINGS,
        title="A later reading",
        core_passage_ids=["passage_2"],
        actor_ids=["shah", "khomeini"],
        reading_summary="The coup becomes a durable political trauma.",
    )

    actor_ids = collect_actor_ids_for_primitives([event, reading])

    assert actor_ids == {"mossadeq", "shah", "khomeini"}


def test_run_multi_book_podcast_marks_failed_on_late_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator = PipelineOrchestrator(_settings_with_artifact_root(tmp_path))

    async def fake_ingest(*args, **kwargs) -> BookRecord:
        source_path = str(args[0])
        title = str(args[1])
        author = str(args[2])
        return BookRecord(
            book_id=Path(source_path).stem,
            title=title,
            author=author,
            source_path=source_path,
            source_type="txt",
        )

    async def fake_decompose(*args, **kwargs):
        return [], ActorMetadata(project_id="proj_1"), {}

    async def fake_extract(*args, **kwargs):
        return {}

    async def fake_map(*args, **kwargs):
        raise AttributeError("'EventPrimitive' object has no attribute 'primary_actor_ids'")

    monkeypatch.setattr(orchestrator, "_ingest_and_index_book", fake_ingest)
    monkeypatch.setattr(orchestrator, "_decompose_theme", fake_decompose)
    monkeypatch.setattr(orchestrator, "_extract_passages", fake_extract)
    monkeypatch.setattr(orchestrator, "_map_synthesis", fake_map)

    with pytest.raises(AttributeError, match="primary_actor_ids"):
        asyncio.run(
            orchestrator.run_multi_book_podcast(
                source_paths=["/tmp/book_a.txt", "/tmp/book_b.txt"],
                theme="Iranian Revolution",
                episode_count=None,
                project_id="proj_1",
            )
        )

    project_payload = json.loads(
        (tmp_path / "proj_1" / "thematic_project.json").read_text(encoding="utf-8")
    )
    run_log = (tmp_path / "proj_1" / "run.log").read_text(encoding="utf-8")

    assert project_payload["status"] == "failed"
    assert len(project_payload["books"]) == 2
    assert '"event_type": "pipeline_error"' in run_log
    assert '"error_type": "AttributeError"' in run_log
