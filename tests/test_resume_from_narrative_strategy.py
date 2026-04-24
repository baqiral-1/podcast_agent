from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    BookRecord,
    NarrativeStrategy,
    PipelineConfig,
    ProjectStatus,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitive,
    SynthesisPrimitivesArtifact,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "resume_from_narrative_strategy.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_narrative_strategy",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
resume_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(resume_script)


def _write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_project_dir(tmp_path: Path) -> Path:
    project_dir = tmp_path / "run_1"
    project_dir.mkdir()

    axis = ThematicAxis(
        axis_id="axis_1",
        name="Axis",
        description="Axis description",
        theme_importance_score=1.0,
    )
    primitive = SynthesisPrimitive(
        id="primitive_1",
        title="Primitive",
        summary="Primitive summary",
        axis_ids=["axis_1"],
        core_passage_ids=[],
        actor_ids=["actor_1"],
    )
    primitives = SynthesisPrimitivesArtifact(
        project_id="run_1",
        primitives_by_family={"epochal_turns": [primitive]},
    )
    synthesis_map = SynthesisMap(
        project_id="run_1",
        primitives_by_family={"epochal_turns": [primitive]},
        episode_candidate_clusters=[
            {
                "cluster_id": "cluster_1",
                "title": "Cluster",
                "summary": "Cluster summary",
                "primary_member_id": "primitive_1",
                "member_ids": ["primitive_1"],
                "actor_ids": ["actor_1"],
                "local_question": "What changed?",
                "local_payoff_shape": "reveal",
            }
        ],
    )
    actor_metadata = ActorMetadata(
        project_id="run_1",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )
    project = ThematicProject(
        project_id="run_1",
        theme="Theme",
        books=[
            BookRecord(
                book_id="book_1",
                title="Book",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        requested_episode_count=1,
        config=PipelineConfig(
            skip_grounding=False,
            skip_audio=False,
            skip_spoken_delivery=True,
        ),
        status=ProjectStatus.ANALYZING,
    )
    corpus = ThematicCorpus(project_id="run_1", axes=[axis])

    _write_json(project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]})
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "synthesis_primitives.json", primitives)
    _write_json(project_dir / "synthesis_map.json", synthesis_map)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    return project_dir


def test_resume_from_narrative_strategy_passes_actor_metadata_and_forces_skips(
    monkeypatch,
    tmp_path,
):
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            calls["orchestrator"] = self

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisMap,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[NarrativeStrategy, dict[str, Any]]:
            calls["narrative_actor_metadata"] = actor_metadata
            calls["narrative_config"] = project.config
            strategy = NarrativeStrategy(
                strategy_type="chronological",
                justification="Test",
                series_arc="Test arc",
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode",
                        driving_question="Question?",
                        arc_summary="Arc summary",
                        cluster_path=[
                            {
                                "occurrence_id": "occ_1",
                                "cluster_id": "cluster_1",
                                "usage": "primary",
                                "emphasis": "anchor",
                            }
                        ],
                        actor_arc_directives=[
                            {
                                "actor_id": "actor_1",
                                "arc_threads": [
                                    {
                                        "thread_id": "thread_1",
                                        "arc_type": "role",
                                        "label": "Role",
                                        "premise": "Premise",
                                    }
                                ],
                            }
                        ],
                    )
                ],
            )
            return strategy, {"unknown_actor_ids": 0}

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            calls["resolved_strategy"] = strategy
            return project.model_copy(update={"episode_count": 1})

        async def _plan_series(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisMap,
            strategy: NarrativeStrategy,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[list[Any], dict[str, Any]]:
            calls["planning_actor_metadata"] = actor_metadata
            calls["planning_config"] = project.config
            return [SimpleNamespace(episode_number=1)], {"unknown_actor_ids": 0}

        async def _produce_episode(
            self,
            plan: Any,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            semaphore: asyncio.Semaphore,
        ) -> tuple[int, Any]:
            calls["production_actor_metadata"] = actor_metadata
            calls["production_config"] = project.config
            return plan.episode_number, SimpleNamespace(episode_number=plan.episode_number)

        def _write_passage_utilization(self, **kwargs: Any) -> None:
            calls["passage_utilization"] = kwargs

        def _build_writing_actor_metrics(
            self,
            project_dir: Path,
            spoken_scripts: list[tuple[int, Any]],
        ) -> dict[str, Any]:
            calls["spoken_scripts"] = spoken_scripts
            return {"completed_episode_count": len(spoken_scripts)}

        def _write_actor_metadata_metrics(self, **kwargs: Any) -> None:
            calls["actor_metadata_metrics"] = kwargs

        async def _render_episode_audio(
            self,
            episode_number: int,
            spoken: Any,
            config: PipelineConfig,
            project_dir: Path,
            semaphore: asyncio.Semaphore,
            *,
            skip_audio: bool,
        ) -> None:
            calls["audio_skip"] = skip_audio

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(resume_script._resume_from_narrative_strategy("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert calls["narrative_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["narrative_config"].skip_grounding is True
    assert calls["narrative_config"].skip_audio is True
    assert calls["narrative_config"].skip_spoken_delivery is False
    assert calls["planning_config"].skip_grounding is True
    assert calls["production_config"].skip_audio is True
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["narrative_strategy"] == {
        "unknown_actor_ids": 0
    }

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
