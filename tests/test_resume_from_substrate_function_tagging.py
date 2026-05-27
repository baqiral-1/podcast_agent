from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _section_progression_helpers import (
    make_section_progression,
    stages_for_purposes,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EventPrimitive,
    NarrativeState,
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveSubstrate,
    ProjectStatus,
    SceneDiscoveryArtifact,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "resume_from_substrate_function_tagging.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_substrate_function_tagging",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
resume_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(resume_script)


def test_resume_from_substrate_function_tagging_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_from_substrate_function_tagging.py",
            "--project-id",
            "run_42",
        ],
    )

    args = resume_script._parse_args()

    assert args.project_id == "run_42"


def test_resume_from_substrate_function_tagging_main_runs_async_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    original_asyncio_run = asyncio.run

    async def fake_resume(project_id: str) -> None:
        calls["project_id"] = project_id

    def fake_asyncio_run(coro: Any) -> None:
        calls["asyncio_run_called"] = True
        original_asyncio_run(coro)

    monkeypatch.setattr(
        resume_script,
        "_parse_args",
        lambda: SimpleNamespace(project_id="run_42"),
    )
    monkeypatch.setattr(
        resume_script,
        "_resume_from_substrate_function_tagging",
        fake_resume,
    )
    monkeypatch.setattr(resume_script.asyncio, "run", fake_asyncio_run)

    resume_script.main()

    assert calls["asyncio_run_called"] is True
    assert calls["project_id"] == "run_42"


def _write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_event_primitive() -> EventPrimitive:
    return EventPrimitive(
        id="primitive_1",
        substrate=PrimitiveSubstrate.EVENTS,
        title="Primitive",
        core_passage_ids=["passage_1"],
        actor_ids=["actor_1"],
        event_type="turning_point",
        what_happened="A decisive event changes the situation.",
        event_result="The balance breaks.",
    )


def _build_substrate_primitives() -> SynthesisPrimitivesArtifact:
    return SynthesisPrimitivesArtifact(project_id="run_1", primitives=[_build_event_primitive()])


def _build_strategy(*, title: str = "Episode") -> NarrativeStrategy:
    return NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title=title,
                arc_summary="Arc summary",
                episode_spine={
                    "listener_problem": "Question?",
                    "episode_answer": "A working claim.",
                    "core_primitive_ids": [
                        "primitive_1",
                        "core_2",
                    ],
                    "support_primitive_roles": {
                        "support_1": "mechanism",
                        "support_2": "mechanism",
                    },
                    "recall_primitive_ids": [],
                },
            )
        ],
        recommended_episode_count=1,
    )


def _build_scene_discovery() -> SceneDiscoveryArtifact:
    return SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A room becomes a decision point.",
                    "scene_jobs": ["opening", "answer"],
                    "anchor_image": "A room and a file.",
                    "why_sceneable": "The beat is visible and oral.",
                    "actor_ids": ["actor_1"],
                }
            ]
        }
    )


def _build_narrative_states() -> tuple[NarrativeState, NarrativeState]:
    state_pre = NarrativeState.model_validate(
        {
            "project_id": "run_1",
            "next_episode_number": 1,
            "listener": {
                "known_explanation_item_ids": [],
                "known_actor_ids": [],
                "questions": [],
                "memory_threads": [],
                "carry_forward_memory": [],
                "last_episode_takeaway": None,
            },
            "host": {
                "mysteries": [],
                "assumptions": [],
                "working_theories": [],
                "recent_revisions": [],
                "confidence_posture": "mixed",
                "last_episode_takeaway": None,
            },
        }
    )
    state_post = NarrativeState.model_validate(
        {
            "project_id": "run_1",
            "next_episode_number": 2,
            "listener": state_pre.listener.model_dump(mode="json"),
            "host": state_pre.host.model_dump(mode="json"),
        }
    )
    return state_pre, state_post


def _episode_architecture() -> EpisodeArchitecture:
    purposes = ["opening", "setup", "setup", "turn", "setup", "closing"]
    sections = [
        ArchitectureSection.model_validate(
            {
                "section_id": f"section_{index + 1}",
                "purpose": purpose,
                "approx_runtime_minutes": 15.0,
                "primitive_ids": ["primitive_1"],
                "listener_tension": f"Question {index + 1}?",
                "section_turn": f"Turn {index + 1}.",
                "transition_logic": f"Transition {index + 1}.",
                "depends_on_section_ids": (
                    [] if index == 0 else [f"section_{index}"]
                ),
                "sets_up_section_ids": (
                    [] if index == len(purposes) - 1 else [f"section_{index + 2}"]
                ),
                "section_progression": make_section_progression(
                    stage, label=f"section_{index + 1}"
                ),
            }
        )
        for index, (purpose, stage) in enumerate(
            zip(purposes, stages_for_purposes(purposes))
        )
    ]
    return EpisodeArchitecture.model_validate(
        {
            "episode_number": 1,
            "major_turn_section_id": "section_4",
            "allowed_recurring_primitive_ids": ["primitive_1"],
            "forbidden_redundancies": [],
            "sections": [section.model_dump(mode="json") for section in sections],
            "architecture_notes": [],
        }
    )


def _build_project_dir(tmp_path: Path) -> Path:
    project_dir = tmp_path / "run_1"
    project_dir.mkdir()

    axis = ThematicAxis(
        axis_id="axis_1",
        name="Axis",
        description="Axis description",
        theme_importance_score=1.0,
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
        requested_episode_count=None,
        config=PipelineConfig(
            skip_grounding=False,
            skip_audio=False,
            skip_spoken_delivery=True,
            synthesis_total_passage_cap=321,
        ),
        status=ProjectStatus.ANALYZING,
    )
    corpus = ThematicCorpus(project_id="run_1", axes=[axis])

    _write_json(
        project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]}
    )
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(project_dir / "substrate_primitives.json", _build_substrate_primitives())
    return project_dir


def test_resume_from_substrate_function_tagging_reruns_tagging_and_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {"order": []}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _tag_substrate_primitives(
            self,
            *,
            project: ThematicProject,
            corpus: ThematicCorpus,
            project_dir: Path,
            primitives: SynthesisPrimitivesArtifact,
            actor_metadata: ActorMetadata,
        ) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
            calls["order"].append("tag_substrate_primitives")
            calls["tagging_config"] = project.config
            calls["tagging_corpus"] = corpus
            calls["tagging_actor_metadata"] = actor_metadata
            calls["tagging_input"] = primitives
            tagged = _build_substrate_primitives()
            _write_json(project_dir / "tagged_primitives.json", tagged)
            return tagged, {
                "primitive_counts_by_substrate": {"events": 1},
                "tagged_counts_by_substrate": {"events": 1},
            }

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisPrimitivesArtifact,
            project_dir: Path,
            actor_metadata: ActorMetadata,
            scene_discovery: SceneDiscoveryArtifact | None = None,
        ) -> tuple[NarrativeStrategy, dict[str, Any]]:
            calls["order"].append("narrative_strategy")
            calls["narrative_actor_metadata"] = actor_metadata
            strategy = _build_strategy()
            _write_json(project_dir / "narrative_strategy.json", strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _discover_scenes(
            self, **kwargs: Any
        ) -> SceneDiscoveryArtifact:
            calls["order"].append("scene_discovery")
            scene_discovery = _build_scene_discovery()
            project_dir = kwargs["project_dir"]
            _write_json(project_dir / "scene_discovery.json", scene_discovery)
            _write_json(
                project_dir / "scene_discovery_diagnostics.json",
                {"warning_count": 0, "warnings": []},
            )
            return scene_discovery

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: SynthesisPrimitivesArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            calls["order"].append("selected_primitives")
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives=[_build_event_primitive()],
            )
            _write_json(project_dir / "retained_primitives.json", synthesis_map)
            return synthesis_map

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            calls["order"].append("resolve_episode_count")
            return project.model_copy(update={"episode_count": 1})

        async def _plan_series_with_narrative_state(
            self, **kwargs: Any
        ) -> tuple[
            list[EpisodeArchitecture],
            list[Any],
            dict[int, NarrativeState],
            dict[int, NarrativeState],
            dict[str, Any],
        ]:
            calls["order"].append("episode_architecture")
            calls["architecture_actor_metadata"] = kwargs["actor_metadata"]
            architecture = _episode_architecture()
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            calls["order"].append("plan_series")
            calls["planning_actor_metadata"] = kwargs["actor_metadata"]
            calls["planning_config"] = kwargs["project"].config
            _write_json(project_dir / "series_plan.json", {"episodes": [{"episode_number": 1}]})
            state_pre, state_post = _build_narrative_states()
            return (
                [architecture],
                [SimpleNamespace(episode_number=1)],
                {1: state_pre},
                {1: state_post},
                {
                    "episode_architecture": {"unknown_actor_ids": 0},
                    "episode_planning": {"unknown_actor_ids": 0},
                },
            )

        async def _produce_episode(
            self,
            plan: Any,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            host_policy: dict[str, Any],
            primitive_lookup: dict[str, Any],
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
            series_explanation_registry: list[Any] | None = None,
            narrative_state_pre: NarrativeState | None = None,
            narrative_state_post: NarrativeState | None = None,
        ) -> tuple[int, Any]:
            calls["order"].append("produce_episode")
            calls["production_actor_metadata"] = actor_metadata
            calls["production_config"] = project.config
            calls["host_policy"] = host_policy
            calls["primitive_lookup"] = primitive_lookup
            return plan.episode_number, SimpleNamespace(
                episode_number=plan.episode_number
            )

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
            calls["order"].append("render_audio")
            calls["audio_skip"] = skip_audio

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))

    assert calls["order"] == [
        "tag_substrate_primitives",
        "scene_discovery",
        "narrative_strategy",
        "selected_primitives",
        "resolve_episode_count",
        "episode_architecture",
        "plan_series",
        "produce_episode",
        "render_audio",
    ]
    assert calls["bound_project_dir"] == project_dir
    assert calls["tagging_corpus"].axes[0].axis_id == "axis_1"
    assert calls["tagging_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["tagging_input"].primitives[0].id == "primitive_1"
    assert calls["narrative_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["architecture_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["tagging_config"].skip_grounding is True
    assert calls["tagging_config"].skip_audio is True
    assert calls["tagging_config"].skip_spoken_delivery is False
    assert calls["planning_config"].skip_audio is True
    assert calls["production_config"].skip_grounding is True
    assert calls["primitive_lookup"]["primitive_1"].id == "primitive_1"
    assert "authorial_policy" in calls["host_policy"]
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["substrate_function_tagging"] == {
        "primitive_counts_by_substrate": {"events": 1},
        "tagged_counts_by_substrate": {"events": 1},
    }

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
    assert final_project.episode_count == 1


def test_resume_from_substrate_function_tagging_rejects_axis_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    mismatched_axis = ThematicAxis(
        axis_id="axis_1",
        name="Changed Axis",
        description="Axis description",
        theme_importance_score=1.0,
    )
    _write_json(
        project_dir / "thematic_axes.json",
        {"axes": [mismatched_axis.model_dump(mode="json")]},
    )

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            raise AssertionError("resume should fail before binding run logger")

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="disagree on axes"):
        asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))


def test_resume_from_substrate_function_tagging_requires_actor_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    (project_dir / "actor_metadata.json").unlink()

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="Missing required artifact"):
        asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))


def test_resume_from_substrate_function_tagging_requires_substrate_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    (project_dir / "substrate_primitives.json").unlink()

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="Missing required artifact"):
        asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))


def test_resume_from_substrate_function_tagging_fails_when_upstream_artifact_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            assert bound_project_dir == project_dir

        async def _tag_substrate_primitives(
            self,
            *,
            project: ThematicProject,
            corpus: ThematicCorpus,
            project_dir: Path,
            primitives: SynthesisPrimitivesArtifact,
            actor_metadata: ActorMetadata,
        ) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
            _write_json(project_dir / "substrate_primitives.json", SynthesisPrimitivesArtifact(project_id="run_1", primitives=[]))
            tagged = _build_substrate_primitives()
            _write_json(project_dir / "tagged_primitives.json", tagged)
            return tagged, {
                "primitive_counts_by_substrate": {"events": 1},
                "tagged_counts_by_substrate": {"events": 1},
            }

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisPrimitivesArtifact,
            project_dir: Path,
            actor_metadata: ActorMetadata,
            scene_discovery: SceneDiscoveryArtifact | None = None,
        ) -> tuple[NarrativeStrategy, dict[str, Any]]:
            strategy = _build_strategy()
            _write_json(project_dir / "narrative_strategy.json", strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _discover_scenes(
            self, **kwargs: Any
        ) -> SceneDiscoveryArtifact:
            scene_discovery = _build_scene_discovery()
            project_dir = kwargs["project_dir"]
            _write_json(project_dir / "scene_discovery.json", scene_discovery)
            return scene_discovery

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: SynthesisPrimitivesArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives=[_build_event_primitive()],
            )
            return synthesis_map

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            return project.model_copy(update={"episode_count": 1})

        async def _plan_series_with_narrative_state(
            self, **_: Any
        ) -> tuple[
            list[EpisodeArchitecture],
            list[Any],
            dict[int, NarrativeState],
            dict[int, NarrativeState],
            dict[str, Any],
        ]:
            state_pre, state_post = _build_narrative_states()
            return (
                [_episode_architecture()],
                [SimpleNamespace(episode_number=1)],
                {1: state_pre},
                {1: state_post},
                {
                    "episode_architecture": {"unknown_actor_ids": 0},
                    "episode_planning": {"unknown_actor_ids": 0},
                },
            )

        async def _produce_episode(
            self,
            plan: Any,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            host_policy: dict[str, Any],
            primitive_lookup: dict[str, Any],
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
            series_explanation_registry: list[Any] | None = None,
            narrative_state_pre: NarrativeState | None = None,
            narrative_state_post: NarrativeState | None = None,
        ) -> tuple[int, Any]:
            return plan.episode_number, SimpleNamespace(
                episode_number=plan.episode_number
            )

        def _write_passage_utilization(self, **kwargs: Any) -> None:
            return None

        def _build_writing_actor_metrics(
            self,
            project_dir: Path,
            spoken_scripts: list[tuple[int, Any]],
        ) -> dict[str, Any]:
            return {"completed_episode_count": len(spoken_scripts)}

        def _write_actor_metadata_metrics(self, **kwargs: Any) -> None:
            return None

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
            return None

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(
        RuntimeError,
        match="Strict integrity check failed: substrate_primitives.json changed during resume",
    ):
        asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED


def test_resume_from_substrate_function_tagging_uses_stage_label_in_resume_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

        async def _tag_substrate_primitives(
            self,
            *,
            project: ThematicProject,
            corpus: ThematicCorpus,
            project_dir: Path,
            primitives: SynthesisPrimitivesArtifact,
            actor_metadata: ActorMetadata,
        ) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
            tagged = _build_substrate_primitives()
            _write_json(project_dir / "tagged_primitives.json", tagged)
            return tagged, {
                "primitive_counts_by_substrate": {"events": 1},
                "tagged_counts_by_substrate": {"events": 1},
            }

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisPrimitivesArtifact,
            project_dir: Path,
            actor_metadata: ActorMetadata,
            scene_discovery: SceneDiscoveryArtifact | None = None,
        ) -> tuple[NarrativeStrategy, dict[str, Any]]:
            strategy = _build_strategy()
            _write_json(project_dir / "narrative_strategy.json", strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _discover_scenes(
            self, **kwargs: Any
        ) -> SceneDiscoveryArtifact:
            scene_discovery = _build_scene_discovery()
            project_dir = kwargs["project_dir"]
            _write_json(project_dir / "scene_discovery.json", scene_discovery)
            return scene_discovery

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: SynthesisPrimitivesArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            return SynthesisMap(
                project_id=project.project_id,
                primitives=[_build_event_primitive()],
            )

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            return project.model_copy(update={"episode_count": 1})

        async def _plan_series_with_narrative_state(
            self, **_: Any
        ) -> tuple[
            list[EpisodeArchitecture],
            list[Any],
            dict[int, NarrativeState],
            dict[int, NarrativeState],
            dict[str, Any],
        ]:
            state_pre, state_post = _build_narrative_states()
            return (
                [_episode_architecture()],
                [SimpleNamespace(episode_number=1)],
                {1: state_pre},
                {1: state_post},
                {
                    "episode_architecture": {"unknown_actor_ids": 0},
                    "episode_planning": {"unknown_actor_ids": 0},
                },
            )

        async def _produce_episode(
            self,
            plan: Any,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            host_policy: dict[str, Any],
            primitive_lookup: dict[str, Any],
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
            series_explanation_registry: list[Any] | None = None,
            narrative_state_pre: NarrativeState | None = None,
            narrative_state_post: NarrativeState | None = None,
        ) -> tuple[int, Any]:
            raise RuntimeError("boom")

        def _write_passage_utilization(self, **kwargs: Any) -> None:
            return None

        def _build_writing_actor_metrics(
            self,
            project_dir: Path,
            spoken_scripts: list[tuple[int, Any]],
        ) -> dict[str, Any]:
            return {"completed_episode_count": len(spoken_scripts)}

        def _write_actor_metadata_metrics(self, **kwargs: Any) -> None:
            return None

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
            return None

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(
        RuntimeError,
        match="Resume failed from substrate function tagging onward: episode 1: boom",
    ):
        asyncio.run(resume_script._resume_from_substrate_function_tagging("run_1"))
