from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
    Path(__file__).resolve().parents[1] / "scripts" / "resume_from_synthesis_mapping.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_synthesis_mapping",
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
    return project_dir


def _build_synthesis_primitives() -> SynthesisPrimitivesArtifact:
    primitive = EventPrimitive(
        id="primitive_1",
        substrate=PrimitiveSubstrate.EVENTS,
        title="Primitive",
        core_passage_ids=["passage_1"],
        actor_ids=["actor_1"],
        event_type="turning_point",
        what_happened="A decisive event changes the situation.",
        event_result="The balance breaks.",
    )
    return SynthesisPrimitivesArtifact(project_id="run_1", primitives=[primitive])


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
                "last_episode_takeaway": "",
            },
            "host": {
                "mysteries": [],
                "assumptions": [],
                "working_theories": [],
                "recent_revisions": [],
                "confidence_posture": "mixed",
                "last_episode_takeaway": "",
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


def test_resume_from_synthesis_mapping_uses_artifacts_and_forces_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {"order": []}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            self.run_logger = SimpleNamespace(log=lambda _event, **kwargs: None)
            self.run_logger = SimpleNamespace(log=lambda _event, **kwargs: None)

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _map_synthesis(
            self,
            *,
            project: ThematicProject,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
            calls["order"].append("map_synthesis")
            calls["map_config"] = project.config
            calls["map_corpus"] = corpus
            calls["map_actor_metadata"] = actor_metadata
            primitives = _build_synthesis_primitives()
            _write_json(project_dir / "synthesis_primitives.json", primitives)
            return primitives, {
                "primitives": {"unknown_actor_ids": 0},
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
            strategy = NarrativeStrategy(
                strategy_type="chronological",
                justification="Test",
                series_arc="Test arc",
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode",
                        arc_summary="Arc summary",
                        episode_spine={
                            "listener_question": "Question?",
                            "argument": "A working claim.",
                            "core_primitive_ids": [
                                "primitive_1",
                                "core_2",
                                "core_3",
                                "core_4",
                                "core_5",
                                "core_6",
                                "core_7",
                            ],
                            "support_primitive_roles": {
                                f"support_{idx}": "mechanism" for idx in range(1, 8)
                            },
                            "recall_primitive_ids": [],
                        },
                    )
                ],
            )
            _write_json(project_dir / "narrative_strategy.json", strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _discover_scenes(
            self, **_: Any
        ) -> SceneDiscoveryArtifact:
            calls["order"].append("scene_discovery")
            scene_discovery = _build_scene_discovery()
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
            primitive = EventPrimitive(
                id="primitive_1",
                substrate=PrimitiveSubstrate.EVENTS,
                title="Primitive",
                core_passage_ids=["passage_1"],
                actor_ids=["actor_1"],
                event_type="turning_point",
                what_happened="A decisive event changes the situation.",
                event_result="The balance breaks.",
            )
            synthesis_map = SynthesisMap(project_id=project.project_id, primitives=[primitive])
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
            architecture = EpisodeArchitecture.model_construct(
                episode_number=1,
                major_turn_section_id="section_03",
                allowed_recurring_primitive_ids=["primitive_1"],
                forbidden_redundancies=[],
                sections=[
                    ArchitectureSection.model_validate(section)
                    for section in [
                        {
                            "section_id": "section_01",
                            "purpose": "opening",
                            "approx_runtime_minutes": 10.0,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q1?",
                            "section_resolution": "R1",
                            "entry_state": "E1",
                            "exit_state": "X1",
                            "transition_logic": "T1",
                            "depends_on_section_ids": [],
                            "sets_up_section_ids": ["section_02"],
                            "argument_role": "frame",
                            "inference_mode": "scene_first",
                            "recurrence_role": "plant",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_02",
                            "purpose": "setup",
                            "approx_runtime_minutes": 10.0,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q2?",
                            "section_resolution": "R2",
                            "entry_state": "E2",
                            "exit_state": "X2",
                            "transition_logic": "T2",
                            "depends_on_section_ids": ["section_01"],
                            "sets_up_section_ids": ["section_03"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_03",
                            "purpose": "turn",
                            "approx_runtime_minutes": 12.5,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q3?",
                            "section_resolution": "R3",
                            "entry_state": "E3",
                            "exit_state": "X3",
                            "transition_logic": "T3",
                            "depends_on_section_ids": ["section_02"],
                            "sets_up_section_ids": ["section_04"],
                            "argument_role": "test_viability",
                            "inference_mode": "contrast_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "reversal",
                            "closure_level": "medium",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_04",
                            "purpose": "setup",
                            "approx_runtime_minutes": 12.5,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q4?",
                            "section_resolution": "R4",
                            "entry_state": "E4",
                            "exit_state": "X4",
                            "transition_logic": "T4",
                            "depends_on_section_ids": ["section_03"],
                            "sets_up_section_ids": ["section_05"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_05",
                            "purpose": "setup",
                            "approx_runtime_minutes": 10.0,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q5?",
                            "section_resolution": "R5",
                            "entry_state": "E5",
                            "exit_state": "X5",
                            "transition_logic": "T5",
                            "depends_on_section_ids": ["section_04"],
                            "sets_up_section_ids": ["section_06"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_06",
                            "purpose": "setup",
                            "approx_runtime_minutes": 10.0,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q6?",
                            "section_resolution": "R6",
                            "entry_state": "E6",
                            "exit_state": "X6",
                            "transition_logic": "T6",
                            "depends_on_section_ids": ["section_05"],
                            "sets_up_section_ids": ["section_07"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_07",
                            "purpose": "setup",
                            "approx_runtime_minutes": 7.5,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q7?",
                            "section_resolution": "R7",
                            "entry_state": "E7",
                            "exit_state": "X7",
                            "transition_logic": "T7",
                            "depends_on_section_ids": ["section_06"],
                            "sets_up_section_ids": ["section_08"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_08",
                            "purpose": "setup",
                            "approx_runtime_minutes": 7.5,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q8?",
                            "section_resolution": "R8",
                            "entry_state": "E8",
                            "exit_state": "X8",
                            "transition_logic": "T8",
                            "depends_on_section_ids": ["section_07"],
                            "sets_up_section_ids": ["section_09"],
                            "argument_role": "establish_mechanism",
                            "inference_mode": "mechanism_first",
                            "recurrence_role": "deepen",
                            "pressure_type": "mass_political",
                            "resolution_type": "escalation",
                            "closure_level": "low",
                            "priority_core_passage_ids": [],
                        },
                        {
                            "section_id": "section_09",
                            "purpose": "closing",
                            "approx_runtime_minutes": 10.0,
                            "primitive_ids": ["primitive_1"],
                            "section_question": "Q9?",
                            "section_resolution": "R9",
                            "entry_state": "E9",
                            "exit_state": "X9",
                            "transition_logic": "T9",
                            "depends_on_section_ids": ["section_08"],
                            "sets_up_section_ids": [],
                            "argument_role": "close",
                            "inference_mode": "aftermath_first",
                            "recurrence_role": "payoff",
                            "pressure_type": "mass_political",
                            "resolution_type": "containment",
                            "closure_level": "high",
                            "priority_core_passage_ids": [],
                        },
                    ]
                ],
                architecture_notes=[],
            )
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            calls["order"].append("plan_series")
            calls["planning_actor_metadata"] = kwargs["actor_metadata"]
            calls["planning_config"] = kwargs["project"].config
            calls["planning_strategy"] = kwargs["strategy"]
            calls["planning_architectures"] = [architecture]
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
            calls["production_strategy_episode"] = strategy_episode
            calls["production_architecture"] = architecture
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

    asyncio.run(resume_script._resume_from_synthesis_mapping("run_1"))

    assert calls["order"] == [
        "map_synthesis",
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
    assert calls["map_corpus"].axes[0].axis_id == "axis_1"
    assert calls["map_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["narrative_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["architecture_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_strategy"].episodes[0].title == "Episode"
    assert calls["planning_architectures"][0].runtime_minutes == 90.0
    assert calls["production_strategy_episode"].title == "Episode"
    assert calls["primitive_lookup"]["primitive_1"].id == "primitive_1"
    assert calls["map_config"].skip_grounding is True
    assert calls["map_config"].skip_audio is True
    assert calls["map_config"].skip_spoken_delivery is False
    assert calls["map_config"].synthesis_total_passage_cap == 321
    assert calls["planning_config"].skip_audio is True
    assert calls["planning_config"].synthesis_total_passage_cap == 321
    assert calls["production_config"].skip_grounding is True
    assert calls["production_config"].synthesis_total_passage_cap == 321
    assert "authorial_policy" in calls["host_policy"]
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["synthesis_primitives"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["episode_architecture"] == {
        "unknown_actor_ids": 0
    }

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
    assert final_project.requested_episode_count is None
    assert final_project.episode_count == 1


def test_resume_from_synthesis_mapping_rejects_axis_mismatch(
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
            self.run_logger = SimpleNamespace(log=lambda _event, **kwargs: None)
            self.run_logger = SimpleNamespace(log=lambda _event, **kwargs: None)

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            raise AssertionError("resume should fail before binding run logger")

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="disagree on axes"):
        asyncio.run(resume_script._resume_from_synthesis_mapping("run_1"))


def test_resume_from_synthesis_mapping_requires_actor_metadata(
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
        asyncio.run(resume_script._resume_from_synthesis_mapping("run_1"))


def test_resume_from_synthesis_mapping_fails_when_upstream_artifact_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            self.run_logger = SimpleNamespace(log=lambda _event, **kwargs: None)

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            assert bound_project_dir == project_dir

        async def _map_synthesis(
            self,
            *,
            project: ThematicProject,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
            mutated_metadata = ActorMetadata(project_id="run_1", actors=[])
            _write_json(project_dir / "actor_metadata.json", mutated_metadata)
            primitives = _build_synthesis_primitives()
            _write_json(project_dir / "synthesis_primitives.json", primitives)
            return primitives, {"primitives": {"unknown_actor_ids": 0}}

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisPrimitivesArtifact,
            project_dir: Path,
            actor_metadata: ActorMetadata,
            scene_discovery: SceneDiscoveryArtifact | None = None,
        ) -> tuple[NarrativeStrategy, dict[str, Any]]:
            strategy = NarrativeStrategy(
                strategy_type="chronological",
                justification="Test",
                series_arc="Test arc",
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode",
                        arc_summary="Arc summary",
                        episode_spine={
                            "listener_question": "Question?",
                            "argument": "A working claim.",
                            "core_primitive_ids": [
                                "primitive_1",
                                "core_2",
                                "core_3",
                                "core_4",
                                "core_5",
                                "core_6",
                                "core_7",
                            ],
                            "support_primitive_roles": {
                                f"support_{idx}": "mechanism" for idx in range(1, 8)
                            },
                            "recall_primitive_ids": [],
                        },
                    )
                ],
            )
            _write_json(project_dir / "narrative_strategy.json", strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _discover_scenes(
            self, **_: Any
        ) -> SceneDiscoveryArtifact:
            scene_discovery = _build_scene_discovery()
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
            primitive = EventPrimitive(
                id="primitive_1",
                substrate=PrimitiveSubstrate.EVENTS,
                title="Primitive",
                core_passage_ids=["passage_1"],
                actor_ids=["actor_1"],
                event_type="turning_point",
                what_happened="A decisive event changes the situation.",
                event_result="The balance breaks.",
            )
            synthesis_map = SynthesisMap(project_id=project.project_id, primitives=[primitive])
            _write_json(project_dir / "retained_primitives.json", synthesis_map)
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
            architecture = EpisodeArchitecture.model_construct(
                episode_number=1,
                major_turn_section_id="section_01",
                allowed_recurring_primitive_ids=["primitive_1"],
                forbidden_redundancies=[],
                sections=[],
                architecture_notes=[],
            )
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
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
        match="Strict integrity check failed: actor_metadata.json changed during resume",
    ):
        asyncio.run(resume_script._resume_from_synthesis_mapping("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED
