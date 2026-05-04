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
    BaseSynthesisPrimitive,
    BookRecord,
    EpisodeArchitecture,
    EpochalTurnPrimitive,
    NarrativeStrategy,
    PipelineConfig,
    ProjectStatus,
    SupportPrimitiveRole,
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
    primitive = BaseSynthesisPrimitive(
        id="primitive_1",
        family="epochal_turns",
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
    enriched_primitive = EpochalTurnPrimitive(
        id="primitive_1",
        family="epochal_turns",
        title="Primitive",
        summary="Primitive summary",
        axis_ids=["axis_1"],
        core_passage_ids=[],
        actor_ids=["actor_1"],
        before_state="The prior balance still holds.",
        after_state="The balance breaks.",
        change_driver="A decisive move forces the turn.",
        irreversibility_reason="The fallout cannot be unwound quickly.",
    )
    synthesis_map = SynthesisMap(
        project_id="run_1",
        primitives_by_family={"epochal_turns": [enriched_primitive]},
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

    _write_json(
        project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]}
    )
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
            synthesis_map: SynthesisPrimitivesArtifact,
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
            persisted_strategy = strategy.model_copy(
                update={
                    "episodes": [
                        strategy.episodes[0].model_copy(
                            update={
                                "title": "Persisted Episode",
                                "episode_spine": strategy.episodes[
                                    0
                                ].episode_spine.model_copy(
                                    update={
                                        "support_primitive_roles": {
                                            **{
                                                key: value
                                                for idx, (key, value) in enumerate(
                                                    strategy.episodes[
                                                        0
                                                    ].episode_spine.support_primitive_roles.items(),
                                                    start=1,
                                                )
                                                if idx <= 6
                                            },
                                            "primitive_support": SupportPrimitiveRole.MECHANISM,
                                        },
                                        "recall_primitive_ids": ["primitive_recall"],
                                    }
                                ),
                            }
                        )
                    ]
                }
            )
            _write_json(project_dir / "narrative_strategy.json", persisted_strategy)
            return strategy, {"unknown_actor_ids": 0}

        async def _enrich_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: SynthesisPrimitivesArtifact,
            strategy: NarrativeStrategy,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> SynthesisMap:
            primitive = EpochalTurnPrimitive(
                id="primitive_1",
                family="epochal_turns",
                title="Primitive",
                summary="Primitive summary",
                axis_ids=["axis_1"],
                core_passage_ids=[],
                actor_ids=["actor_1"],
                before_state="The prior balance still holds.",
                after_state="The balance breaks.",
                change_driver="A decisive move forces the turn.",
                irreversibility_reason="The fallout cannot be unwound quickly.",
            )
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives_by_family={"epochal_turns": [primitive]},
            )
            _write_json(project_dir / "synthesis_map.json", synthesis_map)
            return synthesis_map

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
            episode_architectures: list[EpisodeArchitecture],
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[list[Any], dict[str, Any]]:
            calls["planning_actor_metadata"] = actor_metadata
            calls["planning_config"] = project.config
            calls["planning_strategy"] = strategy
            calls["planning_architectures"] = episode_architectures
            return [SimpleNamespace(episode_number=1)], {"unknown_actor_ids": 0}

        async def _build_episode_architectures(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisMap,
            strategy: NarrativeStrategy,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> tuple[list[EpisodeArchitecture], dict[str, Any]]:
            calls["architecture_actor_metadata"] = actor_metadata
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
            return [architecture], {"unknown_actor_ids": 0}

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
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
        ) -> tuple[int, Any]:
            calls["production_actor_metadata"] = actor_metadata
            calls["production_config"] = project.config
            calls["production_strategy_episode"] = strategy_episode
            calls["production_architecture"] = architecture
            calls["host_policy"] = host_policy
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
    assert calls["architecture_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["resolved_strategy"].episodes[0].title == "Episode"
    assert calls["planning_strategy"].episodes[0].title == "Episode"
    assert calls["planning_architectures"][0].runtime_minutes == 90.0
    assert calls["production_strategy_episode"].title == "Episode"
    assert calls["production_architecture"].episode_number == 1
    assert "authorial_policy" in calls["host_policy"]
    assert calls["host_policy"]["authorial_policy"]["host_moves_are_secondary"] is True
    assert calls["narrative_config"].skip_grounding is True
    assert calls["narrative_config"].skip_audio is True
    assert calls["narrative_config"].skip_spoken_delivery is False
    assert calls["planning_config"].skip_grounding is True
    assert calls["production_config"].skip_audio is True
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["narrative_strategy"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["episode_architecture"] == {
        "unknown_actor_ids": 0
    }

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE


def test_resume_from_narrative_strategy_fails_before_planning_when_persisted_strategy_is_empty(
    monkeypatch,
    tmp_path,
):
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {"planning_called": False}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

        async def _choose_narrative_strategy(
            self,
            *,
            project: ThematicProject,
            synthesis_map: SynthesisPrimitivesArtifact,
            project_dir: Path,
            actor_metadata: ActorMetadata,
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
            _write_json(
                project_dir / "narrative_strategy.json",
                {
                    "strategy_type": "chronological",
                    "justification": "Broken persisted strategy",
                    "series_arc": "Broken",
                    "episodes": [],
                },
            )
            return strategy, {"unknown_actor_ids": 0}

        async def _enrich_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: SynthesisPrimitivesArtifact,
            strategy: NarrativeStrategy,
            corpus: ThematicCorpus,
            project_dir: Path,
            actor_metadata: ActorMetadata,
        ) -> SynthesisMap:
            primitive = EpochalTurnPrimitive(
                id="primitive_1",
                family="epochal_turns",
                title="Primitive",
                summary="Primitive summary",
                axis_ids=["axis_1"],
                core_passage_ids=[],
                actor_ids=["actor_1"],
                before_state="The prior balance still holds.",
                after_state="The balance breaks.",
                change_driver="A decisive move forces the turn.",
                irreversibility_reason="The fallout cannot be unwound quickly.",
            )
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives_by_family={"epochal_turns": [primitive]},
            )
            _write_json(project_dir / "synthesis_map.json", synthesis_map)
            return synthesis_map

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            return project

        async def _plan_series(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
            calls["planning_called"] = True
            return [], {}

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="must contain at least one episode"):
        asyncio.run(resume_script._resume_from_narrative_strategy("run_1"))

    assert calls["planning_called"] is False
    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED
