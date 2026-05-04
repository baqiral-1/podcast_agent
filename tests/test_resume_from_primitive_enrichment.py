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
    / "resume_from_primitive_enrichment.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_primitive_enrichment",
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
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Persisted Episode",
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
                        f"support_{idx}": "mechanism" for idx in range(1, 7)
                    }
                    | {"primitive_support": "mechanism"},
                    "recall_primitive_ids": ["primitive_recall"],
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
        recommended_episode_count=1,
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
    _write_json(project_dir / "narrative_strategy.json", strategy)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    return project_dir


def _episode_architecture() -> EpisodeArchitecture:
    payload = {
        "episode_number": 1,
        "major_turn_section_id": "section_3",
        "allowed_recurring_primitive_ids": ["primitive_1"],
        "forbidden_redundancies": [],
        "sections": [
            {
                "section_id": "section_1",
                "purpose": "opening",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step one.",
                "entry_state": "Start",
                "exit_state": "Move",
                "transition_logic": "Advance.",
                "depends_on_section_ids": [],
                "sets_up_section_ids": ["section_2"],
                "argument_role": "frame",
                "inference_mode": "scene_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_2",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step two.",
                "entry_state": "Move",
                "exit_state": "Turn",
                "transition_logic": "Escalate.",
                "depends_on_section_ids": ["section_1"],
                "sets_up_section_ids": ["section_3"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_3",
                "purpose": "turn",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step three.",
                "entry_state": "Turn",
                "exit_state": "Close",
                "transition_logic": "Pivot.",
                "depends_on_section_ids": ["section_2"],
                "sets_up_section_ids": ["section_4"],
                "argument_role": "test_viability",
                "inference_mode": "contrast_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "reversal",
                "closure_level": "medium",
            },
            {
                "section_id": "section_4",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step four.",
                "entry_state": "Close",
                "exit_state": "Extend",
                "transition_logic": "Press further.",
                "depends_on_section_ids": ["section_3"],
                "sets_up_section_ids": ["section_5"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_5",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step five.",
                "entry_state": "Extend",
                "exit_state": "Resolve",
                "transition_logic": "Narrow to the end.",
                "depends_on_section_ids": ["section_4"],
                "sets_up_section_ids": ["section_6"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_6",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step six.",
                "entry_state": "Resolve",
                "exit_state": "Refine",
                "transition_logic": "Press tighter.",
                "depends_on_section_ids": ["section_5"],
                "sets_up_section_ids": ["section_7"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_7",
                "purpose": "setup",
                "approx_runtime_minutes": 0.125,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step seven.",
                "entry_state": "Refine",
                "exit_state": "Stage close",
                "transition_logic": "Set up the close.",
                "depends_on_section_ids": ["section_6"],
                "sets_up_section_ids": ["section_8"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_8",
                "purpose": "setup",
                "approx_runtime_minutes": 0.125,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Stage the close.",
                "entry_state": "Stage close",
                "exit_state": "End",
                "transition_logic": "Hand off to the ending.",
                "depends_on_section_ids": ["section_7"],
                "sets_up_section_ids": ["section_9"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": "section_9",
                "purpose": "closing",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Resolved.",
                "entry_state": "End",
                "exit_state": "End",
                "transition_logic": "Close the loop.",
                "depends_on_section_ids": ["section_8"],
                "sets_up_section_ids": [],
                "argument_role": "close",
                "inference_mode": "aftermath_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "containment",
                "closure_level": "high",
            },
        ],
        "architecture_notes": [],
    }
    return EpisodeArchitecture.model_construct(
        episode_number=payload["episode_number"],
        major_turn_section_id=payload["major_turn_section_id"],
        allowed_recurring_primitive_ids=payload["allowed_recurring_primitive_ids"],
        forbidden_redundancies=payload["forbidden_redundancies"],
        sections=[
            ArchitectureSection.model_validate(section)
            for section in payload["sections"]
        ],
        architecture_notes=payload["architecture_notes"],
    )


def test_resume_from_primitive_enrichment_uses_persisted_strategy_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

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
            calls["enrichment_strategy"] = strategy
            calls["enrichment_actor_metadata"] = actor_metadata
            calls["enrichment_config"] = project.config
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
            return project.model_copy(
                update={
                    "episode_count": strategy.recommended_episode_count,
                    "recommended_episode_count": strategy.recommended_episode_count,
                }
            )

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
            architecture = _episode_architecture()
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            return [architecture], {"unknown_actor_ids": 0}

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
            calls["planning_strategy"] = strategy
            calls["planning_architectures"] = episode_architectures
            calls["planning_actor_metadata"] = actor_metadata
            calls["planning_config"] = project.config
            return [SimpleNamespace(episode_number=1)], {"unknown_actor_ids": 0}

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
            calls["production_strategy_episode"] = strategy_episode
            calls["production_architecture"] = architecture
            calls["production_actor_metadata"] = actor_metadata
            calls["production_config"] = project.config
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

    asyncio.run(resume_script._resume_from_primitive_enrichment("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert calls["enrichment_strategy"].episodes[0].title == "Persisted Episode"
    assert calls["resolved_strategy"].episodes[0].title == "Persisted Episode"
    assert calls["planning_strategy"].episodes[0].title == "Persisted Episode"
    assert (
        calls["planning_strategy"]
        .episodes[0]
        .episode_spine.support_primitive_roles["primitive_support"]
        == SupportPrimitiveRole.MECHANISM
    )
    assert calls["planning_strategy"].episodes[
        0
    ].episode_spine.recall_primitive_ids == ["primitive_recall"]
    assert calls["planning_architectures"][0].runtime_minutes == 2.0
    assert calls["enrichment_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["architecture_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_strategy_episode"].title == "Persisted Episode"
    assert calls["production_architecture"].episode_number == 1
    assert "authorial_policy" in calls["host_policy"]
    assert calls["host_policy"]["authorial_policy"]["host_moves_are_secondary"] is True
    assert calls["enrichment_config"].skip_grounding is True
    assert calls["enrichment_config"].skip_audio is True
    assert calls["enrichment_config"].skip_spoken_delivery is False
    assert calls["planning_config"].skip_grounding is True
    assert calls["production_config"].skip_audio is True
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["episode_architecture"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["episode_planning"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["writing"] == {
        "completed_episode_count": 1
    }

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
    assert final_project.episode_count == 1


def test_resume_from_primitive_enrichment_fails_when_persisted_strategy_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    _write_json(
        project_dir / "narrative_strategy.json",
        {
            "strategy_type": "chronological",
            "justification": "Broken persisted strategy",
            "series_arc": "Broken",
            "episodes": [],
        },
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

    with pytest.raises(RuntimeError, match="must contain at least one episode"):
        asyncio.run(resume_script._resume_from_primitive_enrichment("run_1"))


def test_resume_from_primitive_enrichment_requires_actor_metadata(
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
        asyncio.run(resume_script._resume_from_primitive_enrichment("run_1"))


def test_resume_from_primitive_enrichment_rejects_architecture_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

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
            return project.model_copy(
                update={
                    "episode_count": 2,
                    "recommended_episode_count": 2,
                }
            )

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
            architecture = _episode_architecture()
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            return [architecture], {"unknown_actor_ids": 0}

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="do not match the resolved episode count"):
        asyncio.run(resume_script._resume_from_primitive_enrichment("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED
