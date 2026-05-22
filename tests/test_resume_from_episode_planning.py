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
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveFunctionTaggingArtifact,
    PrimitiveSubstrate,
    ProjectStatus,
    StrategyEpisode,
    SynthesisMap,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "resume_from_episode_planning.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_episode_planning",
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


def _episode_architecture() -> EpisodeArchitecture:
    sections: list[ArchitectureSection] = []
    purposes = [
        ("section_1", "opening", "frame", "scene_first", "low", "section_2"),
        (
            "section_2",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_3",
        ),
        (
            "section_3",
            "turn",
            "test_viability",
            "contrast_first",
            "medium",
            "section_4",
        ),
        (
            "section_4",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_5",
        ),
        (
            "section_5",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_6",
        ),
        (
            "section_6",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_7",
        ),
        (
            "section_7",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_8",
        ),
        (
            "section_8",
            "setup",
            "establish_mechanism",
            "mechanism_first",
            "low",
            "section_9",
        ),
        ("section_9", "closing", "close", "aftermath_first", "high", None),
    ]
    for idx, (
        section_id,
        purpose,
        argument_role,
        inference_mode,
        closure_level,
        next_section,
    ) in enumerate(purposes, start=1):
        sections.append(
            ArchitectureSection.model_validate(
                {
                    "section_id": section_id,
                    "purpose": purpose,
                    "approx_runtime_minutes": 0.25,
                    "primitive_ids": ["primitive_1"],
                    "section_question": f"Question {idx}?",
                    "section_resolution": f"Resolution {idx}.",
                    "entry_state": f"Entry {idx}",
                    "exit_state": f"Exit {idx}",
                    "transition_logic": f"Transition {idx}.",
                    "depends_on_section_ids": []
                    if idx == 1
                    else [f"section_{idx - 1}"],
                    "sets_up_section_ids": []
                    if next_section is None
                    else [next_section],
                    "argument_role": argument_role,
                    "inference_mode": inference_mode,
                    "recurrence_role": "none",
                    "pressure_type": "mass_political",
                    "resolution_type": "containment"
                    if purpose == "closing"
                    else "escalation",
                    "closure_level": closure_level,
                }
            )
        )
    return EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_3",
        allowed_recurring_primitive_ids=["primitive_1"],
        forbidden_redundancies=[],
        sections=sections,
        architecture_notes=[],
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
    primitive = EventPrimitive(
        id="primitive_1",
        substrate=PrimitiveSubstrate.EVENTS,
        title="Primitive",
        core_passage_ids=["passage_1"],
        actor_ids=["actor_1"],
        event_type="turning_point",
        what_happened="A decisive event changes the field.",
    )
    tagged_primitives = PrimitiveFunctionTaggingArtifact(
        project_id="run_1",
        primitives=[primitive],
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
        recommended_episode_count=1,
        series_explanation_registry=[
            {
                "item_id": "registry_1",
                "label": "Majles",
                "kind": "institution",
                "importance": "foundational",
                "introduction_episode_number": 1,
                "preferred_plain_gloss": "the parliament",
            }
        ],
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
    _write_json(project_dir / "tagged_primitives.json", tagged_primitives)
    _write_json(project_dir / "narrative_strategy.json", strategy)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(
        project_dir / "actor_metadata_metrics.json",
        {"stage_metrics": {"narrative_strategy": {"unknown_actor_ids": 0}}},
    )
    return project_dir


def test_resume_from_episode_planning_passes_host_policy(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: PrimitiveFunctionTaggingArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            calls["materialize_config"] = project.config
            calls["materialize_input"] = synthesis_primitives
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives=[
                    EventPrimitive(
                        id="primitive_1",
                        substrate=PrimitiveSubstrate.EVENTS,
                        title="Primitive",
                        core_passage_ids=["passage_1"],
                        actor_ids=["actor_1"],
                        event_type="turning_point",
                        what_happened="A decisive event changes the field.",
                        event_result="The balance breaks.",
                    )
                ],
            )
            return synthesis_map

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            calls["resolved_strategy"] = strategy
            return project.model_copy(
                update={
                    "episode_count": 1,
                    "recommended_episode_count": strategy.recommended_episode_count,
                }
            )

        async def _build_episode_architectures(
            self,
            **kwargs: Any,
        ) -> tuple[list[EpisodeArchitecture], dict[str, Any]]:
            calls["architecture_config"] = kwargs["project"].config
            calls["architecture_strategy"] = kwargs["strategy"]
            architecture = _episode_architecture()
            return [architecture], {"unknown_actor_ids": 0}

        async def _plan_series(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
            calls["planning_config"] = kwargs["project"].config
            calls["planning_architectures"] = kwargs["episode_architectures"]
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
            primitive_lookup: dict[str, EventPrimitive],
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
            series_explanation_registry: list[Any] | None = None,
        ) -> tuple[int, Any]:
            calls["production_config"] = project.config
            calls["host_policy"] = host_policy
            calls["production_actor_metadata"] = actor_metadata
            calls["production_architecture"] = architecture
            calls["primitive_lookup"] = primitive_lookup
            calls["series_explanation_registry"] = series_explanation_registry
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

    asyncio.run(resume_script._resume_from_episode_planning("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert calls["materialize_config"].skip_grounding is True
    assert calls["materialize_input"].primitives[0].id == "primitive_1"
    assert calls["architecture_config"].skip_spoken_delivery is True
    assert calls["planning_config"].skip_grounding is True
    assert calls["planning_architectures"][0].episode_number == 1
    assert calls["production_config"].skip_audio is True
    assert calls["production_config"].skip_spoken_delivery is True
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_architecture"].episode_number == 1
    assert calls["primitive_lookup"]["primitive_1"].id == "primitive_1"
    assert calls["series_explanation_registry"][0].item_id == "registry_1"
    assert "authorial_policy" in calls["host_policy"]
    assert calls["audio_skip"] is True
    assert calls["actor_metadata_metrics"]["metrics"]["narrative_strategy"] == {
        "unknown_actor_ids": 0
    }
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


def test_resume_from_episode_planning_marks_failed_when_planning_raises(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: PrimitiveFunctionTaggingArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            return SynthesisMap(project_id=project.project_id, primitives=[])

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            return project.model_copy(update={"episode_count": 1})

        async def _build_episode_architectures(
            self,
            **kwargs: Any,
        ) -> tuple[list[EpisodeArchitecture], dict[str, Any]]:
            return [_episode_architecture()], {"unknown_actor_ids": 0}

        async def _plan_series(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(resume_script._resume_from_episode_planning("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED


def test_resume_from_episode_planning_fails_on_partial_production_completion(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

        async def _materialize_selected_primitives(
            self,
            *,
            project: ThematicProject,
            synthesis_primitives: PrimitiveFunctionTaggingArtifact,
            strategy: NarrativeStrategy,
            project_dir: Path,
        ) -> SynthesisMap:
            return SynthesisMap(
                project_id=project.project_id,
                primitives=[
                    EventPrimitive(
                        id="primitive_1",
                        substrate=PrimitiveSubstrate.EVENTS,
                        title="Primitive",
                        core_passage_ids=["passage_1"],
                        actor_ids=["actor_1"],
                        event_type="turning_point",
                        what_happened="A decisive event changes the field.",
                        event_result="The balance breaks.",
                    )
                ],
            )

        def _resolve_episode_count_from_strategy(
            self,
            project: ThematicProject,
            strategy: NarrativeStrategy,
        ) -> ThematicProject:
            return project.model_copy(update={"episode_count": 1})

        async def _build_episode_architectures(
            self,
            **kwargs: Any,
        ) -> tuple[list[EpisodeArchitecture], dict[str, Any]]:
            return [_episode_architecture()], {"unknown_actor_ids": 0}

        async def _plan_series(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
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
            primitive_lookup: dict[str, EventPrimitive],
            semaphore: asyncio.Semaphore,
            spoken_semaphore: asyncio.Semaphore | None = None,
            series_explanation_registry: list[Any] | None = None,
        ) -> tuple[int, Any]:
            return 99, SimpleNamespace(episode_number=99)

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
        match=r"completed episode numbers did not match planned episodes \(\[99\] != \[1\]\)",
    ):
        asyncio.run(resume_script._resume_from_episode_planning("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED


def test_resume_from_episode_planning_requires_tagged_primitives(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    (project_dir / "tagged_primitives.json").unlink()

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(
        RuntimeError,
        match=r"Missing required artifact for strict resume: .*tagged_primitives\.json",
    ):
        asyncio.run(resume_script._resume_from_episode_planning("run_1"))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.FAILED
