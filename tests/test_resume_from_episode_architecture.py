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
    ArchitectureSection,
    BaseSynthesisPrimitive,
    BookRecord,
    EpisodeArchitecture,
    EpochalTurnPrimitive,
    NarrativeStrategy,
    PipelineConfig,
    ProjectStatus,
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
    / "resume_from_episode_architecture.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_episode_architecture",
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
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Test arc",
        recommended_episode_count=1,
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
    architecture = _episode_architecture()

    _write_json(
        project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]}
    )
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "synthesis_primitives.json", primitives)
    _write_json(project_dir / "synthesis_map.json", synthesis_map)
    _write_json(project_dir / "narrative_strategy.json", strategy)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(
        project_dir / "episode_architectures.json",
        {"episodes": [architecture.model_dump(mode="json")]},
    )
    return project_dir


def test_resume_from_episode_architecture_passes_host_policy(
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

        async def _plan_series(self, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
            calls["planning_config"] = kwargs["project"].config
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
            calls["production_config"] = project.config
            calls["host_policy"] = host_policy
            calls["production_actor_metadata"] = actor_metadata
            calls["production_architecture"] = architecture
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

    asyncio.run(resume_script._resume_from_episode_architecture("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert calls["planning_config"].skip_grounding is True
    assert calls["production_config"].skip_spoken_delivery is False
    assert calls["production_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["production_architecture"].episode_number == 1
    assert "authorial_policy" in calls["host_policy"]
    assert calls["audio_skip"] is True

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
