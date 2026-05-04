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
    EpochalTurnPrimitive,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeScript,
    NarrativeStrategy,
    PipelineConfig,
    ProjectStatus,
    SpokenScript,
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
    / "retry_single_episode_from_writing.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "retry_single_episode_from_writing",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
retry_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(retry_script)


def _write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode_plan(episode_number: int) -> EpisodePlan:
    scene_id = f"scene_{episode_number}"
    return EpisodePlan.model_validate(
        {
            "episode_number": episode_number,
            "framing": {
                "opening_image": f"Image {episode_number}",
                "threat_or_unresolved_action": f"Threat {episode_number}",
                "opening_question": f"Question {episode_number}",
                "handoff_scene_card_id": scene_id,
            },
            "scene_cards": [
                {
                    "scene_id": scene_id,
                    "section_id": f"section_{episode_number}_1",
                    "title": f"Scene {episode_number}",
                    "scene_role": "setup",
                    "dominant_primitive_id": "primitive_1",
                    "spine_relation": "set_stakes",
                    "state_effect": "The stakes become legible.",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": [],
                    "estimated_duration_seconds": 60,
                }
            ],
            "target_word_count": 150,
        }
    )


def _episode_architecture(episode_number: int) -> EpisodeArchitecture:
    prefix = f"section_{episode_number}_"
    payload = {
        "episode_number": episode_number,
        "major_turn_section_id": f"{prefix}3",
        "allowed_recurring_primitive_ids": [],
        "forbidden_redundancies": [],
        "sections": [
            {
                "section_id": f"{prefix}1",
                "purpose": "opening",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step one.",
                "entry_state": "Start",
                "exit_state": "Move",
                "transition_logic": "Advance.",
                "depends_on_section_ids": [],
                "sets_up_section_ids": [f"{prefix}2"],
                "argument_role": "frame",
                "inference_mode": "scene_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}2",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step two.",
                "entry_state": "Move",
                "exit_state": "Turn",
                "transition_logic": "Escalate.",
                "depends_on_section_ids": [f"{prefix}1"],
                "sets_up_section_ids": [f"{prefix}3"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}3",
                "purpose": "turn",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step three.",
                "entry_state": "Turn",
                "exit_state": "Close",
                "transition_logic": "Pivot.",
                "depends_on_section_ids": [f"{prefix}2"],
                "sets_up_section_ids": [f"{prefix}4"],
                "argument_role": "test_viability",
                "inference_mode": "contrast_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "reversal",
                "closure_level": "medium",
            },
            {
                "section_id": f"{prefix}4",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step four.",
                "entry_state": "Close",
                "exit_state": "Extend",
                "transition_logic": "Press further.",
                "depends_on_section_ids": [f"{prefix}3"],
                "sets_up_section_ids": [f"{prefix}5"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}5",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step five.",
                "entry_state": "Extend",
                "exit_state": "Resolve",
                "transition_logic": "Narrow to the end.",
                "depends_on_section_ids": [f"{prefix}4"],
                "sets_up_section_ids": [f"{prefix}6"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}6",
                "purpose": "setup",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step six.",
                "entry_state": "Resolve",
                "exit_state": "Refine",
                "transition_logic": "Press tighter.",
                "depends_on_section_ids": [f"{prefix}5"],
                "sets_up_section_ids": [f"{prefix}7"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}7",
                "purpose": "setup",
                "approx_runtime_minutes": 0.125,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Step seven.",
                "entry_state": "Refine",
                "exit_state": "Stage close",
                "transition_logic": "Set up the close.",
                "depends_on_section_ids": [f"{prefix}6"],
                "sets_up_section_ids": [f"{prefix}8"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}8",
                "purpose": "setup",
                "approx_runtime_minutes": 0.125,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Stage the close.",
                "entry_state": "Stage close",
                "exit_state": "End",
                "transition_logic": "Hand off to the ending.",
                "depends_on_section_ids": [f"{prefix}7"],
                "sets_up_section_ids": [f"{prefix}9"],
                "argument_role": "establish_mechanism",
                "inference_mode": "mechanism_first",
                "recurrence_role": "none",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "closure_level": "low",
            },
            {
                "section_id": f"{prefix}9",
                "purpose": "closing",
                "approx_runtime_minutes": 0.25,
                "primitive_ids": ["primitive_1"],
                "section_question": "Question?",
                "section_resolution": "Resolved.",
                "entry_state": "End",
                "exit_state": "End",
                "transition_logic": "Close the loop.",
                "depends_on_section_ids": [f"{prefix}8"],
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
        sections=[ArchitectureSection.model_validate(section) for section in payload["sections"]],
        architecture_notes=payload["architecture_notes"],
    )


def _episode_script(plan: EpisodePlan, title: str) -> EpisodeScript:
    return EpisodeScript.model_validate(
        {
            "episode_number": plan.episode_number,
            "title": title,
            "framing": plan.framing.model_dump(mode="json"),
            "prose_sections": [
                {
                    "section_id": f"section_{plan.episode_number}_spoken",
                    "scene_card_ids": [plan.scene_cards[0].scene_id],
                    "movement_goal": "Advance the argument.",
                    "text": "A short section.",
                    "citations": [],
                    "source_book_ids": ["book_1"],
                }
            ],
            "total_word_count": 3,
            "estimated_duration_seconds": 1,
        }
    )


def _spoken_script(plan: EpisodePlan, title: str) -> SpokenScript:
    return SpokenScript.model_validate(
        {
            "episode_number": plan.episode_number,
            "title": title,
            "framing": plan.framing.model_dump(mode="json"),
            "sections": [
                {
                    "section_id": f"section_{plan.episode_number}_spoken",
                    "text": "A spoken section.",
                }
            ],
            "tts_provider": "openai",
        }
    )


def _build_project_dir(tmp_path: Path) -> tuple[Path, list[EpisodePlan]]:
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
    plans = [_episode_plan(1), _episode_plan(2)]
    architectures = [_episode_architecture(1), _episode_architecture(2)]
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc summary 1",
                episode_spine={
                    "listener_question": "Question 1?",
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
            ),
            StrategyEpisode(
                episode_number=2,
                title="Episode 2",
                arc_summary="Arc summary 2",
                episode_spine={
                    "listener_question": "Question 2?",
                    "argument": "Another working claim.",
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
            ),
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
        requested_episode_count=2,
        config=PipelineConfig(
            skip_grounding=False,
            skip_audio=False,
            skip_spoken_delivery=True,
        ),
        status=ProjectStatus.COMPLETE,
    )
    corpus = ThematicCorpus(project_id="run_1", axes=[axis])

    _write_json(project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]})
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "synthesis_primitives.json", primitives)
    _write_json(project_dir / "synthesis_map.json", synthesis_map)
    _write_json(project_dir / "narrative_strategy.json", strategy)
    _write_json(
        project_dir / "episode_architectures.json",
        {"episodes": [architecture.model_dump(mode="json") for architecture in architectures]},
    )
    _write_json(
        project_dir / "series_plan.json",
        {"episodes": [plan.model_dump(mode="json") for plan in plans]},
    )
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(
        project_dir / "actor_metadata_metrics.json",
        {"stage_metrics": {"episode_planning": {"unknown_actor_ids": 0}, "writing": {"completed_episode_count": 1}}},
    )

    existing_plan = plans[0]
    _write_json(
        project_dir / "episodes" / "1" / "episode_script.json",
        _episode_script(existing_plan, "Episode 1"),
    )
    _write_json(
        project_dir / "episodes" / "1" / "spoken_script.json",
        _spoken_script(existing_plan, "Episode 1"),
    )

    return project_dir, plans


def test_retry_single_episode_from_writing_retries_target_episode_and_regenerates_manifest(
    monkeypatch,
    tmp_path,
):
    project_dir, plans = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _write_episode(
            self,
            plan: EpisodePlan,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            project: ThematicProject,
            corpus: ThematicCorpus,
            ep_dir: Path,
            project_dir: Path,
            actor_metadata: ActorMetadata | None = None,
        ) -> EpisodeScript:
            calls["written_episode_number"] = plan.episode_number
            calls["write_config"] = project.config
            calls["written_strategy_title"] = strategy_episode.title
            calls["written_architecture_episode"] = architecture.episode_number
            script = _episode_script(plan, strategy_episode.title)
            _write_json(ep_dir / "episode_script.json", script)
            _write_json(ep_dir / "spine_diagnostics.json", {"status": "ok"})
            return script

        async def _rewrite_for_speech(
            self,
            episode_number: int,
            script: EpisodeScript,
            project: ThematicProject,
            ep_dir: Path,
            project_dir: Path,
        ) -> SpokenScript:
            calls["spoken_episode_number"] = episode_number
            spoken = _spoken_script(plans[episode_number - 1], script.title)
            _write_json(ep_dir / "spoken_script.json", spoken)
            return spoken

        async def _render_episode_audio(
            self,
            episode_number: int,
            spoken: SpokenScript,
            config: PipelineConfig,
            project_dir: Path,
            semaphore: asyncio.Semaphore,
            *,
            skip_audio: bool,
        ) -> None:
            calls["render_episode_number"] = episode_number
            calls["audio_skip"] = skip_audio
            _write_json(
                project_dir / "episodes" / str(episode_number) / "render_manifest.json",
                {"episode_number": episode_number, "total_segments": len(spoken.sections)},
            )

        def _write_passage_utilization(self, **kwargs: Any) -> None:
            calls["passage_utilization"] = kwargs

        def _build_writing_actor_metrics(
            self,
            project_dir: Path,
            spoken_scripts: list[tuple[int, SpokenScript]],
        ) -> dict[str, Any]:
            calls["spoken_scripts"] = spoken_scripts
            return {"completed_episode_count": len(spoken_scripts)}

        def _write_actor_metadata_metrics(self, **kwargs: Any) -> None:
            calls["actor_metadata_metrics"] = kwargs

    monkeypatch.setattr(
        retry_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(retry_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(retry_script._retry_single_episode_from_writing("run_1", 2))

    assert calls["bound_project_dir"] == project_dir
    assert calls["written_episode_number"] == 2
    assert calls["written_strategy_title"] == "Episode 2"
    assert calls["written_architecture_episode"] == 2
    assert calls["spoken_episode_number"] == 2
    assert calls["render_episode_number"] == 2
    assert calls["write_config"].skip_grounding is True
    assert calls["write_config"].skip_audio is True
    assert calls["write_config"].skip_spoken_delivery is False
    assert calls["audio_skip"] is True
    assert calls["passage_utilization"]["episode_numbers"] == [1, 2]
    assert [episode_number for episode_number, _ in calls["spoken_scripts"]] == [1, 2]
    assert calls["actor_metadata_metrics"]["metrics"]["episode_planning"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["writing"] == {
        "completed_episode_count": 2
    }
    assert (project_dir / "episodes" / "2" / "episode_script.json").exists()
    assert (project_dir / "episodes" / "2" / "spoken_script.json").exists()
    assert (project_dir / "episodes" / "2" / "render_manifest.json").exists()

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
    assert final_project.episode_count == 2


def test_retry_single_episode_from_writing_restores_original_status_on_failure(
    monkeypatch,
    tmp_path,
):
    project_dir, _ = _build_project_dir(tmp_path)

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            return None

        async def _write_episode(self, *args: Any, **kwargs: Any) -> EpisodeScript:
            raise RuntimeError("writing failed")

    monkeypatch.setattr(
        retry_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(retry_script, "PipelineOrchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="writing failed"):
        asyncio.run(retry_script._retry_single_episode_from_writing("run_1", 2))

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE


def test_retry_single_episode_from_writing_requires_matching_architecture(
    monkeypatch,
    tmp_path,
):
    project_dir, _ = _build_project_dir(tmp_path)
    architectures_payload = json.loads(
        (project_dir / "episode_architectures.json").read_text(encoding="utf-8")
    )
    architectures_payload["episodes"] = architectures_payload["episodes"][:1]
    _write_json(project_dir / "episode_architectures.json", architectures_payload)

    monkeypatch.setattr(
        retry_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(
        retry_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="episode_architectures.json does not contain episode_number 2"):
        asyncio.run(retry_script._retry_single_episode_from_writing("run_1", 2))
