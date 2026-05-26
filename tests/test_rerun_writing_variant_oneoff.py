from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_agent.config import Settings as RuntimeSettings
from _section_progression_helpers import make_section_progression
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeScript,
    EventPrimitive,
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveSubstrate,
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
    / "rerun_writing_variant_oneoff.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "rerun_writing_variant_oneoff",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
variant_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(variant_script)


def _write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _host_moves() -> dict[str, list[dict[str, str]]]:
    return {
        "open": [
            {
                "move_type": "orient",
                "note": "Set the listener's footing before the beat turns.",
            }
        ],
        "pivot": [],
        "close": [],
    }


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
                    "host_moves": _host_moves(),
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
                "section_progression": make_section_progression("setup", label=f"{prefix}1"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}2"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}3"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}4"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}5"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}6"),
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
                "section_progression": make_section_progression("advance", label=f"{prefix}7"),
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
                "section_progression": make_section_progression("answer", label=f"{prefix}8"),
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
                "section_progression": make_section_progression("close", label=f"{prefix}9"),
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
        sections=[
            ArchitectureSection.model_validate(section)
            for section in payload["sections"]
        ],
        architecture_notes=payload["architecture_notes"],
    )


def _episode_script(plan: EpisodePlan, title: str, text: str) -> EpisodeScript:
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
                    "text": text,
                    "citations": [],
                    "source_book_ids": ["book_1"],
                }
            ],
            "total_word_count": len(text.split()),
            "estimated_duration_seconds": 1,
        }
    )


def _build_project_dir(tmp_path: Path, *, skip_grounding: bool) -> tuple[Path, list[EpisodePlan]]:
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
        what_happened="A decisive event changes the situation.",
    )
    primitives = SynthesisPrimitivesArtifact(
        project_id="run_1",
        primitives=[primitive],
    )
    enriched_primitive = EventPrimitive(
        id="primitive_1",
        substrate=PrimitiveSubstrate.EVENTS,
        title="Primitive",
        core_passage_ids=["passage_1"],
        actor_ids=["actor_1"],
        event_type="turning_point",
        what_happened="A decisive event changes the situation.",
        event_result="The balance breaks.",
    )
    synthesis_map = SynthesisMap(
        project_id="run_1",
        primitives=[enriched_primitive],
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
            skip_grounding=skip_grounding,
            skip_audio=True,
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

    baseline_plan = plans[1]
    _write_json(
        project_dir / "episodes" / "2" / "episode_script.json",
        _episode_script(baseline_plan, "Episode 2", "A short baseline section."),
    )
    _write_json(
        project_dir / "episodes" / "2" / "spine_diagnostics.json",
        {"status": "baseline"},
    )

    return project_dir, plans


def test_rerun_writing_variant_oneoff_writes_sidecar_outputs_only(monkeypatch, tmp_path):
    project_dir, plans = _build_project_dir(tmp_path, skip_grounding=True)
    source_before = (
        project_dir / "episodes" / "2" / "episode_script.json"
    ).read_text(encoding="utf-8")
    calls: dict[str, Any] = {}
    base_settings = RuntimeSettings()
    settings = base_settings.model_copy(
        update={
            "pipeline": base_settings.pipeline.model_copy(
                update={"artifact_root": tmp_path}
            )
        }
    )

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            self.writing_agent_no_citations = SimpleNamespace(instructions="base")
            calls["resolved_model"] = settings.llm.resolve_model("episode_writing")

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
            host_policy: dict[str, Any] | None = None,
        ) -> EpisodeScript:
            calls["write_episode_number"] = plan.episode_number
            calls["project_dir"] = project_dir
            calls["ep_dir"] = ep_dir
            calls["instructions"] = self.writing_agent_no_citations.instructions
            calls["host_policy"] = host_policy
            script = _episode_script(
                plans[plan.episode_number - 1],
                strategy_episode.title,
                "We enter the square. I think this is the turn. We are left with pressure.",
            )
            _write_json(ep_dir / "episode_script.json", script)
            _write_json(ep_dir / "spine_diagnostics.json", {"status": "variant"})
            return script

    monkeypatch.setattr(
        variant_script,
        "Settings",
        lambda: settings,
    )
    monkeypatch.setattr(variant_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(
        variant_script._rerun_writing_variant_oneoff(
            "run_1",
            2,
            model_name=variant_script.DEFAULT_WRITING_MODEL_NAME,
        )
    )

    experiment_ep_dir = (
        project_dir
        / "writing_experiments"
        / variant_script.VARIANT_LABEL
        / "episodes"
        / "2"
    )
    assert calls["bound_project_dir"] == project_dir / "writing_experiments" / variant_script.VARIANT_LABEL
    assert calls["write_episode_number"] == 2
    assert calls["project_dir"] == project_dir / "writing_experiments" / variant_script.VARIANT_LABEL
    assert calls["ep_dir"] == experiment_ep_dir
    assert "Treat `address_mode = we` and `address_mode = i` as stance signals" in calls["instructions"]
    assert "Avoid companion-tour phrasing" in calls["instructions"]
    assert calls["host_policy"]["pronoun_policy"]["allow_first_person_singular"] is True
    assert calls["resolved_model"] == variant_script.DEFAULT_WRITING_MODEL_NAME

    assert (experiment_ep_dir / "baseline_episode_script.json").exists()
    assert (experiment_ep_dir / "baseline_spine_diagnostics.json").exists()
    assert (experiment_ep_dir / "episode_script.json").exists()
    assert (experiment_ep_dir / "spine_diagnostics.json").exists()
    assert (experiment_ep_dir / "comparison_summary.json").exists()

    comparison = json.loads(
        (experiment_ep_dir / "comparison_summary.json").read_text(encoding="utf-8")
    )
    assert comparison["variant_label"] == variant_script.VARIANT_LABEL
    assert comparison["writing_model"]["schema_name"] == "episode_writing"
    assert comparison["writing_model"]["model_name"] == variant_script.DEFAULT_WRITING_MODEL_NAME
    assert comparison["writing_model"]["provider"] == "anthropic"
    assert comparison["baseline"]["pronouns"]["we_us_our"] == 0
    assert comparison["variant"]["pronouns"]["we_us_our"] == 2
    assert comparison["variant"]["pronouns"]["i_me_my"] == 1
    assert comparison["delta"]["pronouns"]["we_us_our"] == 2
    assert comparison["baseline"]["surface_markers"]["first_person_scene_camera"] == 0
    assert comparison["variant"]["surface_markers"]["first_person_scene_camera"] == 1
    assert comparison["delta"]["surface_markers"]["first_person_scene_camera"] == 1

    source_after = (
        project_dir / "episodes" / "2" / "episode_script.json"
    ).read_text(encoding="utf-8")
    assert source_after == source_before


def test_rerun_writing_variant_oneoff_requires_baseline_script(monkeypatch, tmp_path):
    project_dir, _ = _build_project_dir(tmp_path, skip_grounding=True)
    (project_dir / "episodes" / "2" / "episode_script.json").unlink()
    base_settings = RuntimeSettings()
    settings = base_settings.model_copy(
        update={
            "pipeline": base_settings.pipeline.model_copy(
                update={"artifact_root": tmp_path}
            )
        }
    )

    monkeypatch.setattr(
        variant_script,
        "Settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        variant_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="Baseline episode_script.json is required"):
        asyncio.run(variant_script._rerun_writing_variant_oneoff("run_1", 2))


def test_parse_args_defaults_model_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rerun_writing_variant_oneoff.py",
            "--project-id",
            "run_1",
            "--episode-number",
            "2",
        ],
    )

    args = variant_script._parse_args()

    assert args.project_id == "run_1"
    assert args.episode_number == 2
    assert args.model_name == variant_script.DEFAULT_WRITING_MODEL_NAME
