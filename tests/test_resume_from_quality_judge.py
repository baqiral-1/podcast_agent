from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
    ExcerptArtifact,
    ExcerptRecord,
    NarrativeState,
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveSubstrate,
    ProjectStatus,
    SpokenScript,
    StrategyEpisode,
    SynthesisMap,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "resume_from_quality_judge.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_from_quality_judge",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
resume_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(resume_script)


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
                    "beat_change": "The stakes become legible.",
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
                    "segments": [
                        {
                            "segment_id": f"section_{plan.episode_number}_spoken_seg1",
                            "text": "A spoken section.",
                            "speaker_role": "primary",
                            "tonal_register": "neutral",
                        }
                    ],
                    "tonal_register": "neutral",
                }
            ],
            "tts_provider": "openai",
        }
    )


def _retained_excerpts() -> ExcerptArtifact:
    return ExcerptArtifact(
        project_id="run_1",
        excerpts=[
            ExcerptRecord(
                id="x1",
                title="Proclamation",
                passage_ids=["passage_1"],
                summary="A short proclamation.",
            )
        ],
    )


def _build_project_dir(
    tmp_path: Path, *, include_retained_excerpts: bool = True
) -> tuple[Path, list[EpisodePlan]]:
    project_dir = tmp_path / "run_1"
    project_dir.mkdir()

    axis = ThematicAxis(
        axis_id="axis_1",
        name="Axis",
        description="Axis description",
        theme_importance_score=1.0,
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
    retained_primitives = SynthesisMap(
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
                    "listener_problem": "Question 1?",
                    "episode_answer": "A working claim.",
                    "core_primitive_ids": ["primitive_1", "core_2"],
                    "support_primitive_roles": {
                        "support_1": "mechanism",
                        "support_2": "mechanism",
                    },
                    "recall_primitive_ids": [],
                },
            ),
            StrategyEpisode(
                episode_number=2,
                title="Episode 2",
                arc_summary="Arc summary 2",
                episode_spine={
                    "listener_problem": "Question 2?",
                    "episode_answer": "Another working claim.",
                    "core_primitive_ids": ["primitive_1", "core_2"],
                    "support_primitive_roles": {
                        "support_1": "mechanism",
                        "support_2": "mechanism",
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
    _write_json(project_dir / "retained_primitives.json", retained_primitives)
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
        {"stage_metrics": {"episode_planning": {"unknown_actor_ids": 0}}},
    )
    if include_retained_excerpts:
        _write_json(project_dir / "retained_excerpts.json", _retained_excerpts())

    for plan in plans:
        ep_dir = project_dir / "episodes" / str(plan.episode_number)
        _write_json(
            ep_dir / "episode_script.json", _episode_script(plan, f"Episode {plan.episode_number}")
        )
        _write_json(ep_dir / "narrative_state_pre.json", NarrativeState(project_id="run_1"))
        _write_json(ep_dir / "narrative_state_post.json", NarrativeState(project_id="run_1"))
        _write_json(ep_dir / "spine_diagnostics.json", {"episode": plan.episode_number})
        _write_json(ep_dir / "host_moves_script_diagnostics.json", {"stale": True})

    return project_dir, plans


def test_resume_from_quality_judge_uses_persisted_scripts_and_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir, plans = _build_project_dir(tmp_path)
    calls: dict[str, Any] = {"episodes": []}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _continue_episode_from_script(
            self,
            *,
            plan: EpisodePlan,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            script: EpisodeScript,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            ep_dir: Path,
            host_policy: dict[str, Any],
            primitive_lookup: dict[str, EventPrimitive],
            spoken_semaphore: asyncio.Semaphore,
            series_explanation_registry: list[Any] | None = None,
            narrative_state_pre: NarrativeState | None = None,
            narrative_state_post: NarrativeState | None = None,
            excerpt_by_id: dict[str, ExcerptRecord] | None = None,
            spine_diagnostics: dict[str, Any] | None = None,
            host_moves_diagnostics: dict[str, Any] | None = None,
        ) -> tuple[int, SpokenScript]:
            calls["episodes"].append(
                {
                    "episode_number": plan.episode_number,
                    "script_episode_number": script.episode_number,
                    "strategy_title": strategy_episode.title,
                    "architecture_episode_number": architecture.episode_number,
                    "config": project.config,
                    "corpus_project_id": corpus.project_id,
                    "actor_id": actor_metadata.actors[0].actor_id,
                    "host_policy": host_policy,
                    "primitive_lookup": primitive_lookup,
                    "narrative_state_pre": narrative_state_pre,
                    "narrative_state_post": narrative_state_post,
                    "excerpt_ids": sorted((excerpt_by_id or {}).keys()),
                    "spine_diagnostics": spine_diagnostics,
                    "host_moves_diagnostics": host_moves_diagnostics,
                    "series_explanation_registry": series_explanation_registry or [],
                    "spoken_semaphore": spoken_semaphore,
                }
            )
            _write_json(
                ep_dir / "quality_judgment.json",
                {"episode_number": plan.episode_number, "overall_score": 80},
            )
            _write_json(ep_dir / "style_audit_result.json", {"status": "ok"})
            _write_json(ep_dir / "style_audited_script.json", script)
            spoken = _spoken_script(plan, strategy_episode.title)
            _write_json(ep_dir / "spoken_script.json", spoken)
            return plan.episode_number, spoken

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
            calls.setdefault("render_calls", []).append(
                {"episode_number": episode_number, "skip_audio": skip_audio}
            )
            _write_json(
                project_dir / "episodes" / str(episode_number) / "render_manifest.json",
                {"episode_number": episode_number, "segment_count": len(spoken.sections)},
            )

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        resume_script,
        "resume_from_quality_judge_stage",
        resume_script.resume_from_quality_judge_stage,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_architectures_against_strategy",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_plans_against_strategy_and_architecture",
        lambda **kwargs: None,
    )

    asyncio.run(resume_script._resume_from_quality_judge("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert [item["episode_number"] for item in calls["episodes"]] == [1, 2]
    assert calls["episodes"][0]["script_episode_number"] == 1
    assert calls["episodes"][1]["strategy_title"] == "Episode 2"
    assert calls["episodes"][0]["architecture_episode_number"] == 1
    assert calls["episodes"][0]["config"].skip_grounding is True
    assert calls["episodes"][0]["config"].skip_audio is True
    assert calls["episodes"][0]["config"].skip_spoken_delivery is False
    assert calls["episodes"][0]["corpus_project_id"] == "run_1"
    assert calls["episodes"][0]["actor_id"] == "actor_1"
    assert "authorial_policy" in calls["episodes"][0]["host_policy"]
    assert calls["episodes"][0]["primitive_lookup"]["primitive_1"].id == "primitive_1"
    assert calls["episodes"][0]["narrative_state_pre"].project_id == "run_1"
    assert calls["episodes"][0]["narrative_state_post"].project_id == "run_1"
    assert calls["episodes"][0]["excerpt_ids"] == ["x1"]
    assert calls["episodes"][0]["spine_diagnostics"] == {"episode": 1}
    assert calls["episodes"][0]["host_moves_diagnostics"] == {}
    assert calls["render_calls"] == [
        {"episode_number": 1, "skip_audio": True},
        {"episode_number": 2, "skip_audio": True},
    ]
    assert calls["passage_utilization"]["episode_numbers"] == [1, 2]
    assert [episode_number for episode_number, _ in calls["spoken_scripts"]] == [1, 2]
    assert calls["actor_metadata_metrics"]["metrics"]["episode_planning"] == {
        "unknown_actor_ids": 0
    }
    assert calls["actor_metadata_metrics"]["metrics"]["writing"] == {"completed_episode_count": 2}
    for plan in plans:
        ep_dir = project_dir / "episodes" / str(plan.episode_number)
        assert (ep_dir / "quality_judgment.json").exists()
        assert (ep_dir / "style_audit_result.json").exists()
        assert (ep_dir / "style_audited_script.json").exists()
        assert (ep_dir / "spoken_script.json").exists()
        assert (ep_dir / "render_manifest.json").exists()

    final_project = ThematicProject.model_validate(
        json.loads((project_dir / "thematic_project.json").read_text(encoding="utf-8"))
    )
    assert final_project.status == ProjectStatus.COMPLETE
    assert final_project.episode_count == 2


def test_resume_from_quality_judge_requires_episode_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir, _ = _build_project_dir(tmp_path)
    (project_dir / "episodes" / "2" / "episode_script.json").unlink()

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(
        resume_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_architectures_against_strategy",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_plans_against_strategy_and_architecture",
        lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="episode_script.json"):
        asyncio.run(resume_script._resume_from_quality_judge("run_1"))


def test_resume_from_quality_judge_requires_narrative_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir, _ = _build_project_dir(tmp_path)
    (project_dir / "episodes" / "2" / "narrative_state_pre.json").unlink()

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(
        resume_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_architectures_against_strategy",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_plans_against_strategy_and_architecture",
        lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="narrative_state_pre.json"):
        asyncio.run(resume_script._resume_from_quality_judge("run_1"))


def test_resume_from_quality_judge_validates_persisted_artifact_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _build_project_dir(tmp_path)

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(
        resume_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_architectures_against_strategy",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("episode numbers do not match persisted strategy")
        ),
    )

    with pytest.raises(RuntimeError, match="episode numbers do not match"):
        asyncio.run(resume_script._resume_from_quality_judge("run_1"))


def test_resume_from_quality_judge_strips_removed_pipeline_config_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persisted thematic_project.json from runs that predate the redraft-loop
    removal still carries `style_audit_redraft_cap` and
    `style_audit_in_place_floor` under `config`. PipelineConfig is strict and
    would reject those keys; the resume loader must strip them so the legacy
    run can resume cleanly.
    """
    project_dir, plans = _build_project_dir(tmp_path)

    project_path = project_dir / "thematic_project.json"
    project_payload = json.loads(project_path.read_text(encoding="utf-8"))
    project_payload["config"]["style_audit_redraft_cap"] = 1
    project_payload["config"]["style_audit_in_place_floor"] = 70
    project_path.write_text(json.dumps(project_payload), encoding="utf-8")

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            pass

        async def _continue_episode_from_script(
            self,
            *,
            plan: EpisodePlan,
            strategy_episode: StrategyEpisode,
            architecture: EpisodeArchitecture,
            script: EpisodeScript,
            project: ThematicProject,
            corpus: ThematicCorpus,
            actor_metadata: ActorMetadata,
            project_dir: Path,
            ep_dir: Path,
            host_policy: dict[str, Any],
            primitive_lookup: dict[str, EventPrimitive],
            spoken_semaphore: asyncio.Semaphore,
            series_explanation_registry: list[Any] | None = None,
            narrative_state_pre: NarrativeState | None = None,
            narrative_state_post: NarrativeState | None = None,
            excerpt_by_id: dict[str, ExcerptRecord] | None = None,
            spine_diagnostics: dict[str, Any] | None = None,
            host_moves_diagnostics: dict[str, Any] | None = None,
        ) -> tuple[int, SpokenScript]:
            spoken = _spoken_script(plan, strategy_episode.title)
            _write_json(ep_dir / "spoken_script.json", spoken)
            return plan.episode_number, spoken

        def _write_passage_utilization(self, **kwargs: Any) -> None:
            pass

        def _build_writing_actor_metrics(
            self,
            project_dir: Path,
            spoken_scripts: list[tuple[int, SpokenScript]],
        ) -> dict[str, Any]:
            return {"completed_episode_count": len(spoken_scripts)}

        def _write_actor_metadata_metrics(self, **kwargs: Any) -> None:
            pass

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
            _write_json(
                project_dir / "episodes" / str(episode_number) / "render_manifest.json",
                {"episode_number": episode_number},
            )

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(artifact_root=tmp_path)),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_architectures_against_strategy",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.resume._verify_plans_against_strategy_and_architecture",
        lambda **kwargs: None,
    )

    asyncio.run(resume_script._resume_from_quality_judge("run_1"))

    # Re-saved config should no longer contain the removed keys.
    saved = json.loads(project_path.read_text(encoding="utf-8"))
    assert "style_audit_redraft_cap" not in saved["config"]
    assert "style_audit_in_place_floor" not in saved["config"]
    assert saved["status"] == ProjectStatus.COMPLETE.value
