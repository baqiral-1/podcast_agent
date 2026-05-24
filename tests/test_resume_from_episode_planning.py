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
    BookRecord,
    EpisodeArchitecture,
    EpisodePlan,
    EventPrimitive,
    NarrativeState,
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveSubstrate,
    ProjectStatus,
    SceneDiscoveryArtifact,
    SpokenScript,
    SpokenSection,
    StrategyEpisode,
    SynthesisMap,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "resume_from_episode_planning.py"
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


def _primitive(primitive_id: str) -> EventPrimitive:
    return EventPrimitive(
        id=primitive_id,
        substrate=PrimitiveSubstrate.EVENTS,
        title=primitive_id,
        core_passage_ids=["passage_1"],
        actor_ids=["actor_1"],
        event_type="turning_point",
        what_happened=f"{primitive_id} changes the field.",
        event_result=f"{primitive_id} lands.",
    )


def _strategy() -> NarrativeStrategy:
    return NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Arc",
        recommended_episode_count=1,
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc summary",
                episode_spine={
                    "listener_question": "What changed?",
                    "argument": "A working claim.",
                    "core_primitive_ids": ["primitive_1", "core_2"],
                    "support_primitive_roles": {
                        "support_1": "mechanism",
                        "support_2": "mechanism",
                    },
                    "recall_primitive_ids": [],
                },
                promised_beats=[
                    {
                        "beat_id": "beat_1",
                        "label": "Open from the decree",
                        "kind": "scene",
                        "intended_job": "opening",
                        "source_candidate_ids": ["candidate_01"],
                        "source_primitive_ids": ["primitive_1"],
                        "why_load_bearing": "It is the concrete opening handle.",
                    }
                ],
            )
        ],
    )


def _architecture() -> EpisodeArchitecture:
    return EpisodeArchitecture.model_validate(
        {
            "episode_number": 1,
            "major_turn_section_id": "section_03",
            "answer_section_id": "section_04",
            "residue_section_id": "section_05",
            "promised_beat_decisions": [
                {"beat_id": "beat_1", "decision": "stage", "section_id": "section_01"}
            ],
            "sections": [
                {
                    "section_id": "section_01",
                    "purpose": "opening",
                    "section_anchor": "A decree arrives.",
                    "must_stage_beats": ["The decree lands.", "Everyone recalculates."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "closure_mode": "residue",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
                {
                    "section_id": "section_02",
                    "purpose": "setup",
                    "section_anchor": "The coalition forms.",
                    "must_stage_beats": ["Actors align.", "Pressure sharpens."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["core_2"],
                    "closure_mode": "residue",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
                {
                    "section_id": "section_03",
                    "purpose": "turn",
                    "section_anchor": "The break becomes visible.",
                    "must_stage_beats": ["A line is crossed.", "The balance shifts."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["support_1"],
                    "closure_mode": "turn",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
                {
                    "section_id": "section_04",
                    "purpose": "setup",
                    "section_anchor": "The answer starts to settle.",
                    "must_stage_beats": ["The mechanism is named.", "Its result becomes clear."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["support_2"],
                    "closure_mode": "residue",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
                {
                    "section_id": "section_05",
                    "purpose": "setup",
                    "section_anchor": "The cost spreads outward.",
                    "must_stage_beats": ["The answer carries cost.", "The burden remains."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "closure_mode": "residue",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
                {
                    "section_id": "section_06",
                    "purpose": "closing",
                    "section_anchor": "One image remains.",
                    "must_stage_beats": ["The final image lands.", "The pressure stays alive."],
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "closure_mode": "final_answer",
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
            ],
        }
    )


def _plan() -> EpisodePlan:
    return EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "framing": {
                "opening_image": "An image",
                "threat_or_unresolved_action": "A threat remains.",
                "opening_question": "What now?",
                "handoff_scene_card_id": "scene_01",
            },
            "scene_cards": [
                {
                    "scene_id": "scene_01",
                    "section_id": "section_01",
                    "title": "Opening",
                    "scene_role": "context_setup",
                    "scene_job": "opening",
                    "beat_change": "The decree lands.",
                    "must_land_facts": {"required": ["The decree lands."]},
                    "entry_image": "A document on the desk.",
                    "observable_detail": "Hands stop moving.",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "orient", "target": "stakes"}],
                        "pivot": [],
                        "close": [],
                    },
                },
                {
                    "scene_id": "scene_02",
                    "section_id": "section_02",
                    "title": "Build",
                    "scene_role": "action",
                    "scene_job": "build",
                    "beat_change": "Allies reposition.",
                    "must_land_facts": {"required": ["Allies reposition."]},
                    "entry_image": "People lean over a map.",
                    "observable_detail": "Voices drop.",
                    "primitive_ids": ["core_2"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "clarify", "target": "alignment"}],
                        "pivot": [],
                        "close": [],
                    },
                },
                {
                    "scene_id": "scene_03",
                    "section_id": "section_03",
                    "title": "Turn",
                    "scene_role": "shock",
                    "scene_job": "turn",
                    "beat_change": "The break becomes public.",
                    "must_land_facts": {"required": ["The break becomes public."]},
                    "entry_image": "A room falls silent.",
                    "observable_detail": "Faces change.",
                    "primitive_ids": ["support_1"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "contrast", "target": "before after"}],
                        "pivot": [],
                        "close": [],
                    },
                },
                {
                    "scene_id": "scene_04",
                    "section_id": "section_04",
                    "title": "Answer",
                    "scene_role": "implication",
                    "scene_job": "answer",
                    "beat_change": "The mechanism is now clear.",
                    "must_land_facts": {"required": ["The mechanism is now clear."]},
                    "entry_image": "The pattern snaps into place.",
                    "observable_detail": "The logic is visible.",
                    "primitive_ids": ["support_2"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "evaluate", "target": "meaning"}],
                        "pivot": [],
                        "close": [],
                    },
                },
                {
                    "scene_id": "scene_05",
                    "section_id": "section_05",
                    "title": "Residue",
                    "scene_role": "fallout",
                    "scene_job": "residue",
                    "beat_change": "The cost remains.",
                    "must_land_facts": {"required": ["The cost remains."]},
                    "entry_image": "The room empties.",
                    "observable_detail": "Paper stays on the desk.",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "callback", "target": "opening image"}],
                        "pivot": [],
                        "close": [],
                    },
                },
                {
                    "scene_id": "scene_06",
                    "section_id": "section_06",
                    "title": "Close",
                    "scene_role": "implication",
                    "scene_job": "close",
                    "beat_change": "One image remains.",
                    "must_land_facts": {"required": ["One image remains."]},
                    "entry_image": "The image holds.",
                    "observable_detail": "Nothing resolves cleanly.",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["passage_1"],
                    "estimated_duration_seconds": 60,
                    "host_moves": {
                        "open": [{"move_type": "light_aside", "target": "aftertaste"}],
                        "pivot": [],
                        "close": [],
                    },
                },
            ],
            "answer_scene_card_id": "scene_04",
            "residue_scene_card_id": "scene_05",
            "dropped_support_primitive_reasons": {},
            "target_word_count": 1200,
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
        requested_episode_count=1,
        config=PipelineConfig(
            skip_grounding=False,
            skip_audio=False,
            skip_spoken_delivery=True,
        ),
        status=ProjectStatus.ANALYZING,
    )
    corpus = ThematicCorpus(project_id="run_1", axes=[axis])
    scene_discovery = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A decree lands and the field changes.",
                    "candidate_roles": ["opening"],
                    "anchor_image": "A document hits the desk.",
                    "why_sceneable": "The shift is concrete and audible.",
                    "actor_ids": ["actor_1"],
                }
            ]
        }
    )
    retained_primitives = SynthesisMap(
        project_id="run_1",
        primitives=[
            _primitive("primitive_1"),
            _primitive("core_2"),
            _primitive("support_1"),
            _primitive("support_2"),
        ],
    )
    strategy = _strategy()
    architecture = _architecture()
    plan = _plan()
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

    _write_json(project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]})
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(project_dir / "scene_discovery.json", scene_discovery)
    _write_json(project_dir / "retained_primitives.json", retained_primitives)
    _write_json(project_dir / "narrative_strategy.json", strategy)
    _write_json(
        project_dir / "episode_architectures.json",
        {"episodes": [architecture.model_dump(mode="json")]},
    )
    _write_json(project_dir / "series_plan.json", {"episodes": [plan.model_dump(mode="json")]})
    _write_json(project_dir / "narrative_state_timeline.json", {"episodes": []})
    _write_json(project_dir / "narrative_state_latest.json", state_post)
    episode_dir = project_dir / "episodes" / "1"
    episode_dir.mkdir(parents=True)
    _write_json(episode_dir / "narrative_state_pre.json", state_pre)
    _write_json(episode_dir / "narrative_state_post.json", state_post)
    return project_dir


def test_resume_from_episode_planning_uses_persisted_plan_only(
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

        async def _build_episode_architectures(self, **_: Any) -> None:
            raise AssertionError("episode architecture should not rerun")

        async def _plan_series(self, **_: Any) -> None:
            raise AssertionError("episode planning should not rerun")

        async def _produce_episode(self, plan: EpisodePlan, *args: Any, **kwargs: Any) -> tuple[int, SpokenScript]:
            calls["planned_episode_number"] = plan.episode_number
            calls["produce_config"] = kwargs["semaphore"] is not None
            return (
                1,
                SpokenScript(
                    episode_number=1,
                    title="Episode 1",
                    framing=plan.framing,
                    sections=[SpokenSection(section_id="section_01", text="Text.")],
                    tts_provider="openai",
                ),
            )

        def _write_passage_utilization(self, **_: Any) -> None:
            calls["passage_utilization_written"] = True

        def _build_writing_actor_metrics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"episodes": 1}

        def _write_actor_metadata_metrics(self, **_: Any) -> None:
            calls["actor_metrics_written"] = True

        async def _render_episode_audio(self, *args: Any, **kwargs: Any) -> None:
            calls["audio_rendered"] = True

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(artifact_root=tmp_path),
        ),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(resume_script._resume_from_episode_planning("run_1"))

    assert project_dir.joinpath("tagged_primitives.json").exists() is False
    assert calls["bound_project_dir"] == project_dir
    assert calls["planned_episode_number"] == 1
    assert calls["passage_utilization_written"] is True
    assert calls["actor_metrics_written"] is True
    assert calls["audio_rendered"] is True


def test_resume_from_episode_planning_requires_scene_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path)
    project_dir.joinpath("scene_discovery.json").unlink()

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(artifact_root=tmp_path),
        ),
    )
    monkeypatch.setattr(
        resume_script,
        "PipelineOrchestrator",
        lambda settings: SimpleNamespace(),
    )

    with pytest.raises((RuntimeError, FileNotFoundError), match="scene_discovery.json"):
        asyncio.run(resume_script._resume_from_episode_planning("run_1"))
