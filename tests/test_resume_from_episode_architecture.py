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
    BookRecord,
    EpisodeArchitecture,
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
                    "listener_problem": "What changed?",
                    "episode_answer": "A working claim.",
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
                    "section_progression": make_section_progression("setup", label="section_01"),
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
                    "section_progression": make_section_progression("advance", label="section_02"),
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
                    "section_progression": make_section_progression("advance", label="section_03"),
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
                    "section_progression": make_section_progression("advance", label="section_04"),
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
                    "section_progression": make_section_progression("answer", label="section_05"),
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
                    "section_progression": make_section_progression("close", label="section_06"),
                    "priority_core_passage_ids": [],
                    "key_terms": [],
                    "authorial_passages": [],
                    "term_explanations": [],
                    "actor_explanations": [],
                },
            ],
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


def _build_project_dir(tmp_path: Path, *, include_architecture: bool = True) -> Path:
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
                    "scene_jobs": ["opening"],
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

    _write_json(project_dir / "thematic_axes.json", {"axes": [axis.model_dump(mode="json")]})
    _write_json(project_dir / "thematic_project.json", project)
    _write_json(project_dir / "thematic_corpus.json", corpus)
    _write_json(project_dir / "actor_metadata.json", actor_metadata)
    _write_json(project_dir / "scene_discovery.json", scene_discovery)
    _write_json(project_dir / "retained_primitives.json", retained_primitives)
    _write_json(project_dir / "narrative_strategy.json", strategy)
    if include_architecture:
        _write_json(
            project_dir / "episode_architectures.json",
            {"episodes": [architecture.model_dump(mode="json")]},
        )
    return project_dir


def test_resume_from_episode_architecture_reruns_architecture_then_planning_then_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path, include_architecture=False)
    calls: dict[str, Any] = {"order": []}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

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
            calls["architecture_status"] = kwargs["project"].status
            calls["architecture_scene_discovery"] = kwargs["scene_discovery"]
            calls["architecture_actor_metadata"] = kwargs["actor_metadata"]
            architecture = _architecture()
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            calls["order"].append("plan_series")
            calls["planning_status"] = kwargs["project"].status
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

        async def _produce_episode(self, *args: Any, **kwargs: Any) -> tuple[int, SpokenScript]:
            calls["order"].append("produce_episode")
            calls["host_policy"] = kwargs["host_policy"]
            return (
                1,
                SpokenScript(
                    episode_number=1,
                    title="Episode 1",
                    framing={
                        "opening_image": "An image",
                        "threat_or_unresolved_action": "A threat remains.",
                        "opening_question": "What now?",
                        "handoff_scene_card_id": "scene_01",
                    },
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
            calls["order"].append("render_audio")
            calls["audio_rendered"] = True

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(artifact_root=tmp_path),
        ),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(resume_script._resume_from_episode_architecture("run_1"))

    assert project_dir.joinpath("synthesis_map.json").exists() is False
    assert calls["order"] == [
        "episode_architecture",
        "plan_series",
        "produce_episode",
        "render_audio",
    ]
    assert calls["bound_project_dir"] == project_dir
    assert calls["architecture_status"] == ProjectStatus.PLANNING
    assert (
        calls["architecture_scene_discovery"].candidates[0].candidate_id
        == "candidate_01"
    )
    assert calls["architecture_actor_metadata"].actors[0].actor_id == "actor_1"
    assert calls["planning_status"] == ProjectStatus.PLANNING
    assert calls["host_policy"]["allowed_moves"]
    assert calls["passage_utilization_written"] is True
    assert calls["actor_metrics_written"] is True
    assert calls["audio_rendered"] is True


def test_resume_from_episode_architecture_requires_scene_discovery(
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

    with pytest.raises(RuntimeError, match="scene_discovery.json"):
        asyncio.run(resume_script._resume_from_episode_architecture("run_1"))


def test_resume_from_episode_architecture_does_not_require_persisted_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _build_project_dir(tmp_path, include_architecture=False)
    calls: dict[str, Any] = {}

    class FakeOrchestrator:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def _bind_run_logger(self, bound_project_dir: Path) -> None:
            calls["bound_project_dir"] = bound_project_dir

        async def _plan_series_with_narrative_state(
            self, **kwargs: Any
        ) -> tuple[
            list[EpisodeArchitecture],
            list[Any],
            dict[int, NarrativeState],
            dict[int, NarrativeState],
            dict[str, Any],
        ]:
            calls["architecture_called"] = True
            architecture = _architecture()
            _write_json(
                project_dir / "episode_architectures.json",
                {"episodes": [architecture.model_dump(mode="json")]},
            )
            state_pre, state_post = _build_narrative_states()
            _write_json(project_dir / "series_plan.json", {"episodes": [{"episode_number": 1}]})
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

        async def _produce_episode(self, *args: Any, **kwargs: Any) -> tuple[int, SpokenScript]:
            return (
                1,
                SpokenScript(
                    episode_number=1,
                    title="Episode 1",
                    framing={
                        "opening_image": "An image",
                        "threat_or_unresolved_action": "A threat remains.",
                        "opening_question": "What now?",
                        "handoff_scene_card_id": "scene_01",
                    },
                    sections=[SpokenSection(section_id="section_01", text="Text.")],
                    tts_provider="openai",
                ),
            )

        def _write_passage_utilization(self, **_: Any) -> None:
            return None

        def _build_writing_actor_metrics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"episodes": 1}

        def _write_actor_metadata_metrics(self, **_: Any) -> None:
            return None

        async def _render_episode_audio(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        resume_script,
        "Settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(artifact_root=tmp_path),
        ),
    )
    monkeypatch.setattr(resume_script, "PipelineOrchestrator", FakeOrchestrator)

    asyncio.run(resume_script._resume_from_episode_architecture("run_1"))

    assert calls["bound_project_dir"] == project_dir
    assert calls["architecture_called"] is True
    assert project_dir.joinpath("episode_architectures.json").exists() is True


def test_validate_architecture_transition_accepts_staged_promised_beats():
    """Regression: the staged-promised-beat check must not raise NameError.

    Reproduces the v61 failure where removing the answer/residue block dropped the
    `section_order`/`valid_section_ids` definition that this check relies on.
    """
    from podcast_agent.pipeline.orchestrator import _validate_architecture_transition

    strategy_episode = _strategy().episodes[0]
    architecture = _architecture()  # stages beat_1 into the real section_01

    result = _validate_architecture_transition(
        strategy_episode=strategy_episode,
        architecture=architecture,
    )
    assert result is architecture


def test_validate_architecture_transition_rejects_unknown_staged_section():
    from podcast_agent.pipeline.orchestrator import _validate_architecture_transition

    strategy_episode = _strategy().episodes[0]
    architecture = _architecture()
    # Point the staged decision at a section that does not exist.
    architecture.promised_beat_decisions[0].section_id = "section_99"

    with pytest.raises(RuntimeError, match="staged promised beats into unknown sections"):
        _validate_architecture_transition(
            strategy_episode=strategy_episode,
            architecture=architecture,
        )
