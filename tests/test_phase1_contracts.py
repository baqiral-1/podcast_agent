from __future__ import annotations

from unittest.mock import Mock

import pytest

from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.scene_discovery import SceneDiscoveryAgent
from podcast_agent.langchain.runnables import ComplianceViolationError, RetryableGenerationError
from podcast_agent.llm.base import LLMClient
from podcast_agent.pipeline.orchestrator import (
    _build_host_move_plan_diagnostics,
    _build_spine_plan_diagnostics,
    _validate_plan_transition,
)
from podcast_agent.schemas.models import (
    ArchitectureSection,
    EpisodeArchitecture,
    EpisodePlanDraft,
    EpisodeSpine,
    FramingBlock,
    HostMoveCue,
    PipelineConfig,
    PodcastMode,
    PromisedBeatDecisionRecord,
    SceneCardDraft,
    SceneDiscoveryArtifact,
    SeriesNarratorProfile,
    StrategyEpisode,
    SupportPrimitiveRole,
    resolve_pipeline_config_for_mode,
    scene_discovery_candidate_range_for_mode,
    scene_job_budget_for_mode,
)


def _mock_llm() -> LLMClient:
    return Mock(spec=LLMClient)


def _episode_spine() -> EpisodeSpine:
    return EpisodeSpine(
        listener_problem="Why does the crisis take this shape?",
        episode_answer="One visible turn forces the answer.",
        pressure_line="Pressure narrows the available options.",
        core_primitive_ids=["core_1", "core_2"],
        support_primitive_roles={
            "support_1": SupportPrimitiveRole.MECHANISM,
            "support_2": SupportPrimitiveRole.TEXTURE,
        },
    )


def _strategy_episode() -> StrategyEpisode:
    return StrategyEpisode.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "arc_summary": "One concrete line of pressure narrows into an answer.",
            "episode_spine": _episode_spine().model_dump(mode="json"),
            "promised_beats": [
                {
                    "beat_id": "beat_open",
                    "label": "Open on the visible pressure",
                    "intended_job": "opening",
                    "source_candidate_ids": ["candidate_1"],
                    "source_primitive_ids": ["core_1"],
                    "why_load_bearing": "The opening must stage the pressure concretely.",
                },
                {
                    "beat_id": "beat_answer",
                    "label": "Resolve the listener problem",
                    "intended_job": "answer",
                    "source_candidate_ids": ["candidate_2"],
                    "source_primitive_ids": ["core_2"],
                    "why_load_bearing": "The episode needs one explicit answer-bearing beat.",
                },
                {
                    "beat_id": "beat_residue",
                    "label": "Leave a remainder",
                    "intended_job": "residue",
                    "source_candidate_ids": ["candidate_3"],
                    "source_primitive_ids": ["support_1"],
                    "why_load_bearing": "The episode should leave one burden live after the answer.",
                },
            ],
            "negative_scope": {
                "boundary": "Stay inside the oil crisis.",
                "excluded_topics": ["later regional fallout"],
                "tempting_but_out": ["neighboring diplomatic subplot"],
                "omission_logic": "Do not import adjacent but non-binding material.",
            },
        }
    )


def _section(section_id: str, *, purpose: str = "setup") -> ArchitectureSection:
    return ArchitectureSection(
        section_id=section_id,
        purpose=purpose,
        approx_runtime_minutes=8.0 if purpose != "closing" else 2.0,
        primitive_ids=[f"{section_id}_primitive"],
        section_anchor=f"{section_id} anchor",
        must_stage_beats=[
            f"{section_id} first visible beat.",
            f"{section_id} second visible beat.",
        ],
    )


def _architecture() -> EpisodeArchitecture:
    return EpisodeArchitecture(
        episode_number=1,
        major_turn_section_id="section_04",
        answer_section_id="section_05",
        residue_section_id="section_06",
        promised_beat_decisions=[
            PromisedBeatDecisionRecord(
                beat_id="beat_open", decision="stage", section_id="section_01"
            ),
            PromisedBeatDecisionRecord(
                beat_id="beat_answer", decision="stage", section_id="section_05"
            ),
            PromisedBeatDecisionRecord(
                beat_id="beat_residue", decision="stage", section_id="section_06"
            ),
        ],
        sections=[
            _section("section_01", purpose="opening"),
            _section("section_02"),
            _section("section_03"),
            _section("section_04", purpose="turn"),
            _section("section_05"),
            _section("section_06", purpose="closing"),
        ],
    )


def _scene(
    scene_id: str,
    *,
    section_id: str,
    scene_job: str,
    scene_role: str = "action",
) -> SceneCardDraft:
    role_by_job = {
        "opening": "context_setup",
        "build": scene_role,
        "turn": "shock",
        "answer": "implication",
        "residue": "fallout",
        "close": "implication",
    }
    phase_by_job = {
        "opening": "open",
        "build": "open",
        "turn": "pivot",
        "answer": "pivot",
        "residue": "close",
        "close": "close",
    }
    phase = phase_by_job[scene_job]
    host_moves = {"open": [], "pivot": [], "close": []}
    host_moves[phase] = [{"move_type": "orient", "target": "keep pressure visible"}]
    return SceneCardDraft.model_validate(
        {
            "scene_id": scene_id,
            "section_id": section_id,
            "title": scene_id,
            "scene_role": role_by_job[scene_job],
            "scene_job": scene_job,
            "beat_change": f"{scene_id} changes what the listener can now see.",
            "must_land_facts": [f"{scene_id} lands one concrete fact."],
            "entry_image": f"{scene_id} image",
            "observable_detail": f"{scene_id} detail",
            "primitive_ids": [f"{section_id}_primitive"],
            "passage_ids": [f"{scene_id}_passage"],
            "host_moves": host_moves,
            "estimated_duration_seconds": 90,
        }
    )


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A folder opens.",
        threat_or_unresolved_action="The next order is still unclear.",
        opening_question="What actually changes first?",
        handoff_scene_card_id="scene_01",
    )


def _scene_with_two_phase_cues(
    scene: SceneCardDraft,
    *,
    phase: str,
) -> SceneCardDraft:
    payload = scene.model_dump(mode="json")
    payload["host_moves"][phase] = [
        {"move_type": "callback", "target": "keep the older thread visible"},
        {"move_type": "naming_note", "target": "name the pressure explicitly"},
    ]
    return SceneCardDraft.model_validate(payload)


def test_scene_job_budget_defaults_are_locked() -> None:
    minified_budget = scene_job_budget_for_mode(PodcastMode.MINIFIED)
    assert minified_budget == {
        "total_min": 18,
        "total_max": 21,
        "opening_min": 2,
        "opening_max": 2,
        "build_min": 11,
        "build_max": 14,
        "turn_min": 2,
        "turn_max": 2,
        "answer_min": 1,
        "answer_max": 1,
        "residue_min": 1,
        "residue_max": 1,
        "close_min": 1,
        "close_max": 1,
        "max_recap_build_scenes": 1,
    }
    full_budget = scene_job_budget_for_mode(PodcastMode.FULL)
    assert full_budget == {
        "total_min": 40,
        "total_max": 48,
        "opening_min": 3,
        "opening_max": 3,
        "build_min": 28,
        "build_max": 33,
        "turn_min": 6,
        "turn_max": 9,
        "answer_min": 1,
        "answer_max": 1,
        "residue_min": 1,
        "residue_max": 1,
        "close_min": 1,
        "close_max": 1,
        "max_recap_build_scenes": 1,
    }
    assert (
        full_budget["opening_min"]
        + full_budget["build_min"]
        + full_budget["turn_min"]
        + full_budget["answer_min"]
        + full_budget["residue_min"]
        + full_budget["close_min"]
    ) == full_budget["total_min"]
    assert (
        full_budget["opening_max"]
        + full_budget["build_max"]
        + full_budget["turn_max"]
        + full_budget["answer_max"]
        + full_budget["residue_max"]
        + full_budget["close_max"]
    ) == full_budget["total_max"]
    full_config = resolve_pipeline_config_for_mode(PipelineConfig())
    minified_config = resolve_pipeline_config_for_mode(
        PipelineConfig(podcast_mode=PodcastMode.MINIFIED)
    )
    assert (full_config.scene_card_target_min, full_config.scene_card_target_max) == (
        40,
        48,
    )
    assert (
        minified_config.scene_card_target_min,
        minified_config.scene_card_target_max,
    ) == (18, 21)


def test_host_move_cue_accepts_legacy_note_and_normalizes_to_target() -> None:
    cue = HostMoveCue.model_validate(
        {
            "move_type": "orient",
            "note": "Set the footing.",
            "address_mode": "i",
        }
    )

    assert cue.target == "Set the footing."
    assert cue.address_mode == "i"
    assert "note" not in cue.model_dump(mode="json")


def test_scene_card_draft_maps_legacy_scene_function_to_scene_job() -> None:
    card = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_01",
            "section_id": "section_01",
            "title": "Arrival",
            "scene_role": "setup",
            "scene_function": "scene",
            "beat_change": "Pressure becomes concrete.",
            "passage_ids": ["p1"],
            "host_moves": {"open": [{"move_type": "orient", "note": "Set the footing."}]},
            "estimated_duration_seconds": 90,
        }
    )

    assert card.scene_role == "context_setup"
    assert card.scene_job == "build"


def test_scene_discovery_agent_builds_payload_and_uses_mode_range() -> None:
    agent = SceneDiscoveryAgent(_mock_llm())
    payload = agent.build_payload(
        synthesis_map={"primitives": [{"id": "p1"}]},
        project_metadata={"podcast_mode": "minified"},
        actor_metadata={"actors": [{"actor_id": "actor_1"}]},
        passage_list=[{"passage_id": "passage_1", "text": "Room detail."}],
        scene_discovery_feedback={"issue": "invalid_scene_jobs"},
    )
    artifact = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": f"candidate_{idx + 1:02d}",
                    "primitive_ids": ["p1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A room becomes a decision point.",
                    "scene_jobs": ["opening"],
                    "anchor_image": "A room and a file.",
                    "why_sceneable": "The beat is visible and oral.",
                }
                for idx in range(scene_discovery_candidate_range_for_mode("minified")[0])
            ]
        }
    )

    validated = agent.validate_result(artifact, payload)

    assert payload["project"]["podcast_mode"] == "minified"
    assert "actor_metadata" in payload
    assert payload["scene_discovery_feedback"]["issue"] == "invalid_scene_jobs"
    assert len(validated.candidates) == 18


def test_scene_discovery_agent_builds_richer_mode_specific_instructions() -> None:
    agent = SceneDiscoveryAgent(_mock_llm())

    minified_instructions = agent.build_instructions(
        {"project": {"podcast_mode": "minified"}}
    )
    full_instructions = agent.build_instructions(
        {"project": {"podcast_mode": "full"}}
    )

    assert "This is a `minified` run." in minified_instructions
    assert "Return 18–26 candidates." in minified_instructions
    assert "DISCOVERY WORKFLOW" in minified_instructions
    assert "MERGE VS SEPARATE" in minified_instructions
    assert "`scene_discovery_feedback` (optional): retry feedback" in minified_instructions
    assert "visible consequence, irreversible turn, or immediate aftermath" in minified_instructions
    assert "SELF-CHECK BEFORE RETURNING" in minified_instructions
    assert "This is a `full` run." in full_instructions
    assert "Return 53–79 candidates." in full_instructions


def test_scene_discovery_agent_prepare_retry_payload_adds_role_feedback() -> None:
    agent = SceneDiscoveryAgent(_mock_llm())

    next_payload = agent.prepare_retry_payload(
        {"project": {"podcast_mode": "minified"}},
        RetryableGenerationError(
            "Schema validation failed for scene_discovery",
            data={
                "raw_payload": {
                    "candidates": [
                        {
                            "candidate_id": "candidate_01",
                            "scene_jobs": ["opening", "cost", "complication"],
                        }
                    ]
                }
            },
        ),
    )

    assert next_payload["scene_discovery_feedback"]["issue"] == "invalid_scene_jobs"
    assert next_payload["scene_discovery_feedback"]["candidate_ids"] == ["candidate_01"]
    assert next_payload["scene_discovery_feedback"]["invalid_roles"] == [
        "complication",
        "cost",
    ]


def test_scene_discovery_agent_retries_when_candidate_count_out_of_range() -> None:
    agent = SceneDiscoveryAgent(_mock_llm())
    payload = agent.build_payload(
        synthesis_map={"primitives": [{"id": "p1"}]},
        project_metadata={"podcast_mode": "minified"},
        actor_metadata=None,
        passage_list=[{"passage_id": "passage_1", "text": "Room detail."}],
    )
    artifact = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "primitive_ids": ["p1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A room becomes a decision point.",
                    "scene_jobs": ["opening"],
                    "anchor_image": "A room and a file.",
                    "why_sceneable": "The beat is visible and oral.",
                }
            ]
        }
    )

    with pytest.raises(RetryableGenerationError, match="candidate count"):
        agent.validate_result(artifact, payload)


def test_episode_architecture_agent_requires_answer_residue_and_promised_beat_accounting() -> None:
    agent = EpisodeArchitectureAgent(_mock_llm())
    payload = {
        "episode": _strategy_episode().model_dump(mode="json"),
        "project": {
            "architecture_section_target_min": 6,
            "architecture_section_target_max": 12,
        },
    }
    architecture = EpisodeArchitecture.model_validate(
        {
            "episode_number": 1,
            "major_turn_section_id": "section_04",
            "sections": [_section("section_01", purpose="opening").model_dump(mode="json")]
            + [_section(f"section_0{idx}").model_dump(mode="json") for idx in range(2, 6)]
            + [_section("section_06", purpose="closing").model_dump(mode="json")],
        }
    )

    with pytest.raises(ValueError, match="answer_section_id"):
        agent.validate_result(architecture, payload)


def test_validate_plan_transition_enforces_answer_and_residue_section_ownership() -> None:
    architecture = _architecture()
    strategy_episode = _strategy_episode()
    scenes = [
        _scene("scene_01", section_id="section_01", scene_job="opening"),
        _scene("scene_02", section_id="section_02", scene_job="build"),
        _scene("scene_03", section_id="section_03", scene_job="residue"),
        _scene("scene_04", section_id="section_04", scene_job="turn"),
        _scene("scene_05", section_id="section_05", scene_job="answer"),
        _scene("scene_06", section_id="section_06", scene_job="close"),
    ]
    plan = EpisodePlanDraft.model_construct(
        episode_number=1,
        framing=_framing(),
        scene_cards=scenes,
        answer_scene_card_id="scene_05",
        residue_scene_card_id="scene_03",
        dropped_support_primitive_reasons={},
    )

    with pytest.raises(
        ComplianceViolationError,
        match="Residue scene does not belong to architecture.residue_section_id",
    ):
        _validate_plan_transition(
            strategy_episode=strategy_episode,
            architecture=architecture,
            plan=plan,
        )


def test_build_spine_plan_diagnostics_reports_scene_job_budget_fields() -> None:
    plan = EpisodePlanDraft(
        episode_number=1,
        framing=_framing(),
        scene_cards=[
            _scene("scene_01", section_id="section_01", scene_job="opening"),
            _scene("scene_02", section_id="section_02", scene_job="build"),
            _scene("scene_03", section_id="section_03", scene_job="turn"),
            _scene("scene_04", section_id="section_04", scene_job="answer"),
            _scene("scene_05", section_id="section_05", scene_job="build"),
            _scene("scene_06", section_id="section_06", scene_job="close"),
        ],
        answer_scene_card_id="scene_04",
        residue_scene_card_id=None,
    )

    diagnostics = _build_spine_plan_diagnostics(
        strategy_episode=_strategy_episode(),
        plan=plan,
        scene_job_budget=scene_job_budget_for_mode("minified"),
    )

    assert diagnostics["scene_job_counts"]["opening"] == 1
    assert diagnostics["answer_scene_card_id"] == "scene_04"
    assert diagnostics["residue_scene_present"] is False
    assert "scene_budget_residue_missing" in diagnostics["scene_job_budget_warnings"]
    assert "scene_budget_close_precedes_residue" not in diagnostics["scene_job_budget_warnings"]


def test_planning_instructions_reference_scene_job_budget_and_target() -> None:
    agent = EpisodePlanningAgent(_mock_llm())

    assert "scene_job_budget" in agent.instructions
    assert "`answer_scene_card_id`" in agent.instructions
    assert "`residue_scene_card_id`" in agent.instructions
    assert "`scene_job`" in agent.instructions
    assert "`target`" in agent.instructions
    assert "`scene_function`" not in agent.instructions


def test_episode_planning_agent_allows_two_host_cues_in_one_phase() -> None:
    agent = EpisodePlanningAgent(_mock_llm())
    payload = {"project": {"podcast_mode": "full"}}
    artifact = EpisodePlanDraft(
        episode_number=1,
        framing=_framing(),
        scene_cards=[
            _scene("scene_01", section_id="section_01", scene_job="opening"),
            _scene("scene_02", section_id="section_02", scene_job="build"),
            _scene("scene_03", section_id="section_03", scene_job="turn"),
            _scene("scene_04", section_id="section_04", scene_job="answer"),
            _scene("scene_05", section_id="section_05", scene_job="residue"),
            _scene_with_two_phase_cues(
                _scene("scene_06", section_id="section_06", scene_job="close"),
                phase="close",
            ),
        ],
        answer_scene_card_id="scene_04",
        residue_scene_card_id="scene_05",
    )

    validated = agent.validate_result(artifact, payload)

    assert len(validated.scene_cards[-1].host_moves.close) == 2


def test_validate_plan_transition_allows_two_host_cues_and_keeps_diagnostics() -> None:
    architecture = _architecture()
    strategy_episode = _strategy_episode()
    plan = EpisodePlanDraft(
        episode_number=1,
        framing=_framing(),
        scene_cards=[
            _scene("scene_01", section_id="section_01", scene_job="opening"),
            _scene("scene_02", section_id="section_02", scene_job="build"),
            _scene("scene_03", section_id="section_03", scene_job="turn"),
            _scene("scene_04", section_id="section_04", scene_job="build"),
            _scene("scene_05", section_id="section_05", scene_job="answer"),
            _scene("scene_06", section_id="section_06", scene_job="residue"),
            _scene_with_two_phase_cues(
                _scene("scene_07", section_id="section_06", scene_job="close"),
                phase="close",
            ),
        ],
        answer_scene_card_id="scene_05",
        residue_scene_card_id="scene_06",
        dropped_support_primitive_reasons={},
    )

    validated = _validate_plan_transition(
        strategy_episode=strategy_episode,
        architecture=architecture,
        plan=plan,
    )
    diagnostics, warnings = _build_host_move_plan_diagnostics(
        scene_cards=validated.scene_cards,
        architecture=architecture,
        narrator_profile=SeriesNarratorProfile(),
    )

    assert diagnostics["host_phase_multiple_cues"] == ["scene_07:close"]
    assert any("host_phase_multiple_cues" in warning for warning in warnings)
