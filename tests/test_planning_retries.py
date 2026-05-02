"""Tests for episode-planning postcheck retries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _validate_plan_transition
from podcast_agent.schemas.models import (
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EpisodePlanDraft,
    NarrativeStrategy,
    PipelineConfig,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitive,
    ThematicCorpus,
    ThematicProject,
    VerdictMode,
    EpisodeSpine,
)


def _strategy_episode() -> StrategyEpisode:
    return StrategyEpisode(
        episode_number=1,
        title="Episode 1",
        driving_question="What happened?",
        thematic_focus="Focus",
        arc_summary="Arc",
        episode_spine=EpisodeSpine(
            listener_question="What happened?",
            working_claim="A working claim.",
            target_end_state="A target state.",
            verdict_mode=VerdictMode.CONSTRAIN,
            primary_counterposition="A counterposition.",
            core_primitive_ids=["core_1"],
            support_primitive_roles={
                "support_1": "mechanism",
                "support_2": "stakes",
                "support_3": "consequence",
            },
        ),
    )


def _primitive(
    primitive_id: str,
    title: str,
    *,
    passage_id: str,
) -> SynthesisPrimitive:
    return SynthesisPrimitive(
        id=primitive_id,
        title=title,
        summary=f"{title} summary",
        axis_ids=["axis_1"],
        core_passage_ids=[passage_id],
        narrative_importance_score=0.8,
    )


def _episode_architecture() -> EpisodeArchitecture:
    return EpisodeArchitecture(
        episode_number=1,
        major_turn_section_id="s02",
        sections=[
            ArchitectureSection(
                section_id="s01",
                purpose="opening",
                approx_runtime_minutes=10.0,
                primitive_ids=["support_1"],
                section_question="Q1?",
                section_resolution="R1",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                argument_role="frame",
                inference_mode="scene_first",
                pressure_type="communal",
                resolution_type="redefinition",
            ),
            ArchitectureSection(
                section_id="s02",
                purpose="turn",
                approx_runtime_minutes=30.0,
                primitive_ids=["core_1"],
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["s01"],
                argument_role="test_viability",
                inference_mode="scene_first",
                pressure_type="communal",
                resolution_type="reversal",
            ),
            ArchitectureSection(
                section_id="s03",
                purpose="counterpressure",
                approx_runtime_minutes=30.0,
                primitive_ids=["support_2"],
                section_question="Q3?",
                section_resolution="R3",
                entry_state="E3",
                exit_state="X3",
                transition_logic="T3",
                depends_on_section_ids=["s02"],
                argument_role="convert_event_into_structure",
                inference_mode="scene_first",
                pressure_type="communal",
                resolution_type="escalation",
            ),
            ArchitectureSection(
                section_id="s04",
                purpose="closing",
                approx_runtime_minutes=30.0,
                primitive_ids=["support_3"],
                section_question="Q4?",
                section_resolution="R4",
                entry_state="E4",
                exit_state="X4",
                transition_logic="T4",
                depends_on_section_ids=["s03"],
                argument_role="close",
                inference_mode="scene_first",
                pressure_type="communal",
                resolution_type="containment",
            ),
        ],
    )


def _valid_plan_payload(payload: dict, *, use_invalid_section_primitive: bool) -> dict:
    section_two_primitives = ["core_1", "support_1"] if use_invalid_section_primitive else ["core_1"]
    return {
        "episode_number": payload["strategy_episode"]["episode_number"],
        "framing": {
            "opening_image": "Image",
            "threat_or_unresolved_action": "Threat",
            "opening_question": "Question",
            "handoff_scene_card_id": "scene_1",
        },
        "scene_cards": [
            {
                "scene_id": "scene_1",
                "section_id": "s01",
                "title": "Scene 1",
                "scene_role": "setup",
                "dominant_primitive_id": "support_1",
                "spine_relation": "set_stakes",
                "state_effect": "State 1",
                "primitive_ids": ["support_1"],
                "passage_ids": ["p_support_1"],
                "estimated_duration_seconds": 120,
            },
            {
                "scene_id": "scene_2",
                "section_id": "s02",
                "title": "Scene 2",
                "scene_role": "action",
                "dominant_primitive_id": "core_1",
                "spine_relation": "spine_advance",
                "state_effect": "State 2",
                "primitive_ids": section_two_primitives,
                "passage_ids": ["p_core_1"],
                "estimated_duration_seconds": 120,
            },
            {
                "scene_id": "scene_3",
                "section_id": "s03",
                "title": "Scene 3",
                "scene_role": "consequence",
                "dominant_primitive_id": "support_2",
                "spine_relation": "show_consequence",
                "state_effect": "State 3",
                "primitive_ids": ["support_2"],
                "passage_ids": ["p_support_2"],
                "estimated_duration_seconds": 120,
            },
            {
                "scene_id": "scene_4",
                "section_id": "s04",
                "title": "Scene 4",
                "scene_role": "synthesis",
                "dominant_primitive_id": "support_3",
                "spine_relation": "show_consequence",
                "state_effect": "State 4",
                "primitive_ids": ["support_3"],
                "passage_ids": ["p_support_3"],
                "estimated_duration_seconds": 120,
            },
        ],
    }


def _valid_plan_draft(
    *,
    dropped_support_primitive_reasons: dict[str, str] | None = None,
) -> EpisodePlanDraft:
    payload = _valid_plan_payload(
        {"strategy_episode": {"episode_number": 1}},
        use_invalid_section_primitive=False,
    )
    payload["dropped_support_primitive_reasons"] = dict(
        dropped_support_primitive_reasons or {}
    )
    return EpisodePlanDraft.model_validate(payload)


def _build_orchestrator(monkeypatch: pytest.MonkeyPatch) -> PipelineOrchestrator:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: SimpleNamespace(set_run_logger=lambda logger: None),
    )
    return PipelineOrchestrator()


def test_plan_series_retries_on_primitive_outside_architecture_section(monkeypatch, tmp_path):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)
    orchestrator = _build_orchestrator(monkeypatch)
    orchestrator.episode_planning_agent.max_retry_attempts = 2
    captured_payloads: list[dict] = []

    def fake_planning_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.episode_planning_agent.response_model.model_validate(
            _valid_plan_payload(payload, use_invalid_section_primitive=len(captured_payloads) == 1)
        )

    orchestrator.episode_planning_agent.run = fake_planning_run
    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        episode_count=1,
        config=PipelineConfig(),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="Because.",
        series_arc="Arc.",
        episodes=[_strategy_episode()],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [_primitive("core_1", "Core", passage_id="p_core_1")],
            "decisions_and_nondecisions": [],
            "set_piece_scenes": [],
            "telling_details": [_primitive("support_1", "Support 1", passage_id="p_support_1")],
            "human_costs": [_primitive("support_2", "Support 2", passage_id="p_support_2")],
            "character_engines": [_primitive("support_3", "Support 3", passage_id="p_support_3")],
            "coalitions_and_fault_lines": [],
            "systems_and_operating_logics": [],
            "misreadings_and_fantasies": [],
            "contested_explanations": [],
            "perspective_windows": [],
            "moral_traps": [],
            "afterlives": [],
            "recurring_images_and_symbols": [],
            "ironies_and_reversals": [],
        },
    )
    corpus = ThematicCorpus(project_id="proj")

    plans, _metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            [_episode_architecture()],
            corpus,
            tmp_path,
        )
    )

    assert [plan.episode_number for plan in plans] == [1]
    assert len(captured_payloads) == 2
    assert "planning_feedback" not in captured_payloads[0]
    assert captured_payloads[1]["planning_feedback"]["issue"] == "primitive_outside_architecture_section"
    assert captured_payloads[1]["planning_feedback"]["invalid_scene_primitives"] == {
        "scene_2": ["support_1"]
    }


def test_validate_plan_transition_allows_dropped_support_primitives_missing_after_architecture_filtering():
    plan = _valid_plan_draft(
        dropped_support_primitive_reasons={
            "support_4": "It does not fit the final scene chain."
        }
    )

    validated_plan = _validate_plan_transition(
        strategy_episode=_strategy_episode(),
        architecture=_episode_architecture(),
        plan=plan,
    )

    assert validated_plan == plan
    assert validated_plan.dropped_support_primitive_reasons == {
        "support_4": "It does not fit the final scene chain."
    }


def test_plan_series_raises_after_retry_exhaustion_on_primitive_outside_architecture_section(
    monkeypatch, tmp_path
):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)
    orchestrator = _build_orchestrator(monkeypatch)
    orchestrator.episode_planning_agent.max_retry_attempts = 2
    captured_payloads: list[dict] = []

    def fake_planning_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.episode_planning_agent.response_model.model_validate(
            _valid_plan_payload(payload, use_invalid_section_primitive=True)
        )

    orchestrator.episode_planning_agent.run = fake_planning_run
    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        episode_count=1,
        config=PipelineConfig(),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="Because.",
        series_arc="Arc.",
        episodes=[_strategy_episode()],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [_primitive("core_1", "Core", passage_id="p_core_1")],
            "decisions_and_nondecisions": [],
            "set_piece_scenes": [],
            "telling_details": [_primitive("support_1", "Support 1", passage_id="p_support_1")],
            "human_costs": [_primitive("support_2", "Support 2", passage_id="p_support_2")],
            "character_engines": [_primitive("support_3", "Support 3", passage_id="p_support_3")],
            "coalitions_and_fault_lines": [],
            "systems_and_operating_logics": [],
            "misreadings_and_fantasies": [],
            "contested_explanations": [],
            "perspective_windows": [],
            "moral_traps": [],
            "afterlives": [],
            "recurring_images_and_symbols": [],
            "ironies_and_reversals": [],
        },
    )
    corpus = ThematicCorpus(project_id="proj")

    with pytest.raises(Exception, match="Episode plan used primitives outside their architecture section"):
        asyncio.run(
            orchestrator._plan_series(
                project,
                synthesis_map,
                strategy,
                [_episode_architecture()],
                corpus,
                tmp_path,
            )
        )

    assert len(captured_payloads) == 2
    assert captured_payloads[1]["planning_feedback"]["issue"] == "primitive_outside_architecture_section"
