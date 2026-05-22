"""Tests for episode-planning postcheck retries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _validate_plan_transition
from podcast_agent.schemas.models import (
    ActorExplanationPlan,
    ActorPrimitive,
    ActorMetadata,
    ActorProfile,
    ArtifactPrimitive,
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EpisodePlanDraft,
    EventPrimitive,
    NarrativeStrategy,
    PipelineConfig,
    PrimitiveSalience,
    PrimitiveSubstrate,
    SupportPrimitiveRole,
    StrategyEpisode,
    SynthesisMap,
    ThematicCorpus,
    ThematicProject,
    EpisodeSpine,
)


def _strategy_episode() -> StrategyEpisode:
    return StrategyEpisode.model_construct(
        episode_number=1,
        title="Episode 1",
        thematic_focus="Focus",
        arc_summary="Arc",
        episode_spine=EpisodeSpine.model_construct(
            listener_question="What happened?",
            argument="A working claim.",
            core_primitive_ids=["core_1"],
            support_primitive_roles={
                "support_1": SupportPrimitiveRole.MECHANISM,
                "support_2": SupportPrimitiveRole.STAKES,
                "support_3": SupportPrimitiveRole.CONSEQUENCE,
            },
        ),
    )


def _primitive(
    primitive_id: str,
    title: str,
    *,
    passage_id: str,
) -> object:
    if primitive_id.startswith("support_1"):
        return ArtifactPrimitive(
            id=primitive_id,
            substrate=PrimitiveSubstrate.ARTIFACTS,
            title=title,
            core_passage_ids=[passage_id],
            artifact_type="detail",
            artifact_label=title,
            artifact_detail="A telling detail frames the scene.",
        )
    if primitive_id.startswith("support_2"):
        return EventPrimitive(
            id=primitive_id,
            substrate=PrimitiveSubstrate.EVENTS,
            title=title,
            core_passage_ids=[passage_id],
            actor_ids=["actor_1"],
            event_type="consequence",
            what_happened="Families are driven from their homes.",
            event_result="The cost becomes public.",
        )
    if primitive_id.startswith("support_3"):
        return ActorPrimitive(
            id=primitive_id,
            substrate=PrimitiveSubstrate.ACTOR_PORTRAITS,
            title=title,
            core_passage_ids=[passage_id],
            actor_ids=["actor_1"],
            focus_actor_id="actor_1",
            actor_label="Actor 1",
            goal_or_project="Hold the center.",
            stakes_or_fears="Collapse would be final.",
            operating_pressure="Allies are unreliable.",
        )
    return EventPrimitive(
        id=primitive_id,
        substrate=PrimitiveSubstrate.EVENTS,
        title=title,
        core_passage_ids=[passage_id],
        salience=PrimitiveSalience(score=0.8, justification="Core turn."),
        event_type="turning_point",
        what_happened="A decisive move forces the turn.",
        event_result="The balance breaks.",
    )


def _episode_architecture() -> EpisodeArchitecture:
    return EpisodeArchitecture.model_construct(
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
                "host_moves": {
                    "open": [
                        {
                            "move_type": "orient",
                            "note": "Frame the opening conditions cleanly.",
                        }
                    ]
                },
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
                "host_moves": {
                    "open": [
                        {
                            "move_type": "clarify",
                            "note": "Mark the hinge in the action.",
                        }
                    ]
                },
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
                "host_moves": {
                    "open": [
                        {
                            "move_type": "evaluate",
                            "note": "Make the fallout legible.",
                        }
                    ]
                },
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
                "host_moves": {
                    "open": [
                        {
                            "move_type": "callback",
                            "note": "Close by tying the residue back to the opening.",
                        }
                    ]
                },
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


def test_plan_series_ignores_legacy_scene_primitive_ids(monkeypatch, tmp_path):
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
        primitives=[
            _primitive("core_1", "Core", passage_id="p_core_1"),
            _primitive("support_1", "Support 1", passage_id="p_support_1"),
            _primitive("support_2", "Support 2", passage_id="p_support_2"),
            _primitive("support_3", "Support 3", passage_id="p_support_3"),
        ],
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
    assert len(captured_payloads) == 1
    assert "planning_feedback" not in captured_payloads[0]


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


def test_plan_series_does_not_raise_on_legacy_scene_primitive_ids(
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
        primitives=[
            _primitive("core_1", "Core", passage_id="p_core_1"),
            _primitive("support_1", "Support 1", passage_id="p_support_1"),
            _primitive("support_2", "Support 2", passage_id="p_support_2"),
            _primitive("support_3", "Support 3", passage_id="p_support_3"),
        ],
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
    assert len(captured_payloads) == 1


def test_plan_series_warns_without_retry_on_late_actor_introduction(
    monkeypatch, tmp_path
):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)
    orchestrator = _build_orchestrator(monkeypatch)
    orchestrator.episode_planning_agent.max_retry_attempts = 2
    captured_payloads: list[dict] = []
    warning_messages: list[str] = []
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.logger.warning",
        lambda msg, *args: warning_messages.append(msg % args),
    )

    architecture = _episode_architecture().model_copy(
        update={
            "sections": [
                _episode_architecture().sections[0].model_copy(
                    update={
                        "actor_explanations": [
                            ActorExplanationPlan(
                                actor_id="actor_1",
                                stage="introduce",
                                background_depth="appositive",
                                role_label="the visible political face of the crisis",
                                source_primitive_ids=["support_2"],
                                source_passage_ids=["p_support_2"],
                                intro_facts=["Actor 1 now carries the conflict publicly."],
                                why_now="The actor becomes concrete in the opening section.",
                                preferred_plain_gloss="the actor now carrying the crisis",
                            )
                        ]
                    }
                ),
                *_episode_architecture().sections[1:],
            ]
        }
    )

    def fake_planning_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.episode_planning_agent.response_model.model_validate(
            {
                "episode_number": 1,
                "framing": {
                    "opening_image": "A corridor outside the chamber.",
                    "threat_or_unresolved_action": "No one knows who will take responsibility.",
                    "opening_question": "Who now owns the crisis?",
                    "handoff_scene_card_id": "scene_early",
                },
                "scene_cards": [
                    {
                        "scene_id": "scene_early",
                        "section_id": "s01",
                        "title": "A face in the corridor",
                        "scene_role": "actor_setup",
                        "scene_function": "scene",
                        "beat_change": "The actor is visible before the formal introduction lands.",
                        "actors": [
                            {
                                "name": "Actor 1",
                                "actor_id": "actor_1",
                                "presence": "primary",
                                "role_label": "the visible political face of the crisis",
                                "source_primitive_ids": ["support_2"],
                                "source_passage_ids": ["p_support_2"],
                                "intro_facts": ["Actor 1 now carries the conflict publicly."],
                                "why_now": "The actor becomes concrete in the opening section.",
                            }
                        ],
                        "passage_ids": ["p_support_2"],
                        "host_moves": {
                            "open": [
                                {
                                    "move_type": "orient",
                                    "note": "Enter through the corridor outside the chamber.",
                                }
                            ]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_intro",
                        "section_id": "s01",
                        "title": "The formal naming",
                        "scene_role": "actor_setup",
                        "scene_function": "scene",
                        "beat_change": "The same actor is now explicitly introduced.",
                        "actors": [
                            {
                                "name": "Actor 1",
                                "actor_id": "actor_1",
                                "presence": "primary",
                                "explanation_stage": "introduce",
                                "background_depth": "appositive",
                                "role_label": "the visible political face of the crisis",
                                "source_primitive_ids": ["support_2"],
                                "source_passage_ids": ["p_support_2"],
                                "intro_facts": ["Actor 1 now carries the conflict publicly."],
                                "why_now": "The actor becomes concrete in the opening section.",
                            }
                        ],
                        "passage_ids": ["p_support_2"],
                        "host_moves": {
                            "open": [
                                {
                                    "move_type": "clarify",
                                    "note": "Name the actor once the room has come into focus.",
                                }
                            ]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_2",
                        "section_id": "s02",
                        "title": "The turn gathers",
                        "scene_role": "action",
                        "scene_function": "scene",
                        "beat_change": "The pressure starts to move.",
                        "passage_ids": ["p_core_1"],
                        "host_moves": {
                            "open": [{"move_type": "orient", "note": "Enter through the chamber doors."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_3",
                        "section_id": "s03",
                        "title": "The cost lands",
                        "scene_role": "fallout",
                        "scene_function": "scene",
                        "beat_change": "The cost is now visible.",
                        "passage_ids": ["p_support_2"],
                        "host_moves": {
                            "open": [{"move_type": "evaluate", "note": "Let the cost settle on the room."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_4",
                        "section_id": "s04",
                        "title": "The residue remains",
                        "scene_role": "implication",
                        "scene_function": "landing",
                        "beat_change": "The residue is named.",
                        "passage_ids": ["p_support_3"],
                        "host_moves": {
                            "open": [{"move_type": "callback", "note": "Carry the first corridor image into the close."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                ],
            }
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
        primitives=[
            _primitive("core_1", "Core", passage_id="p_core_1"),
            _primitive("support_1", "Support 1", passage_id="p_support_1"),
            _primitive("support_2", "Support 2", passage_id="p_support_2"),
            _primitive("support_3", "Support 3", passage_id="p_support_3"),
        ],
    )
    corpus = ThematicCorpus(project_id="proj")
    actor_metadata = ActorMetadata(
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor 1",
                actor_type="person",
            )
        ]
    )

    plans, _metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            [architecture],
            corpus,
            tmp_path,
            actor_metadata=actor_metadata,
        )
    )

    assert [plan.episode_number for plan in plans] == [1]
    assert len(captured_payloads) == 1
    assert not any("episode_planning_retry_scheduled" in message for message in warning_messages)
    assert any(
        "late_actor_introduction_scene_links:actor_1:scene_early" in message
        for message in warning_messages
    )


def test_plan_series_warns_without_retry_on_missing_actor_explanation_link(
    monkeypatch, tmp_path
):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)
    orchestrator = _build_orchestrator(monkeypatch)
    orchestrator.episode_planning_agent.max_retry_attempts = 2
    captured_payloads: list[dict] = []
    warning_messages: list[str] = []
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.logger.warning",
        lambda msg, *args: warning_messages.append(msg % args),
    )

    architecture = _episode_architecture().model_copy(
        update={
            "sections": [
                _episode_architecture().sections[0].model_copy(
                    update={
                        "actor_explanations": [
                            ActorExplanationPlan(
                                actor_id="actor_1",
                                stage="introduce",
                                background_depth="appositive",
                                role_label="the visible political face of the crisis",
                                source_primitive_ids=["support_2"],
                                source_passage_ids=["p_support_2"],
                                intro_facts=["Actor 1 now carries the conflict publicly."],
                                why_now="The actor becomes concrete in the opening section.",
                                preferred_plain_gloss="the actor now carrying the crisis",
                            )
                        ]
                    }
                ),
                *_episode_architecture().sections[1:],
            ]
        }
    )

    def fake_planning_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.episode_planning_agent.response_model.model_validate(
            {
                "episode_number": 1,
                "framing": {
                    "opening_image": "A corridor outside the chamber.",
                    "threat_or_unresolved_action": "No one knows who will take responsibility.",
                    "opening_question": "Who now owns the crisis?",
                    "handoff_scene_card_id": "scene_early",
                },
                "scene_cards": [
                    {
                        "scene_id": "scene_early",
                        "section_id": "s01",
                        "title": "A face in the corridor",
                        "scene_role": "actor_setup",
                        "scene_function": "scene",
                        "beat_change": "The actor is visible before any formal explanation lands.",
                        "actors": [
                            {
                                "name": "Actor 1",
                                "actor_id": "actor_1",
                                "presence": "primary",
                                "role_label": "the visible political face of the crisis",
                                "source_primitive_ids": ["support_2"],
                                "source_passage_ids": ["p_support_2"],
                                "intro_facts": ["Actor 1 now carries the conflict publicly."],
                                "why_now": "The actor becomes concrete in the opening section.",
                            }
                        ],
                        "passage_ids": ["p_support_2"],
                        "host_moves": {
                            "open": [
                                {
                                    "move_type": "orient",
                                    "note": "Enter through the corridor outside the chamber.",
                                }
                            ]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_2",
                        "section_id": "s02",
                        "title": "The turn gathers",
                        "scene_role": "action",
                        "scene_function": "scene",
                        "beat_change": "The pressure starts to move.",
                        "passage_ids": ["p_core_1"],
                        "host_moves": {
                            "open": [{"move_type": "orient", "note": "Enter through the chamber doors."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_3",
                        "section_id": "s03",
                        "title": "The cost lands",
                        "scene_role": "fallout",
                        "scene_function": "scene",
                        "beat_change": "The cost is now visible.",
                        "passage_ids": ["p_support_2"],
                        "host_moves": {
                            "open": [{"move_type": "evaluate", "note": "Let the cost settle on the room."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                    {
                        "scene_id": "scene_4",
                        "section_id": "s04",
                        "title": "The residue remains",
                        "scene_role": "implication",
                        "scene_function": "landing",
                        "beat_change": "The residue is named.",
                        "passage_ids": ["p_support_3"],
                        "host_moves": {
                            "open": [{"move_type": "callback", "note": "Carry the first corridor image into the close."}]
                        },
                        "estimated_duration_seconds": 90,
                    },
                ],
            }
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
        primitives=[
            _primitive("core_1", "Core", passage_id="p_core_1"),
            _primitive("support_1", "Support 1", passage_id="p_support_1"),
            _primitive("support_2", "Support 2", passage_id="p_support_2"),
            _primitive("support_3", "Support 3", passage_id="p_support_3"),
        ],
    )
    corpus = ThematicCorpus(project_id="proj")
    actor_metadata = ActorMetadata(
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor 1",
                actor_type="person",
            )
        ]
    )

    plans, _metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            [architecture],
            corpus,
            tmp_path,
            actor_metadata=actor_metadata,
        )
    )

    assert [plan.episode_number for plan in plans] == [1]
    assert len(captured_payloads) == 1
    assert not any("episode_planning_retry_scheduled" in message for message in warning_messages)
    assert any(
        "missing_actor_explanation_scene_links:s01:actor_1:introduce" in message
        for message in warning_messages
    )
