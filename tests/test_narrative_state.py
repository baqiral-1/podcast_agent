from __future__ import annotations

import pytest

from podcast_agent.agents.narrative_state_reconciler import (
    NarrativeStateReconcilerAgent,
)
from podcast_agent.langchain.runnables import ComplianceViolationError
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    _build_continuity_contract,
    _build_continuity_realization_diagnostics,
    _build_host_policy_payload,
    _build_initial_narrative_state,
    _validate_continuity_recap_requirement,
)
from podcast_agent.schemas.models import (
    ContinuityCarryItem,
    HostMoveCue,
    NarrativeState,
    StrategyEpisode,
)
from podcast_agent.schemas.models import SeriesNarratorProfile


def test_strategy_episode_backfills_narrative_agenda_from_legacy_contract() -> None:
    episode = StrategyEpisode.model_validate(
        {
            "episode_number": 2,
            "title": "Episode 2",
            "thematic_focus": "Power moves behind the scenes",
            "arc_summary": "The visible cabinet and the hidden state diverge.",
            "unresolved_questions": ["Who is actually governing?"],
                "episode_spine": {
                    "listener_problem": "Who is actually governing?",
                    "episode_answer": "Parallel institutions are already overruling the cabinet.",
                    "pressure_line": "Formal offices keep losing ground to shadow authority.",
                    "core_primitive_ids": [
                        "primitive_1",
                        "primitive_2",
                        "primitive_3",
                        "primitive_4",
                        "primitive_5",
                        "primitive_6",
                    ],
                    "support_primitive_roles": {
                        "primitive_7": "mechanism",
                        "primitive_8": "stakes",
                    },
                    "recall_primitive_ids": [],
                },
            "authorial_contract": {
                "analysis_weight": "heavy",
                "priority_moves": ["institutional_clarifier"],
                "governing_lenses": ["state capture"],
                "must_clarify_terms": ["velayat_faqih"],
                "must_clarify_institutions": ["revolutionary_council"],
                "introduce_explanation_item_ids": ["velayat_faqih"],
                "remind_explanation_item_ids": ["clerical_networks"],
                "introduce_actor_ids": ["bazargan"],
                "remind_actor_ids": ["khomeini"],
                "callback_obligations": ["Return to the hidden ministries image."],
            },
        }
    )

    assert episode.narrator_contract.analysis_weight == "heavy"
    assert episode.narrator_contract.priority_moves == ["institutional_clarifier"]
    assert episode.narrative_agenda.listener.introduce_explanation_item_ids == [
        "velayat_faqih"
    ]
    assert episode.narrative_agenda.listener.remind_actor_ids == ["khomeini"]
    assert [item.label for item in episode.narrative_agenda.listener.carry_forward_memory] == [
        "velayat_faqih",
        "revolutionary_council",
    ]
    assert (
        episode.narrative_agenda.listener.carry_forward_memory[0].source_episode_number
        == 2
    )
    assert episode.narrative_agenda.listener.question_moves[0].question_id == (
        "episode_2_question_1"
    )
    assert episode.narrative_agenda.listener.question_moves[0].action == "open"
    assert episode.narrative_agenda.listener.memory_thread_moves[0].thread_id == (
        "episode_2_callback_1"
    )


def test_host_move_cue_accepts_epistemic_move_types() -> None:
    assert HostMoveCue.model_validate(
        {"move_type": "uncertainty", "target": "what changed"}
    ).move_type == "uncertainty"
    assert HostMoveCue.model_validate(
        {"move_type": "revision", "target": "one bad ruler story"}
    ).move_type == "revision"
    assert HostMoveCue.model_validate(
        {"move_type": "surprise", "target": "parallel ministries"}
    ).move_type == "surprise"


def test_build_host_policy_payload_uses_narrative_state_context() -> None:
    profile = SeriesNarratorProfile()
    pre_state = _build_initial_narrative_state("proj-1")
    post_state = NarrativeState.model_validate(
        {
            "project_id": "proj-1",
            "next_episode_number": 2,
            "listener": pre_state.listener.model_dump(mode="json"),
            "host": {
                "mysteries": [],
                "assumptions": [],
                "working_theories": [],
                "recent_revisions": ["one bad ruler story"],
                "confidence_posture": "tentative",
                "last_episode_takeaway": "The institutions were weaker than they looked.",
            },
        }
    )

    payload = _build_host_policy_payload(
        profile,
        narrative_state_pre=pre_state,
        narrative_state_post=post_state,
    )

    assert payload["pronoun_policy"]["allow_first_person_singular"] is True
    assert payload["pronoun_policy"]["first_person_singular_allowed_for"] == [
        "uncertainty",
        "revision",
        "surprise",
        "closing_reflection",
    ]
    assert "revision" in payload["allowed_moves"]
    assert payload["authorial_policy"]["host_confidence_posture"] == "tentative"


def test_heuristic_reconciler_updates_listener_and_host_state() -> None:
    llm = HeuristicLLMClient()
    agent = NarrativeStateReconcilerAgent(llm)
    state_pre = _build_initial_narrative_state("proj-1")
    payload = agent.build_payload(
        episode_number=1,
        project_id="proj-1",
        narrative_state_pre=state_pre.model_dump(mode="json"),
        strategy_episode={
            "episode_number": 1,
            "narrative_agenda": {
                "listener": {
                    "introduce_explanation_item_ids": ["velayat_faqih"],
                    "introduce_actor_ids": ["bazargan"],
                    "carry_forward_memory": ["The state is weaker than it looks."],
                    "episode_takeaway": "Parallel institutions are already constraining the cabinet.",
                },
                "host": {
                    "mystery_moves": [
                        {
                            "mystery_id": "army_nonintervention",
                            "action": "open",
                            "text": "Why didn't the army intervene?",
                        }
                    ],
                    "assumption_moves": [
                        {
                            "assumption_id": "one_bad_ruler_story",
                            "action": "introduce",
                            "statement": "This might just be a story about one ruler losing support.",
                        }
                    ],
                    "theory_moves": [
                        {
                            "theory_id": "brittle_institutions",
                            "action": "propose",
                            "statement": "The system may have been hollow before it visibly fell.",
                        }
                    ],
                    "episode_takeaway": "The host now suspects institutional brittleness mattered more than personal unpopularity.",
                },
            },
        },
        architecture={
            "episode_number": 1,
            "sections": [
                {
                    "section_id": "section_1",
                    "question_moves": [
                        {
                            "question_id": "who_is_governing",
                            "action": "open",
                            "text": "Who is actually governing?",
                        }
                    ],
                    "memory_thread_moves": [
                        {
                            "thread_id": "hidden_ministries",
                            "action": "open",
                            "label": "The hidden ministries are already governing behind the cabinet.",
                        }
                    ],
                    "host_mystery_moves": [
                        {
                            "mystery_id": "army_nonintervention",
                            "action": "open",
                            "text": "Why didn't the army intervene?",
                        },
                        {
                            "mystery_id": "army_nonintervention",
                            "action": "advance",
                            "text": "army silence",
                        },
                    ],
                    "host_assumption_moves": [
                        {
                            "assumption_id": "one_bad_ruler_story",
                            "action": "introduce",
                            "statement": "This might just be a story about one ruler losing support.",
                        },
                        {
                            "assumption_id": "one_bad_ruler_story",
                            "action": "weaken",
                            "statement": "one bad ruler story",
                        },
                    ],
                    "host_theory_moves": [
                        {
                            "theory_id": "brittle_institutions",
                            "action": "propose",
                            "statement": "The system may have been hollow before it visibly fell.",
                        }
                    ],
                }
            ],
        },
    )

    result = agent.run(payload)

    assert result.episode_number == 1
    assert result.state_post.next_episode_number == 2
    assert result.state_post.listener.known_explanation_item_ids == ["velayat_faqih"]
    assert result.state_post.listener.known_actor_ids == ["bazargan"]
    assert result.state_post.listener.questions[0].question_id == "who_is_governing"
    assert result.state_post.listener.memory_threads[0].thread_id == "hidden_ministries"
    assert (
        result.state_post.listener.carry_forward_memory[0].label
        == "The state is weaker than it looks."
    )
    assert result.state_post.host.mysteries[0].mystery_id == "army_nonintervention"
    assert result.state_post.host.assumptions[0].status == "weakened"
    assert result.state_post.host.working_theories[0].theory_id == "brittle_institutions"
    assert result.state_post.host.recent_revisions[-2:] == [
        "army silence",
        "one bad ruler story",
    ]


def test_legacy_carry_forward_memory_strings_coerce_to_structured_items() -> None:
    state = NarrativeState.model_validate(
        {
            "project_id": "proj-1",
            "next_episode_number": 2,
            "listener": {
                "carry_forward_memory": ["The parallel ministries are still there."],
            },
            "host": {},
        }
    )

    item = state.listener.carry_forward_memory[0]
    assert isinstance(item, ContinuityCarryItem)
    assert item.label == "The parallel ministries are still there."
    assert item.kind == "takeaway"
    assert item.desired_surface == "body"
    assert item.recommended_action == "keep_live"


def test_continuity_contract_builds_recap_items_from_state() -> None:
    state = NarrativeState.model_validate(
        {
            "project_id": "proj-1",
            "next_episode_number": 2,
            "listener": {
                "carry_forward_memory": [
                    {
                        "item_id": "carry_1",
                        "label": "Earlier pressure still matters.",
                        "kind": "takeaway",
                        "source_episode_number": 1,
                        "priority": "high",
                        "desired_surface": "recap",
                        "recommended_action": "remind",
                    }
                ],
                "last_episode_takeaway": "Earlier pressure still matters.",
            },
            "host": {},
        }
    )

    contract = _build_continuity_contract(
        narrative_state=state,
        episode_number=2,
        phase="pre",
    )

    assert contract["recap_items"][0]["item_id"] == "carry_1"
    assert contract["must_surface_early"][0]["desired_surface"] == "recap"


def test_continuity_recap_validator_requires_recap_for_later_episode() -> None:
    plan = type(
        "PlanStub",
        (),
        {
            "framing": type(
                "FramingStub",
                (),
                {"recap": None},
            )()
        },
    )()

    with pytest.raises(ComplianceViolationError):
        _validate_continuity_recap_requirement(
            episode_number=2,
            plan=plan,
            continuity_contract_pre={
                "recap_items": [
                    {
                        "item_id": "carry_1",
                        "label": "Earlier pressure still matters.",
                    }
                ]
            },
        )


def test_continuity_diagnostics_track_realized_and_missed_items() -> None:
    diagnostics = _build_continuity_realization_diagnostics(
        episode_number=2,
        stage="script",
        framing={
            "recap": "Earlier pressure still matters.",
            "opening_question": "What happens next?",
        },
        ordered_sections=[
            {
                "section_id": "section_1",
                "text": "Earlier pressure still matters in the room.",
            },
            {"section_id": "section_2", "text": "A clean ending that drops the thread."},
        ],
        continuity_contract_pre={
            "recap_items": [
                {"item_id": "carry_1", "label": "Earlier pressure still matters."}
            ],
            "must_surface_early": [
                {"item_id": "carry_1", "label": "Earlier pressure still matters."}
            ],
        },
        continuity_contract_post={
            "must_leave_live": [
                {"item_id": "carry_2", "label": "Unfinished pressure survives."}
            ]
        },
    )

    assert "carry_1" in diagnostics["realized_item_ids"]
    assert "carry_2" in diagnostics["missed_item_ids"]
    assert diagnostics["targeted_feedback"][0]["item_id"] == "carry_2"
