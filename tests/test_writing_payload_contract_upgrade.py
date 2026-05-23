from __future__ import annotations

import pytest

from podcast_agent.langchain.runnables import ComplianceViolationError
from podcast_agent.pipeline.orchestrator import (
    _build_style_audit_sections_payload,
    _build_writer_scene_brief,
    _compute_scene_word_count_bands,
    _validate_plan_transition,
)
from podcast_agent.schemas.models import (
    ArchitectureSection,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodePlanDraft,
    EpisodeScript,
    FramingBlock,
    ProseSection,
    SceneCard,
    SceneCardDraft,
    StrategyEpisode,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A paper changes hands.",
        threat_or_unresolved_action="Nobody knows what the signature will unlock.",
        opening_question="What does this decision make possible?",
        handoff_scene_card_id="scene_1",
    )


def _strategy_episode() -> StrategyEpisode:
    return StrategyEpisode.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "actor_arc_directives": [],
            "promised_beats": [],
                "episode_spine": {
                    "listener_problem": "What changes?",
                    "episode_answer": "The system changes by force.",
                    "pressure_line": "Pressure keeps narrowing the options.",
                    "core_primitive_ids": ["primitive_1", "primitive_2"],
                    "support_primitive_roles": {
                        "primitive_3": "mechanism",
                        "primitive_4": "stakes",
                    },
                    "recall_primitive_ids": [],
                },
            }
        )


def _architecture() -> EpisodeArchitecture:
    section = ArchitectureSection.model_validate(
        {
            "section_id": "section_1",
            "purpose": "turn",
            "approx_runtime_minutes": 8.0,
            "primitive_ids": ["primitive_1"],
            "section_anchor": "A clerk opens the envelope.",
            "must_stage_beats": [
                "The order becomes public.",
                "The pressure changes shape.",
            ],
            "closure_mode": "turn",
            "authorial_passages": [
                {
                    "authorial_passage_id": "ap_1",
                    "passage_id": "p1",
                    "mode": "quote_then_gloss",
                    "placement": "mid",
                    "claim": "The order is doing more than it first appears to do.",
                    "source_primitive_ids": ["primitive_1"],
                }
            ],
        }
    )
    return EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        answer_section_id="section_1",
        residue_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        promised_beat_decisions=[],
        sections=[section],
        architecture_notes=[],
    )


def _scene_card(
    scene_id: str,
    scene_job: str,
    *,
    authorial_passage_ids: list[str] | None = None,
    word_count_priority: str = "default",
    withhold_until: dict | None = None,
) -> dict:
    return {
        "scene_id": scene_id,
        "section_id": "section_1",
        "title": scene_id,
        "scene_role": "action",
        "scene_job": scene_job,
        "beat_change": "The pressure becomes visible in concrete form.",
        "must_land_facts": {
            "required": ["The order is real.", "The room reacts immediately."],
            "strongly_preferred": ["The witnesses know the signature matters."],
            "if_room": ["The corridor outside goes quiet."],
        },
        "entry_image": "A paper changes hands.",
        "observable_detail": "Nobody speaks for a beat.",
        "primitive_ids": ["primitive_1"],
        "passage_ids": ["p1"],
        "authorial_passage_ids": list(authorial_passage_ids or []),
        "word_count_priority": word_count_priority,
        "withhold_until": withhold_until,
        "host_moves": {
            "open": [{"move_type": "orient", "target": "pressure first"}],
            "pivot": [],
            "close": [],
        },
        "estimated_duration_seconds": 120,
    }


def test_scene_card_validates_tiered_facts_and_forward_withholding() -> None:
    card = SceneCardDraft.model_validate(
        _scene_card(
            "scene_1",
            "build",
            withhold_until={
                "subject": "who signed the order",
                "reveal_scene_id": "scene_2",
                "reveal_phase": "pivot",
                "surrogate_label": "the unseen signature",
            },
        )
    )

    assert card.must_land_facts.required == [
        "The order is real.",
        "The room reacts immediately.",
    ]
    assert card.must_land_facts.ordered_facts()[-1] == "The corridor outside goes quiet."
    assert card.word_count_priority == "default"

    with pytest.raises(Exception, match="must not point to an earlier scene card"):
        EpisodePlanDraft.model_validate(
            {
                "episode_number": 1,
                "framing": _framing().model_dump(mode="json"),
                "scene_cards": [
                    _scene_card("scene_1", "build"),
                    _scene_card(
                        "scene_2",
                        "answer",
                        withhold_until={
                            "subject": "the signature",
                            "reveal_scene_id": "scene_1",
                            "reveal_phase": "open",
                        },
                    ),
                    _scene_card("scene_3", "residue"),
                ],
                "answer_scene_card_id": "scene_2",
                "residue_scene_card_id": "scene_3",
            }
        )


def test_scene_card_normalizes_withhold_until_acting_subject_alias() -> None:
    card = SceneCardDraft.model_validate(
        _scene_card(
            "scene_1",
            "build",
            withhold_until={
                "acting_subject": "who signed the order",
                "reveal_scene_id": "scene_2",
                "reveal_phase": "pivot",
                "surrogate_label": "the unseen signature",
            },
        )
    )

    assert card.withhold_until is not None
    assert card.withhold_until.subject == "who signed the order"


def test_validate_plan_transition_requires_scene_pinned_authorial_passages() -> None:
    architecture = _architecture()
    strategy_episode = _strategy_episode()
    unassigned_plan = EpisodePlanDraft.model_validate(
        {
            "episode_number": 1,
            "framing": _framing().model_dump(mode="json"),
                "scene_cards": [
                    _scene_card("scene_1", "build"),
                    _scene_card("scene_2", "answer"),
                    _scene_card("scene_3", "residue"),
                    _scene_card("scene_4", "close"),
                ],
                "answer_scene_card_id": "scene_2",
                "residue_scene_card_id": "scene_3",
        }
    )

    with pytest.raises(
        ComplianceViolationError, match="authorial passages unassigned"
    ):
        _validate_plan_transition(
            strategy_episode=strategy_episode,
            architecture=architecture,
            plan=unassigned_plan,
        )

    assigned_plan = EpisodePlanDraft.model_validate(
        {
            "episode_number": 1,
            "framing": _framing().model_dump(mode="json"),
                "scene_cards": [
                    _scene_card("scene_1", "build", authorial_passage_ids=["ap_1"]),
                    _scene_card("scene_2", "answer"),
                    _scene_card("scene_3", "residue"),
                    _scene_card("scene_4", "close"),
                ],
                "answer_scene_card_id": "scene_2",
                "residue_scene_card_id": "scene_3",
        }
    )

    validated = _validate_plan_transition(
        strategy_episode=strategy_episode,
        architecture=architecture,
        plan=assigned_plan,
    )
    assert validated.scene_cards[0].authorial_passage_ids == ["ap_1"]


def test_build_writer_scene_brief_preserves_new_scene_contract() -> None:
    scene_payload = _scene_card(
        "scene_1",
        "build",
        authorial_passage_ids=["ap_1"],
        word_count_priority="tight",
        withhold_until={
            "subject": "who signed the order",
            "reveal_phase": "pivot",
            "reveal_scene_id": "scene_2",
            "surrogate_label": "the unseen signature",
        },
    )
    scene_payload["authorial_passages"] = [
        {
            "authorial_passage_id": "ap_1",
            "passage_id": "p1",
            "mode": "quote_then_gloss",
            "placement": "mid",
            "claim": "The order changes more than the room.",
            "source_primitive_ids": ["primitive_1"],
            "scene_id": "scene_1",
        }
    ]
    scene_payload["target_word_count_lower"] = 120
    scene_payload["target_word_count_higher"] = 140

    brief = _build_writer_scene_brief(scene_payload)

    assert brief["must_land_facts"]["required"][0] == "The order is real."
    assert brief["why_now"] == "The pressure becomes visible in concrete form."
    assert brief["authorial_passage_ids"] == ["ap_1"]
    assert brief["authorial_passages"][0]["scene_id"] == "scene_1"
    assert brief["word_count_priority"] == "tight"
    assert brief["withhold_until"]["reveal_scene_id"] == "scene_2"


def test_compute_scene_word_count_bands_uses_default_and_tight_ranges() -> None:
    default_scene = SceneCard.model_validate(_scene_card("scene_1", "build"))
    tight_scene = SceneCard.model_validate(
        _scene_card("scene_2", "answer", word_count_priority="tight")
    )

    lower, higher = _compute_scene_word_count_bands(
        [default_scene, tight_scene],
        episode_target_word_count=1000,
    )

    assert lower["scene_1"] < lower["scene_2"]
    assert higher["scene_1"] > higher["scene_2"]


def test_style_audit_payload_enriches_authorial_passages_with_scene_id() -> None:
    architecture = _architecture()
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                _scene_card("scene_1", "build", authorial_passage_ids=["ap_1"]),
                _scene_card("scene_2", "answer"),
                _scene_card("scene_3", "residue"),
            ],
            "answer_scene_card_id": "scene_2",
            "residue_scene_card_id": "scene_3",
            "target_word_count": 600,
        }
    )
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1", "scene_2", "scene_3"],
                movement_goal="pressure rises",
                text="The section text.",
            )
        ],
    )

    payload = _build_style_audit_sections_payload(
        script=script,
        architecture=architecture,
        plan=plan,
    )

    assert payload[0]["authorial_passages"][0]["authorial_passage_id"] == "ap_1"
    assert payload[0]["authorial_passages"][0]["scene_id"] == "scene_1"
