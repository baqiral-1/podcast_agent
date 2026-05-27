from __future__ import annotations

import pytest

from _section_progression_helpers import make_section_progression
from podcast_agent.pipeline.orchestrator import (
    _build_window_actor_metadata,
    _build_writer_scene_brief,
    _normalize_writing_section_outputs,
    _validate_actor_explanation_scene_links,
)
from podcast_agent.schemas.models import (
    ActorExplanationPlan,
    ActorMetadata,
    ActorProfile,
    ArchitectureSection,
    EpisodeArchitecture,
    EpisodePlanDraft,
    EpisodeSpine,
    FramingBlock,
    HostMoveCue,
    HostMovesByPhase,
    NarrativeStrategy,
    SceneCardDraft,
    StrategyEpisode,
)
from podcast_agent.utils.actor_metadata import clean_narrative_strategy_actor_links


def _host_moves() -> HostMovesByPhase:
    return HostMovesByPhase(
        open=[HostMoveCue(move_type="orient", note="Set the footing.")]
    )


def _section(*, actor_id: str = "mossadeq", stage: str = "introduce") -> ArchitectureSection:
    return ArchitectureSection(
        section_id="section_1",
        purpose="opening",
        approx_runtime_minutes=8.0,
        primitive_ids=["p_1"],
        section_anchor="The cabinet chamber.",
        must_stage_beats=["Mossadeq enters office.", "Oil nationalization becomes visible."],
        listener_tension="Can the government keep control of the crisis it has opened?",
        section_turn="The political fight now has a single public face.",
        transition_logic="Move from institutional pressure to a named political actor.",
        section_progression=make_section_progression("setup", label="section_1"),
        actor_explanations=[
            ActorExplanationPlan(
                actor_id=actor_id,
                stage=stage,
                background_depth="appositive",
                role_label="prime minister and constitutional nationalist",
                source_primitive_ids=["p_1"],
                source_passage_ids=["passage_1"],
                intro_facts=["Mossadeq is prime minister."],
                why_now="This is the listener's first concrete encounter with him in the crisis.",
                preferred_plain_gloss="the nationalist prime minister driving oil nationalization",
            )
        ],
    )


def _architecture_with_actor_explanation(
    *, actor_id: str = "mossadeq", stage: str = "introduce"
) -> EpisodeArchitecture:
    sections = [
        _section(actor_id=actor_id, stage=stage),
        ArchitectureSection(
            section_id="section_2",
            purpose="setup",
            approx_runtime_minutes=8.0,
            primitive_ids=["p_2"],
            section_anchor="A memo circulates.",
            must_stage_beats=["Pressure spreads through the ministries.", "The court starts to react."],
            listener_tension="Can the old order absorb the challenge?",
            section_turn="The state begins to harden.",
            transition_logic="Move from public naming to institutional reaction.",
            section_progression=make_section_progression("advance", label="section_2"),
        ),
        ArchitectureSection(
            section_id="section_3",
            purpose="mechanism",
            approx_runtime_minutes=8.0,
            primitive_ids=["p_3"],
            section_anchor="Oil revenue figures.",
            must_stage_beats=["The money flow becomes visible.", "Foreign leverage tightens."],
            listener_tension="How does the pressure actually work?",
            section_turn="The mechanism becomes audible.",
            transition_logic="Show how the pressure moves through the system.",
            section_progression=make_section_progression("advance", label="section_3"),
        ),
        ArchitectureSection(
            section_id="section_4",
            purpose="turn",
            approx_runtime_minutes=8.0,
            primitive_ids=["p_4"],
            section_anchor="A showdown in the street.",
            must_stage_beats=["The crowd turns decisive.", "The balance shifts."],
            listener_tension="Can the confrontation stay contained?",
            section_turn="The confrontation breaks the old balance.",
            transition_logic="Escalate toward the visible turn.",
            section_progression=make_section_progression("advance", label="section_4"),
        ),
        ArchitectureSection(
            section_id="section_5",
            purpose="collapse",
            approx_runtime_minutes=8.0,
            primitive_ids=["p_5"],
            section_anchor="A shuttered office.",
            must_stage_beats=["The government loses room to maneuver.", "The cost becomes immediate."],
            listener_tension="What survives the break?",
            section_turn="The consequences stop looking temporary.",
            transition_logic="Show the immediate fallout of the turn.",
            section_progression=make_section_progression("answer", label="section_5"),
        ),
        ArchitectureSection(
            section_id="section_6",
            purpose="closing",
            approx_runtime_minutes=1.5,
            primitive_ids=["p_6"],
            section_anchor="An emptied chamber.",
            must_stage_beats=["The residue settles.", "The unresolved burden remains."],
            listener_tension="What remains unresolved?",
            section_turn="The answer lands with residue.",
            transition_logic="Land the closing residue without reopening the argument.",
            section_progression=make_section_progression("close", label="section_6"),
        ),
    ]
    return EpisodeArchitecture(
        episode_number=1,
        major_turn_section_id="section_4",
        sections=sections,
    )


def _scene(*, actor_id: str | None, explanation_stage: str | None = None) -> SceneCardDraft:
    actor_payload = {
        "name": "Mohammad Mossadeq",
        "actor_id": actor_id,
        "presence": "primary",
        "explanation_stage": explanation_stage,
        "background_depth": "appositive",
        "preferred_plain_gloss": "the nationalist prime minister driving oil nationalization",
    }
    if explanation_stage == "introduce":
        actor_payload.update(
            {
                "role_label": "prime minister and constitutional nationalist",
                "source_primitive_ids": ["p_1"],
                "source_passage_ids": ["passage_1"],
                "intro_facts": ["Mossadeq is prime minister."],
                "why_now": "This is the listener's first concrete encounter with him in the crisis.",
            }
        )
    return SceneCardDraft(
        scene_id="scene_1",
        section_id="section_1",
        title="Mossadeq arrives",
        scene_role="actor_setup",
        scene_function="scene",
        beat_change="A single political actor now carries the crisis in public.",
        must_land_facts=["Mossadeq is prime minister.", "Oil nationalization is the live issue."],
        actors=[
            actor_payload
        ],
        passage_ids=["passage_1"],
        host_moves=_host_moves().model_dump(mode="json"),
        estimated_duration_seconds=75,
    )


def test_clean_narrative_strategy_actor_links_fills_actor_gloss_and_filters_registry_refs() -> None:
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="mossadeq",
                display_name="Mohammad Mossadeq",
                actor_type="person",
                description="Prime minister who nationalized Iran's oil and became the face of the crisis.",
            )
        ],
    )
    strategy = NarrativeStrategy.model_validate(
        {
            "strategy_type": "chronological",
            "justification": "Chronology carries the causality.",
            "series_arc": "One crisis turns into another.",
            "series_actor_explanation_registry": [
                {
                    "actor_id": "mossadeq",
                    "introduction_episode_number": 1,
                    "first_background_depth": "appositive",
                    "preferred_plain_gloss": "",
                    "later_episode_policy": "brief_reminder",
                },
                {
                    "actor_id": "unknown_actor",
                    "introduction_episode_number": 1,
                    "first_background_depth": "appositive",
                    "preferred_plain_gloss": "unknown",
                    "later_episode_policy": "brief_reminder",
                },
            ],
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode 1",
                    "arc_summary": "Arc",
                    "episode_spine": {
                        "listener_problem": "Why does the crisis take this shape?",
                        "episode_answer": "A claim",
                        "core_primitive_ids": ["core_1", "core_2", "core_3", "core_4", "core_5"],
                        "support_primitive_roles": {
                            "support_1": "mechanism",
                            "support_2": "mechanism",
                        },
                    },
                    "authorial_contract": {
                        "introduce_actor_ids": ["mossadeq", "unknown_actor"],
                        "remind_actor_ids": [],
                    },
                }
            ],
        }
    )

    cleaned, metrics = clean_narrative_strategy_actor_links(strategy, actor_metadata)

    assert [item.actor_id for item in cleaned.series_actor_explanation_registry] == [
        "mossadeq"
    ]
    assert (
        cleaned.series_actor_explanation_registry[0].preferred_plain_gloss
        == "Prime minister who nationalized Iran's oil and became the face of the crisis"
    )
    assert cleaned.episodes[0].authorial_contract.introduce_actor_ids == ["mossadeq"]
    assert metrics["fallback_actor_gloss_count"] == 1
    assert metrics["unknown_actor_ids"] == 1
    assert metrics["unknown_actor_registry_refs"] == 1


def test_build_writer_scene_brief_preserves_actor_explanation_metadata() -> None:
    brief = _build_writer_scene_brief(
        {
            "scene_id": "scene_1",
            "section_id": "section_1",
            "title": "Mossadeq arrives",
            "scene_role": "actor_setup",
            "scene_function": "scene",
            "beat_change": "The public face of the crisis becomes clear.",
            "must_land_facts": ["He is prime minister."],
            "actors": [
                {
                    "name": "Mohammad Mossadeq",
                    "actor_id": "mossadeq",
                    "presence": "primary",
                    "explanation_stage": "introduce",
                    "background_depth": "appositive",
                    "role_label": "prime minister and constitutional nationalist",
                    "source_primitive_ids": ["p_1"],
                    "source_passage_ids": ["passage_1"],
                    "intro_facts": ["Mossadeq is prime minister."],
                    "why_now": "This is the listener's first concrete encounter with him in the crisis.",
                    "preferred_plain_gloss": "the nationalist prime minister driving oil nationalization",
                }
            ],
        }
    )

    assert brief["actors"] == [
        {
            "name": "Mohammad Mossadeq",
            "actor_id": "mossadeq",
            "affiliation": None,
            "presence": "primary",
            "explanation_stage": "introduce",
            "background_depth": "appositive",
            "role_label": "prime minister and constitutional nationalist",
            "source_primitive_ids": ["p_1"],
            "source_passage_ids": ["passage_1"],
            "intro_facts": ["Mossadeq is prime minister."],
            "why_now": "This is the listener's first concrete encounter with him in the crisis.",
            "preferred_plain_gloss": "the nationalist prime minister driving oil nationalization",
        }
    ]


def test_build_window_actor_metadata_includes_section_actor_explanations() -> None:
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="mossadeq",
                display_name="Mohammad Mossadeq",
                actor_type="person",
            ),
            ActorProfile(
                actor_id="shah",
                display_name="Mohammad Reza Shah",
                actor_type="person",
            ),
        ],
    )
    strategy_episode = StrategyEpisode(
        episode_number=1,
        title="Episode 1",
        arc_summary="Arc",
        episode_spine=EpisodeSpine(
            listener_problem="Question?",
            episode_answer="Claim",
            core_primitive_ids=["core_1", "core_2", "core_3", "core_4", "core_5"],
            support_primitive_roles={"support_1": "mechanism", "support_2": "texture"},
        ),
    )
    architecture = _architecture_with_actor_explanation(actor_id="mossadeq")

    payload = _build_window_actor_metadata(
        window_scene_cards=[_scene(actor_id="shah", explanation_stage=None)],
        strategy_episode=strategy_episode,
        architecture=architecture,
        actor_metadata=actor_metadata,
    )

    assert {actor["actor_id"] for actor in payload["actors"]} == {"mossadeq", "shah"}


def test_validate_actor_explanation_scene_links_requires_matching_scene_actor() -> None:
    architecture = _architecture_with_actor_explanation()
    plan = EpisodePlanDraft(
        episode_number=1,
        framing=FramingBlock(
            opening_image="A crowd waits outside.",
            threat_or_unresolved_action="No one knows whether the government can hold.",
            opening_question="Who now carries the crisis?",
            handoff_scene_card_id="scene_1",
        ),
        scene_cards=[_scene(actor_id="mossadeq", explanation_stage=None)],
    )

    warnings = _validate_actor_explanation_scene_links(
        architecture=architecture,
        plan=plan,
    )

    assert warnings == [
        "missing_actor_explanation_scene_links:section_1:mossadeq:introduce"
    ]

    _validate_actor_explanation_scene_links(
        architecture=architecture,
        plan=plan.model_copy(
            update={"scene_cards": [_scene(actor_id="mossadeq", explanation_stage="introduce")]}
        ),
    )


def test_validate_actor_explanation_scene_links_rejects_late_introduction() -> None:
    architecture = _architecture_with_actor_explanation()
    plan = EpisodePlanDraft(
        episode_number=1,
        framing=FramingBlock(
            opening_image="A crowd waits outside.",
            threat_or_unresolved_action="No one knows whether the government can hold.",
            opening_question="Who now carries the crisis?",
            handoff_scene_card_id="scene_early",
        ),
        scene_cards=[
            _scene(actor_id="mossadeq", explanation_stage=None).model_copy(
                update={"scene_id": "scene_early"}
            ),
            _scene(actor_id="mossadeq", explanation_stage="introduce").model_copy(
                update={"scene_id": "scene_intro"}
            ),
        ],
    )

    warnings = _validate_actor_explanation_scene_links(
        architecture=architecture,
        plan=plan,
    )

    assert warnings == [
        "late_actor_introduction_scene_links:mossadeq:scene_early"
    ]


def test_normalize_writing_section_outputs_allows_missing_actor_intro_realization() -> None:
    architecture = _architecture_with_actor_explanation()
    scene = _scene(actor_id="mossadeq", explanation_stage="introduce")

    normalized = _normalize_writing_section_outputs(
        result=type(
            "Payload",
            (),
            {
                "prose_sections": [
                    type(
                        "SectionOutput",
                        (),
                        {
                            "section_id": "section_1",
                            "scene_card_ids": ["scene_1"],
                            "movement_goal": "discover",
                            "text": "Mohammad Mossadeq enters the chamber.",
                            "source_book_ids": [],
                            "citations": [],
                            "actor_explanation_realizations": [],
                        },
                    )()
                ]
            },
        )(),
        architecture=architecture,
        scene_cards=[scene],
        episode_number=1,
        skip_grounding=False,
    )

    assert normalized[0]["actor_explanation_realizations"] == []


def test_normalize_writing_section_outputs_allows_paraphrased_actor_intro_realization() -> None:
    architecture = _architecture_with_actor_explanation()
    scene = _scene(actor_id="mossadeq", explanation_stage="introduce")

    normalized = _normalize_writing_section_outputs(
        result=type(
            "Payload",
            (),
            {
                "prose_sections": [
                    type(
                        "SectionOutput",
                        (),
                        {
                            "section_id": "section_1",
                            "scene_card_ids": ["scene_1"],
                            "movement_goal": "discover",
                            "text": "Mohammad Mossadeq enters the chamber.",
                            "source_book_ids": [],
                            "citations": [],
                            "actor_explanation_realizations": [
                                {
                                    "actor_id": "mossadeq",
                                    "scene_card_id": "scene_1",
                                    "stage": "introduce",
                                    "text_span": "Mohammad Mossadeq enters the chamber.",
                                    "source_passage_ids": ["passage_2"],
                                    "realized_facts": ["Mossadeq arrives as prime minister."],
                                }
                            ],
                        },
                    )()
                ]
            },
        )(),
        architecture=architecture,
        scene_cards=[scene],
        episode_number=1,
        skip_grounding=False,
    )

    assert normalized[0]["actor_explanation_realizations"][0].source_passage_ids == [
        "passage_2"
    ]
