from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    CharacterEnginePrimitive,
    CharacterEnginePrimitiveDelta,
    CoalitionFaultLinePrimitiveDelta,
    DecisionEnrichmentArtifact,
    DecisionPrimitive,
    DecisionPrimitiveDelta,
    HumanCostEnrichmentArtifact,
    HumanCostPrimitive,
    HumanCostPrimitiveDelta,
    IronyReversalPrimitive,
    IronyReversalPrimitiveDelta,
    MoralTrapPrimitive,
    MoralTrapPrimitiveDelta,
    SetPieceScenePrimitive,
    SetPieceScenePrimitiveDelta,
    CoalitionFaultLinePrimitive,
    SystemsOperatingLogicPrimitiveDelta,
)


def test_human_cost_delta_allows_empty_actor_ids() -> None:
    delta = HumanCostPrimitiveDelta.model_validate(
        {
            "id": "hc_qandahar_60000_deaths",
            "family": "human_costs",
            "actor_ids": [],
            "affected_group": "campaign laborers and camp families",
            "cost_type": "mass death",
            "lived_consequence": "The campaign consumes lives at a scale witnesses cannot ignore.",
            "visibility": "widely seen but easily folded into aggregate loss language",
        }
    )

    assert delta.actor_ids == []


def test_human_cost_primitive_allows_empty_actor_ids() -> None:
    primitive = HumanCostPrimitive.model_validate(
        {
            "id": "hc_delhi_expulsion_1857",
            "family": "human_costs",
            "title": "Delhi emptied after the storming",
            "summary": "The expulsion turns imperial collapse into street-level dispossession.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "affected_group": "Delhi households",
            "cost_type": "expulsion",
            "lived_consequence": "Families lose homes, income, and local protection all at once.",
            "visibility": "plain in the city itself, easier to flatten at imperial scale",
        }
    )

    assert primitive.actor_ids == []


def test_decision_delta_allows_empty_actor_ids() -> None:
    delta = DecisionPrimitiveDelta.model_validate(
        {
            "id": "dn_1",
            "family": "decisions_and_nondecisions",
            "actor_ids": [],
            "decision_question": "Should the court move now?",
            "decision_mode": "decision",
            "options_considered": ["advance", "delay"],
            "immediate_consequence": "The move forces the next confrontation.",
        }
    )

    assert delta.actor_ids == []


def test_decision_primitive_allows_empty_actor_ids() -> None:
    primitive = DecisionPrimitive.model_validate(
        {
            "id": "dn_1",
            "family": "decisions_and_nondecisions",
            "title": "The court stalls",
            "summary": "A consequential refusal to commit shapes what happens next.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "decision_question": "Should the court move now?",
            "decision_mode": "decision",
            "options_considered": ["advance", "delay"],
            "immediate_consequence": "The move forces the next confrontation.",
        }
    )

    assert primitive.actor_ids == []


def test_decision_delta_requires_decision_question() -> None:
    with pytest.raises(ValidationError, match="decision_question"):
        DecisionPrimitiveDelta.model_validate(
            {
                "id": "dn_1",
                "family": "decisions_and_nondecisions",
                "actor_ids": ["actor_1"],
                "decision_mode": "decision",
                "options_considered": ["advance", "delay"],
                "immediate_consequence": "The move forces the next confrontation.",
            }
        )


def test_human_cost_artifact_accepts_mughal_style_empty_actor_ids() -> None:
    artifact = HumanCostEnrichmentArtifact.model_validate(
        {
            "project_id": "mughal_v27",
            "family": "human_costs",
            "enriched_primitives": [
                {
                    "id": "hc_bengal_carcase_dow",
                    "family": "human_costs",
                    "actor_ids": [],
                    "affected_group": "Bengal civilians in famine conditions",
                    "cost_type": "starvation",
                    "lived_consequence": "Bodies and hunger make fiscal breakdown physically unavoidable.",
                    "visibility": "visible on the ground, rhetorically softened at a distance",
                }
            ],
        }
    )

    assert artifact.enriched_primitives[0].actor_ids == []


def test_set_piece_scene_delta_allows_empty_actor_ids() -> None:
    delta = SetPieceScenePrimitiveDelta.model_validate(
        {
            "id": "sps_1",
            "family": "set_piece_scenes",
            "actor_ids": [],
            "scene_anchor": "Gunfire breaks the standoff.",
            "scene_outcome": "The crowd scatters in panic.",
            "location": "the main square",
        }
    )

    assert delta.actor_ids == []


def test_set_piece_scene_primitive_allows_empty_actor_ids() -> None:
    primitive = SetPieceScenePrimitive.model_validate(
        {
            "id": "sps_1",
            "family": "set_piece_scenes",
            "title": "The square erupts",
            "summary": "A public confrontation turns into a rout.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "scene_anchor": "Gunfire breaks the standoff.",
            "scene_outcome": "The crowd scatters in panic.",
            "location": "the main square",
        }
    )

    assert primitive.actor_ids == []


def test_character_engine_delta_allows_null_actor_id() -> None:
    delta = CharacterEnginePrimitiveDelta.model_validate(
        {
            "id": "ce_1",
            "family": "character_engines",
            "actor_id": None,
            "goal": "Preserve influence at court.",
            "fear": "Public humiliation by rivals.",
            "constraint": "He lacks direct command of the army.",
            "stakes": "If he fails, his faction loses access to power.",
        }
    )

    assert delta.actor_id is None


def test_character_engine_primitive_allows_null_actor_id() -> None:
    primitive = CharacterEnginePrimitive.model_validate(
        {
            "id": "ce_1",
            "family": "character_engines",
            "title": "Court pressure without one clear owner",
            "summary": "The pressure profile is legible even when one actor is not.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_id": None,
            "goal": "Preserve influence at court.",
            "fear": "Public humiliation by rivals.",
            "constraint": "He lacks direct command of the army.",
            "stakes": "If he fails, his faction loses access to power.",
        }
    )

    assert primitive.actor_id is None


def test_coalition_delta_allows_empty_actor_ids() -> None:
    delta = CoalitionFaultLinePrimitiveDelta.model_validate(
        {
            "id": "cf_1",
            "family": "coalitions_and_fault_lines",
            "actor_ids": [],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "coalition": "A narrow alliance.",
            "shared_interest": "Each side needs short-term support.",
            "fault_line": "Their long-term political aims diverge.",
            "stress_point": "The alliance weakens once the emergency fades.",
        }
    )

    assert delta.actor_ids == []


def test_coalition_primitive_allows_empty_actor_ids() -> None:
    primitive = CoalitionFaultLinePrimitive.model_validate(
        {
            "id": "cf_1",
            "family": "coalitions_and_fault_lines",
            "title": "An uneasy bloc holds for now",
            "summary": "Shared pressure masks deeper divergence.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "coalition": "A narrow alliance.",
            "shared_interest": "Each side needs short-term support.",
            "fault_line": "Their long-term political aims diverge.",
            "stress_point": "The alliance weakens once the emergency fades.",
        }
    )

    assert primitive.actor_ids == []


def test_moral_trap_delta_allows_empty_actor_ids() -> None:
    delta = MoralTrapPrimitiveDelta.model_validate(
        {
            "id": "mt_1",
            "family": "moral_traps",
            "actor_ids": [],
            "competing_obligations": ["Protect civilians", "Preserve the alliance"],
            "compromised_options": ["Call in the strike", "Stand down and abandon the partner"],
            "no_clean_exit_reason": "Any live move sacrifices something the actor is bound to protect.",
        }
    )

    assert delta.actor_ids == []


def test_moral_trap_primitive_allows_empty_actor_ids() -> None:
    primitive = MoralTrapPrimitive.model_validate(
        {
            "id": "mt_1",
            "family": "moral_traps",
            "title": "No clean way out",
            "summary": "Every available move carries a serious violation.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "competing_obligations": ["Protect civilians", "Preserve the alliance"],
            "compromised_options": ["Call in the strike", "Stand down and abandon the partner"],
            "no_clean_exit_reason": "Any live move sacrifices something the actor is bound to protect.",
        }
    )

    assert primitive.actor_ids == []


def test_irony_delta_allows_empty_actor_ids() -> None:
    delta = IronyReversalPrimitiveDelta.model_validate(
        {
            "id": "ir_1",
            "family": "ironies_and_reversals",
            "actor_ids": [],
            "expected_outcome": "The crackdown restores control.",
            "actual_outcome": "The crackdown radicalizes the opposition instead.",
            "reversal_driver": "The coercion creates the coalition it meant to crush.",
        }
    )

    assert delta.actor_ids == []


def test_irony_primitive_allows_empty_actor_ids() -> None:
    primitive = IronyReversalPrimitive.model_validate(
        {
            "id": "ir_1",
            "family": "ironies_and_reversals",
            "title": "The crackdown backfires",
            "summary": "The attempt to restore order produces the opposite effect.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "expected_outcome": "The crackdown restores control.",
            "actual_outcome": "The crackdown radicalizes the opposition instead.",
            "reversal_driver": "The coercion creates the coalition it meant to crush.",
        }
    )

    assert primitive.actor_ids == []


def test_decision_artifact_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="id_note_ignore"):
        DecisionEnrichmentArtifact.model_validate(
            {
                "project_id": "mughal_v27",
                "family": "decisions_and_nondecisions",
                "enriched_primitives": [
                    {
                        "id": "dn_shah_jahan_promote_dara_duplicate",
                        "family": "decisions_and_nondecisions",
                        "actor_ids": ["actor_1"],
                        "decision_question": "Should the emperor lock in one succession path?",
                        "decision_mode": "decision",
                        "options_considered": ["promote Dara", "rebalance succession"],
                        "immediate_consequence": "Court incentives harden around one heir.",
                        "id_note_ignore": "extra field should fail",
                    }
                ],
            }
        )


def test_coalition_delta_validates_enum_fields() -> None:
    delta = CoalitionFaultLinePrimitiveDelta.model_validate(
        {
            "id": "cf_1",
            "family": "coalitions_and_fault_lines",
            "actor_ids": ["actor_1"],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "coalition": "A narrow alliance.",
            "shared_interest": "Each side needs short-term support.",
            "fault_line": "Their long-term political aims diverge.",
            "stress_point": "The alliance weakens once the emergency fades.",
        }
    )

    assert delta.alignment_type.value == "tactical"
    assert delta.coalition_phase.value == "holding"


def test_systems_delta_requires_at_least_two_mechanism_steps() -> None:
    with pytest.raises(ValidationError, match="mechanism_steps"):
        SystemsOperatingLogicPrimitiveDelta.model_validate(
            {
                "id": "sys_1",
                "family": "systems_and_operating_logics",
                "system_name": "Court patronage",
                "mechanism": "Resources move through patronage channels.",
                "mechanism_steps": ["Orders move outward."],
                "inputs": ["orders"],
                "outputs": ["compliance"],
                "failure_mode": "The chain distorts under stress.",
            }
        )
