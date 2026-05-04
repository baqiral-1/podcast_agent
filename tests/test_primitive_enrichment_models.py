from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    CharacterEnginePrimitive,
    CharacterEnginePrimitiveDelta,
    CoalitionFaultLinePrimitive,
    CoalitionFaultLinePrimitiveDelta,
    ContestedExplanationPrimitive,
    ContestedExplanationPrimitiveDelta,
    DecisionEnrichmentArtifact,
    DecisionPrimitive,
    DecisionPrimitiveDelta,
    EpochalTurnPrimitive,
    HumanCostEnrichmentArtifact,
    HumanCostPrimitive,
    HumanCostPrimitiveDelta,
    IronyReversalPrimitive,
    IronyReversalPrimitiveDelta,
    MoralTrapPrimitive,
    MoralTrapPrimitiveDelta,
    SetPieceScenePrimitive,
    SetPieceScenePrimitiveDelta,
    SystemsOperatingLogicPrimitive,
    SystemsOperatingLogicPrimitiveDelta,
)


def _hooks() -> dict[str, str]:
    return {
        "concrete_detail": "A concrete detail lands.",
        "host_lens": "The pressure is visible.",
        "carry_forward": "The residue lingers.",
    }


def _long_phrase(prefix: str, count: int) -> str:
    return " ".join([prefix, *[f"word{i}" for i in range(count - 1)]])


def test_human_cost_delta_allows_empty_actor_ids() -> None:
    delta = HumanCostPrimitiveDelta.model_validate(
        {
            "id": "hc_qandahar_60000_deaths",
            "family": "human_costs",
            "actor_ids": [],
            "affected_group": "campaign laborers and camp families",
            "cost_type": "mass death",
            "concrete_marker": "Bodies stack by the road.",
            "lived_consequence": "The campaign consumes lives at a scale witnesses cannot ignore.",
            "who_saw_it": "widely seen but easily folded into aggregate loss language",
            "narration_hooks": _hooks(),
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
            "concrete_marker": "Doorways stand open and empty.",
            "lived_consequence": "Families lose homes, income, and local protection all at once.",
            "who_saw_it": "plain in the city itself, easier to flatten at imperial scale",
            "narration_hooks": _hooks(),
        }
    )

    assert primitive.actor_ids == []


def test_decision_delta_allows_empty_actor_ids() -> None:
    delta = DecisionPrimitiveDelta.model_validate(
        {
            "id": "dn_1",
            "family": "decisions_and_nondecisions",
            "actor_ids": [],
            "decision_trigger": "Fresh pressure forces a move.",
            "decision_question": "Should the court move now?",
            "decision_mode": "decision",
            "options_considered": ["advance", "delay"],
            "next_result": "The move forces the next confrontation.",
            "narration_hooks": _hooks(),
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
            "decision_trigger": "Fresh pressure forces a move.",
            "decision_question": "Should the court move now?",
            "decision_mode": "decision",
            "options_considered": ["advance", "delay"],
            "next_result": "The move forces the next confrontation.",
            "narration_hooks": _hooks(),
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
                "decision_trigger": "Fresh pressure forces a move.",
                "decision_mode": "decision",
                "options_considered": ["advance", "delay"],
                "next_result": "The move forces the next confrontation.",
                "narration_hooks": _hooks(),
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
                    "concrete_marker": "Corpses lie by the roadside.",
                    "lived_consequence": "Bodies and hunger make fiscal breakdown physically unavoidable.",
                    "who_saw_it": "visible on the ground, rhetorically softened at a distance",
                    "narration_hooks": _hooks(),
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
            "hinge_action": "The first volley scatters the line.",
            "scene_outcome": "The crowd scatters in panic.",
            "location": "the main square",
            "narration_hooks": _hooks(),
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
            "hinge_action": "The first volley scatters the line.",
            "scene_outcome": "The crowd scatters in panic.",
            "location": "the main square",
            "narration_hooks": _hooks(),
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
            "pressure_box": "He lacks direct command and fears public humiliation.",
            "risk_if_it_breaks": "If he fails, his faction loses access to power.",
            "tell": "He keeps doubling down in public.",
            "narration_hooks": _hooks(),
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
            "pressure_box": "He lacks direct command and fears public humiliation.",
            "risk_if_it_breaks": "If he fails, his faction loses access to power.",
            "tell": "He keeps doubling down in public.",
            "narration_hooks": _hooks(),
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
            "alignment_shape": "A narrow alliance.",
            "alignment_basis": "Each side needs short-term support.",
            "fracture_trigger": "The alliance weakens once the emergency fades.",
            "narration_hooks": _hooks(),
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
            "alignment_shape": "A narrow alliance.",
            "alignment_basis": "Each side needs short-term support.",
            "fracture_trigger": "The alliance weakens once the emergency fades.",
            "narration_hooks": _hooks(),
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
            "trap_structure": "Any live move sacrifices something the actor is bound to protect.",
            "narration_hooks": _hooks(),
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
            "trap_structure": "Any live move sacrifices something the actor is bound to protect.",
            "narration_hooks": _hooks(),
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
            "flip_cause": "The coercion creates the coalition it meant to crush.",
            "narration_hooks": _hooks(),
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
            "flip_cause": "The coercion creates the coalition it meant to crush.",
            "narration_hooks": _hooks(),
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
                        "decision_trigger": "Court pressure spikes.",
                        "decision_question": "Should the emperor lock in one succession path?",
                        "decision_mode": "decision",
                        "options_considered": ["promote Dara", "rebalance succession"],
                        "next_result": "Court incentives harden around one heir.",
                        "narration_hooks": _hooks(),
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
            "alignment_shape": "A narrow alliance.",
            "alignment_basis": "Each side needs short-term support.",
            "fracture_trigger": "The alliance weakens once the emergency fades.",
            "narration_hooks": _hooks(),
        }
    )

    assert delta.alignment_type.value == "tactical"
    assert delta.coalition_phase.value == "holding"


def test_systems_delta_requires_at_least_two_operating_chain_steps() -> None:
    with pytest.raises(ValidationError, match="operating_chain"):
        SystemsOperatingLogicPrimitiveDelta.model_validate(
            {
                "id": "sys_1",
                "family": "systems_and_operating_logics",
                "system_name": "Court patronage",
                "operating_chain": ["Orders move outward."],
                "inputs": ["orders"],
                "outputs": ["compliance"],
                "where_it_shows_up": "Orders are delivered face to face.",
                "failure_mode": "The chain distorts under stress.",
                "narration_hooks": _hooks(),
            }
        )


def test_decision_delta_accepts_overlong_fields() -> None:
    delta = DecisionPrimitiveDelta.model_validate(
        {
            "id": "dn_long",
            "family": "decisions_and_nondecisions",
            "actor_ids": [],
            "decision_trigger": _long_phrase("trigger", 24),
            "decision_question": _long_phrase("question", 36),
            "decision_mode": "decision",
            "options_considered": [
                _long_phrase("option", 16),
                _long_phrase("option", 17),
            ],
            "next_result": _long_phrase("result", 34),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 24),
                "host_lens": _long_phrase("lens", 40),
                "carry_forward": _long_phrase("carry", 42),
            },
        }
    )

    assert delta.decision_trigger.startswith("trigger")
    assert len(delta.options_considered) == 2
    assert delta.narration_hooks.concrete_detail.startswith("detail")


def test_set_piece_scene_delta_accepts_overlong_fields() -> None:
    delta = SetPieceScenePrimitiveDelta.model_validate(
        {
            "id": "sps_long",
            "family": "set_piece_scenes",
            "actor_ids": [],
            "scene_anchor": _long_phrase("anchor", 22),
            "hinge_action": _long_phrase("hinge", 21),
            "scene_outcome": _long_phrase("outcome", 23),
            "location": _long_phrase("location", 18),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 26),
                "host_lens": _long_phrase("lens", 35),
                "carry_forward": _long_phrase("carry", 37),
            },
        }
    )

    assert delta.scene_anchor.startswith("anchor")
    assert delta.location.startswith("location")


def test_human_cost_delta_accepts_overlong_fields() -> None:
    delta = HumanCostPrimitiveDelta.model_validate(
        {
            "id": "hc_long",
            "family": "human_costs",
            "actor_ids": [],
            "affected_group": _long_phrase("group", 33),
            "cost_type": _long_phrase("cost", 31),
            "concrete_marker": _long_phrase("marker", 24),
            "lived_consequence": _long_phrase("consequence", 35),
            "who_saw_it": _long_phrase("witness", 33),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 25),
                "host_lens": _long_phrase("lens", 33),
                "carry_forward": _long_phrase("carry", 36),
            },
        }
    )

    assert delta.concrete_marker.startswith("marker")
    assert delta.who_saw_it.startswith("witness")


def test_character_engine_delta_accepts_overlong_fields() -> None:
    delta = CharacterEnginePrimitiveDelta.model_validate(
        {
            "id": "ce_long",
            "family": "character_engines",
            "actor_id": None,
            "goal": _long_phrase("goal", 34),
            "pressure_box": _long_phrase("pressure", 38),
            "risk_if_it_breaks": _long_phrase("risk", 36),
            "tell": _long_phrase("tell", 24),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 28),
                "host_lens": _long_phrase("lens", 34),
                "carry_forward": _long_phrase("carry", 35),
            },
        }
    )

    assert delta.tell.startswith("tell")
    assert delta.narration_hooks.carry_forward.startswith("carry")


def test_contested_explanation_delta_accepts_overlong_candidate_readings() -> None:
    delta = ContestedExplanationPrimitiveDelta.model_validate(
        {
            "id": "cx_long",
            "family": "contested_explanations",
            "candidate_readings": [
                {
                    "label": "Reading A",
                    "claim": _long_phrase("claim", 42),
                    "emphasizes": _long_phrase("emphasizes", 38),
                    "downplays": _long_phrase("downplays", 39),
                    "support_passage_ids": ["p1", "p2"],
                }
            ],
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 29),
                "host_lens": _long_phrase("lens", 31),
                "carry_forward": _long_phrase("carry", 32),
            },
        }
    )

    assert delta.candidate_readings[0].claim.startswith("claim")


def test_epochal_turn_base_accepts_overlong_fields() -> None:
    primitive = EpochalTurnPrimitive.model_validate(
        {
            "id": "et_long",
            "family": "epochal_turns",
            "title": "The order breaks",
            "summary": "An old arrangement gives way to a new one.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "before_state": _long_phrase("before", 36),
            "after_state": _long_phrase("after", 34),
            "change_driver": _long_phrase("driver", 35),
            "proof_of_change": _long_phrase("proof", 24),
            "why_no_return": _long_phrase("return", 33),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 28),
                "host_lens": _long_phrase("lens", 34),
                "carry_forward": _long_phrase("carry", 36),
            },
        }
    )

    assert primitive.before_state.startswith("before")
    assert primitive.irreversibility_reason.startswith("return")


def test_decision_base_accepts_overlong_fields() -> None:
    primitive = DecisionPrimitive.model_validate(
        {
            "id": "dn_long",
            "family": "decisions_and_nondecisions",
            "title": "A wide decision frame",
            "summary": "The base primitive now accepts verbose enrichment output.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "decision_trigger": _long_phrase("trigger", 24),
            "decision_question": _long_phrase("question", 36),
            "decision_mode": "decision",
            "options_considered": [
                _long_phrase("option", 16),
                _long_phrase("option", 17),
            ],
            "next_result": _long_phrase("result", 34),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 24),
                "host_lens": _long_phrase("lens", 40),
                "carry_forward": _long_phrase("carry", 42),
            },
        }
    )

    assert primitive.decision_trigger.startswith("trigger")
    assert primitive.narration_hooks.concrete_detail.startswith("detail")


def test_set_piece_scene_base_accepts_overlong_fields() -> None:
    primitive = SetPieceScenePrimitive.model_validate(
        {
            "id": "sps_long",
            "family": "set_piece_scenes",
            "title": "The room turns",
            "summary": "A staged scene becomes more verbose after enrichment.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "scene_anchor": _long_phrase("anchor", 22),
            "hinge_action": _long_phrase("hinge", 21),
            "scene_outcome": _long_phrase("outcome", 23),
            "location": _long_phrase("location", 18),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 26),
                "host_lens": _long_phrase("lens", 35),
                "carry_forward": _long_phrase("carry", 37),
            },
        }
    )

    assert primitive.scene_anchor.startswith("anchor")
    assert primitive.location.startswith("location")


def test_human_cost_base_accepts_overlong_fields() -> None:
    primitive = HumanCostPrimitive.model_validate(
        {
            "id": "hc_long",
            "family": "human_costs",
            "title": "The harm lands locally",
            "summary": "Human-cost primitives now keep verbose enriched detail.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "affected_group": _long_phrase("group", 33),
            "cost_type": _long_phrase("cost", 31),
            "concrete_marker": _long_phrase("marker", 24),
            "lived_consequence": _long_phrase("consequence", 35),
            "who_saw_it": _long_phrase("witness", 33),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 25),
                "host_lens": _long_phrase("lens", 33),
                "carry_forward": _long_phrase("carry", 36),
            },
        }
    )

    assert primitive.concrete_marker.startswith("marker")
    assert primitive.visibility.startswith("witness")


def test_character_engine_base_accepts_overlong_fields() -> None:
    primitive = CharacterEnginePrimitive.model_validate(
        {
            "id": "ce_long",
            "family": "character_engines",
            "title": "A pressured actor",
            "summary": "Character engines now keep longer enriched phrasing.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_id": None,
            "goal": _long_phrase("goal", 34),
            "pressure_box": _long_phrase("pressure", 38),
            "risk_if_it_breaks": _long_phrase("risk", 36),
            "tell": _long_phrase("tell", 24),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 28),
                "host_lens": _long_phrase("lens", 34),
                "carry_forward": _long_phrase("carry", 35),
            },
        }
    )

    assert primitive.tell.startswith("tell")
    assert primitive.narration_hooks.carry_forward.startswith("carry")


def test_coalition_base_accepts_overlong_fields() -> None:
    primitive = CoalitionFaultLinePrimitive.model_validate(
        {
            "id": "cf_long",
            "family": "coalitions_and_fault_lines",
            "title": "A coalition with a long explanation",
            "summary": "Coalition fields no longer reject verbose enriched output.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "alignment_shape": _long_phrase("shape", 35),
            "alignment_basis": _long_phrase("basis", 33),
            "fracture_trigger": _long_phrase("fracture", 36),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 25),
                "host_lens": _long_phrase("lens", 31),
                "carry_forward": _long_phrase("carry", 32),
            },
        }
    )

    assert primitive.alignment_shape.startswith("shape")


def test_systems_base_accepts_overlong_fields() -> None:
    primitive = SystemsOperatingLogicPrimitive.model_validate(
        {
            "id": "sys_long",
            "family": "systems_and_operating_logics",
            "title": "A system in motion",
            "summary": "System primitives now preserve longer enriched mechanism language.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "system_name": _long_phrase("system", 34),
            "operating_chain": [
                _long_phrase("chain", 18),
                _long_phrase("chain", 19),
            ],
            "inputs": [
                _long_phrase("input", 15),
                _long_phrase("input", 14),
            ],
            "outputs": [
                _long_phrase("output", 16),
                _long_phrase("output", 15),
            ],
            "where_it_shows_up": _long_phrase("where", 34),
            "failure_mode": _long_phrase("failure", 35),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 24),
                "host_lens": _long_phrase("lens", 32),
                "carry_forward": _long_phrase("carry", 33),
            },
        }
    )

    assert primitive.operating_chain[0].startswith("chain")
    assert primitive.mechanism_steps[1].startswith("chain")


def test_contested_explanation_base_accepts_overlong_candidate_readings() -> None:
    primitive = ContestedExplanationPrimitive.model_validate(
        {
            "id": "cx_long",
            "family": "contested_explanations",
            "title": "Competing accounts remain verbose",
            "summary": "Contested explanations keep longer candidate readings after merge.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "candidate_readings": [
                {
                    "label": "Reading A",
                    "claim": _long_phrase("claim", 42),
                    "emphasizes": _long_phrase("emphasizes", 38),
                    "downplays": _long_phrase("downplays", 39),
                    "support_passage_ids": ["p1", "p2"],
                }
            ],
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 29),
                "host_lens": _long_phrase("lens", 31),
                "carry_forward": _long_phrase("carry", 32),
            },
        }
    )

    assert primitive.candidate_readings[0].claim.startswith("claim")


def test_moral_trap_base_accepts_overlong_fields() -> None:
    primitive = MoralTrapPrimitive.model_validate(
        {
            "id": "mt_long",
            "family": "moral_traps",
            "title": "The trap widens",
            "summary": "Moral-trap lists now preserve longer enriched formulations.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "competing_obligations": [
                _long_phrase("obligation", 15),
                _long_phrase("obligation", 16),
            ],
            "compromised_options": [
                _long_phrase("option", 15),
                _long_phrase("option", 14),
            ],
            "trap_structure": _long_phrase("trap", 35),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 24),
                "host_lens": _long_phrase("lens", 31),
                "carry_forward": _long_phrase("carry", 32),
            },
        }
    )

    assert primitive.competing_obligations[0].startswith("obligation")
    assert primitive.no_clean_exit_reason.startswith("trap")


def test_irony_base_accepts_overlong_fields() -> None:
    primitive = IronyReversalPrimitive.model_validate(
        {
            "id": "ir_long",
            "family": "ironies_and_reversals",
            "title": "Backfire in long form",
            "summary": "Irony and reversal primitives now accept longer enriched summaries.",
            "axis_ids": ["axis_1"],
            "core_passage_ids": ["p1"],
            "actor_ids": [],
            "expected_outcome": _long_phrase("expected", 34),
            "actual_outcome": _long_phrase("actual", 35),
            "flip_cause": _long_phrase("flip", 36),
            "narration_hooks": {
                "concrete_detail": _long_phrase("detail", 24),
                "host_lens": _long_phrase("lens", 31),
                "carry_forward": _long_phrase("carry", 32),
            },
        }
    )

    assert primitive.expected_outcome.startswith("expected")
    assert primitive.reversal_driver.startswith("flip")
