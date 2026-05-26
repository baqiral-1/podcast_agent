"""Compact transport helpers for large LLM-facing payloads."""

from __future__ import annotations

from typing import Any

_TRANSPORT_SCHEMA_NAMES = {
    "primitive_substrate_extraction",
    "narrative_strategy_enrichment",
    "narrative_strategy_skeleton",
    "episode_architecture",
    "episode_planning",
}
_TRANSPORT_SCHEMA_PREFIXES = ("primitive_function_tagging_",)

_BASE_CANONICAL_TO_ALIAS: dict[str, str] = {
    "substrate": "sub",
    "core_passage_ids": "core",
    "support_passage_ids": "supp",
    "timeframe": "time",
    "geography": "geo",
    "event_type": "etype",
    "what_happened": "event",
    "act_summary": "act",
    "acting_subject": "subject",
    "utterance_summary": "utter",
    "goal_or_project": "goal",
    "stakes_or_fears": "stakes",
    "mechanism_name": "mech",
    "condition_summary": "cond",
    "reading_summary": "read",
    "narration_hooks": "hooks",
    "passage_id": "pid",
    "passage_ids": "passages",
    "core_primitive_ids": "core_prims",
    "support_primitive_roles": "support_roles",
    "recall_primitive_ids": "recall_prims",
    "series_explanation_registry": "term_registry",
    "series_actor_explanation_registry": "actor_registry",
    "episode_spine": "spine",
    "major_turn_section_id": "major_turn",
    "priority_core_passage_ids": "priority_core",
    "authorial_passages": "authorial",
    "term_explanations": "term_plans",
    "actor_explanations": "actor_plans",
    "host_presence_beats": "host_beats",
    "source_passage_ids": "source_passages",
    "source_primitive_ids": "source_prims",
    "must_land_facts": "facts",
    "scene_cards": "scenes",
    "dropped_support_primitive_reasons": "dropped_support",
    "strategy_skeleton": "skeleton",
    "episode_scene_candidates": "episode_scenes",
}
_SCHEMA_CANONICAL_TO_ALIAS: dict[str, dict[str, str]] = {
    "episode_planning": {
        "episode_number": "ep",
        "framing": "frame",
        "opening_image": "open_img",
        "threat_or_unresolved_action": "threat",
        "opening_question": "open_q",
        "handoff_scene_card_id": "handoff",
        "answer_scene_card_id": "answer_sid",
        "scene_id": "sid",
        "section_id": "sec",
        "title": "ttl",
        "scene_role": "role",
        "scene_job": "job",
        "beat_change": "beat",
        "required": "req",
        "strongly_preferred": "pref",
        "if_room": "room",
        "entry_image": "img",
        "observable_detail": "detail",
        "estimated_duration_seconds": "dur",
        "word_count_priority": "wc",
        "host_moves": "moves",
        "move_type": "type",
        "target": "tgt",
        "surface_mode": "surf",
        "address_mode": "addr",
        "withhold_until": "withhold",
        "reveal_phase": "phase",
        "reveal_scene_id": "reveal_sid",
        "surrogate_label": "label",
        "primitive_ids": "prims",
        "authorial_passage_ids": "authorial_ids",
    }
}
_PRIMITIVE_CANONICAL_TO_ALIAS: dict[str, str] = {
    **_BASE_CANONICAL_TO_ALIAS,
    "actor_ids": "actors",
}
_PRESERVE_SUBTREES = frozenset(
    {
        "synthesis_feedback",
        "function_feedback",
        "strategy_feedback",
        "strategy_skeleton_feedback",
        "strategy_enrichment_feedback",
        "architecture_feedback",
        "planning_feedback",
    }
)


def _transport_key_map(schema_name: str, *, reverse: bool) -> dict[str, str]:
    if (
        schema_name == "primitive_substrate_extraction"
        or schema_name.startswith(_TRANSPORT_SCHEMA_PREFIXES)
    ):
        key_map = _PRIMITIVE_CANONICAL_TO_ALIAS
    else:
        key_map = {
            **_BASE_CANONICAL_TO_ALIAS,
            **_SCHEMA_CANONICAL_TO_ALIAS.get(schema_name, {}),
        }
    if reverse:
        return {alias: canonical for canonical, alias in key_map.items()}
    return key_map


def transport_enabled_for_schema(schema_name: str) -> bool:
    if schema_name in _TRANSPORT_SCHEMA_NAMES:
        return True
    return any(schema_name.startswith(prefix) for prefix in _TRANSPORT_SCHEMA_PREFIXES)


def encode_transport_payload(schema_name: str, payload: Any) -> Any:
    if not transport_enabled_for_schema(schema_name):
        return payload
    return _transform_keys(
        payload,
        _transport_key_map(schema_name, reverse=False),
        preserve_subtrees=True,
    )


def decode_transport_payload(schema_name: str, payload: Any) -> Any:
    if not transport_enabled_for_schema(schema_name):
        return payload
    return _transform_keys(
        payload,
        _transport_key_map(schema_name, reverse=True),
        preserve_subtrees=False,
    )


def _transform_keys(
    value: Any,
    key_map: dict[str, str],
    *,
    preserve_subtrees: bool,
) -> Any:
    if isinstance(value, list):
        return [
            _transform_keys(item, key_map, preserve_subtrees=preserve_subtrees)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    transformed: dict[str, Any] = {}
    for key, item in value.items():
        next_key = key_map.get(key, key)
        if preserve_subtrees and key in _PRESERVE_SUBTREES:
            transformed[next_key] = item
            continue
        transformed[next_key] = _transform_keys(
            item, key_map, preserve_subtrees=preserve_subtrees
        )
    return transformed
