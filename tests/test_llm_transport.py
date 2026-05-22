from __future__ import annotations

from podcast_agent.schemas.models import EpisodePlanDraft
from podcast_agent.llm.transport import (
    decode_transport_payload,
    encode_transport_payload,
)


def test_encode_transport_payload_uses_compact_keys_for_repeated_fields() -> None:
    payload = {
        "project_id": "proj",
        "podcast_mode": "full",
        "base_primitives": [
            {
                "id": "e1",
                "substrate": "events",
                "core_passage_ids": ["000001"],
                "support_passage_ids": ["000002"],
                "timeframe": "1979",
                "geography": "Tehran",
                "actor_ids": ["actor_1"],
                "event_type": "occupation",
                "what_happened": "Students seize the embassy compound.",
            }
        ],
    }

    encoded = encode_transport_payload("primitive_function_tagging_events", payload)

    primitive = encoded["base_primitives"][0]
    assert primitive["sub"] == "events"
    assert primitive["core"] == ["000001"]
    assert primitive["supp"] == ["000002"]
    assert primitive["time"] == "1979"
    assert primitive["geo"] == "Tehran"
    assert primitive["actors"] == ["actor_1"]
    assert primitive["etype"] == "occupation"
    assert primitive["event"] == "Students seize the embassy compound."


def test_encode_transport_payload_preserves_feedback_subtrees() -> None:
    payload = {
        "project_id": "proj",
        "synthesis_feedback": {
            "primitive_snapshot": {
                "substrate": "events",
                "core_passage_ids": ["000001"],
                "what_happened": "A crisis lands publicly.",
            }
        },
    }

    encoded = encode_transport_payload("primitive_substrate_extraction", payload)

    snapshot = encoded["synthesis_feedback"]["primitive_snapshot"]
    assert snapshot["substrate"] == "events"
    assert snapshot["core_passage_ids"] == ["000001"]
    assert snapshot["what_happened"] == "A crisis lands publicly."


def test_decode_transport_payload_restores_canonical_names() -> None:
    payload = {
        "episode_number": 1,
        "scenes": [
            {
                "section_id": "sec_1",
                "title": "Arrival",
                "scene_role": "context_setup",
                "scene_function": "scene",
                "beat_change": "The room is now charged.",
                "facts": ["The crowd is waiting."],
                "estimated_duration_seconds": 45,
                "host_moves": {"open": [{"move_type": "orient"}]},
                "passages": ["000001"],
            }
        ],
        "dropped_support": {"m1": "Redundant with stronger scene evidence."},
        "framing": {
            "opening_image": "A crowded gate.",
            "threat_or_unresolved_action": "No one knows who will fire first.",
            "opening_question": "Who actually controls the square?",
            "handoff_scene_card_id": "scene_1",
        },
    }

    decoded = decode_transport_payload("episode_planning", payload)

    assert "scene_cards" in decoded
    assert decoded["scene_cards"][0]["must_land_facts"] == ["The crowd is waiting."]
    assert decoded["scene_cards"][0]["passage_ids"] == ["000001"]
    assert decoded["dropped_support_primitive_reasons"]["m1"] == (
        "Redundant with stronger scene evidence."
    )


def test_decode_transport_payload_restores_primitive_actor_ids() -> None:
    payload = {
        "project_id": "proj",
        "primitives": [
            {
                "id": "e1",
                "sub": "events",
                "actors": ["actor_1", "actor_2"],
                "event": "Students seize the embassy compound.",
            }
        ],
    }

    decoded = decode_transport_payload("primitive_substrate_extraction", payload)

    assert decoded["primitives"][0]["actor_ids"] == ["actor_1", "actor_2"]
    assert "actors" not in decoded["primitives"][0]


def test_decode_transport_payload_preserves_scene_card_actors_for_planning() -> None:
    payload = {
        "episode_number": 1,
        "framing": {
            "opening_image": "A crowded gate.",
            "threat_or_unresolved_action": "No one knows who will fire first.",
            "opening_question": "Who actually controls the square?",
            "handoff_scene_card_id": "scene_1",
        },
        "scenes": [
            {
                "scene_id": "scene_1",
                "section_id": "sec_1",
                "title": "Arrival",
                "scene_role": "actor_setup",
                "scene_function": "scene",
                "beat_change": "The room is now charged.",
                "facts": ["The crowd is waiting."],
                "actors": [
                    {
                        "name": "Actor 1",
                        "actor_id": "actor_1",
                        "presence": "primary",
                    }
                ],
                "estimated_duration_seconds": 45,
                "host_moves": {"open": [{"move_type": "orient"}]},
                "passages": ["000001"],
            }
        ],
    }

    decoded = decode_transport_payload("episode_planning", payload)

    assert decoded["scene_cards"][0]["actors"] == [
        {"name": "Actor 1", "actor_id": "actor_1", "presence": "primary"}
    ]
    assert "actor_ids" not in decoded["scene_cards"][0]


def test_decoded_planning_payload_with_scene_actors_validates() -> None:
    payload = {
        "episode_number": 1,
        "framing": {
            "opening_image": "A crowded gate.",
            "threat_or_unresolved_action": "No one knows who will fire first.",
            "opening_question": "Who actually controls the square?",
            "handoff_scene_card_id": "scene_1",
        },
        "scenes": [
            {
                "scene_id": "scene_1",
                "section_id": "sec_1",
                "title": "Arrival",
                "scene_role": "actor_setup",
                "scene_function": "scene",
                "beat_change": "The room is now charged.",
                "facts": ["The crowd is waiting."],
                "actors": [
                    {
                        "name": "Actor 1",
                        "actor_id": "actor_1",
                        "presence": "primary",
                    }
                ],
                "estimated_duration_seconds": 45,
                "host_moves": {"open": [{"move_type": "orient"}]},
                "passages": ["000001"],
            }
        ],
    }

    decoded = decode_transport_payload("episode_planning", payload)
    validated = EpisodePlanDraft.model_validate(decoded)

    assert validated.scene_cards[0].actors[0].actor_id == "actor_1"
