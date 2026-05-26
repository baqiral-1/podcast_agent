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


def test_encode_transport_payload_uses_planning_specific_compact_keys() -> None:
    payload = {
        "episode_number": 1,
        "framing": {
            "opening_image": "A crowded gate.",
            "threat_or_unresolved_action": "No one knows who will fire first.",
            "opening_question": "Who actually controls the square?",
            "handoff_scene_card_id": "scene_1",
        },
        "scene_cards": [
            {
                "scene_id": "scene_1",
                "section_id": "sec_1",
                "title": "Arrival",
                "scene_role": "actor_setup",
                "scene_job": "opening",
                "beat_change": "The room is now charged.",
                "must_land_facts": {"required": ["The crowd is waiting."]},
                "entry_image": "A crowded gate.",
                "observable_detail": "The gate is still shut.",
                "estimated_duration_seconds": 45,
                "host_moves": {"open": [{"move_type": "orient", "target": "crowd outside"}]},
                "passage_ids": ["000001"],
            }
        ],
    }

    encoded = encode_transport_payload("episode_planning", payload)

    assert encoded["ep"] == 1
    assert encoded["frame"]["open_img"] == "A crowded gate."
    scene = encoded["scenes"][0]
    assert scene["sid"] == "scene_1"
    assert scene["sec"] == "sec_1"
    assert scene["ttl"] == "Arrival"
    assert scene["role"] == "actor_setup"
    assert scene["job"] == "opening"
    assert scene["beat"] == "The room is now charged."
    assert scene["facts"]["req"] == ["The crowd is waiting."]
    assert scene["img"] == "A crowded gate."
    assert scene["detail"] == "The gate is still shut."
    assert scene["dur"] == 45
    assert scene["moves"]["open"][0]["type"] == "orient"
    assert scene["moves"]["open"][0]["tgt"] == "crowd outside"


def test_decode_transport_payload_restores_planning_specific_compact_keys() -> None:
    payload = {
        "ep": 1,
        "frame": {
            "open_img": "A crowded gate.",
            "threat": "No one knows who will fire first.",
            "open_q": "Who actually controls the square?",
            "handoff": "scene_1",
        },
        "scenes": [
            {
                "sid": "scene_1",
                "sec": "sec_1",
                "ttl": "Arrival",
                "role": "actor_setup",
                "job": "opening",
                "beat": "The room is now charged.",
                "facts": {"req": ["The crowd is waiting."]},
                "img": "A crowded gate.",
                "detail": "The gate is still shut.",
                "dur": 45,
                "moves": {"open": [{"type": "orient", "tgt": "crowd outside"}]},
                "passages": ["000001"],
            }
        ],
    }

    decoded = decode_transport_payload("episode_planning", payload)

    assert decoded["episode_number"] == 1
    assert decoded["framing"]["opening_image"] == "A crowded gate."
    scene = decoded["scene_cards"][0]
    assert scene["scene_id"] == "scene_1"
    assert scene["section_id"] == "sec_1"
    assert scene["scene_role"] == "actor_setup"
    assert scene["scene_job"] == "opening"
    assert scene["must_land_facts"]["required"] == ["The crowd is waiting."]
    assert scene["entry_image"] == "A crowded gate."
    assert scene["observable_detail"] == "The gate is still shut."
    assert scene["estimated_duration_seconds"] == 45
    assert scene["host_moves"]["open"][0]["move_type"] == "orient"
    assert scene["host_moves"]["open"][0]["target"] == "crowd outside"
    assert scene["passage_ids"] == ["000001"]


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


def test_planning_transport_has_no_residue_alias_and_round_trips() -> None:
    # The residue concept was removed: residue_scene_card_id/residue_sid must not
    # survive encode/decode, and the canonical answer_scene_card_id round-trips.
    payload = {
        "episode_number": 1,
        "framing": {
            "opening_image": "A crowded gate.",
            "threat_or_unresolved_action": "No one knows who will fire first.",
            "opening_question": "Who actually controls the square?",
            "handoff_scene_card_id": "scene_1",
        },
        "scene_cards": [
            {
                "scene_id": "scene_1",
                "section_id": "sec_1",
                "title": "Arrival",
                "scene_role": "actor_setup",
                "scene_job": "opening",
                "beat_change": "The room is now charged.",
                "must_land_facts": {"required": ["The crowd is waiting."]},
                "entry_image": "A crowded gate.",
                "observable_detail": "The gate is still shut.",
                "estimated_duration_seconds": 45,
                "host_moves": {"open": [{"move_type": "orient", "target": "crowd"}]},
                "passage_ids": ["000001"],
            },
            {
                "scene_id": "scene_2",
                "section_id": "sec_2",
                "title": "The answer",
                "scene_role": "implication",
                "scene_job": "answer",
                "beat_change": "The answer becomes clear.",
                "must_land_facts": {"required": ["The decision lands."]},
                "entry_image": "A signed order on the desk.",
                "observable_detail": "Ink still wet.",
                "estimated_duration_seconds": 60,
                "host_moves": {"pivot": [{"move_type": "clarify", "target": "the decision"}]},
                "passage_ids": ["000002"],
            },
            {
                "scene_id": "scene_3",
                "section_id": "sec_3",
                "title": "Exit",
                "scene_role": "implication",
                "scene_job": "close",
                "beat_change": "The episode exits.",
                "must_land_facts": {"required": ["Nothing reopens."]},
                "entry_image": "Lights down.",
                "observable_detail": "An empty hall.",
                "estimated_duration_seconds": 30,
                "host_moves": {"close": [{"move_type": "callback", "target": "clean exit"}]},
                "passage_ids": ["000003"],
            },
        ],
        "answer_scene_card_id": "scene_2",
    }

    encoded = encode_transport_payload("episode_planning", payload)
    assert "residue_scene_card_id" not in encoded
    assert "residue_sid" not in encoded
    assert encoded["answer_sid"] == "scene_2"

    decoded = decode_transport_payload("episode_planning", encoded)
    assert "residue_sid" not in decoded
    assert "residue_scene_card_id" not in decoded
    assert decoded["answer_scene_card_id"] == "scene_2"

    validated = EpisodePlanDraft.model_validate(decoded)
    assert validated.answer_scene_card_id == "scene_2"
    assert not hasattr(validated, "residue_scene_card_id")
