from __future__ import annotations

import math

from podcast_agent.pipeline.orchestrator import (
    _FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE,
    _build_function_tagging_passage_list,
    _split_sentences,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    BaseSynthesisPrimitive,
    ExtractedPassage,
    PrimitiveSubstrate,
)


def _actor_metadata() -> ActorMetadata:
    return ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )


def test_function_tagging_passage_list_is_flat_and_unique() -> None:
    shared_text = " ".join(
        [
            "Shared signal one now.",
            "Shared signal two now.",
            "Shared signal three now.",
            "Shared signal four now.",
            "Shared signal five now.",
            "Shared signal six now.",
            "Shared signal seven now.",
            "Shared signal eight now.",
        ]
    )
    support_text = " ".join(
        [
            "Support signal one now.",
            "Support signal two now.",
            "Support signal three now.",
            "Support signal four now.",
            "Support signal five now.",
            "Support signal six now.",
            "Support signal seven now.",
            "Support signal eight now.",
        ]
    )
    primitives = [
        BaseSynthesisPrimitive(
            id="event_1",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Turn",
            core_passage_ids=["p1"],
            support_passage_ids=["p1", "p2"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
        BaseSynthesisPrimitive(
            id="event_2",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Second turn",
            core_passage_ids=[],
            support_passage_ids=["p1"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
    ]
    passage_lookup = {
        "p1": ExtractedPassage(
            passage_id="p1",
            book_id="b1",
            chunk_ids=["c1"],
            text=shared_text,
            full_text=shared_text,
            chapter_ref="ch. 1",
            axis_id="axis_1",
        ),
        "p2": ExtractedPassage(
            passage_id="p2",
            book_id="b1",
            chunk_ids=["c2"],
            text=support_text,
            full_text=support_text,
            chapter_ref="ch. 2",
            axis_id="axis_1",
        ),
    }

    passage_list = _build_function_tagging_passage_list(
        primitives=primitives,
        passage_lookup=passage_lookup,
        actor_metadata=_actor_metadata(),
        trim_profile=_FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE,
    )

    assert [item["passage_id"] for item in passage_list] == ["p1", "p2"]
    assert all(set(item) == {"passage_id", "text"} for item in passage_list)


def test_function_tagging_passage_list_uses_15_percent_core_and_7_point_5_percent_support() -> None:
    repeated_text = " ".join(
        [
            "Alpha sentence one now.",
            "Alpha sentence two now.",
            "Alpha sentence three now.",
            "Alpha sentence four now.",
            "Alpha sentence five now.",
            "Alpha sentence six now.",
            "Alpha sentence seven now.",
            "Alpha sentence eight now.",
            "Alpha sentence nine now.",
            "Alpha sentence ten now.",
        ]
    )
    primitives = [
        BaseSynthesisPrimitive(
            id="event_1",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Core turn",
            core_passage_ids=["p1"],
            support_passage_ids=[],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
        BaseSynthesisPrimitive(
            id="event_2",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Support turn",
            core_passage_ids=[],
            support_passage_ids=["p2"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
    ]
    passage_lookup = {
        "p1": ExtractedPassage(
            passage_id="p1",
            book_id="b1",
            chunk_ids=["c1"],
            text=repeated_text,
            full_text=repeated_text,
            chapter_ref="ch. 1",
            axis_id="axis_1",
        ),
        "p2": ExtractedPassage(
            passage_id="p2",
            book_id="b1",
            chunk_ids=["c2"],
            text=repeated_text,
            full_text=repeated_text,
            chapter_ref="ch. 2",
            axis_id="axis_1",
        ),
    }

    passage_list = _build_function_tagging_passage_list(
        primitives=primitives,
        passage_lookup=passage_lookup,
        actor_metadata=_actor_metadata(),
        trim_profile=_FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE,
    )

    text_by_id = {item["passage_id"]: item["text"] for item in passage_list}
    assert len(_split_sentences(text_by_id["p1"])) == 1
    assert len(_split_sentences(text_by_id["p2"])) == 1


def test_function_tagging_passage_list_shared_passage_respects_25_percent_cap() -> None:
    shared_text = " ".join(
        [
            "Alpha pivot one now.",
            "Alpha pivot two now.",
            "Alpha pivot three now.",
            "Beta contest one now.",
            "Beta contest two now.",
            "Beta contest three now.",
            "Context spare one now.",
            "Context spare two now.",
        ]
    )
    primitives = [
        BaseSynthesisPrimitive(
            id="event_1",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Alpha pivot",
            core_passage_ids=["p1"],
            support_passage_ids=[],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
        BaseSynthesisPrimitive(
            id="event_2",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Beta contest",
            core_passage_ids=[],
            support_passage_ids=["p1"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
    ]
    passage_lookup = {
        "p1": ExtractedPassage(
            passage_id="p1",
            book_id="b1",
            chunk_ids=["c1"],
            text=shared_text,
            full_text=shared_text,
            chapter_ref="ch. 1",
            axis_id="axis_1",
        )
    }

    passage_list = _build_function_tagging_passage_list(
        primitives=primitives,
        passage_lookup=passage_lookup,
        actor_metadata=_actor_metadata(),
        trim_profile=_FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE,
    )

    assert [item["passage_id"] for item in passage_list] == ["p1"]
    final_sentences = _split_sentences(passage_list[0]["text"])
    assert len(final_sentences) == 2
    assert any("Alpha" in sentence for sentence in final_sentences)
    assert any("Beta" in sentence for sentence in final_sentences)
    combined_words = sum(len(sentence.split()) for sentence in final_sentences)
    assert combined_words <= math.floor(len(shared_text.split()) * 0.25)
