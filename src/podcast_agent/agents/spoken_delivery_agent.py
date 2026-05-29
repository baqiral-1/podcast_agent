"""Stage 13: oral rewriter (replaces the prior "compliance + paraphrase"
spoken delivery pass).

The agent now restructures paragraph order, breaks em-dash chains, varies
register section-by-section, marks high-quotability excerpts for actor
voicing, and reframes number runs sensorily — while preserving every fact,
date, name, number, and verbatim quotation.

Response shape replaces the old single-text fallback with batch-only output
in the new ``SpokenSection`` / ``SpokenSegment`` schema. The render manifest
consumes ``SpokenSegment.speaker_id`` for per-segment voice override
(actor voicing) and ``SpokenSegment.delivery_instructions`` for per-segment
TTS instructions (when the provider supports them — currently
``gpt-4o-mini-tts`` only).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import spoken_delivery_instructions
from podcast_agent.schemas.models import SpokenSection


class SpokenDeliveryResponse(BaseModel):
    """Batch oral-rewriter output.

    Always uses the segment-aware ``SpokenSection`` schema; the prior dual
    ``text`` / ``sections`` shape is gone. A batch may carry 1..N sections.
    """

    sections: list[SpokenSection] = Field(default_factory=list, min_length=1)


class SpokenDeliveryAgent(Agent):
    """Oral rewriter for one contiguous batch of prose sections."""

    schema_name = "spoken_delivery"
    response_model = SpokenDeliveryResponse
    instructions = spoken_delivery_instructions()

    def build_payload(
        self,
        episode_number: int,
        script: dict,
        max_words_per_segment: int,
        tts_provider: str,
        tts_provider_capabilities: dict | None = None,
        quotability_marks: list[dict] | None = None,
        host_policy: dict | None = None,
        narrative_state_pre: dict | None = None,
        narrative_state_post: dict | None = None,
        continuity_contract_pre: dict | None = None,
        continuity_contract_post: dict | None = None,
        previous_spoken_tail: str | None = None,
        field_semantics: dict | None = None,
        actor_voice_catalog: dict[str, str] | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "script": script,
            "max_words_per_segment": max_words_per_segment,
            "tts_provider": tts_provider,
        }
        if tts_provider_capabilities is not None:
            payload["tts_provider_capabilities"] = tts_provider_capabilities
        if quotability_marks is not None:
            payload["quotability_marks"] = list(quotability_marks)
        if actor_voice_catalog is not None:
            payload["actor_voice_catalog"] = dict(actor_voice_catalog)
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if narrative_state_pre is not None:
            payload["narrative_state_pre"] = narrative_state_pre
        if narrative_state_post is not None:
            payload["narrative_state_post"] = narrative_state_post
        if continuity_contract_pre is not None:
            payload["continuity_contract_pre"] = continuity_contract_pre
        if continuity_contract_post is not None:
            payload["continuity_contract_post"] = continuity_contract_post
        if previous_spoken_tail:
            payload["previous_spoken_tail"] = previous_spoken_tail
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        return payload
