"""Stage 13: spoken delivery rewrite agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import spoken_delivery_instructions
from podcast_agent.schemas.models import SpeechHints


class SpokenDeliveryResponse(BaseModel):
    text: str
    speech_hints: SpeechHints


class SpokenDeliveryAgent(Agent):
    """Rewrites one prose section at a time for spoken delivery."""

    schema_name = "spoken_delivery"
    response_model = SpokenDeliveryResponse
    instructions = spoken_delivery_instructions()

    def build_payload(
        self,
        episode_number: int,
        section: dict,
        max_words_per_segment: int,
        tts_provider: str,
        previous_spoken_tail: str | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "section": section,
            "max_words_per_segment": max_words_per_segment,
            "tts_provider": tts_provider,
        }
        if previous_spoken_tail:
            payload["previous_spoken_tail"] = previous_spoken_tail
        return payload
