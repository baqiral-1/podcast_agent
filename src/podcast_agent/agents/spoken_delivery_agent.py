"""Stage 13: spoken delivery rewrite agent."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import spoken_delivery_instructions
from podcast_agent.schemas.models import SpeechHints


class SpokenDeliveryBatchSection(BaseModel):
    section_id: str
    text: str
    speech_hints: SpeechHints


class SpokenDeliveryResponse(BaseModel):
    text: str | None = None
    speech_hints: SpeechHints | None = None
    sections: list[SpokenDeliveryBatchSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response(self) -> "SpokenDeliveryResponse":
        if self.sections:
            return self
        if self.text is None or self.speech_hints is None:
            raise ValueError("spoken delivery response must provide sections or text + speech_hints")
        return self


class SpokenDeliveryAgent(Agent):
    """Rewrites one contiguous batch of prose sections for spoken delivery."""

    schema_name = "spoken_delivery"
    response_model = SpokenDeliveryResponse
    instructions = spoken_delivery_instructions()

    def build_payload(
        self,
        episode_number: int,
        script: dict,
        max_words_per_segment: int,
        tts_provider: str,
        host_policy: dict | None = None,
        narrative_state_pre: dict | None = None,
        narrative_state_post: dict | None = None,
        continuity_contract_pre: dict | None = None,
        continuity_contract_post: dict | None = None,
        previous_spoken_tail: str | None = None,
        field_semantics: dict | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "script": script,
            "max_words_per_segment": max_words_per_segment,
            "tts_provider": tts_provider,
        }
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
        prose_sections = list(script.get("prose_sections", []) or [])
        if len(prose_sections) == 1:
            payload["section"] = prose_sections[0]
        if previous_spoken_tail:
            payload["previous_spoken_tail"] = previous_spoken_tail
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        return payload
