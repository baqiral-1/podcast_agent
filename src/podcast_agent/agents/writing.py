"""Stage 10: episode writing agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import (
    episode_writing_instructions,
    episode_writing_no_citations_instructions,
)
from podcast_agent.schemas.models import ProseSection, ScriptTransition, WindowMapEntry


class EpisodeWritingResponse(BaseModel):
    batch_id: str
    prose_sections: list[ProseSection] = Field(default_factory=list)
    transitions: list[ScriptTransition] = Field(default_factory=list)
    window_map: list[WindowMapEntry] = Field(default_factory=list)


class ProseSectionNoCitations(BaseModel):
    section_id: str
    scene_card_ids: list[str] = Field(default_factory=list)
    movement_goal: str
    text: str
    source_book_ids: list[str] = Field(default_factory=list)


class ScriptTransitionNoCitations(BaseModel):
    transition_id: str
    after_section_id: str
    before_section_id: str | None = None
    text: str
    source_book_ids: list[str] = Field(default_factory=list)


class EpisodeWritingNoCitationsResponse(BaseModel):
    batch_id: str
    prose_sections: list[ProseSectionNoCitations] = Field(default_factory=list)
    transitions: list[ScriptTransitionNoCitations] = Field(default_factory=list)
    window_map: list[WindowMapEntry] = Field(default_factory=list)


class WritingAgent(Agent):
    """Drafts a section-based episode batch from scene-card windows."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingResponse
    instructions = episode_writing_instructions()

    def build_payload(
        self,
        episode_number: int,
        batch_id: str,
        episode_plan: dict,
        active_scene_card_ids: list[str],
        passages: list[dict],
        book_metadata: list[dict],
        previous_sections: list[dict] | None = None,
        previous_transitions: list[dict] | None = None,
        skip_grounding: bool = False,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "batch_id": batch_id,
            "plan": episode_plan,
            "active_scene_card_ids": active_scene_card_ids,
            "passages": passages,
            "books": book_metadata,
            "skip_grounding": skip_grounding,
        }
        if previous_sections:
            payload["previous_sections"] = previous_sections
        if previous_transitions:
            payload["previous_transitions"] = previous_transitions
        return payload


class WritingAgentNoCitations(WritingAgent):
    """Drafts a section-based episode batch without citation requirements."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingNoCitationsResponse
    instructions = episode_writing_no_citations_instructions()
