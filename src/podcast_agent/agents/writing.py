"""Stage 10: episode writing agent."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import (
    episode_writing_instructions,
    episode_writing_no_citations_instructions,
)
from podcast_agent.schemas.models import ProseSection


_TEASER_LINE_RE = re.compile(
    r"(?im)^\s*(?:next time|coming up next|in the next episode|on the next episode)\b\s*[:.,-]?"
)


def _validate_no_teaser_lines(
    sections: list[ProseSection] | list["ProseSectionNoCitations"],
) -> None:
    for section in sections:
        if _TEASER_LINE_RE.search(section.text):
            raise ValueError(
                f"section {section.section_id!r} contains next-episode teaser copy"
            )


class EpisodeWritingResponse(BaseModel):
    prose_sections: list[ProseSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_teaser_lines(self) -> "EpisodeWritingResponse":
        _validate_no_teaser_lines(self.prose_sections)
        return self


class ProseSectionNoCitations(BaseModel):
    section_id: str
    scene_card_ids: list[str] = Field(default_factory=list)
    movement_goal: str
    text: str
    source_book_ids: list[str] = Field(default_factory=list)


class EpisodeWritingNoCitationsResponse(BaseModel):
    prose_sections: list[ProseSectionNoCitations] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_teaser_lines(self) -> "EpisodeWritingNoCitationsResponse":
        _validate_no_teaser_lines(self.prose_sections)
        return self


class WritingAgent(Agent):
    """Drafts a section-based episode batch from scene-card windows."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingResponse
    instructions = episode_writing_instructions()

    def build_payload(
        self,
        episode_number: int,
        episode_plan: dict,
        passages: list[dict],
        book_metadata: list[dict],
        episode_target_word_count_lower: int | None = None,
        episode_target_word_count_higher: int | None = None,
        skip_grounding: bool = False,
        actor_metadata: dict | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "plan": episode_plan,
            "passages": passages,
            "books": book_metadata,
            "skip_grounding": skip_grounding,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if episode_target_word_count_lower is not None:
            payload["episode_target_word_count_lower"] = int(episode_target_word_count_lower)
        if episode_target_word_count_higher is not None:
            payload["episode_target_word_count_higher"] = int(episode_target_word_count_higher)
        return payload


class WritingAgentNoCitations(WritingAgent):
    """Drafts a section-based episode batch without citation requirements."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingNoCitationsResponse
    instructions = episode_writing_no_citations_instructions()
