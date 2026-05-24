"""Stage 10: episode writing agent."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import (
    episode_writing_instructions,
    episode_writing_no_citations_instructions,
)
from podcast_agent.schemas.models import ActorExplanationRealization, Citation


class SectionProse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    scene_card_ids: list[str]
    movement_goal: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    source_book_ids: list[str] = Field(default_factory=list)
    actor_explanation_realizations: list[ActorExplanationRealization] = Field(
        default_factory=list
    )


class EpisodeWritingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prose_sections: list[SectionProse] = Field(default_factory=list)


class SectionProseNoCitations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    scene_card_ids: list[str]
    movement_goal: str
    text: str
    source_book_ids: list[str] = Field(default_factory=list)
    actor_explanation_realizations: list[ActorExplanationRealization] = Field(
        default_factory=list
    )


class EpisodeWritingNoCitationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prose_sections: list[SectionProseNoCitations] = Field(default_factory=list)


class WritingAgent(Agent):
    """Drafts section-level prose from ordered scene cards."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingResponse
    instructions = episode_writing_instructions()

    def build_payload(
        self,
        episode_number: int,
        strategy_episode: dict,
        architecture: dict,
        episode_plan: dict,
        passages: list[dict],
        book_metadata: list[dict],
        episode_target_word_count_lower: int | None = None,
        episode_target_word_count_higher: int | None = None,
        skip_grounding: bool = False,
        host_policy: dict | None = None,
        narrative_state_pre: dict | None = None,
        narrative_state_post: dict | None = None,
        continuity_contract_pre: dict | None = None,
        continuity_contract_post: dict | None = None,
        scene_primitive_briefs: dict[str, list[dict[str, object]]] | None = None,
        actor_metadata: dict | None = None,
        writing_feedback: str | None = None,
        prior_window_continuity: dict | None = None,
        field_semantics: dict | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "strategy_episode": strategy_episode,
            "architecture": architecture,
            "plan": episode_plan,
            "passages": passages,
            "books": book_metadata,
            "skip_grounding": skip_grounding,
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
        if scene_primitive_briefs is not None:
            payload["scene_primitive_briefs"] = scene_primitive_briefs
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if writing_feedback:
            payload["writing_feedback"] = str(writing_feedback)
        if prior_window_continuity is not None:
            payload["prior_window_continuity"] = prior_window_continuity
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        if episode_target_word_count_lower is not None:
            payload["episode_target_word_count_lower"] = int(
                episode_target_word_count_lower
            )
        if episode_target_word_count_higher is not None:
            payload["episode_target_word_count_higher"] = int(
                episode_target_word_count_higher
            )
        return payload


class WritingAgentNoCitations(WritingAgent):
    """Drafts section-level prose without citation requirements."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingNoCitationsResponse
    instructions = episode_writing_no_citations_instructions()
