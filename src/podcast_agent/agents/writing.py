"""Stage 10: Episode writing agent."""

from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.schemas.models import Citation, EpisodeScript, ScriptSegment

logger = logging.getLogger(__name__)


class EpisodeWritingResponse(BaseModel):
    title: str
    segments: list[ScriptSegment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ScriptSegmentNoCitations(BaseModel):
    segment_id: str | None = None
    text: str
    segment_type: Literal["intro", "body", "transition", "outro", "recap", "bridge"] = "body"
    beat_id: str | None = None
    source_book_ids: list[str] = Field(default_factory=list)
    attribution_level: Literal["none", "light", "full"] = "none"


class EpisodeWritingResponseNoCitations(BaseModel):
    title: str
    segments: list[ScriptSegmentNoCitations] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class WritingAgent(Agent):
    """Transforms an episode plan into a narrative-first script with selective attribution."""

    schema_name = "episode_writing"
    response_model = EpisodeWritingResponse
    instructions = (
        "You are a narrator telling a true story. You have absorbed the research of "
        "multiple historians and are now telling the story in your own voice. "
        "Transform the episode plan into a complete narration script.\n\n"
        "Requirements:\n"
        "- Follow the beat sequence provided in the plan\n"
        "- Target plan.target_duration_minutes for total runtime\n"
        "- Use each beat's estimated_duration_seconds as pacing guidance while writing\n"
        "- Use the assigned passages as source material, but do not organize by author\n"
        "- Follow the narrative_instruction and attribution_level for each beat\n"
        "- Default mode: omniscient narrator telling a story (no author names)\n"
        "- attribution_level rules:\n"
        "  - none: no author names or explicit source comparisons\n"
        "  - light: brief attribution as a narrative aside\n"
        "  - full: surface a genuine disagreement as a dramatic moment\n"
        "- Every factual claim must include a Citation linking it to a specific passage_id, "
        "book_id, and chunk_ids\n"
        "- Cross-book claims must cite passages from each referenced book\n"
        "- Avoid academic comparison language and list-style attribution\n\n"
        "Author naming rules:\n"
        "- Name authors only when required by attribution_level light or full\n"
        "- Prefer indirect attribution when allowed\n"
        "- Do not exceed the episode author name limit provided in the payload\n\n"
        "For each segment, specify:\n"
        "- segment_type: intro, body, transition, outro, recap, or bridge\n"
        "- beat_id: which beat this segment belongs to\n"
        "- source_book_ids: books referenced in this segment\n"
        "- attribution_level: none, light, or full\n"
        "- citations: inline citations\n\n"
        "Return a JSON object with title, segments, and citations"
    )
    instructions_no_citations = (
        "You are a narrator telling a true story. You have absorbed the research of "
        "multiple historians and are now telling the story in your own voice. "
        "Transform the episode plan into a complete narration script.\n\n"
        "Requirements:\n"
        "- Follow the beat sequence provided in the plan\n"
        "- Target plan.target_duration_minutes for total runtime\n"
        "- Use each beat's estimated_duration_seconds as pacing guidance while writing\n"
        "- Use the assigned passages as source material, but do not organize by author\n"
        "- Follow the narrative_instruction and attribution_level for each beat\n"
        "- Default mode: omniscient narrator telling a story (no author names)\n"
        "- attribution_level rules:\n"
        "  - none: no author names or explicit source comparisons\n"
        "  - light: brief attribution as a narrative aside\n"
        "  - full: surface a genuine disagreement as a dramatic moment\n"
        "- Avoid academic comparison language and list-style attribution\n\n"
        "Author naming rules:\n"
        "- Name authors only when required by attribution_level light or full\n"
        "- Prefer indirect attribution when allowed\n"
        "- Do not exceed the episode author name limit provided in the payload\n\n"
        "For each segment, specify:\n"
        "- segment_type: intro, body, transition, outro, recap, or bridge\n"
        "- beat_id: which beat this segment belongs to\n"
        "- source_book_ids: books referenced in this segment\n"
        "- attribution_level: none, light, or full\n\n"
        "Return a JSON object with title and segments."
    )

    def run(self, payload: dict) -> BaseModel:
        """Execute the agent with optional citation-free instructions."""
        use_no_citations = bool(payload.get("skip_grounding"))
        instructions = self.instructions_no_citations if use_no_citations else self.instructions
        response_model = (
            EpisodeWritingResponseNoCitations if use_no_citations else self.response_model
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retry_attempts + 1):
            with llm_semaphore_for(self.schema_name):
                try:
                    return self.llm.generate_json(
                        schema_name=self.schema_name,
                        instructions=instructions,
                        payload=payload,
                        response_model=response_model,
                        attempt=attempt,
                        max_attempts=self.max_retry_attempts,
                    )
                except (TransientLLMError, RetryableGenerationError) as exc:
                    last_exc = exc
                    if attempt < self.max_retry_attempts:
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        logger.warning(
                            "Agent %s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                            self.schema_name, attempt, self.max_retry_attempts,
                            type(exc).__name__, exc, backoff,
                        )
                        time.sleep(backoff)
                    continue
                except Exception:
                    raise
        raise last_exc  # type: ignore[misc]

    def build_payload(
        self,
        episode_number: int,
        episode_plan: dict,
        passages: list[dict],
        book_metadata: list[dict],
        *,
        max_author_names_per_episode: int,
        prefer_indirect_attribution: bool,
        skip_grounding: bool = False,
    ) -> dict:
        return {
            "episode_number": episode_number,
            "plan": episode_plan,
            "passages": passages,
            "books": book_metadata,
            "max_author_names_per_episode": max_author_names_per_episode,
            "prefer_indirect_attribution": prefer_indirect_attribution,
            "skip_grounding": skip_grounding,
        }
