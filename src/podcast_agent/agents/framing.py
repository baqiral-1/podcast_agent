"""Episode framing agent."""

from __future__ import annotations

from json import JSONDecodeError

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.prompts.episode_framing import build_episode_framing_instructions
from podcast_agent.schemas.models import EpisodeFraming
from podcast_agent.utils.text import truncate_words


class EpisodeFramingAgent(Agent):
    """Generate recap/next framing text for one episode."""

    schema_name = "episode_framing"
    instructions = ""
    response_model = EpisodeFraming
    recap_openers = [
        "In the previous episode,",
        "Previously, in our story,",
        "Last time, on this story,",
        "Earlier in the series,",
        "Earlier in our story,",
        "Previously, we followed,",
        "Last time, we followed,",
        "Previously, we saw,",
        "Last time, we saw,",
        "Earlier, we saw,",
    ]
    next_openers = [
        "Next time,",
        "In the next episode,",
        "Next, we follow,",
        "Coming up next,",
        "Next, we see,",
        "Next, we turn to,",
        "In the next chapter of this story,",
        "Next on this journey,",
        "Next in our story,",
        "Next, the story moves to,",
    ]

    def __init__(
        self,
        llm,
        *,
        recap_min_words: int,
        recap_max_words: int,
        next_min_words: int,
        next_max_words: int,
        max_recap_source_words: int = 900,
        retry_attempts: int = 1,
    ) -> None:
        super().__init__(llm)
        self.recap_min_words = recap_min_words
        self.recap_max_words = recap_max_words
        self.next_min_words = next_min_words
        self.next_max_words = next_max_words
        self.max_recap_source_words = max_recap_source_words
        self.retry_attempts = retry_attempts

    def build_payload(
        self,
        *,
        episode_id: str,
        episode_title: str,
        recap_source: str,
        current_themes: list[str],
        next_themes: list[str] | None,
        current_outline: str,
        next_outline: str | None,
        has_previous: bool,
        has_next: bool,
    ) -> dict:
        return {
            "episode_id": episode_id,
            "episode_title": episode_title,
            "recap_source": truncate_words(recap_source, self.max_recap_source_words),
            "current_themes": current_themes,
            "next_themes": next_themes or [],
            "current_outline": current_outline,
            "next_outline": next_outline or "",
            "has_previous": has_previous,
            "has_next": has_next,
            "recap_min_words": self.recap_min_words,
            "recap_max_words": self.recap_max_words,
            "next_min_words": self.next_min_words,
            "next_max_words": self.next_max_words,
        }

    def generate(self, payload: dict) -> EpisodeFraming:
        instructions = build_episode_framing_instructions(
            recap_min_words=self.recap_min_words,
            recap_max_words=self.recap_max_words,
            next_min_words=self.next_min_words,
            next_max_words=self.next_max_words,
            recap_openers=self.recap_openers,
            next_openers=self.next_openers,
        )
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                return self.llm.generate_json(
                    schema_name=self.schema_name,
                    instructions=instructions,
                    payload=payload,
                    response_model=self.response_model,
                )
            except (ValidationError, JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                instructions = self._retry_instructions(instructions, last_error)
        raise RuntimeError(f"Episode framing failed after retry: {last_error}") from last_error

    def _retry_instructions(self, instructions: str, error: Exception) -> str:
        return f"{instructions} Rewrite to satisfy the framing requirements. Error: {error}."
