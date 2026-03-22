"""Episode framing agent."""

from __future__ import annotations

from json import JSONDecodeError

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.prompts.episode_framing import build_episode_framing_instructions
from podcast_agent.schemas.models import EpisodeFraming


class EpisodeFramingAgent(Agent):
    """Generate recap/current/next framing text for one episode."""

    schema_name = "episode_framing"
    instructions = ""
    response_model = EpisodeFraming

    def __init__(
        self,
        llm,
        *,
        recap_words: int,
        current_words: int,
        next_min_words: int,
        next_max_words: int,
        max_recap_source_words: int = 900,
        retry_attempts: int = 1,
    ) -> None:
        super().__init__(llm)
        self.recap_words = recap_words
        self.current_words = current_words
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
            "recap_source": self._truncate_words(recap_source, self.max_recap_source_words),
            "current_themes": current_themes,
            "next_themes": next_themes or [],
            "current_outline": current_outline,
            "next_outline": next_outline or "",
            "has_previous": has_previous,
            "has_next": has_next,
            "recap_words": self.recap_words,
            "current_words": self.current_words,
            "next_min_words": self.next_min_words,
            "next_max_words": self.next_max_words,
        }

    def generate(self, payload: dict) -> EpisodeFraming:
        instructions = build_episode_framing_instructions(
            recap_words=self.recap_words,
            current_words=self.current_words,
            next_min_words=self.next_min_words,
            next_max_words=self.next_max_words,
        )
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                framing = self.llm.generate_json(
                    schema_name=self.schema_name,
                    instructions=instructions,
                    payload=payload,
                    response_model=self.response_model,
                )
            except (ValidationError, JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                instructions = self._retry_instructions(instructions, payload, last_error)
                continue
            violations = self._validate_word_counts(framing, payload)
            if not violations:
                return framing
            last_error = ValueError("; ".join(violations))
            if attempt >= self.retry_attempts:
                break
            instructions = self._retry_instructions(instructions, payload, last_error)
        raise RuntimeError(f"Episode framing failed after retry: {last_error}") from last_error

    def _validate_word_counts(self, framing: EpisodeFraming, payload: dict) -> list[str]:
        violations: list[str] = []
        has_previous = payload.get("has_previous", False)
        has_next = payload.get("has_next", False)
        if has_previous:
            recap_words = self._word_count(framing.recap)
            if recap_words != self.recap_words:
                violations.append(f"recap has {recap_words} words (expected {self.recap_words})")
        if framing.current_summary:
            current_words = self._word_count(framing.current_summary)
            if current_words != self.current_words:
                violations.append(
                    f"current_summary has {current_words} words (expected {self.current_words})"
                )
        if has_next:
            next_words = self._word_count(framing.next_overview)
            if not (self.next_min_words <= next_words <= self.next_max_words):
                violations.append(
                    f"next_overview has {next_words} words (expected {self.next_min_words}-{self.next_max_words})"
                )
        return violations

    def _retry_instructions(self, instructions: str, payload: dict, error: Exception) -> str:
        del payload
        return (
            f"{instructions} The previous output violated word-count requirements. "
            f"Use exact counts and rewrite. Error: {error}."
        )

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()
