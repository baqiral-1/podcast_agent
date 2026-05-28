"""Stage 5: Theme decomposition agent."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import theme_decomposition_instructions
from podcast_agent.schemas.models import BookRecord, ChapterInfo, ThematicAxis


class ThemeDecompositionResponse(BaseModel):
    axes: list[ThematicAxis] = Field(default_factory=list)
    actor_metadata: dict[str, Any] = Field(default_factory=dict)


class ThemeDecompositionAgent(Agent):
    """Decomposes a user theme into 12-20 strong thematic axes spanning all books.

    This stage is the intentional consumer of chapter-level analysis and
    synthesized per-book summaries. Later stages use retrieved passage evidence
    instead of chapter-level context.
    """

    schema_name = "theme_decomposition"
    response_model = ThemeDecompositionResponse
    instructions = theme_decomposition_instructions()
    _MAX_THEMES_TOUCHED = 3
    _MAX_MAJOR_ACTORS = 3
    _MAX_KEY_EVENTS = 2
    _MAX_THEME_ITEM_CHARS = 64
    _MAX_ACTOR_ITEM_CHARS = 64
    _MAX_EVENT_ITEM_CHARS = 180
    _WHITESPACE_RE = re.compile(r"\s+")

    @classmethod
    def _normalize_and_truncate(cls, value: str, max_chars: int) -> str:
        normalized = cls._WHITESPACE_RE.sub(" ", value).strip()
        if not normalized:
            return ""
        return normalized[:max_chars]

    @classmethod
    def _compact_string_list(
        cls, values: list[str], max_items: int, max_chars: int,
    ) -> list[str]:
        compacted: list[str] = []
        for value in values:
            trimmed = cls._normalize_and_truncate(value, max_chars)
            if not trimmed:
                continue
            compacted.append(trimmed)
            if len(compacted) >= max_items:
                break
        return compacted

    @staticmethod
    def _compact_chapter_analysis(chapter: ChapterInfo) -> dict[str, Any] | None:
        analysis = chapter.analysis
        if analysis is None:
            return None
        return {
            "themes_touched": ThemeDecompositionAgent._compact_string_list(
                analysis.themes_touched,
                ThemeDecompositionAgent._MAX_THEMES_TOUCHED,
                ThemeDecompositionAgent._MAX_THEME_ITEM_CHARS,
            ),
            "major_actors": ThemeDecompositionAgent._compact_string_list(
                analysis.major_actors,
                ThemeDecompositionAgent._MAX_MAJOR_ACTORS,
                ThemeDecompositionAgent._MAX_ACTOR_ITEM_CHARS,
            ),
            "key_events_or_arguments": ThemeDecompositionAgent._compact_string_list(
                analysis.key_events_or_arguments,
                ThemeDecompositionAgent._MAX_KEY_EVENTS,
                ThemeDecompositionAgent._MAX_EVENT_ITEM_CHARS,
            ),
        }

    def build_payload(
        self,
        theme: str,
        sub_themes: list[str] | None,
        theme_elaboration: str | None,
        books: list[BookRecord],
        axis_count_min: int,
        axis_count_max: int,
        actor_count_min: int = 10,
        actor_count_max: int = 40,
        book_summaries: dict[str, str] | None = None,
    ) -> dict:
        summary_by_book = book_summaries or {}
        book_summaries = []
        for book in books:
            chapter_info: list[dict[str, Any]] = []
            for ch in book.chapters:
                entry = {
                    "analysis": self._compact_chapter_analysis(ch),
                }
                chapter_info.append(entry)
            book_summaries.append({
                "book_id": book.book_id,
                "title": book.title,
                "author": book.author,
                "book_summary": summary_by_book.get(book.book_id, ""),
                "chapters": chapter_info,
            })
        return {
            "theme": theme,
            "sub_themes": sub_themes or [],
            "theme_elaboration": theme_elaboration or "",
            "axis_count_min": axis_count_min,
            "axis_count_max": axis_count_max,
            "actor_count_min": actor_count_min,
            "actor_count_max": actor_count_max,
            "books": book_summaries,
        }

    def build_instructions(self, payload: dict) -> str:
        axis_count_min = int(payload.get("axis_count_min", 12))
        axis_count_max = int(payload.get("axis_count_max", 20))
        actor_count_min = int(payload.get("actor_count_min", 10))
        actor_count_max = int(payload.get("actor_count_max", 40))
        return theme_decomposition_instructions(
            axis_count_min=axis_count_min,
            axis_count_max=axis_count_max,
            actor_count_min=actor_count_min,
            actor_count_max=actor_count_max,
        )
