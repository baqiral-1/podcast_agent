"""Stage 5: Theme decomposition agent."""

from __future__ import annotations

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

    @staticmethod
    def _compact_chapter_analysis(chapter: ChapterInfo) -> dict[str, Any] | None:
        analysis = chapter.analysis
        if analysis is None:
            return None
        return {
            "themes_touched": analysis.themes_touched,
            "major_actors": analysis.major_actors,
            "key_events_or_arguments": analysis.key_events_or_arguments,
        }

    def build_payload(
        self,
        theme: str,
        sub_themes: list[str] | None,
        theme_elaboration: str | None,
        books: list[BookRecord],
        book_summaries: dict[str, str] | None = None,
    ) -> dict:
        summary_by_book = book_summaries or {}
        book_summaries = []
        for book in books:
            chapter_info: list[dict[str, Any]] = []
            for ch in book.chapters:
                entry = {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
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
            "books": book_summaries,
        }
