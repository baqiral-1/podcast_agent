"""Stage 5: Theme decomposition agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import theme_decomposition_instructions
from podcast_agent.schemas.models import BookRecord, ThematicAxis


class ThemeDecompositionResponse(BaseModel):
    axes: list[ThematicAxis] = Field(default_factory=list)


class ThemeDecompositionAgent(Agent):
    """Decomposes a user theme into 10-15 strong thematic axes spanning all books.

    This stage is the intentional consumer of chapter summaries and synthesized
    per-book summaries. Later stages use retrieved passage evidence instead of
    chapter-summary context.
    """

    schema_name = "theme_decomposition"
    response_model = ThemeDecompositionResponse
    instructions = theme_decomposition_instructions()

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
            # Chapter summaries are provided here for thematic axis discovery only.
            chapter_info = []
            for ch in book.chapters:
                entry = {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
                    "summary": ch.summary,
                }
                if ch.analysis is not None:
                    entry.update({
                        "themes_touched": list(ch.analysis.themes_touched),
                        "major_tensions": list(ch.analysis.major_tensions),
                        "causal_shifts": list(ch.analysis.causal_shifts),
                        "narrative_hooks": list(ch.analysis.narrative_hooks),
                        "retrieval_keywords": list(ch.analysis.retrieval_keywords),
                    })
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
