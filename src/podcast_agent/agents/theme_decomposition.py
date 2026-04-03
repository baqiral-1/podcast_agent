"""Stage 5: Theme decomposition agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookRecord, ThematicAxis


class ThemeDecompositionResponse(BaseModel):
    axes: list[ThematicAxis] = Field(default_factory=list)


class ThemeDecompositionAgent(Agent):
    """Decomposes a user theme into 5-15 thematic axes spanning all books."""

    schema_name = "theme_decomposition"
    response_model = ThemeDecompositionResponse
    instructions = (
        "You are a thematic analyst preparing a cross-book podcast. Given a theme and "
        "summaries of N books, decompose the theme into 5-15 specific analytical lenses "
        "(thematic axes).\n\n"
        "Each axis should be:\n"
        "- Narrow enough to drive targeted passage retrieval\n"
        "- Broad enough that at least 2 of the books have something meaningful to say about it\n\n"
        "For each axis, provide:\n"
        "- axis_id: a unique identifier\n"
        "- name: a short descriptive label (e.g., 'Cognitive Biases in High-Stakes Decisions')\n"
        "- description: 2-3 sentences explaining what this axis covers\n"
        "- guiding_questions: 3-5 questions this axis seeks to answer across the books\n"
        "- relevance_by_book: a mapping of book_id to relevance score (0.0-1.0)\n"
        "- keywords: terms useful for retrieval augmentation\n\n"
        "Return a JSON object with an 'axes' array."
    )

    def build_payload(
        self,
        theme: str,
        theme_elaboration: str | None,
        books: list[BookRecord],
    ) -> dict:
        book_summaries = []
        for book in books:
            chapter_info = [
                {"title": ch.title, "summary": ch.summary}
                for ch in book.chapters
            ]
            book_summaries.append({
                "book_id": book.book_id,
                "title": book.title,
                "author": book.author,
                "chapters": chapter_info,
            })
        return {
            "theme": theme,
            "theme_elaboration": theme_elaboration or "",
            "books": book_summaries,
        }
