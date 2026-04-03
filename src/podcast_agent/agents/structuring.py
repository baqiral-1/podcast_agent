"""Stage 2: Chapter structuring agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookRecord, ChapterInfo


class StructuredChapterResponse(BaseModel):
    chapters: list[ChapterInfo] = Field(default_factory=list)


class StructuringAgent(Agent):
    """Identifies chapter boundaries and produces chapter-level summaries."""

    schema_name = "structuring"
    response_model = StructuredChapterResponse
    instructions = (
        "You are a book analyst. Given a section of text from a book, identify chapter "
        "or major section boundaries. Return the title and character offset for each boundary. "
        "If no explicit chapter headings exist, identify natural thematic breaks.\n\n"
        "For each chapter, provide:\n"
        "- chapter_id: a unique identifier\n"
        "- title: the chapter title or a descriptive label\n"
        "- start_index: character offset where the chapter starts\n"
        "- end_index: character offset where the chapter ends\n"
        "- word_count: approximate word count\n"
        "- summary: a 2-3 sentence summary of the chapter content\n\n"
        "Return a JSON object with a 'chapters' array."
    )

    def build_payload(self, book_record: BookRecord, text_window: str, window_offset: int = 0) -> dict:
        return {
            "book_id": book_record.book_id,
            "title": book_record.title,
            "author": book_record.author,
            "text_window": text_window,
            "window_character_offset": window_offset,
        }
