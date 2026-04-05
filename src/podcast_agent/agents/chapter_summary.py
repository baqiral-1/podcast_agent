"""Stage 2b: Chapter summary agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import ChapterAnalysis


class ChapterSummaryResponse(BaseModel):
    summary: str = Field(default="")
    analysis: ChapterAnalysis | None = None


class ChapterSummaryAgent(Agent):
    """Generates chapter summaries used by theme decomposition."""

    schema_name = "chapter_summary"
    response_model = ChapterSummaryResponse
    instructions = (
        "You are a book analyst. Given a chapter of a book, produce a concise 4-6 sentence "
        "summary focused on thematic analysis. Capture key events or arguments, the central "
        "actors or institutions, major tensions/disagreements, and any meaningful causal shifts.\n\n"
        "Also return structured chapter analysis for downstream planning and retrieval. "
        "Populate only details clearly supported by the chapter text, keep lists tight, and "
        "prefer concrete historical terms over abstract labels.\n\n"
        "Return a JSON object with 'summary' and optional 'analysis' fields."
    )

    def build_payload(self, *, book_id: str, title: str, author: str, chapter_title: str, chapter_text: str) -> dict:
        return {
            "book_id": book_id,
            "title": title,
            "author": author,
            "chapter_title": chapter_title,
            "chapter_text": chapter_text,
        }
