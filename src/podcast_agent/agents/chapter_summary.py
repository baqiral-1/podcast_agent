"""Stage 2b: Chapter summary agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import chapter_summary_instructions
from podcast_agent.schemas.models import ChapterAnalysis


class ChapterSummaryResponse(BaseModel):
    summary: str = Field(default="")
    analysis: ChapterAnalysis | None = None


class ChapterSummaryAgent(Agent):
    """Generates chapter summaries used by theme decomposition."""

    schema_name = "chapter_summary"
    response_model = ChapterSummaryResponse
    instructions = chapter_summary_instructions()

    def build_payload(
        self,
        *,
        theme: str,
        sub_themes: list[str] | None,
        theme_elaboration: str | None,
        book_id: str,
        title: str,
        author: str,
        chapter_title: str,
        chapter_text: str,
    ) -> dict:
        return {
            "theme": theme,
            "sub_themes": sub_themes or [],
            "theme_elaboration": theme_elaboration or "",
            "book_id": book_id,
            "title": title,
            "author": author,
            "chapter_title": chapter_title,
            "chapter_text": chapter_text,
        }
