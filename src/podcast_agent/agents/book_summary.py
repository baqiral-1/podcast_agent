"""Theme-conditioned book summary agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import book_summary_instructions


class BookSummaryResponse(BaseModel):
    summary: str = Field(default="")


class BookSummaryAgent(Agent):
    """Synthesizes a per-book summary used for theme decomposition."""

    schema_name = "book_summary"
    response_model = BookSummaryResponse
    instructions = book_summary_instructions()

    def build_payload(
        self,
        *,
        theme: str,
        sub_themes: list[str] | None,
        theme_elaboration: str | None,
        book_id: str,
        title: str,
        author: str,
        chapters: list[dict[str, Any]],
    ) -> dict:
        return {
            "theme": theme,
            "sub_themes": sub_themes or [],
            "theme_elaboration": theme_elaboration or "",
            "book_id": book_id,
            "title": title,
            "author": author,
            "chapters": chapters,
        }
