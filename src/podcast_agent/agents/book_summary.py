"""Theme-conditioned book summary agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent


class BookSummaryResponse(BaseModel):
    summary: str = Field(default="")


class BookSummaryAgent(Agent):
    """Synthesizes a per-book summary used for theme decomposition."""

    schema_name = "book_summary"
    response_model = BookSummaryResponse
    instructions = (
        "You are a thematic analyst preparing a cross-book podcast. Given the project theme "
        "plus optional sub-themes "
        "and chapter-level summaries for a single book, write one concise book summary for "
        "axis discovery.\n\n"
        "Focus on the material most relevant to the theme, highlighting recurring patterns, "
        "major tensions, and arguments or events likely to matter across books. Use the "
        "chapter summaries as evidence. Keep the summary tight and information-dense.\n\n"
        "Return a JSON object with a 'summary' field."
    )

    def build_payload(
        self,
        *,
        theme: str,
        sub_themes: list[str] | None,
        theme_elaboration: str | None,
        book_id: str,
        title: str,
        author: str,
        chapters: list[dict[str, str]],
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
