"""Analysis agent for theme and episode-cluster extraction."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookAnalysis, BookStructure


class AnalysisAgent(Agent):
    """Agent that extracts themes and candidate multi-chapter episode clusters."""

    schema_name = "book_analysis"
    instructions = "Analyze the book and propose multi-chapter episode clusters."
    response_model = BookAnalysis

    def build_payload(self, structure: BookStructure) -> dict:
        return {"structure": structure.model_dump(mode="python")}

    def analyze(self, structure: BookStructure) -> BookAnalysis:
        """Run book analysis."""

        return self.run(self.build_payload(structure))
