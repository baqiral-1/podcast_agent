"""Episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookAnalysis, BookStructure, SeriesPlan


class EpisodePlanningAgent(Agent):
    """Agent that turns analysis into a hierarchical series plan."""

    schema_name = "series_plan"
    instructions = "Create a hierarchical series and episode plan for a single-narrator podcast."
    response_model = SeriesPlan

    def __init__(self, llm, min_episode_minutes: int = 30, spoken_words_per_minute: int = 130) -> None:
        super().__init__(llm)
        self.min_episode_minutes = min_episode_minutes
        self.spoken_words_per_minute = spoken_words_per_minute

    def build_payload(self, structure: BookStructure, analysis: BookAnalysis) -> dict:
        return {
            "structure": structure.model_dump(mode="python"),
            "analysis": analysis.model_dump(mode="python"),
            "target_episode_words": self.min_episode_minutes * self.spoken_words_per_minute,
        }

    def plan(self, structure: BookStructure, analysis: BookAnalysis) -> SeriesPlan:
        """Run episode planning."""

        return self.run(self.build_payload(structure, analysis))
