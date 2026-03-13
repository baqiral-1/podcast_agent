"""Writing agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodePlan, EpisodeScript, RetrievalHit


class WritingAgent(Agent):
    """Agent that turns an episode plan into a cited script."""

    schema_name = "episode_script"
    instructions = "Write a grounded single-narrator podcast script with claim-level citations."
    response_model = EpisodeScript

    def __init__(self, llm, min_episode_minutes: int = 30, spoken_words_per_minute: int = 130) -> None:
        super().__init__(llm)
        self.min_episode_minutes = min_episode_minutes
        self.spoken_words_per_minute = spoken_words_per_minute

    def build_payload(self, episode_plan: EpisodePlan, retrieval_hits: list[RetrievalHit]) -> dict:
        return {
            "episode_plan": episode_plan.model_dump(mode="python"),
            "retrieval_hits": [hit.model_dump(mode="python") for hit in retrieval_hits],
            "target_episode_words": self.min_episode_minutes * self.spoken_words_per_minute,
        }

    def write(self, episode_plan: EpisodePlan, retrieval_hits: list[RetrievalHit]) -> EpisodeScript:
        """Generate an episode script."""

        return self.run(self.build_payload(episode_plan, retrieval_hits))
