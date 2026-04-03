"""Stage 14: Episode framing agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodeFraming


class EpisodeFramingAgent(Agent):
    """Generates recap and preview bridges that connect episodes into a series."""

    schema_name = "episode_framing"
    response_model = EpisodeFraming
    instructions = (
        "You are a podcast framing specialist for a multi-book synthesis series. "
        "Generate episode bridges that connect episodes.\n\n"
        "Produce:\n"
        "- recap (episodes 2+): A 30-60 second opening that reminds the listener what was "
        "covered previously. For multi-book podcasts, reference cross-book connections: "
        "'Last time, we explored how [Author A] and [Author B] approach [topic] from "
        "opposite angles. Today, [Author C] adds a twist neither of them anticipated.'\n"
        "- preview (all except last): A 15-30 second closing tease for the next episode. "
        "Create anticipation without spoiling.\n"
        "- cold_open (optional): A provocative quote or question before the recap to hook "
        "the listener immediately.\n\n"
        "Set recap to null for episode 1. Set preview to null for the last episode.\n\n"
        "Return a JSON EpisodeFraming object."
    )

    def build_payload(
        self,
        episode_number: int,
        total_episodes: int,
        current_episode_summary: str,
        previous_episode_summary: str | None,
        next_episode_summary: str | None,
        book_metadata: list[dict],
    ) -> dict:
        return {
            "episode_number": episode_number,
            "total_episodes": total_episodes,
            "current_episode": current_episode_summary,
            "previous_episode": previous_episode_summary,
            "next_episode": next_episode_summary,
            "books": book_metadata,
        }
