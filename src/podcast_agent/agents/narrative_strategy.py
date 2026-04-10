"""Stage 8: narrative strategy agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import narrative_strategy_instructions
from podcast_agent.schemas.models import NarrativeStrategy


class NarrativeStrategyAgent(Agent):
    """Chooses the macro-level series structure from cluster-first synthesis."""

    schema_name = "narrative_strategy"
    response_model = NarrativeStrategy
    instructions = narrative_strategy_instructions()

    def build_payload(
        self,
        synthesis_map: dict,
        thematic_axes: list[dict],
        project_metadata: dict,
        episode_count: int | None,
        strategy_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "thematic_axes": thematic_axes,
            "project": project_metadata,
        }
        if episode_count is not None:
            payload["requested_episode_count"] = episode_count
        if strategy_feedback is not None:
            payload["strategy_feedback"] = strategy_feedback
        return payload
