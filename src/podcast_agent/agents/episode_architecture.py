"""Stage 9: episode architecture agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_architecture_instructions
from podcast_agent.schemas.models import EpisodeArchitecture


class EpisodeArchitectureAgent(Agent):
    """Turns one strategy episode into a binding section architecture."""

    schema_name = "episode_architecture"
    response_model = EpisodeArchitecture
    instructions = episode_architecture_instructions()

    def build_payload(
        self,
        episode: dict,
        synthesis_map: dict,
        project_metadata: dict,
        core_passages: list[dict],
        narrator_profile: dict | None = None,
        actor_metadata: dict | None = None,
        architecture_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "episode": episode,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "core_passages": core_passages,
        }
        if narrator_profile is not None:
            payload["narrator_profile"] = narrator_profile
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if architecture_feedback is not None:
            payload["architecture_feedback"] = architecture_feedback
        return payload
