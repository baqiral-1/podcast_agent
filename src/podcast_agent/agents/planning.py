"""Stage 9: episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_planning_instructions
from podcast_agent.schemas.models import EpisodePlanDraft


class EpisodePlanningAgent(Agent):
    """Expands one episode cluster path into framing plus scene cards."""

    schema_name = "episode_planning"
    response_model = EpisodePlanDraft
    instructions = episode_planning_instructions()

    def build_instructions(self, payload: dict) -> str:
        project = payload.get("project")
        if not isinstance(project, dict):
            return self.instructions
        return episode_planning_instructions(
            scene_card_target_min=int(project.get("scene_card_target_min", 32)),
            scene_card_target_max=int(project.get("scene_card_target_max", 40)),
        )

    def build_payload(
        self,
        strategy_episode: dict,
        architecture: dict,
        synthesis_map: dict,
        project_metadata: dict,
        available_passages: list[dict],
        host_policy: dict | None = None,
        actor_metadata: dict | None = None,
        planning_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "strategy_episode": strategy_episode,
            "architecture": architecture,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "available_passages": available_passages,
        }
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if planning_feedback is not None:
            payload["planning_feedback"] = planning_feedback
        return payload
