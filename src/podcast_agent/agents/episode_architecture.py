"""Stage 9: episode architecture agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_architecture_instructions
from podcast_agent.schemas.models import (
    EpisodeArchitecture,
    validate_episode_architecture_targets,
)


class EpisodeArchitectureAgent(Agent):
    """Turns one strategy episode into a binding section architecture."""

    schema_name = "episode_architecture"
    response_model = EpisodeArchitecture
    instructions = episode_architecture_instructions()

    @staticmethod
    def _section_target_bounds(payload: dict) -> tuple[int, int]:
        project = payload.get("project")
        if not isinstance(project, dict):
            return 9, 12
        return (
            int(project.get("architecture_section_target_min", 9)),
            int(project.get("architecture_section_target_max", 12)),
        )

    def build_instructions(self, payload: dict) -> str:
        section_target_min, section_target_max = self._section_target_bounds(payload)
        return episode_architecture_instructions(
            section_target_min=section_target_min,
            section_target_max=section_target_max,
        )

    def validate_result(
        self, result: EpisodeArchitecture, payload: dict
    ) -> EpisodeArchitecture:
        section_target_min, section_target_max = self._section_target_bounds(payload)
        validate_episode_architecture_targets(
            result,
            section_target_min=section_target_min,
            section_target_max=section_target_max,
        )
        return result

    def build_payload(
        self,
        episode: dict,
        synthesis_map: dict,
        project_metadata: dict,
        core_passages: list[dict],
        series_explanation_registry: list[dict] | None = None,
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
        if series_explanation_registry is not None:
            payload["series_explanation_registry"] = series_explanation_registry
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if architecture_feedback is not None:
            payload["architecture_feedback"] = architecture_feedback
        return payload
