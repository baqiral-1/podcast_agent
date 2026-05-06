"""Stage 8: narrative strategy agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import narrative_strategy_instructions
from podcast_agent.schemas.models import (
    NarrativeStrategy,
    validate_episode_spine_targets,
)


class NarrativeStrategyAgent(Agent):
    """Chooses the macro-level series structure from evidence-pack synthesis."""

    schema_name = "narrative_strategy"
    response_model = NarrativeStrategy
    instructions = narrative_strategy_instructions()

    @staticmethod
    def _primitive_target_bounds(payload: dict) -> tuple[int, int, int, int, int]:
        project = payload.get("project")
        if not isinstance(project, dict):
            return 5, 7, 5, 7, 2
        return (
            int(project.get("episode_spine_core_primitive_target_min", 5)),
            int(project.get("episode_spine_core_primitive_target_max", 7)),
            int(project.get("episode_spine_support_primitive_target_min", 5)),
            int(project.get("episode_spine_support_primitive_target_max", 7)),
            int(project.get("episode_spine_recall_primitive_target_max", 2)),
        )

    def build_instructions(self, payload: dict) -> str:
        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
        ) = self._primitive_target_bounds(payload)
        return narrative_strategy_instructions(
            core_primitive_target_min=core_target_min,
            core_primitive_target_max=core_target_max,
            support_primitive_target_min=support_target_min,
            support_primitive_target_max=support_target_max,
            recall_primitive_target_max=recall_target_max,
        )

    def validate_result(self, result: NarrativeStrategy, payload: dict) -> NarrativeStrategy:
        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
        ) = self._primitive_target_bounds(payload)
        for episode in result.episodes:
            validate_episode_spine_targets(
                episode.episode_spine,
                core_target_min=core_target_min,
                core_target_max=core_target_max,
                support_target_min=support_target_min,
                support_target_max=support_target_max,
                recall_target_max=recall_target_max,
            )
        return result

    def build_payload(
        self,
        synthesis_map: dict,
        project_metadata: dict,
        episode_count: int | None,
        recommended_episode_count_min: int,
        recommended_episode_count_max: int,
        actor_metadata: dict | None = None,
        strategy_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "recommended_episode_count_min": recommended_episode_count_min,
            "recommended_episode_count_max": recommended_episode_count_max,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if episode_count is not None:
            payload["requested_episode_count"] = episode_count
        if strategy_feedback is not None:
            payload["strategy_feedback"] = strategy_feedback
        return payload
