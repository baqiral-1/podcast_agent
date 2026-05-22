"""Stage 8: scene discovery agent."""

from __future__ import annotations

import logging

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import scene_discovery_instructions
from podcast_agent.schemas.models import (
    PodcastMode,
    SceneDiscoveryArtifact,
    scene_discovery_candidate_range_for_mode,
)

logger = logging.getLogger(__name__)


class SceneDiscoveryAgent(Agent):
    """Discovers a shared series-wide pool of sceneable candidates."""

    schema_name = "scene_discovery"
    response_model = SceneDiscoveryArtifact
    instructions = scene_discovery_instructions()

    @staticmethod
    def _podcast_mode(payload: dict) -> PodcastMode:
        project = payload.get("project")
        if not isinstance(project, dict):
            return PodcastMode.FULL
        raw_mode = project.get("podcast_mode", PodcastMode.FULL.value)
        try:
            return PodcastMode(raw_mode)
        except ValueError:
            return PodcastMode.FULL

    def validate_result(
        self, result: SceneDiscoveryArtifact, payload: dict
    ) -> SceneDiscoveryArtifact:
        mode = self._podcast_mode(payload)
        target_min, target_max = scene_discovery_candidate_range_for_mode(mode)
        candidate_count = len(result.candidates)
        if candidate_count < target_min or candidate_count > target_max:
            warning = (
                "scene_discovery_candidate_count_out_of_range: "
                f"{candidate_count} (target {target_min}-{target_max})"
            )
            logger.warning("%s", warning)
            run_logger = getattr(self.llm, "run_logger", None)
            if run_logger is not None:
                run_logger.log(
                    "scene_discovery_warning",
                    warning=warning,
                    candidate_count=candidate_count,
                    target_min=target_min,
                    target_max=target_max,
                )
        return result

    def build_instructions(self, payload: dict) -> str:
        mode = self._podcast_mode(payload)
        target_min, target_max = scene_discovery_candidate_range_for_mode(mode)
        return scene_discovery_instructions(
            candidate_target_min=target_min,
            candidate_target_max=target_max,
            podcast_mode=mode,
        )

    def build_payload(
        self,
        synthesis_map: dict,
        project_metadata: dict,
        actor_metadata: dict | None,
        passage_list: list[dict],
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "passage_list": passage_list,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        return payload
