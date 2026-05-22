"""Stage 8: narrative strategy agent."""

from __future__ import annotations

import logging

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import narrative_strategy_instructions
from podcast_agent.schemas.models import (
    NarrativeStrategy,
    PodcastMode,
    authorial_passage_target_for_mode,
    authorial_passage_target_range_for_mode,
    validate_episode_spine_targets,
)

logger = logging.getLogger(__name__)


class NarrativeStrategyAgent(Agent):
    """Chooses the macro-level series structure from evidence-pack synthesis."""

    schema_name = "narrative_strategy"
    response_model = NarrativeStrategy
    instructions = narrative_strategy_instructions()

    @staticmethod
    def _primitive_target_bounds(payload: dict) -> tuple[int, int, int, int, int]:
        project = payload.get("project")
        if not isinstance(project, dict):
            return 6, 8, 7, 10, 2
        return (
            int(project.get("episode_spine_core_primitive_target_min", 6)),
            int(project.get("episode_spine_core_primitive_target_max", 8)),
            int(project.get("episode_spine_support_primitive_target_min", 7)),
            int(project.get("episode_spine_support_primitive_target_max", 10)),
            int(project.get("episode_spine_recall_primitive_target_max", 2)),
        )

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

    def build_instructions(self, payload: dict) -> str:
        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
        ) = self._primitive_target_bounds(payload)
        mode = self._podcast_mode(payload)
        authorial_target_min, authorial_target_max = (
            authorial_passage_target_range_for_mode(mode)
        )
        return narrative_strategy_instructions(
            core_primitive_target_min=core_target_min,
            core_primitive_target_max=core_target_max,
            support_primitive_target_min=support_target_min,
            support_primitive_target_max=support_target_max,
            recall_primitive_target_max=recall_target_max,
            authorial_passage_target_min=authorial_target_min,
            authorial_passage_target_max=authorial_target_max,
            podcast_mode=mode,
        )

    def validate_result(self, result: NarrativeStrategy, payload: dict) -> NarrativeStrategy:
        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
        ) = self._primitive_target_bounds(payload)
        mode = self._podcast_mode(payload)
        synthesis_map = payload.get("synthesis_map")
        primitive_by_id: dict[str, dict] = {}
        if isinstance(synthesis_map, dict):
            for primitive in synthesis_map.get("primitives", []) or []:
                if not isinstance(primitive, dict):
                    continue
                primitive_id = str(primitive.get("id", "") or "").strip()
                if primitive_id:
                    primitive_by_id[primitive_id] = primitive
        for episode in result.episodes:
            validate_episode_spine_targets(
                episode.episode_spine,
                core_target_min=core_target_min,
                core_target_max=core_target_max,
                support_target_min=support_target_min,
                support_target_max=support_target_max,
                recall_target_max=recall_target_max,
            )
            if primitive_by_id:
                if not any(
                    primitive_by_id.get(primitive_id, {}).get("substrate")
                    in {"events", "acts"}
                    for primitive_id in episode.episode_spine.core_primitive_ids
                ):
                    warning = (
                        "narrative_strategy_core_missing_event_or_act: "
                        "each episode core should include at least one events or acts primitive"
                    )
                    logger.warning("%s episode=%s", warning, episode.episode_number)
                    run_logger = getattr(self.llm, "run_logger", None)
                    if run_logger is not None:
                        run_logger.log(
                            "narrative_strategy_warning",
                            warning=warning,
                            episode_number=episode.episode_number,
                        )
                for primitive_id in episode.episode_spine.recall_primitive_ids:
                    functions = primitive_by_id.get(primitive_id, {}).get("functions", [])
                    if "recurrence" not in list(functions or []):
                        warning = (
                            "narrative_strategy_recall_missing_recurrence: "
                            "recall_primitive_ids should reference recurrence-tagged primitives "
                            f"(primitive_id={primitive_id})"
                        )
                        logger.warning("%s episode=%s", warning, episode.episode_number)
                        run_logger = getattr(self.llm, "run_logger", None)
                        if run_logger is not None:
                            run_logger.log(
                                "narrative_strategy_warning",
                                warning=warning,
                                episode_number=episode.episode_number,
                                primitive_id=primitive_id,
                            )
        narrator_profile = result.narrator_profile
        if (
            "target_authorial_passages_per_episode"
            not in narrator_profile.model_fields_set
        ):
            result = result.model_copy(
                update={
                    "narrator_profile": narrator_profile.model_copy(
                        update={
                            "target_authorial_passages_per_episode": (
                                authorial_passage_target_for_mode(mode)
                            )
                        }
                    )
                }
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
