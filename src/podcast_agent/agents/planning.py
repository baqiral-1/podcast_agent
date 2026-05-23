"""Stage 9: episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_planning_instructions
from podcast_agent.schemas.models import (
    EpisodePlanDraft,
    SceneJob,
    scene_job_budget_for_mode,
)


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
            scene_card_target_min=int(project.get("scene_card_target_min", 30)),
            scene_card_target_max=int(project.get("scene_card_target_max", 36)),
        )

    def validate_result(self, result: EpisodePlanDraft, payload: dict) -> EpisodePlanDraft:
        if not result.answer_scene_card_id:
            raise ValueError("episode plan must include answer_scene_card_id")
        if not result.residue_scene_card_id:
            raise ValueError("episode plan must include residue_scene_card_id")
        close_scene_count = sum(
            1 for scene in result.scene_cards if scene.scene_job == SceneJob.CLOSE
        )
        if close_scene_count != 1:
            raise ValueError("episode plan must include exactly one close scene")
        for scene in result.scene_cards:
            for phase in ("open", "pivot", "close"):
                if len(getattr(scene.host_moves, phase)) > 1:
                    raise ValueError(
                        "fresh episode plans must use at most one host cue per phase"
                    )
        return result

    def build_payload(
        self,
        strategy_episode: dict,
        architecture: dict,
        synthesis_map: dict,
        project_metadata: dict,
        scene_job_budget: dict | None,
        available_passages: list[dict],
        host_policy: dict | None = None,
        actor_metadata: dict | None = None,
        planning_feedback: dict | None = None,
        field_semantics: dict | None = None,
    ) -> dict:
        payload = {
            "strategy_episode": strategy_episode,
            "architecture": architecture,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "available_passages": available_passages,
        }
        if scene_job_budget is None:
            raw_mode = project_metadata.get("podcast_mode", "full")
            scene_job_budget = scene_job_budget_for_mode(raw_mode)
        payload["scene_job_budget"] = scene_job_budget
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if planning_feedback is not None:
            payload["planning_feedback"] = planning_feedback
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        return payload
