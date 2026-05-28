"""Stage 9: episode architecture agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_architecture_instructions
from podcast_agent.schemas.models import (
    EpisodeArchitecture,
    PodcastMode,
    authorial_passage_target_range_for_mode,
    dense_section_authorial_passage_range_for_mode,
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
            return 11, 14
        return (
            int(project.get("architecture_section_target_min", 11)),
            int(project.get("architecture_section_target_max", 14)),
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
        section_target_min, section_target_max = self._section_target_bounds(payload)
        mode = self._podcast_mode(payload)
        authorial_target_min, authorial_target_max = (
            authorial_passage_target_range_for_mode(mode)
        )
        dense_section_min, dense_section_max = (
            dense_section_authorial_passage_range_for_mode(mode)
        )
        return episode_architecture_instructions(
            section_target_min=section_target_min,
            section_target_max=section_target_max,
            authorial_passage_target_min=authorial_target_min,
            authorial_passage_target_max=authorial_target_max,
            dense_section_authorial_passage_min=dense_section_min,
            dense_section_authorial_passage_max=dense_section_max,
            podcast_mode=mode,
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
        episode = payload.get("episode")
        promised_beats = []
        if isinstance(episode, dict):
            promised_beats = list(episode.get("promised_beats", []) or [])
        if promised_beats and len(result.promised_beat_decisions) != len(promised_beats):
            raise ValueError(
                "episode architecture must account for every promised beat exactly once"
            )
        return result

    def build_payload(
        self,
        episode: dict,
        synthesis_map: dict,
        project_metadata: dict,
        core_passages: list[dict],
        support_passages: list[dict],
        episode_scenes: list[dict] | None = None,
        series_explanation_registry: list[dict] | None = None,
        series_actor_explanation_registry: list[dict] | None = None,
        narrator_profile: dict | None = None,
        narrative_state: dict | None = None,
        actor_metadata: dict | None = None,
        architecture_feedback: dict | None = None,
        excerpts: list[dict] | None = None,
    ) -> dict:
        payload = {
            "episode": episode,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "core_passages": core_passages,
            "support_passages": support_passages,
        }
        if excerpts:
            payload["excerpts"] = excerpts
        if episode_scenes is not None:
            payload["episode_scenes"] = episode_scenes
        if narrator_profile is not None:
            payload["narrator_profile"] = narrator_profile
        if narrative_state is not None:
            payload["narrative_state"] = narrative_state
        if series_explanation_registry is not None:
            payload["series_explanation_registry"] = series_explanation_registry
        if series_actor_explanation_registry is not None:
            payload["series_actor_explanation_registry"] = (
                series_actor_explanation_registry
            )
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if architecture_feedback is not None:
            payload["architecture_feedback"] = architecture_feedback
        return payload
