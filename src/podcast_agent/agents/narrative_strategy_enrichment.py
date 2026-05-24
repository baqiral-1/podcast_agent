"""Stage 8b: narrative strategy enrichment agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import narrative_strategy_enrichment_instructions
from podcast_agent.schemas.models import (
    NarrativeStrategyEnrichment,
    PodcastMode,
    PromisedBeat,
    PromisedBeatKind,
    SceneJob,
    authorial_passage_target_for_mode,
    authorial_passage_target_range_for_mode,
)


class NarrativeStrategyEnrichmentAgent(Agent):
    """Adds narrator, registry, agenda, and promised-beat layers to a fixed skeleton."""

    schema_name = "narrative_strategy_enrichment"
    response_model = NarrativeStrategyEnrichment
    instructions = narrative_strategy_enrichment_instructions()

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
        mode = self._podcast_mode(payload)
        authorial_target_min, authorial_target_max = (
            authorial_passage_target_range_for_mode(mode)
        )
        return narrative_strategy_enrichment_instructions(
            authorial_passage_target_min=authorial_target_min,
            authorial_passage_target_max=authorial_target_max,
            podcast_mode=mode,
        )

    def validate_result(
        self, result: NarrativeStrategyEnrichment, payload: dict
    ) -> NarrativeStrategyEnrichment:
        mode = self._podcast_mode(payload)
        strategy_skeleton = payload.get("strategy_skeleton")
        skeleton_episodes = []
        if isinstance(strategy_skeleton, dict):
            skeleton_episodes = list(strategy_skeleton.get("episodes", []) or [])
        expected_numbers = sorted(
            int(episode.get("episode_number"))
            for episode in skeleton_episodes
            if isinstance(episode, dict) and isinstance(episode.get("episode_number"), int)
        )
        actual_numbers = sorted(episode.episode_number for episode in result.episodes)
        if expected_numbers and actual_numbers != expected_numbers:
            raise ValueError(
                "narrative_strategy_enrichment episode numbers must match strategy_skeleton"
            )
        for episode in result.episodes:
            if not episode.promised_beats:
                fallback_primitive_id = ""
                for skeleton_episode in skeleton_episodes:
                    if (
                        isinstance(skeleton_episode, dict)
                        and skeleton_episode.get("episode_number") == episode.episode_number
                    ):
                        spine = skeleton_episode.get("episode_spine", {}) or {}
                        if isinstance(spine, dict):
                            core_ids = list(spine.get("core_primitive_ids", []) or [])
                            if core_ids:
                                fallback_primitive_id = str(core_ids[0])
                        break
                if not fallback_primitive_id:
                    fallback_primitive_id = f"episode_{episode.episode_number}_core"
                episode.promised_beats = [
                    PromisedBeat(
                        beat_id=f"episode_{episode.episode_number}_answer",
                        label=f"Episode {episode.episode_number} answer commitment",
                        kind=PromisedBeatKind.MECHANISM,
                        intended_job=SceneJob.ANSWER,
                        source_primitive_ids=[fallback_primitive_id],
                        why_load_bearing="Backfilled during enrichment validation.",
                    )
                ]
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
        strategy_skeleton: dict,
        synthesis_map: dict,
        project_metadata: dict,
        episode_scene_candidates: list[dict],
        actor_metadata: dict | None = None,
        strategy_enrichment_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "strategy_skeleton": strategy_skeleton,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "episode_scene_candidates": episode_scene_candidates,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if strategy_enrichment_feedback is not None:
            payload["strategy_enrichment_feedback"] = strategy_enrichment_feedback
        return payload
