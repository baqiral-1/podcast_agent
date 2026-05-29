"""Stage 8b: narrative strategy enrichment agent."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import narrative_strategy_enrichment_instructions
from podcast_agent.schemas.models import (
    NarrativeStrategyEnrichment,
    PodcastMode,
    PromisedBeat,
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
        authorial_target_min, authorial_target_max = authorial_passage_target_range_for_mode(mode)
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
                        intended_job=SceneJob.ANSWER,
                        source_primitive_ids=[fallback_primitive_id],
                        why_load_bearing="Backfilled during enrichment validation.",
                    )
                ]
        narrator_profile = result.narrator_profile
        if "target_authorial_passages_per_episode" not in narrator_profile.model_fields_set:
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

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        feedback = self._build_retry_feedback(exc)
        if feedback is None:
            return payload
        next_payload = dict(payload)
        next_payload["strategy_enrichment_feedback"] = feedback
        return next_payload

    def _build_retry_feedback(
        self,
        exc: RetryableGenerationError,
    ) -> dict[str, Any] | None:
        validation_exc = exc.__cause__
        raw_payload = exc.data.get("raw_payload")
        if not isinstance(validation_exc, ValidationError) or not isinstance(raw_payload, dict):
            return None

        episode_constraints: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        episodes = raw_payload.get("episodes")
        if not isinstance(episodes, list):
            return None

        for episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            episode_number = episode.get("episode_number", episode_index + 1)
            issue_types: list[str] = []
            required_fixes: list[str] = []

            host = (
                (episode.get("narrative_agenda") or {}).get("host")
                if isinstance(episode.get("narrative_agenda"), dict)
                else {}
            )
            if isinstance(host, dict):
                assumption_moves = host.get("assumption_moves")
                if isinstance(assumption_moves, list):
                    for move_index, move in enumerate(assumption_moves):
                        if not isinstance(move, dict):
                            continue
                        action = str(move.get("action", "") or "").strip()
                        statement = str(move.get("statement", "") or "").strip()
                        revised_statement = str(move.get("revised_statement", "") or "").strip()
                        assumption_id = str(move.get("assumption_id", "") or "").strip()
                        if action == "introduce" and not statement:
                            issue = {
                                "issue": "host_assumption_introduce_missing_statement",
                                "episode_number": episode_number,
                                "episode_index": episode_index,
                                "move_index": move_index,
                                "assumption_id": assumption_id,
                                "required_fix": "Include `statement` for assumption_moves.introduce.",
                            }
                            issues.append(issue)
                            issue_types.append(issue["issue"])
                            required_fixes.append(issue["required_fix"])
                        if action == "revise":
                            if not statement:
                                issue = {
                                    "issue": "host_assumption_revise_missing_statement",
                                    "episode_number": episode_number,
                                    "episode_index": episode_index,
                                    "move_index": move_index,
                                    "assumption_id": assumption_id,
                                    "required_fix": "Include both `statement` and `revised_statement` for assumption_moves.revise.",
                                }
                                issues.append(issue)
                                issue_types.append(issue["issue"])
                                required_fixes.append(issue["required_fix"])
                            elif not revised_statement:
                                issue = {
                                    "issue": "host_assumption_revise_missing_revised_statement",
                                    "episode_number": episode_number,
                                    "episode_index": episode_index,
                                    "move_index": move_index,
                                    "assumption_id": assumption_id,
                                    "required_fix": "Include both `statement` and `revised_statement` for assumption_moves.revise.",
                                }
                                issues.append(issue)
                                issue_types.append(issue["issue"])
                                required_fixes.append(issue["required_fix"])

            promised_beats = episode.get("promised_beats")
            if isinstance(promised_beats, list):
                for beat_index, beat in enumerate(promised_beats):
                    if not isinstance(beat, dict) or "kind" not in beat:
                        continue
                    beat_id = str(beat.get("beat_id", "") or "").strip()
                    issue = {
                        "issue": "forbidden_promised_beat_kind_field",
                        "episode_number": episode_number,
                        "episode_index": episode_index,
                        "beat_index": beat_index,
                        "beat_id": beat_id,
                        "required_fix": "Remove `kind`; promised beats now use only `intended_job` plus sources and `why_load_bearing`.",
                    }
                    issues.append(issue)
                    issue_types.append(issue["issue"])
                    required_fixes.append(issue["required_fix"])

            if issue_types:
                episode_constraints.append(
                    {
                        "episode_number": episode_number,
                        "episode_index": episode_index,
                        "issue_types": issue_types,
                        "required_fix": " ".join(dict.fromkeys(required_fixes)),
                    }
                )

        if not issues:
            return None

        return {
            "issue": "schema_validation_failed",
            "issues": issues,
            "episode_constraints_by_number": episode_constraints,
            "canonical_field_names": {
                "scene_jobs": "scene_jobs",
                "promised_beats[].intended_job": "intended_job",
                "promised_beats[].kind": "remove this field",
                "narrative_agenda.host.assumption_moves[].statement": "statement",
                "narrative_agenda.host.assumption_moves[].revised_statement": "revised_statement",
            },
            "instruction": (
                "Revise only the invalid episodes or invalid items. Keep unaffected "
                "content unchanged. For `assumption_moves.revise`, include both "
                "`statement` and `revised_statement`. Do not emit promised-beat "
                "`kind`; use only `intended_job`, source ids, and `why_load_bearing`."
            ),
        }

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
