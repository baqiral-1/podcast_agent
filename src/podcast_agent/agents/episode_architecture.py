"""Stage 9: episode architecture agent."""

from __future__ import annotations

from pydantic import ValidationError

from podcast_agent.agents._architecture_feedback import (
    INNER_RETRY_INSTRUCTION_HEADER,
    build_section_context_from_architecture_payload,
    format_validation_errors,
    layer_in_episode_support_passages,
    shape_specific_appendix,
)
from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import episode_architecture_instructions
from podcast_agent.schemas.models import (
    EpisodeArchitecture,
    PodcastMode,
    authorial_passage_target_range_for_mode,
    dense_section_authorial_passage_range_for_mode,
    validate_density_budget,
    validate_episode_architecture_targets,
    validate_episode_runtime_envelope,
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
            return 12, 16
        return (
            int(project.get("architecture_section_target_min", 12)),
            int(project.get("architecture_section_target_max", 16)),
        )

    @staticmethod
    def _density_bounds(payload: dict) -> dict[str, float | int]:
        project = payload.get("project") or {}
        if not isinstance(project, dict):
            project = {}
        return {
            "max_dense_sections": int(project.get("max_dense_sections_per_episode", 2)),
            "dense_min": float(project.get("dense_section_runtime_min_minutes", 16.0)),
            "dense_max": float(project.get("dense_section_runtime_max_minutes", 22.0)),
            "section_floor": float(project.get("section_runtime_floor_minutes", 6.0)),
            "section_ceiling": float(project.get("section_runtime_ceiling_minutes", 14.0)),
            "episode_min": float(project.get("min_episode_minutes", 124.0)),
            "episode_max": float(project.get("max_episode_minutes", 145.0)),
            "policy": str(project.get("series_runtime_policy", "warn")),
        }

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
        authorial_target_min, authorial_target_max = authorial_passage_target_range_for_mode(mode)
        dense_section_min, dense_section_max = dense_section_authorial_passage_range_for_mode(mode)
        density_bounds = self._density_bounds(payload)
        return episode_architecture_instructions(
            section_target_min=section_target_min,
            section_target_max=section_target_max,
            authorial_passage_target_min=authorial_target_min,
            authorial_passage_target_max=authorial_target_max,
            dense_section_authorial_passage_min=dense_section_min,
            dense_section_authorial_passage_max=dense_section_max,
            podcast_mode=mode,
            max_dense_sections_per_episode=int(density_bounds["max_dense_sections"]),
            dense_section_runtime_min=float(density_bounds["dense_min"]),
            dense_section_runtime_max=float(density_bounds["dense_max"]),
            section_runtime_floor=float(density_bounds["section_floor"]),
            section_runtime_ceiling=float(density_bounds["section_ceiling"]),
            target_episode_runtime_min=float(density_bounds["episode_min"]),
            target_episode_runtime_max=float(density_bounds["episode_max"]),
        )

    def validate_result(self, result: EpisodeArchitecture, payload: dict) -> EpisodeArchitecture:
        section_target_min, section_target_max = self._section_target_bounds(payload)
        validate_episode_architecture_targets(
            result,
            section_target_min=section_target_min,
            section_target_max=section_target_max,
        )
        bounds = self._density_bounds(payload)
        # Density budget enforced here so the agent retry harness can feed
        # violations back as architecture_feedback. validate_density_budget
        # never raises; we convert violations to ValueError so the orchestrator
        # sees them as a retryable compliance error.
        density_warnings = validate_density_budget(
            result,
            max_dense_sections=int(bounds["max_dense_sections"]),
            dense_runtime_range=(
                float(bounds["dense_min"]),
                float(bounds["dense_max"]),
            ),
            nondense_runtime_ceiling=float(bounds["section_ceiling"]),
            section_runtime_floor=float(bounds["section_floor"]),
        )
        if density_warnings and bounds["policy"] == "enforce":
            raise ValueError("architecture density violations: " + "; ".join(density_warnings))
        # Episode runtime envelope. Same enforce/warn split.
        validate_episode_runtime_envelope(
            result,
            min_episode_minutes=float(bounds["episode_min"]),
            max_episode_minutes=float(bounds["episode_max"]),
            policy="enforce" if bounds["policy"] == "enforce" else "warn",
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

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        """Inject the previous attempt's validation failure into the next call.

        The inner LLM retry loop (Agent.run) calls this before each retry on
        a RetryableGenerationError. Without this override the same prompt
        would be resent verbatim and the same violation would recur — which
        is exactly what happened on episode 11 of iranian_revolution_v74,
        where three identical peripheral_touch grounding failures burned in
        a row before the orchestrator's outer retry took over.

        We write into the same ``architecture_feedback`` key the
        orchestrator's outer retry uses, so the prompt only describes one
        feedback channel. ``source=inner_retry`` is purely diagnostic.
        """
        cause = exc.__cause__
        if isinstance(cause, ValidationError):
            data = getattr(exc, "data", None) or {}
            raw_payload = data.get("raw_payload")
            section_context, sections_by_index = (
                build_section_context_from_architecture_payload(raw_payload)
                if isinstance(raw_payload, dict)
                else ({}, {})
            )
            support_passage_ids = [
                str(item.get("passage_id"))
                for item in payload.get("support_passages", []) or []
                if isinstance(item, dict) and item.get("passage_id")
            ]
            if section_context:
                layer_in_episode_support_passages(section_context, support_passage_ids)
            instruction = (
                INNER_RETRY_INSTRUCTION_HEADER
                + format_validation_errors(cause)
                + shape_specific_appendix(
                    cause,
                    section_context=section_context or None,
                    sections_by_index=sections_by_index or None,
                )
            )
            feedback: dict = {
                "issue": "schema_validation_failed",
                "source": "inner_retry",
                "instruction": instruction,
            }
        else:
            # JSON parse error or other RetryableGenerationError without a
            # wrapped Pydantic cause. Surface parse coordinates if present
            # so the model knows its prior output was structurally invalid.
            data = getattr(exc, "data", None) or {}
            instruction = (
                "The previous attempt's response was not valid JSON for the "
                "episode_architecture schema. Re-emit a complete, well-formed JSON "
                "object — no truncation, no trailing commentary, no markdown fences. "
                f"Underlying parser error: {exc}"
            )
            feedback = {
                "issue": "generation_unparseable",
                "source": "inner_retry",
                "instruction": instruction,
            }
            if data.get("parse_error_line") is not None:
                feedback["parse_error_line"] = data["parse_error_line"]
                feedback["parse_error_column"] = data["parse_error_column"]
        next_payload = dict(payload)
        next_payload["architecture_feedback"] = feedback
        return next_payload

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
            payload["series_actor_explanation_registry"] = series_actor_explanation_registry
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if architecture_feedback is not None:
            payload["architecture_feedback"] = architecture_feedback
        return payload
