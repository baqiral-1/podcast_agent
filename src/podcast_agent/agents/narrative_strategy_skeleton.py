"""Stage 8a: narrative strategy skeleton agent."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import narrative_strategy_skeleton_instructions
from podcast_agent.schemas.models import (
    NarrativeStrategySkeleton,
    PodcastMode,
    validate_episode_spine_targets,
)

logger = logging.getLogger(__name__)
_EPISODE_SCOPE_PATH_RE = re.compile(r"episodes\.(\d+)\.scope\b")


class NarrativeStrategySkeletonAgent(Agent):
    """Chooses the macro series partition before narrator/enrichment details."""

    schema_name = "narrative_strategy_skeleton"
    response_model = NarrativeStrategySkeleton
    instructions = narrative_strategy_skeleton_instructions()

    @staticmethod
    def _primitive_target_bounds(
        payload: dict,
    ) -> tuple[int, int, int, int, int, int, int]:
        project = payload.get("project")
        if not isinstance(project, dict):
            return 6, 10, 8, 12, 2, 12, 20
        return (
            int(project.get("episode_spine_core_primitive_target_min", 6)),
            int(project.get("episode_spine_core_primitive_target_max", 10)),
            int(project.get("episode_spine_support_primitive_target_min", 8)),
            int(project.get("episode_spine_support_primitive_target_max", 12)),
            int(project.get("episode_spine_recall_primitive_target_max", 2)),
            int(project.get("episode_spine_excerpt_target_min", 12)),
            int(project.get("episode_spine_excerpt_target_max", 20)),
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
            excerpt_target_min,
            excerpt_target_max,
        ) = self._primitive_target_bounds(payload)
        return narrative_strategy_skeleton_instructions(
            core_primitive_target_min=core_target_min,
            core_primitive_target_max=core_target_max,
            support_primitive_target_min=support_target_min,
            support_primitive_target_max=support_target_max,
            recall_primitive_target_max=recall_target_max,
            excerpt_target_min=excerpt_target_min,
            excerpt_target_max=excerpt_target_max,
        )

    def validate_result(
        self, result: NarrativeStrategySkeleton, payload: dict
    ) -> NarrativeStrategySkeleton:
        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
            excerpt_target_min,
            excerpt_target_max,
        ) = self._primitive_target_bounds(payload)
        project = payload.get("project")
        has_project_context = isinstance(project, dict)
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
            if (
                not episode.negative_scope.boundary.strip()
                or not episode.negative_scope.omission_logic.strip()
            ):
                if not has_project_context:
                    raise ValueError(
                        f"strategy skeleton episode {episode.episode_number} must include negative_scope boundary and omission_logic"
                    )
                episode.negative_scope = episode.negative_scope.model_copy(
                    update={
                        "boundary": episode.negative_scope.boundary
                        or "Stay inside the episode's declared listener problem.",
                        "omission_logic": episode.negative_scope.omission_logic
                        or "Leave neighboring material out unless it directly advances the episode answer.",
                    }
                )
            validate_episode_spine_targets(
                episode.episode_spine,
                core_target_min=core_target_min,
                core_target_max=core_target_max,
                support_target_min=support_target_min,
                support_target_max=support_target_max,
                recall_target_max=recall_target_max,
                excerpt_target_min=excerpt_target_min,
                excerpt_target_max=excerpt_target_max,
            )
            if primitive_by_id:
                if not any(
                    primitive_by_id.get(primitive_id, {}).get("substrate") in {"events", "acts"}
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
        return result

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        feedback = self._build_retry_feedback(payload, exc)
        if feedback is None:
            return payload
        next_payload = dict(payload)
        next_payload["strategy_skeleton_feedback"] = feedback
        return next_payload

    def _build_retry_feedback(
        self,
        payload: dict[str, Any],
        exc: RetryableGenerationError,
    ) -> dict[str, Any] | None:
        validation_exc = exc.__cause__
        if not isinstance(validation_exc, ValidationError):
            return None

        (
            core_target_min,
            core_target_max,
            support_target_min,
            support_target_max,
            recall_target_max,
            _excerpt_target_min,
            _excerpt_target_max,
        ) = self._primitive_target_bounds(payload)
        raw_payload = exc.data.get("raw_payload")
        episode_feedback = self._episode_feedback_from_raw_payload(
            raw_payload=raw_payload,
            core_target_min=core_target_min,
            core_target_max=core_target_max,
            support_target_min=support_target_min,
            support_target_max=support_target_max,
            recall_target_max=recall_target_max,
        )
        issues = self._issues_from_validation_errors(
            validation_exc,
            episode_feedback_by_index={
                int(item["episode_index"]): item for item in episode_feedback
            },
        )

        if not episode_feedback and not issues:
            return {
                "issue": "schema_validation_failed",
                "required_ranges": {
                    "core_primitive_ids": f"{core_target_min}-{core_target_max}",
                    "support_primitive_roles": (f"{support_target_min}-{support_target_max}"),
                    "recall_primitive_ids_max": recall_target_max,
                },
                "canonical_field_names": {
                    "negative_scope": "negative_scope",
                    "episode_spine": "episode_spine",
                    "core_primitive_ids": "core_primitive_ids",
                    "support_primitive_roles": "support_primitive_roles",
                    "recall_primitive_ids": "recall_primitive_ids",
                },
                "instruction": (
                    "Revise only the invalid episodes. Use canonical field names, "
                    "satisfy the required primitive-count ranges exactly. If an "
                    "episode is overfull, trim or demote primitives rather than "
                    "adding more. If an episode is underfull, strengthen it or "
                    "reduce episode count rather than forcing a thin partition."
                ),
            }

        has_overfull = any(
            issue_type.endswith("_overfull")
            for item in episode_feedback
            for issue_type in item["issue_types"]
        )
        has_underfull = any(
            issue_type.endswith("_underfull")
            for item in episode_feedback
            for issue_type in item["issue_types"]
        )
        instruction_parts = [
            "Revise only the invalid episodes.",
            "Use canonical field names, especially `negative_scope` instead of `scope`.",
            "Satisfy the required primitive-count ranges exactly.",
        ]
        if has_overfull:
            instruction_parts.append(
                "If an episode is overfull, trim or demote primitives rather than adding more."
            )
        if has_underfull:
            instruction_parts.append(
                "If an episode is underfull, strengthen it with materially load-bearing primitives or merge/reduce episode count."
            )
        instruction_parts.append(
            "If the current partition still cannot satisfy those ranges, reduce episode count rather than forcing thin or bloated episodes."
        )
        return {
            "issue": "schema_validation_failed",
            "issues": issues,
            "episode_constraints_by_number": episode_feedback,
            "required_ranges": {
                "core_primitive_ids": f"{core_target_min}-{core_target_max}",
                "support_primitive_roles": f"{support_target_min}-{support_target_max}",
                "recall_primitive_ids_max": recall_target_max,
            },
            "canonical_field_names": {
                "negative_scope": "negative_scope",
                "episode_spine": "episode_spine",
                "core_primitive_ids": "core_primitive_ids",
                "support_primitive_roles": "support_primitive_roles",
                "recall_primitive_ids": "recall_primitive_ids",
            },
            "instruction": " ".join(instruction_parts),
        }

    @staticmethod
    def _count_direction(
        count: int,
        minimum: int,
        maximum: int,
    ) -> str | None:
        if count < minimum:
            return "underfull"
        if count > maximum:
            return "overfull"
        return None

    @classmethod
    def _episode_required_fix(
        cls,
        *,
        core_direction: str | None,
        support_direction: str | None,
        recall_direction: str | None,
        has_scope_alias: bool,
    ) -> tuple[str, str]:
        if core_direction == "overfull":
            fix = (
                "Demote or remove non-thesis primitives from core. If support is "
                "also overfull, narrow episode scope before keeping the current "
                "episode count."
            )
            action = "trim_core"
        elif support_direction == "overfull":
            fix = (
                "Trim duplicate or subordinate support before changing core. Keep "
                "only materially distinct grounding, pressure, or consequence."
            )
            action = "trim_support"
        elif core_direction == "underfull":
            fix = (
                "Strengthen the episode with genuinely load-bearing primitives or "
                "merge/reduce episode count instead of forcing a thin partition."
            )
            action = "merge_episode"
        elif support_direction == "underfull":
            fix = (
                "Add only materially distinct support that broadens grounding, "
                "pressure, or consequence without inflating core."
            )
            action = "strengthen_episode"
        elif recall_direction == "overfull":
            fix = (
                "Trim recall to the configured maximum and keep only materially "
                "necessary recurrence callbacks."
            )
            action = "trim_support"
        else:
            fix = "Correct the schema validation errors without changing valid unaffected episodes."
            action = "strengthen_episode"

        if has_scope_alias:
            fix = f"{fix} Use `negative_scope`, not `scope`."
        return fix, action

    @classmethod
    def _episode_feedback_from_raw_payload(
        cls,
        *,
        raw_payload: Any,
        core_target_min: int,
        core_target_max: int,
        support_target_min: int,
        support_target_max: int,
        recall_target_max: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_payload, dict):
            return []
        episodes = raw_payload.get("episodes")
        if not isinstance(episodes, list):
            return []

        feedback: list[dict[str, Any]] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            episode_number = episode.get("episode_number")
            if not isinstance(episode_number, int):
                episode_number = index + 1
            spine = episode.get("episode_spine")
            if not isinstance(spine, dict):
                spine = episode.get("spine")
            if not isinstance(spine, dict):
                continue
            core_ids = spine.get("core_primitive_ids")
            if not isinstance(core_ids, list):
                core_ids = spine.get("core_prims")
            support_roles = spine.get("support_primitive_roles")
            if not isinstance(support_roles, dict):
                support_roles = spine.get("support_roles")
            recall_ids = spine.get("recall_primitive_ids")
            if not isinstance(recall_ids, list):
                recall_ids = spine.get("recall_prims")

            core_count = len(core_ids) if isinstance(core_ids, list) else 0
            support_count = len(support_roles) if isinstance(support_roles, dict) else 0
            recall_count = len(recall_ids) if isinstance(recall_ids, list) else 0
            core_direction = cls._count_direction(
                core_count,
                core_target_min,
                core_target_max,
            )
            support_direction = cls._count_direction(
                support_count,
                support_target_min,
                support_target_max,
            )
            recall_direction = cls._count_direction(
                recall_count,
                0,
                recall_target_max,
            )
            has_scope_alias = "scope" in episode and "negative_scope" not in episode
            issue_types: list[str] = []
            direction_by_field: dict[str, str] = {}
            if core_direction is not None:
                issue_types.append(f"core_primitive_count_{core_direction}")
                direction_by_field["core_primitive_ids"] = core_direction
            if support_direction is not None:
                issue_types.append(f"support_primitive_count_{support_direction}")
                direction_by_field["support_primitive_roles"] = support_direction
            if recall_direction is not None:
                issue_types.append(f"recall_primitive_count_{recall_direction}")
                direction_by_field["recall_primitive_ids"] = recall_direction
            if has_scope_alias:
                issue_types.append("forbidden_scope_alias")
            if issue_types:
                required_fix, recommended_action = cls._episode_required_fix(
                    core_direction=core_direction,
                    support_direction=support_direction,
                    recall_direction=recall_direction,
                    has_scope_alias=has_scope_alias,
                )
                feedback.append(
                    {
                        "episode_index": index,
                        "episode_number": episode_number,
                        "issue_types": issue_types,
                        "direction_by_field": direction_by_field,
                        "actual_counts": {
                            "core_primitive_ids": core_count,
                            "support_primitive_roles": support_count,
                            "recall_primitive_ids": recall_count,
                        },
                        "required_fix": required_fix,
                        "recommended_action": recommended_action,
                    }
                )
        return feedback

    @staticmethod
    def _issues_from_validation_errors(
        validation_exc: ValidationError,
        *,
        episode_feedback_by_index: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        feedback_by_index = episode_feedback_by_index or {}
        issues: list[dict[str, Any]] = []
        seen_scope_paths: set[str] = set()
        for error in validation_exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            message = str(error.get("msg", "")).strip()
            error_type = str(error.get("type", "")).strip()
            issue = "schema_validation_failed"
            episode_index = next(
                (int(part) for part in error.get("loc", ()) if isinstance(part, int)),
                None,
            )
            direction_by_field = (
                feedback_by_index.get(episode_index, {}).get("direction_by_field", {})
                if episode_index is not None
                else {}
            )
            if "core_primitive_ids must contain" in message:
                issue = (
                    f"core_primitive_count_{direction_by_field['core_primitive_ids']}"
                    if "core_primitive_ids" in direction_by_field
                    else "core_primitive_count_out_of_range"
                )
            elif "support_primitive_roles must contain" in message:
                issue = (
                    f"support_primitive_count_{direction_by_field['support_primitive_roles']}"
                    if "support_primitive_roles" in direction_by_field
                    else "support_primitive_count_out_of_range"
                )
            elif "recall_primitive_ids must contain at most" in message:
                issue = (
                    f"recall_primitive_count_{direction_by_field['recall_primitive_ids']}"
                    if "recall_primitive_ids" in direction_by_field
                    else "recall_primitive_count_out_of_range"
                )
            elif _EPISODE_SCOPE_PATH_RE.search(loc):
                if loc in seen_scope_paths:
                    continue
                seen_scope_paths.add(loc)
                issue = "forbidden_scope_alias"
                message = "Use `negative_scope`, not `scope`."
            issues.append(
                {
                    "path": loc,
                    "issue": issue,
                    "error_type": error_type,
                    "message": message,
                }
            )
        return issues

    def build_payload(
        self,
        synthesis_map: dict,
        project_metadata: dict,
        scene_discovery: dict | None,
        episode_count: int | None,
        recommended_episode_count_min: int,
        recommended_episode_count_max: int,
        actor_metadata: dict | None = None,
        strategy_skeleton_feedback: dict | None = None,
        human_thread_candidates: list[dict] | None = None,
        excerpts: list[dict] | None = None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "recommended_episode_count_min": recommended_episode_count_min,
            "recommended_episode_count_max": recommended_episode_count_max,
        }
        if excerpts is not None:
            payload["excerpts"] = excerpts
        if scene_discovery is not None:
            payload["scene_discovery"] = scene_discovery
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if human_thread_candidates is not None:
            payload["human_thread_candidates"] = human_thread_candidates
        if episode_count is not None:
            payload["requested_episode_count"] = episode_count
        if strategy_skeleton_feedback is not None:
            payload["strategy_skeleton_feedback"] = strategy_skeleton_feedback
        return payload
