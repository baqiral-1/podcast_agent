"""Stage 8: scene discovery agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import scene_discovery_instructions
from podcast_agent.schemas.models import (
    PodcastMode,
    SceneDiscoveryArtifact,
    scene_discovery_candidate_range_for_mode,
)

_ALLOWED_CANDIDATE_ROLES = {
    "opening",
    "build",
    "turn",
    "answer",
}


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
            raise RetryableGenerationError(
                "scene_discovery candidate count fell outside the configured range.",
                data={
                    "issue": "candidate_count_out_of_range",
                    "candidate_count": candidate_count,
                    "target_min": target_min,
                    "target_max": target_max,
                },
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
        next_payload["scene_discovery_feedback"] = feedback
        return next_payload

    def _build_retry_feedback(self, exc: RetryableGenerationError) -> dict[str, object] | None:
        issue = str(exc.data.get("issue", "") or "").strip()
        if issue == "candidate_count_out_of_range":
            candidate_count = int(exc.data.get("candidate_count", 0) or 0)
            target_min = int(exc.data.get("target_min", 0) or 0)
            target_max = int(exc.data.get("target_max", 0) or 0)
            return {
                "issue": issue,
                "candidate_count": candidate_count,
                "target_min": target_min,
                "target_max": target_max,
                "instruction": (
                    "Revise only the candidate count. Keep valid candidates, ordering, "
                    "and references unchanged while returning a total inside the target range."
                ),
            }

        raw_payload = exc.data.get("raw_payload")
        if not isinstance(raw_payload, dict):
            return None
        candidates = raw_payload.get("candidates")
        if not isinstance(candidates, list):
            return None

        invalid_roles: set[str] = set()
        candidate_ids: list[str] = []
        legacy_field_candidate_ids: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            used_legacy_field = "candidate_roles" in candidate
            roles = candidate.get("scene_jobs")
            if roles is None and used_legacy_field:
                roles = candidate.get("candidate_roles")
            if not isinstance(roles, list):
                continue
            candidate_id = str(candidate.get("candidate_id", "") or "").strip()
            if used_legacy_field and candidate_id:
                legacy_field_candidate_ids.append(candidate_id)
            current_invalid_roles = [
                str(role).strip()
                for role in roles
                if str(role).strip() and str(role).strip() not in _ALLOWED_CANDIDATE_ROLES
            ]
            if not current_invalid_roles:
                continue
            invalid_roles.update(current_invalid_roles)
            if candidate_id:
                candidate_ids.append(candidate_id)

        if legacy_field_candidate_ids and not invalid_roles:
            return {
                "issue": "forbidden_candidate_roles_field",
                "candidate_ids": legacy_field_candidate_ids,
                "instruction": (
                    "Revise only scene_jobs. Do not emit candidate_roles. Every "
                    "scene_jobs entry must be one of `opening`, `build`, `turn`, "
                    "or `answer`. Keep valid candidates, scene sketches, "
                    "images, and references unchanged."
                ),
            }
        if not invalid_roles:
            return None
        return {
            "issue": "invalid_scene_jobs",
            "candidate_ids": candidate_ids,
            "invalid_roles": sorted(invalid_roles),
            "instruction": (
                "Revise only scene_jobs. Every entry must be one of "
                "`opening`, `build`, `turn`, or `answer`. "
                "Keep valid candidates, scene sketches, images, and references unchanged."
            ),
        }

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
        scene_discovery_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "passage_list": passage_list,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if scene_discovery_feedback is not None:
            payload["scene_discovery_feedback"] = scene_discovery_feedback
        return payload
