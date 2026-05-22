"""Stage 7b: per-substrate primitive function tagging agent."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import primitive_function_tagging_instructions
from podcast_agent.schemas.models import (
    PrimitiveFunctionTaggingArtifact,
    PrimitiveFunctionTaggingOverlayArtifact,
    PrimitiveSubstrate,
    apply_primitive_enrichment_overlay,
)

_FUNCTION_VALIDATION_PATTERN = re.compile(
    r"^(?P<function>[a-z_]+) (?P<detail>"
    r"requires its paired justification payload|"
    r"justification is not allowed unless the function tag is present"
    r")$"
)

_DEFERRED_SUBSTRATE_FIELDS: dict[str, tuple[str, ...]] = {
    PrimitiveSubstrate.EVENTS.value: ("event_result",),
    PrimitiveSubstrate.ACTS.value: ("immediate_result",),
    PrimitiveSubstrate.ACTOR_PORTRAITS.value: ("operating_pressure",),
    PrimitiveSubstrate.MECHANISMS.value: ("operating_chain", "inputs", "outputs"),
    PrimitiveSubstrate.CONDITIONS.value: ("active_tension",),
    PrimitiveSubstrate.ARTIFACTS.value: ("artifact_detail",),
}

_COMMON_OVERLAY_FIELDS = frozenset(
    {
        "functions",
        "salience",
        "pivot",
        "stake",
        "texture",
        "cost",
        "complication",
        "recurrence",
        "contest",
        "narration_hooks",
    }
)


class PrimitiveFunctionTaggingAgent(Agent):
    """Assigns narrative functions, justifications, and salience to one substrate batch."""

    response_model = PrimitiveFunctionTaggingOverlayArtifact

    def __init__(
        self,
        llm,
        *,
        substrate: PrimitiveSubstrate | str,
        max_retry_attempts: int = 3,
    ) -> None:
        self.substrate = PrimitiveSubstrate(substrate)
        self.schema_name = f"primitive_function_tagging_{self.substrate.value}"
        self.instructions = primitive_function_tagging_instructions(self.substrate)
        super().__init__(llm, max_retry_attempts=max_retry_attempts)

    def build_instructions(self, payload: dict) -> str:
        return primitive_function_tagging_instructions(self.substrate)

    def build_payload(
        self,
        *,
        project_id: str,
        podcast_mode: str,
        base_primitives: list[dict[str, Any]],
        passage_list: list[dict[str, str]],
        actor_metadata: dict | None = None,
        function_feedback: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "project_id": project_id,
            "podcast_mode": podcast_mode,
            "substrate": self.substrate.value,
            "base_primitives": base_primitives,
            "passage_list": passage_list,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if function_feedback is not None:
            payload["function_feedback"] = function_feedback
        return payload

    def validate_result(
        self,
        result: PrimitiveFunctionTaggingOverlayArtifact,
        payload: dict,
    ) -> PrimitiveFunctionTaggingOverlayArtifact:
        base_primitives = payload.get("base_primitives")
        expected_ids: list[str] = []
        if isinstance(base_primitives, list):
            for item in base_primitives:
                if not isinstance(item, dict):
                    continue
                primitive_id = str(item.get("id", "") or "").strip()
                if primitive_id:
                    expected_ids.append(primitive_id)
        returned_ids = list(result.overlays_by_id)
        if set(returned_ids) != set(expected_ids):
            missing_ids = [
                primitive_id for primitive_id in expected_ids if primitive_id not in result.overlays_by_id
            ]
            unexpected_ids = [
                primitive_id for primitive_id in returned_ids if primitive_id not in expected_ids
            ]
            raise RetryableGenerationError(
                "primitive_function_tagging must return exactly one overlay per input primitive id",
                data={
                    "issue": "mismatched_enrichment_overlay_ids",
                    "expected_primitive_ids": expected_ids,
                    "returned_primitive_ids": returned_ids,
                    "missing_primitive_ids": missing_ids,
                    "unexpected_primitive_ids": unexpected_ids,
                    "substrate": self.substrate.value,
                },
            )
        invalid_overlay_field_errors = self._collect_invalid_overlay_field_errors(result)
        if invalid_overlay_field_errors:
            raise RetryableGenerationError(
                "primitive_function_tagging returned fields outside the allowed enrichment overlay",
                data={
                    "issue": "invalid_overlay_fields_for_substrate",
                    "substrate": self.substrate.value,
                    "validation_errors": invalid_overlay_field_errors,
                },
            )
        merged_artifact = self._build_merged_artifact(payload, result)
        deferred_errors = self._collect_missing_deferred_field_errors(
            payload=payload,
            result=merged_artifact,
        )
        if deferred_errors:
            issue_types = {error["issue"] for error in deferred_errors}
            top_level_issue = (
                next(iter(issue_types))
                if len(issue_types) == 1
                else "missing_deferred_substrate_field"
            )
            raise RetryableGenerationError(
                "primitive_function_tagging omitted deferred substrate-detail fields",
                data={
                    "issue": top_level_issue,
                    "substrate": self.substrate.value,
                    "validation_errors": deferred_errors,
                },
            )
        return result

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        feedback = self._build_function_feedback(payload, exc)
        if feedback is None:
            return payload
        next_payload = dict(payload)
        next_payload["function_feedback"] = feedback
        return next_payload

    def _build_function_feedback(
        self,
        payload: dict[str, Any],
        exc: RetryableGenerationError,
    ) -> dict[str, Any] | None:
        issue = str(exc.data.get("issue") or "").strip()
        if issue in {
            "mismatched_enrichment_overlay_ids",
            "invalid_overlay_fields_for_substrate",
            "missing_deferred_substrate_field",
            "invalid_deferred_substrate_shape",
        }:
            feedback = {key: value for key, value in exc.data.items() if key != "raw_payload"}
            feedback["instruction"] = (
                "Correct only the named issue. Preserve all valid unaffected overlays "
                "exactly, fill deferred substrate-detail fields from the supplied evidence "
                "when requested, and keep unaffected primitive ids stable."
            )
            return feedback

        validation_exc = exc.__cause__
        if not isinstance(validation_exc, ValidationError):
            return None

        validation_errors = self._extract_validation_errors(payload, validation_exc)
        if not validation_errors:
            return {
                "issue": "schema_validation_failed",
                "substrate": self.substrate.value,
                "instruction": (
                    "Correct only the named issue. Preserve all valid unaffected "
                    "overlays exactly and keep unaffected primitive ids stable."
                ),
            }

        issue_types = {item["issue"] for item in validation_errors}
        top_level_issue = (
            next(iter(issue_types)) if len(issue_types) == 1 else "schema_validation_failed"
        )
        return {
            "issue": top_level_issue,
            "substrate": self.substrate.value,
            "validation_errors": validation_errors,
            "instruction": (
                "Correct only the named primitive/function mismatch. Preserve all valid "
                "unaffected overlays exactly and keep unaffected primitive ids stable."
            ),
        }

    def _extract_validation_errors(
        self,
        payload: dict[str, Any],
        validation_exc: ValidationError,
    ) -> list[dict[str, Any]]:
        feedback_errors: list[dict[str, Any]] = []
        for error in validation_exc.errors():
            loc = error.get("loc", ())
            primitive_index = self._primitive_index_from_loc(loc)
            path = ".".join(str(part) for part in loc)
            primitive_id = self._primitive_id_from_loc(loc)
            if primitive_id is None:
                primitive_id = self._primitive_id_for_index(payload, primitive_index)
            if primitive_index is None and primitive_id is not None:
                primitive_index = self._primitive_index_for_id(payload, primitive_id)
            message = str(error.get("msg", "")).strip()
            function_name, issue_type, required_fix = self._classify_function_error(
                message
            )
            feedback_errors.append(
                {
                    "path": path,
                    "error_type": str(error.get("type", "")),
                    "message": message,
                    "primitive_index": primitive_index,
                    "primitive_id": primitive_id,
                    "substrate": self.substrate.value,
                    "function": function_name,
                    "issue": issue_type,
                    "required_fix": required_fix,
                }
            )
        return feedback_errors

    @staticmethod
    def _primitive_index_from_loc(loc: tuple[Any, ...] | list[Any]) -> int | None:
        if (
            len(loc) >= 2
            and loc[0] == "overlays_by_id"
            and isinstance(loc[1], str)
        ):
            return None
        if len(loc) >= 2 and loc[0] == "primitives" and isinstance(loc[1], int):
            return int(loc[1])
        return None

    @staticmethod
    def _primitive_id_from_loc(loc: tuple[Any, ...] | list[Any]) -> str | None:
        if (
            len(loc) >= 2
            and loc[0] == "overlays_by_id"
            and isinstance(loc[1], str)
        ):
            return str(loc[1])
        return None

    @staticmethod
    def _primitive_id_for_index(
        payload: dict[str, Any],
        primitive_index: int | None,
    ) -> str | None:
        if primitive_index is None:
            return None
        base_primitives = payload.get("base_primitives")
        if not isinstance(base_primitives, list):
            return None
        if primitive_index < 0 or primitive_index >= len(base_primitives):
            return None
        primitive = base_primitives[primitive_index]
        if not isinstance(primitive, dict):
            return None
        primitive_id = primitive.get("id")
        return str(primitive_id) if primitive_id is not None else None

    @staticmethod
    def _primitive_index_for_id(
        payload: dict[str, Any],
        primitive_id: str,
    ) -> int | None:
        base_primitives = payload.get("base_primitives")
        if not isinstance(base_primitives, list):
            return None
        for primitive_index, primitive in enumerate(base_primitives):
            if not isinstance(primitive, dict):
                continue
            if str(primitive.get("id") or "").strip() == primitive_id:
                return primitive_index
        return None

    @staticmethod
    def _classify_function_error(
        message: str,
    ) -> tuple[str | None, str, str]:
        normalized_message = message.removeprefix("Value error, ").strip()
        match = _FUNCTION_VALIDATION_PATTERN.match(normalized_message)
        if match is None:
            return (
                None,
                "schema_validation_failed",
                "Correct the schema validation error without changing valid unaffected overlays.",
            )
        function_name = match.group("function")
        detail = match.group("detail")
        if detail == "requires its paired justification payload":
            return (
                function_name,
                "tag_without_payload",
                f"Either add a valid `{function_name}` payload or remove `{function_name}` from `functions`.",
            )
        return (
            function_name,
            "payload_without_tag",
            f"Either add `{function_name}` to `functions` or remove the `{function_name}` payload.",
        )

    def _collect_invalid_overlay_field_errors(
        self,
        result: PrimitiveFunctionTaggingOverlayArtifact,
    ) -> list[dict[str, Any]]:
        allowed_fields = _COMMON_OVERLAY_FIELDS | set(
            _DEFERRED_SUBSTRATE_FIELDS.get(self.substrate.value, ())
        )
        errors: list[dict[str, Any]] = []
        for primitive_id, overlay in result.overlays_by_id.items():
            for field_name in overlay.model_fields_set:
                if field_name in allowed_fields:
                    continue
                errors.append(
                    {
                        "path": f"overlays_by_id.{primitive_id}.{field_name}",
                        "error_type": "invalid_overlay_field",
                        "message": (
                            f"Field `{field_name}` is not allowed in the "
                            f"`{self.substrate.value}` enrichment overlay."
                        ),
                        "primitive_index": None,
                        "primitive_id": primitive_id,
                        "substrate": self.substrate.value,
                        "field": field_name,
                        "issue": "invalid_overlay_fields_for_substrate",
                        "required_fix": (
                            f"Remove `{field_name}` from the returned overlay for this "
                            f"`{self.substrate.value}` batch."
                        ),
                    }
                )
        return errors

    def _build_merged_artifact(
        self,
        payload: dict[str, Any],
        result: PrimitiveFunctionTaggingOverlayArtifact,
    ) -> PrimitiveFunctionTaggingArtifact:
        base_primitives = payload.get("base_primitives")
        if not isinstance(base_primitives, list):
            raise RetryableGenerationError(
                "primitive_function_tagging payload is missing base_primitives",
                data={
                    "issue": "schema_validation_failed",
                    "substrate": self.substrate.value,
                },
            )
        merged_primitives: list[dict[str, Any]] = []
        for primitive in base_primitives:
            if not isinstance(primitive, dict):
                continue
            primitive_id = str(primitive.get("id", "") or "").strip()
            if not primitive_id or primitive_id not in result.overlays_by_id:
                continue
            merged_primitives.append(
                apply_primitive_enrichment_overlay(
                    primitive,
                    result.overlays_by_id[primitive_id],
                )
            )
        try:
            return PrimitiveFunctionTaggingArtifact.model_validate(
                {
                    "project_id": result.project_id,
                    "primitives": merged_primitives,
                }
            )
        except ValidationError as validation_exc:
            raise RetryableGenerationError(
                "Schema validation failed for merged primitive_function_tagging output",
                data={"raw_payload": result.model_dump(mode="json")},
            ) from validation_exc

    def _collect_missing_deferred_field_errors(
        self,
        *,
        payload: dict[str, Any],
        result: PrimitiveFunctionTaggingArtifact,
    ) -> list[dict[str, Any]]:
        deferred_fields = _DEFERRED_SUBSTRATE_FIELDS.get(self.substrate.value, ())
        if not deferred_fields:
            return []
        errors: list[dict[str, Any]] = []
        for primitive in result.primitives:
            primitive_index = self._primitive_index_for_id(payload, primitive.id)
            for field_name in deferred_fields:
                value = getattr(primitive, field_name, None)
                if isinstance(value, list):
                    is_missing = not value
                else:
                    is_missing = not str(value or "").strip()
                if not is_missing:
                    continue
                issue_type = (
                    "invalid_deferred_substrate_shape"
                    if field_name == "operating_chain"
                    else "missing_deferred_substrate_field"
                )
                errors.append(
                    {
                        "path": f"overlays_by_id.{primitive.id}.{field_name}",
                        "error_type": "missing_deferred_substrate_field",
                        "message": (
                            f"Deferred substrate-detail field `{field_name}` must be filled before returning."
                        ),
                        "primitive_index": primitive_index,
                        "primitive_id": primitive.id,
                        "substrate": self.substrate.value,
                        "field": field_name,
                        "issue": issue_type,
                        "required_fix": self._required_fix_for_deferred_field(field_name),
                    }
                )
        return errors

    def _required_fix_for_deferred_field(self, field_name: str) -> str:
        if field_name == "operating_chain":
            return (
                "Fill a non-empty `operating_chain` from the supplied evidence for this "
                f"`{self.substrate.value}` primitive before returning."
            )
        if field_name == "artifact_detail":
            return (
                "Fill a non-empty `artifact_detail` from the supplied evidence for this "
                f"`{self.substrate.value}` primitive before returning."
            )
        return (
            f"Fill non-empty `{field_name}` from the supplied evidence for this "
            f"`{self.substrate.value}` primitive before returning."
        )
