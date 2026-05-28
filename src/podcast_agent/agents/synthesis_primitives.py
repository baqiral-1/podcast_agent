"""Stage 7a: primitive substrate extraction agent."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.prompts import primitive_substrate_extraction_instructions
from podcast_agent.schemas.models import (
    PRIMITIVE_SUBSTRATES,
    PrimitiveSubstrate,
    RawCorePrimitivesArtifact,
    SynthesisPrimitivesArtifact,
)

_PRIMITIVE_ID_PREFIXES: dict[str, str] = {
    PrimitiveSubstrate.EVENTS.value: "e",
    PrimitiveSubstrate.ACTS.value: "a",
    PrimitiveSubstrate.ACTOR_PORTRAITS.value: "r",
    PrimitiveSubstrate.MECHANISMS.value: "m",
    PrimitiveSubstrate.CONDITIONS.value: "c",
    PrimitiveSubstrate.ARTIFACTS.value: "f",
    PrimitiveSubstrate.READINGS.value: "d",
}


class PrimitiveSubstrateExtractionAgent(Agent):
    """Extracts substrate-first primitives from the full synthesis evidence surface."""

    schema_name = "primitive_substrate_extraction"
    response_model = RawCorePrimitivesArtifact
    instructions = primitive_substrate_extraction_instructions()

    def build_instructions(self, payload: dict) -> str:
        target_ranges = payload.get("primitive_target_ranges")
        return primitive_substrate_extraction_instructions(
            target_ranges=target_ranges if isinstance(target_ranges, dict) else None
        )

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        feedback = self._build_synthesis_feedback(exc)
        if feedback is None:
            return payload
        next_payload = dict(payload)
        next_payload["synthesis_feedback"] = feedback
        return next_payload

    def build_payload(
        self,
        project_id: str,
        podcast_mode: str,
        axes_summary: list[dict],
        passages_by_axis: dict[str, list[dict]],
        cross_book_pairs: list[dict],
        book_metadata: list[dict],
        primitive_target_ranges: dict[str, tuple[int, int]] | None = None,
        actor_metadata: dict | None = None,
        synthesis_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "project_id": project_id,
            "podcast_mode": podcast_mode,
            "axes": axes_summary,
            "passages_by_axis": passages_by_axis,
            "cross_book_pairs": cross_book_pairs,
            "books": book_metadata,
        }
        if primitive_target_ranges is not None:
            payload["primitive_target_ranges"] = primitive_target_ranges
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if synthesis_feedback is not None:
            payload["synthesis_feedback"] = synthesis_feedback
        return payload

    def validate_result(
        self,
        result: RawCorePrimitivesArtifact,
        payload: dict,
    ) -> SynthesisPrimitivesArtifact:
        counters: defaultdict[str, int] = defaultdict(int)
        primitives: list[dict[str, Any]] = []
        for substrate in PRIMITIVE_SUBSTRATES:
            for primitive in result.primitives_by_substrate()[substrate]:
                counters[substrate] += 1
                primitive_payload = primitive.model_dump(mode="json")
                primitive_payload["substrate"] = substrate
                primitive_payload["id"] = (
                    f"{_PRIMITIVE_ID_PREFIXES[substrate]}{counters[substrate]}"
                )
                primitives.append(primitive_payload)
        return SynthesisPrimitivesArtifact.model_validate(
            {
                "project_id": result.project_id,
                "primitives": primitives,
                "quality_score": result.quality_score,
                "quality_notes": result.quality_notes,
            }
        )

    def _build_synthesis_feedback(
        self,
        exc: RetryableGenerationError,
    ) -> dict[str, Any] | None:
        validation_exc = exc.__cause__
        if not isinstance(validation_exc, ValidationError):
            return None

        raw_payload = exc.data.get("raw_payload")
        if not isinstance(raw_payload, dict):
            return None

        validation_errors = self._extract_validation_errors(raw_payload, validation_exc)
        if not validation_errors:
            return {
                "issue": "schema_validation_failed",
                "instruction": (
                    "Correct only the named primitive/field errors. Preserve valid "
                    "unaffected primitives as closely as possible and keep output order unchanged."
                ),
            }

        issue_types = {item["issue"] for item in validation_errors}
        top_level_issue = (
            next(iter(issue_types)) if len(issue_types) == 1 else "schema_validation_failed"
        )
        return {
            "issue": top_level_issue,
            "validation_errors": validation_errors,
            "instruction": (
                "Correct only the named primitive/field errors. Preserve valid unaffected "
                "primitives as closely as possible, keep output order unchanged, and do not "
                "substitute `title` for missing substrate-required fields."
            ),
        }

    def _extract_validation_errors(
        self,
        raw_payload: dict[str, Any],
        validation_exc: ValidationError,
    ) -> list[dict[str, Any]]:
        feedback_errors: list[dict[str, Any]] = []
        for error in validation_exc.errors():
            loc = error.get("loc", ())
            primitive_index = self._primitive_index_from_loc(raw_payload, loc)
            primitive_snapshot = self._primitive_snapshot_for_loc(raw_payload, loc)
            substrate = self._substrate_for_error(loc, primitive_snapshot)
            field_name = self._field_name_from_loc(loc)
            issue, required_fix = self._classify_error(
                field_name=field_name,
                error_type=str(error.get("type", "")),
                message=str(error.get("msg", "")).strip(),
                substrate=substrate,
            )
            feedback_errors.append(
                {
                    "path": ".".join(str(part) for part in loc),
                    "error_type": str(error.get("type", "")),
                    "message": str(error.get("msg", "")).strip(),
                    "primitive_index": primitive_index,
                    "primitive_id": self._primitive_id_from_snapshot(primitive_snapshot),
                    "substrate": substrate,
                    "field": field_name,
                    "issue": issue,
                    "required_fix": required_fix,
                    "primitive_snapshot": primitive_snapshot,
                }
            )
        return feedback_errors

    @staticmethod
    def _primitive_index_from_loc(
        raw_payload: dict[str, Any],
        loc: tuple[Any, ...] | list[Any],
    ) -> int | None:
        if len(loc) < 2 or not isinstance(loc[0], str) or not isinstance(loc[1], int):
            return None
        substrate = loc[0]
        if substrate not in PRIMITIVE_SUBSTRATES:
            return None
        primitive_index = int(loc[1])
        offset = 0
        for bucket in PRIMITIVE_SUBSTRATES:
            bucket_items = raw_payload.get(bucket, [])
            if bucket == substrate:
                if not isinstance(bucket_items, list) or primitive_index >= len(bucket_items):
                    return None
                return offset + primitive_index
            if isinstance(bucket_items, list):
                offset += len(bucket_items)
        return None

    @staticmethod
    def _field_name_from_loc(loc: tuple[Any, ...] | list[Any]) -> str | None:
        if not loc:
            return None
        last = loc[-1]
        return last if isinstance(last, str) else None

    @staticmethod
    def _primitive_snapshot_for_loc(
        raw_payload: dict[str, Any],
        loc: tuple[Any, ...] | list[Any],
    ) -> dict[str, Any] | None:
        if len(loc) < 2 or not isinstance(loc[0], str) or not isinstance(loc[1], int):
            return None
        substrate = loc[0]
        if substrate not in PRIMITIVE_SUBSTRATES:
            return None
        bucket = raw_payload.get(substrate)
        if not isinstance(bucket, list):
            return None
        primitive_index = int(loc[1])
        if primitive_index < 0 or primitive_index >= len(bucket):
            return None
        primitive = bucket[primitive_index]
        if not isinstance(primitive, dict):
            return None
        return {"substrate": substrate, **primitive}

    @staticmethod
    def _primitive_id_from_snapshot(
        primitive_snapshot: dict[str, Any] | None,
    ) -> str | None:
        if not isinstance(primitive_snapshot, dict):
            return None
        primitive_id = primitive_snapshot.get("id")
        return str(primitive_id) if primitive_id is not None else None

    @staticmethod
    def _substrate_for_error(
        loc: tuple[Any, ...] | list[Any],
        primitive_snapshot: dict[str, Any] | None,
    ) -> str | None:
        if len(loc) >= 1 and isinstance(loc[0], str) and loc[0] in PRIMITIVE_SUBSTRATES:
            return str(loc[0])
        if isinstance(primitive_snapshot, dict):
            substrate = primitive_snapshot.get("substrate")
            if isinstance(substrate, str) and substrate:
                return substrate
        return None

    @staticmethod
    def _classify_error(
        *,
        field_name: str | None,
        error_type: str,
        message: str,
        substrate: str | None,
    ) -> tuple[str, str]:
        substrate_label = substrate or "primitive"
        if error_type == "missing" and field_name:
            return (
                "missing_required_substrate_field",
                f"Add a non-empty `{field_name}` for this `{substrate_label}` primitive. Do not rely on `title` as a substitute.",
            )
        if error_type in {"literal_error", "enum"} or "Input should be" in message:
            field_label = field_name or "field"
            return (
                "invalid_substrate_enum",
                f"Replace `{field_label}` with one of the schema-allowed literal values for this `{substrate_label}` primitive.",
            )
        if field_name:
            return (
                "invalid_substrate_shape",
                f"Correct `{field_name}` so it matches the required shape for this `{substrate_label}` primitive.",
            )
        return (
            "schema_validation_failed",
            "Correct the schema validation error without changing valid unaffected primitives.",
        )


SynthesisPrimitivesAgent = PrimitiveSubstrateExtractionAgent
