"""Stage 8b: primitive enrichment agent."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.prompts import primitive_enrichment_instructions
from podcast_agent.schemas.models import (
    PRIMITIVE_ENRICHMENT_ARTIFACT_MODEL_BY_FAMILY,
    PrimitiveEnrichmentArtifact,
    RICH_SYNTHESIS_PRIMITIVE_FAMILY_SET,
)

logger = logging.getLogger(__name__)


class PrimitiveEnrichmentAgent(Agent):
    """Enriches selected rich-family primitives after narrative strategy."""

    schema_name = "primitive_enrichment"
    response_model = PrimitiveEnrichmentArtifact
    instructions = primitive_enrichment_instructions()

    def instructions_for_family(self, family: str) -> str:
        if family not in RICH_SYNTHESIS_PRIMITIVE_FAMILY_SET:
            raise ValueError(
                f"Unknown primitive enrichment family: {family}"
            )
        return primitive_enrichment_instructions(family)

    def response_model_for_family(self, family: str) -> type[BaseModel]:
        try:
            return PRIMITIVE_ENRICHMENT_ARTIFACT_MODEL_BY_FAMILY[family]
        except KeyError as exc:
            raise ValueError(f"Unknown primitive enrichment family: {family}") from exc

    def run(self, payload: dict) -> BaseModel:
        family = str(payload.get("family", "")).strip()
        if family not in RICH_SYNTHESIS_PRIMITIVE_FAMILY_SET:
            raise ValueError(f"Unknown primitive enrichment family: {family or '<missing>'}")

        instructions = self.instructions_for_family(family)
        response_model = self.response_model_for_family(family)
        last_exc: Exception | None = None
        retry_payload = dict(payload)
        for attempt in range(1, self.max_retry_attempts + 1):
            with llm_semaphore_for(self.schema_name):
                try:
                    return self.llm.generate_json(
                        schema_name=self.schema_name,
                        instructions=instructions,
                        payload=retry_payload,
                        response_model=response_model,
                        attempt=attempt,
                        max_attempts=self.max_retry_attempts,
                    )
                except (TransientLLMError, RetryableGenerationError) as exc:
                    last_exc = exc
                    if isinstance(exc, RetryableGenerationError):
                        feedback = self._build_enrichment_feedback(
                            family=family,
                            exc=exc,
                        )
                        if feedback is not None:
                            retry_payload = dict(payload)
                            retry_payload["enrichment_feedback"] = feedback
                    if attempt < self.max_retry_attempts:
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        self._log_retry_scheduled(
                            attempt=attempt,
                            backoff=backoff,
                            exc=exc,
                        )
                        logger.warning(
                            "Agent %s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                            self.schema_name,
                            attempt,
                            self.max_retry_attempts,
                            type(exc).__name__,
                            exc,
                            backoff,
                        )
                        time.sleep(backoff)
                    continue
                except Exception:
                    raise
        raise last_exc  # type: ignore[misc]

    def build_payload(
        self,
        *,
        project_id: str,
        family: str,
        base_primitives: list[dict],
        evidence_by_primitive_id: dict[str, dict[str, list[dict[str, str]]]],
        actor_metadata: dict | None = None,
        enrichment_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "project_id": project_id,
            "family": family,
            "base_primitives": base_primitives,
            "evidence_by_primitive_id": evidence_by_primitive_id,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if enrichment_feedback is not None:
            payload["enrichment_feedback"] = enrichment_feedback
        return payload

    def _build_enrichment_feedback(
        self,
        *,
        family: str,
        exc: RetryableGenerationError,
    ) -> dict[str, Any] | None:
        cause = exc.__cause__
        if not isinstance(cause, ValidationError):
            return None

        raw_payload = exc.data.get("raw_payload")
        validation_errors = [
            self._format_validation_error(raw_payload=raw_payload, error=error)
            for error in cause.errors()
        ]
        if not validation_errors:
            return None

        return {
            "issue": "schema_validation_failed",
            "family": family,
            "validation_errors": validation_errors,
            "instruction": (
                "Correct only the listed schema validation issues. Preserve ids, family, order, "
                "and all valid unaffected fields. Do not add unsupported keys."
            ),
        }

    def _format_validation_error(
        self,
        *,
        raw_payload: Any,
        error: dict[str, Any],
    ) -> dict[str, Any]:
        loc = tuple(error.get("loc", ()))
        primitive_id: str | None = None
        if (
            len(loc) >= 2
            and loc[0] == "enriched_primitives"
            and isinstance(loc[1], int)
            and isinstance(raw_payload, dict)
        ):
            enriched = raw_payload.get("enriched_primitives")
            if isinstance(enriched, list) and 0 <= loc[1] < len(enriched):
                item = enriched[loc[1]]
                if isinstance(item, dict):
                    candidate = item.get("id")
                    if isinstance(candidate, str) and candidate.strip():
                        primitive_id = candidate.strip()

        return {
            "path": ".".join(str(part) for part in loc),
            "error_type": str(error.get("type", "")),
            "message": str(error.get("msg", "")),
            "primitive_id": primitive_id,
        }
