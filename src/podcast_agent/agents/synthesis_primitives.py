"""Stage 7a: synthesis primitives agent."""

from __future__ import annotations

import logging
import time

from podcast_agent.agents.base import Agent
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.prompts import synthesis_primitives_instructions
from podcast_agent.schemas.models import SynthesisPrimitivesArtifact

logger = logging.getLogger(__name__)


class SynthesisPrimitivesAgent(Agent):
    """Extracts grounded synthesis primitives from the full evidence surface."""

    schema_name = "synthesis_primitives"
    response_model = SynthesisPrimitivesArtifact
    instructions = synthesis_primitives_instructions()

    def build_payload(
        self,
        project_id: str,
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

    def run(self, payload: dict) -> SynthesisPrimitivesArtifact:
        target_ranges = payload.get("primitive_target_ranges")
        instructions = synthesis_primitives_instructions(
            target_ranges=target_ranges if isinstance(target_ranges, dict) else None
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retry_attempts + 1):
            with llm_semaphore_for(self.schema_name):
                try:
                    return self.llm.generate_json(
                        schema_name=self.schema_name,
                        instructions=instructions,
                        payload=payload,
                        response_model=self.response_model,
                        attempt=attempt,
                        max_attempts=self.max_retry_attempts,
                    )
                except (TransientLLMError, RetryableGenerationError) as exc:
                    last_exc = exc
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
