"""Base class for LLM-backed agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

from podcast_agent.langchain.runnables import (
    RetryableGenerationError,
    TransientLLMError,
    is_transient_error,
)
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.llm.transport import encode_transport_payload

logger = logging.getLogger(__name__)


def _extract_payload_episode_number(payload: dict) -> int | None:
    for key in ("episode_number",):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("strategy_episode", "architecture", "episode"):
        value = payload.get(key)
        if isinstance(value, dict):
            episode_number = value.get("episode_number")
            if isinstance(episode_number, int):
                return episode_number
    return None


class Agent(ABC):
    """Shared base class for JSON-producing agents."""

    schema_name: str
    instructions: str
    response_model: type[BaseModel]

    def __init__(self, llm: LLMClient, *, max_retry_attempts: int = 3) -> None:
        self.llm = llm
        self.max_retry_attempts = max_retry_attempts

    def _log_retry_scheduled(
        self,
        *,
        payload: dict,
        attempt: int,
        backoff: float,
        exc: Exception,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        run_logger.log(
            "llm_retry_scheduled",
            schema_name=self.schema_name,
            episode_number=_extract_payload_episode_number(payload),
            attempt=attempt,
            max_attempts=self.max_retry_attempts,
            next_attempt=attempt + 1,
            backoff_seconds=backoff,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def build_instructions(self, payload: dict) -> str:
        """Construct the instructions sent to the LLM."""

        return self.instructions

    def validate_result(self, result: BaseModel, payload: dict) -> BaseModel:
        """Apply any post-parse validation that depends on runtime payload."""

        return result

    def prepare_retry_payload(
        self,
        payload: dict,
        exc: RetryableGenerationError,
    ) -> dict:
        """Optionally adjust the next-attempt payload after a retryable generation error."""

        return payload

    def build_llm_payload(self, payload: dict) -> dict:
        """Build the transport payload sent to the model."""

        encoded = encode_transport_payload(self.schema_name, payload)
        return encoded if isinstance(encoded, dict) else payload

    def run(self, payload: dict) -> BaseModel:
        """Execute the agent with retry and concurrency gating."""
        last_exc: Exception | None = None
        current_payload = payload
        instructions = self.build_instructions(current_payload)
        for attempt in range(1, self.max_retry_attempts + 1):
            with llm_semaphore_for(self.schema_name):
                try:
                    llm_payload = self.build_llm_payload(current_payload)
                    result = self.llm.generate_json(
                        schema_name=self.schema_name,
                        instructions=instructions,
                        payload=llm_payload,
                        response_model=self.response_model,
                        attempt=attempt,
                        max_attempts=self.max_retry_attempts,
                    )
                    return self.validate_result(result, current_payload)
                except (TransientLLMError, RetryableGenerationError) as exc:
                    last_exc = exc
                    if attempt < self.max_retry_attempts:
                        if isinstance(exc, RetryableGenerationError):
                            current_payload = self.prepare_retry_payload(
                                current_payload, exc
                            )
                            instructions = self.build_instructions(current_payload)
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        self._log_retry_scheduled(
                            payload=current_payload,
                            attempt=attempt,
                            backoff=backoff,
                            exc=exc,
                        )
                        logger.warning(
                            "Agent %s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                            self.schema_name, attempt, self.max_retry_attempts,
                            type(exc).__name__, exc, backoff,
                        )
                        time.sleep(backoff)
                    continue
                except Exception as exc:
                    if not is_transient_error(exc):
                        raise
                    last_exc = exc
                    if attempt < self.max_retry_attempts:
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        self._log_retry_scheduled(
                            payload=current_payload,
                            attempt=attempt,
                            backoff=backoff,
                            exc=exc,
                        )
                        logger.warning(
                            "Agent %s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                            self.schema_name, attempt, self.max_retry_attempts,
                            type(exc).__name__, exc, backoff,
                        )
                        time.sleep(backoff)
                    continue
        raise last_exc  # type: ignore[misc]

    @abstractmethod
    def build_payload(self, *args, **kwargs) -> dict:
        """Construct the payload sent to the LLM."""
