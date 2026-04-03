"""Base class for LLM-backed agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.concurrency import llm_semaphore_for

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Shared base class for JSON-producing agents."""

    schema_name: str
    instructions: str
    response_model: type[BaseModel]

    def __init__(self, llm: LLMClient, *, max_retry_attempts: int = 3) -> None:
        self.llm = llm
        self.max_retry_attempts = max_retry_attempts

    def run(self, payload: dict) -> BaseModel:
        """Execute the agent with retry and concurrency gating."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retry_attempts + 1):
            with llm_semaphore_for(self.schema_name):
                try:
                    return self.llm.generate_json(
                        schema_name=self.schema_name,
                        instructions=self.instructions,
                        payload=payload,
                        response_model=self.response_model,
                        attempt=attempt,
                        max_attempts=self.max_retry_attempts,
                    )
                except (TransientLLMError, RetryableGenerationError) as exc:
                    last_exc = exc
                    if attempt < self.max_retry_attempts:
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        logger.warning(
                            "Agent %s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                            self.schema_name, attempt, self.max_retry_attempts,
                            type(exc).__name__, exc, backoff,
                        )
                        time.sleep(backoff)
                    continue
                except Exception:
                    raise
        raise last_exc  # type: ignore[misc]

    @abstractmethod
    def build_payload(self, *args, **kwargs) -> dict:
        """Construct the payload sent to the LLM."""
