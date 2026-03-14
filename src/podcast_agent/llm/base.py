"""Base interfaces for model-backed agent inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


PromptPayload = dict[str, Any]


class LLMContentFilterError(RuntimeError):
    """Raised when the model response is blocked by content filtering."""


class LLMClient(ABC):
    """Abstract interface for JSON-producing LLM calls."""

    def __init__(self) -> None:
        self.run_logger = None

    def set_run_logger(self, run_logger: Any) -> None:
        """Attach a run logger used for prompt/response tracing."""

        self.run_logger = run_logger

    @abstractmethod
    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Generate and validate structured JSON for a response model."""
