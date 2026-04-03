"""Base interfaces for model-backed agent inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
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
        *,
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> BaseModel:
        """Generate and validate structured JSON for a response model."""


def _serialize_prompt_component(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def prompt_log_metadata(instructions: Any, payload: Any) -> dict[str, Any]:
    instructions_text = _serialize_prompt_component(instructions)
    payload_text = _serialize_prompt_component(payload)
    return {
        "instructions_char_count": len(instructions_text),
        "payload_char_count": len(payload_text),
        "instructions_sha256": hashlib.sha256(instructions_text.encode("utf-8")).hexdigest(),
        "payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
    }
