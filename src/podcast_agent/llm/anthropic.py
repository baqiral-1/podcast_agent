"""Anthropic JSON client for agent inference."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from podcast_agent.config import LLMConfig
from podcast_agent.llm.base import LLMClient, LLMContentFilterError, PromptPayload
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.llm.openai_compatible import (
    HTTPTransport,
    _unwrap_response_payload,
)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLMClient(LLMClient):
    """Client that requests schema-constrained JSON from Anthropic via tool_use."""

    def __init__(self, config: LLMConfig, transport: HTTPTransport | None = None) -> None:
        super().__init__()
        self.config = config
        self.transport = transport or HTTPTransport()

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Request structured JSON and validate it against the response model."""

        if not self.config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic LLM client.")

        with llm_semaphore_for(schema_name):
            selected_model = self.config.model_overrides.get(schema_name, self.config.model_name)
            endpoint = f"{self.config.anthropic_base_url.rstrip('/')}/v1/messages"
            user_content = json.dumps(
                {
                    "schema_name": schema_name,
                    "payload": payload,
                },
                default=str,
            )
            request_payload = {
                "model": selected_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.anthropic_max_tokens,
                "system": instructions,
                "tools": [
                    {
                        "name": "respond",
                        "description": "Respond with structured data matching the requested schema.",
                        "input_schema": response_model.model_json_schema(),
                    }
                ],
                "tool_choice": {"type": "tool", "name": "respond"},
                "messages": [{"role": "user", "content": user_content}],
            }
            headers = {
                "x-api-key": self.config.anthropic_api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            }
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_request",
                    client="anthropic",
                    schema_name=schema_name,
                    instructions=instructions,
                    payload=user_content,
                    model=selected_model,
                    timeout_seconds=self.config.timeout_seconds,
                )
            try:
                response = self.transport.post_json(
                    url=endpoint,
                    headers=headers,
                    payload=request_payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_response",
                        client="anthropic",
                        schema_name=schema_name,
                        response=response.body,
                    )
                _raise_for_anthropic_stop_reason(response.body)
                tool_input = _extract_tool_use_input(response.body)
                normalized_payload = _unwrap_response_payload(tool_input)
                return response_model.model_validate_json(json.dumps(normalized_payload))
            except Exception as exc:
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_error",
                        client="anthropic",
                        schema_name=schema_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                raise


def _extract_tool_use_input(body: dict[str, Any]) -> dict[str, Any]:
    """Extract the pre-parsed input dict from a tool_use content block."""
    content = body.get("content")
    if isinstance(content, list):
        for item in content:
            if item.get("type") == "tool_use" and item.get("name") == "respond":
                tool_input = item.get("input")
                if isinstance(tool_input, dict):
                    return tool_input
    raise RuntimeError("LLM response did not include a tool_use block named 'respond'.")


def _raise_for_anthropic_stop_reason(body: dict[str, Any]) -> None:
    stop_reason = body.get("stop_reason")
    if stop_reason == "max_tokens":
        raise RuntimeError("LLM response was truncated because it hit the completion token limit.")
    if stop_reason == "content_filter":
        raise LLMContentFilterError("LLM response was blocked by content filtering.")
    # "tool_use" and "end_turn" are both success cases.


__all__ = ["AnthropicLLMClient"]
