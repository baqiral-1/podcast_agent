"""Anthropic JSON client for agent inference."""

from __future__ import annotations

import json
from datetime import UTC, datetime
import os
import threading
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from podcast_agent.config import LLMConfig
from podcast_agent.llm.base import LLMClient, LLMContentFilterError, PromptPayload, prompt_log_metadata
from podcast_agent.llm.concurrency import llm_semaphore_for
from podcast_agent.llm.openai_compatible import (
    HTTPTransport,
    LLMTransportHTTPError,
    _unwrap_response_payload,
)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLMClient(LLMClient):
    """Client that requests schema-constrained JSON from Anthropic via tool_use."""
    _heartbeat_interval_seconds = 120.0

    def __init__(self, config: LLMConfig, transport: HTTPTransport | None = None) -> None:
        super().__init__()
        self.config = config
        self.transport = transport or HTTPTransport()
        self._prompt_cache_supported: bool | None = None

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
            selected_model = self.config.model_name
            endpoint = f"{self.config.anthropic_base_url.rstrip('/')}/v1/messages"
            user_content = json.dumps(
                {
                    "schema_name": schema_name,
                    "payload": payload,
                },
                default=str,
            )
            prompt_caching_enabled = self._resolve_prompt_caching_enabled()
            request_payload = _build_request_payload(
                selected_model=selected_model,
                temperature=self.config.temperature,
                max_tokens=self.config.anthropic_max_tokens,
                instructions=instructions,
                input_schema=response_model.model_json_schema(),
                user_content=user_content,
                prompt_caching_enabled=prompt_caching_enabled,
            )
            headers = {
                "x-api-key": self.config.anthropic_api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            }
            request_uuid = uuid4().hex
            request_started_at = datetime.now(UTC).isoformat()
            request_started_monotonic = time.monotonic()
            stop_heartbeat, heartbeat_thread = self._start_inflight_heartbeat(
                request_uuid=request_uuid,
                schema_name=schema_name,
                model=selected_model,
                timeout_seconds=self.config.timeout_seconds,
                request_started_monotonic=request_started_monotonic,
            )
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_request",
                    request_uuid=request_uuid,
                    client="anthropic",
                    schema_name=schema_name,
                    model=selected_model,
                    endpoint=endpoint,
                    request_started_at=request_started_at,
                    timeout_seconds=self.config.timeout_seconds,
                    prompt_caching_enabled=prompt_caching_enabled,
                    **prompt_log_metadata(instructions, user_content),
                )
            try:
                try:
                    response = self.transport.post_json_sse(
                        url=endpoint,
                        headers=headers,
                        payload=request_payload,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                except LLMTransportHTTPError as exc:
                    if (
                        prompt_caching_enabled
                        and self.config.anthropic_prompt_caching_auto_fallback
                        and _is_prompt_cache_rejection(exc)
                    ):
                        prompt_caching_enabled = False
                        request_payload = _build_request_payload(
                            selected_model=selected_model,
                            temperature=self.config.temperature,
                            max_tokens=self.config.anthropic_max_tokens,
                            instructions=instructions,
                            input_schema=response_model.model_json_schema(),
                            user_content=user_content,
                            prompt_caching_enabled=False,
                        )
                        if self.run_logger is not None:
                            self.run_logger.log(
                                "anthropic_prompt_cache_fallback",
                                request_uuid=request_uuid,
                                schema_name=schema_name,
                                status_code=exc.status_code,
                                error_message=exc.response_text[:500],
                            )
                        self._prompt_cache_supported = False
                        response = self.transport.post_json_sse(
                            url=endpoint,
                            headers=headers,
                            payload=request_payload,
                            timeout_seconds=self.config.timeout_seconds,
                        )
                    else:
                        raise
                if self.run_logger is not None:
                    response_body = response.body
                    usage = response_body.get("usage")
                    provider_request_id = (response.response_headers or {}).get("x-request-id")
                    self.run_logger.log(
                        "llm_response_meta",
                        request_uuid=request_uuid,
                        client="anthropic",
                        schema_name=schema_name,
                        response_id=response_body.get("id"),
                        provider_request_id=provider_request_id,
                        status_code=response.status_code,
                        status=response_body.get("stop_reason"),
                        model=response_body.get("model", selected_model),
                        response_created_at=response_body.get("created_at"),
                        response_completed_at=response_body.get("completed_at"),
                        elapsed_ms=response.elapsed_ms,
                        time_to_headers_ms=response.time_to_headers_ms,
                        response_bytes=response.response_bytes,
                        prompt_caching_enabled=prompt_caching_enabled,
                        prompt_tokens=(usage or {}).get("input_tokens"),
                        completion_tokens=(usage or {}).get("output_tokens"),
                        cache_creation_input_tokens=_extract_cache_token_usage(usage, "cache_creation_input_tokens"),
                        cache_read_input_tokens=_extract_cache_token_usage(usage, "cache_read_input_tokens"),
                        total_tokens=(usage or {}).get("total_tokens"),
                    )
                    self.run_logger.log(
                        "llm_response",
                        request_uuid=request_uuid,
                        client="anthropic",
                        schema_name=schema_name,
                        response=response_body,
                    )
                _raise_for_anthropic_stop_reason(response.body)
                tool_input = _extract_tool_use_input(response.body)
                normalized_payload = _unwrap_response_payload(tool_input)
                return response_model.model_validate_json(json.dumps(normalized_payload))
            except Exception as exc:
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_error",
                        request_uuid=request_uuid,
                        client="anthropic",
                        schema_name=schema_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        instructions=instructions,
                        payload=user_content,
                    )
                raise
            finally:
                stop_heartbeat.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=0.1)

    def _resolve_prompt_caching_enabled(self) -> bool:
        if not self.config.anthropic_prompt_caching_enabled:
            return False
        if self._prompt_cache_supported is False:
            return False
        return True

    def _start_inflight_heartbeat(
        self,
        *,
        request_uuid: str,
        schema_name: str,
        model: str,
        timeout_seconds: float,
        request_started_monotonic: float,
    ) -> tuple[threading.Event, threading.Thread | None]:
        if self.run_logger is None:
            return threading.Event(), None
        if os.getenv("PODCAST_AGENT_DISABLE_LLM_HEARTBEAT", "").strip().lower() in {"1", "true", "yes"}:
            return threading.Event(), None
        stop = threading.Event()

        def emit_heartbeat() -> None:
            while not stop.wait(self._heartbeat_interval_seconds):
                elapsed_ms = int((time.monotonic() - request_started_monotonic) * 1000)
                self.run_logger.log(
                    "llm_inflight_heartbeat",
                    request_uuid=request_uuid,
                    client="anthropic",
                    schema_name=schema_name,
                    model=model,
                    api_mode="anthropic_messages",
                    elapsed_ms=elapsed_ms,
                    timeout_seconds=timeout_seconds,
                )

        thread = threading.Thread(
            target=emit_heartbeat,
            name=f"llm-heartbeat-{request_uuid[:8]}",
            daemon=True,
        )
        thread.start()
        return stop, thread


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


def _build_request_payload(
    *,
    selected_model: str,
    temperature: float,
    max_tokens: int,
    instructions: str,
    input_schema: dict[str, Any],
    user_content: str,
    prompt_caching_enabled: bool,
) -> dict[str, Any]:
    if prompt_caching_enabled:
        system: str | list[dict[str, Any]] = [
            {
                "type": "text",
                "text": instructions,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = instructions
    return {
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "system": system,
        "tools": [
            {
                "name": "respond",
                "description": "Respond with structured data matching the requested schema.",
                "input_schema": input_schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": "respond"},
        "messages": [{"role": "user", "content": user_content}],
    }


def _extract_cache_token_usage(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    direct_value = usage.get(key)
    if isinstance(direct_value, int):
        return direct_value
    if key == "cache_creation_input_tokens":
        nested = usage.get("cache_creation")
    elif key == "cache_read_input_tokens":
        nested = usage.get("cache_read")
    else:
        nested = None
    if isinstance(nested, dict):
        return sum(value for value in nested.values() if isinstance(value, int))
    return None


def _is_prompt_cache_rejection(exc: LLMTransportHTTPError) -> bool:
    if exc.status_code not in {400, 404, 422}:
        return False
    message = exc.response_text.lower()
    has_cache_marker = "cache_control" in message or "prompt cach" in message
    has_rejection_marker = any(
        marker in message
        for marker in (
            "unsupported",
            "unknown field",
            "not allowed",
            "invalid request",
            "extra inputs are not permitted",
        )
    )
    return has_cache_marker and has_rejection_marker
