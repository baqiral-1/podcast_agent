"""OpenAI-compatible JSON client for agent inference."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
import threading
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel

from podcast_agent.config import LLMConfig, Settings
from podcast_agent.llm.base import LLMClient, LLMContentFilterError, PromptPayload, prompt_log_metadata
from podcast_agent.llm.concurrency import configure_llm_semaphore, llm_semaphore_for


@dataclass(frozen=True)
class HTTPResponse:
    """Normalized HTTP response payload."""

    status_code: int
    body: dict[str, Any]
    response_headers: dict[str, str] | None = None
    elapsed_ms: int | None = None
    time_to_headers_ms: int | None = None
    response_bytes: int | None = None


class LLMTransportHTTPError(RuntimeError):
    """Transport error raised when the OpenAI-compatible API returns an HTTP error."""

    def __init__(self, status_code: int, response_text: str) -> None:
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"LLM request failed with status {status_code}: {response_text}")


class HTTPTransport:
    """Minimal JSON transport for OpenAI-compatible APIs."""

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> HTTPResponse:
        """POST JSON and parse the JSON response body."""

        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(url=url, data=data, headers=headers, method="POST")
        started_at = time.monotonic()
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                headers_ms = int((time.monotonic() - started_at) * 1000)
                response_headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
                raw_body = response.read()
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                body = json.loads(raw_body.decode("utf-8"))
                return HTTPResponse(
                    status_code=response.status,
                    body=body,
                    response_headers=response_headers,
                    elapsed_ms=elapsed_ms,
                    time_to_headers_ms=headers_ms,
                    response_bytes=len(raw_body),
                )
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise LLMTransportHTTPError(status_code=exc.code, response_text=message) from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"LLM request timed out after {timeout_seconds} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"LLM request failed with transport error: {exc}") from exc

    def post_json_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> HTTPResponse:
        """POST JSON and consume an SSE (server-sent events) stream.

        Returns a single HTTPResponse whose body is the reconstructed
        Anthropic Messages response assembled from the streamed events.
        """

        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(url=url, data=data, headers=headers, method="POST")
        started_at = time.monotonic()
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                headers_ms = int((time.monotonic() - started_at) * 1000)
                response_headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
                body, total_bytes = _consume_anthropic_sse(response)
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                return HTTPResponse(
                    status_code=response.status,
                    body=body,
                    response_headers=response_headers,
                    elapsed_ms=elapsed_ms,
                    time_to_headers_ms=headers_ms,
                    response_bytes=total_bytes,
                )
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise LLMTransportHTTPError(status_code=exc.code, response_text=message) from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"LLM request timed out after {timeout_seconds} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"LLM request failed with transport error: {exc}") from exc


def _consume_anthropic_sse(response) -> tuple[dict[str, Any], int]:
    """Read an Anthropic SSE stream and reconstruct the Messages API response body.

    Returns (reconstructed_body, total_bytes_read).
    """

    message: dict[str, Any] = {}
    content_blocks: list[dict[str, Any]] = []
    # Accumulate partial JSON per content block index
    json_accumulators: dict[int, list[str]] = {}
    total_bytes = 0

    for raw_line in response:
        total_bytes += len(raw_line)
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

        if not line.startswith("data: "):
            continue

        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "message_start":
            msg = event.get("message", {})
            message = {
                "id": msg.get("id"),
                "type": msg.get("type", "message"),
                "role": msg.get("role", "assistant"),
                "model": msg.get("model"),
                "usage": msg.get("usage", {}),
            }

        elif event_type == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            # Ensure list is long enough
            while len(content_blocks) <= idx:
                content_blocks.append({})
            content_blocks[idx] = dict(block)
            if block.get("type") == "tool_use":
                json_accumulators[idx] = []

        elif event_type == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "input_json_delta" and idx in json_accumulators:
                json_accumulators[idx].append(delta.get("partial_json", ""))
            elif delta_type == "text_delta" and idx < len(content_blocks):
                content_blocks[idx].setdefault("text", "")
                content_blocks[idx]["text"] += delta.get("text", "")

        elif event_type == "content_block_stop":
            idx = event.get("index", 0)
            if idx in json_accumulators:
                full_json = "".join(json_accumulators.pop(idx))
                if idx < len(content_blocks):
                    try:
                        content_blocks[idx]["input"] = json.loads(full_json)
                    except json.JSONDecodeError:
                        content_blocks[idx]["input"] = full_json

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            if "stop_reason" in delta:
                message["stop_reason"] = delta["stop_reason"]
            usage = event.get("usage", {})
            if usage:
                message.setdefault("usage", {}).update(usage)

    message["content"] = content_blocks
    return message, total_bytes


class OpenAICompatibleLLMClient(LLMClient):
    """Client that requests schema-constrained JSON from an OpenAI-compatible API."""

    _reasoning_excluded_schemas = {"structured_chapter", "grounding_report"}
    _allowed_reasoning_efforts = {"low", "medium", "high", "xhigh"}
    _heartbeat_interval_seconds = 120.0

    def __init__(self, config: LLMConfig, transport: HTTPTransport | None = None) -> None:
        super().__init__()
        self.config = config
        self.transport = transport or HTTPTransport()

    def _should_use_responses_api(self) -> bool:
        parsed = urlparse(self.config.base_url)
        return parsed.netloc == "api.openai.com"

    def _resolve_reasoning_effort(self, schema_name: str) -> str | None:
        if schema_name in self._reasoning_excluded_schemas:
            return None
        raw_effort = self.config.reasoning_effort
        if raw_effort is None:
            return None
        effort = raw_effort.strip().lower()
        if effort == "none":
            return None
        if effort not in self._allowed_reasoning_efforts:
            raise ValueError(
                "Invalid reasoning effort. Expected one of: none, low, medium, high, xhigh."
            )
        return effort

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Request structured JSON and validate it against the response model."""

        if not self.config.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the default OpenAI-compatible LLM client."
            )
        with llm_semaphore_for(schema_name):
            selected_model = self.config.model_overrides.get(schema_name, self.config.model_name)
            reasoning_effort = self._resolve_reasoning_effort(schema_name)
            use_responses_api = self._should_use_responses_api()
            system_text = (
                f"{instructions}\n"
                "Return only a JSON object that matches the requested schema. "
                "Do not wrap the response in markdown or prose. "
                "Do not repeat wrapper keys such as schema_name, payload, or expected_schema."
            )
            user_text = json.dumps(
                {
                    "schema_name": schema_name,
                    "payload": payload,
                    "expected_schema": response_model.model_json_schema(),
                },
                default=str,
            )
            if use_responses_api:
                endpoint = f"{self.config.base_url.rstrip('/')}/v1/responses"
                request_payload = {
                    "model": selected_model,
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                    ],
                    "text": {"format": {"type": "json_object"}},
                }
                if reasoning_effort is not None:
                    request_payload["reasoning"] = {"effort": reasoning_effort}
            else:
                endpoint = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
                request_payload = {
                    "model": selected_model,
                    "temperature": self.config.temperature,
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    "response_format": {"type": "json_object"},
                }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            request_uuid = uuid4().hex
            request_started_at = datetime.now(UTC).isoformat()
            request_started_monotonic = time.monotonic()
            stop_heartbeat, heartbeat_thread = self._start_inflight_heartbeat(
                request_uuid=request_uuid,
                schema_name=schema_name,
                model=selected_model,
                api_mode="responses" if use_responses_api else "chat_completions",
                timeout_seconds=self.config.timeout_seconds,
                request_started_monotonic=request_started_monotonic,
            )
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_request",
                    request_uuid=request_uuid,
                    client="openai-compatible",
                    schema_name=schema_name,
                    model=selected_model,
                    endpoint=endpoint,
                    request_started_at=request_started_at,
                    timeout_seconds=self.config.timeout_seconds,
                    api_mode="responses" if use_responses_api else "chat_completions",
                    reasoning_effort=reasoning_effort or "none",
                    **prompt_log_metadata(system_text, user_text),
                )
            try:
                response = self.transport.post_json(
                    url=endpoint,
                    headers=headers,
                    payload=request_payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
                if self.run_logger is not None:
                    response_body = response.body
                    usage = response_body.get("usage")
                    response_id = response_body.get("id")
                    provider_request_id = (response.response_headers or {}).get("x-request-id")
                    self.run_logger.log(
                        "llm_response_meta",
                        request_uuid=request_uuid,
                        client="openai-compatible",
                        schema_name=schema_name,
                        response_id=response_id,
                        provider_request_id=provider_request_id,
                        status_code=response.status_code,
                        status=response_body.get("status"),
                        model=response_body.get("model", selected_model),
                        response_created_at=response_body.get("created_at"),
                        response_completed_at=response_body.get("completed_at"),
                        elapsed_ms=response.elapsed_ms,
                        time_to_headers_ms=response.time_to_headers_ms,
                        response_bytes=response.response_bytes,
                        prompt_tokens=(usage or {}).get("input_tokens"),
                        completion_tokens=(usage or {}).get("output_tokens"),
                        total_tokens=(usage or {}).get("total_tokens"),
                    )
                    self.run_logger.log(
                        "llm_response",
                        request_uuid=request_uuid,
                        client="openai-compatible",
                        schema_name=schema_name,
                        response=response_body,
                    )
                if not use_responses_api:
                    _raise_for_finish_reason(response.body)
                    content = _extract_message_content(response.body)
                else:
                    content = _extract_responses_content(response.body)
                normalized_json = _normalize_json_content(content)
                normalized_payload = _unwrap_response_payload(json.loads(normalized_json))
                return response_model.model_validate_json(json.dumps(normalized_payload))
            except Exception as exc:
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_error",
                        request_uuid=request_uuid,
                        client="openai-compatible",
                        schema_name=schema_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        instructions=system_text,
                        payload=user_text,
                    )
                raise
            finally:
                stop_heartbeat.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=0.1)

    def _start_inflight_heartbeat(
        self,
        *,
        request_uuid: str,
        schema_name: str,
        model: str,
        api_mode: str,
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
                    client="openai-compatible",
                    schema_name=schema_name,
                    model=model,
                    api_mode=api_mode,
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


class RoutingLLMClient(LLMClient):
    """Routes LLM requests to provider-specific clients."""

    def __init__(
        self,
        *,
        default_provider: str,
        clients: dict[str, LLMClient],
        provider_overrides: dict[str, str],
    ) -> None:
        super().__init__()
        self.default_provider = default_provider
        self.clients = clients
        self.provider_overrides = provider_overrides

    def set_run_logger(self, run_logger: Any) -> None:
        self.run_logger = run_logger
        for client in self.clients.values():
            if hasattr(client, "set_run_logger"):
                client.set_run_logger(run_logger)

    def client_for_schema(self, schema_name: str) -> LLMClient:
        provider = self.provider_overrides.get(schema_name, self.default_provider)
        client = self.clients.get(provider)
        if client is None:
            raise ValueError(f"Unsupported LLM provider '{provider}' for schema '{schema_name}'.")
        return client

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
    ) -> BaseModel:
        return self.client_for_schema(schema_name).generate_json(
            schema_name=schema_name,
            instructions=instructions,
            payload=payload,
            response_model=response_model,
        )


def build_llm_client(settings: Settings) -> LLMClient:
    """Construct the configured LLM client."""

    configure_llm_semaphore(
        _resolve_llm_parallelism(settings),
        per_schema={
            "structured_chapter": settings.pipeline.structuring_parallelism,
            "beat_script": settings.pipeline.beat_parallelism,
            "grounding_report": settings.pipeline.grounding_parallelism,
        },
    )
    config = settings.llm
    default_provider = _resolve_default_provider(config)
    provider_overrides = {
        schema_name: _normalize_provider(provider)
        for schema_name, provider in config.provider_overrides.items()
    }
    if provider_overrides:
        _validate_provider_overrides(provider_overrides, config.model_overrides)
        providers = set(provider_overrides.values())
        providers.add(default_provider)
        clients = {provider: _build_llm_client_for_provider(provider, config) for provider in providers}
        return RoutingLLMClient(
            default_provider=default_provider,
            clients=clients,
            provider_overrides=provider_overrides,
        )
    return _build_llm_client_for_provider(default_provider, config)


def _build_llm_client_for_provider(provider: str, config: LLMConfig) -> LLMClient:
    if provider == "heuristic":
        from podcast_agent.llm.heuristic import HeuristicLLMClient

        return HeuristicLLMClient()
    if provider == "anthropic":
        from podcast_agent.llm.anthropic import AnthropicLLMClient

        return AnthropicLLMClient(config)
    if provider == "openai-compatible":
        return OpenAICompatibleLLMClient(config)
    raise ValueError(f"Unsupported LLM provider '{provider}'.")


def _resolve_default_provider(config: LLMConfig) -> str:
    llm_provider = _normalize_provider(config.llm_provider)
    provider = _normalize_provider(config.provider)
    if llm_provider == "heuristic" or provider == "heuristic":
        return "heuristic"
    if llm_provider == "anthropic" or provider == "anthropic":
        return "anthropic"
    if llm_provider == "openai-compatible" or provider == "openai-compatible":
        return "openai-compatible"
    raise ValueError(f"Unsupported LLM provider '{config.llm_provider}'.")


def _resolve_llm_parallelism(settings: Settings) -> int:
    pipeline = settings.pipeline
    return max(1, pipeline.episode_parallelism)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "openai-compatible"
    return normalized


def _validate_provider_overrides(
    provider_overrides: dict[str, str], model_overrides: dict[str, str]
) -> None:
    supported = {"openai-compatible", "anthropic", "heuristic"}
    for schema_name, provider in provider_overrides.items():
        if provider not in supported:
            raise ValueError(
                f"Unsupported LLM provider '{provider}' for schema '{schema_name}'. "
                f"Supported providers: {', '.join(sorted(supported))}."
            )
        if schema_name not in model_overrides:
            raise ValueError(
                f"Schema '{schema_name}' specifies provider '{provider}' but has no model override. "
                "Set model_overrides and provider_overrides in configuration to set both."
            )


def _extract_message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not choices:
        raise RuntimeError("LLM response did not include any choices.")
    message = choices[0].get("message", {})
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"LLM refused the request: {refusal}")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "".join(text_parts)
    raise RuntimeError("LLM response did not include parseable content.")


def _extract_responses_content(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    outputs = body.get("output", [])
    if not outputs:
        raise RuntimeError("LLM response did not include any output items.")
    text_parts: list[str] = []
    for item in outputs:
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type")
            if block_type in {"output_text", "text"}:
                text_parts.append(block.get("text", ""))
    if text_parts:
        return "".join(text_parts)
    raise RuntimeError("LLM response did not include parseable output text.")


def _raise_for_finish_reason(body: dict[str, Any]) -> None:
    choices = body.get("choices")
    if not choices:
        return
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "content_filter":
        raise LLMContentFilterError("LLM response was blocked by content filtering.")
    if finish_reason == "length":
        raise RuntimeError("LLM response was truncated because it hit the completion token limit.")


def _normalize_json_content(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("LLM response was not valid JSON.") from None
        cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response JSON must be an object.")
    return json.dumps(parsed)


def _unwrap_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "payload" in payload and isinstance(payload["payload"], dict):
        wrapped_payload = payload["payload"]
        if "draft" in wrapped_payload and isinstance(wrapped_payload["draft"], dict):
            return wrapped_payload["draft"]
        return wrapped_payload
    if "result" in payload and isinstance(payload["result"], dict):
        return payload["result"]
    return payload
