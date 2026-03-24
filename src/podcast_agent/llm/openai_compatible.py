"""OpenAI-compatible JSON client for agent inference."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from pydantic import BaseModel

from podcast_agent.config import LLMConfig, Settings
from podcast_agent.llm.base import LLMClient, LLMContentFilterError, PromptPayload
from podcast_agent.llm.concurrency import configure_llm_semaphore, llm_semaphore_for


@dataclass(frozen=True)
class HTTPResponse:
    """Normalized HTTP response payload."""

    status_code: int
    body: dict[str, Any]


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
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                return HTTPResponse(status_code=response.status, body=body)
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise LLMTransportHTTPError(status_code=exc.code, response_text=message) from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"LLM request timed out after {timeout_seconds} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"LLM request failed with transport error: {exc}") from exc


class OpenAICompatibleLLMClient(LLMClient):
    """Client that requests schema-constrained JSON from an OpenAI-compatible API."""

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

        if not self.config.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the default OpenAI-compatible LLM client."
            )
        with llm_semaphore_for(schema_name):
            selected_model = self.config.model_overrides.get(schema_name, self.config.model_name)
            endpoint = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
            request_payload = {
                "model": selected_model,
                "temperature": self.config.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{instructions}\n"
                            "Return only a JSON object that matches the requested schema. "
                            "Do not wrap the response in markdown or prose. "
                            "Do not repeat wrapper keys such as schema_name, payload, or expected_schema."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "schema_name": schema_name,
                                "payload": payload,
                                "expected_schema": response_model.model_json_schema(),
                            },
                            default=str,
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_request",
                    client="openai-compatible",
                    schema_name=schema_name,
                    instructions=request_payload["messages"][0]["content"],
                    payload=request_payload["messages"][1]["content"],
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
                        client="openai-compatible",
                        schema_name=schema_name,
                        response=response.body,
                    )
                _raise_for_finish_reason(response.body)
                content = _extract_message_content(response.body)
                normalized_json = _normalize_json_content(content)
                normalized_payload = _unwrap_response_payload(json.loads(normalized_json))
                return response_model.model_validate_json(json.dumps(normalized_payload))
            except Exception as exc:
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_error",
                        client="openai-compatible",
                        schema_name=schema_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                raise


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
                "Use --agent-model AGENT=PROVIDER:MODEL to set both."
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
