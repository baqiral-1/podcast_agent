"""LangChain-backed LLM client implementation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from podcast_agent.config import LLMConfig, Settings
from podcast_agent.langchain.cache import configure_llm_cache
from podcast_agent.langchain.prompts import build_prompt_template
from podcast_agent.langchain.runnables import (
    RetryableGenerationError,
    TransientLLMError,
    is_json_parse_error,
    is_transient_error,
)
from podcast_agent.llm.base import (
    LLMClient,
    LLMContentFilterError,
    PromptPayload,
    prompt_log_metadata,
)
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.llm.json_utils import normalize_json_content, unwrap_response_payload
from podcast_agent.schemas.models import ChapterAnalysis


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "openai-compatible"
    return normalized


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


def _resolve_model(config: LLMConfig, schema_name: str) -> str:
    return config.model_overrides.get(schema_name, config.model_name)


@dataclass(frozen=True)
class _ProviderTarget:
    provider: str
    model: str
    max_tokens: int | None
    temperature: float
    timeout_seconds: float


def _apply_schema_caps(
    payload: dict[str, Any], response_model: type, schema_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply list maxItems caps before validation for selected schemas."""
    if schema_name != "chapter_summary":
        return payload, []

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return payload, []

    capped_payload = dict(payload)
    capped_analysis = dict(analysis)
    truncations: list[dict[str, Any]] = []

    for name, model_field in ChapterAnalysis.model_fields.items():
        field_value = capped_analysis.get(name)
        if not isinstance(field_value, list):
            continue
        max_length: int | None = None
        for metadata in model_field.metadata:
            metadata_max = getattr(metadata, "max_length", None)
            if isinstance(metadata_max, int):
                max_length = metadata_max
                break
        if max_length is None or len(field_value) <= max_length:
            continue
        capped_analysis[name] = field_value[:max_length]
        truncations.append(
            {
                "path": f"analysis.{name}",
                "original_length": len(field_value),
                "capped_length": max_length,
            }
        )

    if truncations:
        capped_payload["analysis"] = capped_analysis
    return capped_payload, truncations


class LangChainLLMClient(LLMClient):
    """LangChain wrapper implementing the repository LLMClient interface."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.config = settings.llm
        self.prompt_template = build_prompt_template()
        configure_llm_cache(settings.langchain)
        self._model_cache: dict[_ProviderTarget, Any] = {}

    def client_for_schema(self, schema_name: str) -> "LangChainLLMClient":
        # Retain schema-based routing inside generate_json.
        return self

    def with_overrides(self, **updates: Any) -> "LangChainLLMClient":
        next_settings = self.settings.model_copy(
            update={"llm": self.settings.llm.model_copy(update=updates)}
        )
        client = LangChainLLMClient(next_settings)
        if self.run_logger is not None:
            client.set_run_logger(self.run_logger)
        return client

    def _provider_for_schema(self, schema_name: str) -> str:
        overrides = {
            key: _normalize_provider(value) for key, value in self.config.provider_overrides.items()
        }
        default_provider = _resolve_default_provider(self.config)
        return overrides.get(schema_name, default_provider)

    def _build_model(self, schema_name: str) -> Any:
        provider = self._provider_for_schema(schema_name)
        model = self.config.resolve_model(schema_name)
        temperature = self.config.resolve_temperature(schema_name)
        timeout_seconds = self.config.resolve_timeout_seconds(schema_name)
        max_tokens: int | None = None
        if provider == "anthropic":
            max_tokens = self.config.resolve_anthropic_max_tokens(schema_name)
        target = _ProviderTarget(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        cached = self._model_cache.get(target)
        if cached is not None:
            return cached
        if provider == "heuristic":
            model_client = HeuristicLLMClient()
        elif provider == "anthropic":
            if not self.config.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic LLM client.")
            model_client = ChatAnthropic(
                model=model,
                anthropic_api_key=self.config.anthropic_api_key,
                anthropic_api_url=self.config.anthropic_base_url,
                max_tokens=max_tokens,
                default_request_timeout=timeout_seconds,
                temperature=temperature,
                max_retries=0,
            )
        elif provider == "openai-compatible":
            if not self.config.api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the OpenAI LLM client.")
            model_kwargs: dict[str, Any] = {}
            if self.config.reasoning_effort is not None:
                effort = self.config.reasoning_effort.strip().lower()
                if effort != "none":
                    model_kwargs["reasoning"] = {"effort": effort}
            model_client = ChatOpenAI(
                model=model,
                openai_api_key=self.config.api_key,
                openai_api_base=self.config.base_url,
                request_timeout=timeout_seconds,
                temperature=temperature,
                model_kwargs=model_kwargs,
                max_retries=0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider '{provider}'.")
        self._model_cache[target] = model_client
        return model_client

    def _build_messages(
        self,
        *,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type,
    ) -> tuple[str, str, list[Any]]:
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
        messages = self.prompt_template.format_messages(
            system_text=system_text,
            user_text=user_text,
        )
        provider = self._provider_for_schema(schema_name)
        if provider == "anthropic" and self.config.anthropic_prompt_caching_enabled:
            messages = [
                SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                ),
                HumanMessage(content=user_text),
            ]
        return system_text, user_text, messages

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type,
        *,
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> Any:
        model_client = self._build_model(schema_name)
        if isinstance(model_client, HeuristicLLMClient):
            return model_client.generate_json(
                schema_name=schema_name,
                instructions=instructions,
                payload=payload,
                response_model=response_model,
            )
        system_text, user_text, messages = self._build_messages(
            schema_name=schema_name,
            instructions=instructions,
            payload=payload,
            response_model=response_model,
        )
        request_uuid = uuid4().hex
        request_started_at = datetime.now(UTC).isoformat()
        request_started_monotonic = time.monotonic()
        resolved_model = self.config.resolve_model(schema_name)
        if self.run_logger is not None:
            self.run_logger.log(
                "llm_request",
                request_uuid=request_uuid,
                client="langchain",
                schema_name=schema_name,
                model=resolved_model,
                attempt=attempt,
                max_attempts=max_attempts,
                request_started_at=request_started_at,
                timeout_seconds=self.config.resolve_timeout_seconds(schema_name),
                system_text=system_text,
                user_text=user_text,
                **prompt_log_metadata(system_text, user_text),
            )
        try:
            invoke_kwargs: dict[str, Any] = {}
            provider = self._provider_for_schema(schema_name)
            if provider == "openai-compatible" and self.config.openai_prompt_caching_enabled:
                cache_key = f"{schema_name}:{prompt_log_metadata(system_text, user_text)['payload_sha256']}"
                invoke_kwargs["prompt_cache_key"] = cache_key
            response = None
            response_metadata: dict[str, Any] | None = None
            content = None
            if self.config.heartbeat_enabled and hasattr(model_client, "stream"):
                last_heartbeat = request_started_monotonic
                last_usage: dict[str, Any] | None = None
                chunks: list[str] = []
                try:
                    stream_kwargs = dict(invoke_kwargs)
                    if "prompt_cache_key" in stream_kwargs:
                        try:
                            stream_iter = model_client.stream(messages, **stream_kwargs)
                        except TypeError:
                            stream_kwargs.pop("prompt_cache_key", None)
                            stream_iter = model_client.stream(messages, **stream_kwargs)
                    else:
                        stream_iter = model_client.stream(messages, **stream_kwargs)
                    for chunk in stream_iter:
                        chunk_content = getattr(chunk, "content", "")
                        if isinstance(chunk_content, list):
                            chunk_content = "".join(str(part) for part in chunk_content)
                        chunks.append(str(chunk_content))
                        chunk_meta = getattr(chunk, "response_metadata", None)
                        if isinstance(chunk_meta, dict):
                            response_metadata = chunk_meta
                            usage_meta = chunk_meta.get("usage")
                            if isinstance(usage_meta, dict):
                                last_usage = usage_meta
                        now = time.monotonic()
                        if (
                            self.run_logger is not None
                            and now - last_heartbeat >= self.config.heartbeat_interval_seconds
                        ):
                            usage_snapshot = last_usage or {}
                            self.run_logger.log(
                                "llm_heartbeat",
                                request_uuid=request_uuid,
                                client="langchain",
                                schema_name=schema_name,
                                attempt=attempt,
                                elapsed_ms=int((now - request_started_monotonic) * 1000),
                                input_tokens=usage_snapshot.get("input_tokens")
                                or usage_snapshot.get("prompt_tokens"),
                                output_tokens=usage_snapshot.get("output_tokens")
                                or usage_snapshot.get("completion_tokens"),
                                cache_read_tokens=usage_snapshot.get("cache_read_input_tokens"),
                            )
                            last_heartbeat = now
                    content = "".join(chunks)
                except (TypeError, AttributeError):
                    content = None
            if content is None:
                if "prompt_cache_key" in invoke_kwargs:
                    try:
                        response = model_client.invoke(messages, **invoke_kwargs)
                    except TypeError:
                        invoke_kwargs.pop("prompt_cache_key", None)
                        response = model_client.invoke(messages)
                else:
                    response = model_client.invoke(messages, **invoke_kwargs)
                content = response.content
                if isinstance(content, list):
                    content = "".join(str(part) for part in content)
                if response_metadata is None:
                    response_metadata = getattr(response, "response_metadata", {}) or {}
            try:
                normalized_json = normalize_json_content(str(content))
                normalized_payload = unwrap_response_payload(json.loads(normalized_json))
            except Exception as parse_exc:
                if is_json_parse_error(parse_exc):
                    raise RetryableGenerationError(
                        f"JSON parsing failed for {schema_name}: {parse_exc}",
                        data={"raw_content": str(content)},
                    ) from parse_exc
                raise
            normalized_payload, cap_truncations = _apply_schema_caps(
                normalized_payload, response_model, schema_name
            )
            if self.run_logger is not None and cap_truncations:
                self.run_logger.log(
                    "llm_schema_cap_filter",
                    request_uuid=request_uuid,
                    client="langchain",
                    schema_name=schema_name,
                    truncation_count=len(cap_truncations),
                    truncations=cap_truncations,
                )
            if self.run_logger is not None:
                response_metadata = response_metadata or {}
                provider_request_id = (
                    response_metadata.get("request_id")
                    or response_metadata.get("request-id")
                    or response_metadata.get("x-request-id")
                )
                usage = response_metadata.get("usage", {}) if isinstance(response_metadata, dict) else {}
                self.run_logger.log(
                    "llm_response_meta",
                    request_uuid=request_uuid,
                    client="langchain",
                    schema_name=schema_name,
                    attempt=attempt,
                    response_id=response_metadata.get("id") if isinstance(response_metadata, dict) else None,
                    provider_request_id=provider_request_id,
                    model=(
                        response_metadata.get("model", resolved_model)
                        if isinstance(response_metadata, dict)
                        else resolved_model
                    ),
                    elapsed_ms=int((time.monotonic() - request_started_monotonic) * 1000),
                    input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
                    output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
                    cache_read_tokens=usage.get("cache_read_input_tokens"),
                )
                self.run_logger.log(
                    "llm_response",
                    request_uuid=request_uuid,
                    client="langchain",
                    schema_name=schema_name,
                    response=normalized_payload,
                )
            try:
                return response_model.model_validate_json(json.dumps(normalized_payload))
            except Exception as validation_exc:
                raise RetryableGenerationError(
                    f"Schema validation failed for {schema_name}: {validation_exc}",
                    data={"raw_payload": normalized_payload},
                ) from validation_exc
        except (LLMContentFilterError, RetryableGenerationError, TransientLLMError):
            raise
        except Exception as exc:
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_error",
                    request_uuid=request_uuid,
                    client="langchain",
                    schema_name=schema_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    instructions=system_text,
                    payload=user_text,
                )
            if is_transient_error(exc):
                raise TransientLLMError(str(exc)) from exc
            raise


def build_llm_client(settings: Settings) -> LLMClient:
    default_provider = _resolve_default_provider(settings.llm)
    if default_provider == "heuristic":
        return HeuristicLLMClient()
    return LangChainLLMClient(settings)
