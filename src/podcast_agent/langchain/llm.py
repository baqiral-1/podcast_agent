"""LangChain-backed LLM client implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)
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


def _is_claude_opus_4_7(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized == "claude-opus-4-7" or normalized.startswith("claude-opus-4-7-")


@dataclass(frozen=True)
class _ProviderTarget:
    provider: str
    model: str
    max_tokens: int | None
    temperature: float | None
    timeout_seconds: float
    thinking_budget: int | None = None
    thinking_effort: str | None = None


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Field access that works on both TypedDicts (dicts) and plain objects."""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class _ThinkingState:
    """Accumulates per-request signals that prove extended thinking ran.

    langchain-anthropic 1.4.0 emits each `thinking_delta` and each
    `signature_delta` as a separate `AIMessageChunk` whose `content` is a
    single-element list like `{"type":"thinking","index":N,...}`. Multiple
    deltas for one logical thinking block share an `index`. We deduplicate by
    index so `block_count` reflects distinct blocks, not delta events.
    """

    indices_seen: set[int] = field(default_factory=set)
    indices_signed: set[int] = field(default_factory=set)
    signatures: list[str] = field(default_factory=list)
    char_count: int = 0
    redacted_count: int = 0
    buffer: list[str] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.indices_seen)

    @property
    def signed_count(self) -> int:
        return len(self.indices_signed)

    def signature_fingerprint(self) -> str | None:
        if not self.signatures:
            return None
        joined = "".join(self.signatures).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:16]


def _resolve_stop_reason(
    response_metadata: dict[str, Any] | None, response: Any
) -> str | None:
    """Pull `stop_reason` from wherever langchain put it.

    Streaming: langchain-anthropic sets `response_metadata["stop_reason"]` on
    the final delta chunk. Non-streaming: the field lands in the AIMessage's
    `response_metadata` after langchain-core merges `llm_output`. We check
    both defensively.
    """

    if isinstance(response_metadata, dict):
        stop_reason = response_metadata.get("stop_reason")
        if stop_reason is not None:
            return stop_reason
    if response is not None:
        resp_meta = getattr(response, "response_metadata", None)
        if isinstance(resp_meta, dict):
            stop_reason = resp_meta.get("stop_reason")
            if stop_reason is not None:
                return stop_reason
    return None


def _extract_thinking_part(
    part: Any, fallback_index: int
) -> tuple[str | None, str | None, str | None, int]:
    """Return (part_type, thinking_text, signature, index) from a content part."""

    if isinstance(part, dict):
        part_type = part.get("type")
        thinking_text = part.get("thinking") or part.get("text")
        signature = part.get("signature")
        index = part.get("index", fallback_index)
    else:
        part_type = getattr(part, "type", None)
        thinking_text = getattr(part, "thinking", None) or getattr(part, "text", None)
        signature = getattr(part, "signature", None)
        index = getattr(part, "index", fallback_index)
    try:
        index = int(index) if index is not None else fallback_index
    except (TypeError, ValueError):
        index = fallback_index
    return part_type, thinking_text, signature, index


def _filter_and_track_thinking(
    parts: list[Any],
    state: _ThinkingState,
    *,
    log_content: bool,
) -> str:
    """Strip thinking / redacted_thinking blocks from a content-list chunk
    while updating state. Returns the text portion joined as a string."""

    text_parts: list[str] = []
    for idx, part in enumerate(parts):
        part_type, thinking_text, signature, index = _extract_thinking_part(
            part, idx
        )
        if part_type == "thinking":
            state.indices_seen.add(index)
            if thinking_text:
                state.char_count += len(thinking_text)
                if log_content:
                    state.buffer.append(str(thinking_text))
            if signature:
                state.indices_signed.add(index)
                state.signatures.append(str(signature))
            continue
        if part_type == "redacted_thinking":
            state.redacted_count += 1
            continue
        part_text = (
            part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        )
        text_parts.append(part_text if part_text is not None else str(part))
    return "".join(text_parts)


def _merge_usage(
    usage_metadata_obj: Any, raw_usage: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge langchain-anthropic 1.x `usage_metadata` with raw Anthropic `usage` dict.

    `usage_metadata` in langchain 1.x is a `TypedDict` (a `dict` at runtime);
    field access is via `obj["key"]`, not `obj.key`. We also look at
    `response_metadata["usage"]` defensively in case a future langchain
    version mirrors extra fields there.

    NOTE: Anthropic does NOT expose a separate thinking-token counter. Per
    the extended-thinking docs, thinking tokens are counted inside
    `usage.output_tokens` and billing is based on those. The `thinking_tokens`
    field this helper may produce is therefore always `None` against
    langchain-anthropic 1.4.0 + Anthropic SDK 0.87.0 — it exists only to
    auto-pick up a future LC `output_token_details["reasoning"]` key if
    Anthropic ever adds one. Use thinking *block count*, *signature*, and
    *char count* (tracked separately in `_ThinkingState`) as the empirical
    proof that thinking ran.
    """
    out: dict[str, Any] = {}
    if usage_metadata_obj is not None:
        input_tokens = _get_field(usage_metadata_obj, "input_tokens")
        output_tokens = _get_field(usage_metadata_obj, "output_tokens")
        if input_tokens is not None:
            out["input_tokens"] = input_tokens
        if output_tokens is not None:
            out["output_tokens"] = output_tokens
        otd = _get_field(usage_metadata_obj, "output_token_details") or {}
        itd = _get_field(usage_metadata_obj, "input_token_details") or {}
        reasoning = _get_field(otd, "reasoning")
        if reasoning is not None:
            out["thinking_tokens"] = reasoning
        cache_read = _get_field(itd, "cache_read")
        if cache_read is not None:
            out["cache_read_input_tokens"] = cache_read
        cache_creation = _get_field(itd, "cache_creation")
        if cache_creation is not None:
            out["cache_creation_input_tokens"] = cache_creation
    raw = raw_usage if isinstance(raw_usage, dict) else {}
    for src_key, dst_key in (
        ("input_tokens", "input_tokens"),
        ("prompt_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("completion_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
    ):
        if dst_key in out:
            continue
        value = raw.get(src_key)
        if value is not None:
            out[dst_key] = value
    return out


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


def _summarize_retryable_error(exc: Exception) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if isinstance(exc, RetryableGenerationError):
        data = exc.data or {}
        if data:
            summary["data_keys"] = sorted(str(key) for key in data.keys())
        raw_content = data.get("raw_content")
        parse_error_line = data.get("parse_error_line")
        if isinstance(parse_error_line, int):
            summary["parse_error_line"] = parse_error_line
        parse_error_column = data.get("parse_error_column")
        if isinstance(parse_error_column, int):
            summary["parse_error_column"] = parse_error_column
        parse_error_char = data.get("parse_error_char")
        if isinstance(parse_error_char, int):
            summary["parse_error_char"] = parse_error_char
        if isinstance(raw_content, str):
            summary["raw_content_chars"] = len(raw_content)
            summary["raw_content_head"] = raw_content[:2000]
            if isinstance(parse_error_char, int):
                window_start = max(0, parse_error_char - 500)
                window_end = min(len(raw_content), parse_error_char + 500)
                summary["raw_content_error_window_start"] = window_start
                summary["raw_content_error_window_end"] = window_end
                summary["raw_content_error_window"] = raw_content[window_start:window_end]
        raw_response_type = data.get("raw_response_type")
        if raw_response_type:
            summary["raw_response_type"] = raw_response_type
        raw_response_parts = data.get("raw_response_parts")
        if raw_response_parts:
            summary["raw_response_parts"] = raw_response_parts
        raw_payload = data.get("raw_payload")
        if raw_payload is not None:
            try:
                summary["raw_payload_chars"] = len(json.dumps(raw_payload, default=str))
            except Exception:
                summary["raw_payload_type"] = type(raw_payload).__name__
    return summary


class LangChainLLMClient(LLMClient):
    """LangChain wrapper implementing the repository LLMClient interface."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.config = settings.llm
        self.prompt_template = build_prompt_template()
        configure_llm_cache(settings.langchain)
        self._model_cache: dict[_ProviderTarget, Any] = {}
        self._thinking_logged: set[tuple[str, _ProviderTarget]] = set()

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
        temperature_for_target: float | None = temperature
        timeout_seconds = self.config.resolve_timeout_seconds(schema_name)
        max_tokens: int | None = None
        thinking_budget: int | None = None
        thinking_effort: str | None = None
        legacy_budget_for_effort: int | None = None
        model_supports_adaptive_thinking = False
        if provider == "anthropic":
            max_tokens = self.config.resolve_anthropic_max_tokens(schema_name)
            model_supports_adaptive_thinking = _is_claude_opus_4_7(model)
            if model_supports_adaptive_thinking:
                legacy_budget_for_effort = self.config.resolve_thinking_budget(schema_name)
                thinking_effort = self.config.resolve_anthropic_thinking_effort(schema_name)
                temperature_for_target = None
            else:
                thinking_budget = self.config.resolve_thinking_budget(schema_name)
            if thinking_budget is not None and temperature != 1.0:
                logger.warning(
                    "Extended thinking requires temperature=1.0; overriding %.1f for %s",
                    temperature, schema_name,
                )
                temperature = 1.0
                temperature_for_target = 1.0
        target = _ProviderTarget(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature_for_target,
            timeout_seconds=timeout_seconds,
            thinking_budget=thinking_budget,
            thinking_effort=thinking_effort,
        )
        self._log_thinking_config(
            schema_name,
            target,
            legacy_budget_for_effort=legacy_budget_for_effort,
        )
        cached = self._model_cache.get(target)
        if cached is not None:
            return cached
        if provider == "heuristic":
            model_client = HeuristicLLMClient()
        elif provider == "anthropic":
            if not self.config.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic LLM client.")
            anthropic_kwargs: dict[str, Any] = {
                "model": model,
                "anthropic_api_key": self.config.anthropic_api_key,
                "anthropic_api_url": self.config.anthropic_base_url,
                "max_tokens": max_tokens,
                "default_request_timeout": timeout_seconds,
                "max_retries": 0,
            }
            if not model_supports_adaptive_thinking:
                anthropic_kwargs["temperature"] = temperature
            if model_supports_adaptive_thinking and thinking_effort is not None:
                anthropic_kwargs["thinking"] = {"type": "adaptive"}
                anthropic_kwargs["model_kwargs"] = {
                    "output_config": {"effort": thinking_effort}
                }
                logger.info(
                    "Adaptive thinking enabled for %s with effort %s",
                    schema_name,
                    thinking_effort,
                )
                if (
                    legacy_budget_for_effort is not None
                    and schema_name
                    not in self.config.anthropic_thinking_effort_overrides
                ):
                    logger.warning(
                        "Using deprecated thinking_budget_tokens for %s; "
                        "mapped budget %d to adaptive effort %s.",
                        schema_name,
                        legacy_budget_for_effort,
                        thinking_effort,
                    )
            elif thinking_budget is not None:
                anthropic_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                logger.info(
                    "Extended thinking enabled for %s with budget %d tokens",
                    schema_name, thinking_budget,
                )
            model_client = ChatAnthropic(**anthropic_kwargs)
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

    def _log_thinking_config(
        self,
        schema_name: str,
        target: _ProviderTarget,
        *,
        legacy_budget_for_effort: int | None = None,
    ) -> None:
        if self.run_logger is None or target.provider != "anthropic":
            return
        key = (schema_name, target)
        if key in self._thinking_logged:
            return
        self._thinking_logged.add(key)
        if target.thinking_effort is not None:
            mode = "adaptive"
            budget_tokens = legacy_budget_for_effort
        elif target.thinking_budget is not None:
            mode = "legacy"
            budget_tokens = target.thinking_budget
        else:
            mode = "disabled"
            budget_tokens = None
        self.run_logger.log(
            "llm_thinking_config",
            schema_name=schema_name,
            model=target.model,
            mode=mode,
            effort=target.thinking_effort,
            budget_tokens=budget_tokens,
            # We never pass the `display` key on the thinking config; the
            # Anthropic API default is "summarized" (thinking summaries are
            # returned in content). Logging the effective value avoids
            # ambiguity for log readers.
            display="summarized" if mode != "disabled" else None,
        )

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
            last_usage: dict[str, Any] = {}
            thinking_state = _ThinkingState()
            log_thinking_content = bool(
                getattr(self.config, "log_thinking_content", False)
            )
            if self.config.heartbeat_enabled and hasattr(model_client, "stream"):
                last_heartbeat = request_started_monotonic
                chunks: list[str] = []
                aggregated_chunk: Any = None
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
                        try:
                            aggregated_chunk = (
                                chunk if aggregated_chunk is None else aggregated_chunk + chunk
                            )
                        except Exception:
                            aggregated_chunk = chunk
                        chunk_content = getattr(chunk, "content", "")
                        if isinstance(chunk_content, list):
                            chunk_content = _filter_and_track_thinking(
                                chunk_content,
                                thinking_state,
                                log_content=log_thinking_content,
                            )
                        chunks.append(str(chunk_content))
                        chunk_meta = getattr(chunk, "response_metadata", None)
                        if isinstance(chunk_meta, dict):
                            response_metadata = chunk_meta
                        merged_usage = _merge_usage(
                            getattr(aggregated_chunk, "usage_metadata", None),
                            (
                                getattr(aggregated_chunk, "response_metadata", None)
                                or {}
                            ).get("usage")
                            if isinstance(
                                getattr(aggregated_chunk, "response_metadata", None),
                                dict,
                            )
                            else None,
                        )
                        if merged_usage:
                            last_usage = merged_usage
                        now = time.monotonic()
                        if (
                            self.run_logger is not None
                            and now - last_heartbeat >= self.config.heartbeat_interval_seconds
                        ):
                            self.run_logger.log(
                                "llm_heartbeat",
                                request_uuid=request_uuid,
                                client="langchain",
                                schema_name=schema_name,
                                attempt=attempt,
                                elapsed_ms=int((now - request_started_monotonic) * 1000),
                                input_tokens=last_usage.get("input_tokens"),
                                output_tokens=last_usage.get("output_tokens"),
                                cache_read_tokens=last_usage.get("cache_read_input_tokens"),
                                thinking_tokens=last_usage.get("thinking_tokens"),
                            )
                            last_heartbeat = now
                    content = "".join(chunks)
                    if aggregated_chunk is not None:
                        merged_usage = _merge_usage(
                            getattr(aggregated_chunk, "usage_metadata", None),
                            (
                                getattr(aggregated_chunk, "response_metadata", None)
                                or {}
                            ).get("usage")
                            if isinstance(
                                getattr(aggregated_chunk, "response_metadata", None),
                                dict,
                            )
                            else None,
                        )
                        if merged_usage:
                            last_usage = merged_usage
                        agg_meta = getattr(aggregated_chunk, "response_metadata", None)
                        if isinstance(agg_meta, dict):
                            response_metadata = agg_meta
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
                    content = _filter_and_track_thinking(
                        content,
                        thinking_state,
                        log_content=log_thinking_content,
                    )
                if response_metadata is None:
                    response_metadata = getattr(response, "response_metadata", {}) or {}
                merged_usage = _merge_usage(
                    getattr(response, "usage_metadata", None),
                    response_metadata.get("usage")
                    if isinstance(response_metadata, dict)
                    else None,
                )
                if merged_usage:
                    last_usage = merged_usage
            try:
                normalized_json = normalize_json_content(str(content))
                normalized_payload = unwrap_response_payload(json.loads(normalized_json))
            except Exception as parse_exc:
                if is_json_parse_error(parse_exc):
                    # Capture the original response structure for debugging
                    raw_response = getattr(response, "content", None) if response else None
                    raw_structure = None
                    if isinstance(raw_response, list):
                        raw_structure = [
                            {
                                "type": p.get("type") if isinstance(p, dict) else getattr(p, "type", "?"),
                                "text_head": (
                                    (p.get("text") if isinstance(p, dict) else getattr(p, "text", ""))
                                    or ""
                                )[:200],
                            }
                            for p in raw_response
                        ]
                    parse_error_line = None
                    parse_error_column = None
                    parse_error_char = None
                    if isinstance(parse_exc, json.JSONDecodeError):
                        parse_error_line = parse_exc.lineno
                        parse_error_column = parse_exc.colno
                        parse_error_char = parse_exc.pos
                    raise RetryableGenerationError(
                        f"JSON parsing failed for {schema_name}: {parse_exc}",
                        data={
                            "raw_content": str(content),
                            "parse_error_line": parse_error_line,
                            "parse_error_column": parse_error_column,
                            "parse_error_char": parse_error_char,
                            "raw_response_type": type(raw_response).__name__ if raw_response is not None else "none",
                            "raw_response_parts": raw_structure,
                        },
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
                if log_thinking_content and thinking_state.buffer:
                    joined_thinking = "".join(thinking_state.buffer)
                    self.run_logger.log(
                        "llm_thinking_content",
                        request_uuid=request_uuid,
                        client="langchain",
                        schema_name=schema_name,
                        char_count=len(joined_thinking),
                        content_head=joined_thinking[:2000],
                        truncated=len(joined_thinking) > 2000,
                    )
                stop_reason = _resolve_stop_reason(response_metadata, response)
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
                    input_tokens=last_usage.get("input_tokens"),
                    output_tokens=last_usage.get("output_tokens"),
                    cache_read_tokens=last_usage.get("cache_read_input_tokens"),
                    cache_creation_tokens=last_usage.get("cache_creation_input_tokens"),
                    thinking_tokens=last_usage.get("thinking_tokens"),
                    thinking_block_count=thinking_state.block_count,
                    thinking_blocks_with_signature=thinking_state.signed_count,
                    thinking_char_count=thinking_state.char_count,
                    redacted_thinking_block_count=thinking_state.redacted_count,
                    thinking_signature_sample=thinking_state.signature_fingerprint(),
                    stop_reason=stop_reason,
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
        except (LLMContentFilterError, RetryableGenerationError, TransientLLMError) as exc:
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_retryable_error",
                    request_uuid=request_uuid,
                    client="langchain",
                    schema_name=schema_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    will_retry=attempt < max_attempts,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    **_summarize_retryable_error(exc),
                )
            raise
        except Exception as exc:
            if is_transient_error(exc):
                if self.run_logger is not None:
                    self.run_logger.log(
                        "llm_retryable_error",
                        request_uuid=request_uuid,
                        client="langchain",
                        schema_name=schema_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        will_retry=attempt < max_attempts,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                raise TransientLLMError(str(exc)) from exc
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
            raise


def build_llm_client(settings: Settings) -> LLMClient:
    default_provider = _resolve_default_provider(settings.llm)
    if default_provider == "heuristic":
        return HeuristicLLMClient()
    return LangChainLLMClient(settings)
