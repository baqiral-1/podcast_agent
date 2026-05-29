from __future__ import annotations

from typing import Any

from anthropic import APIStatusError as AnthropicAPIStatusError
import httpx
import pytest
from pydantic import BaseModel, Field

from podcast_agent.agents.chapter_summary import ChapterSummaryResponse
from podcast_agent.config import AgentConfig, LLMConfig, Settings
import podcast_agent.langchain.llm as llm_module
from podcast_agent.langchain.runnables import RetryableGenerationError
from podcast_agent.langchain.llm import (
    _apply_schema_caps,
    _merge_usage,
    _summarize_retryable_error,
)


def test_apply_schema_caps_truncates_chapter_summary_analysis_lists() -> None:
    payload = {
        "analysis": {
            "themes_touched": [f"theme-{idx}" for idx in range(10)],
            "major_actors": [f"actor-{idx}" for idx in range(11)],
        },
    }

    capped, truncations = _apply_schema_caps(
        payload, ChapterSummaryResponse, "chapter_summary"
    )

    analysis = capped["analysis"]
    assert len(analysis["themes_touched"]) == 8
    assert len(analysis["major_actors"]) == 8
    assert any(t["path"] == "analysis.major_actors" for t in truncations)


def test_apply_schema_caps_is_noop_for_non_chapter_summary() -> None:
    class DummyResponse(BaseModel):
        tags: list[str] = Field(default_factory=list, max_length=2)

    payload = {"tags": ["a", "b", "c"]}
    capped, truncations = _apply_schema_caps(payload, DummyResponse, "episode_planning")

    assert capped == payload
    assert truncations == []


def test_summarize_retryable_error_compacts_retry_payload() -> None:
    exc = RetryableGenerationError(
        "JSON parsing failed",
        data={"raw_content": "x" * 20, "raw_payload": {"a": [1, 2, 3]}},
    )
    summary = _summarize_retryable_error(exc)
    assert summary["data_keys"] == ["raw_content", "raw_payload"]
    assert summary["raw_content_chars"] == 20
    assert summary["raw_payload_chars"] > 0


def test_summarize_retryable_error_includes_parse_error_window_middle() -> None:
    raw_content = "".join(chr(65 + (idx % 26)) for idx in range(3000))
    exc = RetryableGenerationError(
        "JSON parsing failed",
        data={
            "raw_content": raw_content,
            "parse_error_line": 22,
            "parse_error_column": 7,
            "parse_error_char": 1500,
        },
    )

    summary = _summarize_retryable_error(exc)

    assert summary["parse_error_line"] == 22
    assert summary["parse_error_column"] == 7
    assert summary["parse_error_char"] == 1500
    assert summary["raw_content_error_window_start"] == 1000
    assert summary["raw_content_error_window_end"] == 2000
    assert summary["raw_content_error_window"] == raw_content[1000:2000]


def test_summarize_retryable_error_includes_parse_error_window_start_boundary() -> None:
    raw_content = "abcdef" * 200
    exc = RetryableGenerationError(
        "JSON parsing failed",
        data={"raw_content": raw_content, "parse_error_char": 120},
    )

    summary = _summarize_retryable_error(exc)

    assert summary["raw_content_error_window_start"] == 0
    assert summary["raw_content_error_window_end"] == 620
    assert summary["raw_content_error_window"] == raw_content[:620]


def test_summarize_retryable_error_includes_parse_error_window_end_boundary() -> None:
    raw_content = "abcdef" * 200
    parse_error_char = len(raw_content) - 80
    exc = RetryableGenerationError(
        "JSON parsing failed",
        data={"raw_content": raw_content, "parse_error_char": parse_error_char},
    )

    summary = _summarize_retryable_error(exc)

    assert summary["raw_content_error_window_start"] == parse_error_char - 500
    assert summary["raw_content_error_window_end"] == len(raw_content)
    assert summary["raw_content_error_window"] == raw_content[parse_error_char - 500 :]


def test_summarize_retryable_error_ignores_non_retryable_error() -> None:
    summary = _summarize_retryable_error(ValueError("boom"))
    assert summary == {}


class _FakeAnthropicModel:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeAnthropicModel.last_kwargs = kwargs


def test_build_model_uses_adaptive_thinking_for_opus_48(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    _FakeAnthropicModel.last_kwargs = None
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            temperature=0.2,
            thinking_budget_tokens={"custom_schema": 20_000},
        )
    )
    client = llm_module.LangChainLLMClient(settings)

    model_client = client._build_model("custom_schema")

    assert isinstance(model_client, _FakeAnthropicModel)
    kwargs = _FakeAnthropicModel.last_kwargs
    assert kwargs is not None
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["model_kwargs"] == {"output_config": {"effort": "high"}}
    assert "temperature" not in kwargs


def test_build_model_omits_thinking_for_opus_48_without_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    _FakeAnthropicModel.last_kwargs = None
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={},
        )
    )
    client = llm_module.LangChainLLMClient(settings)

    model_client = client._build_model("custom_schema")

    assert isinstance(model_client, _FakeAnthropicModel)
    kwargs = _FakeAnthropicModel.last_kwargs
    assert kwargs is not None
    assert "thinking" not in kwargs
    assert "model_kwargs" not in kwargs
    assert "temperature" not in kwargs


def test_build_model_uses_adaptive_thinking_for_opus_47(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: 4.7 must still take the adaptive-thinking path after the
    4.8 migration broadened the predicate."""

    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    _FakeAnthropicModel.last_kwargs = None
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-7",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            temperature=0.2,
            thinking_budget_tokens={"custom_schema": 20_000},
        )
    )
    client = llm_module.LangChainLLMClient(settings)

    model_client = client._build_model("custom_schema")

    assert isinstance(model_client, _FakeAnthropicModel)
    kwargs = _FakeAnthropicModel.last_kwargs
    assert kwargs is not None
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["model_kwargs"] == {"output_config": {"effort": "high"}}
    assert "temperature" not in kwargs


def test_build_model_keeps_legacy_thinking_for_non_opus_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    _FakeAnthropicModel.last_kwargs = None
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={"custom_schema": 20_000},
            agent_configs={
                "custom_schema": AgentConfig(
                    model_name="claude-sonnet-4-6",
                    temperature=0.2,
                )
            },
        )
    )
    client = llm_module.LangChainLLMClient(settings)

    model_client = client._build_model("custom_schema")

    assert isinstance(model_client, _FakeAnthropicModel)
    kwargs = _FakeAnthropicModel.last_kwargs
    assert kwargs is not None
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 20_000}
    assert kwargs["temperature"] == 1.0
    assert "model_kwargs" not in kwargs


class _StubRunLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, event_type: str, **payload: Any) -> None:
        self.events.append((event_type, payload))


def test_build_model_emits_thinking_config_event_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={"custom_schema": 20_000},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)

    client._build_model("custom_schema")
    client._build_model("custom_schema")

    thinking_events = [e for e in run_logger.events if e[0] == "llm_thinking_config"]
    assert len(thinking_events) == 1
    _, payload = thinking_events[0]
    assert payload["mode"] == "adaptive"
    assert payload["effort"] == "high"
    assert payload["budget_tokens"] == 20_000
    assert payload["schema_name"] == "custom_schema"
    assert payload["model"] == "claude-opus-4-8"


def test_build_model_emits_thinking_config_event_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={"custom_schema": 20_000},
            agent_configs={
                "custom_schema": AgentConfig(
                    model_name="claude-sonnet-4-6",
                    temperature=0.2,
                )
            },
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)

    client._build_model("custom_schema")

    thinking_events = [e for e in run_logger.events if e[0] == "llm_thinking_config"]
    assert len(thinking_events) == 1
    _, payload = thinking_events[0]
    assert payload["mode"] == "legacy"
    assert payload["effort"] is None
    assert payload["budget_tokens"] == 20_000


def test_merge_usage_reads_langchain_usage_metadata() -> None:
    class _UM:
        input_tokens = 120
        output_tokens = 350
        input_token_details = {"cache_read": 40, "cache_creation": 10}
        output_token_details = {"reasoning": 2048}

    merged = _merge_usage(_UM(), None)
    assert merged["input_tokens"] == 120
    assert merged["output_tokens"] == 350
    assert merged["thinking_tokens"] == 2048
    assert merged["cache_read_input_tokens"] == 40
    assert merged["cache_creation_input_tokens"] == 10


def test_merge_usage_handles_typeddict_usage_metadata() -> None:
    # langchain_core.messages.UsageMetadata is a TypedDict (runtime: dict);
    # the real-world shape must work, not just attribute-style.
    merged = _merge_usage(
        {
            "input_tokens": 120,
            "output_tokens": 350,
            "input_token_details": {"cache_read": 40, "cache_creation": 10},
            "output_token_details": {"reasoning": 2048},
        },
        None,
    )
    assert merged == {
        "input_tokens": 120,
        "output_tokens": 350,
        "thinking_tokens": 2048,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10,
    }


def test_merge_usage_falls_back_to_raw_usage_dict() -> None:
    merged = _merge_usage(
        None,
        {
            "input_tokens": 5,
            "output_tokens": 7,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 3,
        },
    )
    assert merged["input_tokens"] == 5
    assert merged["output_tokens"] == 7
    assert merged["cache_read_input_tokens"] == 2
    assert merged["cache_creation_input_tokens"] == 3
    # thinking_tokens is NOT in raw fallback — Anthropic SDK does not expose it.
    assert "thinking_tokens" not in merged


class _FakeAIMessageChunk:
    """Minimal AIMessageChunk-like object that supports `+` aggregation."""

    def __init__(
        self,
        content: Any,
        *,
        usage_metadata: Any = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}

    def __add__(self, other: "_FakeAIMessageChunk") -> "_FakeAIMessageChunk":
        merged_content = (
            (self.content if isinstance(self.content, str) else "")
            + (other.content if isinstance(other.content, str) else "")
        )
        # When either side has list content, prefer concatenation of lists.
        if isinstance(self.content, list) or isinstance(other.content, list):
            left = self.content if isinstance(self.content, list) else []
            right = other.content if isinstance(other.content, list) else []
            merged_content = left + right
        return _FakeAIMessageChunk(
            merged_content,
            usage_metadata=other.usage_metadata or self.usage_metadata,
            response_metadata={**self.response_metadata, **other.response_metadata},
        )


class _FakeStreamingClient:
    def __init__(self, chunks: list[_FakeAIMessageChunk]) -> None:
        self._chunks = chunks

    def stream(self, messages: Any, **_: Any) -> Any:
        return iter(self._chunks)


class _FakeFailingStreamingClient:
    def __init__(
        self,
        exc: Exception,
        *,
        chunks: list[_FakeAIMessageChunk] | None = None,
        fail_on_stream: bool = False,
        fail_after_chunks: int = 0,
    ) -> None:
        self._exc = exc
        self._chunks = chunks or []
        self._fail_on_stream = fail_on_stream
        self._fail_after_chunks = fail_after_chunks

    def stream(self, messages: Any, **_: Any) -> Any:
        if self._fail_on_stream:
            raise self._exc

        def _iterator() -> Any:
            emitted = 0
            for chunk in self._chunks:
                if emitted >= self._fail_after_chunks:
                    raise self._exc
                emitted += 1
                yield chunk
            raise self._exc

        return _iterator()


class _FakeInvokeClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def invoke(self, messages: Any, **_: Any) -> Any:
        raise self._exc


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeStructuredAPIStatusError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.type = error_type


def _anthropic_overloaded_error(
    *,
    status_code: int = 529,
    message: str = "provider status failure",
) -> AnthropicAPIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    body = {
        "type": "error",
        "error": {
            "details": None,
            "type": "overloaded_error",
            "message": "Overloaded",
        },
    }
    return AnthropicAPIStatusError(message, response=response, body=body)


class _DummySchema(BaseModel):
    k: int


def _make_client_with_stub_stream(
    chunks: list[_FakeAIMessageChunk],
    monkeypatch: pytest.MonkeyPatch,
    *,
    log_thinking_content: bool = False,
) -> tuple[llm_module.LangChainLLMClient, _StubRunLogger]:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={"custom_schema": 20_000},
            log_thinking_content=log_thinking_content,
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)
    # Swap the cached model for a streaming stub that yields the fake chunks.
    stub = _FakeStreamingClient(chunks)
    monkeypatch.setattr(client, "_build_model", lambda _schema: stub)
    return client, run_logger


def test_generate_json_records_thinking_tokens_from_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk('{"k":'),
        _FakeAIMessageChunk(
            '1}',
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "output_token_details": {"reasoning": 2048},
                "input_token_details": {"cache_read": 10},
            },
            response_metadata={"id": "msg_1", "model": "claude-opus-4-8"},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(chunks, monkeypatch)

    result = client.generate_json(
        schema_name="custom_schema",
        instructions="be terse",
        payload={"user_text": "hello"},
        response_model=_DummySchema,
    )
    assert isinstance(result, _DummySchema) and result.k == 1

    meta_events = [p for (t, p) in run_logger.events if t == "llm_response_meta"]
    assert len(meta_events) == 1
    meta = meta_events[0]
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 50
    assert meta["thinking_tokens"] == 2048
    assert meta["cache_read_tokens"] == 10


def test_generate_json_counts_thinking_by_index_not_by_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # langchain-anthropic streams each thinking_delta / signature_delta as its
    # own AIMessageChunk with `content=[{"type":"thinking","index":N,...}]`.
    # Many deltas for one logical block share an index; we must dedupe.
    chunks = [
        _FakeAIMessageChunk(
            [{"type": "thinking", "thinking": "part one ", "index": 0}]
        ),
        _FakeAIMessageChunk(
            [{"type": "thinking", "thinking": "part two.", "index": 0}]
        ),
        _FakeAIMessageChunk(
            [{"type": "thinking", "signature": "sig-abc", "index": 0}]
        ),
        _FakeAIMessageChunk(
            '{"k":1}',
            usage_metadata={"input_tokens": 50, "output_tokens": 30},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=False
    )

    client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )

    meta = [p for (t, p) in run_logger.events if t == "llm_response_meta"][0]
    assert meta["thinking_block_count"] == 1
    assert meta["thinking_blocks_with_signature"] == 1
    assert meta["thinking_char_count"] == len("part one ") + len("part two.")
    assert meta["redacted_thinking_block_count"] == 0
    assert meta["thinking_signature_sample"] is not None
    assert len(meta["thinking_signature_sample"]) == 16


def test_generate_json_counts_two_distinct_thinking_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk(
            [{"type": "thinking", "thinking": "aaa", "index": 0}]
        ),
        _FakeAIMessageChunk(
            [{"type": "thinking", "signature": "sig-0", "index": 0}]
        ),
        _FakeAIMessageChunk(
            [{"type": "thinking", "thinking": "bbb", "index": 1}]
        ),
        _FakeAIMessageChunk(
            [{"type": "thinking", "signature": "sig-1", "index": 1}]
        ),
        _FakeAIMessageChunk(
            '{"k":1}',
            usage_metadata={"input_tokens": 10, "output_tokens": 10},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=False
    )

    client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )

    meta = [p for (t, p) in run_logger.events if t == "llm_response_meta"][0]
    assert meta["thinking_block_count"] == 2
    assert meta["thinking_blocks_with_signature"] == 2


def test_generate_json_records_redacted_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk(
            [
                {"type": "redacted_thinking", "data": "opaque-ciphertext"},
                {"type": "text", "text": '{"k":1}'},
            ],
            usage_metadata={"input_tokens": 10, "output_tokens": 10},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=False
    )

    # Should not raise — `data` is not treated as text/JSON.
    result = client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )
    assert result.k == 1

    meta = [p for (t, p) in run_logger.events if t == "llm_response_meta"][0]
    assert meta["redacted_thinking_block_count"] == 1
    assert meta["thinking_block_count"] == 0
    assert meta["thinking_blocks_with_signature"] == 0


def test_generate_json_signature_fingerprint_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run_with_signatures(sigs: list[str]) -> dict[str, Any]:
        chunks = [
            _FakeAIMessageChunk(
                [{"type": "thinking", "signature": sig, "index": i}]
            )
            for i, sig in enumerate(sigs)
        ] + [
            _FakeAIMessageChunk(
                '{"k":1}',
                usage_metadata={"input_tokens": 1, "output_tokens": 1},
            ),
        ]
        client, run_logger = _make_client_with_stub_stream(
            chunks, monkeypatch, log_thinking_content=False
        )
        client.generate_json(
            schema_name="custom_schema",
            instructions="x",
            payload={"user_text": "y"},
            response_model=_DummySchema,
        )
        return [p for (t, p) in run_logger.events if t == "llm_response_meta"][0]

    meta_a = _run_with_signatures(["sig-a", "sig-b"])
    meta_b = _run_with_signatures(["sig-a", "sig-b"])
    meta_c = _run_with_signatures(["sig-b", "sig-a"])  # order matters
    assert meta_a["thinking_signature_sample"] == meta_b["thinking_signature_sample"]
    assert meta_a["thinking_signature_sample"] != meta_c["thinking_signature_sample"]


def test_generate_json_logs_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk(
            '{"k":1}',
            usage_metadata={"input_tokens": 5, "output_tokens": 5},
            response_metadata={"stop_reason": "end_turn"},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=False
    )

    client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )

    meta = [p for (t, p) in run_logger.events if t == "llm_response_meta"][0]
    assert meta["stop_reason"] == "end_turn"


def test_generate_json_logs_transient_provider_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)
    monkeypatch.setattr(
        client,
        "_build_model",
        lambda _schema: _FakeInvokeClient(
            _FakeAPIStatusError("Internal server error", status_code=500)
        ),
    )

    with pytest.raises(llm_module.TransientLLMError, match="Internal server error"):
        client.generate_json(
            schema_name="custom_schema",
            instructions="x",
            payload={"user_text": "y"},
            response_model=_DummySchema,
            attempt=1,
            max_attempts=2,
        )

    retryable_events = [p for (t, p) in run_logger.events if t == "llm_retryable_error"]
    error_events = [p for (t, p) in run_logger.events if t == "llm_error"]
    assert len(retryable_events) == 1
    assert retryable_events[0]["error_type"] == "_FakeAPIStatusError"
    assert retryable_events[0]["will_retry"] is True
    assert error_events == []


def test_generate_json_logs_structured_overload_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)
    monkeypatch.setattr(
        client,
        "_build_model",
        lambda _schema: _FakeInvokeClient(
            _FakeStructuredAPIStatusError(
                "provider status failure",
                body={
                    "type": "error",
                    "error": {
                        "details": None,
                        "type": "overloaded_error",
                        "message": "Overloaded",
                    },
                },
            )
        ),
    )

    with pytest.raises(llm_module.TransientLLMError, match="provider status failure"):
        client.generate_json(
            schema_name="custom_schema",
            instructions="x",
            payload={"user_text": "y"},
            response_model=_DummySchema,
            attempt=1,
            max_attempts=2,
        )

    retryable_events = [p for (t, p) in run_logger.events if t == "llm_retryable_error"]
    error_events = [p for (t, p) in run_logger.events if t == "llm_error"]
    assert len(retryable_events) == 1
    assert retryable_events[0]["error_type"] == "_FakeStructuredAPIStatusError"
    assert retryable_events[0]["will_retry"] is True
    assert error_events == []


def test_generate_json_logs_real_anthropic_overload_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)
    monkeypatch.setattr(
        client,
        "_build_model",
        lambda _schema: _FakeInvokeClient(_anthropic_overloaded_error()),
    )

    with pytest.raises(llm_module.TransientLLMError, match="provider status failure"):
        client.generate_json(
            schema_name="custom_schema",
            instructions="x",
            payload={"user_text": "y"},
            response_model=_DummySchema,
            attempt=1,
            max_attempts=2,
        )

    retryable_events = [p for (t, p) in run_logger.events if t == "llm_retryable_error"]
    error_events = [p for (t, p) in run_logger.events if t == "llm_error"]
    assert len(retryable_events) == 1
    assert retryable_events[0]["error_type"] == "APIStatusError"
    assert retryable_events[0]["will_retry"] is True
    assert error_events == []


@pytest.mark.parametrize(
    ("stub", "expected_error_type"),
    [
        (
            lambda: _FakeFailingStreamingClient(
                _anthropic_overloaded_error(),
                fail_on_stream=True,
            ),
            "TransientLLMError",
        ),
        (
            lambda: _FakeFailingStreamingClient(
                _anthropic_overloaded_error(),
                chunks=[],
                fail_after_chunks=0,
            ),
            "TransientLLMError",
        ),
        (
            lambda: _FakeFailingStreamingClient(
                _anthropic_overloaded_error(),
                chunks=[_FakeAIMessageChunk('{"k":1}')],
                fail_after_chunks=1,
            ),
            "TransientLLMError",
        ),
    ],
)
def test_generate_json_logs_streaming_anthropic_overload_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
    stub,
    expected_error_type: str,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)
    monkeypatch.setattr(client, "_build_model", lambda _schema: stub())

    with pytest.raises(llm_module.TransientLLMError, match="provider status failure"):
        client.generate_json(
            schema_name="custom_schema",
            instructions="x",
            payload={"user_text": "y"},
            response_model=_DummySchema,
            attempt=1,
            max_attempts=2,
        )

    retryable_events = [p for (t, p) in run_logger.events if t == "llm_retryable_error"]
    error_events = [p for (t, p) in run_logger.events if t == "llm_error"]
    assert len(retryable_events) == 1
    assert retryable_events[0]["error_type"] == expected_error_type
    assert retryable_events[0]["will_retry"] is True
    assert error_events == []


def test_thinking_config_event_includes_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeAnthropicModel)
    settings = Settings(
        llm=LLMConfig(
            llm_provider="anthropic",
            provider="anthropic",
            model_name="claude-opus-4-8",
            anthropic_api_key="test-key",
            provider_overrides={"custom_schema": "anthropic"},
            thinking_budget_tokens={"custom_schema": 20_000},
        )
    )
    client = llm_module.LangChainLLMClient(settings)
    run_logger = _StubRunLogger()
    client.set_run_logger(run_logger)

    client._build_model("custom_schema")

    evt = next(p for (t, p) in run_logger.events if t == "llm_thinking_config")
    assert evt["display"] == "summarized"


def test_generate_json_captures_thinking_content_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk(
            [
                {"type": "thinking", "thinking": "stepwise reasoning..."},
                {"type": "text", "text": '{"k":1}'},
            ],
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=True
    )

    result = client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )
    assert result.k == 1

    thinking_events = [p for (t, p) in run_logger.events if t == "llm_thinking_content"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["char_count"] == len("stepwise reasoning...")
    assert thinking_events[0]["content_head"] == "stepwise reasoning..."
    assert thinking_events[0]["truncated"] is False


def test_generate_json_skips_thinking_content_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeAIMessageChunk(
            [
                {"type": "thinking", "thinking": "hidden reasoning"},
                {"type": "text", "text": '{"k":1}'},
            ],
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        ),
    ]
    client, run_logger = _make_client_with_stub_stream(
        chunks, monkeypatch, log_thinking_content=False
    )

    client.generate_json(
        schema_name="custom_schema",
        instructions="x",
        payload={"user_text": "y"},
        response_model=_DummySchema,
    )

    thinking_events = [t for (t, _) in run_logger.events if t == "llm_thinking_content"]
    assert thinking_events == []
