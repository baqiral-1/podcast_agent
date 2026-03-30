"""Tests for the Anthropic LLM client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_agent.config import LLMConfig
from podcast_agent.llm.anthropic import AnthropicLLMClient
from podcast_agent.schemas.models import GroundingReport
from podcast_agent.llm.openai_compatible import (
    HTTPResponse,
    LLMTransportHTTPError,
    _consume_anthropic_sse,
)
from podcast_agent.run_logging import RunLogger


class FakeTransport:
    """Small fake transport for offline client tests."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.last_payload: dict | None = None

    def post_json(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
        self.last_payload = payload
        return HTTPResponse(status_code=200, body=self.body)

    def post_json_sse(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
        self.last_payload = payload
        return HTTPResponse(status_code=200, body=self.body)


def test_anthropic_client_parses_schema_constrained_json() -> None:
    transport = FakeTransport(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_fake_id",
                    "name": "respond",
                    "input": {
                        "episode_id": "episode-1",
                        "overall_status": "pass",
                        "claim_assessments": [],
                        "validated_at": "2026-03-13T00:00:00Z",
                    },
                }
            ],
            "stop_reason": "tool_use",
        }
    )
    client = AnthropicLLMClient(
        config=LLMConfig(anthropic_api_key="test-key", model_name="claude-test"),
        transport=transport,
    )

    result = client.generate_json(
        schema_name="grounding_report",
        instructions="Validate the claims.",
        payload={"script": {"episode_id": "episode-1"}},
        response_model=GroundingReport,
    )

    assert result.episode_id == "episode-1"
    assert transport.last_payload is not None
    assert transport.last_payload["model"] == "claude-test"
    assert "tools" in transport.last_payload
    assert transport.last_payload["tool_choice"] == {"type": "tool", "name": "respond"}
    system_blocks = transport.last_payload["system"]
    assert isinstance(system_blocks, list)
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_client_raises_for_max_tokens_stop_reason() -> None:
    transport = FakeTransport(
        {
            "content": [{"type": "text", "text": '{"episode_id":"episode-1"}'}],
            "stop_reason": "max_tokens",
        }
    )
    client = AnthropicLLMClient(
        config=LLMConfig(anthropic_api_key="test-key", model_name="claude-test"),
        transport=transport,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate_json(
            schema_name="grounding_report",
            instructions="Validate the claims.",
            payload={"script": {"episode_id": "episode-1"}},
            response_model=GroundingReport,
        )

    assert "completion token limit" in str(exc_info.value)


def test_anthropic_client_handles_quotes_in_reason_field() -> None:
    """Regression test: reason fields with literal double-quotes must not cause JSONDecodeError."""
    transport = FakeTransport(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_fake_id",
                    "name": "respond",
                    "input": {
                        "episode_id": "episode-1",
                        "overall_status": "fail",
                        "claim_assessments": [
                            {
                                "claim_id": "c1",
                                "status": "unsupported",
                                "reason": 'the shikastah (literally "broken writing") script',
                                "evidence_chunk_ids": [],
                            }
                        ],
                        "validated_at": "2026-03-13T00:00:00Z",
                    },
                }
            ],
            "stop_reason": "tool_use",
        }
    )
    client = AnthropicLLMClient(
        config=LLMConfig(anthropic_api_key="test-key", model_name="claude-test"),
        transport=transport,
    )

    result = client.generate_json(
        schema_name="grounding_report",
        instructions="Validate.",
        payload={},
        response_model=GroundingReport,
    )

    assert '"broken writing"' in result.claim_assessments[0].reason


def test_anthropic_client_logs_prompt_metadata_for_success(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "usage": {
                "input_tokens": 123,
                "output_tokens": 45,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 60,
            },
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_fake_id",
                    "name": "respond",
                    "input": {
                        "episode_id": "episode-1",
                        "overall_status": "pass",
                        "claim_assessments": [],
                        "validated_at": "2026-03-13T00:00:00Z",
                    },
                }
            ],
            "stop_reason": "tool_use",
        }
    )
    run_logger = RunLogger(tmp_path / "runs")
    run_logger.bind_run("anthropic-success")
    client = AnthropicLLMClient(
        config=LLMConfig(anthropic_api_key="test-key", model_name="claude-test"),
        transport=transport,
    )
    client.set_run_logger(run_logger)

    client.generate_json(
        schema_name="grounding_report",
        instructions="Validate the claims.",
        payload={"script": {"episode_id": "episode-1"}},
        response_model=GroundingReport,
    )

    lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "anthropic-success" / "run.log").read_text(encoding="utf-8").splitlines()
    ]
    request_events = [line for line in lines if line["event_type"] == "llm_request"]
    response_meta_events = [line for line in lines if line["event_type"] == "llm_response_meta"]

    assert request_events
    assert response_meta_events
    payload = request_events[-1]["payload"]
    assert "instructions" not in payload
    assert "payload" not in payload
    assert payload["instructions_char_count"] > 0
    assert payload["payload_char_count"] > 0
    assert len(payload["instructions_sha256"]) == 64
    assert len(payload["payload_sha256"]) == 64
    assert payload["request_uuid"]
    assert payload["request_started_at"]
    assert payload["endpoint"].endswith("/v1/messages")
    response_meta_payload = response_meta_events[-1]["payload"]
    assert response_meta_payload["request_uuid"] == payload["request_uuid"]
    assert response_meta_payload["prompt_caching_enabled"] is True
    assert response_meta_payload["cache_creation_input_tokens"] == 30
    assert response_meta_payload["cache_read_input_tokens"] == 60


def test_anthropic_client_logs_full_prompt_payload_on_error(tmp_path: Path) -> None:
    class TimeoutTransport:
        def post_json_sse(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
            raise TimeoutError("timed out")

    run_logger = RunLogger(tmp_path / "runs")
    run_logger.bind_run("anthropic-error")
    client = AnthropicLLMClient(
        config=LLMConfig(anthropic_api_key="test-key", model_name="claude-test"),
        transport=TimeoutTransport(),
    )
    client.set_run_logger(run_logger)

    with pytest.raises(TimeoutError):
        client.generate_json(
            schema_name="grounding_report",
            instructions="Validate the claims.",
            payload={"script": {"episode_id": "episode-1"}},
            response_model=GroundingReport,
        )

    lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "anthropic-error" / "run.log").read_text(encoding="utf-8").splitlines()
    ]
    error_events = [line for line in lines if line["event_type"] == "llm_error"]

    assert error_events
    payload = error_events[-1]["payload"]
    assert payload["instructions"] == "Validate the claims."
    assert '"schema_name": "grounding_report"' in payload["payload"]
    assert payload["request_uuid"]


def test_anthropic_client_falls_back_when_cache_control_rejected() -> None:
    class RejectThenSucceedTransport:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def post_json_sse(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
            self.calls.append(payload)
            if len(self.calls) == 1:
                raise LLMTransportHTTPError(
                    status_code=400,
                    response_text='{"error":{"type":"invalid_request_error","message":"cache_control is not allowed"}}',
                )
            return HTTPResponse(
                status_code=200,
                body={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_fake_id",
                            "name": "respond",
                            "input": {
                                "episode_id": "episode-1",
                                "overall_status": "pass",
                                "claim_assessments": [],
                                "validated_at": "2026-03-13T00:00:00Z",
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                },
            )

    transport = RejectThenSucceedTransport()
    client = AnthropicLLMClient(
        config=LLMConfig(
            anthropic_api_key="test-key",
            model_name="claude-test",
            anthropic_prompt_caching_enabled=True,
            anthropic_prompt_caching_auto_fallback=True,
        ),
        transport=transport,
    )

    result = client.generate_json(
        schema_name="grounding_report",
        instructions="Validate.",
        payload={"script": {"episode_id": "episode-1"}},
        response_model=GroundingReport,
    )

    assert result.episode_id == "episode-1"
    assert len(transport.calls) == 2
    assert isinstance(transport.calls[0]["system"], list)
    assert transport.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(transport.calls[1]["system"], str)

    second_result = client.generate_json(
        schema_name="grounding_report",
        instructions="Validate.",
        payload={"script": {"episode_id": "episode-2"}},
        response_model=GroundingReport,
    )
    assert second_result.episode_id == "episode-1"
    assert len(transport.calls) == 3
    assert isinstance(transport.calls[2]["system"], str)


class FakeSSEStream:
    """In-memory iterable that simulates an HTTP response with SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)


def test_consume_anthropic_sse_reconstructs_tool_use_response() -> None:
    """Verify that streaming SSE events are reassembled into the same body shape as a non-streaming response."""
    sse_lines = [
        'data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-opus-4-6","usage":{"input_tokens":500}}}\n',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"respond","input":""}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"episode_id\\""}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": \\"ep-1\\","}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"overall_status\\": \\"pass\\","}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"claim_assessments\\": [],"}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"validated_at\\": \\"2026-03-13T00:00:00Z\\"}"}}\n',
        'data: {"type":"content_block_stop","index":0}\n',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":150}}\n',
        'data: {"type":"message_stop"}\n',
    ]
    stream = FakeSSEStream(sse_lines)
    body, total_bytes = _consume_anthropic_sse(stream)

    assert body["id"] == "msg_123"
    assert body["model"] == "claude-opus-4-6"
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] == 500
    assert body["usage"]["output_tokens"] == 150
    assert len(body["content"]) == 1
    block = body["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "respond"
    assert block["input"]["episode_id"] == "ep-1"
    assert block["input"]["overall_status"] == "pass"
    assert block["input"]["claim_assessments"] == []
