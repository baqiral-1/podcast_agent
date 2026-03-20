"""Tests for the Anthropic LLM client."""

from __future__ import annotations

import pytest

from podcast_agent.config import LLMConfig
from podcast_agent.llm.anthropic import AnthropicLLMClient
from podcast_agent.schemas.models import GroundingReport
from podcast_agent.llm.openai_compatible import HTTPResponse


class FakeTransport:
    """Small fake transport for offline client tests."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.last_payload: dict | None = None

    def post_json(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
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
