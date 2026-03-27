"""Tests for the OpenAI-compatible LLM and TTS clients."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from podcast_agent.config import LLMConfig, TTSConfig
from podcast_agent.llm.base import LLMContentFilterError
from podcast_agent.llm.openai_compatible import (
    HTTPResponse,
    LLMTransportHTTPError,
    OpenAICompatibleLLMClient,
)
from podcast_agent.schemas.models import GroundingReport
from podcast_agent.tts.openai_compatible import BinaryHTTPTransport, OpenAICompatibleTTSClient


class FakeTransport:
    """Small fake transport for offline client tests."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.last_payload: dict | None = None

    def post_json(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
        self.last_payload = payload
        return HTTPResponse(status_code=200, body=self.body)


class FakeBinaryTransport(BinaryHTTPTransport):
    """Fake binary transport for speech synthesis tests."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.last_payload: dict | None = None

    def post_json_for_bytes(self, url: str, headers: dict[str, str], payload: dict, timeout_seconds: float) -> bytes:
        self.last_payload = payload
        return self.data


class FailingHTTPTransport:
    """Fake transport that raises a typed HTTP transport error."""

    def post_json(self, url: str, headers: dict, payload: dict, timeout_seconds: float) -> HTTPResponse:
        raise LLMTransportHTTPError(status_code=400, response_text='{"error":"parse error"}')


def test_openai_compatible_client_parses_schema_constrained_json() -> None:
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"episode_id":"episode-1","overall_status":"pass",'
                                '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}'
                            ),
                        }
                    ],
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key"),
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
    assert transport.last_payload["text"]["format"]["type"] == "json_object"


def test_openai_compatible_client_unwraps_payload_echoes() -> None:
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"schema_name":"grounding_report","payload":'
                                '{"episode_id":"episode-1","overall_status":"pass",'
                                '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}}'
                            ),
                        }
                    ],
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key"),
        transport=transport,
    )

    result = client.generate_json(
        schema_name="grounding_report",
        instructions="Validate the claims.",
        payload={"script": {"episode_id": "episode-1"}},
        response_model=GroundingReport,
    )

    assert result.episode_id == "episode-1"


def test_openai_compatible_client_uses_schema_model_override() -> None:
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"episode_id":"episode-1","overall_status":"pass",'
                                '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}'
                            ),
                        }
                    ],
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(
            api_key="test-key",
            model_name="gpt-4o-mini",
            model_overrides={"grounding_report": "gpt-4o"},
        ),
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
    assert transport.last_payload["model"] == "gpt-4o"


def test_openai_compatible_client_raises_for_content_filter_finish_reason() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"content": '{"episode_id":"episode-1"}'},
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key", base_url="https://example.com"),
        transport=transport,
    )

    try:
        client.generate_json(
            schema_name="grounding_report",
            instructions="Validate the claims.",
            payload={"script": {"episode_id": "episode-1"}},
            response_model=GroundingReport,
        )
    except LLMContentFilterError as exc:
        assert "content filtering" in str(exc)
    else:
        raise AssertionError("Expected content filter error to be raised")


def test_openai_compatible_client_raises_for_length_finish_reason() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"episode_id":"episode-1"}'},
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key", base_url="https://example.com"),
        transport=transport,
    )

    try:
        client.generate_json(
            schema_name="grounding_report",
            instructions="Validate the claims.",
            payload={"script": {"episode_id": "episode-1"}},
            response_model=GroundingReport,
        )
    except RuntimeError as exc:
        assert "completion token limit" in str(exc)
    else:
        raise AssertionError("Expected length truncation error to be raised")


def test_openai_compatible_client_preserves_http_status_and_body() -> None:
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key", base_url="https://example.com"),
        transport=FailingHTTPTransport(),
    )

    with pytest.raises(LLMTransportHTTPError) as exc_info:
        client.generate_json(
            schema_name="grounding_report",
            instructions="Validate the claims.",
            payload={"script": {"episode_id": "episode-1"}},
            response_model=GroundingReport,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.response_text == '{"error":"parse error"}'
    assert str(exc_info.value) == 'LLM request failed with status 400: {"error":"parse error"}'


def test_openai_compatible_tts_client_returns_audio_bytes(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = FakeBinaryTransport(b"fake-audio")
    client = OpenAICompatibleTTSClient(
        config=TTSConfig(
            model_name="gpt-4o-mini-tts",
            voice="ballad",
            audio_format="mp3",
            instructions="Speak with dramatic restraint.",
        ),
        transport=transport,
    )

    audio_bytes = client.synthesize(
        "Testing the TTS path.",
        voice="nova",
        audio_format="wav",
        instructions="Deliver this with ominous gravity.",
    )

    assert audio_bytes == b"fake-audio"
    assert transport.last_payload is not None
    assert transport.last_payload["model"] == "gpt-4o-mini-tts"
    assert transport.last_payload["voice"] == "nova"
    assert transport.last_payload["format"] == "wav"
    assert transport.last_payload["instructions"] == "Deliver this with ominous gravity."
    assert transport.last_payload["input"] == "Testing the TTS path."


def test_openai_compatible_client_sets_reasoning_effort_excluding_grounding() -> None:
    class DummyResponse(BaseModel):
        value: str

    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"value":"ok"}'},
                    ],
                }
            ]
        }
    )
    client = OpenAICompatibleLLMClient(
        config=LLMConfig(api_key="test-key", reasoning_effort="high"),
        transport=transport,
    )

    result = client.generate_json(
        schema_name="beat_script",
        instructions="Respond with a value.",
        payload={},
        response_model=DummyResponse,
    )

    assert result.value == "ok"
    assert transport.last_payload is not None
    assert transport.last_payload["reasoning"]["effort"] == "high"

    transport.last_payload = None
    transport.body = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"episode_id":"episode-1","overall_status":"pass",'
                            '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}'
                        ),
                    }
                ],
            }
        ]
    }
    client.generate_json(
        schema_name="grounding_report",
        instructions="Validate the claims.",
        payload={"script": {"episode_id": "episode-1"}},
        response_model=GroundingReport,
    )
    assert transport.last_payload is not None
    assert "reasoning" not in transport.last_payload
