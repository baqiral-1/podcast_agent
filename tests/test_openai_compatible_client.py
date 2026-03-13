"""Tests for the OpenAI-compatible LLM and TTS clients."""

from __future__ import annotations

from podcast_agent.config import LLMConfig, TTSConfig
from podcast_agent.llm.openai_compatible import HTTPResponse, OpenAICompatibleLLMClient
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


def test_openai_compatible_client_parses_schema_constrained_json() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"episode_id":"episode-1","overall_status":"pass",'
                            '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}'
                        )
                    }
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
    assert transport.last_payload["response_format"]["type"] == "json_object"


def test_openai_compatible_client_unwraps_payload_echoes() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"schema_name":"grounding_report","payload":'
                            '{"episode_id":"episode-1","overall_status":"pass",'
                            '"claim_assessments":[],"validated_at":"2026-03-13T00:00:00Z"}}'
                        )
                    }
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


def test_openai_compatible_tts_client_returns_audio_bytes(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = FakeBinaryTransport(b"fake-audio")
    client = OpenAICompatibleTTSClient(
        config=TTSConfig(model_name="gpt-4o-mini-tts", voice="alloy", audio_format="mp3"),
        transport=transport,
    )

    audio_bytes = client.synthesize("Testing the TTS path.", voice="nova", audio_format="wav")

    assert audio_bytes == b"fake-audio"
    assert transport.last_payload is not None
    assert transport.last_payload["model"] == "gpt-4o-mini-tts"
    assert transport.last_payload["voice"] == "nova"
    assert transport.last_payload["format"] == "wav"
    assert transport.last_payload["input"] == "Testing the TTS path."
