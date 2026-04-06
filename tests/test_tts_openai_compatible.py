"""Unit tests for OpenAI-compatible TTS request mapping."""

from __future__ import annotations

from podcast_agent.config import Settings, TTSConfig
from podcast_agent.tts.kokoro import KokoroTTSClient
from podcast_agent.tts.openai_compatible import OpenAICompatibleTTSClient, build_tts_client


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post_json_for_bytes(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> bytes:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return b"audio"


class TestOpenAICompatibleTTSClient:
    def test_synthesize_uses_per_call_instructions_and_speed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        transport = _FakeTransport()
        client = OpenAICompatibleTTSClient(
            TTSConfig(
                instructions="base profile",
                speed=1.0,
            ),
            transport=transport,
        )

        result = client.synthesize(
            "Narration text.",
            voice="nova",
            audio_format="wav",
            instructions="segment profile",
            speed=1.2,
        )

        assert result == b"audio"
        assert len(transport.calls) == 1
        payload = transport.calls[0]["payload"]
        assert transport.calls[0]["url"] == "https://api.openai.com/v1/audio/speech"
        assert payload["voice"] == "nova"
        assert payload["format"] == "wav"
        assert payload["instructions"] == "segment profile"
        assert payload["speed"] == 1.2


class TestBuildTTSClient:
    def test_kokoro_with_ballad_maps_to_previous_default_voice(self):
        settings = Settings(tts=TTSConfig(provider="kokoro", voice="ballad"))

        client = build_tts_client(settings)

        assert isinstance(client, KokoroTTSClient)
        assert client.config.voice == "af_heart"
