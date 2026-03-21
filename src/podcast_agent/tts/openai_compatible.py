"""OpenAI-compatible speech synthesis client."""

from __future__ import annotations

import json
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from podcast_agent.config import Settings, TTSConfig
from podcast_agent.tts.base import TTSClient
from podcast_agent.tts.kokoro import KokoroTTSClient


class BinaryHTTPTransport:
    """Minimal binary transport for speech endpoints."""

    def post_json_for_bytes(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS request failed with status {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"TTS request failed: {exc.reason}") from exc


class OpenAICompatibleTTSClient(TTSClient):
    """Client for OpenAI-compatible text-to-speech endpoints."""

    def __init__(self, config: TTSConfig, transport: BinaryHTTPTransport | None = None) -> None:
        super().__init__()
        self.config = config
        self.transport = transport or BinaryHTTPTransport()

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        api_key = __import__("os").getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible TTS synthesis.")
        payload = {
            "model": self.config.model_name,
            "voice": voice or self.config.voice,
            "input": text,
            "format": audio_format or self.config.audio_format,
            "instructions": instructions or self.config.instructions,
            "speed": self.config.speed,
        }
        endpoint = f"{(__import__('os').getenv('OPENAI_BASE_URL') or 'https://api.openai.com').rstrip('/')}/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.run_logger is not None:
            self.run_logger.log(
                "tts_request",
                client="openai-compatible",
                model=self.config.model_name,
                voice=payload["voice"],
                audio_format=payload["format"],
                instructions=payload["instructions"],
                text=text,
            )
        audio_bytes = self.transport.post_json_for_bytes(
            url=endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        if self.run_logger is not None:
            self.run_logger.log(
                "tts_response",
                client="openai-compatible",
                model=self.config.model_name,
                voice=payload["voice"],
                audio_format=payload["format"],
                instructions=payload["instructions"],
                byte_count=len(audio_bytes),
            )
        return audio_bytes


def build_tts_client(settings: Settings) -> TTSClient:
    provider = settings.tts.provider.lower()
    if provider in {"openai-compatible", "openai"}:
        return OpenAICompatibleTTSClient(settings.tts)
    if provider == "kokoro":
        if settings.tts.voice == "ballad":
            return KokoroTTSClient(settings.tts.model_copy(update={"voice": "af_heart"}))
        return KokoroTTSClient(settings.tts)
    raise ValueError(f"Unsupported TTS provider '{settings.tts.provider}'.")
