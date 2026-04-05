"""Base interface for text-to-speech synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TTSClient(ABC):
    """Abstract speech synthesis client."""

    def __init__(self) -> None:
        self.run_logger = None

    def set_run_logger(self, run_logger: Any) -> None:
        """Attach run logger for request tracing."""

        self.run_logger = run_logger

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """Synthesize speech audio bytes from text."""
