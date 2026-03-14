"""Runtime configuration for the podcast agent."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatabaseConfig(BaseModel):
    """Connection settings for PostgreSQL-backed storage."""

    model_config = ConfigDict(frozen=True)

    dsn: str | None = Field(
        default_factory=lambda: os.getenv("DATABASE_URL"),
        description="PostgreSQL DSN used by the repository layer.",
    )


class LLMConfig(BaseModel):
    """Shared LLM defaults for all agents."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="openai-compatible")
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
    )
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class TTSConfig(BaseModel):
    """Text-to-speech settings for audio synthesis."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="openai-compatible")
    model_name: str = Field(default="gpt-4o-mini-tts")
    voice: str = Field(default="alloy")
    audio_format: str = Field(default="mp3")
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class PipelineConfig(BaseModel):
    """Top-level configuration for orchestration."""

    model_config = ConfigDict(frozen=True)

    artifact_root: Path = Field(default=Path(".podcast_agent") / "runs")
    embedding_dimensions: int = Field(default=8, ge=4)
    max_chunk_words: int = Field(default=180, ge=40)
    chunk_overlap_words: int = Field(default=30, ge=0)
    max_repair_attempts: int = Field(default=2, ge=0)
    minimum_source_words_per_episode: int = Field(default=50000, ge=1000)
    spoken_words_per_minute: int = Field(default=130, ge=80)
    max_structuring_chapter_words: int = Field(default=2500, ge=500)
    structuring_window_words: int = Field(default=1800, ge=300)
    structuring_window_overlap_words: int = Field(default=150, ge=0)


class Settings(BaseModel):
    """Aggregate application settings."""

    model_config = ConfigDict(frozen=True)

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
