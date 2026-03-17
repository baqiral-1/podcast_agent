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
    voice: str = Field(default="ballad")
    audio_format: str = Field(default="mp3")
    instructions: str = Field(
        default=(
            "Voice Affect: Low, hushed, and suspenseful; convey tension and intrigue.\n\n"
            "Tone: Deeply serious and mysterious, maintaining an undercurrent of unease throughout.\n\n"
            "Pacing: Slow, deliberate, pausing slightly after suspenseful moments to heighten drama.\n\n"
            "Emotion: Restrained yet intense—voice should subtly tremble or tighten at key suspenseful points.\n\n"
            'Emphasis: Highlight sensory descriptions ("footsteps echoed," "heart hammering," '
            '"shadows melting into darkness") to amplify atmosphere.\n\n'
            "Pronunciation: Slightly elongated vowels and softened consonants for an eerie, haunting effect.\n\n"
            'Pauses: Insert meaningful pauses after phrases like "only shadows melting into darkness," '
            "and especially before the final line, to enhance suspense dramatically."
        )
    )
    speed: float = Field(default=1.1, gt=0.0, le=4.0)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class PipelineConfig(BaseModel):
    """Top-level configuration for orchestration."""

    model_config = ConfigDict(frozen=True)

    artifact_root: Path = Field(default=Path(".podcast_agent") / "runs")
    embedding_dimensions: int = Field(default=8, ge=4)
    max_chunk_words: int = Field(default=180, ge=40)
    chunk_overlap_words: int = Field(default=30, ge=0)
    max_repair_attempts: int = Field(default=2, ge=0)
    episode_parallelism: int = Field(default=3, ge=1)
    audio_parallelism: int = Field(default=8, ge=1)
    audio_retry_attempts: int = Field(default=2, ge=0)
    beat_parallelism: int = Field(default=4, ge=1)
    beat_write_retry_attempts: int = Field(default=2, ge=0)
    beat_write_timeout_seconds: float = Field(default=120.0, gt=0.0)
    grounding_parallelism: int = Field(default=5, ge=1)
    minimum_source_words_per_episode: int = Field(default=50000, ge=1000)
    min_episode_source_ratio: float = Field(default=0.3, gt=0.0, le=1.0)
    spoken_words_per_minute: int = Field(default=130, ge=80)
    max_episode_minutes: int = Field(default=360, ge=1)
    max_analysis_payload_bytes: int = Field(default=500000, ge=10000)
    max_planning_payload_bytes: int = Field(default=500000, ge=10000)
    max_analysis_payload_bytes_with_episode_count: int = Field(default=1000000, ge=10000)
    max_planning_payload_bytes_with_episode_count: int = Field(default=1000000, ge=10000)
    section_beat_target_words: int = Field(default=1200, ge=200)
    beat_evidence_window_size: int = Field(default=8, ge=1)
    coverage_warning_min_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    max_structuring_chapter_words: int = Field(default=2500, ge=500)
    max_structuring_llm_chapter_words: int = Field(default=75000, ge=1000)
    structuring_parallelism: int = Field(default=5, ge=1)
    structuring_window_words: int = Field(default=1800, ge=300)
    structuring_window_overlap_words: int = Field(default=150, ge=0)


class Settings(BaseModel):
    """Aggregate application settings."""

    model_config = ConfigDict(frozen=True)

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
