"""Runtime configuration for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsn: str | None = Field(
        default_factory=lambda: os.getenv("DATABASE_URL"),
        description="PostgreSQL DSN used by PGVector and chunk storage.",
    )


class AgentConfig(BaseModel):
    """Per-agent LLM configuration."""

    model_config = ConfigDict(frozen=True)

    model_name: str | None = None
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retry_attempts: int | None = None
    concurrency_limit: int | None = None


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER") or os.getenv("LLM_TYPE") or "anthropic"
    )
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    model_name: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "claude-opus-4-6"))
    model_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Per-schema model overrides keyed by schema_name.",
    )
    provider_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Per-schema LLM provider overrides keyed by schema_name.",
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT"),
    )
    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    )
    anthropic_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("ANTHROPIC_MAX_TOKENS", "100000")), ge=1
    )
    anthropic_max_tokens_overrides: dict[str, int] = Field(
        default_factory=lambda: {
            "structuring": 64000,
            "passage_extraction": 64000,
            "episode_framing": 64000,
        },
        description="Per-schema max_tokens overrides for Anthropic models.",
    )
    anthropic_prompt_caching_enabled: bool = Field(
        default_factory=lambda: _env_bool("ANTHROPIC_PROMPT_CACHING_ENABLED", True)
    )
    anthropic_prompt_caching_auto_fallback: bool = Field(
        default_factory=lambda: _env_bool("ANTHROPIC_PROMPT_CACHING_AUTO_FALLBACK", True)
    )
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    timeout_seconds_overrides: dict[str, float] = Field(
        default_factory=lambda: {
            "passage_extraction": 900.0,
            "synthesis_mapping": 1200.0,
            "episode_planning": 900.0,
            "episode_writing": 1200.0,
        },
        description="Per-schema timeout overrides in seconds.",
    )
    heartbeat_enabled: bool = Field(default=True)
    heartbeat_interval_seconds: float = Field(default=60.0, gt=0.0)
    openai_prompt_caching_enabled: bool = Field(
        default_factory=lambda: _env_bool("OPENAI_PROMPT_CACHING_ENABLED", True),
    )

    # Per-agent configuration keyed by agent schema_name
    agent_configs: dict[str, AgentConfig] = Field(
        default_factory=lambda: {
            "structuring": AgentConfig(model_name="claude-haiku-4-5", temperature=0.3, max_retry_attempts=3, concurrency_limit=10),
            "chapter_summary": AgentConfig(model_name="claude-haiku-4-5", temperature=0.3, max_retry_attempts=3, concurrency_limit=10),
            "book_summary": AgentConfig(model_name="claude-opus-4-6", temperature=0.3, max_retry_attempts=3, concurrency_limit=10),
            "theme_decomposition": AgentConfig(model_name="claude-opus-4-6", temperature=0.7, max_retry_attempts=2, concurrency_limit=6),
            "passage_extraction": AgentConfig(model_name="claude-opus-4-6", temperature=0.1, max_retry_attempts=3, concurrency_limit=8),
            "synthesis_mapping": AgentConfig(model_name="claude-opus-4-6", temperature=0.8, max_retry_attempts=2, concurrency_limit=3),
            "narrative_strategy": AgentConfig(model_name="claude-opus-4-6", temperature=0.5, max_retry_attempts=2, concurrency_limit=6),
            "episode_planning": AgentConfig(model_name="claude-opus-4-6", temperature=0.5, max_retry_attempts=2, concurrency_limit=6),
            "episode_writing": AgentConfig(model_name="claude-opus-4-6", temperature=0.6, max_retry_attempts=2, concurrency_limit=5),
            "source_weaving": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.5, max_retry_attempts=2, concurrency_limit=6),
            "grounding_validation": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.2, max_retry_attempts=2, concurrency_limit=6),
            "repair": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.3, max_retry_attempts=2, concurrency_limit=6),
            "spoken_delivery": AgentConfig(model_name="claude-opus-4-6", temperature=0.7, max_retry_attempts=2, concurrency_limit=6),
            "episode_framing": AgentConfig(model_name="claude-haiku-4-5", temperature=0.7, max_retry_attempts=2, concurrency_limit=15),
        },
        description="Per-agent LLM config overrides keyed by schema_name.",
    )

    def resolve_anthropic_max_tokens(self, schema_name: str) -> int:
        override = self.anthropic_max_tokens_overrides.get(schema_name)
        if override is None:
            resolved = self.anthropic_max_tokens
        else:
            resolved = override
        model_name = self.resolve_model(schema_name)
        if model_name and model_name.strip().lower().startswith("claude-haiku"):
            resolved = min(64000, resolved)
        return max(1, resolved)

    def resolve_temperature(self, schema_name: str) -> float:
        agent_cfg = self.agent_configs.get(schema_name)
        if agent_cfg and agent_cfg.temperature is not None:
            return agent_cfg.temperature
        return self.temperature

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_openai_base_url(cls, value: str) -> str:
        if value is None:
            return value
        base = str(value).rstrip("/")
        if base == "https://api.openai.com":
            return "https://api.openai.com/v1"
        return value

    def resolve_model(self, schema_name: str) -> str:
        agent_cfg = self.agent_configs.get(schema_name)
        if agent_cfg and agent_cfg.model_name is not None:
            return agent_cfg.model_name
        return self.model_overrides.get(schema_name, self.model_name)

    def resolve_max_retry_attempts(self, schema_name: str) -> int:
        agent_cfg = self.agent_configs.get(schema_name)
        if agent_cfg and agent_cfg.max_retry_attempts is not None:
            return agent_cfg.max_retry_attempts
        return 2

    def resolve_concurrency_limit(self, schema_name: str) -> int | None:
        agent_cfg = self.agent_configs.get(schema_name)
        if agent_cfg and agent_cfg.concurrency_limit is not None:
            return agent_cfg.concurrency_limit
        return None

    def resolve_timeout_seconds(self, schema_name: str) -> float:
        return self.timeout_seconds_overrides.get(schema_name, self.timeout_seconds)


class TTSConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="openai-compatible")
    model_name: str = Field(default="gpt-4o-mini-tts")
    voice: str = Field(default="ballad")
    audio_format: str = Field(default="mp3")
    instructions: str = Field(
        default=(
            "Narrate as a clear, grounded documentary host.\n\n"
            "Keep the delivery controlled, natural, and easy to follow.\n\n"
            "Use serious tone when the material is weighty, but avoid melodrama unless the segment guidance asks for it.\n\n"
            "Favor clean diction and steady pacing."
        )
    )
    speed: float = Field(default=1, gt=0.0, le=4.0)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    kokoro_parallelism: int = Field(default=2, ge=1)
    kokoro_worker_threads: int = Field(default=4, ge=1)
    kokoro_chunk_min_words: int = Field(default=550, ge=100)
    kokoro_chunk_max_words: int = Field(default=600, ge=100)

    @model_validator(mode="after")
    def validate_kokoro_chunk_bounds(self) -> "TTSConfig":
        if self.kokoro_chunk_max_words < self.kokoro_chunk_min_words:
            raise ValueError("kokoro_chunk_max_words must be >= kokoro_chunk_min_words")
        return self


class SpokenDeliveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    timeout_seconds: float = Field(default=10800.0, gt=0.0)
    chunk_min_words: int = Field(default=700, ge=100)
    chunk_max_words: int = Field(default=900, ge=100)


class PipelineRuntimeConfig(BaseModel):
    """Orchestration-level runtime parameters."""

    model_config = ConfigDict(frozen=True)

    artifact_root: Path = Field(default=Path("runs"))
    embedding_dimensions: int = Field(default=8, ge=4)
    max_chunk_words: int = Field(default=400, ge=50)
    chunk_overlap_words: int = Field(default=50, ge=0)
    min_chunk_words: int = Field(default=80, ge=10)
    max_repair_attempts: int = Field(default=3, ge=0)
    episode_write_concurrency: int = Field(default=5, ge=1)
    tts_concurrency: int = Field(default=4, ge=1)
    llm_global_max_concurrency: int = Field(default=30, ge=1)
    audio_retry_attempts: int = Field(default=3, ge=0)
    spoken_words_per_minute: int = Field(default=110, ge=80)
    # Thematic intelligence
    max_axes: int = Field(default=30, ge=1)
    min_axes: int = Field(default=25, ge=1)
    passage_retrieval_percentage: float = Field(default=0.25, gt=0.0, le=1.0)
    passage_retrieval_min_per_book: int = Field(default=20, ge=1)
    passage_retrieval_max_per_book: int = Field(default=50, ge=1)
    axis_candidate_target_total: int = Field(default=250, ge=1)
    admission_floor_per_book: int = Field(default=2, ge=0)
    retrieval_conf_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    retrieval_size_basis: Literal["total_words"] = "total_words"
    retrieval_size_exponent: float = Field(default=0.68, ge=0.0)
    retrieval_relevance_power: float = Field(default=1.2, ge=0.0)
    retrieval_soft_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    chapter_penalty_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    rerank_top_k: int = Field(default=30, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    spoken_chunk_max_words: int = Field(default=250, ge=50)

    @model_validator(mode="after")
    def validate_retrieval_budget_bounds(self) -> PipelineRuntimeConfig:
        if self.passage_retrieval_max_per_book < self.passage_retrieval_min_per_book:
            raise ValueError(
                "passage_retrieval_max_per_book must be >= passage_retrieval_min_per_book"
            )
        return self


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(default_factory=lambda: os.getenv("EMBEDDINGS_PROVIDER", "openai"))
    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small")
    )
    dimensions: int | None = Field(
        default_factory=lambda: int(os.getenv("EMBEDDINGS_DIMENSIONS", "0")) or None
    )
    batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDINGS_BATCH_SIZE", "100")), ge=1)
    timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("EMBEDDINGS_TIMEOUT_SECONDS", "60.0")), gt=0.0
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(
        default_factory=lambda: os.getenv("RETRIEVAL_COLLECTION_NAME", "podcast_agent_chunks")
    )
    oversample_factor: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_OVERSAMPLE_FACTOR", "3")), ge=1
    )
    max_oversample: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_MAX_OVERSAMPLE", "150")), ge=1
    )


class LangChainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_backend: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_CACHE_BACKEND", "sqlite"))
    cache_path: str = Field(
        default_factory=lambda: os.getenv(
            "LANGCHAIN_CACHE_PATH", str(Path(".podcast_agent") / "langchain_cache.sqlite")
        )
    )
    cache_redis_url: str | None = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_CACHE_REDIS_URL")
    )
    cache_enabled: bool = Field(default_factory=lambda: _env_bool("LANGCHAIN_CACHE_ENABLED", True))


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    spoken_delivery: SpokenDeliveryConfig = Field(default_factory=SpokenDeliveryConfig)
    pipeline: PipelineRuntimeConfig = Field(default_factory=PipelineRuntimeConfig)
