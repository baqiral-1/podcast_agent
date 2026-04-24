"""Runtime configuration for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_ANTHROPIC_THINKING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


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
    model_name: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "claude-opus-4-7"))
    model_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Per-schema model overrides keyed by schema_name.",
    )
    provider_overrides: dict[str, str] = Field(
        default_factory=lambda: {"synthesis_primitives": "anthropic"},
        description="Per-schema LLM provider overrides keyed by schema_name.",
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT") or "xhigh",
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
            "passage_extraction": 64000,
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
            "passage_extraction": 480.0,
            "synthesis_primitives": 3360.0,
            "synthesis_consolidation": 2520.0,
            "narrative_strategy": 900.0,
            "theme_decomposition": 900.0,
            "episode_planning": 1500.0,
            "episode_writing": 1500.0,
            "spoken_delivery": 1200.0,
        },
        description="Per-schema timeout overrides in seconds.",
    )
    thinking_budget_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "narrative_strategy": 30000,
            "episode_planning": 30000,
            "episode_writing": 30000,
            "spoken_delivery": 30000,
            "synthesis_primitives": 30000,
            "synthesis_consolidation": 30000,
            "theme_decomposition": 30000,
        },
        description=(
            "Legacy per-schema thinking budget in tokens. "
            "When Anthropic adaptive thinking is used, these values are mapped "
            "to effort tiers unless an explicit effort override is configured."
        ),
    )
    log_thinking_content: bool = Field(
        default_factory=lambda: _env_bool("LLM_LOG_THINKING_CONTENT", False),
        description=(
            "When true, capture extended-thinking block text into a separate "
            "`llm_thinking_content` run.log event (truncated). Off by default."
        ),
    )
    anthropic_thinking_effort_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-schema Anthropic adaptive thinking effort override. "
            "Allowed values: low, medium, high, xhigh, max."
        ),
    )
    heartbeat_enabled: bool = Field(default=True)
    heartbeat_interval_seconds: float = Field(default=60.0, gt=0.0)
    openai_prompt_caching_enabled: bool = Field(
        default_factory=lambda: _env_bool("OPENAI_PROMPT_CACHING_ENABLED", True),
    )

    # Per-agent configuration keyed by agent schema_name
    agent_configs: dict[str, AgentConfig] = Field(
        default_factory=lambda: {
            "chapter_summary": AgentConfig(model_name="claude-haiku-4-5", temperature=0.3, max_retry_attempts=3, concurrency_limit=48),
            "book_summary": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.3, max_retry_attempts=3, concurrency_limit=15),
            "theme_decomposition": AgentConfig(model_name="claude-opus-4-7", temperature=0.7, max_retry_attempts=2, concurrency_limit=6),
            "passage_extraction": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.1, max_retry_attempts=4, concurrency_limit=26),
            "synthesis_primitives": AgentConfig(model_name="claude-opus-4-7", temperature=0.8, max_retry_attempts=2, concurrency_limit=3),
            "synthesis_consolidation": AgentConfig(model_name="claude-opus-4-7", temperature=0.6, max_retry_attempts=2, concurrency_limit=4),
            "narrative_strategy": AgentConfig(model_name="claude-opus-4-7", temperature=0.5, max_retry_attempts=2, concurrency_limit=6),
            "episode_planning": AgentConfig(model_name="claude-opus-4-7", temperature=0.5, max_retry_attempts=3, concurrency_limit=9),
            "episode_writing": AgentConfig(model_name="claude-opus-4-7", temperature=0.6, max_retry_attempts=2, concurrency_limit=9),
            "grounding_validation": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.2, max_retry_attempts=2, concurrency_limit=6),
            "repair": AgentConfig(model_name="claude-sonnet-4-6", temperature=0.3, max_retry_attempts=2, concurrency_limit=6),
            "spoken_delivery": AgentConfig(model_name="claude-opus-4-7", temperature=0.7, max_retry_attempts=2, concurrency_limit=9),
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

    def resolve_thinking_budget(self, schema_name: str) -> int | None:
        """Return the extended-thinking token budget for *schema_name*, or None if disabled."""
        return self.thinking_budget_tokens.get(schema_name)

    @field_validator("anthropic_thinking_effort_overrides", mode="before")
    @classmethod
    def _validate_anthropic_thinking_effort_overrides(
        cls, value: dict[str, str] | None
    ) -> dict[str, str]:
        if value is None:
            return {}
        normalized: dict[str, str] = {}
        for schema_name, effort_value in value.items():
            effort = str(effort_value).strip().lower()
            if effort not in _ANTHROPIC_THINKING_EFFORTS:
                allowed = ", ".join(sorted(_ANTHROPIC_THINKING_EFFORTS))
                raise ValueError(
                    f"Invalid anthropic thinking effort '{effort_value}' for "
                    f"schema '{schema_name}'. Allowed: {allowed}."
                )
            normalized[str(schema_name)] = effort
        return normalized

    def resolve_anthropic_thinking_effort(self, schema_name: str) -> str | None:
        override = self.anthropic_thinking_effort_overrides.get(schema_name)
        if override is not None:
            return override
        budget_tokens = self.resolve_thinking_budget(schema_name)
        if budget_tokens is None:
            return None
        if budget_tokens <= 12_000:
            return "medium"
        if budget_tokens <= 25_000:
            return "high"
        return "xhigh"


class TTSConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="openai-compatible")
    model_name: str = Field(default="tts-1-hd")
    voice: str = Field(default="fable")
    audio_format: str = Field(default="mp3")
    instructions: str = Field(
        default=(
            "Narrate as a clear, grounded documentary host.\n\n"
            "Keep the delivery controlled, natural, and easy to follow.\n\n"
            "Use serious tone when the material is weighty, but avoid melodrama unless the segment guidance asks for it.\n\n"
            "Favor clean diction and steady pacing."
        )
    )
    speed: float = Field(default=0.9, gt=0.0, le=4.0)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    kokoro_parallelism: int = Field(default=2, ge=1)
    kokoro_worker_threads: int = Field(default=2, ge=1)
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_root: Path = Field(default=Path("runs"))
    embedding_dimensions: int = Field(default=8, ge=4)
    max_chunk_words: int = Field(default=1000, ge=50)
    chunk_overlap_words: int = Field(default=75, ge=0)
    min_chunk_words: int = Field(default=80, ge=10)
    max_repair_attempts: int = Field(default=3, ge=0)
    episode_planning_concurrency: int = Field(default=6, ge=1)
    episode_write_concurrency: int = Field(default=6, ge=1)
    tts_concurrency: int = Field(default=5, ge=1)
    llm_global_max_concurrency: int = Field(default=30, ge=1)
    audio_retry_attempts: int = Field(default=3, ge=0)
    spoken_words_per_minute: int = Field(default=130, ge=80)
    # Thematic intelligence
    max_axes: int = Field(default=15, ge=1)
    min_axes: int = Field(default=10, ge=1)
    passage_retrieval_percentage: float = Field(default=0.25, gt=0.0, le=1.0)
    passage_retrieval_min_per_book: int = Field(default=10, ge=1)
    passage_retrieval_max_per_book: int = Field(default=25, ge=1)
    axis_candidate_target_total: int = Field(default=60, ge=1)
    pre_axis_total_budget: int = Field(default=1500, ge=1)
    pre_axis_floor: int = Field(default=30, ge=0)
    pre_axis_relevance_power: float = Field(default=1.3, ge=0.0)
    pre_axis_cross_axis_reuse_penalty: float = Field(default=0.25, ge=0.0, le=1.0)
    admission_floor_per_book: int = Field(default=0, ge=0)
    retrieval_relevance_power: float = Field(default=2.5, ge=0.0)
    retrieval_soft_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    chapter_penalty_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    mmr_enabled: bool = True
    mmr_synthesis_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    mmr_planning_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    synthesis_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    synthesis_axis_min: int = Field(default=10, ge=0)
    synthesis_axis_max: int = Field(default=15, ge=1)
    synthesis_total_passage_cap: int = Field(default=500, ge=1)
    synthesis_floor_budget_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    synthesis_axis_floor_min: int = Field(default=10, ge=0)
    synthesis_axis_floor_max: int = Field(default=15, ge=1)
    synthesis_axis_ceiling_multiplier: float = Field(default=1.4, ge=1.0)
    synthesis_trim_top_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    synthesis_trim_mid_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    synthesis_trim_top_keep_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    synthesis_trim_mid_keep_fraction: float = Field(default=0.40, gt=0.0, le=1.0)
    synthesis_trim_tail_keep_fraction: float = Field(default=0.30, gt=0.0, le=1.0)
    planning_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    planning_axis_min: int = Field(default=10, ge=0)
    planning_axis_max: int = Field(default=15, ge=1)
    planning_total_passage_cap: int = Field(default=300, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    passage_extraction_concurrency: int = Field(default=16, ge=1)
    spoken_chunk_max_words: int = Field(default=250, ge=50)

    @model_validator(mode="after")
    def validate_retrieval_budget_bounds(self) -> PipelineRuntimeConfig:
        if self.passage_retrieval_max_per_book < self.passage_retrieval_min_per_book:
            raise ValueError(
                "passage_retrieval_max_per_book must be >= passage_retrieval_min_per_book"
            )
        if self.synthesis_axis_max < self.synthesis_axis_min:
            raise ValueError("synthesis_axis_max must be >= synthesis_axis_min")
        if self.synthesis_axis_floor_max < self.synthesis_axis_floor_min:
            raise ValueError("synthesis_axis_floor_max must be >= synthesis_axis_floor_min")
        if self.planning_axis_max < self.planning_axis_min:
            raise ValueError("planning_axis_max must be >= planning_axis_min")
        if (self.synthesis_trim_top_fraction + self.synthesis_trim_mid_fraction) > 1.0:
            raise ValueError("synthesis trim top and mid fractions must sum to <= 1.0")
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
