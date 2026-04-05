"""Unit tests for configuration."""

from __future__ import annotations

import pytest

from podcast_agent.config import (
    AgentConfig,
    DatabaseConfig,
    EmbeddingsConfig,
    LLMConfig,
    PipelineRuntimeConfig,
    RetrievalConfig,
    Settings,
    TTSConfig,
)


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.temperature == 1.0
        assert config.timeout_seconds == 600.0
        assert config.base_url == "https://api.openai.com/v1"

    def test_resolve_temperature_with_agent_config(self):
        config = LLMConfig(
            agent_configs={
                "structuring": AgentConfig(temperature=0.3),
                "synthesis_mapping": AgentConfig(temperature=0.8),
            }
        )
        assert config.resolve_temperature("structuring") == 0.3
        assert config.resolve_temperature("synthesis_mapping") == 0.8
        assert config.resolve_temperature("unknown_agent") == 1.0

    def test_normalizes_openai_base_url(self):
        config = LLMConfig(base_url="https://api.openai.com")
        assert config.base_url == "https://api.openai.com/v1"

    def test_preserves_non_openai_base_url(self):
        custom = "https://example.com/custom"
        config = LLMConfig(base_url=custom)
        assert config.base_url == custom

    def test_resolve_model_with_agent_config(self):
        config = LLMConfig(
            model_name="claude-opus-4-6",
            agent_configs={
                "structuring": AgentConfig(model_name="claude-haiku-4-5"),
            },
        )
        assert config.resolve_model("structuring") == "claude-haiku-4-5"
        assert config.resolve_model("unknown") == "claude-opus-4-6"

    def test_resolve_model_with_legacy_overrides(self):
        config = LLMConfig(
            model_name="claude-opus-4-6",
            model_overrides={"beat_script": "claude-sonnet-4-6"},
        )
        assert config.resolve_model("beat_script") == "claude-sonnet-4-6"

    def test_agent_config_takes_precedence(self):
        config = LLMConfig(
            model_name="claude-opus-4-6",
            model_overrides={"structuring": "claude-sonnet-4-6"},
            agent_configs={
                "structuring": AgentConfig(model_name="claude-haiku-4-5"),
            },
        )
        assert config.resolve_model("structuring") == "claude-haiku-4-5"

    def test_default_agent_temperatures(self):
        config = LLMConfig()
        assert config.resolve_temperature("structuring") == 0.3
        assert config.resolve_temperature("synthesis_mapping") == 0.8
        assert config.resolve_temperature("grounding_validation") == 0.2
        assert config.resolve_temperature("spoken_delivery") == 0.7


class TestTTSConfig:
    def test_defaults(self):
        config = TTSConfig()
        assert config.provider == "openai-compatible"
        assert config.voice == "ballad"

    def test_kokoro_chunk_validation(self):
        with pytest.raises(ValueError, match="kokoro_chunk_max_words"):
            TTSConfig(kokoro_chunk_min_words=600, kokoro_chunk_max_words=500)


class TestPipelineRuntimeConfig:
    def test_defaults(self):
        config = PipelineRuntimeConfig()
        assert config.max_chunk_words == 400
        assert config.chunk_overlap_words == 50
        assert config.max_repair_attempts == 3
        assert config.episode_write_concurrency == 5
        assert config.tts_concurrency == 4
        assert config.spoken_words_per_minute == 110

    def test_thematic_defaults(self):
        config = PipelineRuntimeConfig()
        assert config.max_axes == 15
        assert config.min_axes == 5
        assert config.passage_retrieval_percentage == 0.25
        assert config.passage_retrieval_min_per_book == 20
        assert config.passage_retrieval_max_per_book == 50
        assert config.axis_candidate_target_total == 180
        assert config.admission_floor_per_book == 2
        assert config.retrieval_conf_weight == 0.2
        assert config.retrieval_size_basis == "total_words"
        assert config.retrieval_size_exponent == 0.68
        assert config.retrieval_relevance_power == 1.2
        assert config.retrieval_soft_threshold == 0.35
        assert config.chapter_penalty_weight == 0.05
        assert config.rerank_top_k == 30
        assert config.synthesis_quality_threshold == 0.5

    def test_retrieval_budget_bounds_validation(self):
        with pytest.raises(ValueError, match="passage_retrieval_max_per_book"):
            PipelineRuntimeConfig(
                passage_retrieval_min_per_book=21,
                passage_retrieval_max_per_book=20,
            )


class TestLLMConfigResolvers:
    def test_resolve_max_retry_attempts_from_agent_config(self):
        config = LLMConfig()
        assert config.resolve_max_retry_attempts("structuring") == 3
        assert config.resolve_max_retry_attempts("chapter_summary") == 3
        assert config.resolve_max_retry_attempts("book_summary") == 3
        assert config.resolve_max_retry_attempts("passage_extraction") == 3
        assert config.resolve_max_retry_attempts("synthesis_mapping") == 2

    def test_resolve_max_retry_attempts_default_for_unknown(self):
        config = LLMConfig()
        assert config.resolve_max_retry_attempts("unknown_agent") == 2

    def test_resolve_timeout_seconds_uses_schema_override(self):
        config = LLMConfig()
        assert config.resolve_timeout_seconds("passage_extraction") == 360.0
        assert config.resolve_timeout_seconds("synthesis_mapping") == 1200.0
        assert config.resolve_timeout_seconds("unknown_agent") == 600.0

    def test_resolve_concurrency_limit_from_agent_config(self):
        config = LLMConfig()
        assert config.resolve_concurrency_limit("structuring") == 10
        assert config.resolve_concurrency_limit("chapter_summary") == 10
        assert config.resolve_concurrency_limit("book_summary") == 10
        assert config.resolve_concurrency_limit("passage_extraction") == 8
        assert config.resolve_concurrency_limit("synthesis_mapping") == 3
        assert config.resolve_concurrency_limit("episode_writing") == 5

    def test_resolve_concurrency_limit_none_for_unknown(self):
        config = LLMConfig()
        assert config.resolve_concurrency_limit("unknown_agent") is None

    def test_resolve_model_from_agent_config(self):
        config = LLMConfig()
        assert config.resolve_model("structuring") == "claude-haiku-4-5"
        assert config.resolve_model("synthesis_mapping") == "claude-opus-4-6"
        assert config.resolve_model("narrative_strategy") == "claude-opus-4-6"
        assert config.resolve_model("grounding_validation") == "claude-sonnet-4-6"
        assert config.resolve_model("episode_framing") == "claude-haiku-4-5"

    def test_resolve_temperature_from_agent_config(self):
        config = LLMConfig()
        assert config.resolve_temperature("structuring") == 0.3
        assert config.resolve_temperature("synthesis_mapping") == 0.8
        assert config.resolve_temperature("grounding_validation") == 0.2

    def test_resolve_anthropic_max_tokens_clamps_haiku(self):
        config = LLMConfig()
        assert config.resolve_anthropic_max_tokens("chapter_summary") == 64000
        assert config.resolve_anthropic_max_tokens("synthesis_mapping") == 100000

    def test_resolve_anthropic_max_tokens_respects_override(self):
        config = LLMConfig(
            anthropic_max_tokens=120000,
            anthropic_max_tokens_overrides={"chapter_summary": 50000, "synthesis_mapping": 90000},
            agent_configs={
                "chapter_summary": AgentConfig(model_name="claude-haiku-4-5"),
                "synthesis_mapping": AgentConfig(model_name="claude-opus-4-6"),
            },
        )
        assert config.resolve_anthropic_max_tokens("chapter_summary") == 50000
        assert config.resolve_anthropic_max_tokens("synthesis_mapping") == 90000

    def test_resolve_anthropic_max_tokens_caps_haiku_override(self):
        config = LLMConfig(
            anthropic_max_tokens_overrides={"chapter_summary": 90000},
            agent_configs={"chapter_summary": AgentConfig(model_name="claude-haiku-4-5")},
        )
        assert config.resolve_anthropic_max_tokens("chapter_summary") == 64000

    def test_all_agents_have_model_assigned(self):
        config = LLMConfig()
        expected_agents = [
            "structuring", "book_summary", "theme_decomposition", "passage_extraction",
            "synthesis_mapping", "narrative_strategy", "episode_planning",
            "episode_writing", "source_weaving", "grounding_validation",
            "repair", "spoken_delivery", "episode_framing",
        ]
        for agent_name in expected_agents:
            model = config.resolve_model(agent_name)
            assert model.startswith("claude-"), f"{agent_name} should have a claude model, got {model}"

    def test_all_agents_have_concurrency(self):
        config = LLMConfig()
        for agent_name in config.agent_configs:
            limit = config.resolve_concurrency_limit(agent_name)
            assert limit is not None and limit >= 1, f"{agent_name} needs a concurrency limit"


class TestSettings:
    def test_default_construction(self):
        settings = Settings()
        assert settings.llm is not None
        assert settings.pipeline is not None
        assert settings.tts is not None

    def test_frozen(self):
        settings = Settings()
        with pytest.raises(Exception):
            settings.llm = LLMConfig()
