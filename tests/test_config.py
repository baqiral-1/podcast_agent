"""Unit tests for configuration defaults and resolvers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.config import AgentConfig, LLMConfig, PipelineRuntimeConfig, TTSConfig


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.temperature == 1.0
        assert config.timeout_seconds == 600.0
        assert config.base_url == "https://api.openai.com/v1"

    def test_resolve_temperature_with_agent_config(self):
        config = LLMConfig(
            agent_configs={
                "chapter_summary": AgentConfig(temperature=0.3),
                "synthesis_primitives": AgentConfig(temperature=0.8),
            }
        )
        assert config.resolve_temperature("chapter_summary") == 0.3
        assert config.resolve_temperature("synthesis_primitives") == 0.8
        assert config.resolve_temperature("unknown_agent") == 1.0

    def test_default_agent_temperatures(self):
        config = LLMConfig()
        assert config.resolve_temperature("chapter_summary") == 0.3
        assert config.resolve_temperature("synthesis_primitives") == 0.8
        assert config.resolve_temperature("synthesis_consolidation") == 0.6
        assert config.resolve_temperature("grounding_validation") == 0.2
        assert config.resolve_temperature("style_audit") == 0.2

    def test_resolve_model_defaults(self):
        config = LLMConfig()
        assert config.resolve_model("chapter_summary") == "claude-haiku-4-5"
        assert config.resolve_model("synthesis_primitives") == "claude-opus-4-6"
        assert config.resolve_model("synthesis_consolidation") == "claude-opus-4-6"
        assert config.resolve_model("style_audit") == "claude-sonnet-4-6"

    def test_resolve_max_retry_attempts_defaults(self):
        config = LLMConfig()
        assert config.resolve_max_retry_attempts("chapter_summary") == 3
        assert config.resolve_max_retry_attempts("passage_extraction") == 4
        assert config.resolve_max_retry_attempts("synthesis_primitives") == 2
        assert config.resolve_max_retry_attempts("style_audit") == 2
        assert config.resolve_max_retry_attempts("unknown_agent") == 2

    def test_resolve_timeout_seconds_defaults(self):
        config = LLMConfig()
        assert config.resolve_timeout_seconds("passage_extraction") == 480.0
        assert config.resolve_timeout_seconds("synthesis_primitives") == 1200.0
        assert config.resolve_timeout_seconds("synthesis_consolidation") == 900.0
        assert config.resolve_timeout_seconds("style_audit") == 600.0
        assert config.resolve_timeout_seconds("unknown_agent") == 600.0

    def test_resolve_concurrency_limit_defaults(self):
        config = LLMConfig()
        assert config.resolve_concurrency_limit("chapter_summary") == 25
        assert config.resolve_concurrency_limit("synthesis_primitives") == 3
        assert config.resolve_concurrency_limit("synthesis_consolidation") == 4
        assert config.resolve_concurrency_limit("episode_planning") == 8
        assert config.resolve_concurrency_limit("style_audit") == 8
        assert config.resolve_concurrency_limit("unknown_agent") is None

    def test_resolve_thinking_budget_defaults(self):
        config = LLMConfig()
        assert config.resolve_thinking_budget("theme_decomposition") == 20_000
        assert config.resolve_thinking_budget("synthesis_primitives") == 20_000
        assert config.resolve_thinking_budget("synthesis_consolidation") == 15_000
        assert config.resolve_thinking_budget("episode_planning") == 30_000
        assert config.resolve_thinking_budget("style_audit") == 8_000
        assert config.resolve_thinking_budget("chapter_summary") is None

    def test_anthropic_max_tokens_uses_override_for_style_audit(self):
        config = LLMConfig()
        assert config.resolve_anthropic_max_tokens("style_audit") == 64000

    def test_normalizes_openai_base_url(self):
        config = LLMConfig(base_url="https://api.openai.com")
        assert config.base_url == "https://api.openai.com/v1"


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
        assert config.episode_write_concurrency == 8
        assert config.tts_concurrency == 4
        assert config.spoken_words_per_minute == 120

    def test_thematic_defaults(self):
        config = PipelineRuntimeConfig()
        assert config.max_axes == 15
        assert config.min_axes == 10
        assert config.passage_retrieval_percentage == 0.25
        assert config.pre_axis_total_budget == 1800
        assert config.post_axis_total_budget == 1200
        assert config.post_axis_cap == 120
        assert config.synthesis_axis_pct == 1.0
        assert config.synthesis_axis_min == 10
        assert config.synthesis_axis_max == 15
        assert config.planning_axis_pct == 1.0
        assert config.planning_axis_min == 10
        assert config.planning_axis_max == 15
        assert config.synthesis_total_passage_cap == 750
        assert config.planning_total_passage_cap == 300
        assert config.passage_extraction_concurrency == 8
        assert config.llm_global_max_concurrency == 30

    def test_rejects_removed_retrieval_weighting_fields(self):
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(retrieval_conf_weight=0.2)

    def test_retrieval_budget_bounds_validation(self):
        with pytest.raises(ValueError, match="passage_retrieval_max_per_book"):
            PipelineRuntimeConfig(
                passage_retrieval_min_per_book=21,
                passage_retrieval_max_per_book=20,
            )

    def test_axis_budget_bounds_validation(self):
        with pytest.raises(ValueError, match="post_axis_cap"):
            PipelineRuntimeConfig(post_axis_floor=50, post_axis_cap=20)
        with pytest.raises(ValueError, match="synthesis_axis_max"):
            PipelineRuntimeConfig(synthesis_axis_min=40, synthesis_axis_max=20)
        with pytest.raises(ValueError, match="planning_axis_max"):
            PipelineRuntimeConfig(planning_axis_min=50, planning_axis_max=30)
