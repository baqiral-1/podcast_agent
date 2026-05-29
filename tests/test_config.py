"""Unit tests for configuration defaults and resolvers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.config import (
    AgentConfig,
    LLMConfig,
    PipelineRuntimeConfig,
    TTSConfig,
)


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.temperature == 1.0
        assert config.timeout_seconds == 600.0
        assert config.base_url == "https://api.openai.com/v1"
        assert config.reasoning_effort == "xhigh"
        assert (
            config.provider_overrides["primitive_substrate_extraction"] == "anthropic"
        )

    def test_resolve_temperature_with_agent_config(self):
        config = LLMConfig(
            agent_configs={
                "chapter_summary": AgentConfig(temperature=0.3),
                "primitive_substrate_extraction": AgentConfig(temperature=0.8),
            }
        )
        assert config.resolve_temperature("chapter_summary") == 0.3
        assert config.resolve_temperature("synthesis_primitives") == 0.8
        assert config.resolve_temperature("primitive_substrate_extraction") == 0.8
        assert config.resolve_temperature("unknown_agent") == 1.0

    def test_default_agent_temperatures(self):
        config = LLMConfig()
        assert config.resolve_temperature("chapter_summary") == 0.3
        assert config.resolve_temperature("synthesis_primitives") == 0.8
        assert config.resolve_temperature("narrative_strategy_skeleton") == 0.5
        assert config.resolve_temperature("narrative_strategy_enrichment") == 0.5
        assert config.resolve_temperature("quality_judge") == 0.2

    def test_resolve_model_defaults(self):
        config = LLMConfig()
        assert config.resolve_model("chapter_summary") == "claude-haiku-4-5"
        assert config.resolve_model("synthesis_primitives") == "claude-opus-4-8"
        assert config.resolve_model("narrative_strategy") == "claude-opus-4-8"
        assert config.resolve_model("theme_decomposition") == "claude-sonnet-4-6"

    def test_resolve_max_retry_attempts_defaults(self):
        config = LLMConfig()
        assert config.resolve_max_retry_attempts("chapter_summary") == 3
        assert config.resolve_max_retry_attempts("passage_extraction") == 4
        assert config.resolve_max_retry_attempts("synthesis_primitives") == 2
        assert config.resolve_max_retry_attempts("narrative_strategy_enrichment") == 2
        assert config.resolve_max_retry_attempts("episode_planning") == 3
        assert config.resolve_max_retry_attempts("unknown_agent") == 2

    def test_resolve_timeout_seconds_defaults(self):
        config = LLMConfig()
        assert config.resolve_timeout_seconds("passage_extraction") == 480.0
        assert config.resolve_timeout_seconds("synthesis_primitives") == 3360.0
        assert config.resolve_timeout_seconds("narrative_strategy") == 900.0
        assert config.resolve_timeout_seconds("narrative_state_reconciliation") == 900.0
        assert config.resolve_timeout_seconds("unknown_agent") == 600.0

    def test_resolve_concurrency_limit_defaults(self):
        config = LLMConfig()
        assert config.resolve_concurrency_limit("chapter_summary") == 48
        assert config.resolve_concurrency_limit("book_summary") == 15
        assert config.resolve_concurrency_limit("passage_extraction") == 26
        assert config.resolve_concurrency_limit("synthesis_primitives") == 3
        assert config.resolve_concurrency_limit("episode_architecture") == 6
        assert config.resolve_concurrency_limit("episode_planning") == 9
        assert config.resolve_concurrency_limit("episode_writing") == 9
        assert config.resolve_concurrency_limit("spoken_delivery") == 9
        assert config.resolve_concurrency_limit("unknown_agent") is None

    def test_resolve_thinking_budget_defaults(self):
        config = LLMConfig()
        assert config.resolve_thinking_budget("theme_decomposition") == 30_000
        assert config.resolve_thinking_budget("synthesis_primitives") == 30_000
        assert config.resolve_thinking_budget("narrative_strategy") == 30_000
        assert config.resolve_thinking_budget("episode_planning") == 30_000
        assert config.resolve_thinking_budget("chapter_summary") is None

    def test_resolve_anthropic_thinking_effort_defaults_from_legacy_budgets(self):
        config = LLMConfig()
        assert config.resolve_anthropic_thinking_effort("episode_writing") == "xhigh"
        assert (
            config.resolve_anthropic_thinking_effort("theme_decomposition") == "xhigh"
        )
        assert (
            config.resolve_anthropic_thinking_effort("synthesis_primitives") == "xhigh"
        )
        assert (
            config.resolve_anthropic_thinking_effort("narrative_strategy")
            == "xhigh"
        )
        assert config.resolve_anthropic_thinking_effort("chapter_summary") is None

    def test_resolve_provider_uses_canonical_aliases(self):
        config = LLMConfig()
        assert config.resolve_provider("primitive_substrate_extraction") == "anthropic"
        assert config.resolve_provider("synthesis_primitives") == "anthropic"

    def test_resolve_anthropic_thinking_effort_prefers_explicit_override(self):
        config = LLMConfig(
            thinking_budget_tokens={"custom_schema": 35_000},
            anthropic_thinking_effort_overrides={"custom_schema": "LOW"},
        )
        assert config.resolve_anthropic_thinking_effort("custom_schema") == "low"

    def test_rejects_invalid_anthropic_thinking_effort_override(self):
        with pytest.raises(ValidationError, match="Invalid anthropic thinking effort"):
            LLMConfig(anthropic_thinking_effort_overrides={"custom_schema": "turbo"})

    def test_normalizes_openai_base_url(self):
        config = LLMConfig(base_url="https://api.openai.com")
        assert config.base_url == "https://api.openai.com/v1"


class TestTTSConfig:
    def test_defaults(self):
        config = TTSConfig()
        assert config.provider == "openai-compatible"
        assert config.model_name == "tts-1-hd"
        assert config.voice == "fable"
        assert config.speed == 0.9

    def test_kokoro_chunk_validation(self):
        with pytest.raises(ValueError, match="kokoro_chunk_max_words"):
            TTSConfig(kokoro_chunk_min_words=600, kokoro_chunk_max_words=500)


class TestPipelineRuntimeConfig:
    def test_defaults(self):
        config = PipelineRuntimeConfig()
        assert config.max_chunk_words == 750
        assert config.chunk_overlap_words == 30
        assert config.max_repair_attempts == 3
        assert config.episode_architecture_concurrency == 12
        assert config.episode_planning_concurrency == 12
        assert config.episode_write_concurrency == 8
        assert config.spoken_delivery_concurrency == 8
        assert config.tts_concurrency == 5
        assert config.spoken_words_per_minute == 145

    def test_thematic_defaults(self):
        config = PipelineRuntimeConfig()
        assert config.max_axes == 20
        assert config.min_axes == 12
        assert config.passage_retrieval_percentage == 0.25
        assert config.pre_axis_total_budget == 1500
        assert config.passage_retrieval_min_per_book == 10
        assert config.passage_retrieval_max_per_book == 25
        assert config.axis_candidate_target_total == 60
        assert config.pre_axis_floor == 30
        assert config.admission_floor_per_book == 0
        assert config.retrieval_relevance_power == 2.5
        assert config.synthesis_axis_pct == 1.0
        assert config.synthesis_axis_min == 12
        assert config.synthesis_axis_max == 20
        assert config.synthesis_floor_budget_fraction == 0.0
        assert config.synthesis_axis_floor_min == 0
        assert config.synthesis_axis_floor_max == 0
        assert config.synthesis_axis_ceiling_multiplier == 1.68
        assert config.synthesis_trim_top_fraction == 0.10
        assert config.synthesis_trim_mid_fraction == 0.20
        assert config.synthesis_trim_top_keep_fraction == 0.35
        assert config.synthesis_trim_mid_keep_fraction == 0.25
        assert config.synthesis_trim_tail_keep_fraction == 0.15
        assert config.planning_axis_pct == 1.0
        assert config.planning_axis_min == 10
        assert config.planning_axis_max == 15
        assert config.synthesis_total_passage_cap == 750
        assert config.planning_total_passage_cap == 300
        assert config.architecture_section_target_min == 12
        assert config.architecture_section_target_max == 18
        assert config.scene_card_target_min == 36
        assert config.scene_card_target_max == 42
        assert config.passage_extraction_concurrency == 16
        assert config.llm_global_max_concurrency == 30

    def test_rejects_removed_retrieval_weighting_fields(self):
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(retrieval_conf_weight=0.2)
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(rerank_top_k=30)
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(post_axis_total_budget=1200)
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(post_axis_floor=20)
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(post_axis_cap=125)
        with pytest.raises(ValidationError):
            PipelineRuntimeConfig(post_axis_signal_power=2.5)

    def test_retrieval_budget_bounds_validation(self):
        with pytest.raises(ValueError, match="passage_retrieval_max_per_book"):
            PipelineRuntimeConfig(
                passage_retrieval_min_per_book=21,
                passage_retrieval_max_per_book=20,
            )

    def test_axis_budget_bounds_validation(self):
        with pytest.raises(ValueError, match="synthesis_axis_max"):
            PipelineRuntimeConfig(synthesis_axis_min=40, synthesis_axis_max=20)
        with pytest.raises(ValueError, match="synthesis_axis_floor_max"):
            PipelineRuntimeConfig(
                synthesis_axis_floor_min=15, synthesis_axis_floor_max=10
            )
        with pytest.raises(ValueError, match="planning_axis_max"):
            PipelineRuntimeConfig(planning_axis_min=50, planning_axis_max=30)
        with pytest.raises(ValueError, match="architecture_section_target_max"):
            PipelineRuntimeConfig(
                architecture_section_target_min=9,
                architecture_section_target_max=8,
            )
        with pytest.raises(ValueError, match="synthesis trim top and mid fractions"):
            PipelineRuntimeConfig(
                synthesis_trim_top_fraction=0.75,
                synthesis_trim_mid_fraction=0.30,
            )
