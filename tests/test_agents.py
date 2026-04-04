"""Unit tests for all agents."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from podcast_agent.agents.base import Agent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.agents.framing import EpisodeFramingAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import SeriesPlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.source_weaving import SourceWeavingAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.structuring import StructuringAgent
from podcast_agent.agents.synthesis_mapping import SynthesisMappingAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterInfo,
    EpisodeFraming,
    GroundingReport,
    NarrativeStrategy,
    ThematicAxis,
)


def _mock_llm():
    return MagicMock()


class TestStructuringAgent:
    def test_schema_name(self):
        agent = StructuringAgent(_mock_llm())
        assert agent.schema_name == "structuring"

    def test_build_payload(self):
        agent = StructuringAgent(_mock_llm())
        book = BookRecord(
            book_id="b1", title="Test", author="A",
            source_path="/test.txt", source_type="txt",
        )
        payload = agent.build_payload(book, "Some text window", window_offset=100)
        assert payload["book_id"] == "b1"
        assert payload["text_window"] == "Some text window"
        assert payload["window_character_offset"] == 100


class TestThemeDecompositionAgent:
    def test_schema_name(self):
        agent = ThemeDecompositionAgent(_mock_llm())
        assert agent.schema_name == "theme_decomposition"

    def test_build_payload(self):
        agent = ThemeDecompositionAgent(_mock_llm())
        books = [
            BookRecord(
                book_id="b1", title="Book 1", author="Author A",
                source_path="/a.txt", source_type="txt",
                chapters=[ChapterInfo(
                    title="Ch1", start_index=0, end_index=100,
                    word_count=50, summary="Summary.",
                )],
            ),
            BookRecord(
                book_id="b2", title="Book 2", author="Author B",
                source_path="/b.txt", source_type="txt",
            ),
        ]
        payload = agent.build_payload(
            theme="AI ethics", theme_elaboration="Focus on bias", books=books,
        )
        assert payload["theme"] == "AI ethics"
        assert len(payload["books"]) == 2
        assert payload["books"][0]["chapters"][0]["title"] == "Ch1"
        assert payload["books"][0]["chapters"][0]["summary"] == "Summary."


class TestChapterSummaryAgent:
    def test_schema_name(self):
        agent = ChapterSummaryAgent(_mock_llm())
        assert agent.schema_name == "chapter_summary"

    def test_instructions_target_longer_summary(self):
        agent = ChapterSummaryAgent(_mock_llm())
        assert "4-6 sentence" in agent.instructions


class TestPassageExtractionAgent:
    def test_schema_name(self):
        agent = PassageExtractionAgent(_mock_llm())
        assert agent.schema_name == "passage_extraction"

    def test_build_payload(self):
        agent = PassageExtractionAgent(_mock_llm())
        payload = agent.build_payload(
            axis_id="ax1", axis_name="Decision Making",
            axis_description="How decisions are made",
            candidate_passages=[
                {"passage_id": "p1", "text": "Some passage", "book_id": "b1"},
            ],
        )
        assert payload["axis_id"] == "ax1"
        assert len(payload["candidate_passages"]) == 1

    def test_instructions_require_different_books_for_cross_pairs(self):
        agent = PassageExtractionAgent(_mock_llm())
        assert "MUST connect passages from different books" in agent.instructions
        assert "Never pair two passages from the same book" in agent.instructions


class TestSynthesisMappingAgent:
    def test_schema_name(self):
        agent = SynthesisMappingAgent(_mock_llm())
        assert agent.schema_name == "synthesis_mapping"

    def test_build_payload(self):
        agent = SynthesisMappingAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj1",
            axes_summary=[{"axis_id": "ax1", "name": "Test"}],
            passages_by_axis={"ax1": [{"passage_id": "p1"}]},
            cross_book_pairs=[],
            book_metadata=[{"book_id": "b1", "title": "Book 1"}],
        )
        assert payload["project_id"] == "proj1"
        assert "chapters" not in payload["books"][0]


class TestNarrativeStrategyAgent:
    def test_schema_name(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        assert agent.schema_name == "narrative_strategy"

    def test_build_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map_summary={"insight_count": 5},
            project_metadata={"theme": "Test"},
            episode_count=3,
        )
        assert payload["requested_episode_count"] == 3

    def test_build_payload_without_episode_override(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map_summary={"insight_count": 5},
            project_metadata={"theme": "Test"},
            episode_count=None,
        )
        assert "requested_episode_count" not in payload


class TestSeriesPlanningAgent:
    def test_schema_name(self):
        agent = SeriesPlanningAgent(_mock_llm())
        assert agent.schema_name == "series_planning"

    def test_build_payload(self):
        agent = SeriesPlanningAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map_summary={}, narrative_strategy={},
            project_metadata={}, episode_count=3, passages_summary={},
        )
        assert payload["episode_count"] == 3
        assert "chapters" not in payload["project"]

    def test_instructions_target_75_to_100_minute_episode(self):
        agent = SeriesPlanningAgent(_mock_llm())
        assert "75-100 minute episode" in agent.instructions


class TestWritingAgent:
    def test_schema_name(self):
        agent = WritingAgent(_mock_llm())
        assert agent.schema_name == "episode_writing"

    def test_build_payload(self):
        agent = WritingAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1, episode_plan={"title": "Ep 1"},
            passages=[{"text": "passage"}],
            book_metadata=[{"book_id": "b1"}],
            max_author_names_per_episode=2,
            prefer_indirect_attribution=True,
            skip_grounding=True,
        )
        assert payload["episode_number"] == 1
        assert payload["skip_grounding"] is True
        assert "chapters" not in payload["books"][0]


class TestSourceWeavingAgent:
    def test_schema_name(self):
        agent = SourceWeavingAgent(_mock_llm())
        assert agent.schema_name == "source_weaving"

    def test_build_payload(self):
        agent = SourceWeavingAgent(_mock_llm())
        payload = agent.build_payload(
            moment={"synthesis_instruction": "contrast"},
            passages=[{"text": "A"}, {"text": "B"}],
            books=[{"title": "Book A"}, {"title": "Book B"}],
        )
        assert payload["moment"]["synthesis_instruction"] == "contrast"


class TestGroundingValidationAgent:
    def test_schema_name(self):
        agent = GroundingValidationAgent(_mock_llm())
        assert agent.schema_name == "grounding_validation"

    def test_build_payload(self):
        agent = GroundingValidationAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1, script={"segments": []},
            passages={"p1": {"text": "passage"}},
        )
        assert payload["episode_number"] == 1


class TestRepairAgent:
    def test_schema_name(self):
        agent = RepairAgent(_mock_llm())
        assert agent.schema_name == "repair"

    def test_build_payload(self):
        agent = RepairAgent(_mock_llm())
        payload = agent.build_payload(
            failing_segments=[{"text": "bad claim"}],
            failure_reasons=[{"status": "FABRICATED"}],
            passages={"p1": {"text": "actual passage"}},
        )
        assert len(payload["failing_segments"]) == 1


class TestSpokenDeliveryAgent:
    def test_schema_name(self):
        agent = SpokenDeliveryAgent(_mock_llm())
        assert agent.schema_name == "spoken_delivery"

    def test_build_payload(self):
        agent = SpokenDeliveryAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            script_segments=[{"text": "content"}],
            max_words_per_segment=250,
            tts_provider="openai",
        )
        assert payload["max_words_per_segment"] == 250


class TestEpisodeFramingAgent:
    def test_schema_name(self):
        agent = EpisodeFramingAgent(_mock_llm())
        assert agent.schema_name == "episode_framing"

    def test_build_payload(self):
        agent = EpisodeFramingAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=2, total_episodes=5,
            current_episode_summary="Current ep",
            previous_episode_summary="Previous ep",
            next_episode_summary="Next ep",
            book_metadata=[{"title": "Book A"}],
        )
        assert payload["episode_number"] == 2
        assert payload["total_episodes"] == 5


class TestAgentRetry:
    """Test retry behavior in Agent.run()."""

    def test_retries_on_retryable_error(self):
        """Agent retries on RetryableGenerationError and succeeds on 3rd attempt."""
        mock_llm = _mock_llm()
        mock_result = MagicMock()
        mock_llm.generate_json.side_effect = [
            RetryableGenerationError("validation failed"),
            RetryableGenerationError("validation failed again"),
            mock_result,
        ]
        agent = StructuringAgent(mock_llm, max_retry_attempts=3)
        result = agent.run({"test": "payload"})
        assert result == mock_result
        assert mock_llm.generate_json.call_count == 3

    def test_retries_on_transient_error(self):
        """Agent retries on TransientLLMError."""
        mock_llm = _mock_llm()
        mock_result = MagicMock()
        mock_llm.generate_json.side_effect = [
            TransientLLMError("timeout"),
            mock_result,
        ]
        agent = StructuringAgent(mock_llm, max_retry_attempts=3)
        result = agent.run({"test": "payload"})
        assert result == mock_result
        assert mock_llm.generate_json.call_count == 2

    def test_non_retryable_error_propagates_immediately(self):
        """Non-retryable errors are raised immediately without retry."""
        mock_llm = _mock_llm()
        mock_llm.generate_json.side_effect = ValueError("non-retryable")
        agent = StructuringAgent(mock_llm, max_retry_attempts=3)
        with pytest.raises(ValueError, match="non-retryable"):
            agent.run({"test": "payload"})
        assert mock_llm.generate_json.call_count == 1

    def test_exhausted_retries_raises_last_error(self):
        """After all retries exhausted, the last error is raised."""
        mock_llm = _mock_llm()
        mock_llm.generate_json.side_effect = RetryableGenerationError("always fails")
        agent = StructuringAgent(mock_llm, max_retry_attempts=2)
        with pytest.raises(RetryableGenerationError, match="always fails"):
            agent.run({"test": "payload"})
        assert mock_llm.generate_json.call_count == 2

    def test_attempt_numbers_passed_to_llm(self):
        """Agent passes attempt/max_attempts to generate_json."""
        mock_llm = _mock_llm()
        mock_result = MagicMock()
        mock_llm.generate_json.side_effect = [
            RetryableGenerationError("fail"),
            mock_result,
        ]
        agent = StructuringAgent(mock_llm, max_retry_attempts=3)
        agent.run({"test": "payload"})
        calls = mock_llm.generate_json.call_args_list
        assert calls[0].kwargs["attempt"] == 1
        assert calls[0].kwargs["max_attempts"] == 3
        assert calls[1].kwargs["attempt"] == 2

    def test_default_max_retry_attempts(self):
        agent = StructuringAgent(_mock_llm())
        assert agent.max_retry_attempts == 3

    def test_custom_max_retry_attempts(self):
        agent = StructuringAgent(_mock_llm(), max_retry_attempts=5)
        assert agent.max_retry_attempts == 5


class TestAllAgentsHaveRequiredAttributes:
    """Verify every agent has schema_name, instructions, and response_model."""

    @pytest.mark.parametrize("agent_class", [
        StructuringAgent,
        ThemeDecompositionAgent,
        PassageExtractionAgent,
        SynthesisMappingAgent,
        NarrativeStrategyAgent,
        SeriesPlanningAgent,
        WritingAgent,
        SourceWeavingAgent,
        GroundingValidationAgent,
        RepairAgent,
        SpokenDeliveryAgent,
        EpisodeFramingAgent,
    ])
    def test_agent_attributes(self, agent_class):
        agent = agent_class(_mock_llm())
        assert hasattr(agent, "schema_name")
        assert isinstance(agent.schema_name, str)
        assert len(agent.schema_name) > 0
        assert hasattr(agent, "instructions")
        assert isinstance(agent.instructions, str)
        assert len(agent.instructions) > 0
        assert hasattr(agent, "response_model")
