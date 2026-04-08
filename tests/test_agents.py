"""Unit tests for all agents."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from podcast_agent.agents.base import Agent
from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent, ChapterSummaryResponse
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.agents.framing import EpisodeFramingAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.source_weaving import SourceWeavingAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.structuring import StructuringAgent
from podcast_agent.agents.synthesis_mapping import SynthesisMappingAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterAnalysis,
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
                    analysis=ChapterAnalysis(
                        themes_touched=["bias"],
                        major_tensions=["fairness vs scale"],
                        causal_shifts=["policy changes deployment"],
                        narrative_hooks=["The tradeoff becomes visible."],
                        retrieval_keywords=["bias", "governance"],
                    ),
                )],
            ),
            BookRecord(
                book_id="b2", title="Book 2", author="Author B",
                source_path="/b.txt", source_type="txt",
            ),
        ]
        payload = agent.build_payload(
            theme="AI ethics",
            sub_themes=["bias", "governance"],
            theme_elaboration="Focus on bias",
            books=books,
            book_summaries={"b1": "Book 1 summary."},
        )
        assert payload["theme"] == "AI ethics"
        assert payload["sub_themes"] == ["bias", "governance"]
        assert len(payload["books"]) == 2
        assert payload["books"][0]["book_summary"] == "Book 1 summary."
        assert payload["books"][1]["book_summary"] == ""
        assert payload["books"][0]["chapters"][0]["title"] == "Ch1"
        assert payload["books"][0]["chapters"][0]["summary"] == "Summary."
        assert payload["books"][0]["chapters"][0]["themes_touched"] == ["bias"]

    def test_instructions_target_20_to_25_axes(self):
        agent = ThemeDecompositionAgent(_mock_llm())
        assert "20-25 strong analytical lenses" in agent.instructions


class TestBookSummaryAgent:
    def test_schema_name(self):
        agent = BookSummaryAgent(_mock_llm())
        assert agent.schema_name == "book_summary"

    def test_build_payload(self):
        agent = BookSummaryAgent(_mock_llm())
        payload = agent.build_payload(
            theme="AI ethics",
            sub_themes=["bias", "governance"],
            theme_elaboration="Focus on bias",
            book_id="b1",
            title="Book 1",
            author="Author A",
            chapters=[{"title": "Ch1", "summary": "Summary."}],
        )
        assert payload["theme"] == "AI ethics"
        assert payload["sub_themes"] == ["bias", "governance"]
        assert payload["book_id"] == "b1"
        assert payload["chapters"][0]["summary"] == "Summary."

    def test_instructions_reference_theme_and_chapter_summaries(self):
        agent = BookSummaryAgent(_mock_llm())
        assert "project theme" in agent.instructions
        assert "chapter-level summaries" in agent.instructions


class TestChapterSummaryAgent:
    def test_schema_name(self):
        agent = ChapterSummaryAgent(_mock_llm())
        assert agent.schema_name == "chapter_summary"

    def test_instructions_target_longer_summary(self):
        agent = ChapterSummaryAgent(_mock_llm())
        assert "4-6 sentence" in agent.instructions
        assert "project theme" in agent.instructions
        assert "do not force keyword mentions" in agent.instructions

    def test_build_payload_includes_theme_context_and_normalizes_optional_fields(self):
        agent = ChapterSummaryAgent(_mock_llm())
        payload = agent.build_payload(
            theme="Iranian Revolution",
            sub_themes=None,
            theme_elaboration=None,
            book_id="b1",
            title="Book 1",
            author="Author A",
            chapter_title="Chapter 1",
            chapter_text="Some text",
        )
        assert payload["theme"] == "Iranian Revolution"
        assert payload["sub_themes"] == []
        assert payload["theme_elaboration"] == ""
        assert payload["chapter_title"] == "Chapter 1"

    def test_response_model_allows_more_than_six_key_events_or_arguments(self):
        response = ChapterSummaryResponse.model_validate(
            {
                "summary": "Summary.",
                "analysis": {
                    "key_events_or_arguments": [f"event-{idx}" for idx in range(7)],
                },
            }
        )
        assert response.analysis is not None
        assert len(response.analysis.key_events_or_arguments) == 7


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

    def test_instructions_target_insight_volume(self):
        agent = SynthesisMappingAgent(_mock_llm())
        assert "Generate between 40 and 50 insights in the insights array." in agent.instructions
        assert "Generate between 7 and 8 merged narratives" in agent.instructions


class TestNarrativeStrategyAgent:
    def test_schema_name(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        assert agent.schema_name == "narrative_strategy"

    def test_build_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"insight_count": 5},
            thematic_axes=[{"axis_id": "a1"}],
            project_metadata={"theme": "Test"},
            episode_count=3,
        )
        assert payload["requested_episode_count"] == 3
        assert payload["thematic_axes"][0]["axis_id"] == "a1"

    def test_build_payload_without_episode_override(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"insight_count": 5},
            thematic_axes=[],
            project_metadata={"theme": "Test"},
            episode_count=None,
        )
        assert "requested_episode_count" not in payload

    def test_instructions_define_output_schema(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        assert "episode assignment plan" in agent.instructions
        assert "Choose the strategy using:" in agent.instructions
        assert "narrative progression, contrast, and payoff" in agent.instructions
        assert "If project.requested_episode_count is provided, use it as a planning hint for arc shape." in agent.instructions
        assert "Output schema (strict):" in agent.instructions
        assert "episode_arc_details: array of objects" in agent.instructions
        assert "recommended_episode_count between 7 and 8" in agent.instructions
        assert "narrative_stakes: string" in agent.instructions
        assert "episode_assignments: array of objects" in agent.instructions
        assert "driving_question: string" in agent.instructions
        assert "strategy_type: one of thesis_driven, debate, chronological, convergence, mosaic" in agent.instructions
        assert "Return only a JSON object matching this schema." in agent.instructions
        assert "Every SynthesisInsight with podcast_potential > 0.5 must appear in at least one " in agent.instructions
        assert "Target 5-7 insights per episode" in agent.instructions
        assert "usually 3-4 axes" in agent.instructions
        assert (
            "prioritize full coverage of SynthesisInsights with podcast_potential > 0.5"
            in agent.instructions
        )
        assert "exactly one merged_narrative_id" in agent.instructions

    def test_heuristic_strategy_caps_recommended_episode_count_to_eight(self):
        heuristic = HeuristicLLMClient()
        result = heuristic._generate_narrative_strategy(
            {"requested_episode_count": 20, "synthesis_map": {}, "thematic_axes": []}
        )
        assert result["recommended_episode_count"] == 8

    def test_heuristic_strategy_assigns_one_merged_narrative_per_episode_when_available(self):
        heuristic = HeuristicLLMClient()
        result = heuristic._generate_narrative_strategy(
            {
                "synthesis_map": {
                    "merged_narratives": [
                        {"merged_narrative_id": "merged_narrative_001"},
                        {"merged_narrative_id": "merged_narrative_002"},
                    ],
                    "insights": [],
                },
                "thematic_axes": [{"axis_id": "axis_1", "description": "", "guiding_questions": []}],
            }
        )
        assert result["episode_assignments"]
        assert all(item["merged_narrative_id"] is not None for item in result["episode_assignments"])

    def test_heuristic_strategy_targets_three_to_four_axes_when_available(self):
        heuristic = HeuristicLLMClient()
        thematic_axes = [
            {"axis_id": f"axis_{idx}", "description": f"Axis {idx}", "guiding_questions": [f"Q{idx}?"]}
            for idx in range(1, 7)
        ]
        result = heuristic._generate_narrative_strategy(
            {
                "requested_episode_count": 8,
                "synthesis_map": {"insights": []},
                "thematic_axes": thematic_axes,
            }
        )
        axis_counts = [len(item["axes"]) for item in result["episode_assignments"]]
        assert axis_counts
        assert all(3 <= count <= 4 for count in axis_counts)


class TestEpisodePlanningAgent:
    def test_schema_name(self):
        agent = EpisodePlanningAgent(_mock_llm())
        assert agent.schema_name == "episode_planning"

    def test_build_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            episode_assignment={"episode_number": 1},
            narrative_strategy={},
            synthesis_map={},
            project_metadata={},
            available_passages={},
            previous_episode=None,
            next_episode=None,
        )
        assert payload["episode_assignment"]["episode_number"] == 1

    def test_instructions_bind_driving_question(self):
        agent = EpisodePlanningAgent(_mock_llm())
        assert "episode_assignment.driving_question" in agent.instructions
        assert "payoff_shape" in agent.instructions

    def test_instructions_target_140_minute_episode(self):
        agent = EpisodePlanningAgent(_mock_llm())
        assert "140 minutes" in agent.instructions
        assert "125 minutes" in agent.instructions
        assert "70-80 beats" in agent.instructions
        assert "summary_text" in agent.instructions
        assert "full_text" in agent.instructions
        assert "project.book_size_share_by_id" in agent.instructions


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

    def test_instructions_include_runtime_targets(self):
        agent = WritingAgent(_mock_llm())
        assert "plan.target_word_count" in agent.instructions
        assert "payload.beat_word_targets" in agent.instructions
        assert "estimated_duration_seconds" in agent.instructions
        assert "plan.target_word_count" in agent.instructions_no_citations
        assert "payload.beat_word_targets" in agent.instructions_no_citations
        assert "estimated_duration_seconds" in agent.instructions_no_citations
        assert "Target plan.target_duration_minutes for total runtime" not in agent.instructions
        assert "Target plan.target_duration_minutes for total runtime" not in agent.instructions_no_citations
        assert "Required writing constraints" in agent.instructions
        assert "plan.narrative_spine" in agent.instructions
        assert "plan.cross_references" in agent.instructions
        assert "plan.book_balance" in agent.instructions
        assert "synthesis_instruction" in agent.instructions
        assert "transition_hint" in agent.instructions
        assert "Required writing constraints" in agent.instructions_no_citations
        assert "plan.narrative_spine" in agent.instructions_no_citations
        assert "plan.cross_references" in agent.instructions_no_citations
        assert "plan.book_balance" in agent.instructions_no_citations


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

    def test_instructions_define_strict_speech_hints_contract(self):
        agent = SpokenDeliveryAgent(_mock_llm())
        assert "speech_hints" in agent.instructions
        assert "style" in agent.instructions
        assert "pause_before_ms" in agent.instructions
        assert "pace" in agent.instructions
        assert "render_strategy" in agent.instructions
        assert "raw SSML, XML, or markup tags" in agent.instructions


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
        BookSummaryAgent,
        ThemeDecompositionAgent,
        PassageExtractionAgent,
        SynthesisMappingAgent,
        NarrativeStrategyAgent,
        EpisodePlanningAgent,
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
