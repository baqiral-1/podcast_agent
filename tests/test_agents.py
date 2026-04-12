"""Unit tests for active agents in the redesigned pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent, ChapterSummaryResponse
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.synthesis_consolidation import SynthesisConsolidationAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.schemas.models import BookRecord, ChapterAnalysis, ChapterInfo


def _mock_llm() -> MagicMock:
    return MagicMock()


class TestCoreAgents:
    def test_chapter_summary_agent_payload(self):
        agent = ChapterSummaryAgent(_mock_llm())
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["state failure"],
            theme_elaboration="Focus on causal escalation.",
            book_id="b1",
            title="Book 1",
            author="Author A",
            chapter_title="Chapter 1",
            chapter_text="Window text.",
        )
        assert agent.schema_name == "chapter_summary"
        assert payload["chapter_text"] == "Window text."
        assert payload["theme"] == "War on terror"

    def test_chapter_summary_response_allows_more_than_six_key_events(self):
        response = ChapterSummaryResponse.model_validate(
            {
                "summary": "Summary.",
                "analysis": {"key_events_or_arguments": [f"event-{idx}" for idx in range(7)]},
            }
        )
        assert len(response.analysis.key_events_or_arguments) == 7

    def test_book_summary_agent_payload(self):
        agent = BookSummaryAgent(_mock_llm())
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["causality"],
            theme_elaboration="",
            book_id="b1",
            title="Book 1",
            author="Author A",
            chapters=[{"title": "Ch 1", "summary": "Summary."}],
        )
        assert agent.schema_name == "book_summary"
        assert payload["chapters"][0]["summary"] == "Summary."

    def test_theme_decomposition_agent_payload(self):
        agent = ThemeDecompositionAgent(_mock_llm())
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["state failure"],
            theme_elaboration="Trace the escalation.",
            books=[],
            book_summaries={},
        )
        assert agent.schema_name == "theme_decomposition"
        assert payload["sub_themes"] == ["state failure"]
        assert "10-15 strong thematic axes" in agent.instructions
        assert "`books`" in agent.instructions

    def test_theme_decomposition_payload_omits_removed_chapter_analysis_fields(self):
        agent = ThemeDecompositionAgent(_mock_llm())
        book = BookRecord(
            book_id="b1",
            title="Book 1",
            author="Author A",
            source_path="/tmp/book.txt",
            source_type="txt",
            chapters=[
                ChapterInfo(
                    chapter_id="ch1",
                    title="Chapter 1",
                    start_index=0,
                    end_index=100,
                    word_count=100,
                    summary="Summary",
                    analysis=ChapterAnalysis(
                        themes_touched=["theme"],
                        major_tensions=["tension"],
                    ),
                )
            ],
        )
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["state failure"],
            theme_elaboration="Trace the escalation.",
            books=[book],
            book_summaries={"b1": "Book summary"},
        )
        chapter = payload["books"][0]["chapters"][0]
        assert "themes_touched" in chapter
        assert "major_tensions" in chapter
        assert "causal_shifts" not in chapter
        assert "narrative_hooks" not in chapter
        assert "retrieval_keywords" not in chapter

    def test_passage_extraction_agent_payload(self):
        agent = PassageExtractionAgent(_mock_llm())
        payload = agent.build_payload(
            axis_id="axis_1",
            axis_name="Escalation",
            axis_description="How escalation compounds.",
            candidate_passages=[{"passage_id": "p1", "book_id": "b1", "text": "Text"}],
        )
        assert agent.schema_name == "passage_extraction"
        assert payload["candidate_passages"][0]["passage_id"] == "p1"
        assert "`candidate_passages`" in agent.instructions


class TestRedesignedAgents:
    def test_synthesis_primitives_agent_payload_and_instructions(self):
        agent = SynthesisPrimitivesAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            axes_summary=[{"axis_id": "axis_1"}],
            passages_by_axis={"axis_1": [{"passage_id": "p1"}]},
            cross_book_pairs=[],
            book_metadata=[{"book_id": "b1"}],
            synthesis_feedback={"issue": "thin_grounding"},
        )
        assert agent.schema_name == "synthesis_primitives"
        assert payload["synthesis_feedback"]["issue"] == "thin_grounding"
        assert "Do not emit episode architecture" in agent.instructions
        assert "`passages_by_axis`" in agent.instructions

    def test_synthesis_consolidation_agent_payload(self):
        agent = SynthesisConsolidationAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            primitives={"turning_points": []},
            axes_summary=[{"axis_id": "axis_1"}],
            book_metadata=[{"book_id": "b1"}],
            series_size_hint=3,
            consolidation_feedback={"issue": "cluster_density"},
        )
        assert agent.schema_name == "synthesis_consolidation"
        assert payload["series_size_hint"] == 3
        assert payload["consolidation_feedback"]["issue"] == "cluster_density"

    def test_narrative_strategy_agent_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"episode_candidate_clusters": []},
            thematic_axes=[{"axis_id": "axis_1"}],
            project_metadata={"theme": "War on terror"},
            episode_count=3,
            strategy_feedback={"issue": "cluster_home_collision"},
        )
        assert agent.schema_name == "narrative_strategy"
        assert payload["requested_episode_count"] == 3
        assert payload["strategy_feedback"]["issue"] == "cluster_home_collision"

    def test_episode_planning_agent_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            episode={"episode_number": 1},
            synthesis_map={"episode_candidate_clusters": []},
            project_metadata={"theme": "War on terror"},
            available_passages=[{"passage_id": "p1"}],
            planning_feedback={"issue": "uncovered_primary_occurrences"},
        )
        assert agent.schema_name == "episode_planning"
        assert payload["planning_feedback"]["issue"] == "uncovered_primary_occurrences"
        assert "`available_passages`" in agent.instructions

    def test_writing_agent_payload(self):
        agent = WritingAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            batch_id="batch_2",
            episode_plan={"episode_number": 1},
            active_scene_card_ids=["scene_2"],
            passages=[{"passage_id": "p1"}],
            book_metadata=[{"book_id": "b1"}],
            batch_target_word_count_lower=120,
            batch_target_word_count_higher=180,
            skip_grounding=True,
        )
        assert agent.schema_name == "episode_writing"
        assert payload["skip_grounding"] is True
        assert payload["batch_target_word_count_lower"] == 120
        assert payload["batch_target_word_count_higher"] == 180
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "previous_transitions" not in payload
        assert "`active_scene_card_ids`" in agent.instructions
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`batch_target_word_count_lower`" in agent.instructions
        assert "`batch_target_word_count_higher`" in agent.instructions
        assert "`passages[].text`" in agent.instructions

    def test_writing_agent_no_citations_instructions_and_schema(self):
        agent = WritingAgentNoCitations(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            batch_id="batch_1",
            episode_plan={"episode_number": 1},
            active_scene_card_ids=["scene_1"],
            passages=[{"passage_id": "p1", "text": "Evidence"}],
            book_metadata=[{"book_id": "b1"}],
            batch_target_word_count_lower=140,
            batch_target_word_count_higher=220,
            skip_grounding=True,
        )
        assert agent.schema_name == "episode_writing"
        assert payload["skip_grounding"] is True
        assert payload["batch_target_word_count_lower"] == 140
        assert payload["batch_target_word_count_higher"] == 220
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "previous_transitions" not in payload
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`batch_target_word_count_lower`" in agent.instructions
        assert "`batch_target_word_count_higher`" in agent.instructions
        assert "Do not include a `citations` field" in agent.instructions

    def test_grounding_validation_agent_payload(self):
        agent = GroundingValidationAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            script={"prose_sections": []},
            passages={"p1": {"passage_id": "p1"}},
        )
        assert agent.schema_name == "grounding_validation"
        assert payload["cited_passages"]["p1"]["passage_id"] == "p1"
        assert "`cited_passages`" in agent.instructions

    def test_repair_agent_payload(self):
        agent = RepairAgent(_mock_llm())
        payload = agent.build_payload(
            failing_sections=[{"section_id": "section_1"}],
            failing_transitions=[{"transition_id": "transition_1"}],
            failure_reasons=[{"text_unit_id": "section_1", "status": "UNSUPPORTED"}],
            passages={"p1": {"passage_id": "p1"}},
        )
        assert agent.schema_name == "repair"
        assert payload["failure_reasons"][0]["text_unit_id"] == "section_1"

    def test_spoken_delivery_agent_payload(self):
        agent = SpokenDeliveryAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            script={"prose_sections": []},
            max_words_per_segment=250,
            tts_provider="openai",
        )
        assert agent.schema_name == "spoken_delivery"
        assert payload["max_words_per_segment"] == 250
        assert "You are the `narrative_historian` stage" in agent.instructions
        assert "Output Format:" not in agent.instructions
        assert "`section_id`" in agent.instructions
        assert "`transition_id`" in agent.instructions


class TestHeuristicClient:
    def test_synthesis_primitives_agent_run_returns_valid_model(self):
        agent = SynthesisPrimitivesAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                project_id="proj",
                axes_summary=[{"axis_id": "axis_1"}],
                passages_by_axis={"axis_1": [{"passage_id": "p1"}, {"passage_id": "p2"}]},
                cross_book_pairs=[],
                book_metadata=[{"book_id": "b1"}],
            )
        )
        assert result.project_id == "proj"
        assert result.turning_points[0].id == "tp_001"

    def test_narrative_strategy_agent_run_returns_cluster_path_episodes(self):
        agent = NarrativeStrategyAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                synthesis_map={
                    "episode_candidate_clusters": [
                        {"cluster_id": f"cluster_{idx}", "title": f"Cluster {idx}"}
                        for idx in range(1, 7)
                    ]
                },
                thematic_axes=[{"axis_id": "axis_1"}],
                project_metadata={"theme": "War on terror"},
                episode_count=6,
            )
        )
        assert result.recommended_episode_count == 6
        assert result.episodes[0].cluster_path[0].usage == "primary"
