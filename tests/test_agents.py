"""Unit tests for active agents in the redesigned pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

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
            chapters=[{"title": "Ch 1", "analysis": {"themes_touched": ["t1"]}}],
        )
        assert agent.schema_name == "book_summary"
        assert payload["chapters"][0]["analysis"]["themes_touched"] == ["t1"]

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
        assert "Produce between 10 and 40 actors" in agent.instructions
        assert "`actor_metadata`" in agent.instructions
        assert "`books`" in agent.instructions
        assert "Actor scalar string fields" in agent.instructions
        assert "`uncertainty_notes` when there is no caveat" in agent.instructions
        assert "Actor list-of-string fields" in agent.instructions
        assert "`chapter_refs` must be an array of objects" in agent.instructions
        assert "`evidence_confidence` must be one of: `high`, `medium`, `low`" in agent.instructions
        assert "`narrative_functions` values must be drawn from" in agent.instructions
        assert "top-level `actor_metadata.relationships`" in agent.instructions
        assert "relationship `confidence` is optional" in agent.instructions
        assert "`motivations`" not in agent.instructions

    def test_theme_decomposition_payload_includes_analysis_object_only(self):
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
                    analysis=ChapterAnalysis(
                        themes_touched=["theme"],
                        major_actors=["actor"],
                        key_events_or_arguments=["event"],
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
        assert "summary" not in chapter
        assert chapter["analysis"]["themes_touched"] == ["theme"]
        assert chapter["analysis"]["major_actors"] == ["actor"]
        assert chapter["analysis"]["key_events_or_arguments"] == ["event"]
        assert set(chapter["analysis"]) == {
            "themes_touched",
            "major_actors",
            "key_events_or_arguments",
        }

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
            passages_by_axis={"axis_1": [{"book_id": "b1", "passages": [{"passage_id": "p1"}]}]},
            cross_book_pairs=[{"passage_a_id": "p1", "passage_b_id": "p2"}],
            book_metadata=[{"book_id": "b1"}],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            synthesis_feedback={"issue": "thin_grounding"},
        )
        assert agent.schema_name == "synthesis_primitives"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["cross_book_pairs"][0]["passage_a_id"] == "p1"
        assert payload["passages_by_axis"]["axis_1"][0]["book_id"] == "b1"
        assert payload["synthesis_feedback"]["issue"] == "thin_grounding"
        assert "PRIORITY RULES (govern everything below)" in agent.instructions
        assert "passages_by_axis: evidence grouped by axis" in agent.instructions
        assert "epochal_turns (12–20)" in agent.instructions
        assert "Score every primitive on 0.0–1.0" in agent.instructions
        assert "Scores must be meaningfully non-flat" in agent.instructions

    def test_synthesis_consolidation_agent_payload(self):
        agent = SynthesisConsolidationAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            primitives={"primitives_by_family": {"epochal_turns": []}},
            axes_summary=[{"axis_id": "axis_1"}],
            book_metadata=[{"book_id": "b1"}],
            series_size_hint=3,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            consolidation_feedback={"issue": "cluster_density"},
        )
        assert agent.schema_name == "synthesis_consolidation"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["series_size_hint"] == 3
        assert payload["consolidation_feedback"]["issue"] == "cluster_density"
        assert "`EvidencePack`" in agent.instructions
        assert "`primitive_ids`" in agent.instructions
        assert "`primary_job`" not in agent.instructions
        assert "`dependency_pack_ids`" not in agent.instructions
        assert "Aim for roughly 45-65 evidence packs" in agent.instructions
        assert "Do not allocate packs to episodes." in agent.instructions
        assert "`actor_ids` only when canonical actors are genuinely central" in agent.instructions

    def test_narrative_strategy_agent_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"evidence_packs": []},
            thematic_axes=[{"axis_id": "axis_1"}],
            project_metadata={"theme": "War on terror"},
            episode_count=3,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            strategy_feedback={"issue": "cluster_home_collision"},
        )
        assert agent.schema_name == "narrative_strategy"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["requested_episode_count"] == 3
        assert payload["strategy_feedback"]["issue"] == "cluster_home_collision"
        assert "default `EpisodeSpine`" in agent.instructions
        assert "`spine_pack_ids` must contain 1-3 packs" in agent.instructions
        assert "Support packs must be typed with exactly one role each" in agent.instructions
        assert "Default total pack budget is 5-7 packs per episode" in agent.instructions
        assert "Infer pack role and recall eligibility" in agent.instructions
        assert "preclassified" in agent.instructions
        assert "`actor_arc_directives` must contain only the 1-4 actors" in agent.instructions
        assert "`arc_threads`" in agent.instructions
        assert "`arc_type`" in agent.instructions
        assert "`actor_arc_summary`" not in agent.instructions

    def test_episode_planning_agent_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            episode={"episode_number": 1},
            synthesis_map={"evidence_packs": []},
            project_metadata={"theme": "War on terror"},
            available_passages=[{"passage_id": "p1"}],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            planning_feedback={"issue": "missing_spine_coverage"},
        )
        assert agent.schema_name == "episode_planning"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["planning_feedback"]["issue"] == "missing_spine_coverage"
        assert "`available_passages`" in agent.instructions
        assert "`estimated_duration_seconds`" not in agent.instructions
        assert "PRIORITY RULES" in agent.instructions
        assert "No scene without passage support" in agent.instructions
        assert "Target 35-45 scene cards" in agent.instructions
        assert "Group scene cards into 6-10 contiguous batches" in agent.instructions
        assert "Every scene card must set `batch_id`" in agent.instructions
        assert "Do not distribute primitives evenly by default." in agent.instructions
        assert "Preserve `episode.actor_arc_directives`" in agent.instructions
        assert "`dominant_pack_id`" in agent.instructions
        assert "`spine_relation`" in agent.instructions
        assert "`state_effect`" in agent.instructions
        assert "`arc_bindings`" in agent.instructions
        assert "Set scene actor `presence` as `primary`, `secondary`, or `background`" in agent.instructions
        assert "`coverage_depth`" not in agent.instructions
        assert "`setup`, `shock`, `action`, `consequence`, `reaction`, `contestation`, `synthesis`" in agent.instructions
        assert "Prefer observable detail, local consequence, and partial legibility over abstract summary." in agent.instructions
        assert "`action` and `consequence` scenes normally have at least one actor." in agent.instructions
        assert "Scene-card `scene_role` describes the whole scene's narrative job." in agent.instructions
        assert "`arc_bindings[].scene_role` is the actor's role inside the scene" in agent.instructions
        assert "`scene_role`: `driver`, `blocked`, `counterforce`, or `subject`" in agent.instructions
        assert "`scene_use`: `introduce`, `develop`, `complicate`, `stage_choice`, `show_consequence`, `pay_off`, or `avoid`" in agent.instructions
        assert "`weight`: optional; `light`, `standard`, or `strong`" in agent.instructions
        assert "Do not bind an actor just because they are named in the evidence." in agent.instructions
        assert "Do not mix them." in agent.instructions
        assert "`actor_throughline`" not in agent.instructions

    def test_writing_agent_payload(self):
        agent = WritingAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            episode_plan={"episode_number": 1},
            passages=[{"passage_id": "p1"}],
            book_metadata=[{"book_id": "b1"}],
            episode_target_word_count_lower=120,
            episode_target_word_count_higher=180,
            skip_grounding=True,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 120
        assert payload["episode_target_word_count_higher"] == 180
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "Draft all `plan.scene_cards` in order." in agent.instructions
        assert "next-episode teaser copy" in agent.instructions
        assert "`plan.framing.preview` is rendered separately" in agent.instructions
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "`passages[].text`" in agent.instructions
        assert agent.instructions.count("Optional `actor_metadata`") == 1
        assert agent.instructions.count("Do not cite actor metadata.") == 1
        assert "Passage evidence wins if actor metadata and passages conflict." in agent.instructions
        assert "target ranges already encode narrative importance" in agent.instructions
        assert "Write one prose item for each input `plan.scene_cards[]` item." in agent.instructions
        assert "Return one output item per input scene card" in agent.instructions
        assert "Target 8-12 prose sections for the episode" not in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`action`: show named actors doing concrete things" in agent.instructions
        assert "Do not write standalone transition paragraphs" in agent.instructions
        assert "Resolve each scene actor `arc_bindings[].thread_id` against `plan.actor_arc_directives[].arc_threads[]`" in agent.instructions
        assert "Use arc thread `premise`, `pressure`, `movement`, and `payoff` as narrative guidance" in agent.instructions
        assert "Use `arc_bindings[].scene_use` as the actor arc operation for the scene" in agent.instructions
        assert "`introduce`: establish the actor's episode function" in agent.instructions
        assert "`avoid`: keep the actor present without foregrounding the arc" in agent.instructions
        assert "Use `arc_bindings[].weight` to scale narrative attention" in agent.instructions
        assert "Do not restate the same actor function" in agent.instructions

    def test_writing_response_rejects_teaser_line(self):
        with pytest.raises(ValidationError, match="next-episode teaser copy"):
            WritingAgent(_mock_llm()).response_model.model_validate(
                {
                    "scene_prose": [
                        {
                            "scene_card_id": "scene_1",
                            "movement_goal": "discover",
                            "text": "The scene lands.\n\nNext time: another story begins.",
                        }
                    ]
                }
            )

    def test_writing_response_allows_ordinary_next_time_phrase(self):
        response = WritingAgent(_mock_llm()).response_model.model_validate(
            {
                "scene_prose": [
                    {
                        "scene_card_id": "scene_1",
                        "movement_goal": "discover",
                        "text": "But the next time you hear the official story, remember the archive.",
                    }
                ]
            }
        )

        assert response.scene_prose[0].scene_card_id == "scene_1"

    def test_writing_agent_no_citations_instructions_and_schema(self):
        agent = WritingAgentNoCitations(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            episode_plan={"episode_number": 1},
            passages=[{"passage_id": "p1", "text": "Evidence"}],
            book_metadata=[{"book_id": "b1"}],
            episode_target_word_count_lower=140,
            episode_target_word_count_higher=220,
            skip_grounding=True,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 140
        assert payload["episode_target_word_count_higher"] == 220
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "Draft the full episode" in agent.instructions
        assert "next-episode teaser copy" in agent.instructions
        assert "`plan.framing.preview` is rendered separately" in agent.instructions
        assert "Target total narration for this call within" in agent.instructions
        assert "Write one prose item for each input scene card." in agent.instructions
        assert "Return one output item per input scene card" in agent.instructions
        assert "Aim to deliver the episode in 8-12 prose sections" not in agent.instructions
        assert "Optional `actor_metadata`" in agent.instructions
        assert "Passage evidence wins if actor metadata and passages conflict." in agent.instructions
        assert "Do not cite actor metadata." in agent.instructions
        assert "Do not include a `citations` field" in agent.instructions
        assert "Populate `source_book_ids`" in agent.instructions
        assert "target ranges already encode narrative importance" in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`action`: write an observable beat: named actors doing concrete things" in agent.instructions
        assert "Do not output standalone transitions." in agent.instructions
        assert "Resolve each scene actor `arc_bindings[].thread_id` against `plan.actor_arc_directives[].arc_threads[]`" in agent.instructions
        assert "Use arc thread `premise`, `pressure`, `movement`, and `payoff` as narrative guidance" in agent.instructions
        assert "Use `arc_bindings[].scene_use` as the actor arc operation for the scene" in agent.instructions
        assert "`introduce`: establish the actor's episode function" in agent.instructions
        assert "`avoid`: keep the actor present without foregrounding the arc" in agent.instructions
        assert "Use `arc_bindings[].weight` to scale narrative attention" in agent.instructions
        assert "Do not restate the same actor function" in agent.instructions

    def test_writing_no_citations_response_rejects_teaser_line(self):
        with pytest.raises(ValidationError, match="next-episode teaser copy"):
            WritingAgentNoCitations(_mock_llm()).response_model.model_validate(
                {
                    "scene_prose": [
                        {
                            "scene_card_id": "scene_1",
                            "movement_goal": "discover",
                            "text": "The scene lands.\n\nIn the next episode, another story begins.",
                            "source_book_ids": ["b1"],
                        }
                    ]
                }
            )

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
        assert "You are the `oral_rewriter` stage" in agent.instructions
        assert "Your job is to recast a literary draft into compelling spoken narration for audio." in agent.instructions
        assert "TRANSFORMATION MANDATE" in agent.instructions
        assert "HOW PARAGRAPHS WORK" in agent.instructions
        assert "Do not tell the listener that a moment matters." in agent.instructions
        assert "Before returning, check:" in agent.instructions
        assert "Narrator-nudge tells that point at the listener or prep the moment:" in agent.instructions
        assert "Return only valid JSON matching expected_schema exactly." in agent.instructions
        assert "Return exactly two top-level keys: text, speech_hints." in agent.instructions
        assert "No wrapper keys. No extra fields." in agent.instructions
        assert "The response applies to the single input `script.prose_sections[0]`." in agent.instructions
        assert "Do not return `section_id`." in agent.instructions
        assert "Do not firm up hedged language. Do not soften firm language." in agent.instructions
        assert "Add `speech_hints.pronunciation_hints` only" in agent.instructions
        assert "JSON matches expected_schema exactly?" in agent.instructions
        assert "Output Format:" not in agent.instructions
        assert "You are the `narrative_historian` stage" not in agent.instructions
        assert "Use the following narration style" not in agent.instructions
        assert "Do not overstate single-cause explanations for partition" not in agent.instructions
        assert "Every sentence in the original script serves a purpose." not in agent.instructions
        assert "Do not simply delete structural sentences" not in agent.instructions
        assert "speech_hints" in agent.instructions
        assert "transition_id" not in agent.instructions


class TestHeuristicClient:
    def test_synthesis_primitives_agent_run_returns_valid_model(self):
        agent = SynthesisPrimitivesAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                project_id="proj",
                axes_summary=[{"axis_id": "axis_1"}],
                passages_by_axis={
                    "axis_1": [{"book_id": "b1", "passages": [{"passage_id": "p1"}, {"passage_id": "p2"}]}]
                },
                cross_book_pairs=[],
                book_metadata=[{"book_id": "b1"}],
            )
        )
        assert result.project_id == "proj"
        assert result.primitives_by_family["epochal_turns"][0].id == "et_001"
        assert result.primitives_by_family["misreadings_and_fantasies"][0].id == "mf_001"

    def test_narrative_strategy_agent_run_returns_spine_first_episodes(self):
        agent = NarrativeStrategyAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                synthesis_map={
                    "evidence_packs": [
                        {
                            "pack_id": f"pack_{idx}",
                            "title": f"Pack {idx}",
                            "local_summary": f"Summary {idx}",
                            "primitive_ids": [f"primitive_{idx}"],
                        }
                        for idx in range(1, 7)
                    ]
                },
                thematic_axes=[{"axis_id": "axis_1"}],
                project_metadata={"theme": "War on terror"},
                episode_count=6,
            )
        )
        assert result.recommended_episode_count == 6
        assert result.episodes[0].episode_spine.spine_pack_ids == ["pack_1"]
