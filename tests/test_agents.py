"""Unit tests for active agents in the redesigned pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent, ChapterSummaryResponse
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
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
        assert "12-20 strong thematic axes" in agent.instructions
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
        assert "epochal_turns (25–30)" in agent.instructions
        assert "decisions_and_nondecisions (25–30)" in agent.instructions
        assert "telling_details (15–24)" in agent.instructions
        assert "character_engines (15–23)" in agent.instructions
        assert "perspective_windows (11–15)" in agent.instructions
        assert "moral_traps (10–15)" in agent.instructions
        assert "afterlives (11–18)" in agent.instructions
        assert "recurring_images_and_symbols (10–15)" in agent.instructions
        assert "ironies_and_reversals (15–16)" in agent.instructions
        assert "worlds_in_collision" not in agent.instructions
        assert "Score every primitive on 0.0–1.0" in agent.instructions
        assert "Score every primitive on 0.0–1.0 using five distinct questions:" in agent.instructions

    def test_narrative_strategy_agent_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"primitives_by_family": {"epochal_turns": []}},
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
        assert "Turn the primitive synthesis map into a series structure." in agent.instructions
        assert "`core_primitive_ids`" in agent.instructions
        assert "`support_primitive_roles`" in agent.instructions
        assert "`recall_primitive_ids`" in agent.instructions
        assert "`core_primitive_ids` must contain 7-10 primitives." in agent.instructions
        assert "`support_primitive_roles` must contain 10-14 primitives." in agent.instructions
        assert "`actor_arc_directives` must contain only the 2-4 actors" in agent.instructions
        assert "`arc_threads`" in agent.instructions
        assert "`arc_type`" in agent.instructions
        assert "SUPPORT ROLE DEFINITIONS" in agent.instructions
        assert "Choose `chronological` when:" in agent.instructions
        assert "`thematic_axes` are guardrails, not assignment units." in agent.instructions
        assert "Build each episode around one controlling proposition expressed through one explicit set of core primitives." in agent.instructions
        assert "Keep the listener-facing question narrow and concrete." in agent.instructions
        assert "`actor_arc_summary`" not in agent.instructions

    def test_episode_architecture_agent_payload(self):
        agent = EpisodeArchitectureAgent(_mock_llm())
        payload = agent.build_payload(
            episode={"episode_number": 1, "title": "Episode 1"},
            synthesis_map={"primitives_by_family": {"epochal_turns": []}},
            project_metadata={"theme": "War on terror"},
            core_passages=[{"passage_id": "p1"}],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            architecture_feedback={"issue": "missing_turn"},
        )
        assert agent.schema_name == "episode_architecture"
        assert payload["core_passages"][0]["passage_id"] == "p1"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["architecture_feedback"]["issue"] == "missing_turn"
        assert "Convert the episode spine into 8-12 binding sections." in agent.instructions
        assert "The final section must use `purpose` = `closing`." in agent.instructions
        assert "approx_runtime_minutes` at or below 2.0" in agent.instructions
        assert "Do not build a second ending." in agent.instructions
        assert "Place 10-14 support primitives maximum across sections." in agent.instructions
        assert "Do not place all assigned support primitives by default." in agent.instructions
        assert "Ensure the sum of `sections[].approx_runtime_minutes` lands within the project's allowed episode runtime range." in agent.instructions
        assert "`priority_core_passage_ids`" in agent.instructions

    def test_episode_planning_agent_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_episode={"episode_number": 1, "episode_spine": {"listener_question": "Question?"}},
            architecture={"episode_number": 1},
            synthesis_map={"primitives_by_family": {"epochal_turns": []}},
            project_metadata={"theme": "War on terror"},
            available_passages=[{"passage_id": "p1"}],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            planning_feedback={"issue": "missing_spine_coverage"},
        )
        assert agent.schema_name == "episode_planning"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["planning_feedback"]["issue"] == "missing_spine_coverage"
        assert "`available_passages`" in agent.instructions
        assert "`estimated_duration_seconds`" in agent.instructions
        assert "PRIORITY RULES" in agent.instructions
        assert "Every scene card must be grounded in provided `passage_ids`." in agent.instructions
        assert "Target 45–55 scene cards" in agent.instructions
        assert "Build each section through accumulation." in agent.instructions
        assert "Do not distribute primitives evenly by default." in agent.instructions
        assert "If a primitive is not clearly classified in the inputs, treat it as support." in agent.instructions
        assert "core-led scenes        ~60–70% of total runtime" in agent.instructions
        assert "support-led scenes     ~30–35%" in agent.instructions
        assert "The final architecture `closing` section must expand to exactly one scene" in agent.instructions
        assert "It must keep `estimated_duration_seconds` ≤ 120." in agent.instructions
        assert "The framing should orient the listener without pre-explaining the episode." in agent.instructions
        assert "should create curiosity in the same territory as" in agent.instructions
        assert "drawn ONLY from this scene's" in agent.instructions
        assert "If architecture omitted a support or recall primitive" in agent.instructions
        assert "When a section has `priority_core_passage_ids`, prefer those passages" in agent.instructions
        assert "Use `actors[]` for scene actors." in agent.instructions
        assert "`actors[].arc_bindings[].scene_role` describes an ACTOR's role inside the" in agent.instructions
        assert "`coverage_depth`" not in agent.instructions
        assert "Do not distribute primitives evenly by default." in agent.instructions
        assert "`section_id`" in agent.instructions
        assert "`dominant_primitive_id`" in agent.instructions
        assert "`spine_relation`" in agent.instructions
        assert "`state_effect`" in agent.instructions
        assert "setup, shock, action, consequence, reaction, contestation, synthesis" in agent.instructions
        assert "Do NOT bind an actor just because they're named in the evidence." in agent.instructions
        assert "Do not mix them." in agent.instructions
        assert "`batch_id`" not in agent.instructions
        assert "`facet_id`" not in agent.instructions
        assert "`actor_throughline`" not in agent.instructions

    def test_writing_agent_payload(self):
        agent = WritingAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            strategy_episode={"episode_number": 1},
            architecture={"episode_number": 1, "sections": []},
            episode_plan={"episode_number": 1},
            passages=[{"passage_id": "p1"}],
            book_metadata=[{"book_id": "b1"}],
            episode_target_word_count_lower=120,
            episode_target_word_count_higher=180,
            skip_grounding=True,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["prior_window_continuity"]["completed_scene_count"] == 1
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 120
        assert payload["episode_target_word_count_higher"] == 180
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["architecture"]["episode_number"] == 1
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "Draft all `plan.scene_cards` in order." in agent.instructions
        assert "`strategy_episode.episode_spine.core_primitive_ids` as the episode's load-bearing material" in agent.instructions
        assert "next-episode teaser copy" not in agent.instructions
        assert "`plan.framing.preview` is rendered separately" not in agent.instructions
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "computed at 130 WPM" in agent.instructions
        assert "computed at 150 WPM" in agent.instructions
        assert "`passages[].text`" in agent.instructions
        assert agent.instructions.count("Optional `actor_metadata`") == 1
        assert agent.instructions.count("Do not cite actor metadata.") == 1
        assert "Passage evidence wins if actor metadata and passages conflict." in agent.instructions
        assert "Optional `prior_window_continuity`" in agent.instructions
        assert "Treat it as reference-only guidance for handoff, pacing, and continuity." in agent.instructions
        assert "When `prior_window_continuity` is present, use it only to maintain local continuity across the split." in agent.instructions
        assert "`prior_window_continuity` is reference-only." in agent.instructions
        assert "target ranges already encode narrative importance" in agent.instructions
        assert "Write one prose item for each input `plan.scene_cards[]` item." in agent.instructions
        assert "Return one output item per input scene card" in agent.instructions
        assert "Target 8-12 prose sections for the episode" not in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`action`: show named actors doing concrete things" in agent.instructions
        assert "Do not write standalone transition paragraphs" in agent.instructions
        assert "Resolve each scene actor `arc_bindings[].thread_id` against `strategy_episode.actor_arc_directives[].arc_threads[]`" in agent.instructions
        assert "Use arc thread `premise`, `pressure`, `movement`, and `resolution` as narrative guidance" in agent.instructions
        assert "Use `arc_bindings[].scene_use` as the actor arc operation for the scene" in agent.instructions
        assert "`introduce`: establish the actor's episode function" in agent.instructions
        assert "`avoid`: keep the actor present without foregrounding the arc" in agent.instructions
        assert "Use `arc_bindings[].weight` to scale narrative attention" in agent.instructions
        assert "Do not restate the same actor function" in agent.instructions

    def test_writing_response_allows_teaser_line(self):
        response = WritingAgent(_mock_llm()).response_model.model_validate(
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

        assert response.scene_prose[0].scene_card_id == "scene_1"

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
            strategy_episode={"episode_number": 1},
            architecture={"episode_number": 1, "sections": []},
            episode_plan={"episode_number": 1},
            passages=[{"passage_id": "p1", "text": "Evidence"}],
            book_metadata=[{"book_id": "b1"}],
            episode_target_word_count_lower=140,
            episode_target_word_count_higher=220,
            skip_grounding=True,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["prior_window_continuity"]["completed_scene_count"] == 1
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 140
        assert payload["episode_target_word_count_higher"] == 220
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["architecture"]["episode_number"] == 1
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "`strategy_episode.episode_spine.core_primitive_ids` are the episode's load-bearing material" in agent.instructions
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "Draft the full episode" in agent.instructions
        assert "next-episode teaser copy" not in agent.instructions
        assert "`plan.framing.preview` is rendered separately" not in agent.instructions
        assert "Target total narration for this call within" in agent.instructions
        assert "Write one prose item for each input scene card." in agent.instructions
        assert "Return one output item per input scene card" in agent.instructions
        assert "Aim to deliver the episode in 8-12 prose sections" not in agent.instructions
        assert "Optional `actor_metadata`" in agent.instructions
        assert "Passage evidence wins if actor metadata and passages conflict." in agent.instructions
        assert "Do not cite actor metadata." in agent.instructions
        assert "Optional `prior_window_continuity`" in agent.instructions
        assert "Treat it as reference-only guidance for handoff, pacing, and continuity." in agent.instructions
        assert "When `prior_window_continuity` is present, use it only to maintain local continuity across the split." in agent.instructions
        assert "`prior_window_continuity` is reference-only." in agent.instructions
        assert "Do not include a `citations` field" in agent.instructions
        assert "Populate `source_book_ids`" in agent.instructions
        assert "target ranges already encode narrative importance" in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`action`: write an observable beat: named actors doing concrete things" in agent.instructions
        assert "Do not output standalone transitions." in agent.instructions
        assert "Resolve each scene actor `arc_bindings[].thread_id` against `strategy_episode.actor_arc_directives[].arc_threads[]`" in agent.instructions
        assert "Use arc thread `premise`, `pressure`, `movement`, and `resolution` as narrative guidance" in agent.instructions
        assert "Use `arc_bindings[].scene_use` as the actor arc operation for the scene" in agent.instructions
        assert "`introduce`: establish the actor's episode function" in agent.instructions
        assert "`avoid`: keep the actor present without foregrounding the arc" in agent.instructions
        assert "Use `arc_bindings[].weight` to scale narrative attention" in agent.instructions
        assert "Do not restate the same actor function" in agent.instructions

    def test_writing_no_citations_response_allows_teaser_line(self):
        response = WritingAgentNoCitations(_mock_llm()).response_model.model_validate(
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

        assert response.scene_prose[0].scene_card_id == "scene_1"

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
        assert "Your job is to turn one already-written batch of episode prose into spoken narration" in agent.instructions
        assert "INPUT" in agent.instructions
        assert "`script.prose_sections[].text` is the canonical source for the batch." in agent.instructions
        assert "`script.framing` is episode-level scaffolding carried through the pipeline" in agent.instructions
        assert "PRIORITY RULES" in agent.instructions
        assert "WORK ORDER" in agent.instructions
        assert "PLANNING WORKFLOW" in agent.instructions
        assert "Do not silently repair any source contradiction" not in agent.instructions
        assert "Do not fix a source contradiction invisibly just to make the narration cleaner." in agent.instructions
        assert "PODCAST QUALITY" in agent.instructions
        assert "TTS AND SPEECH HINTS" in agent.instructions
        assert "SELF-CHECK BEFORE RETURNING" in agent.instructions
        assert "Return only valid JSON matching expected_schema exactly." in agent.instructions
        assert "Return exactly two top-level keys:" in agent.instructions
        assert "No wrapper keys." in agent.instructions
        assert "Rewrite all of it into one continuous spoken passage in `text`." in agent.instructions
        assert "If `previous_spoken_tail` is present, continue rather than restart." in agent.instructions
        assert "Do not repeat or paraphrase the previous tail" in agent.instructions
        assert "Add `speech_hints.pronunciation_hints` only" in agent.instructions
        assert "Add at most 8 pronunciation hints" in agent.instructions
        assert "Does the JSON match `expected_schema` exactly?" in agent.instructions
        assert "Output Format:" not in agent.instructions
        assert "You are the `narrative_historian` stage" not in agent.instructions
        assert "Use the following narration style" not in agent.instructions
        assert "Do not overstate single-cause explanations for partition" not in agent.instructions
        assert "Every sentence in the original script serves a purpose." not in agent.instructions
        assert "Do not simply delete structural sentences" not in agent.instructions
        assert "speech_hints" in agent.instructions
        assert "transition_id" not in agent.instructions
        assert "previous_spoken_text" not in payload
        assert "previous_spoken_tail" not in payload
        assert "upcoming_batches_summary" not in payload
        assert "batch_index" not in payload
        assert "batch_count" not in payload


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
                    "primitives_by_family": {
                        "epochal_turns": [{"id": f"primitive_{idx}"} for idx in range(1, 7)]
                    }
                },
                thematic_axes=[{"axis_id": "axis_1"}],
                project_metadata={"theme": "War on terror"},
                episode_count=6,
            )
        )
        assert result.recommended_episode_count == 6
        assert result.episodes[0].episode_spine.core_primitive_ids == [
            "primitive_1",
            "primitive_2",
            "primitive_3",
            "primitive_4",
            "primitive_5",
        ]
