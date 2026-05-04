"""Unit tests for active agents in the redesigned pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent, ChapterSummaryResponse
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_enrichment import PrimitiveEnrichmentAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.langchain.runnables import RetryableGenerationError, TransientLLMError
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.prompts.instructions import section_local_spoken_delivery_instructions
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterAnalysis,
    ChapterInfo,
    DecisionEnrichmentArtifact,
    HumanCostEnrichmentArtifact,
)


def _mock_llm() -> MagicMock:
    return MagicMock()


def _hooks() -> dict[str, str]:
    return {
        "concrete_detail": "A concrete detail lands.",
        "host_lens": "The pressure is visible.",
        "carry_forward": "The residue lingers.",
    }


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
        assert "chapter_id" not in chapter
        assert "title" not in chapter
        assert chapter["analysis"]["themes_touched"] == ["theme"]
        assert chapter["analysis"]["major_actors"] == ["actor"]
        assert chapter["analysis"]["key_events_or_arguments"] == ["event"]
        assert set(chapter["analysis"]) == {
            "themes_touched",
            "major_actors",
            "key_events_or_arguments",
        }

    def test_theme_decomposition_payload_compacts_analysis_fields(self):
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
                        themes_touched=["  theme one  ", "theme two", "theme three", "theme four"],
                        major_actors=["actor one", "actor two", "actor three", "actor four"],
                        key_events_or_arguments=[
                            " ".join(["event"] * 80),
                            "event two",
                            "event three",
                        ],
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
        analysis = payload["books"][0]["chapters"][0]["analysis"]
        assert analysis["themes_touched"] == ["theme one", "theme two", "theme three"]
        assert analysis["major_actors"] == ["actor one", "actor two", "actor three"]
        assert len(analysis["key_events_or_arguments"]) == 2
        assert len(analysis["key_events_or_arguments"][0]) == 180
        assert analysis["key_events_or_arguments"][1] == "event two"

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
        assert "Score passages relative to the other candidates for this axis." in agent.instructions
        assert "Do not tag everything as `exemplifies`." in agent.instructions
        assert "A pair should teach something a single passage cannot." in agent.instructions
        assert "Self-check before returning:" in agent.instructions
        assert "Every input passage appears exactly once in `passages`." in agent.instructions


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
        assert "epochal_turns (30–38)" in agent.instructions
        assert "decisions_and_nondecisions (25–32)" in agent.instructions
        assert "set_piece_scenes (18–31)" in agent.instructions
        assert "telling_details (4–8)" in agent.instructions
        assert "human_costs (17–22)" in agent.instructions
        assert "character_engines (14–22)" in agent.instructions
        assert "coalitions_and_fault_lines (11–16)" in agent.instructions
        assert "systems_and_operating_logics (12–18)" in agent.instructions
        assert "perspective_windows (2–5)" in agent.instructions
        assert "moral_traps (4–7)" in agent.instructions
        assert "afterlives (5–10)" in agent.instructions
        assert "recurring_images_and_symbols (2–5)" in agent.instructions
        assert "ironies_and_reversals (10–13)" in agent.instructions
        assert "worlds_in_collision" not in agent.instructions
        assert "Score every primitive on 0.0–1.0" in agent.instructions
        assert "Score every primitive on 0.0–1.0 using five distinct questions:" in agent.instructions

    def test_narrative_strategy_agent_payload(self):
        agent = NarrativeStrategyAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"primitives_by_family": {"epochal_turns": []}},
            project_metadata={"theme": "War on terror"},
            episode_count=3,
            recommended_episode_count_min=8,
            recommended_episode_count_max=12,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            strategy_feedback={"issue": "cluster_home_collision"},
        )
        assert agent.schema_name == "narrative_strategy"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["requested_episode_count"] == 3
        assert payload["recommended_episode_count_min"] == 8
        assert payload["recommended_episode_count_max"] == 12
        assert payload["strategy_feedback"]["issue"] == "cluster_home_collision"
        assert "Turn the primitive synthesis map into a series structure." in agent.instructions
        assert "`recommended_episode_count_min`" in agent.instructions
        assert "`recommended_episode_count_max`" in agent.instructions
        assert (
            "Otherwise, produce between `recommended_episode_count_min` and "
            "`recommended_episode_count_max` episodes, inclusive."
        ) in agent.instructions
        assert "`core_primitive_ids`" in agent.instructions
        assert "`support_primitive_roles`" in agent.instructions
        assert "`recall_primitive_ids`" in agent.instructions
        assert "`core_primitive_ids` must contain 5-7 primitives." in agent.instructions
        assert "`support_primitive_roles` must contain 5-7 primitives." in agent.instructions
        assert "`actor_arc_directives` must contain only the 2-4 actors" in agent.instructions
        assert "`arc_threads`" in agent.instructions
        assert "`arc_type`" in agent.instructions
        assert "SUPPORT ROLE DEFINITIONS" in agent.instructions
        assert "Choose `chronological` when:" in agent.instructions
        assert "FIRST-PASS GROUPING WORKFLOW" in agent.instructions
        assert "PRIMITIVE-FIRST DISCIPLINE" in agent.instructions
        assert "`thematic_axes`" not in agent.instructions
        assert "Build each episode around one controlling proposition expressed through one explicit set of core primitives." in agent.instructions
        assert "Keep the listener-facing question narrow and concrete." in agent.instructions
        assert "`actor_arc_summary`" not in agent.instructions

    def test_primitive_enrichment_agent_payload(self):
        agent = PrimitiveEnrichmentAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            family="epochal_turns",
            base_primitives=[{"id": "et_1", "family": "epochal_turns"}],
            evidence_by_primitive_id={
                "et_1": {
                    "core_passages": [{"passage_id": "p1", "text": "Text"}],
                    "support_passages": [],
                }
            },
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
        )
        assert agent.schema_name == "primitive_enrichment"
        assert payload["family"] == "epochal_turns"
        assert payload["evidence_by_primitive_id"]["et_1"]["core_passages"][0]["text"] == "Text"
        assert "primitive_ids_by_role" not in payload
        assert "project" not in payload
        instructions = agent.instructions_for_family("epochal_turns")
        assert "`base_primitives`" in instructions
        assert "`evidence_by_primitive_id`" in instructions
        assert "primitive-scoped trimmed evidence with `core_passages` and `support_passages`" in instructions
        assert "`actor_metadata` (optional): slim canonical actor context with `one_line_role`" in instructions
        assert "Goal:" in instructions
        assert "INPUT AND AUTHORITY" in instructions
        assert "Authority order:" in instructions
        assert "WORKING METHOD" in instructions
        assert "TASK" in instructions
        assert "REQUIRED FIELDS" in instructions
        assert "WHAT GOOD LOOKS LIKE" in instructions
        assert "FIELD DISTINCTIONS" in instructions
        assert "WHEN TO STAY NARROW" in instructions
        assert "FAILURE AND AMBIGUITY HANDLING" in instructions
        assert "OUTPUT CONTRACT" in instructions
        assert "SELF-CHECK BEFORE RETURNING" in instructions
        assert "Return one enriched delta for each selected primitive in this batch." in instructions
        assert "Do not add new primitives, merge primitives, split primitives, change type" in instructions
        assert "Treat core passages as decisive evidence." in instructions
        assert "Multiple primitives may share a passage." in instructions
        assert "You are enriching epochal-turn primitives." in instructions
        assert "Return these fields for every selected primitive: `before_state`, `after_state`, `change_driver`, `proof_of_change`, `why_no_return`, `narration_hooks`" in instructions
        assert "A strong epochal turn marks a genuine break in the rules of the story" in instructions
        assert "`proof_of_change` gives the concrete sign that made the turn undeniable" in instructions
        assert "Do not upgrade ordinary escalation, battle intensity, or local drama into an epochal break" in instructions
        assert "primitive_ids_by_role" not in instructions
        assert "rich family" not in instructions
        assert "requested `family`" not in instructions
        assert "family-specific" not in instructions

    def test_primitive_enrichment_family_specific_instructions_remain_rich(self):
        agent = PrimitiveEnrichmentAgent(_mock_llm())

        instructions = agent.instructions_for_family("human_costs")

        assert "INPUT AND AUTHORITY" in instructions
        assert "WORKING METHOD" in instructions
        assert "FAILURE AND AMBIGUITY HANDLING" in instructions
        assert "SELF-CHECK BEFORE RETURNING" in instructions
        assert "You are enriching human-cost primitives." in instructions
        assert "Return these fields for every selected primitive: `actor_ids`, `affected_group`, `cost_type`, `concrete_marker`, `lived_consequence`, `who_saw_it`, `narration_hooks`" in instructions
        assert "A strong human-cost primitive makes the harm land on a concrete group in lived terms" in instructions
        assert "`concrete_marker` is the most speakable physical or social sign of that harm." in instructions
        assert "Leave `actor_ids` empty when the harm is diffuse" in instructions
        assert "Do not invent actor linkage just to satisfy schema pressure." in instructions
        assert "If the evidence only supports generalized hardship" in instructions
        assert "primitive_ids_by_role" not in instructions
        assert "`character_engines`: `actor_id`, `goal`, `pressure_box`, `risk_if_it_breaks`" not in instructions
        assert "`systems_and_operating_logics`: `system_name`, `operating_chain`, `inputs`" not in instructions
        assert "rich family" not in instructions
        assert "requested `family`" not in instructions
        assert "family-specific fields" not in instructions
        assert "FAMILY-SPECIFIC ENRICHMENT RULES" not in instructions

    def test_primitive_enrichment_systems_prompt_is_type_native(self):
        agent = PrimitiveEnrichmentAgent(_mock_llm())

        instructions = agent.instructions_for_family("systems_and_operating_logics")

        assert "You are enriching systems and operating-logics primitives." in instructions
        assert "Return these fields for every selected primitive: `system_name`, `operating_chain`, `inputs`, `outputs`, `where_it_shows_up`, `failure_mode`, `narration_hooks`" in instructions
        assert "A strong systems primitive describes a concrete operating chain" in instructions
        assert "`operating_chain` should give 2-4 short ordered concrete steps inside the chain." in instructions
        assert "Describe a real operating chain, not abstract thesis prose." in instructions
        assert "Bad pattern: 'The political system processes pressure through institutions.'" in instructions
        assert "If the passage only shows one bottleneck or one distorted channel" in instructions
        assert "This family must describe a real operating chain" not in instructions
        assert "rich family" not in instructions

    def test_primitive_enrichment_thin_family_prompts_gain_more_specificity(self):
        agent = PrimitiveEnrichmentAgent(_mock_llm())

        coalition = agent.instructions_for_family("coalitions_and_fault_lines")
        assert "Return these fields for every selected primitive: `actor_ids`, `alignment_type`, `coalition_phase`, `alignment_shape`, `alignment_basis`, `fracture_trigger`, `narration_hooks`" in coalition
        assert "`alignment_type` classifies the kind of alignment: tactical, strategic, institutional, or situational." in coalition
        assert "`coalition_phase` says whether the evidence shows the coalition forming, holding, fracturing, or breaking." in coalition
        assert "A strong coalition primitive shows why actors align" in coalition
        assert "`alignment_basis` is the practical reason it holds for now." in coalition
        assert "Do not confuse simple coexistence, shared enemies, or momentary simultaneity with a meaningful coalition." in coalition
        assert "If the evidence only supports a narrow tactical alignment" in coalition

        moral_trap = agent.instructions_for_family("moral_traps")
        assert "A strong moral-trap primitive shows an actor bound by real duties or loyalties" in moral_trap
        assert "Do not confuse a hard preference, policy disagreement, or sad outcome with a moral trap" in moral_trap
        assert "If the evidence only shows a difficult choice without conflicting obligations" in moral_trap

        reversal = agent.instructions_for_family("ironies_and_reversals")
        assert "A strong irony or reversal shows actors aiming at one result and producing a meaningfully inverted result" in reversal
        assert "Do not label every failed plan a reversal" in reversal
        assert "If the evidence shows only setback or friction" in reversal

        scene = agent.instructions_for_family("set_piece_scenes")
        assert "A strong set-piece scene is playable" in scene
        assert "Do not turn broad chronology, campaign background, or multi-month drift into a pseudo-scene." in scene
        assert "If the evidence gives atmosphere and consequence but no clear hinge moment" in scene

        decision = agent.instructions_for_family("decisions_and_nondecisions")
        assert "Return these fields for every selected primitive: `actor_ids`, `decision_trigger`, `decision_question`, `decision_mode`, `options_considered`, `next_result`, `narration_hooks`" in decision
        assert "`decision_question` should name the live hinge the actor is resolving" in decision
        assert "`enrichment_feedback` (optional): retry feedback" in decision

    def test_primitive_enrichment_agent_payload_includes_feedback(self):
        agent = PrimitiveEnrichmentAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            family="human_costs",
            base_primitives=[{"id": "hc_1", "family": "human_costs"}],
            evidence_by_primitive_id={
                "hc_1": {
                    "core_passages": [{"passage_id": "p1", "text": "Text"}],
                    "support_passages": [],
                }
            },
            enrichment_feedback={"issue": "schema_validation_failed"},
        )

        assert payload["enrichment_feedback"]["issue"] == "schema_validation_failed"

    def test_primitive_enrichment_run_dispatches_family_specific_schema_and_prompt(self):
        llm = _mock_llm()
        llm.generate_json.return_value = HumanCostEnrichmentArtifact.model_validate(
            {
                "project_id": "proj",
                "family": "human_costs",
                "enriched_primitives": [
                    {
                        "id": "hc_1",
                        "family": "human_costs",
                        "actor_ids": [],
                        "affected_group": "camp followers",
                        "cost_type": "displacement",
                        "concrete_marker": "Families carry bedding into the road.",
                        "lived_consequence": "Households lose shelter and income.",
                        "who_saw_it": "plain to witnesses but politically deniable",
                        "narration_hooks": _hooks(),
                    }
                ],
            }
        )
        agent = PrimitiveEnrichmentAgent(llm)

        result = agent.run(
            {
                "project_id": "proj",
                "family": "human_costs",
                "base_primitives": [{"id": "hc_1", "family": "human_costs", "actor_ids": []}],
                "evidence_by_primitive_id": {},
            }
        )

        assert isinstance(result, HumanCostEnrichmentArtifact)
        kwargs = llm.generate_json.call_args.kwargs
        assert kwargs["response_model"] is HumanCostEnrichmentArtifact
        assert "You are enriching human-cost primitives." in kwargs["instructions"]
        assert "Return these fields for every selected primitive: `actor_ids`, `affected_group`, `cost_type`, `concrete_marker`, `lived_consequence`, `who_saw_it`, `narration_hooks`" in kwargs["instructions"]
        assert "Leave `actor_ids` empty when the harm is diffuse" in kwargs["instructions"]
        assert "`character_engines`: `actor_id`, `goal`, `pressure_box`, `risk_if_it_breaks`" not in kwargs["instructions"]

    def test_primitive_enrichment_run_passes_feedback_to_retry_attempt(self):
        llm = _mock_llm()
        invalid_payload = {
            "project_id": "proj",
            "family": "decisions_and_nondecisions",
            "enriched_primitives": [
                {
                    "id": "dn_1",
                    "family": "decisions_and_nondecisions",
                    "actor_ids": ["actor_1"],
                    "decision_trigger": "Fresh pressure forces a move.",
                    "decision_question": "Should the court move now?",
                    "decision_mode": "decision",
                    "options_considered": ["advance", "delay"],
                    "next_result": "The move forces the next confrontation.",
                    "decision_query": "",
                    "narration_hooks": _hooks(),
                }
            ],
        }

        try:
            DecisionEnrichmentArtifact.model_validate(invalid_payload)
        except Exception as validation_exc:
            try:
                raise RetryableGenerationError(
                    "Schema validation failed for primitive_enrichment",
                    data={"raw_payload": invalid_payload},
                ) from validation_exc
            except RetryableGenerationError as retry_exc:
                first_exc = retry_exc
        else:
            raise AssertionError("expected validation failure for invalid decision enrichment payload")

        llm.generate_json.side_effect = [
            first_exc,
            DecisionEnrichmentArtifact.model_validate(
                {
                    "project_id": "proj",
                    "family": "decisions_and_nondecisions",
                    "enriched_primitives": [
                        {
                            "id": "dn_1",
                            "family": "decisions_and_nondecisions",
                            "actor_ids": ["actor_1"],
                            "decision_trigger": "Fresh pressure forces a move.",
                            "decision_question": "Should the court move now?",
                            "decision_mode": "decision",
                            "options_considered": ["advance", "delay"],
                            "next_result": "The move forces the next confrontation.",
                            "narration_hooks": _hooks(),
                        }
                    ],
                }
            ),
        ]
        agent = PrimitiveEnrichmentAgent(llm, max_retry_attempts=2)

        with patch("podcast_agent.agents.primitive_enrichment.time.sleep", return_value=None):
            result = agent.run(
                {
                    "project_id": "proj",
                    "family": "decisions_and_nondecisions",
                    "base_primitives": [{"id": "dn_1", "family": "decisions_and_nondecisions", "actor_ids": []}],
                    "evidence_by_primitive_id": {},
                }
            )

        assert isinstance(result, DecisionEnrichmentArtifact)
        first_kwargs = llm.generate_json.call_args_list[0].kwargs
        second_kwargs = llm.generate_json.call_args_list[1].kwargs
        assert "enrichment_feedback" not in first_kwargs["payload"]
        feedback = second_kwargs["payload"]["enrichment_feedback"]
        assert feedback["issue"] == "schema_validation_failed"
        assert feedback["family"] == "decisions_and_nondecisions"
        assert feedback["validation_errors"] == [
            {
                "path": "enriched_primitives.0.decision_query",
                "error_type": "extra_forbidden",
                "message": "Extra inputs are not permitted",
                "primitive_id": "dn_1",
            }
        ]

    def test_primitive_enrichment_transient_retry_does_not_add_feedback(self):
        llm = _mock_llm()
        llm.generate_json.side_effect = [
            TransientLLMError("timeout"),
            HumanCostEnrichmentArtifact.model_validate(
                {
                    "project_id": "proj",
                    "family": "human_costs",
                    "enriched_primitives": [
                        {
                            "id": "hc_1",
                            "family": "human_costs",
                            "actor_ids": [],
                            "affected_group": "camp followers",
                            "cost_type": "displacement",
                            "concrete_marker": "Families carry bedding into the road.",
                            "lived_consequence": "Households lose shelter and income.",
                            "who_saw_it": "plain to witnesses but politically deniable",
                            "narration_hooks": _hooks(),
                        }
                    ],
                }
            ),
        ]
        agent = PrimitiveEnrichmentAgent(llm, max_retry_attempts=2)

        with patch("podcast_agent.agents.primitive_enrichment.time.sleep", return_value=None):
            result = agent.run(
                {
                    "project_id": "proj",
                    "family": "human_costs",
                    "base_primitives": [{"id": "hc_1", "family": "human_costs", "actor_ids": []}],
                    "evidence_by_primitive_id": {},
                }
            )

        assert isinstance(result, HumanCostEnrichmentArtifact)
        first_kwargs = llm.generate_json.call_args_list[0].kwargs
        second_kwargs = llm.generate_json.call_args_list[1].kwargs
        assert "enrichment_feedback" not in first_kwargs["payload"]
        assert "enrichment_feedback" not in second_kwargs["payload"]

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
        assert "Convert the episode spine into 9-12 binding sections." in agent.instructions
        assert "The final section must use `purpose` = `closing`." in agent.instructions
        assert "approx_runtime_minutes` at or below 2.0" in agent.instructions
        assert "Do not build a second ending." in agent.instructions
        assert "Target 6-10 support-primitive placements across sections." in agent.instructions
        assert "Do not place all assigned support primitives by default." in agent.instructions
        assert "Ensure the sum of `sections[].approx_runtime_minutes` lands within the project's allowed episode runtime range." in agent.instructions
        assert "`priority_core_passage_ids`" in agent.instructions
        assert "`section_anchor`" in agent.instructions
        assert "`must_stage_beats`" in agent.instructions
        assert "are not scene cards, scene counts" in agent.instructions

    def test_episode_planning_agent_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_episode={"episode_number": 1, "episode_spine": {"listener_question": "Question?"}},
            architecture={"episode_number": 1},
            synthesis_map={"primitives_by_family": {"epochal_turns": []}},
            project_metadata={"theme": "War on terror"},
            available_passages=[{"passage_id": "p1"}],
            host_policy={"target_host_moves_per_episode": 7, "target_policy": "soft_target"},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            planning_feedback={"issue": "missing_spine_coverage"},
        )
        assert agent.schema_name == "episode_planning"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["host_policy"]["target_host_moves_per_episode"] == 7
        assert payload["planning_feedback"]["issue"] == "missing_spine_coverage"
        assert "`available_passages`" in agent.instructions
        assert "`estimated_duration_seconds`" in agent.instructions
        assert "PRIORITY RULES" in agent.instructions
        assert "Every scene card must be grounded in provided `passage_ids`." in agent.instructions
        assert "Target 27–36 scene cards" in agent.instructions
        assert "Build each section through accumulation." in agent.instructions
        assert "The final architecture `closing` section must expand to exactly one scene" in agent.instructions
        assert "It must keep `estimated_duration_seconds` ≤ 120." in agent.instructions
        assert "The framing should orient the listener without pre-explaining the episode." in agent.instructions
        assert "Treat `architecture.section_anchor` as the section-local opening handle." in agent.instructions
        assert "every item in `must_stage_beats`." in agent.instructions
        assert "should create curiosity in the same territory as" in agent.instructions
        assert "When a section has `priority_core_passage_ids`, prefer those passages" in agent.instructions
        assert "`host_policy`" in agent.instructions
        assert "host_move.placement" in agent.instructions
        assert "Do not force filler host moves" in agent.instructions
        assert "No first-person singular." in agent.instructions
        assert "`section_id`" in agent.instructions
        assert "`scene_function`" in agent.instructions
        assert "context_setup, actor_setup, action, shock, contestation, reaction," in agent.instructions
        assert "fallout, implication" in agent.instructions
        assert "scene, hinge, mechanism, turn, landing, callback, afterlife" in agent.instructions

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
            host_policy={"target_host_moves_per_episode": 7},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["host_policy"]["target_host_moves_per_episode"] == 7
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
        assert "Treat it as narrative scaffolding, not factual authority." in agent.instructions
        assert "Keep claims grounded in each card's `must_land_facts` and `passage_ids`." in agent.instructions
        assert "`host_policy`" in agent.instructions
        assert "no first-person singular" in agent.instructions
        assert "host_move.placement" in agent.instructions
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
        assert "Follow scene-role intent:" in agent.instructions
        assert "Follow scene-function intent:" in agent.instructions
        assert "Structural cards should stay concrete and brief." in agent.instructions
        assert "Realize a planned `host_move` as one distinct audible line or clause" in agent.instructions

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
            host_policy={"target_host_moves_per_episode": 7},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["host_policy"]["target_host_moves_per_episode"] == 7
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
        assert "Passages are evidence." in agent.instructions
        assert "Do not cite scaffolding, assert it as fact, or use it to fill evidence gaps." in agent.instructions
        assert "`host_policy`" in agent.instructions
        assert "no first-person singular" in agent.instructions
        assert "let it shape the scene at its planned" in agent.instructions
        assert "Optional `prior_window_continuity`" in agent.instructions
        assert "Treat it as reference-only guidance for handoff, pacing, and continuity." in agent.instructions
        assert "When `prior_window_continuity` is present, use it only to maintain local continuity across the split." in agent.instructions
        assert "`prior_window_continuity` is reference-only." in agent.instructions
        assert "Do not include a `citations` field" in agent.instructions
        assert "Populate `source_book_ids`" in agent.instructions
        assert "target ranges already encode narrative importance" in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "SCENE FUNCTIONS" in agent.instructions
        assert "Structural cards must stay concrete and brief." in agent.instructions
        assert "`action`: write an observable beat: named actors doing concrete things" in agent.instructions
        assert "Do not output standalone transitions." in agent.instructions
        assert "Realize a planned host move as one distinct audible line or clause" in agent.instructions
        assert "Let host-marked scenes feel slightly more authored, but not more analytical." in agent.instructions

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
            script={
                "framing": {"opening_image": "", "threat_or_unresolved_action": "", "opening_question": "", "handoff_scene_card_id": "scene_1"},
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "movement_goal": "continue",
                        "scene_card_ids": ["scene_1"],
                        "host_moves": [{"move_type": "callback", "placement": "close", "note": "Carry residue.", "max_sentences": 1}],
                        "text": "Section text",
                    }
                ],
            },
            max_words_per_segment=250,
            tts_provider="openai",
            host_policy={"target_host_moves_per_episode": 7},
        )
        assert agent.schema_name == "spoken_delivery"
        assert payload["max_words_per_segment"] == 250
        assert payload["host_policy"]["target_host_moves_per_episode"] == 7
        assert "You are the `oral_rewriter` stage" in agent.instructions
        assert "turn one already-written batch of episode prose into spoken narration" in agent.instructions
        assert "INPUT" in agent.instructions
        assert "`script.prose_sections[].host_moves` are host-guidance control signals." in agent.instructions
        assert "`host_policy`" in agent.instructions
        assert "TRANSFORMATION MANDATE" in agent.instructions
        assert "Be faithful to the content. Do not be faithful to the delivery mechanism." in agent.instructions
        assert "Do not draft from source sentences. Draft from extracted content moves." in agent.instructions
        assert "PLANNING WORKFLOW" in agent.instructions
        assert "factual and chronological fidelity" in agent.instructions
        assert "Extract the batch into content moves" in agent.instructions
        assert "treating the past as a physical place" in agent.instructions
        assert "Prioritize the weather of a scene. Replace abstract summaries with physical friction." in agent.instructions
        assert "Pivot from geopolitical forces to immediate pressure on individuals." in agent.instructions
        assert "Open from something concrete already present in the batch." in agent.instructions
        assert "PODCAST QUALITY" in agent.instructions
        assert "Do not explain a beat before staging it." in agent.instructions
        assert "Preserve staircase beats when the material earns them." in agent.instructions
        assert "If you must choose between a cleaner essay sentence and a sharper spoken sentence, choose the sharper spoken sentence." in agent.instructions
        assert "Slight expansion is allowed only when it restores audible force, a clearer referent, or a stronger landing." in agent.instructions
        assert "When the material turns coercive, humiliating, or irreversible, allow the prose to become barer and more percussive." in agent.instructions
        assert "Avoid AI-cliches and topic-announcing transitions." in agent.instructions
        assert "TTS AND SPEECH HINTS" in agent.instructions
        assert "Return only valid JSON matching `expected_schema` exactly" in agent.instructions
        assert "If `previous_spoken_tail` is present, continue rather than restart." in agent.instructions
        assert "Do not repeat it, paraphrase it, summarize it, or import facts from it" in agent.instructions
        assert "SELF-CHECK BEFORE RETURNING" in agent.instructions
        assert "Does `speech_hints` remain minimal and genuinely useful?" in agent.instructions
        assert "Did you preserve the batch's hardest lines instead of moderating them?" in agent.instructions
        assert "`script.prose_sections`" in agent.instructions
        assert "`script.framing`" in agent.instructions
        assert "upcoming_batches_summary" not in agent.instructions
        assert "Rewrite all of it into one continuous spoken passage" not in agent.instructions
        assert "speech_hints" in agent.instructions
        assert "section" in payload
        assert "previous_spoken_text" not in payload
        assert "previous_spoken_tail" not in payload
        assert "upcoming_batches_summary" not in payload
        assert "batch_index" not in payload
        assert "batch_count" not in payload

    def test_section_local_spoken_prompt_stays_trimmed(self):
        prompt = section_local_spoken_delivery_instructions()
        assert len(prompt.split()) <= 1810


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
                        "epochal_turns": [{"id": f"primitive_{idx}"} for idx in range(1, 8)]
                    }
                },
                project_metadata={"theme": "War on terror"},
                episode_count=7,
                recommended_episode_count_min=8,
                recommended_episode_count_max=12,
            )
        )
        assert result.recommended_episode_count == 7
        assert result.episodes[0].episode_spine.core_primitive_ids == [
            "primitive_1",
            "primitive_2",
            "primitive_3",
            "primitive_4",
            "primitive_5",
        ]
        assert list(result.episodes[0].episode_spine.support_primitive_roles.keys()) == [
            "primitive_6",
            "primitive_7",
            "primitive_008",
            "primitive_009",
            "primitive_010",
        ]

    def test_narrative_strategy_agent_run_uses_payload_min_when_no_override(self):
        agent = NarrativeStrategyAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                synthesis_map={
                    "primitives_by_family": {
                        "epochal_turns": [{"id": f"primitive_{idx}"} for idx in range(1, 8)]
                    }
                },
                project_metadata={"theme": "War on terror"},
                episode_count=None,
                recommended_episode_count_min=8,
                recommended_episode_count_max=12,
            )
        )
        assert result.recommended_episode_count == 8
