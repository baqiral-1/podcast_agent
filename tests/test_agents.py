"""Unit tests for active agents in the redesigned pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic import APIStatusError as AnthropicAPIStatusError
import httpx
import pytest
from pydantic import ValidationError

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import (
    ChapterSummaryAgent,
    ChapterSummaryResponse,
)
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy_enrichment import (
    NarrativeStrategyEnrichmentAgent,
)
from podcast_agent.agents.narrative_strategy_skeleton import (
    NarrativeStrategySkeletonAgent,
)
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_function_tagging import PrimitiveFunctionTaggingAgent
from podcast_agent.agents.quality_judge import QualityJudgeAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.style_audit import StyleAuditAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.langchain.runnables import (
    RetryableGenerationError,
    TransientLLMError,
)
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import _build_host_policy_payload
from podcast_agent.prompts.instructions import (
    episode_architecture_instructions,
    episode_writing_instructions,
    episode_writing_no_citations_instructions,
    spoken_delivery_instructions,
)
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterAnalysis,
    ChapterInfo,
    PrimitiveFunctionTaggingOverlayArtifact,
    PrimitiveSubstrate,
    SeriesNarratorProfile,
    primitive_substrate_target_ranges_for_mode,
)


def _anthropic_overloaded_error(
    *,
    status_code: int = 529,
    message: str = "provider status failure",
) -> AnthropicAPIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    body = {
        "type": "error",
        "error": {
            "details": None,
            "type": "overloaded_error",
            "message": "Overloaded",
        },
    }
    return AnthropicAPIStatusError(message, response=response, body=body)


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
            axis_count_min=12,
            axis_count_max=20,
            books=[],
            book_summaries={},
        )
        assert agent.schema_name == "theme_decomposition"
        assert payload["sub_themes"] == ["state failure"]
        assert payload["axis_count_min"] == 12
        assert payload["axis_count_max"] == 20
        assert payload["actor_count_min"] == 10
        assert payload["actor_count_max"] == 40
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

    def test_theme_decomposition_agent_run_uses_payload_axis_count_min(self):
        agent = ThemeDecompositionAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                theme="War on terror",
                sub_themes=["state failure"],
                theme_elaboration="Trace the escalation.",
                axis_count_min=4,
                axis_count_max=6,
                books=[],
                book_summaries={},
            )
        )
        assert len(result.axes) == 4

    def test_theme_decomposition_agent_run_uses_payload_actor_count_range(self):
        llm = _mock_llm()
        llm.generate_json.return_value = MagicMock()
        agent = ThemeDecompositionAgent(llm)
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["state failure"],
            theme_elaboration="Trace the escalation.",
            axis_count_min=4,
            axis_count_max=6,
            actor_count_min=5,
            actor_count_max=12,
            books=[],
            book_summaries={},
        )

        agent.run(payload)

        assert llm.generate_json.call_count == 1
        instructions = llm.generate_json.call_args.kwargs["instructions"]
        assert "Produce between 5 and 12 actors" in instructions

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
            axis_count_min=12,
            axis_count_max=20,
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
                        themes_touched=[
                            "  theme one  ",
                            "theme two",
                            "theme three",
                            "theme four",
                        ],
                        major_actors=[
                            "actor one",
                            "actor two",
                            "actor three",
                            "actor four",
                        ],
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
            axis_count_min=12,
            axis_count_max=20,
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
        assert (
            "Score passages relative to the other candidates for this axis." in agent.instructions
        )
        assert "Do not tag everything as `exemplifies`." in agent.instructions
        assert "A pair should teach something a single passage cannot." in agent.instructions
        assert "Self-check before returning:" in agent.instructions
        assert "Every input passage appears exactly once in `passages`." in agent.instructions


class TestRedesignedAgents:
    def test_synthesis_primitives_agent_payload_and_instructions(self):
        agent = SynthesisPrimitivesAgent(_mock_llm())
        payload = agent.build_payload(
            project_id="proj",
            podcast_mode="minified",
            axes_summary=[{"axis_id": "axis_1"}],
            passages_by_axis={"axis_1": [{"book_id": "b1", "passages": [{"passage_id": "p1"}]}]},
            cross_book_pairs=[{"passage_a_id": "p1", "passage_b_id": "p2"}],
            book_metadata=[{"book_id": "b1"}],
            primitive_target_ranges=primitive_substrate_target_ranges_for_mode("minified"),
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            synthesis_feedback={"issue": "thin_grounding"},
        )
        assert agent.schema_name == "primitive_substrate_extraction"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["cross_book_pairs"][0]["passage_a_id"] == "p1"
        assert payload["passages_by_axis"]["axis_1"][0]["book_id"] == "b1"
        assert payload["primitive_target_ranges"] == primitive_substrate_target_ranges_for_mode(
            "minified"
        )
        assert payload["synthesis_feedback"]["issue"] == "thin_grounding"
        instructions = agent.build_instructions(payload)
        assert "primitive_substrate_extraction" in instructions
        assert "SUBSTRATE ONTOLOGY" in instructions
        assert "FIELDS PRESENT ON EVERY PRIMITIVE" in instructions
        assert "SUBSTRATE REQUIRED FIELD MATRIX" in instructions
        assert "SCHEMA-BOUND FIELD RULES" in instructions
        assert "SOFT COUNT GUIDANCE" in instructions
        assert "events (19–27)" in instructions

    def test_narrative_strategy_skeleton_agent_payload(self):
        agent = NarrativeStrategySkeletonAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"primitives": []},
            project_metadata={"theme": "War on terror"},
            scene_discovery={"candidates": [{"candidate_id": "c1"}]},
            episode_count=3,
            recommended_episode_count_min=10,
            recommended_episode_count_max=16,
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            strategy_skeleton_feedback={"issue": "weak_episode"},
        )
        instructions = agent.build_instructions(payload)
        assert agent.schema_name == "narrative_strategy_skeleton"
        assert payload["strategy_skeleton_feedback"]["issue"] == "weak_episode"
        assert "`promised_beats`" in instructions
        assert "Do NOT include:" in instructions
        assert "`narrative_agenda`" in instructions
        assert "`core_primitive_ids` must contain 6–10 primitives." in instructions
        assert "`negative_scope -> scope`" not in instructions
        assert "COMPACT TRANSPORT KEYS" not in instructions
        assert "prefer canonical field names" not in instructions
        assert "not a holding area for every later-useful primitive" in instructions
        assert "Do not let tail episodes become everything left over." in instructions
        assert "trim or demote primitives rather than adding more" in instructions
        assert "usually spend one directive slot on a human carrier" in instructions
        assert (
            "Sovereigns or state heads alone are not always a sufficient character spine"
            in instructions
        )

    def test_narrative_strategy_skeleton_agent_prepare_retry_payload_adds_targeted_feedback(
        self,
    ):
        agent = NarrativeStrategySkeletonAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"primitives": []},
            project_metadata={
                "theme": "War on terror",
                "episode_spine_core_primitive_target_min": 8,
                "episode_spine_core_primitive_target_max": 11,
                "episode_spine_support_primitive_target_min": 9,
                "episode_spine_support_primitive_target_max": 13,
                "episode_spine_recall_primitive_target_max": 3,
            },
            scene_discovery=None,
            episode_count=None,
            recommended_episode_count_min=8,
            recommended_episode_count_max=12,
        )
        raw_payload = {
            "episodes": [
                {
                    "episode_number": 1,
                    "scope": {"boundary": "Wrong key"},
                    "spine": {
                        "core_prims": ["e1"],
                        "support_roles": {"s1": "mechanism"},
                        "recall_prims": [],
                    },
                }
            ]
        }
        exc = RetryableGenerationError(
            "Schema validation failed for narrative_strategy_skeleton",
            data={"raw_payload": raw_payload},
        )
        exc.__cause__ = ValidationError.from_exception_data(
            "NarrativeStrategySkeleton",
            [
                {
                    "type": "value_error",
                    "loc": ("episodes", 0, "episode_spine"),
                    "msg": "Value error, core_primitive_ids must contain 2-11 primitive ids",
                    "input": raw_payload["episodes"][0]["spine"],
                    "ctx": {
                        "error": ValueError("core_primitive_ids must contain 2-11 primitive ids")
                    },
                },
                {
                    "type": "extra_forbidden",
                    "loc": ("episodes", 0, "scope"),
                    "msg": "Extra inputs are not permitted",
                    "input": raw_payload["episodes"][0]["scope"],
                },
            ],
        )

        next_payload = agent.prepare_retry_payload(payload, exc)

        feedback = next_payload["strategy_skeleton_feedback"]
        assert feedback["required_ranges"]["core_primitive_ids"] == "8-11"
        assert feedback["canonical_field_names"]["negative_scope"] == "negative_scope"
        assert feedback["episode_constraints_by_number"][0]["episode_number"] == 1
        assert (
            feedback["episode_constraints_by_number"][0]["direction_by_field"]["core_primitive_ids"]
            == "underfull"
        )
        assert (
            feedback["episode_constraints_by_number"][0]["direction_by_field"][
                "support_primitive_roles"
            ]
            == "underfull"
        )
        assert feedback["episode_constraints_by_number"][0]["recommended_action"] == "merge_episode"
        assert (
            "Use `negative_scope`, not `scope`."
            in feedback["episode_constraints_by_number"][0]["required_fix"]
        )
        assert (
            "forbidden_scope_alias" in feedback["episode_constraints_by_number"][0]["issue_types"]
        )
        assert (
            "core_primitive_count_underfull"
            in feedback["episode_constraints_by_number"][0]["issue_types"]
        )
        assert (
            "support_primitive_count_underfull"
            in feedback["episode_constraints_by_number"][0]["issue_types"]
        )
        assert feedback["issues"][0]["issue"] == "core_primitive_count_underfull"

    def test_narrative_strategy_skeleton_agent_prepare_retry_payload_marks_overfull_episodes(
        self,
    ):
        agent = NarrativeStrategySkeletonAgent(_mock_llm())
        payload = agent.build_payload(
            synthesis_map={"primitives": []},
            project_metadata={
                "theme": "Iranian Revolution",
                "episode_spine_core_primitive_target_min": 8,
                "episode_spine_core_primitive_target_max": 11,
                "episode_spine_support_primitive_target_min": 9,
                "episode_spine_support_primitive_target_max": 13,
                "episode_spine_recall_primitive_target_max": 3,
            },
            scene_discovery=None,
            episode_count=None,
            recommended_episode_count_min=8,
            recommended_episode_count_max=12,
        )
        raw_payload = {
            "episodes": [
                {
                    "episode_number": 8,
                    "spine": {
                        "core_prims": [f"e{i}" for i in range(12)],
                        "support_roles": {f"s{i}": "mechanism" for i in range(14)},
                        "recall_prims": ["r1", "r2", "r3"],
                    },
                }
            ]
        }
        exc = RetryableGenerationError(
            "Schema validation failed for narrative_strategy_skeleton",
            data={"raw_payload": raw_payload},
        )
        exc.__cause__ = ValidationError.from_exception_data(
            "NarrativeStrategySkeleton",
            [
                {
                    "type": "value_error",
                    "loc": ("episodes", 0, "episode_spine"),
                    "msg": "Value error, core_primitive_ids must contain 2-11 primitive ids",
                    "input": raw_payload["episodes"][0]["spine"],
                    "ctx": {
                        "error": ValueError("core_primitive_ids must contain 2-11 primitive ids")
                    },
                },
            ],
        )

        next_payload = agent.prepare_retry_payload(payload, exc)

        feedback = next_payload["strategy_skeleton_feedback"]
        assert (
            feedback["episode_constraints_by_number"][0]["direction_by_field"]["core_primitive_ids"]
            == "overfull"
        )
        assert (
            feedback["episode_constraints_by_number"][0]["direction_by_field"][
                "support_primitive_roles"
            ]
            == "overfull"
        )
        assert feedback["episode_constraints_by_number"][0]["recommended_action"] == "trim_core"
        assert (
            "Demote or remove non-thesis primitives from core."
            in feedback["episode_constraints_by_number"][0]["required_fix"]
        )
        assert (
            "core_primitive_count_overfull"
            in feedback["episode_constraints_by_number"][0]["issue_types"]
        )
        assert (
            "support_primitive_count_overfull"
            in feedback["episode_constraints_by_number"][0]["issue_types"]
        )
        assert (
            "If an episode is overfull, trim or demote primitives rather than adding more."
            in feedback["instruction"]
        )

    def test_narrative_strategy_enrichment_agent_payload(self):
        agent = NarrativeStrategyEnrichmentAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_skeleton={"episodes": [{"episode_number": 1}]},
            synthesis_map={"primitives": [{"id": "primitive_1"}]},
            project_metadata={"podcast_mode": "minified"},
            episode_scene_candidates=[
                {"episode_number": 1, "candidates": [{"candidate_id": "c1"}]}
            ],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            strategy_enrichment_feedback={"issue": "bad_registry"},
        )
        instructions = agent.build_instructions(payload)
        assert agent.schema_name == "narrative_strategy_enrichment"
        assert payload["strategy_enrichment_feedback"]["issue"] == "bad_registry"
        assert "The skeleton is binding." in instructions
        assert "`episode_scene_candidates`" in instructions
        assert "`series_explanation_registry` is top-level output" in instructions
        assert (
            "`assumption_moves.revise` must include both `statement` and `revised_statement`"
            in instructions
        )
        assert "- `kind`:" not in instructions
        assert (
            "`target_authorial_passages_per_episode` should usually land around 12–16."
            in instructions
        )

    def test_narrative_strategy_enrichment_agent_prepare_retry_payload_adds_targeted_feedback(
        self,
    ):
        agent = NarrativeStrategyEnrichmentAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_skeleton={"episodes": [{"episode_number": 1}]},
            synthesis_map={"primitives": [{"id": "primitive_1"}]},
            project_metadata={"podcast_mode": "full"},
            episode_scene_candidates=[],
        )
        raw_payload = {
            "episodes": [
                {
                    "episode_number": 1,
                    "narrative_agenda": {
                        "host": {
                            "assumption_moves": [
                                {
                                    "assumption_id": "a1",
                                    "action": "revise",
                                    "revised_statement": "The revised theory.",
                                }
                            ]
                        }
                    },
                    "promised_beats": [
                        {
                            "beat_id": "beat_1",
                            "label": "A structural promise",
                            "kind": "scene",
                            "intended_job": "opening",
                            "source_candidate_ids": ["candidate_1"],
                            "why_load_bearing": "It matters.",
                        }
                    ],
                }
            ]
        }
        exc = RetryableGenerationError(
            "Schema validation failed for narrative_strategy_enrichment",
            data={"raw_payload": raw_payload},
        )
        exc.__cause__ = ValidationError.from_exception_data(
            "NarrativeStrategyEnrichment",
            [
                {
                    "type": "value_error",
                    "loc": (
                        "episodes",
                        0,
                        "narrative_agenda",
                        "host",
                        "assumption_moves",
                        0,
                    ),
                    "msg": "Value error, host assumption introduce/revise moves must include statement",
                    "input": raw_payload["episodes"][0]["narrative_agenda"]["host"][
                        "assumption_moves"
                    ][0],
                    "ctx": {
                        "error": ValueError(
                            "host assumption introduce/revise moves must include statement"
                        )
                    },
                },
                {
                    "type": "extra_forbidden",
                    "loc": ("episodes", 0, "promised_beats", 0, "kind"),
                    "msg": "Extra inputs are not permitted",
                    "input": "scene",
                },
            ],
        )

        next_payload = agent.prepare_retry_payload(payload, exc)

        feedback = next_payload["strategy_enrichment_feedback"]
        assert feedback["issue"] == "schema_validation_failed"
        assert feedback["episode_constraints_by_number"][0]["episode_number"] == 1
        assert feedback["episode_constraints_by_number"][0]["issue_types"] == [
            "host_assumption_revise_missing_statement",
            "forbidden_promised_beat_kind_field",
        ]
        assert feedback["canonical_field_names"]["promised_beats[].kind"] == "remove this field"
        assert "Remove `kind`" in feedback["episode_constraints_by_number"][0]["required_fix"]
        assert "assumption_moves.revise" in feedback["instruction"]

    def test_primitive_function_tagging_agent_payload(self):
        agent = PrimitiveFunctionTaggingAgent(_mock_llm(), substrate=PrimitiveSubstrate.EVENTS)
        payload = agent.build_payload(
            project_id="proj",
            podcast_mode="full",
            base_primitives=[
                {
                    "id": "event_1",
                    "substrate": "events",
                    "title": "Turn",
                    "core_passage_ids": ["p1"],
                    "event_type": "crisis",
                    "what_happened": "A crisis lands publicly.",
                }
            ],
            passage_list=[{"passage_id": "p1", "text": "Text"}],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
        )
        assert agent.schema_name == "primitive_function_tagging_events"
        assert payload["substrate"] == "events"
        assert payload["passage_list"][0]["text"] == "Text"
        instructions = agent.build_instructions(payload)
        assert "`base_primitives`" in instructions
        assert "`passage_list`" in instructions
        assert "`core_passage_ids`" in instructions
        assert "`support_passage_ids`" in instructions
        assert "`function_feedback` (optional): retry feedback" in instructions
        assert "Goal:" in instructions
        assert "INPUT AND AUTHORITY" in instructions
        assert "WORKING METHOD" in instructions
        assert "FUNCTION-JUSTIFICATION DISCIPLINE" in instructions
        assert "SALIENCE SCORING METHOD" in instructions
        assert "NARRATION HOOKS" in instructions
        assert "OUTPUT CONTRACT" in instructions
        assert "SELF-CHECK BEFORE RETURNING" in instructions
        assert "narration_hooks" in instructions
        assert "overlays_by_id" in instructions
        # A set authorial_move must carry a spoken plain_gloss (see
        # _build_narration_hook_gloss_warnings).
        assert "whenever `authorial_move` is set" in instructions
        assert "without a usable spoken `plain_gloss` is not allowed" in instructions

    def test_primitive_function_tagging_retry_feedback_includes_invalid_hook_path(self):
        llm = _mock_llm()
        agent = PrimitiveFunctionTaggingAgent(
            llm,
            substrate=PrimitiveSubstrate.EVENTS,
            max_retry_attempts=2,
        )
        invalid_payload = {
            "project_id": "proj",
            "overlays_by_id": {
                "event_1": {
                    "functions": ["pivot"],
                    "pivot": {
                        "what_changed": "The event changes the field.",
                        "irreversibility": "med",
                    },
                    "salience": {
                        "score": 0.8,
                        "justification": "Load-bearing turn.",
                    },
                    "event_result": "The turn hardens the conflict.",
                    "narration_hooks": {**_hooks(), "authorial_move": "naming_note"},
                }
            },
        }

        try:
            PrimitiveFunctionTaggingOverlayArtifact.model_validate(invalid_payload)
        except Exception as validation_exc:
            try:
                raise RetryableGenerationError(
                    "Schema validation failed for primitive_function_tagging_events",
                    data={"raw_payload": invalid_payload},
                ) from validation_exc
            except RetryableGenerationError as retry_exc:
                first_exc = retry_exc
        else:
            raise AssertionError("expected validation failure for invalid tagging payload")

        llm.generate_json.side_effect = [
            first_exc,
            PrimitiveFunctionTaggingOverlayArtifact.model_validate(
                {
                    "project_id": "proj",
                    "overlays_by_id": {
                        "event_1": {
                            "functions": ["pivot"],
                            "pivot": {
                                "what_changed": "The event changes the field.",
                                "irreversibility": "med",
                            },
                            "salience": {
                                "score": 0.8,
                                "justification": "Load-bearing turn.",
                            },
                            "event_result": "The turn hardens the conflict.",
                            "narration_hooks": {
                                **_hooks(),
                                "authorial_move": "none",
                            },
                        }
                    },
                }
            ),
        ]

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(
                {
                    "project_id": "proj",
                    "podcast_mode": "full",
                    "substrate": "events",
                    "base_primitives": [
                        {
                            "id": "event_1",
                            "substrate": "events",
                            "title": "Turn",
                            "core_passage_ids": ["p1"],
                            "event_type": "crisis",
                            "what_happened": "A crisis lands publicly.",
                            "event_result": "The turn hardens the conflict.",
                        }
                    ],
                    "passage_list": [],
                }
            )

        assert "event_1" in result.overlays_by_id
        first_kwargs = llm.generate_json.call_args_list[0].kwargs
        second_kwargs = llm.generate_json.call_args_list[1].kwargs
        assert "function_feedback" not in first_kwargs["payload"]
        feedback = second_kwargs["payload"]["function_feedback"]
        assert feedback["issue"] == "schema_validation_failed"
        assert feedback["substrate"] == "events"
        assert feedback["validation_errors"] == [
            {
                "path": "overlays_by_id.event_1.narration_hooks.authorial_move",
                "error_type": "literal_error",
                "message": "Input should be 'none', 'quote_then_gloss', 'doctrinal_unpack', 'institutional_clarifier', 'causal_compression', 'comparative_aside' or 'verdict_landing'",
                "primitive_index": 0,
                "primitive_id": "event_1",
                "substrate": "events",
                "function": None,
                "issue": "schema_validation_failed",
                "required_fix": "Correct the schema validation error without changing valid unaffected overlays.",
            }
        ]

    def test_theme_decomposition_agent_retries_transient_error(self):
        llm = _mock_llm()
        expected = object()
        llm.generate_json.side_effect = [
            TransientLLMError("timeout"),
            expected,
        ]
        agent = ThemeDecompositionAgent(llm, max_retry_attempts=2)
        payload = agent.build_payload(
            theme="War on terror",
            sub_themes=["state failure"],
            theme_elaboration="Trace the escalation.",
            axis_count_min=12,
            axis_count_max=20,
            books=[],
            book_summaries={},
        )

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(payload)

        assert result is expected
        assert llm.generate_json.call_count == 2

    def test_writing_agent_retries_raw_transient_provider_error(self):
        llm = _mock_llm()
        llm.generate_json.side_effect = [
            _anthropic_overloaded_error(),
            WritingAgent.response_model.model_validate(
                {
                    "prose_sections": [
                        {
                            "section_id": "section_1",
                            "scene_card_ids": ["scene_1"],
                            "movement_goal": "discover",
                            "text": "Draft text.",
                            "source_book_ids": ["book_1"],
                        }
                    ]
                }
            ),
        ]
        agent = WritingAgent(llm, max_retry_attempts=2)
        payload = agent.build_payload(
            episode_number=1,
            strategy_episode={},
            architecture={"sections": [{"section_id": "section_1"}]},
            episode_plan={"scene_cards": [{"scene_id": "scene_1", "section_id": "section_1"}]},
            passages=[],
            book_metadata=[],
        )

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(payload)

        assert result.prose_sections[0].scene_card_ids == ["scene_1"]
        assert llm.generate_json.call_count == 2
        assert llm.generate_json.call_args_list[0].kwargs["attempt"] == 1
        assert llm.generate_json.call_args_list[1].kwargs["attempt"] == 2

    def test_spoken_delivery_agent_retries_raw_transient_provider_error(self):
        llm = _mock_llm()
        llm.generate_json.side_effect = [
            _anthropic_overloaded_error(),
            SpokenDeliveryAgent.response_model.model_validate(
                {
                    "sections": [
                        {
                            "section_id": "section_1",
                            "segments": [
                                {
                                    "segment_id": "section_1_seg1",
                                    "text": "Spoken delivery draft.",
                                    "speaker_role": "primary",
                                    "tonal_register": "neutral",
                                }
                            ],
                            "tonal_register": "neutral",
                        }
                    ]
                }
            ),
        ]
        agent = SpokenDeliveryAgent(llm, max_retry_attempts=2)
        payload = agent.build_payload(
            episode_number=1,
            script={
                "framing": {
                    "opening_image": "",
                    "threat_or_unresolved_action": "",
                    "opening_question": "",
                    "handoff_scene_card_id": "scene_1",
                },
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "continue",
                        "text": "Section text.",
                    }
                ],
            },
            max_words_per_segment=250,
            tts_provider="openai",
        )

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(payload)

        assert result.sections[0].section_id == "section_1"
        assert llm.generate_json.call_count == 2
        assert llm.generate_json.call_args_list[0].kwargs["attempt"] == 1
        assert llm.generate_json.call_args_list[1].kwargs["attempt"] == 2

    def test_episode_architecture_agent_payload(self):
        agent = EpisodeArchitectureAgent(_mock_llm())
        payload = agent.build_payload(
            episode={"episode_number": 1, "title": "Episode 1"},
            synthesis_map={"primitives": []},
            project_metadata={"theme": "War on terror"},
            core_passages=[{"passage_id": "p1"}],
            support_passages=[{"passage_id": "p2"}],
            episode_scenes=[{"candidate_id": "candidate_1"}],
            series_explanation_registry=[
                {
                    "item_id": "registry_1",
                    "label": "taqlid",
                    "kind": "term",
                    "importance": "foundational",
                    "introduction_episode_number": 1,
                    "preferred_plain_gloss": "follow a recognized jurist",
                }
            ],
            series_actor_explanation_registry=[
                {
                    "actor_id": "kermit_roosevelt",
                    "introduction_episode_number": 2,
                    "first_background_depth": "appositive",
                    "preferred_plain_gloss": "the CIA field officer helping run the coup operation",
                    "later_episode_policy": "brief_reminder",
                }
            ],
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            architecture_feedback={"issue": "missing_turn"},
        )
        assert agent.schema_name == "episode_architecture"
        instructions = agent.build_instructions(payload)
        assert payload["core_passages"][0]["passage_id"] == "p1"
        assert payload["support_passages"][0]["passage_id"] == "p2"
        assert payload["episode_scenes"][0]["candidate_id"] == "candidate_1"
        assert payload["series_explanation_registry"][0]["item_id"] == "registry_1"
        assert payload["series_actor_explanation_registry"][0]["actor_id"] == "kermit_roosevelt"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["architecture_feedback"]["issue"] == "missing_turn"
        assert "Convert the episode spine into 12–18 binding sections." in instructions
        assert (
            "The final section must use `purpose` = `closing` (aligned with `stage = close`)."
            in instructions
        )
        assert "approx_runtime_minutes` at or below 2.0" in instructions
        assert "Do not build a second ending." in instructions
        assert "Target 8-12 support-primitive placements across sections." in instructions
        assert "Do not place all assigned support primitives by default." in instructions
        assert "RUNTIME AND DENSITY BUDGET" in instructions
        assert "`priority_core_passage_ids`" in instructions
        assert "`support_passages`" in instructions
        assert "`episode_scenes`" in instructions
        assert "`section_anchor`" in instructions
        assert "`must_stage_beats`" in instructions
        assert "`series_explanation_registry`" in instructions
        assert "`series_actor_explanation_registry`" in instructions
        assert "`term_explanations`" in instructions
        assert "`actor_explanations`" in instructions
        assert "`question_moves`" in instructions
        assert "`memory_thread_moves`" in instructions
        assert "`host_mystery_moves`" in instructions
        assert "`host_assumption_moves`" in instructions
        assert "`host_theory_moves`" in instructions
        assert "24–32 total `authorial_passages`" in instructions
        assert "are not scene cards, scene counts" in instructions
        assert (
            "The first `must_stage_beats` item should usually open from the section" in instructions
        )
        assert "`section_progression`" in instructions
        assert "Exactly one section must use `section_progression.stage = answer`." in instructions
        assert "`closure_mode`" not in instructions
        assert "last stage allowed to mutate season state" in instructions
        assert "Do not leave state changes implicit inside `must_stage_beats`." in instructions
        assert (
            "Use `series_actor_explanation_registry`, `introduce_actor_ids`, and `remind_actor_ids`"
            in instructions
        )
        assert (
            "Treat those actor-registry fields as routing metadata only, not as copy-ready prose for `actor_explanations`."
            in instructions
        )
        assert "Attach `source_primitive_ids`," in instructions
        assert "`source_passage_ids`, `intro_facts`, `role_label`, and `why_now`" in instructions

    def test_episode_architecture_agent_builds_minified_instructions_from_payload(self):
        agent = EpisodeArchitectureAgent(_mock_llm())
        payload = agent.build_payload(
            episode={"episode_number": 1, "title": "Episode 1"},
            synthesis_map={"primitives": []},
            project_metadata={
                "podcast_mode": "minified",
                "architecture_section_target_min": 7,
                "architecture_section_target_max": 9,
            },
            core_passages=[{"passage_id": "p1"}],
            support_passages=[],
        )

        instructions = agent.build_instructions(payload)

        assert "Convert the episode spine into 7–9 binding sections." in instructions
        assert (
            "Most minified episodes should carry 12–16 total `authorial_passages`." in instructions
        )
        assert "Dense minified sections may use 2–5 `authorial_passages`" in instructions

    def test_episode_planning_agent_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_episode={
                "episode_number": 1,
                "episode_spine": {"listener_question": "Question?"},
            },
            architecture={"episode_number": 1},
            synthesis_map={"primitives": []},
            project_metadata={"theme": "War on terror"},
            scene_job_budget=None,
            available_passages=[{"passage_id": "p1"}],
            host_policy={
                "target_full_phase_scene_coverage_target": 0.8,
            },
            continuity_contract_pre={"recap_items": [{"item_id": "carry_1"}]},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            planning_feedback={"issue": "missing_spine_coverage"},
        )
        assert agent.schema_name == "episode_planning"
        instructions = agent.build_instructions(payload)
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["host_policy"]["target_full_phase_scene_coverage_target"] == 0.8
        assert payload["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
        assert payload["planning_feedback"]["issue"] == "missing_spine_coverage"
        assert "`available_passages`" in instructions
        assert "`estimated_duration_seconds`" in instructions
        assert "PRIORITY RULES" in instructions
        assert "Every scene card must be grounded in provided `passage_ids`." in instructions
        assert "scene cards for this episode." in instructions
        assert "Build each section through accumulation." in instructions
        assert (
            "The final architecture `closing` section must expand to exactly one scene"
            in instructions
        )
        assert "It must keep `estimated_duration_seconds` ≤ 120." in instructions
        assert (
            "The framing should orient the listener without pre-explaining the episode."
            in instructions
        )
        assert (
            "Treat `architecture.section_anchor` as the section-local opening handle,"
            in instructions
        )
        assert "`section_sonic_plan`" in instructions
        assert (
            "`section_sonic_plan.obligation` may only be `required` or `preferred`." in instructions
        )
        assert "scene-local derived realization" in instructions
        assert "should not repeat that anchor verbatim" in instructions
        assert "`must_stage_beats`, `section_progression`" in instructions
        assert "`closure_mode`" not in instructions
        assert "should create curiosity in the same territory as" in instructions
        assert (
            "When a section has `priority_core_passage_ids`, prefer those passages" in instructions
        )
        assert "When a section has `actor_explanations`" in instructions
        assert "`explanation_stage = introduce` or `reminder`." in instructions
        assert "`continuity_contract_pre` are read-only" in instructions
        assert "`continuity_contract_pre`" in instructions
        assert "Episode 1, set this to null" in instructions
        assert "may not invent new" in instructions
        assert "memory-thread, or host-state progression not already in strategy" in instructions
        assert "Carry the section plan's `background_depth`, `role_label`," in instructions
        assert "Do not add copied registry glosses or lifted prose when placing the" in instructions
        assert "Copy the section plan's `background_depth`, `role_label`," not in instructions
        assert "`host_policy`" in instructions
        assert "Every scene card carries a `host_moves` object with" in instructions
        assert "`host_moves` must be non-empty on structural scenes" in instructions
        assert "Default to one populated phase with one cue." in instructions
        assert "Ordinary `build` cards may leave all buckets empty." in instructions
        assert (
            "Omit optional fields entirely instead of returning blank strings or empty arrays."
            in instructions
        )
        assert "COMPACT TRANSPORT KEYS" not in instructions
        assert "`audible_detail -> audio`" not in instructions
        assert "Treat `must_land_facts` as the card's factual spine." in instructions
        assert "Every `note` must contain both:" in instructions
        assert "Write notes anchor-first." in instructions
        assert "Move-type-specific note rules:" in instructions
        assert "Phase-specific note rules:" in instructions
        assert "Weak vs. strong note examples:" in instructions
        assert (
            "Use Bazargan's office and the hidden ministries to show that the Council is already governing behind the cabinet."
            in instructions
        )
        assert "State the through-line." in instructions
        assert "`i` for taste, candid uncertainty, or comparison" in instructions
        assert (
            "Use `i` / `we` / `you` when brief, scene-rooted, and earning their keep"
            in instructions
        )
        assert "`section_id`" in instructions
        assert "`scene_function`" in instructions
        assert (
            "The answer card must live inside the section whose `section_progression.stage`"
            in instructions
        )
        assert "After-pressure content" in instructions
        assert "`afterpressure`-stage sections" in instructions
        assert "If `planning_feedback.issue = answer_scene_wrong_section`" in instructions
        assert "residue_scene_wrong_section" not in instructions
        assert "context_setup, actor_setup, action, shock, contestation, reaction," in instructions
        assert (
            "Ordinary sections should usually contain at most one `implication` card."
            in instructions
        )
        assert (
            'When the real job is "what changed because of this," prefer `fallout`.' in instructions
        )
        assert (
            "Sections that open in `context_setup` or `actor_setup` should usually pick up a concrete event, confrontation, or consequence beat inside the same section."
            in instructions
        )
        assert "Target 36–42 scene cards for this episode." in agent.instructions

    def test_episode_planning_agent_build_llm_payload_keeps_canonical_keys(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = {
            "episode_number": 1,
            "framing": {"opening_image": "A crowded gate."},
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "estimated_duration_seconds": 45,
                }
            ],
        }

        llm_payload = agent.build_llm_payload(payload)

        assert llm_payload == payload
        assert "scene_cards" in llm_payload
        assert "scenes" not in llm_payload
        assert llm_payload["scene_cards"][0]["estimated_duration_seconds"] == 45
        assert "dur" not in llm_payload["scene_cards"][0]

    def test_episode_planning_agent_builds_minified_instructions_from_payload(self):
        agent = EpisodePlanningAgent(_mock_llm())
        payload = agent.build_payload(
            strategy_episode={"episode_number": 1},
            architecture={"episode_number": 1},
            synthesis_map={"primitives": []},
            project_metadata={
                "scene_card_target_min": 21,
                "scene_card_target_max": 26,
            },
            scene_job_budget=None,
            available_passages=[{"passage_id": "p1"}],
        )

        instructions = agent.build_instructions(payload)

        assert "Target 21–26 scene cards for this episode." in instructions
        assert "fallout, implication" in agent.instructions
        assert "scene, hinge, mechanism, turn, landing, callback, afterlife" in agent.instructions
        assert "`term_explanations`" in agent.instructions
        assert "Every `note` must contain both:" in agent.instructions
        assert "Move-type-specific note rules:" in agent.instructions
        assert "Phase-specific note rules:" in agent.instructions
        assert "Use its `allowed_moves` as binding" in agent.instructions
        assert (
            "`clarify` after complexity, `contrast` to kill a false reading" in agent.instructions
        )
        assert (
            "Avoid section shapes that are effectively `setup -> implication -> implication`"
            in agent.instructions
        )

    def test_style_audit_agent_payload(self):
        agent = StyleAuditAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            title="Episode 1",
            sections=[
                {
                    "section_id": "section_1",
                    "text": "Section text.",
                    "term_explanations": [],
                }
            ],
            host_policy={"target_full_phase_scene_coverage_target": 0.75},
            continuity_contract_pre={"recap_items": [{"item_id": "carry_1"}]},
            continuity_contract_post={"must_leave_live": [{"item_id": "carry_2"}]},
            series_explanation_registry=[
                {
                    "item_id": "registry_1",
                    "label": "taqlid",
                    "kind": "term",
                    "importance": "foundational",
                    "introduction_episode_number": 1,
                    "preferred_plain_gloss": "follow a recognized jurist",
                }
            ],
        )
        assert agent.schema_name == "style_audit"
        assert payload["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
        assert payload["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
        assert payload["series_explanation_registry"][0]["item_id"] == "registry_1"
        assert payload["host_policy"]["target_full_phase_scene_coverage_target"] == 0.75
        assert "optional `series_explanation_registry`" in agent.instructions
        assert "optional `continuity_contract_pre`" in agent.instructions
        assert "optional `continuity_contract_post`" in agent.instructions
        assert "spoken_style_contract = anti_academic_oral" in agent.instructions
        assert "`term_explanations`" in agent.instructions
        assert "`actor_explanations`" in agent.instructions
        assert 'Visible production-frame phrasing such as "This series..."' in agent.instructions

    def test_build_host_policy_payload_includes_spoken_style_contract(self):
        payload = _build_host_policy_payload(SeriesNarratorProfile())

        assert payload["spoken_style_contract"] == "anti_academic_oral"
        assert payload["authorial_policy"]["comparative_aside_tolerance"] == "high"
        assert payload["allowed_moves"][-3:] == [
            "uncertainty",
            "revision",
            "surprise",
        ]

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
            host_policy={"target_full_phase_scene_coverage_target": 0.8},
            continuity_contract_pre={"recap_items": [{"item_id": "carry_1"}]},
            continuity_contract_post={"must_leave_live": [{"item_id": "carry_2"}]},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["host_policy"]["target_full_phase_scene_coverage_target"] == 0.8
        assert payload["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
        assert payload["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
        assert payload["prior_window_continuity"]["completed_scene_count"] == 1
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 120
        assert payload["episode_target_word_count_higher"] == 180
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["architecture"]["episode_number"] == 1
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert "Draft all `plan.scene_cards` in order." in agent.instructions
        assert (
            "`strategy_episode.episode_spine.core_primitive_ids`" in agent.instructions
            and "episode's load-bearing material" in agent.instructions
        )
        assert (
            "Use support and recall primitives only in service of those core primitives."
            in agent.instructions
        )
        assert "next-episode teaser copy" not in agent.instructions
        assert "`plan.framing.preview` is rendered separately" not in agent.instructions
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "`passages[].text`" in agent.instructions
        assert agent.instructions.count("Optional `actor_metadata`") == 1
        assert "Treat it as narrative scaffolding, not factual authority." in agent.instructions
        assert "`must_land_facts`" in agent.instructions
        assert "`passage_ids`" in agent.instructions
        assert "`host_policy`" in agent.instructions
        assert "use `I`, `we`" in agent.instructions
        assert "`you` freely" in agent.instructions
        assert "avoid filler" in agent.instructions
        assert "self-performance" in agent.instructions
        assert "Write this to be heard, not admired on the page." in agent.instructions
        assert "The target is not historical prose with some personality." in agent.instructions
        assert "Read each scene's phase buckets in order" in agent.instructions
        assert "Optional `continuity_contract_pre`" in agent.instructions
        assert "Optional `continuity_contract_post`" in agent.instructions
        assert "Optional `prior_window_continuity`" in agent.instructions
        assert "reference-only guidance for handoff, pacing, and continuity" in agent.instructions
        assert (
            "maintain local continuity across the split" in agent.instructions
            and "not factual authority" in agent.instructions
        )
        assert "target ranges already encode narrative importance" in agent.instructions
        assert (
            "Return one prose section per contiguous section window in the input plan."
            in agent.instructions
        )
        assert "Return one output item per input section" in agent.instructions
        assert "`architecture.sections[].must_stage_beats`" in agent.instructions
        assert "section-level obligations" in agent.instructions
        assert "`section_progression`" in agent.instructions
        assert "`inherited_pressure`" in agent.instructions
        assert "`section_sonic_plan`" in agent.instructions
        assert "Spend `section_sonic_plan.opening_anchor` in the first 1-2" in agent.instructions
        assert "Do not reproduce `section_sonic_plan.opening_anchor` verbatim" in agent.instructions
        assert "`term_explanations`" in agent.instructions
        assert "`actor_explanations`" in agent.instructions
        assert "In planned `authorial_passages`, you may quote then gloss" in agent.instructions
        assert "For `term_explanations.stage = define`" in agent.instructions
        assert "For `term_explanations.stage = reminder`" in agent.instructions
        assert "For `actors[].explanation_stage = introduce`" in agent.instructions
        assert "actor_explanation_realizations" in agent.instructions
        assert (
            "Build actor introductions from `role_label`, `source_passage_ids`"
            in agent.instructions
        )
        assert "actor_metadata` when present, and the" in agent.instructions
        assert "Do not use self-referential announcer lines in body prose" in agent.instructions
        assert "Target 8-12 prose sections for the episode" not in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`audible_detail`" in agent.instructions
        assert "SCENE ROLE SEMANTICS" in agent.instructions
        assert "SCENE JOB SEMANTICS" in agent.instructions
        assert "`action`: stage a materially consequential move" in agent.instructions
        assert (
            "`build`: carry most accumulation, setup, mechanism, contest, reaction, and consequence work."
            in agent.instructions
        )
        assert "Do not write standalone transition paragraphs" in agent.instructions
        assert "Structural cards should stay concrete and brief." in agent.instructions
        assert "Planned `host_moves` should shape the scene's narration" in agent.instructions
        assert "translate each host target into concrete scene" in agent.instructions.lower()
        assert "Do not surface control words unless the" in agent.instructions
        assert (
            "For compact output, you may omit `section_id`, `scene_card_ids`," in agent.instructions
        )
        assert "Omit empty `source_book_ids`." in agent.instructions

    def test_writing_response_allows_teaser_line(self):
        response = WritingAgent(_mock_llm()).response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "The scene lands.\n\nNext time: another story begins.",
                    }
                ]
            }
        )

        assert response.prose_sections[0].scene_card_ids == ["scene_1"]

    def test_writing_response_allows_compact_section_metadata(self):
        response = WritingAgent(_mock_llm()).response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "text": "The scene lands.",
                    }
                ]
            }
        )

        assert response.prose_sections[0].section_id == ""
        assert response.prose_sections[0].scene_card_ids == []
        assert response.prose_sections[0].movement_goal == ""

    def test_writing_response_allows_ordinary_next_time_phrase(self):
        response = WritingAgent(_mock_llm()).response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "But the next time you hear the official story, remember the archive.",
                    }
                ]
            }
        )

        assert response.prose_sections[0].scene_card_ids == ["scene_1"]

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
            host_policy={"target_full_phase_scene_coverage_target": 0.8},
            continuity_contract_pre={"recap_items": [{"item_id": "carry_1"}]},
            continuity_contract_post={"must_leave_live": [{"item_id": "carry_2"}]},
            actor_metadata={"actors": [{"actor_id": "actor_1"}]},
            prior_window_continuity={"completed_scene_count": 1},
        )
        assert agent.schema_name == "episode_writing"
        assert payload["actor_metadata"]["actors"][0]["actor_id"] == "actor_1"
        assert payload["host_policy"]["target_full_phase_scene_coverage_target"] == 0.8
        assert payload["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
        assert payload["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
        assert payload["prior_window_continuity"]["completed_scene_count"] == 1
        assert payload["skip_grounding"] is True
        assert payload["episode_target_word_count_lower"] == 140
        assert payload["episode_target_word_count_higher"] == 220
        assert payload["strategy_episode"]["episode_number"] == 1
        assert payload["architecture"]["episode_number"] == 1
        assert "scene_word_count_targets" not in payload
        assert "previous_sections" not in payload
        assert (
            "`strategy_episode.episode_spine.core_primitive_ids` are the episode's load-bearing material"
            in agent.instructions
        )
        assert "`target_word_count_lower`" in agent.instructions
        assert "`target_word_count_higher`" in agent.instructions
        assert "`episode_target_word_count_lower`" in agent.instructions
        assert "`episode_target_word_count_higher`" in agent.instructions
        assert "Draft the full episode" in agent.instructions
        assert "next-episode teaser copy" not in agent.instructions
        assert "`plan.framing.preview` is rendered separately" not in agent.instructions
        assert "Target total narration for this call within" in agent.instructions
        assert (
            "Return one prose section per contiguous section window in the input plan."
            in agent.instructions
        )
        assert "Return one output item per input section" in agent.instructions
        assert "Aim to deliver the episode in 8-12 prose sections" not in agent.instructions
        assert "Optional `actor_metadata`" in agent.instructions
        assert "Passages are evidence." in agent.instructions
        assert (
            "Do not cite scaffolding, assert it as fact, or use it to fill evidence gaps."
            in agent.instructions
        )
        assert "`host_policy`" in agent.instructions
        assert "use `I`, `we`" in agent.instructions
        assert "`you` freely" in agent.instructions
        assert "avoid filler" in agent.instructions
        assert "self-performance" in agent.instructions
        assert "Read the scene's `host_moves` phase buckets in order" in agent.instructions
        assert "Optional `continuity_contract_pre`" in agent.instructions
        assert "Optional `continuity_contract_post`" in agent.instructions
        assert "Optional `prior_window_continuity`" in agent.instructions
        assert (
            "`prior_window_continuity` is reference-only." in agent.instructions
            or "reference-only." in agent.instructions
        )
        assert (
            "maintain local continuity across the split" in agent.instructions
            or "cannot override the current window" in agent.instructions in agent.instructions
        )
        assert "`prior_window_continuity` is reference-only." in agent.instructions
        assert "Do not include a `citations` field" in agent.instructions
        assert "concrete scene leverage" in agent.instructions
        assert "no leaked host-target control phrasing" in agent.instructions
        assert "omit the field when empty rather than guessing" in agent.instructions
        assert (
            "For compact output, you may omit `section_id`, `scene_card_ids`, and"
            in agent.instructions
        )
        assert "Target total narration for this call within" in agent.instructions
        assert "`entry_image`" in agent.instructions
        assert "`audible_detail`" in agent.instructions
        assert "`section_sonic_plan`" in agent.instructions
        assert (
            "`section_sonic_plan.obligation` may only be `required` or `preferred`."
            in agent.instructions
        )
        assert (
            "Treat scene-level `audible_detail` as a local derived realization"
            in agent.instructions
        )
        assert "SCENE ROLES AND JOBS" in agent.instructions
        assert "Structural cards must stay concrete and brief." in agent.instructions
        assert "SCENE ROLE SEMANTICS" in agent.instructions
        assert "SCENE JOB SEMANTICS" in agent.instructions
        assert "`action`: stage a materially consequential move" in agent.instructions
        assert (
            "`build`: carry most accumulation, setup, mechanism, contest, reaction, and consequence work."
            in agent.instructions
        )
        assert "Do not expose scaffolding" in agent.instructions
        assert "no meta-transitions" in agent.instructions
        assert "let `surface_mode` and `address_mode` decide" in agent.instructions
        assert "Distinct host lines are allowed" in agent.instructions
        assert "not the default" in agent.instructions
        assert "Planned `authorial_passages` may be more explanatory" in agent.instructions
        assert "For `term_explanations.stage = define`" in agent.instructions
        assert "For `term_explanations.stage = reminder`" in agent.instructions
        assert "For `actors[].explanation_stage = introduce`" in agent.instructions
        assert "Do not use self-referential announcer lines in body prose" in agent.instructions
        assert "spoken historical narration" in agent.instructions
        assert "spoken_style_contract = anti_academic_oral" in agent.instructions
        assert "Host-line archetypes are welcome when earned" in agent.instructions
        assert "Write this to be heard, not admired on the page." in agent.instructions
        assert "Do not become more oral by getting much shorter." in agent.instructions

    def test_writing_no_citations_response_allows_teaser_line(self):
        response = WritingAgentNoCitations(_mock_llm()).response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "The scene lands.\n\nIn the next episode, another story begins.",
                        "source_book_ids": ["b1"],
                    }
                ]
            }
        )

        assert response.prose_sections[0].scene_card_ids == ["scene_1"]

    def test_quality_judge_agent_payload(self):
        agent = QualityJudgeAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            title="Test Episode",
            framing={
                "opening_image": "x",
                "threat_or_unresolved_action": "y",
                "opening_question": "z",
                "handoff_scene_card_id": "sc01",
            },
            prose_sections=[{"section_id": "s1", "text": "hello"}],
            architecture_summary=[
                {
                    "section_id": "s1",
                    "purpose": "opening",
                    "stage": "setup",
                    "is_dense": False,
                    "approx_runtime_minutes": 5.0,
                    "must_stage_beats": [],
                }
            ],
            excerpt_staging=[],
            rubric_thresholds={"criterion_floor_for_in_place_fixes": 70},
            style_audit_lint_flags={"tic_counts": {}, "tic_locations": {}, "by_section": {}},
        )
        assert agent.schema_name == "quality_judge"
        assert payload["episode_number"] == 1
        assert payload["thresholds"]["criterion_floor_for_in_place_fixes"] == 70

    def test_spoken_delivery_agent_payload(self):
        agent = SpokenDeliveryAgent(_mock_llm())
        payload = agent.build_payload(
            episode_number=1,
            script={
                "framing": {
                    "opening_image": "",
                    "threat_or_unresolved_action": "",
                    "opening_question": "",
                    "handoff_scene_card_id": "scene_1",
                },
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "movement_goal": "continue",
                        "scene_card_ids": ["scene_1"],
                        "text": "Section text",
                    }
                ],
            },
            max_words_per_segment=250,
            tts_provider="openai",
            tts_provider_capabilities={
                "provider": "openai",
                "supports_per_segment_instructions": False,
                "voice_catalog": ["fable", "onyx"],
            },
            quotability_marks=[{"section_id": "section_1", "excerpt_id": "x1"}],
            host_policy={"target_full_phase_scene_coverage_target": 0.8},
            continuity_contract_pre={"recap_items": [{"item_id": "carry_1"}]},
            continuity_contract_post={"must_leave_live": [{"item_id": "carry_2"}]},
            actor_voice_catalog={"actor_male_1": "onyx"},
        )
        # Change 4 contract: agent schema, payload shape, prompt content
        assert agent.schema_name == "spoken_delivery"
        assert payload["max_words_per_segment"] == 250
        assert payload["host_policy"]["target_full_phase_scene_coverage_target"] == 0.8
        assert payload["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
        assert payload["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
        assert payload["tts_provider_capabilities"]["supports_per_segment_instructions"] is False
        assert payload["quotability_marks"][0]["excerpt_id"] == "x1"
        assert payload["actor_voice_catalog"]["actor_male_1"] == "onyx"
        # Oral rewriter prompt mentions its core mandates
        assert "You are the `oral_rewriter` stage" in agent.instructions
        assert "ORAL REWRITING MANDATE" in agent.instructions
        assert "Restructure paragraph order" in agent.instructions
        assert "em-dash" in agent.instructions
        assert "register" in agent.instructions
        assert "actor voice" in agent.instructions.lower()
        assert "Preservation contract" in agent.instructions

    def test_spoken_delivery_prompt_stays_trimmed(self):
        prompt = spoken_delivery_instructions()
        # The new oral rewriter prompt is naturally longer than the old
        # compliance pass; allow up to ~2500 words.
        assert len(prompt.split()) <= 2500

    def test_episode_architecture_prompt_includes_field_contract(self):
        prompt = episode_architecture_instructions()
        assert "`authorial_passages.mode` carries the explanatory job and must be one of:" in prompt
        assert "`quote_then_gloss`" in prompt
        assert "`doctrinal_unpack`" in prompt
        assert "`institutional_clarifier`" in prompt
        assert "`causal_compression`" in prompt
        assert "`comparative_aside`" in prompt
        assert "`verdict_landing`" in prompt
        assert "`authorial_passages.placement` carries explanatory placement inside the" in prompt
        assert "section and must be one of: `open`, `mid`, `close`;" in prompt
        assert "may not use `close`" in prompt
        assert "Treat `comparative_aside` as comparison-with-return" in prompt
        assert "Prefer `placement = mid` for `comparative_aside`" in prompt
        assert "prefer them in your JSON output when possible" not in prompt
        assert "COMPACT TRANSPORT KEYS" not in prompt
        assert "`answer_section_id -> answer_section`" not in prompt
        assert "`residue_section_id -> residue_section`" not in prompt
        assert "`promised_beat_decisions -> promised_decisions`" not in prompt
        assert "`episode_spine -> spine`" not in prompt
        assert "`major_turn_section_id -> major_turn`" not in prompt
        assert "`grounding_actor_candidates` (optional)" not in prompt
        assert (
            "Prefer the `human_thread` members as the recurring human carrier across sections"
            in prompt
        )
        assert "reserve `episode.actor_arc_directives` for non-thread supporting actors" in prompt
        assert "`thread_binding` (required when the episode has a `human_thread`" in prompt
        assert "Adjacent explanatory sections should usually differ in `open_mode`" in prompt
        assert "`open_mode` sets the rhetorical shape of the section's opening" in prompt
        assert (
            "Sections built mostly from `mechanisms`, `conditions`, or `readings` should usually tie those abstractions to an event, act, artifact, attached excerpt, or recurring human pressure thread"
            in prompt
        )
        assert (
            "Sections carry `section_sonic_plan` whenever their primary excerpt is audible"
            in prompt
        )
        assert "`section_sonic_plan.obligation` must be exactly" in prompt
        assert "`required` or `preferred`." in prompt

    @pytest.mark.parametrize(
        "prompt_builder",
        [episode_writing_instructions, episode_writing_no_citations_instructions],
    )
    def test_writing_prompts_include_host_stance_guidance(self, prompt_builder):
        prompt = prompt_builder()
        assert "Treat `address_mode = we` and `address_mode = i` as stance signals" in prompt
        assert "Let the scene begin inside the world." in prompt
        assert (
            "Treat `spoken_style_contract = anti_academic_oral` as the default narrator mode."
            in prompt
        )
        assert (
            "For `comparative_aside`, prefer: scene fact -> carried comparison -> explicit snap-back."
            in prompt
        )
        assert "Write this to be heard, not admired on the page." in prompt
        assert "forceful host explaining history out loud." in prompt
        assert "claim_certainty = probable" in prompt
        assert "scene or factual pressure" in prompt
        assert "plain-English interpretation" in prompt
        assert "Avoid companion-tour phrasing" in prompt
        assert "If a first-person clause adds no real insight, comparison, surprise," in prompt
        assert "Prefer one inhabited clause of judgment or comparison" in prompt
        assert "Do not become more oral by getting much shorter." in prompt
        assert (
            "`implication`: land earned interpretation after evidence or consequence has already become legible."
            in prompt
        )
        assert "Scene jobs are structural jobs, not licenses for abstraction." in prompt
        assert (
            "When the real job is materially staged disagreement or rival reading, prefer `contestation`."
            in prompt
        )
        assert "Do not restart consecutive scenes with the same explanatory frame" in prompt


class TestHeuristicClient:
    def test_synthesis_primitives_agent_run_returns_valid_model(self):
        agent = SynthesisPrimitivesAgent(HeuristicLLMClient())
        result = agent.run(
            agent.build_payload(
                project_id="proj",
                podcast_mode="full",
                axes_summary=[{"axis_id": "axis_1"}],
                passages_by_axis={
                    "axis_1": [
                        {
                            "book_id": "b1",
                            "passages": [{"passage_id": "p1"}, {"passage_id": "p2"}],
                        }
                    ]
                },
                cross_book_pairs=[],
                book_metadata=[{"book_id": "b1"}],
            )
        )
        assert result.project_id == "proj"
        assert result.primitives[0].id == "e1"
        assert result.primitives[0].substrate.value == "events"

    def test_synthesis_primitives_agent_run_retries_on_transient_error(self):
        llm = _mock_llm()
        expected = SynthesisPrimitivesAgent(HeuristicLLMClient()).response_model.model_validate(
            {
                "project_id": "proj",
                "events": [
                    {
                        "title": "Recovered event",
                        "core_passage_ids": ["p1"],
                        "event_type": "crisis",
                        "what_happened": "A concrete event is returned after retry.",
                    }
                ],
            }
        )
        llm.generate_json.side_effect = [
            TransientLLMError("Connection error."),
            expected,
        ]
        agent = SynthesisPrimitivesAgent(llm, max_retry_attempts=2)
        payload = agent.build_payload(
            project_id="proj",
            podcast_mode="full",
            axes_summary=[{"axis_id": "axis_1"}],
            passages_by_axis={"axis_1": []},
            cross_book_pairs=[],
            book_metadata=[],
        )

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(payload)

        assert result.project_id == expected.project_id
        assert result.primitives[0].id == "e1"

    def test_synthesis_primitives_agent_run_retries_on_overloaded_provider_error(self):
        llm = _mock_llm()
        expected = SynthesisPrimitivesAgent(HeuristicLLMClient()).response_model.model_validate(
            {
                "project_id": "proj",
                "events": [
                    {
                        "title": "Recovered event",
                        "core_passage_ids": ["p1"],
                        "event_type": "crisis",
                        "what_happened": "A concrete event is returned after retry.",
                    }
                ],
            }
        )
        llm.generate_json.side_effect = [
            _anthropic_overloaded_error(),
            expected,
        ]
        agent = SynthesisPrimitivesAgent(llm, max_retry_attempts=2)
        payload = agent.build_payload(
            project_id="proj",
            podcast_mode="full",
            axes_summary=[{"axis_id": "axis_1"}],
            passages_by_axis={"axis_1": []},
            cross_book_pairs=[],
            book_metadata=[],
        )

        with patch("podcast_agent.agents.base.time.sleep", return_value=None):
            result = agent.run(payload)

        assert result.project_id == expected.project_id
        assert result.primitives[0].id == "e1"
        assert llm.generate_json.call_count == 2
