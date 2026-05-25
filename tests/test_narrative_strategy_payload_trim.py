"""Tests for narrative strategy runtime payload compaction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_narrative_strategy_scene_discovery_payload,
    _build_scene_discovery_synthesis_map_payload,
    _build_narrative_strategy_actor_metadata_payload,
    _build_narrative_strategy_project_metadata_payload,
    _build_narrative_strategy_synthesis_map_payload,
    _compact_narrative_strategy_runtime_payload,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    BookRecord,
    EpisodeSpine,
    EventPrimitive,
    NarrativeStrategy,
    NarrativeStrategyEnrichment,
    NarrativeStrategySkeleton,
    PipelineConfig,
    PrimitiveSalience,
    PrimitiveSubstrate,
    SeriesActorExplanationItem,
    SeriesNarratorProfile,
    SeriesExplanationItem,
    SceneDiscoveryArtifact,
    SynthesisPrimitivesArtifact,
    StrategyEpisode,
    ThematicProject,
)


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def test_compact_narrative_strategy_runtime_payload_elides_empty_values():
    payload = _compact_narrative_strategy_runtime_payload(
        {
            "drop_none": None,
            "drop_empty_string": "",
            "keep_zero": 0,
            "keep_zero_float": 0.0,
            "keep_false": False,
            "nested": {
                "drop_empty_list": [],
                "keep_text": "value",
            },
            "items": [
                {},
                {"drop": "", "keep": "x"},
                [],
                0,
                False,
            ],
        }
    )

    assert payload == {
        "keep_zero": 0,
        "keep_zero_float": 0.0,
        "keep_false": False,
        "nested": {"keep_text": "value"},
        "items": [{"keep": "x"}, 0, False],
    }


def test_build_narrative_strategy_payload_helpers_trim_fields():
    synthesis_map = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives=[
            EventPrimitive(
                id="et_1",
                substrate=PrimitiveSubstrate.EVENTS,
                title="Turn",
                core_passage_ids=["p1"],
                support_passage_ids=[],
                geography="Delhi",
                actor_ids=["actor_1", "actor_2"],
                salience=PrimitiveSalience(score=0.82, justification="High signal."),
                event_type="turning_point",
                what_happened="A decisive turn shifts the field.",
            )
        ],
        quality_score=0.6,
        quality_notes=[],
    )
    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
                description="Primary actor.",
                aliases=["One"],
                book_ids=["b1"],
                narrative_functions=["decision_maker"],
                goals_or_motivational_pressures=["Hold power"],
                constraints=["Weak treasury"],
                stakes=["Dynasty"],
                transformations=["Hardens"],
                uncertainty_notes="Sparse on motive.",
                narrative_importance_score=0.7,
            )
        ],
        relationships=[
            ActorRelationship(
                source_actor_id="actor_1",
                target_actor_id="actor_1",
                relationship_type="other",
                description="Self-conception matters.",
            )
        ],
        quality_notes=["verbose actor metadata"],
    )

    synthesis_payload = _build_narrative_strategy_synthesis_map_payload(synthesis_map)
    project_payload = _build_narrative_strategy_project_metadata_payload(project)
    actor_payload = _build_narrative_strategy_actor_metadata_payload(actor_metadata)

    primitive = synthesis_payload["primitives"][0]
    assert "affected_actor_ids" not in primitive
    assert "actor_tags" not in primitive
    assert "institution_tags" not in primitive
    assert "support_passage_ids" not in primitive
    assert "timeframe" not in primitive
    assert "axis_ids" not in primitive
    assert "candidate_readings" not in primitive
    assert primitive["geography"] == "Delhi"
    assert "sub_themes" not in project_payload
    assert actor_payload == {
        "actors": [
            {
                "actor_id": "actor_1",
                "display_name": "Actor One",
                "aliases": ["One"],
                "actor_type": "person",
                "description": "Primary actor.",
                "goals_or_motivational_pressures": ["Hold power"],
                "constraints": ["Weak treasury"],
                "stakes": ["Dynasty"],
                "transformations": ["Hardens"],
                "narrative_importance_score": 0.7,
            }
        ],
    }


def test_build_scene_discovery_payload_trims_heavy_primitive_fields():
    synthesis_map = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives=[
            EventPrimitive(
                id="et_1",
                substrate=PrimitiveSubstrate.EVENTS,
                title="Turn",
                core_passage_ids=["p1"],
                support_passage_ids=["p2"],
                geography="Delhi",
                actor_ids=["actor_1", "actor_2"],
                functions=["cost", "complication"],
                salience=PrimitiveSalience(score=0.82, justification="High signal."),
                event_type="turning_point",
                what_happened="A decisive turn shifts the field.",
                event_result="A political order weakens.",
                cost={
                    "who_paid": "Civilians",
                    "what_was_paid": "Security and livelihood",
                },
                complication={
                    "what_is_compromised": "Clean blame",
                    "why_no_clean_option": "Every path worsens something.",
                },
                narration_hooks={
                    "concrete_detail": "A sealed file on a desk.",
                    "host_lens": "The state narrows its options.",
                    "carry_forward": "Returns later as panic.",
                    "plain_gloss": "A turning point with real fallout.",
                    "why_it_matters": "It makes the later collapse legible.",
                    "best_use": "opening",
                    "natural_host_move": "orient",
                    "listener_confusion": "Why does this matter now?",
                    "authorial_move": "causal_compression",
                },
            )
        ],
        quality_score=0.6,
        quality_notes=[],
    )

    payload = _build_scene_discovery_synthesis_map_payload(synthesis_map)
    primitive = payload["primitives"][0]

    assert primitive["salience"] == {"score": 0.82}
    assert primitive["narration_hooks"] == {
        "plain_gloss": "A turning point with real fallout.",
        "why_it_matters": "It makes the later collapse legible.",
        "best_use": "opening",
        "natural_host_move": "orient",
    }
    assert primitive["cost"] == {
        "who_paid": "Civilians",
        "what_was_paid": "Security and livelihood",
    }
    assert primitive["complication"] == {
        "what_is_compromised": "Clean blame",
        "why_no_clean_option": "Every path worsens something.",
    }
    assert "concrete_detail" not in primitive["narration_hooks"]
    assert "host_lens" not in primitive["narration_hooks"]
    assert "carry_forward" not in primitive["narration_hooks"]
    assert "listener_confusion" not in primitive["narration_hooks"]
    assert "authorial_move" not in primitive["narration_hooks"]
    assert primitive["event_result"] == "A political order weakens."


def test_build_narrative_strategy_scene_discovery_payload_drops_passage_ids():
    scene_discovery = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "primitive_ids": ["p1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A decree lands.",
                    "scene_jobs": ["opening"],
                    "anchor_image": "A sheet of paper on a desk.",
                    "why_sceneable": "The action is visible.",
                    "quote_anchor": "The room goes quiet.",
                    "actor_ids": ["actor_1"],
                }
            ]
        }
    )
    payload = _build_narrative_strategy_scene_discovery_payload(scene_discovery)
    assert payload is not None
    candidate = payload["candidates"][0]
    assert candidate["candidate_id"] == "candidate_01"
    assert "passage_ids" not in candidate


def test_choose_narrative_strategy_uses_trimmed_runtime_payload(
    monkeypatch, tmp_path: Path
):
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: heuristic,
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: DummyTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    captured: dict[str, Any] = {}

    def fake_skeleton_run(payload: dict[str, Any]) -> NarrativeStrategySkeleton:
        captured["skeleton_payload"] = payload
        return NarrativeStrategySkeleton.model_validate(
            {
                "strategy_type": "chronological",
                "justification": "Because it fits.",
                "series_arc": "Arc",
                "recommended_episode_count": 1,
                "episodes": [
                    {
                        "episode_number": 1,
                        "title": "Episode 1",
                        "arc_summary": "Arc",
                        "episode_spine": {
                            "listener_question": "Question?",
                            "argument": "Claim",
                            "core_primitive_ids": [
                                "et_1",
                                "core_2",
                                "core_3",
                                "core_4",
                                "core_5",
                                "core_6",
                                "core_7",
                            ],
                            "support_primitive_roles": {
                                f"support_{idx}": "mechanism"
                                for idx in range(1, 8)
                            },
                        },
                        "negative_scope": {
                            "boundary": "Stay on the main pressure line.",
                            "excluded_topics": [],
                            "tempting_but_out": [],
                            "omission_logic": "Leave neighboring material out unless it advances the answer.",
                        },
                    }
                ],
            }
        )

    def fake_enrichment_run(payload: dict[str, Any]) -> NarrativeStrategyEnrichment:
        captured["enrichment_payload"] = payload
        return NarrativeStrategyEnrichment.model_validate(
            {
                "episodes": [
                    {
                        "episode_number": 1,
                        "promised_beats": [
                            {
                                "beat_id": "beat_1",
                                "label": "Opening beat",
                                "intended_job": "opening",
                                "source_candidate_ids": ["candidate_01"],
                                "source_primitive_ids": ["et_1"],
                                "why_load_bearing": "Concrete opening obligation.",
                            }
                        ],
                    }
                ]
            }
        )

    orchestrator.narrative_strategy_skeleton_agent.run = fake_skeleton_run
    orchestrator.narrative_strategy_enrichment_agent.run = fake_enrichment_run
    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    synthesis_map = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives=[
            EventPrimitive(
                id="et_1",
                substrate=PrimitiveSubstrate.EVENTS,
                title="Turn",
                core_passage_ids=["p1"],
                support_passage_ids=[],
                geography="Delhi",
                actor_ids=["actor_1", "actor_2"],
                salience=PrimitiveSalience(score=0.82, justification="High signal."),
                event_type="turning_point",
                what_happened="A decisive turn shifts the field.",
            )
        ],
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
                description="Primary actor.",
                aliases=["One"],
                book_ids=["b1"],
                narrative_functions=["decision_maker"],
                goals_or_motivational_pressures=["Hold power"],
                constraints=["Weak treasury"],
                stakes=["Dynasty"],
                transformations=["Hardens"],
                uncertainty_notes="Sparse on motive.",
                narrative_importance_score=0.7,
            )
        ],
        relationships=[
            ActorRelationship(
                source_actor_id="actor_1",
                target_actor_id="actor_1",
                relationship_type="other",
                description="Self-conception matters.",
            )
        ],
        quality_notes=["verbose actor metadata"],
    )
    scene_discovery = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "primitive_ids": ["et_1"],
                    "passage_ids": ["passage_1"],
                    "scene_sketch": "A decree lands.",
                    "scene_jobs": ["opening"],
                    "anchor_image": "A decree on a desk.",
                    "why_sceneable": "The act is visible.",
                }
            ]
        }
    )

    asyncio.run(
        orchestrator._choose_narrative_strategy(
            project,
            synthesis_map,
            tmp_path,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
        )
    )

    payload = captured["skeleton_payload"]
    assert payload["recommended_episode_count_min"] == 10
    assert payload["recommended_episode_count_max"] == 16
    primitive = payload["synthesis_map"]["primitives"][0]
    assert "affected_actor_ids" not in primitive
    assert "actor_tags" not in primitive
    assert "institution_tags" not in primitive
    assert "axis_ids" not in primitive
    assert "thematic_axes" not in payload
    assert "sub_themes" not in payload["project"]
    assert "book_ids" not in payload["actor_metadata"]["actors"][0]
    assert "narrative_functions" not in payload["actor_metadata"]["actors"][0]
    assert "relationships" not in payload["actor_metadata"]
    assert "quality_notes" not in payload["actor_metadata"]
    assert "passage_ids" not in payload["scene_discovery"]["candidates"][0]
    enrichment_payload = captured["enrichment_payload"]
    assert enrichment_payload["strategy_skeleton"]["episodes"][0]["episode_number"] == 1
    assert enrichment_payload["episode_scene_candidates"][0]["episode_number"] == 1


def test_narrative_strategy_accepts_series_explanation_registry_contract() -> None:
    strategy = NarrativeStrategy.model_validate(
        {
            "strategy_type": "chronological",
            "justification": "Fits the material.",
            "series_arc": "An arc.",
            "narrator_profile": SeriesNarratorProfile().model_dump(mode="json"),
            "series_explanation_registry": [
                SeriesExplanationItem(
                    item_id="registry_1",
                    label="taqlid",
                    aliases=["emulation"],
                    kind="term",
                    importance="foundational",
                    introduction_episode_number=1,
                    preferred_plain_gloss="follow a recognized jurist",
                ).model_dump(mode="json")
            ],
            "series_actor_explanation_registry": [
                SeriesActorExplanationItem(
                    actor_id="mohammad_mossadeq",
                    introduction_episode_number=1,
                    first_background_depth="appositive",
                    preferred_plain_gloss="the nationalist prime minister driving oil nationalization",
                    later_episode_policy="brief_reminder",
                ).model_dump(mode="json")
            ],
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode 1",
                    "arc_summary": "Arc",
                    "episode_spine": {
                        "listener_question": "Question?",
                        "argument": "Claim",
                        "core_primitive_ids": [
                            "core_1",
                            "core_2",
                            "core_3",
                            "core_4",
                            "core_5",
                            "core_6",
                            "core_7",
                        ],
                        "support_primitive_roles": {
                            f"support_{idx}": "mechanism" for idx in range(1, 6)
                        },
                    },
                    "authorial_contract": {
                        "introduce_explanation_item_ids": ["registry_1"],
                        "remind_explanation_item_ids": [],
                        "introduce_actor_ids": ["mohammad_mossadeq"],
                        "remind_actor_ids": [],
                        "callback_obligations": [
                            "Recall that clerical authority already had an operating structure."
                        ],
                    },
                }
            ],
        }
    )

    assert strategy.series_explanation_registry[0].item_id == "registry_1"
    assert (
        strategy.series_actor_explanation_registry[0].actor_id
        == "mohammad_mossadeq"
    )
    assert strategy.episodes[0].authorial_contract.introduce_explanation_item_ids == [
        "registry_1"
    ]
    assert strategy.episodes[0].authorial_contract.introduce_actor_ids == [
        "mohammad_mossadeq"
    ]
