"""Tests for narrative strategy runtime payload compaction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_narrative_strategy_actor_metadata_payload,
    _build_narrative_strategy_project_metadata_payload,
    _build_narrative_strategy_synthesis_map_payload,
    _build_narrative_strategy_thematic_axes_payload,
    _compact_narrative_strategy_runtime_payload,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    BookRecord,
    EpisodeSpine,
    ExtractedPassage,
    NarrativeStrategy,
    PipelineConfig,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitive,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
    VerdictMode,
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
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                SynthesisPrimitive(
                    id="et_1",
                    title="Turn",
                    summary="A decisive turn.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    support_passage_ids=[],
                    timeframe=None,
                    geography="Delhi",
                    primary_actor_ids=["actor_1"],
                    affected_actor_ids=["actor_2"],
                    actor_ids=["actor_1", "actor_2"],
                    actor_tags=["tag"],
                    institution_tags=["institution"],
                    unresolved_actor_tags=[],
                    narrative_importance_score=0.82,
                    candidate_readings=[],
                )
            ]
        },
        quality_score=0.6,
        quality_notes=[],
    )
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description.",
                theme_importance_score=0.9,
                guiding_questions=["What changed?"],
                relevance_by_book={"b1": 1.0},
                keywords=["state", "court"],
            )
        ],
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
    axis_payload = _build_narrative_strategy_thematic_axes_payload(corpus)
    project_payload = _build_narrative_strategy_project_metadata_payload(project)
    actor_payload = _build_narrative_strategy_actor_metadata_payload(actor_metadata)

    primitive = synthesis_payload["primitives_by_family"]["epochal_turns"][0]
    assert "affected_actor_ids" not in primitive
    assert "actor_tags" not in primitive
    assert "institution_tags" not in primitive
    assert "support_passage_ids" not in primitive
    assert "timeframe" not in primitive
    assert "candidate_readings" not in primitive
    assert primitive["geography"] == "Delhi"

    assert axis_payload == [
        {
            "axis_id": "axis_1",
            "name": "Axis",
            "description": "Axis description.",
            "theme_importance_score": 0.9,
            "guiding_questions": ["What changed?"],
        }
    ]
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


def test_choose_narrative_strategy_uses_trimmed_runtime_payload(monkeypatch, tmp_path: Path):
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
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

    def fake_run(payload: dict[str, Any]) -> NarrativeStrategy:
        captured["payload"] = payload
        return NarrativeStrategy(
            strategy_type="chronological",
            justification="Because it fits.",
            series_arc="Arc",
            episodes=[
                StrategyEpisode(
                    episode_number=1,
                    title="Episode 1",
                    driving_question="Question?",
                    arc_summary="Arc",
                    episode_spine=EpisodeSpine(
                        listener_question="Question?",
                        working_claim="Claim",
                        target_end_state="End state",
                        verdict_mode=VerdictMode.CONSTRAIN,
                        primary_counterposition="Counter",
                        core_primitive_ids=["et_1"],
                    ),
                )
            ],
        )

    orchestrator.narrative_strategy_agent.run = fake_run
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
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description.",
                theme_importance_score=0.9,
                guiding_questions=["What changed?"],
                relevance_by_book={"b1": 1.0},
                keywords=["state", "court"],
            )
        ],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="text",
                    axis_id="axis_1",
                )
            ]
        },
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                SynthesisPrimitive(
                    id="et_1",
                    title="Turn",
                    summary="A decisive turn.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    geography="Delhi",
                    primary_actor_ids=["actor_1"],
                    affected_actor_ids=["actor_2"],
                    actor_ids=["actor_1", "actor_2"],
                    actor_tags=["tag"],
                    institution_tags=["institution"],
                    narrative_importance_score=0.82,
                )
            ]
        },
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

    asyncio.run(
        orchestrator._choose_narrative_strategy(
            project,
            synthesis_map,
            corpus,
            tmp_path,
            actor_metadata,
        )
    )

    payload = captured["payload"]
    primitive = payload["synthesis_map"]["primitives_by_family"]["epochal_turns"][0]
    assert "affected_actor_ids" not in primitive
    assert "actor_tags" not in primitive
    assert "institution_tags" not in primitive
    assert "relevance_by_book" not in payload["thematic_axes"][0]
    assert "keywords" not in payload["thematic_axes"][0]
    assert payload["thematic_axes"][0]["guiding_questions"] == ["What changed?"]
    assert "sub_themes" not in payload["project"]
    assert "book_ids" not in payload["actor_metadata"]["actors"][0]
    assert "narrative_functions" not in payload["actor_metadata"]["actors"][0]
    assert "relationships" not in payload["actor_metadata"]
    assert "quality_notes" not in payload["actor_metadata"]
