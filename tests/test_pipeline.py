"""Focused tests for active orchestrator helpers in the redesigned pipeline."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _cluster_batch_group_sizes,
    _compute_scene_word_count_targets,
    _build_scene_card_count_warnings,
    _build_scene_card_primitive_warnings,
    _build_passage_lookup,
    _estimate_duration_seconds_from_words,
    _flatten_synthesis_primitives,
    _resolve_synthesis_bm25_keep_fraction_by_passage,
    _split_sentences,
    _script_total_word_count,
    _trim_candidate_texts_by_bm25,
)
from podcast_agent.schemas.models import (
    BookRecord,
    ClusterPathOccurrence,
    EpisodeCandidateCluster,
    EpisodeScript,
    EpisodePlan,
    ExtractedPassage,
    FramingBlock,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProseSection,
    SceneCard,
    ScriptTransition,
    SpokenScript,
    SpokenSection,
    SpokenTransition,
    StrategyEpisode,
    SynthesisConsolidationResult,
    SynthesisPrimitivesArtifact,
    SynthesisMap,
    ThematicCorpus,
    ThematicAxis,
    ThematicProject,
    TurningPoint,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="Image",
        threat_or_unresolved_action="Threat",
        opening_question="Question",
        handoff_scene_card_id="scene_1",
    )


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def test_build_scene_card_count_warnings_for_under_target():
    warnings = _build_scene_card_count_warnings(
        scene_card_count=18,
        scene_card_target_min=25,
        scene_card_target_max=40,
    )
    assert len(warnings) == 1
    assert warnings[0].startswith("scene_card_count_below_target")


def test_build_scene_card_primitive_warnings_reports_density_and_unknown_ids():
    cards = [
        SceneCard(
            scene_id="scene_1",
            title="Scene 1",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p1"],
        ),
        SceneCard(
            scene_id="scene_2",
            title="Scene 2",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["tp_1", "tp_2", "tp_3"],
            passage_ids=["p2"],
        ),
    ]
    warnings = _build_scene_card_primitive_warnings(
        scene_cards=cards,
        primitive_pool_ids={"tp_1", "tp_2"},
        primitive_min=1,
        primitive_max=2,
    )
    assert any(warning.startswith("normal_scene_primitive_density_out_of_range") for warning in warnings)
    assert any(warning.startswith("scene_card_unknown_primitive_ids") for warning in warnings)


def test_script_total_word_count_counts_sections_and_transitions():
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="One two three",
            )
        ],
        transitions=[
            ScriptTransition(
                transition_id="transition_1",
                after_section_id="section_1",
                text="Four five",
            )
        ],
    )
    assert _script_total_word_count(script) == 5


def test_estimate_duration_seconds_from_words_handles_zero_rate():
    assert _estimate_duration_seconds_from_words(120, 120) == 60
    assert _estimate_duration_seconds_from_words(120, 0) == 0


def test_compute_scene_word_count_targets_uses_scene_durations():
    scenes = [
        SceneCard(
            scene_id="scene_1",
            title="Scene 1",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p1"],
            estimated_duration_seconds=30,
        ),
        SceneCard(
            scene_id="scene_2",
            title="Scene 2",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p2"],
            estimated_duration_seconds=90,
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=1000, words_per_minute=120.0)
    assert targets == {"scene_1": 60, "scene_2": 180}


def test_compute_scene_word_count_targets_falls_back_to_even_split_when_no_durations():
    scenes = [
        SceneCard(
            scene_id="scene_1",
            title="Scene 1",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p1"],
            estimated_duration_seconds=0,
        ),
        SceneCard(
            scene_id="scene_2",
            title="Scene 2",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p2"],
            estimated_duration_seconds=0,
        ),
        SceneCard(
            scene_id="scene_3",
            title="Scene 3",
            scene_role="consequence",
            dominant_cluster_occurrence_id="occ_3",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p3"],
            estimated_duration_seconds=0,
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=10, words_per_minute=120.0)
    assert targets == {"scene_1": 4, "scene_2": 3, "scene_3": 3}


@pytest.mark.parametrize(
    ("cluster_count", "expected"),
    [
        (0, []),
        (1, [1]),
        (2, [1, 1]),
        (3, [2, 1]),
        (4, [2, 2]),
        (5, [3, 2]),
        (6, [3, 3]),
    ],
)
def test_cluster_batch_group_sizes(cluster_count: int, expected: list[int]):
    assert _cluster_batch_group_sizes(cluster_count) == expected


def test_trim_candidate_texts_by_bm25_keeps_one_quarter_of_sentences():
    axis = ThematicAxis(
        axis_id="axis_1",
        name="Query",
        description="Axis description",
        theme_importance_score=0.7,
    )
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "One sentence. Two sentence. Three sentence. "
                "Four sentence. Five sentence. Six sentence."
            ),
        },
        {
            "passage_id": "p2",
            "book_id": "b1",
            "text": "Alpha sentence. Beta sentence.",
        },
    ]

    _trim_candidate_texts_by_bm25(axis, candidates, keep_fraction=0.25)

    assert len(_split_sentences(candidates[0]["text"])) == 2
    assert len(_split_sentences(candidates[1]["text"])) == 1


def test_resolve_synthesis_bm25_keep_fraction_by_passage_uses_relevance_tiers():
    passages = [
        ExtractedPassage(
            passage_id=f"p{idx}",
            book_id="b1",
            chunk_ids=[f"c{idx}"],
            text=f"Passage {idx}",
            axis_id="axis_1",
            relevance_score=1.0 - (0.01 * idx),
            quotability_score=0.5,
        )
        for idx in range(1, 11)
    ]

    keep_fraction_by_passage_id, tier_counts = _resolve_synthesis_bm25_keep_fraction_by_passage(passages)

    assert tier_counts == {
        "top_half_passages": 1,
        "middle_third_passages": 2,
        "rest_quarter_passages": 7,
    }
    assert keep_fraction_by_passage_id["p1"] == 0.5
    assert keep_fraction_by_passage_id["p2"] == pytest.approx(1 / 3)
    assert keep_fraction_by_passage_id["p3"] == pytest.approx(1 / 3)
    assert keep_fraction_by_passage_id["p4"] == 0.25


def test_trim_candidate_texts_by_bm25_supports_per_passage_keep_fractions():
    axis = ThematicAxis(
        axis_id="axis_1",
        name="Query",
        description="Axis description",
        theme_importance_score=0.7,
    )
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "One sentence. Two sentence. Three sentence. "
                "Four sentence. Five sentence. Six sentence."
            ),
        },
        {
            "passage_id": "p2",
            "book_id": "b1",
            "text": (
                "Alpha sentence. Beta sentence. Gamma sentence. "
                "Delta sentence. Epsilon sentence. Zeta sentence."
            ),
        },
    ]

    _trim_candidate_texts_by_bm25(
        axis,
        candidates,
        keep_fraction=0.25,
        keep_fraction_by_passage_id={"p1": 0.5, "p2": 0.25},
    )

    assert len(_split_sentences(candidates[0]["text"])) == 3
    assert len(_split_sentences(candidates[1]["text"])) == 2


def test_build_passage_lookup_and_flatten_primitives():
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Text",
                    axis_id="axis_1",
                )
            ]
        },
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        turning_points=[
            TurningPoint(
                id="tp_1",
                title="Turn",
                summary="Summary",
                core_passage_ids=["p1"],
            )
        ],
    )
    assert _build_passage_lookup(corpus)["p1"].book_id == "b1"
    assert _flatten_synthesis_primitives(synthesis_map)["tp_1"].title == "Turn"


def test_orchestrator_initializes_redesigned_agents(monkeypatch):
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

    assert orchestrator.synthesis_primitives_agent.schema_name == "synthesis_primitives"
    assert orchestrator.synthesis_consolidation_agent.schema_name == "synthesis_consolidation"
    assert not hasattr(orchestrator, "synthesis_mapping_agent")


def test_map_synthesis_caps_total_passages_and_keeps_cross_pair_priority(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_primitives_run(payload: dict):
        captured["payload"] = payload
        return SynthesisPrimitivesArtifact(project_id="proj")

    orchestrator.synthesis_primitives_agent.run = fake_primitives_run
    orchestrator.synthesis_consolidation_agent.run = lambda payload: SynthesisConsolidationResult(project_id="proj")

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        config=PipelineConfig(
            synthesis_total_passage_cap=4,
            synthesis_axis_pct=1.0,
            synthesis_axis_min=2,
            synthesis_axis_max=2,
        ),
    )
    axis_1 = ThematicAxis(axis_id="axis_1", name="Axis 1", description="A1", theme_importance_score=0.95)
    axis_2 = ThematicAxis(axis_id="axis_2", name="Axis 2", description="A2", theme_importance_score=0.45)
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[axis_1, axis_2],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(passage_id="p1", book_id="b1", chunk_ids=["c1"], text="P1", axis_id="axis_1", relevance_score=0.9, quotability_score=0.8),
                ExtractedPassage(passage_id="p2", book_id="b1", chunk_ids=["c2"], text="P2", axis_id="axis_1", relevance_score=0.7, quotability_score=0.7),
                ExtractedPassage(passage_id="p3", book_id="b1", chunk_ids=["c3"], text="P3", axis_id="axis_1", relevance_score=0.6, quotability_score=0.6),
            ],
            "axis_2": [
                ExtractedPassage(passage_id="p4", book_id="b1", chunk_ids=["c4"], text="P4", axis_id="axis_2", relevance_score=0.95, quotability_score=0.9),
                ExtractedPassage(passage_id="p5", book_id="b1", chunk_ids=["c5"], text="P5", axis_id="axis_2", relevance_score=0.85, quotability_score=0.8),
                ExtractedPassage(passage_id="p6", book_id="b1", chunk_ids=["c6"], text="P6", axis_id="axis_2", relevance_score=0.1, quotability_score=0.1),
            ],
        },
        cross_book_pairs=[
            PassagePair(
                passage_a_id="p1",
                passage_b_id="p6",
                relationship="contradicts",
                strength=0.9,
                axis_id="axis_2",
            )
        ],
    )

    asyncio.run(orchestrator._map_synthesis(project, corpus, tmp_path))

    payload = captured["payload"]
    passages = payload["passages_by_axis"]
    for axis_passages in passages.values():
        for item in axis_passages:
            assert set(item.keys()) == {"passage_id", "book_id", "text"}
    all_ids = [item["passage_id"] for axis_passages in passages.values() for item in axis_passages]
    assert len(all_ids) == 4
    assert "p1" in all_ids
    assert "p6" in all_ids


def test_write_episode_passes_full_text_to_writing_agent(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_writing_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "batch_id": payload["batch_id"],
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                    }
                ],
                "transitions": [],
                "window_map": [{"batch_id": payload["batch_id"], "section_ids": ["section_1"], "transition_ids": []}],
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "title": "Scene 1",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 300,
                }
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for writing.",
                    axis_id="axis_1",
                )
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    payload = captured["payload"]
    assert payload["passages"][0]["text"] == "Full text evidence for writing."


def test_write_episode_uses_single_batch_for_many_scene_cards(monkeypatch, tmp_path):
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
    captured_payloads: list[dict] = []

    def fake_writing_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "batch_id": payload["batch_id"],
                "prose_sections": [
                    {
                        "section_id": f"section_{payload['batch_id']}",
                        "scene_card_ids": payload["active_scene_card_ids"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                    }
                ],
                "transitions": [],
                "window_map": [{
                    "batch_id": payload["batch_id"],
                    "section_ids": [f"section_{payload['batch_id']}"],
                    "transition_ids": [],
                }],
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": f"scene_{idx}",
                    "title": f"Scene {idx}",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": f"occ_{idx}",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": [f"p{idx}"],
                    "estimated_duration_seconds": 300,
                }
                for idx in range(1, 27)
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id=f"p{idx}",
                    book_id="b1",
                    chunk_ids=[f"c{idx}"],
                    text=f"Trimmed text {idx}",
                    full_text=f"Full text evidence {idx}.",
                    axis_id="axis_1",
                )
                for idx in range(1, 27)
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    assert len(captured_payloads) == 2
    assert [payload["batch_id"] for payload in captured_payloads] == ["batch_1", "batch_2"]
    assert [len(payload["active_scene_card_ids"]) for payload in captured_payloads] == [13, 13]
    assert [len(payload["passages"]) for payload in captured_payloads] == [13, 13]
    assert [payload["batch_target_word_count_lower"] for payload in captured_payloads] == [7150, 7150]
    assert [payload["batch_target_word_count_higher"] for payload in captured_payloads] == [9100, 9100]

    for payload in captured_payloads:
        assert "previous_sections" not in payload
        assert "previous_transitions" not in payload
        payload_scene_cards = payload["plan"]["scene_cards"]
        assert [scene["scene_id"] for scene in payload_scene_cards] == payload["active_scene_card_ids"]
        assert payload["plan"]["framing"]["handoff_scene_card_id"] == payload["active_scene_card_ids"][0]
        assert all("estimated_duration_seconds" not in scene for scene in payload_scene_cards)
        assert all(scene["target_word_count_lower"] == 550 for scene in payload_scene_cards)
        assert all(scene["target_word_count_higher"] == 700 for scene in payload_scene_cards)


def test_write_episode_uses_no_citation_agent_when_skip_grounding(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fail_if_standard_agent_used(_payload: dict):
        raise AssertionError("standard writing agent should not be used when skip_grounding=True")

    def fake_no_citation_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.writing_agent_no_citations.response_model.model_validate(
            {
                "batch_id": payload["batch_id"],
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                        "source_book_ids": ["b1"],
                    }
                ],
                "transitions": [],
                "window_map": [{"batch_id": payload["batch_id"], "section_ids": ["section_1"], "transition_ids": []}],
            }
        )

    orchestrator.writing_agent.run = fail_if_standard_agent_used
    orchestrator.writing_agent_no_citations.run = fake_no_citation_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        config=PipelineConfig(skip_grounding=True),
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "title": "Scene 1",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 300,
                }
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for writing.",
                    axis_id="axis_1",
                )
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    script = asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    payload = captured["payload"]
    assert payload["skip_grounding"] is True
    assert "scene_word_count_targets" not in payload
    assert "previous_sections" not in payload
    assert "previous_transitions" not in payload
    assert payload["batch_target_word_count_lower"] == 550
    assert payload["batch_target_word_count_higher"] == 700
    scene = payload["plan"]["scene_cards"][0]
    assert scene["scene_id"] == "scene_1"
    assert "estimated_duration_seconds" not in scene
    assert scene["target_word_count_lower"] == 550
    assert scene["target_word_count_higher"] == 700
    assert script.prose_sections[0].citations == []


def test_validate_grounding_uses_full_text_lookup(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_grounding_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.grounding_agent.response_model.model_validate({"episode_number": 1})

    orchestrator.grounding_agent.run = fake_grounding_run
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="Draft.",
            )
        ],
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for grounding.",
                    axis_id="axis_1",
                )
            ]
        },
    )

    asyncio.run(orchestrator._validate_grounding(1, script, corpus, tmp_path, tmp_path))

    payload = captured["payload"]
    assert payload["cited_passages"]["p1"]["text"] == "Full text evidence for grounding."


def test_render_episode_audio_logs_warning_when_transitions_mismatch(monkeypatch, tmp_path):
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
    logged_events: list[tuple[str, dict]] = []
    original_log = orchestrator.run_logger.log

    def capture_log(event_type: str, **payload):
        logged_events.append((event_type, payload))
        original_log(event_type, **payload)

    orchestrator.run_logger.log = capture_log  # type: ignore[method-assign]

    spoken = SpokenScript(
        episode_number=4,
        title="Episode 4",
        framing=_framing(),
        sections=[
            SpokenSection(section_id="sec_01", text="One."),
            SpokenSection(section_id="sec_02", text="Two."),
            SpokenSection(section_id="sec_03", text="Three."),
        ],
        transitions=[
            SpokenTransition(transition_id="tr_01_02", text="Bridge one."),
        ],
    )

    config = PipelineConfig(skip_audio=True)
    asyncio.run(
        orchestrator._render_episode_audio(
            episode_number=4,
            spoken=spoken,
            config=config,
            project_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
            skip_audio=True,
        )
    )

    warning_events = [payload for event_type, payload in logged_events if event_type == "spoken_transition_mismatch_warning"]
    assert warning_events
    assert warning_events[0]["section_count"] == 3
    assert warning_events[0]["transition_count"] == 1
    assert warning_events[0]["expected_transition_count"] == 2
    assert (tmp_path / "episodes" / "4" / "render_manifest.json").exists()


def test_plan_series_parallelizes_episode_planning_with_configured_limit(monkeypatch, tmp_path):
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
    in_flight = 0
    max_in_flight = 0
    payloads_by_episode: dict[int, dict] = {}
    lock = threading.Lock()

    def fake_planning_run(payload: dict):
        nonlocal in_flight, max_in_flight
        episode_number = int(payload["episode"]["episode_number"])
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            payloads_by_episode[episode_number] = payload
        try:
            time.sleep(0.05)
            primary_occurrence_id = payload["episode"]["cluster_path"][0]["occurrence_id"]
            return orchestrator.episode_planning_agent.response_model.model_validate(
                {
                    "episode_number": episode_number,
                    "title": payload["episode"]["title"],
                    "driving_question": payload["episode"]["driving_question"],
                    "thematic_focus": payload["episode"]["thematic_focus"],
                    "arc_summary": payload["episode"]["arc_summary"],
                    "unresolved_questions": payload["episode"]["unresolved_questions"],
                    "framing": {
                        "opening_image": "Image",
                        "threat_or_unresolved_action": "Threat",
                        "opening_question": "Question",
                        "handoff_scene_card_id": "scene_1",
                    },
                    "scene_cards": [
                        {
                            "scene_id": "scene_1",
                            "title": "Scene 1",
                            "scene_role": "setup",
                            "dominant_cluster_occurrence_id": primary_occurrence_id,
                            "passage_ids": [],
                            "primitive_ids": [],
                        }
                    ],
                    "target_duration_minutes": 140.0,
                }
            )
        finally:
            with lock:
                in_flight -= 1

    orchestrator.episode_planning_agent.run = fake_planning_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        episode_count=3,
        config=PipelineConfig(episode_planning_concurrency=2),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="test",
        series_arc="test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                driving_question="Q1",
                arc_summary="A1",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_1",
                        cluster_id="ec_1",
                        usage="primary",
                    )
                ],
            ),
            StrategyEpisode(
                episode_number=2,
                title="Episode 2",
                driving_question="Q2",
                arc_summary="A2",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_2",
                        cluster_id="ec_2",
                        usage="primary",
                    )
                ],
            ),
            StrategyEpisode(
                episode_number=3,
                title="Episode 3",
                driving_question="Q3",
                arc_summary="A3",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_3",
                        cluster_id="ec_3",
                        usage="primary",
                    )
                ],
            ),
        ],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        episode_candidate_clusters=[
            EpisodeCandidateCluster(
                cluster_id="ec_1",
                title="C1",
                summary="S1",
                primary_member_id="tp_1",
                member_ids=["tp_1"],
                local_question="L1",
                local_payoff_shape="reveal",
            ),
            EpisodeCandidateCluster(
                cluster_id="ec_2",
                title="C2",
                summary="S2",
                primary_member_id="tp_2",
                member_ids=["tp_2"],
                local_question="L2",
                local_payoff_shape="reveal",
            ),
            EpisodeCandidateCluster(
                cluster_id="ec_3",
                title="C3",
                summary="S3",
                primary_member_id="tp_3",
                member_ids=["tp_3"],
                local_question="L3",
                local_payoff_shape="reveal",
            ),
        ],
        turning_points=[
            TurningPoint(id="tp_1", title="T1", summary="TS1", core_passage_ids=["p1"]),
            TurningPoint(id="tp_2", title="T2", summary="TS2", core_passage_ids=["p2"]),
            TurningPoint(id="tp_3", title="T3", summary="TS3", core_passage_ids=["p3"]),
        ],
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed planning text 1",
                    full_text="Full planning text 1",
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p2",
                    book_id="b1",
                    chunk_ids=["c2"],
                    text="Trimmed planning text 2",
                    full_text="",
                    chapter_ref="Chapter 2",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p3",
                    book_id="b1",
                    chunk_ids=["c3"],
                    text="Trimmed planning text 3",
                    full_text="Full planning text 3",
                    chapter_ref="Chapter 3",
                    axis_id="axis_1",
                ),
            ]
        },
    )

    plans = asyncio.run(
        orchestrator._plan_series(project, synthesis_map, strategy, corpus, tmp_path)
    )

    assert [plan.episode_number for plan in plans] == [1, 2, 3]
    assert 1 < max_in_flight <= 2
    assert sorted(payloads_by_episode) == [1, 2, 3]
    expected_passage_text_by_episode = {
        1: "Full planning text 1",
        2: "Trimmed planning text 2",
        3: "Full planning text 3",
    }
    for episode_number in [1, 2, 3]:
        synthesis_payload = payloads_by_episode[episode_number]["synthesis_map"]
        assert [
            cluster["cluster_id"] for cluster in synthesis_payload["episode_candidate_clusters"]
        ] == [f"ec_{episode_number}"]
        assert [item["id"] for item in synthesis_payload["turning_points"]] == [
            f"tp_{episode_number}"
        ]
        available_passages = payloads_by_episode[episode_number]["available_passages"]
        assert [passage["passage_id"] for passage in available_passages] == [f"p{episode_number}"]
        assert [passage["text"] for passage in available_passages] == [
            expected_passage_text_by_episode[episode_number]
        ]
