"""Focused tests for active orchestrator helpers in the redesigned pipeline."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_passage_lookup,
    _estimate_duration_seconds_from_words,
    _flatten_synthesis_primitives,
    _scene_cards_to_batches,
    _split_sentences,
    _script_total_word_count,
    _trim_candidate_texts_by_bm25,
)
from podcast_agent.schemas.models import (
    BookRecord,
    EpisodeScript,
    EpisodePlan,
    ExtractedPassage,
    FramingBlock,
    PassagePair,
    PipelineConfig,
    ProseSection,
    SceneCard,
    ScriptTransition,
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


def test_scene_cards_to_batches_splits_into_two_contiguous_windows():
    cards = [
        SceneCard(
            scene_id=f"scene_{idx}",
            title=f"Scene {idx}",
            scene_role="setup",
            dominant_cluster_occurrence_id=f"occ_{idx}",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            passage_ids=[f"p{idx}"],
        )
        for idx in range(1, 5)
    ]
    batches = _scene_cards_to_batches(cards)
    assert [[card.scene_id for card in batch] for batch in batches] == [
        ["scene_1", "scene_2"],
        ["scene_3", "scene_4"],
    ]


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


def test_trim_candidate_texts_by_bm25_keeps_one_third_of_sentences():
    axis = ThematicAxis(
        axis_id="axis_1",
        name="Query",
        description="Axis description",
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

    _trim_candidate_texts_by_bm25(axis, candidates)

    assert len(_split_sentences(candidates[0]["text"])) == 2
    assert len(_split_sentences(candidates[1]["text"])) == 1


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
    assert orchestrator.style_audit_agent.schema_name == "style_audit"
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
    orchestrator.synthesis_consolidation_agent.run = lambda payload: SynthesisMap(project_id="proj")

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
    axis_1 = ThematicAxis(axis_id="axis_1", name="Axis 1", description="A1")
    axis_2 = ThematicAxis(axis_id="axis_2", name="Axis 2", description="A2")
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
