"""Integration tests for the pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _save_json,
    _load_json,
    _trim_candidate_texts_by_bm25,
    _select_synthesis_passages,
    _select_top_passages_for_synthesis,
)
from podcast_agent.retrieval.vector_store import RetrievalHit
from podcast_agent.agents.passage_extraction import PassageExtractionResponse, PassageExtractionScore
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterInfo,
    ClaimAssessment,
    EpisodeBeat,
    EpisodeFraming,
    EpisodePlan,
    EpisodeScript,
    ExtractedPassage,
    GroundingReport,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProjectStatus,
    ScriptSegment,
    SpokenScript,
    SpokenSegment,
    SynthesisInsight,
    SynthesisMap,
    SynthesisTag,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
    InsightType,
)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


class TestArtifactPersistence:
    def test_save_and_load_json(self, tmp_path):
        data = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "sub" / "test.json"
        _save_json(path, data)
        loaded = _load_json(path)
        assert loaded == data

    def test_save_pydantic_model(self, tmp_path):
        book = BookRecord(
            book_id="b1", title="Test", author="A",
            source_path="/test.txt", source_type="txt",
        )
        path = tmp_path / "book.json"
        _save_json(path, book)
        loaded = _load_json(path)
        assert loaded["book_id"] == "b1"

    def test_load_missing_returns_none(self, tmp_path):
        assert _load_json(tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# Orchestrator construction
# ---------------------------------------------------------------------------


class TestOrchestratorConstruction:
    def test_creates_all_agents(self):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
        )
        orch = PipelineOrchestrator(settings)
        assert orch.structuring_agent is not None
        assert orch.theme_decomposition_agent is not None
        assert orch.passage_extraction_agent is not None
        assert orch.synthesis_mapping_agent is not None
        assert orch.narrative_strategy_agent is not None
        assert orch.series_planning_agent is not None
        assert orch.writing_agent is not None
        assert orch.source_weaving_agent is not None
        assert orch.grounding_agent is not None
        assert orch.repair_agent is not None
        assert orch.spoken_delivery_agent is not None
        assert orch.framing_agent is not None


# ---------------------------------------------------------------------------
# Phase 1 tests
# ---------------------------------------------------------------------------


class TestBookIngestion:
    def test_minimum_two_books_required(self, tmp_path):
        """Pipeline should fail if fewer than 2 books succeed."""
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        # Only 1 source file, and it doesn't exist
        with pytest.raises(RuntimeError, match="Minimum 2 required"):
            asyncio.run(orch.run_multi_book_podcast(
                source_paths=["/nonexistent/book1.txt"],
                theme="test",
                episode_count=2,
            ))

    def test_bm25_trims_top_third_sentences(self):
        axis = ThematicAxis(
            name="alpha beta",
            description="",
        )
        candidates = [
            {"text": "alpha beta. alpha. gamma."},
        ]
        _trim_candidate_texts_by_bm25(axis, candidates)
        assert candidates[0]["text"] == "alpha beta."

    def test_retrieval_candidates_logged(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def __init__(self, hits_by_book):
                self.hits_by_book = hits_by_book
                self.calls = []

            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                self.calls.append(k_per_book)
                return {bid: self.hits_by_book[bid] for bid in book_ids}

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        axis = ThematicAxis(
            axis_id="axis_01",
            name="Axis",
            description="Desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a1",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                ),
                RetrievalHit(
                    chunk_id="a2",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a2",
                    score=0.2,
                    metadata={"chapter_id": "ch1"},
                ),
                RetrievalHit(
                    chunk_id="a3",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a3",
                    score=0.3,
                    metadata={"chapter_id": "ch1"},
                ),
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id="b1",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b1",
                    score=0.1,
                    metadata={"chapter_id": "ch2"},
                ),
                RetrievalHit(
                    chunk_id="b2",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b2",
                    score=0.2,
                    metadata={"chapter_id": "ch2"},
                ),
                RetrievalHit(
                    chunk_id="b3",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b3",
                    score=0.3,
                    metadata={"chapter_id": "ch2"},
                ),
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)
        def _scores_from_payload(payload):
            return PassageExtractionResponse(
                passages=[
                    PassageExtractionScore(
                        passage_id=c["passage_id"],
                        relevance_score=0.5,
                        quotability_score=0.5,
                        synthesis_tags=[],
                    )
                    for c in payload["candidate_passages"]
                ],
                cross_book_pairs=[],
            )

        orch.passage_extraction_agent.run = MagicMock(side_effect=_scores_from_payload)

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(passages_per_axis_per_book=2, rerank_top_k=1),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        asyncio.run(orch._extract_passages(project, [axis], project_dir))

        log_path = (
            project_dir
            / "stage_artifacts"
            / "passage_extraction"
            / "retrieval_candidates_axis_01.json"
        )
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        for book in data["books"]:
            used = [c for c in book["candidates"] if c["used"]]
            assert len(used) == 2

    def test_passage_extraction_rehydrates_scores(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def __init__(self, hits_by_book):
                self.hits_by_book = hits_by_book

            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                return {bid: self.hits_by_book[bid] for bid in book_ids}

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        axis = ThematicAxis(
            axis_id="axis_01",
            name="Axis",
            description="Desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a1",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                ),
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id="b1",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b1",
                    score=0.2,
                    metadata={"chapter_id": "ch2"},
                ),
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)

        def fake_run(payload):
            passages = [
                PassageExtractionScore(
                    passage_id=c["passage_id"],
                    relevance_score=0.9,
                    quotability_score=0.8,
                    synthesis_tags=[SynthesisTag.INDEPENDENT],
                )
                for c in payload["candidate_passages"]
            ]
            return PassageExtractionResponse(passages=passages, cross_book_pairs=[])

        orch.passage_extraction_agent.run = MagicMock(side_effect=fake_run)

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(passages_per_axis_per_book=1, rerank_top_k=1),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        corpus = asyncio.run(orch._extract_passages(project, [axis], project_dir))
        axis_passages = corpus.passages_by_axis[axis.axis_id]
        assert axis_passages
        expected_by_chunk = {"a1": "text a1", "b1": "text b1"}
        for passage in axis_passages:
            assert passage.text == expected_by_chunk[passage.chunk_ids[0]]
            assert passage.relevance_score == 0.9
            assert passage.quotability_score == 0.8
            assert SynthesisTag.INDEPENDENT in passage.synthesis_tags

    def test_passage_extraction_retries_on_low_coverage(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def __init__(self, hits_by_book):
                self.hits_by_book = hits_by_book

            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                return {bid: self.hits_by_book[bid] for bid in book_ids}

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        axis = ThematicAxis(
            axis_id="axis_01",
            name="Axis",
            description="Desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a1",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                ),
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id="b1",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b1",
                    score=0.2,
                    metadata={"chapter_id": "ch2"},
                ),
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)

        bad_response = PassageExtractionResponse(passages=[], cross_book_pairs=[])
        orch.passage_extraction_agent.run = MagicMock(side_effect=[bad_response, bad_response])

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(passages_per_axis_per_book=1, rerank_top_k=1),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        with pytest.raises(RuntimeError, match="fewer than 85%"):
            asyncio.run(orch._extract_passages(project, [axis], project_dir))
        assert orch.passage_extraction_agent.run.call_count == 2

    def test_passage_extraction_ignores_duplicates(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def __init__(self, hits_by_book):
                self.hits_by_book = hits_by_book

            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                return {bid: self.hits_by_book[bid] for bid in book_ids}

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        axis = ThematicAxis(
            axis_id="axis_01",
            name="Axis",
            description="Desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a1",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                ),
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id="b1",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b1",
                    score=0.2,
                    metadata={"chapter_id": "ch2"},
                ),
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)

        def fake_run(payload):
            candidates = payload["candidate_passages"]
            scores = [
                PassageExtractionScore(
                    passage_id=candidates[0]["passage_id"],
                    relevance_score=0.9,
                    quotability_score=0.8,
                    synthesis_tags=[SynthesisTag.INDEPENDENT],
                ),
                PassageExtractionScore(
                    passage_id=candidates[0]["passage_id"],
                    relevance_score=0.85,
                    quotability_score=0.75,
                    synthesis_tags=[SynthesisTag.INDEPENDENT],
                ),
                PassageExtractionScore(
                    passage_id=candidates[1]["passage_id"],
                    relevance_score=0.7,
                    quotability_score=0.6,
                    synthesis_tags=[SynthesisTag.INDEPENDENT],
                ),
            ]
            return PassageExtractionResponse(passages=scores, cross_book_pairs=[])

        orch.passage_extraction_agent.run = MagicMock(side_effect=fake_run)

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(passages_per_axis_per_book=1, rerank_top_k=1),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        corpus = asyncio.run(orch._extract_passages(project, [axis], project_dir))
        axis_passages = corpus.passages_by_axis[axis.axis_id]
        assert len(axis_passages) == 2

    def test_passage_extraction_filters_and_caps_cross_pairs(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def __init__(self, hits_by_book):
                self.hits_by_book = hits_by_book

            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                return {bid: self.hits_by_book[bid] for bid in book_ids}

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        axis = ThematicAxis(
            axis_id="axis_01",
            name="Axis",
            description="Desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="text a1",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                ),
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id="b1",
                    book_id="book-b",
                    chapter_id="ch2",
                    text="text b1",
                    score=0.2,
                    metadata={"chapter_id": "ch2"},
                ),
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)

        def fake_run(payload):
            passages = [
                PassageExtractionScore(
                    passage_id=c["passage_id"],
                    relevance_score=0.9,
                    quotability_score=0.8,
                    synthesis_tags=[SynthesisTag.INDEPENDENT],
                )
                for c in payload["candidate_passages"]
            ]
            cross_pairs = [
                PassagePair(
                    passage_a_id="p1",
                    passage_b_id="p2",
                    relationship=SynthesisTag.AGREES_WITH,
                    strength=0.99,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p3",
                    passage_b_id="p4",
                    relationship=SynthesisTag.EXTENDS,
                    strength=0.95,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p5",
                    passage_b_id="p6",
                    relationship=SynthesisTag.CONTEXTUALIZES,
                    strength=0.9,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p7",
                    passage_b_id="p8",
                    relationship=SynthesisTag.EXEMPLIFIES,
                    strength=0.85,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p9",
                    passage_b_id="p10",
                    relationship=SynthesisTag.CONTRADICTS,
                    strength=0.8,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p11",
                    passage_b_id="p12",
                    relationship=SynthesisTag.CONTEXTUALIZES,
                    strength=0.7,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="p13",
                    passage_b_id="p14",
                    relationship=SynthesisTag.EXEMPLIFIES,
                    strength=0.6,
                    axis_id="axis_01",
                ),
            ]
            return PassageExtractionResponse(passages=passages, cross_book_pairs=cross_pairs)

        orch.passage_extraction_agent.run = MagicMock(side_effect=fake_run)

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(passages_per_axis_per_book=1, rerank_top_k=1),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        corpus = asyncio.run(orch._extract_passages(project, [axis], project_dir))
        relationships = {pair.relationship for pair in corpus.cross_book_pairs}
        assert SynthesisTag.AGREES_WITH not in relationships
        assert SynthesisTag.EXTENDS not in relationships
        assert len(corpus.cross_book_pairs) == 5


def test_select_synthesis_passages_top10_plus_pairs():
    passages = []
    for i in range(12):
        passages.append(
            ExtractedPassage(
                passage_id=f"p{i}",
                book_id="book",
                chunk_ids=[f"c{i}"],
                text=f"text {i}",
                axis_id="axis_01",
                relevance_score=1.0 - (i * 0.01),
                quotability_score=0.5,
                synthesis_tags=[SynthesisTag.INDEPENDENT],
            )
        )
    cross_pair_ids = {"p11"}
    selected = _select_synthesis_passages(passages, cross_pair_ids)
    selected_ids = {p.passage_id for p in selected}
    assert len(selected) == 11
    assert "p11" in selected_ids
    assert "p10" not in selected_ids

    def test_missing_database_url_fails_fast(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
            asyncio.run(orch.run_multi_book_podcast(
                source_paths=["/nonexistent/book1.txt", "/nonexistent/book2.txt"],
                theme="test",
                episode_count=2,
            ))


# ---------------------------------------------------------------------------
# Thematic axis validation
# ---------------------------------------------------------------------------


class TestThematicAxisValidation:
    def test_axis_requires_two_book_relevance(self):
        axis = ThematicAxis(
            axis_id="ax1", name="Test",
            description="Test axis",
            relevance_by_book={"b1": 0.9, "b2": 0.7, "b3": 0.1},
        )
        books_above = sum(1 for s in axis.relevance_by_book.values() if s >= 0.3)
        assert books_above == 2

    def test_axis_with_all_books_relevant(self):
        axis = ThematicAxis(
            axis_id="ax1", name="Universal",
            description="Relevant everywhere",
            relevance_by_book={"b1": 0.8, "b2": 0.7, "b3": 0.6},
        )
        books_above = sum(1 for s in axis.relevance_by_book.values() if s >= 0.3)
        assert books_above == 3


# ---------------------------------------------------------------------------
# Synthesis quality gate
# ---------------------------------------------------------------------------


class TestSynthesisQualityGate:
    def test_quality_score_computation(self):
        """Verify quality score formula from the plan."""
        insights = [
            SynthesisInsight(
                insight_type=InsightType.AGREEMENT,
                title="A", description="D",
                passage_ids=["p1", "p2"],
                podcast_potential=0.8,
            ),
            SynthesisInsight(
                insight_type=InsightType.DISAGREEMENT,
                title="B", description="D",
                passage_ids=["p3", "p4"],
                podcast_potential=0.9,
            ),
        ]
        # All insights involve 2+ books (by passage), 2 types out of 6
        fraction_multi_book = 1.0
        diversity = 2 / 6
        avg_potential = (0.8 + 0.9) / 2
        fraction_in_threads = 0.0  # No threads
        expected = 0.3 * fraction_multi_book + 0.3 * diversity + 0.2 * avg_potential + 0.2 * fraction_in_threads
        assert abs(expected - 0.57) < 0.01

    def test_low_quality_does_not_hard_fail(self):
        """Low quality should warn, not crash."""
        sm = SynthesisMap(
            project_id="proj1",
            quality_score=0.3,
        )
        config = PipelineConfig(synthesis_quality_threshold=0.5)
        # This should trigger a warning but not an error
        assert sm.quality_score < config.synthesis_quality_threshold


# ---------------------------------------------------------------------------
# Synthesis selection
# ---------------------------------------------------------------------------


class TestSynthesisSelection:
    def _make_passage(
        self,
        passage_id: str,
        relevance_score: float,
        *,
        quotability_score: float = 0.0,
        axis_id: str = "ax1",
    ) -> ExtractedPassage:
        return ExtractedPassage(
            passage_id=passage_id,
            book_id="b1",
            chunk_ids=["c1"],
            text="text",
            axis_id=axis_id,
            relevance_score=relevance_score,
            quotability_score=quotability_score,
        )

    def test_select_top_passages_uses_top_ten_cap(self):
        passages = [
            self._make_passage("p1", 0.1),
            self._make_passage("p2", 0.2),
            self._make_passage("p3", 0.3),
            self._make_passage("p4", 0.4),
            self._make_passage("p5", 0.5),
        ]
        selected = _select_top_passages_for_synthesis(passages)
        assert len(selected) == 5
        assert [p.passage_id for p in selected] == ["p5", "p4", "p3", "p2", "p1"]

    def test_select_top_passages_breaks_ties_by_quotability(self):
        passages = [
            self._make_passage("p1", 0.5, quotability_score=0.2),
            self._make_passage("p2", 0.5, quotability_score=0.8),
        ]
        selected = _select_top_passages_for_synthesis(passages)
        assert selected[0].passage_id == "p2"

    def test_select_synthesis_passages_includes_cross_pairs(self):
        passages = [
            self._make_passage("a", 0.1),
            self._make_passage("b", 0.9),
            self._make_passage("c", 0.2),
            self._make_passage("d", 0.3),
            self._make_passage("e", 0.4),
        ]
        selected = _select_synthesis_passages(passages, {"a"})
        assert [p.passage_id for p in selected] == ["b", "e", "d", "c", "a"]


# ---------------------------------------------------------------------------
# Repair loop logic
# ---------------------------------------------------------------------------


class TestRepairLogic:
    def test_passed_report_skips_repair(self):
        report = GroundingReport(
            episode_number=1,
            overall_status="PASSED",
            grounding_score=0.95,
        )
        assert report.overall_status == "PASSED"

    def test_failing_claims_identified(self):
        report = GroundingReport(
            episode_number=1,
            claim_assessments=[
                ClaimAssessment(
                    claim_text="Correct claim",
                    cited_passage_id="p1",
                    status="SUPPORTED",
                ),
                ClaimAssessment(
                    claim_text="Bad claim",
                    cited_passage_id="p2",
                    status="FABRICATED",
                    explanation="Not in source.",
                ),
            ],
            overall_status="NEEDS_REPAIR",
            grounding_score=0.5,
        )
        failing = [ca for ca in report.claim_assessments if ca.status in ("UNSUPPORTED", "FABRICATED")]
        assert len(failing) == 1
        assert failing[0].claim_text == "Bad claim"


# ---------------------------------------------------------------------------
# Episode plan constraints
# ---------------------------------------------------------------------------


class TestEpisodePlanConstraints:
    def test_book_balance_sums_to_one(self):
        plan = EpisodePlan(
            episode_number=1, title="Ep 1",
            book_balance={"b1": 0.6, "b2": 0.4},
        )
        assert abs(sum(plan.book_balance.values()) - 1.0) < 0.01

    def test_cross_references_after_first_episode(self):
        """Episodes after the first should have cross-references."""
        from podcast_agent.schemas.models import CrossReference
        plan = EpisodePlan(
            episode_number=2, title="Ep 2",
            cross_references=[
                CrossReference(
                    from_book_id="b1", to_book_id="b2",
                    connection_type="disagrees",
                    bridge_note="While Author A argues...",
                ),
            ],
        )
        assert len(plan.cross_references) >= 1


# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------


class TestSkipFlags:
    def test_pipeline_config_accepts_skip_grounding(self):
        config = PipelineConfig(skip_grounding=True)
        assert config.skip_grounding is True

    def test_pipeline_config_accepts_skip_spoken_delivery(self):
        config = PipelineConfig(skip_spoken_delivery=True)
        assert config.skip_spoken_delivery is True

    def test_pipeline_config_accepts_skip_audio(self):
        config = PipelineConfig(skip_audio=True)
        assert config.skip_audio is True

    def test_pipeline_config_defaults_to_false(self):
        config = PipelineConfig()
        assert config.skip_grounding is False
        assert config.skip_spoken_delivery is False
        assert config.skip_audio is False

    def test_skip_both(self):
        config = PipelineConfig(skip_grounding=True, skip_spoken_delivery=True, skip_audio=True)
        assert config.skip_grounding is True
        assert config.skip_spoken_delivery is True
        assert config.skip_audio is True

    def test_skip_spoken_delivery_produces_valid_spoken_script(self):
        """When skipping spoken delivery, raw segments become SpokenSegments."""
        from podcast_agent.schemas.models import SpokenScript, SpokenSegment, ScriptSegment
        script_segments = [
            ScriptSegment(segment_id="s1", text="First segment content."),
            ScriptSegment(segment_id="s2", text="Second segment content."),
        ]
        spoken = SpokenScript(
            episode_number=1,
            title="Test",
            segments=[
                SpokenSegment(
                    segment_id=seg.segment_id,
                    text=seg.text,
                    max_words=250,
                )
                for seg in script_segments
            ],
            tts_provider="openai",
        )
        assert len(spoken.segments) == 2
        assert spoken.segments[0].text == "First segment content."

    def test_skip_audio_writes_render_manifest_only(self, tmp_path):
        """Skipping audio should still write render_manifest.json without TTS."""
        from unittest.mock import MagicMock

        from podcast_agent.schemas.models import SpokenScript, SpokenSegment

        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        orch.tts_client = MagicMock()

        project_dir = tmp_path / "run"
        spoken = SpokenScript(
            episode_number=1,
            title="Test",
            segments=[
                SpokenSegment(segment_id="s1", text="Hello world.", max_words=250),
            ],
            tts_provider="openai",
        )

        asyncio.run(
            orch._render_episode_audio(
                1,
                spoken,
                None,
                project_dir,
                asyncio.Semaphore(1),
                skip_audio=True,
            )
        )

        render_path = project_dir / "episodes" / "1" / "render_manifest.json"
        audio_manifest_path = project_dir / "episodes" / "1" / "audio_manifest.json"
        audio_dir = project_dir / "episodes" / "1" / "audio"

        assert render_path.exists()
        assert not audio_manifest_path.exists()
        assert not audio_dir.exists()
        orch.tts_client.synthesize.assert_not_called()


# ---------------------------------------------------------------------------
# Stage logging
# ---------------------------------------------------------------------------


class TestStageLog:
    def test_stage_log_context_manager(self, tmp_path):
        """Verify _stage_log creates artifacts and captures output summary."""
        from podcast_agent.pipeline.orchestrator import _stage_log
        from podcast_agent.run_logging import RunLogger

        run_logger = RunLogger(tmp_path)
        run_logger.bind_run("test_run")

        async def run_stage():
            async with _stage_log(
                run_logger, "test_stage", tmp_path, input_key="input_value",
            ) as ctx:
                ctx["output_summary"] = {"result": "success", "count": 42}

        asyncio.run(run_stage())

        # Check artifacts were created
        stage_dir = tmp_path / "stage_artifacts" / "test_stage"
        assert (stage_dir / "input.json").exists()
        assert (stage_dir / "output.json").exists()

        input_data = json.loads((stage_dir / "input.json").read_text())
        assert input_data["input_key"] == "input_value"

        output_data = json.loads((stage_dir / "output.json").read_text())
        assert output_data["result"] == "success"
        assert output_data["count"] == 42
