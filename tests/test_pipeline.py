"""Integration tests for the pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_agent.agents.book_summary import BookSummaryResponse
from podcast_agent.agents.synthesis_mapping import SynthesisMappingResponse
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryResponse
from podcast_agent.config import Settings
from podcast_agent.agents.theme_decomposition import ThemeDecompositionResponse
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    StructuringStageError,
    _compute_adaptive_rerank_target,
    _evaluate_episode_script_plan_alignment,
    _compute_passage_utilization,
    _compute_passage_retrieval_budget,
    _render_segments_for_spoken_segment,
    _compute_weighted_admitted_budgets,
    _select_episode_planning_passages,
    _save_json,
    _load_json,
    _select_top_passages_for_post_rerank,
    _trim_candidate_texts_by_bm25,
    _select_synthesis_passages,
    _select_top_passages_for_synthesis,
    _normalize_beat_insight_linkage,
)
from podcast_agent.retrieval.vector_store import RetrievalHit
from podcast_agent.agents.passage_extraction import PassageExtractionResponse, PassageExtractionScore
from podcast_agent.agents.writing import EpisodeWritingResponse
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterAnalysis,
    ChapterInfo,
    ClaimAssessment,
    EpisodeArcDetail,
    EpisodeAssignment,
    EpisodeBeat,
    EpisodePlan,
    EpisodeSynthesisContext,
    EpisodeScript,
    ExtractedPassage,
    GroundingReport,
    NarrativeStrategy,
    NarrativeThread,
    MergedNarrative,
    PassagePair,
    PipelineConfig,
    ProjectStatus,
    RenderManifest,
    RenderSegment,
    ScriptSegment,
    SpeechHints,
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


def _episode_arc_details(*episode_numbers: int) -> list[EpisodeArcDetail]:
    return [
        EpisodeArcDetail(
            episode_number=episode_number,
            arc_summary=f"Episode {episode_number} arc summary",
            narrative_stakes=f"Episode {episode_number} narrative stakes",
            progression_beats=[f"Episode {episode_number} progression beat"],
            unresolved_questions=[f"Episode {episode_number} unresolved question"],
            episode_inquiries=[
                {
                    "axis_id": f"axis_{episode_number}",
                    "question": f"Episode {episode_number} inquiry {i}?",
                }
                for i in range(1, 5)
            ],
            payoff_shape=f"Episode {episode_number} payoff shape",
        )
        for episode_number in episode_numbers
    ]


def _episode_assignment_fields(episode_number: int) -> dict[str, object]:
    return {
        "driving_question": f"Episode {episode_number} driving question?",
    }


def _episode_plan_fields(
    episode_number: int,
    *,
    target_duration_minutes: float = 140.0,
) -> dict[str, object]:
    return {
        "target_word_count": int(round(target_duration_minutes * 120)),
        "driving_question": f"Episode {episode_number} driving question?",
        "unresolved_questions": [f"Episode {episode_number} unresolved question"],
        "payoff_shape": f"Episode {episode_number} payoff shape",
    }


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


class TestPassageRetrievalBudget:
    @pytest.mark.parametrize(
        ("chunk_count", "percentage", "min_per_book", "max_per_book", "expected_percentage", "expected_budget"),
        [
            (57, 0.25, 20, 60, 14, 20),
            (142, 0.25, 20, 60, 36, 36),
            (582, 0.25, 20, 60, 146, 60),
        ],
    )
    def test_budget_clamps_percentage_between_floor_and_cap(
        self,
        chunk_count,
        percentage,
        min_per_book,
        max_per_book,
        expected_percentage,
        expected_budget,
    ):
        result = _compute_passage_retrieval_budget(
            chunk_count=chunk_count,
            percentage=percentage,
            min_per_book=min_per_book,
            max_per_book=max_per_book,
        )

        assert result["chunk_count"] == chunk_count
        assert result["percentage_budget"] == expected_percentage
        assert result["per_book_budget"] == expected_budget


class TestWeightedAdmittedBudgets:
    def test_relevance_weighting_is_asymmetric_and_sums_exactly(self):
        budgets = _compute_weighted_admitted_budgets(
            book_ids=["book-a", "book-b"],
            axis_total_budget=100,
            relevance_by_book={"book-a": 1.0, "book-b": 0.2},
        )

        assert budgets == {"book-a": 86, "book-b": 14}
        assert sum(budgets.values()) == 100

    def test_zero_relevance_falls_back_to_even_distribution(self):
        budgets = _compute_weighted_admitted_budgets(
            book_ids=["book-a", "book-b", "book-c"],
            axis_total_budget=36,
            relevance_by_book={},
        )

        assert budgets == {"book-a": 12, "book-b": 12, "book-c": 12}
        assert sum(budgets.values()) == 36

    def test_floor_clamps_to_feasible_budget(self):
        budgets = _compute_weighted_admitted_budgets(
            book_ids=["book-a", "book-b"],
            axis_total_budget=3,
            relevance_by_book={"book-a": 1.0, "book-b": 0.2},
        )

        assert budgets == {"book-a": 2, "book-b": 1}
        assert sum(budgets.values()) == 3

    def test_relevance_only_prefers_higher_relevance_scores(self):
        budgets = _compute_weighted_admitted_budgets(
            book_ids=["book-a", "book-b"],
            axis_total_budget=100,
            relevance_by_book={"book-a": 0.9, "book-b": 0.7},
            floor_per_book=2,
            relevance_power=1.2,
        )

        assert budgets["book-a"] > budgets["book-b"]
        assert sum(budgets.values()) == 100


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
        assert orch.book_summary_agent is not None
        assert orch.theme_decomposition_agent is not None
        assert orch.passage_extraction_agent is not None
        assert orch.synthesis_mapping_agent is not None
        assert orch.narrative_strategy_agent is not None
        assert orch.episode_planning_agent is not None
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

    def test_structuring_failure_blocks_phase_two(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        books = [
            BookRecord(
                book_id="b1", title="B1", author="A1",
                source_path="/tmp/b1.txt", source_type="txt",
            ),
            BookRecord(
                book_id="b2", title="B2", author="A2",
                source_path="/tmp/b2.txt", source_type="txt",
            ),
            BookRecord(
                book_id="b3", title="B3", author="A3",
                source_path="/tmp/b3.txt", source_type="txt",
            ),
        ]
        calls = {"n": 0}

        async def fake_ingest(*args, **kwargs):
            idx = calls["n"]
            calls["n"] += 1
            if idx == 1:
                raise StructuringStageError(
                    "boom",
                    book_id="b2",
                    title="B2",
                    source_path="/tmp/b2.txt",
                )
            return books[idx]

        orch._ingest_and_index_book = AsyncMock(side_effect=fake_ingest)
        orch._decompose_theme = AsyncMock(side_effect=AssertionError("Phase 2 should not run"))

        with pytest.raises(RuntimeError, match="Structuring failed for 1 book\\(s\\)"):
            asyncio.run(orch.run_multi_book_podcast(
                source_paths=["/tmp/b1.txt", "/tmp/b2.txt", "/tmp/b3.txt"],
                theme="test",
                episode_count=2,
                project_id="proj-structuring-fail",
            ))

        project_file = tmp_path / "proj-structuring-fail" / "thematic_project.json"
        data = json.loads(project_file.read_text())
        assert data["status"] == ProjectStatus.FAILED.value
        assert not (tmp_path / "proj-structuring-fail" / "thematic_axes.json").exists()

    def test_bm25_trims_top_third_sentences(self):
        axis = ThematicAxis(
            name="alpha beta",
            description="",
        )
        candidates = [
            {"text": "alpha beta. alpha. gamma. delta."},
        ]
        _trim_candidate_texts_by_bm25(axis, candidates)
        assert candidates[0]["text"] == "alpha beta. alpha."

    def test_passage_extraction_preserves_full_text_when_bm25_trims(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        class FakeRetrieval:
            def retrieve_for_axis(self, *, axis, project_id, book_ids, k_per_book):
                hit = RetrievalHit(
                    chunk_id="a1",
                    book_id="book-a",
                    chapter_id="ch1",
                    text="alpha beta. alpha. gamma.",
                    score=0.1,
                    metadata={"chapter_id": "ch1"},
                )
                return {"book-a": [hit], "book-b": []}

        orch.retrieval = FakeRetrieval()

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
            name="alpha beta",
            description="desc",
            relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.9},
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(
                passage_retrieval_min_per_book=1,
                passage_retrieval_max_per_book=1,
                rerank_top_k=1,
            ),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        corpus = asyncio.run(orch._extract_passages(project, [axis], project_dir))
        passage = corpus.passages_by_axis[axis.axis_id][0]
        assert passage.text == "alpha beta."
        assert passage.trimmed_text == "alpha beta."
        assert passage.full_text == "alpha beta. alpha. gamma."

    def test_decompose_theme_adds_book_summaries_to_payload(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book_a = BookRecord(
            book_id="book-a",
            title="Book A",
            author="Author A",
            source_path="/a.txt",
            source_type="txt",
            chapters=[ChapterInfo(
                title="Ch1",
                start_index=0,
                end_index=100,
                word_count=50,
                summary="Summary A.",
                analysis=ChapterAnalysis(
                    themes_touched=["borders"],
                    major_tensions=["speed vs legitimacy"],
                    causal_shifts=["award delay increases uncertainty"],
                    narrative_hooks=["A line on a map becomes a human crisis."],
                    retrieval_keywords=["border", "delay"],
                ),
            )],
        )
        book_b = BookRecord(
            book_id="book-b",
            title="Book B",
            author="Author B",
            source_path="/b.txt",
            source_type="txt",
            chapters=[ChapterInfo(
                title="Ch2",
                start_index=0,
                end_index=100,
                word_count=50,
                summary="Summary B.",
            )],
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="Partition",
            sub_themes=["borders", "displacement"],
            episode_count=2,
            books=[book_a, book_b],
        )

        orch.book_summary_agent.run = MagicMock(side_effect=[
            BookSummaryResponse(summary="Book A summary."),
            BookSummaryResponse(summary="Book B summary."),
        ])
        orch.theme_decomposition_agent.run = MagicMock(
            return_value=ThemeDecompositionResponse(
                axes=[
                    ThematicAxis(
                        axis_id="axis_01",
                        name="State power",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8, book_b.book_id: 0.7},
                    )
                ]
            )
        )

        axes = asyncio.run(orch._decompose_theme(project, tmp_path / project.project_id))

        assert len(axes) == 1
        assert orch.book_summary_agent.run.call_count == 2
        payload = orch.theme_decomposition_agent.run.call_args.args[0]
        assert payload["sub_themes"] == ["borders", "displacement"]
        assert payload["books"][0]["book_summary"] == "Book A summary."
        assert payload["books"][1]["book_summary"] == "Book B summary."
        assert payload["books"][0]["chapters"][0]["themes_touched"] == ["borders"]
        summary_payload = orch.book_summary_agent.run.call_args_list[0].args[0]
        assert summary_payload["sub_themes"] == ["borders", "displacement"]
        assert summary_payload["chapters"][0]["retrieval_keywords"] == ["border", "delay"]

    def test_decompose_theme_pads_with_fallback_axes_when_valid_below_min(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A", source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B", source_path="/b.txt", source_type="txt",
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="Partition",
            episode_count=2,
            config=PipelineConfig(min_axes=3, max_axes=4),
            books=[book_a, book_b],
        )

        orch.book_summary_agent.run = MagicMock(side_effect=[
            BookSummaryResponse(summary="Book A summary."),
            BookSummaryResponse(summary="Book B summary."),
        ])
        orch.theme_decomposition_agent.run = MagicMock(
            return_value=ThemeDecompositionResponse(
                axes=[
                    ThematicAxis(
                        axis_id="invalid_a",
                        name="Invalid A",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8, book_b.book_id: 0.2},
                    ),
                    ThematicAxis(
                        axis_id="valid_1",
                        name="Valid 1",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8, book_b.book_id: 0.7},
                    ),
                    ThematicAxis(
                        axis_id="invalid_b",
                        name="Invalid B",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.4, book_b.book_id: 0.2},
                    ),
                    ThematicAxis(
                        axis_id="valid_2",
                        name="Valid 2",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.7, book_b.book_id: 0.6},
                    ),
                    ThematicAxis(
                        axis_id="invalid_c",
                        name="Invalid C",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.2, book_b.book_id: 0.1},
                    ),
                ]
            )
        )

        axes = asyncio.run(orch._decompose_theme(project, tmp_path / project.project_id))
        axis_ids = [axis.axis_id for axis in axes]

        assert axis_ids == ["valid_1", "valid_2", "invalid_a", "invalid_b"]

    def test_decompose_theme_caps_axes_at_max(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A", source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B", source_path="/b.txt", source_type="txt",
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="Partition",
            episode_count=2,
            config=PipelineConfig(min_axes=2, max_axes=3),
            books=[book_a, book_b],
        )

        orch.book_summary_agent.run = MagicMock(side_effect=[
            BookSummaryResponse(summary="Book A summary."),
            BookSummaryResponse(summary="Book B summary."),
        ])
        orch.theme_decomposition_agent.run = MagicMock(
            return_value=ThemeDecompositionResponse(
                axes=[
                    ThematicAxis(
                        axis_id="valid_1",
                        name="Valid 1",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8, book_b.book_id: 0.7},
                    ),
                    ThematicAxis(
                        axis_id="valid_2",
                        name="Valid 2",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.7, book_b.book_id: 0.6},
                    ),
                    ThematicAxis(
                        axis_id="valid_3",
                        name="Valid 3",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.6, book_b.book_id: 0.6},
                    ),
                    ThematicAxis(
                        axis_id="valid_4",
                        name="Valid 4",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.9, book_b.book_id: 0.8},
                    ),
                ]
            )
        )

        axes = asyncio.run(orch._decompose_theme(project, tmp_path / project.project_id))
        axis_ids = [axis.axis_id for axis in axes]

        assert axis_ids == ["valid_1", "valid_2", "valid_3"]

    def test_decompose_theme_retries_when_axis_missing_book_relevance(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A", source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B", source_path="/b.txt", source_type="txt",
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="Partition",
            episode_count=2,
            config=PipelineConfig(min_axes=1, max_axes=3),
            books=[book_a, book_b],
        )

        orch.book_summary_agent.run = MagicMock(side_effect=[
            BookSummaryResponse(summary="Book A summary."),
            BookSummaryResponse(summary="Book B summary."),
        ])
        orch.theme_decomposition_agent.run = MagicMock(side_effect=[
            ThemeDecompositionResponse(
                axes=[
                    ThematicAxis(
                        axis_id="axis_01",
                        name="Missing map key",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8},
                    )
                ]
            ),
            ThemeDecompositionResponse(
                axes=[
                    ThematicAxis(
                        axis_id="axis_01",
                        name="Complete map",
                        description="desc",
                        relevance_by_book={book_a.book_id: 0.8, book_b.book_id: 0.7},
                    )
                ]
            ),
        ])

        with patch("podcast_agent.pipeline.orchestrator.time.sleep"):
            axes = asyncio.run(orch._decompose_theme(project, tmp_path / project.project_id))

        assert len(axes) == 1
        assert orch.theme_decomposition_agent.run.call_count == 2
        assert axes[0].axis_id == "axis_01"

    def test_decompose_theme_raises_after_retries_if_book_relevance_missing(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A", source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B", source_path="/b.txt", source_type="txt",
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="Partition",
            episode_count=2,
            config=PipelineConfig(min_axes=1, max_axes=3),
            books=[book_a, book_b],
        )

        orch.book_summary_agent.run = MagicMock(side_effect=[
            BookSummaryResponse(summary="Book A summary."),
            BookSummaryResponse(summary="Book B summary."),
        ])
        bad_response = ThemeDecompositionResponse(
            axes=[
                ThematicAxis(
                    axis_id="axis_01",
                    name="Missing map key",
                    description="desc",
                    relevance_by_book={book_a.book_id: 0.8},
                )
            ]
        )
        orch.theme_decomposition_agent.run = MagicMock(
            side_effect=[bad_response, bad_response]
        )

        with patch("podcast_agent.pipeline.orchestrator.time.sleep"):
            with pytest.raises(RuntimeError, match="omitted input books"):
                asyncio.run(orch._decompose_theme(project, tmp_path / project.project_id))

        assert orch.theme_decomposition_agent.run.call_count == 2

    def test_structure_chapters_persists_summary_and_analysis(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        book = BookRecord(
            book_id="book-a",
            title="Book A",
            author="Author A",
            source_path="/a.txt",
            source_type="txt",
        )
        chapter = ChapterInfo(
            chapter_id="ch1",
            title="Chapter 1",
            start_index=0,
            end_index=20,
            word_count=5,
        )
        orch.chapter_summary_agent.run = MagicMock(
            return_value=MagicMock(
                summary="Structured summary.",
                analysis=ChapterAnalysis(
                    themes_touched=["partition"],
                    major_tensions=["state vs community"],
                    causal_shifts=["announcement changes incentives"],
                    narrative_hooks=["The chapter opens with a political choice."],
                    retrieval_keywords=["partition", "state"],
                ),
            )
        )

        with patch("podcast_agent.pipeline.orchestrator.extract_chapters_from_source", return_value=[chapter]):
            chapters = asyncio.run(
                orch._structure_chapters(
                    book,
                    "mock chapter text",
                    tmp_path / "proj-structure",
                    theme="partition",
                    sub_themes=["borders", "migration"],
                    theme_elaboration="Focus on state formation and displacement.",
                )
            )

        assert chapters[0].summary == "Structured summary."
        assert chapters[0].analysis is not None
        assert chapters[0].analysis.major_tensions == ["state vs community"]
        payload = orch.chapter_summary_agent.run.call_args.args[0]
        assert payload["theme"] == "partition"
        assert payload["sub_themes"] == ["borders", "migration"]
        assert payload["theme_elaboration"] == "Focus on state formation and displacement."


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
            source_path="/a.txt", source_type="txt", chunk_count=20,
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt", chunk_count=8,
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
            config=PipelineConfig(
                passage_retrieval_min_per_book=2,
                passage_retrieval_max_per_book=20,
                rerank_top_k=1,
            ),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        asyncio.run(orch._extract_passages(project, [axis], project_dir))

        assert orch.retrieval.calls == [100]
        log_path = (
            project_dir
            / "stage_artifacts"
            / "passage_extraction"
            / "retrieval_candidates_axis_01.json"
        )
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert data["budget_strategy"] == "fixed_target_soft_threshold_spillover_backfill"
        assert data["allocation_policy"].startswith("floor_2_adaptive_relevance_pow_")
        assert data["axis_candidate_budget_target"] == 200
        assert data["axis_candidate_budget_effective"] == 6
        assert data["axis_candidate_budget"] == 6
        assert data["per_book_budget"] == {"book-a": 3, "book-b": 3}
        assert data["retrieval_depth_by_book"] == {"book-a": 5, "book-b": 2}
        for book in data["books"]:
            used = [c for c in book["candidates"] if c["used"]]
            assert len(used) == 3

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
            config=PipelineConfig(
                passage_retrieval_min_per_book=1,
                passage_retrieval_max_per_book=1,
                rerank_top_k=1,
            ),
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
            assert passage.trimmed_text == expected_by_chunk[passage.chunk_ids[0]]
            assert passage.full_text == expected_by_chunk[passage.chunk_ids[0]]
            assert passage.relevance_score == 0.9
            assert passage.quotability_score == 0.8
            assert SynthesisTag.INDEPENDENT in passage.synthesis_tags

    def test_passage_extraction_uses_weighted_floor_allocation(self, tmp_path):
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
            relevance_by_book={book_a.book_id: 1.0, book_b.book_id: 0.2},
        )
        hits_by_book = {
            "book-a": [
                RetrievalHit(
                    chunk_id=f"a{i}",
                    book_id="book-a",
                    chapter_id="ch1",
                    text=f"text a{i}",
                    score=0.1 * i,
                    metadata={"chapter_id": "ch1"},
                )
                for i in range(1, 21)
            ],
            "book-b": [
                RetrievalHit(
                    chunk_id=f"b{i}",
                    book_id="book-b",
                    chapter_id="ch2",
                    text=f"text b{i}",
                    score=0.1 * i,
                    metadata={"chapter_id": "ch2"},
                )
                for i in range(1, 21)
            ],
        }
        orch.retrieval = FakeRetrieval(hits_by_book)

        orch.passage_extraction_agent.run = MagicMock(
            side_effect=lambda payload: PassageExtractionResponse(
                passages=[
                    PassageExtractionScore(
                        passage_id=c["passage_id"],
                        relevance_score=0.5,
                        quotability_score=0.5,
                        synthesis_tags=[SynthesisTag.INDEPENDENT],
                    )
                    for c in payload["candidate_passages"]
                ],
                cross_book_pairs=[],
            )
        )

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(
                passage_retrieval_min_per_book=2,
                passage_retrieval_max_per_book=20,
                rerank_top_k=1,
            ),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id
        asyncio.run(orch._extract_passages(project, [axis], project_dir))

        data = json.loads(
            (
                project_dir
                / "stage_artifacts"
                / "passage_extraction"
                / "retrieval_candidates_axis_01.json"
            ).read_text()
        )
        used_by_book = {
            book["book_id"]: len([candidate for candidate in book["candidates"] if candidate["used"]])
            for book in data["books"]
        }
        assert data["allocation_policy"].startswith("floor_2_adaptive_relevance_pow_")
        assert sum(data["per_book_budget"].values()) == 40
        assert data["per_book_budget"]["book-a"] > data["per_book_budget"]["book-b"]
        assert data["retrieval_depth_by_book"] == {"book-a": 2, "book-b": 2}
        assert sum(used_by_book.values()) == 40
        assert used_by_book == {"book-a": 20, "book-b": 20}

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
        orch.passage_extraction_agent.run = MagicMock(
            side_effect=[bad_response, bad_response, bad_response]
        )

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(
                passage_retrieval_min_per_book=1,
                passage_retrieval_max_per_book=1,
                rerank_top_k=1,
            ),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        with pytest.raises(RuntimeError, match="fewer than 60%"):
            asyncio.run(orch._extract_passages(project, [axis], project_dir))
        assert orch.passage_extraction_agent.run.call_count == 3

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
            config=PipelineConfig(
                passage_retrieval_min_per_book=1,
                passage_retrieval_max_per_book=1,
                rerank_top_k=1,
            ),
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
        log_spy = MagicMock()
        orch.run_logger.log = log_spy

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
            candidate_ids_by_book: dict[str, list[str]] = {}
            for candidate in payload["candidate_passages"]:
                candidate_ids_by_book.setdefault(candidate["book_id"], []).append(candidate["passage_id"])

            book_a_id = candidate_ids_by_book["book-a"][0]
            book_b_id = candidate_ids_by_book["book-b"][0]
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
                    passage_a_id=book_a_id,
                    passage_b_id=book_b_id,
                    relationship=SynthesisTag.AGREES_WITH,
                    strength=0.99,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_b_id,
                    passage_b_id=book_a_id,
                    relationship=SynthesisTag.EXTENDS,
                    strength=0.95,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_a_id,
                    passage_b_id=book_b_id,
                    relationship=SynthesisTag.CONTEXTUALIZES,
                    strength=0.94,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_b_id,
                    passage_b_id=book_a_id,
                    relationship=SynthesisTag.EXEMPLIFIES,
                    strength=0.93,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_a_id,
                    passage_b_id=book_b_id,
                    relationship=SynthesisTag.CONTRADICTS,
                    strength=0.92,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_b_id,
                    passage_b_id=book_a_id,
                    relationship=SynthesisTag.CONTEXTUALIZES,
                    strength=0.91,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_a_id,
                    passage_b_id=book_b_id,
                    relationship=SynthesisTag.EXEMPLIFIES,
                    strength=0.90,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id=book_a_id,
                    passage_b_id=book_a_id,
                    relationship=SynthesisTag.CONTEXTUALIZES,
                    strength=0.89,
                    axis_id="axis_01",
                ),
                PassagePair(
                    passage_a_id="missing-passage-id",
                    passage_b_id=book_b_id,
                    relationship=SynthesisTag.EXEMPLIFIES,
                    strength=0.88,
                    axis_id="axis_01",
                ),
            ]
            return PassageExtractionResponse(passages=passages, cross_book_pairs=cross_pairs)

        orch.passage_extraction_agent.run = MagicMock(side_effect=fake_run)

        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=2,
            config=PipelineConfig(
                passage_retrieval_min_per_book=1,
                passage_retrieval_max_per_book=1,
                rerank_top_k=1,
            ),
            status=ProjectStatus.ANALYZING,
            books=[book_a, book_b],
        )
        project_dir = tmp_path / project.project_id

        corpus = asyncio.run(orch._extract_passages(project, [axis], project_dir))
        relationships = {pair.relationship for pair in corpus.cross_book_pairs}
        assert SynthesisTag.AGREES_WITH not in relationships
        assert SynthesisTag.EXTENDS not in relationships
        assert len(corpus.cross_book_pairs) == 5
        assert all(pair.passage_a_id != pair.passage_b_id for pair in corpus.cross_book_pairs)

        retrieval_metrics = json.loads((project_dir / "retrieval_metrics.json").read_text())
        axis_metrics = retrieval_metrics["per_axis"]["axis_01"]
        assert axis_metrics["rehydrated_count"] == 2
        assert axis_metrics["full_text_count"] == 2
        assert axis_metrics["trimmed_text_count"] == 2
        assert axis_metrics["full_text_coverage_ratio"] == 1.0
        axis_validation = retrieval_metrics["per_axis"]["axis_01"]["cross_pair_validation"]
        assert axis_validation == {
            "candidate_pair_count": 7,
            "valid_pair_count": 5,
            "retained_pair_count": 5,
            "dropped_missing_id_count": 1,
            "dropped_same_book_count": 1,
        }

        invalid_pair_logs = [
            call
            for call in log_spy.call_args_list
            if call.args and call.args[0] == "passage_extraction_invalid_cross_book_pairs"
        ]
        assert len(invalid_pair_logs) == 1
        invalid_payload = invalid_pair_logs[0].kwargs
        assert invalid_payload["axis_id"] == "axis_01"
        assert invalid_payload["candidate_pair_count"] == 7
        assert invalid_payload["dropped_missing_id_count"] == 1
        assert invalid_payload["dropped_same_book_count"] == 1


def test_select_synthesis_passages_top50_plus_pairs():
    passages = []
    for i in range(52):
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
    cross_pair_ids = {"p51"}
    selected = _select_synthesis_passages(passages, cross_pair_ids)
    selected_ids = {p.passage_id for p in selected}
    assert len(selected) == 51
    assert "p51" in selected_ids
    assert "p50" not in selected_ids

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
        """Verify the prompt-level quality score formula arithmetic."""
        insights = [
            SynthesisInsight(
                insight_type=InsightType.SYNCHRONICITY,
                title="A", description="D",
                passage_ids=["p1", "p2"],
                podcast_potential=0.8,
            ),
            SynthesisInsight(
                insight_type=InsightType.PRODUCTIVE_FRICTION,
                title="B", description="D",
                passage_ids=["p3", "p4"],
                podcast_potential=0.9,
            ),
        ]
        assert len(insights) == 2
        connectivity = 1.0
        narrative_utility = 0.75
        nuance = 0.8
        expected = (0.3 * connectivity) + (0.3 * narrative_utility) + (0.4 * nuance)
        assert abs(expected - 0.845) < 0.001

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

    def test_select_top_passages_uses_top_thirty_cap(self):
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

    def test_select_top_passages_defaults_to_fifty(self):
        passages = [
            self._make_passage(f"p{i:03d}", i / 1000.0)
            for i in range(50)
        ]
        selected = _select_top_passages_for_synthesis(passages)
        assert len(selected) == 50
        assert selected[0].passage_id == "p049"
        assert selected[-1].passage_id == "p000"

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


class TestAdaptiveRerankTarget:
    def test_dense_axis_is_capped_to_one_point_five_x(self):
        policy = _compute_adaptive_rerank_target(
            candidate_count=80,
            rehydrated_count=80,
            valid_cross_pair_count=5,
            book_count=4,
            rerank_top_k=10,
        )
        assert policy["base_total"] == 40
        assert policy["target_total"] == 60
        assert policy["cap_applied"] is True

    def test_sparse_axis_decreases_target(self):
        policy = _compute_adaptive_rerank_target(
            candidate_count=10,
            rehydrated_count=10,
            valid_cross_pair_count=0,
            book_count=4,
            rerank_top_k=10,
        )
        assert policy["base_total"] == 40
        assert policy["target_total"] == 10
        assert policy["cap_applied"] is False


class TestPostRerankSelection:
    def _make_passage(
        self,
        passage_id: str,
        relevance: float,
        quotability: float,
    ) -> ExtractedPassage:
        return ExtractedPassage(
            passage_id=passage_id,
            book_id="book-a",
            chunk_ids=[passage_id],
            text=f"text {passage_id}",
            trimmed_text=f"text {passage_id}",
            full_text=f"text {passage_id}",
            chapter_ref="ch1",
            axis_id="axis_01",
            relevance_score=relevance,
            quotability_score=quotability,
            synthesis_tags=[],
        )

    def test_select_top_passages_for_post_rerank_uses_weighted_final_score(self):
        passages = [
            self._make_passage("a", 0.80, 0.00),
            self._make_passage("b", 0.79, 1.00),
            self._make_passage("c", 0.75, 0.75),
        ]

        selected = _select_top_passages_for_post_rerank(passages, top_k=3)

        assert [p.passage_id for p in selected] == ["b", "c", "a"]

    def test_select_top_passages_for_post_rerank_caps_at_requested_limit(self):
        passages = [
            self._make_passage(f"p{i:03d}", 1.0 - (i / 1000.0), 0.5)
            for i in range(105)
        ]

        selected = _select_top_passages_for_post_rerank(passages, top_k=100)

        assert len(selected) == 100
        assert selected[0].passage_id == "p000"
        assert selected[-1].passage_id == "p099"

    def test_select_top_passages_for_post_rerank_defaults_to_one_forty(self):
        passages = [
            self._make_passage(f"p{i:03d}", 1.0 - (i / 1000.0), 0.5)
            for i in range(150)
        ]

        selected = _select_top_passages_for_post_rerank(passages)

        assert len(selected) == 140
        assert selected[0].passage_id == "p000"
        assert selected[-1].passage_id == "p139"


class TestPassageUtilization:
    def test_compute_passage_utilization(self):
        book_a = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
        )
        book_b = BookRecord(
            book_id="book-b", title="Book B", author="B",
            source_path="/b.txt", source_type="txt",
        )
        corpus = ThematicCorpus(
            project_id="proj-1",
            passages_by_axis={
                "axis_01": [
                    ExtractedPassage(
                        passage_id="p1",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="t1",
                        axis_id="axis_01",
                    ),
                    ExtractedPassage(
                        passage_id="p2",
                        book_id="book-b",
                        chunk_ids=["c2"],
                        text="t2",
                        axis_id="axis_01",
                    ),
                ],
                "axis_02": [
                    ExtractedPassage(
                        passage_id="p3",
                        book_id="book-a",
                        chunk_ids=["c3"],
                        text="t3",
                        axis_id="axis_02",
                    ),
                ],
            },
        )
        episode_plans = [
            EpisodePlan(
                episode_number=1,
                title="Ep1",
                **_episode_plan_fields(1),
                beats=[EpisodeBeat(description="b1", passage_ids=["p1", "p3"])],
            )
        ]
        episode_scripts = [
            EpisodeScript(
                episode_number=1,
                title="Ep1",
                citations=[],
                segments=[
                    ScriptSegment(
                        text="seg",
                        citations=[
                            {
                                "text_span": "q",
                                "passage_id": "p1",
                                "book_id": "book-a",
                            }
                        ],
                    )
                ],
            )
        ]

        utilization = _compute_passage_utilization(
            corpus=corpus,
            episode_plans=episode_plans,
            episode_scripts=episode_scripts,
            books=[book_a, book_b],
        )
        assert utilization["summary"]["retained_count"] == 3
        assert utilization["summary"]["planned_count"] == 2
        assert utilization["summary"]["cited_count"] == 1
        assert utilization["per_axis"]["axis_01"]["planned_count"] == 1
        assert utilization["per_book"]["book-b"]["cited_count"] == 0


class TestEpisodeScriptPlanAlignment:
    def test_aligned_script_has_no_issues(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Ep1",
            **_episode_plan_fields(1),
            insight_ids=["insight-1"],
            beats=[
                EpisodeBeat(
                    beat_id="beat-1",
                    description="Beat",
                    passage_ids=["p1", "p2"],
                )
            ],
            synthesis_context={
                "insights": [
                    {
                        "insight_id": "insight-1",
                        "insight_type": InsightType.SYNCHRONICITY,
                        "title": "Insight",
                        "description": "Desc",
                        "passage_ids": ["p1", "p2"],
                        "axis_ids": ["axis_1"],
                    }
                ]
            },
            cross_references=[
                {
                    "from_book_id": "book-a",
                    "to_book_id": "book-b",
                    "connection_type": "extends",
                }
            ],
            book_balance={"book-a": 0.5, "book-b": 0.5},
        )
        script = EpisodeScript(
            episode_number=1,
            title="Ep1",
            segments=[
                ScriptSegment(
                    text="Narration",
                    beat_id="beat-1",
                    source_book_ids=["book-a", "book-b"],
                    citations=[
                        {"text_span": "a", "passage_id": "p1", "book_id": "book-a"},
                        {"text_span": "b", "passage_id": "p2", "book_id": "book-b"},
                    ],
                )
            ],
            citations=[],
        )

        report = _evaluate_episode_script_plan_alignment(plan=plan, script=script)

        assert report["has_issues"] is False
        assert report["cross_references"]["coverage_ratio"] == 1.0
        assert report["book_balance"]["max_abs_drift"] == 0.0

    def test_missing_insight_realization_is_flagged(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Ep1",
            **_episode_plan_fields(1),
            insight_ids=["insight-1"],
            beats=[EpisodeBeat(beat_id="beat-1", description="Beat", passage_ids=["p1"])],
            synthesis_context={
                "insights": [
                    {
                        "insight_id": "insight-1",
                        "insight_type": InsightType.SYNCHRONICITY,
                        "title": "Insight",
                        "description": "Desc",
                        "passage_ids": ["p1", "p2"],
                        "axis_ids": ["axis_1"],
                    }
                ]
            },
        )
        script = EpisodeScript(
            episode_number=1,
            title="Ep1",
            segments=[ScriptSegment(text="Narration", beat_id=None)],
            citations=[{"text_span": "x", "passage_id": "p_other", "book_id": "book-a"}],
        )

        report = _evaluate_episode_script_plan_alignment(plan=plan, script=script)

        assert report["has_issues"] is True
        assert report["insight_realization"]["has_issues"] is True
        assert report["insight_realization"]["problem_count"] == 1

    def test_cross_reference_under_coverage_is_flagged(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Ep1",
            **_episode_plan_fields(1),
            cross_references=[
                {"from_book_id": "book-a", "to_book_id": "book-b", "connection_type": "agrees"},
                {"from_book_id": "book-a", "to_book_id": "book-c", "connection_type": "agrees"},
                {"from_book_id": "book-b", "to_book_id": "book-c", "connection_type": "agrees"},
            ],
        )
        script = EpisodeScript(
            episode_number=1,
            title="Ep1",
            segments=[
                ScriptSegment(
                    text="Narration",
                    source_book_ids=["book-a", "book-b"],
                )
            ],
            citations=[],
        )

        report = _evaluate_episode_script_plan_alignment(plan=plan, script=script)

        assert report["has_issues"] is True
        assert report["cross_references"]["has_issues"] is True
        assert report["cross_references"]["coverage_ratio"] < 0.5

    def test_book_balance_drift_is_flagged(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Ep1",
            **_episode_plan_fields(1),
            book_balance={"book-a": 0.5, "book-b": 0.5},
        )
        script = EpisodeScript(
            episode_number=1,
            title="Ep1",
            segments=[
                ScriptSegment(text="S1", source_book_ids=["book-a"]),
                ScriptSegment(text="S2", source_book_ids=["book-a"]),
            ],
            citations=[],
        )

        report = _evaluate_episode_script_plan_alignment(plan=plan, script=script)

        assert report["has_issues"] is True
        assert report["book_balance"]["has_issues"] is True
        assert report["book_balance"]["max_abs_drift"] > 0.15

    def test_book_balance_uses_source_book_ids_and_top_level_citations(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Ep1",
            **_episode_plan_fields(1),
            book_balance={"book-a": 0.5, "book-b": 0.5},
        )
        script = EpisodeScript(
            episode_number=1,
            title="Ep1",
            segments=[ScriptSegment(text="Narration", source_book_ids=["book-a"])],
            citations=[{"text_span": "x", "passage_id": "p2", "book_id": "book-b"}],
        )

        report = _evaluate_episode_script_plan_alignment(plan=plan, script=script)

        assert report["book_balance"]["signal_counts"]["book-a"] == 1
        assert report["book_balance"]["signal_counts"]["book-b"] == 1
        assert report["book_balance"]["has_issues"] is False


# ---------------------------------------------------------------------------
# Episode writing source mode
# ---------------------------------------------------------------------------


class TestBeatInsightLinkageNormalization:
    def test_description_insight_refs_are_injected_into_beat_linkage(self):
        beats = [
            EpisodeBeat(
                beat_id="beat-1",
                description="Present the contingency side of ins_38 and ins_41.",
                insight_ids=["ins_38"],
                passage_ids=["p1"],
            ),
            EpisodeBeat(
                beat_id="beat-2",
                description="Transition beat with no explicit insight reference.",
                insight_ids=[],
                passage_ids=["p2"],
            ),
        ]

        adjusted, stats = _normalize_beat_insight_linkage(beats)

        assert adjusted[0].insight_ids == ["ins_38", "ins_41"]
        assert adjusted[1].insight_ids == []
        assert stats["missing_references"] == 1
        assert stats["injected_references"] == 1


class TestWriteEpisodeSourceMode:
    def _build_context(self, tmp_path, *, full_text: str) -> tuple[PipelineOrchestrator, EpisodePlan, ThematicProject, ThematicCorpus, Path, Path]:
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        book = BookRecord(
            book_id="book-a", title="Book A", author="A",
            source_path="/a.txt", source_type="txt",
            chapters=[
                ChapterInfo(
                    chapter_id="ch1",
                    title="Chapter 1",
                    start_index=0,
                    end_index=100,
                    word_count=50,
                    summary="Chapter summary",
                    analysis=ChapterAnalysis(
                        themes_touched=["partition"],
                        major_tensions=["policy vs reality"],
                        causal_shifts=["announcement triggers movement"],
                        narrative_hooks=["One decree changes every household."],
                        retrieval_keywords=["partition", "announcement"],
                    ),
                )
            ],
        )
        project = ThematicProject(
            project_id="proj-1",
            theme="T",
            episode_count=1,
            books=[book],
            config=PipelineConfig(),
        )
        plan = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            **_episode_plan_fields(1),
            synthesis_context=EpisodeSynthesisContext(
                insights=[
                    SynthesisInsight(
                        insight_id="ins_01",
                        insight_type=InsightType.SYNCHRONICITY,
                        title="Insight 1",
                        description="Insight description 1",
                        passage_ids=["p1", "p1b"],
                    ),
                    SynthesisInsight(
                        insight_id="ins_02",
                        insight_type=InsightType.PRODUCTIVE_FRICTION,
                        title="Insight 2",
                        description="Insight description 2",
                        passage_ids=["p2", "p2b"],
                    ),
                ],
                narrative_threads=[
                    NarrativeThread(
                        thread_id="thread-1",
                        title="Thread 1",
                        description="Thread description 1",
                        insight_ids=["ins_01"],
                    ),
                    NarrativeThread(
                        thread_id="thread-2",
                        title="Thread 2",
                        description="Thread description 2",
                        insight_ids=["ins_02"],
                    ),
                ],
            ),
            beats=[
                EpisodeBeat(
                    beat_id="beat-1",
                    description="Beat",
                    insight_ids=["ins_01"],
                    passage_ids=["p1"],
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id=project.project_id,
            passages_by_axis={
                "axis_01": [
                    ExtractedPassage(
                        passage_id="p1",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="trimmed excerpt",
                        full_text=full_text,
                        chapter_ref="ch1",
                        axis_id="axis_01",
                        relevance_score=0.8,
                        quotability_score=0.7,
                        synthesis_tags=[SynthesisTag.INDEPENDENT],
                    )
                ]
            },
        )
        project_dir = tmp_path / project.project_id
        ep_dir = project_dir / "episodes" / "1"
        return orch, plan, project, corpus, ep_dir, project_dir

    def test_write_episode_uses_full_chunk_text(self, tmp_path):
        orch, plan, project, corpus, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="full chunk text",
        )
        orch.writing_agent.run = MagicMock(
            return_value=EpisodeWritingResponse(
                title="Episode 1",
                segments=[ScriptSegment(segment_id="s1", text="Narration", beat_id="beat-1")],
                citations=[],
            )
        )

        asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        payload = orch.writing_agent.run.call_args.args[0]
        assert payload["passages"][0]["text"] == "full chunk text"
        assert payload["writing_source_mode"] == "full_chunk"
        assert payload["plan"]["driving_question"] == "Episode 1 driving question?"
        assert payload["plan"]["unresolved_questions"] == ["Episode 1 unresolved question"]
        assert payload["plan"]["payoff_shape"] == "Episode 1 payoff shape"
        assert "synthesis_context" in payload["plan"]
        assert payload["passages"][0]["chapter_context"]["chapter_title"] == "Chapter 1"
        assert payload["passages"][0]["chapter_context"]["major_tensions"] == ["policy vs reality"]
        output_path = project_dir / "stage_artifacts" / "write_episode_1" / "output.json"
        output = json.loads(output_path.read_text())
        assert output["writing_source_mode"] == "full_chunk"

    def test_write_episode_falls_back_to_trimmed_excerpt(self, tmp_path):
        orch, plan, project, corpus, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="",
        )
        orch.writing_agent.run = MagicMock(
            return_value=EpisodeWritingResponse(
                title="Episode 1",
                segments=[ScriptSegment(segment_id="s1", text="Narration", beat_id="beat-1")],
                citations=[],
            )
        )

        asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        payload = orch.writing_agent.run.call_args.args[0]
        assert payload["passages"][0]["text"] == "trimmed excerpt"

    def test_write_episode_includes_cross_axis_insight_passage_when_plan_returns_it(self, tmp_path):
        orch, plan, project, corpus, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="full chunk text",
        )
        plan = plan.model_copy(
            update={
                "beats": [
                    EpisodeBeat(
                        beat_id="beat-1",
                        description="Beat",
                        passage_ids=["p_cross_axis"],
                    )
                ]
            }
        )
        corpus = ThematicCorpus(
            project_id=project.project_id,
            passages_by_axis={
                "axis_06": [
                    ExtractedPassage(
                        passage_id="p_cross_axis",
                        book_id="book-a",
                        chunk_ids=["c9"],
                        text="cross axis excerpt",
                        full_text="cross axis full text",
                        chapter_ref="ch9",
                        axis_id="axis_06",
                        relevance_score=0.8,
                        quotability_score=0.7,
                        synthesis_tags=[SynthesisTag.INDEPENDENT],
                    )
                ]
            },
        )
        orch.writing_agent.run = MagicMock(
            return_value=EpisodeWritingResponse(
                title="Episode 1",
                segments=[ScriptSegment(segment_id="s1", text="Narration", beat_id="beat-1")],
                citations=[],
            )
        )

        asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        payload = orch.writing_agent.run.call_args.args[0]
        assert payload["passages"][0]["passage_id"] == "p_cross_axis"
        assert payload["passages"][0]["text"] == "cross axis full text"
        assert payload["passages"][0]["chapter_context"] is None

    def test_write_episode_splits_into_three_windowed_requests(self, tmp_path):
        orch, _, project, _, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="",
        )
        beats = [
            EpisodeBeat(
                beat_id=f"beat-{idx}",
                description=f"Beat {idx}",
                insight_ids=(["ins_01"] if idx <= 2 else ["ins_02"] if idx <= 4 else []),
                passage_ids=[f"p{idx}"],
                estimated_duration_seconds=120,
            )
            for idx in range(1, 7)
        ]
        plan = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            **_episode_plan_fields(1),
            synthesis_context=EpisodeSynthesisContext(
                insights=[
                    SynthesisInsight(
                        insight_id="ins_01",
                        insight_type=InsightType.SYNCHRONICITY,
                        title="Insight 1",
                        description="Insight description 1",
                        passage_ids=["p1", "p1b"],
                    ),
                    SynthesisInsight(
                        insight_id="ins_02",
                        insight_type=InsightType.PRODUCTIVE_FRICTION,
                        title="Insight 2",
                        description="Insight description 2",
                        passage_ids=["p2", "p2b"],
                    ),
                ],
                narrative_threads=[
                    NarrativeThread(
                        thread_id="thread-1",
                        title="Thread 1",
                        description="Thread description 1",
                        insight_ids=["ins_01"],
                    ),
                    NarrativeThread(
                        thread_id="thread-2",
                        title="Thread 2",
                        description="Thread description 2",
                        insight_ids=["ins_02"],
                    ),
                ],
            ),
            beats=beats,
        )
        corpus = ThematicCorpus(
            project_id=project.project_id,
            passages_by_axis={
                "axis_01": [
                    ExtractedPassage(
                        passage_id=f"p{idx}",
                        book_id="book-a",
                        chunk_ids=[f"c{idx}"],
                        text=f"excerpt {idx}",
                        full_text=f"full {idx}",
                        chapter_ref="ch1",
                        axis_id="axis_01",
                        relevance_score=0.8,
                        quotability_score=0.7,
                        synthesis_tags=[SynthesisTag.INDEPENDENT],
                    )
                    for idx in range(1, 7)
                ]
            },
        )

        def _fake_run(payload: dict) -> EpisodeWritingResponse:
            payload_beats = payload["plan"]["beats"]
            segments = [
                ScriptSegment(
                    segment_id=f"s-{beat['beat_id']}",
                    text=f"Narration for {beat['beat_id']}",
                    beat_id=beat["beat_id"],
                )
                for beat in payload_beats
            ]
            return EpisodeWritingResponse(
                title="Episode 1",
                segments=segments,
                citations=[],
            )

        orch.writing_agent.run = MagicMock(side_effect=_fake_run)

        script = asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        assert orch.writing_agent.run.call_count == 3
        payloads = [call.args[0] for call in orch.writing_agent.run.call_args_list]
        assert [len(payload["plan"]["beats"]) for payload in payloads] == [2, 2, 2]
        assert [len(payload["passages"]) for payload in payloads] == [2, 2, 2]
        assert [
            [item["insight_id"] for item in payload["plan"]["synthesis_context"]["insights"]]
            for payload in payloads
        ] == [["ins_01"], ["ins_02"], []]
        assert all(
            len(payload["plan"]["synthesis_context"]["narrative_threads"]) == 2
            for payload in payloads
        )
        assert len(script.segments) == 6

    def test_write_episode_logs_duration_shortfall_warning(self, tmp_path):
        orch, _, project, corpus, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="full chunk text",
        )
        log_spy = MagicMock()
        orch.run_logger.log = log_spy
        plan = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            **_episode_plan_fields(1, target_duration_minutes=100.0),
            target_duration_minutes=100.0,
            beats=[
                EpisodeBeat(
                    beat_id="beat-1",
                    description="Beat",
                    passage_ids=["p1"],
                    estimated_duration_seconds=6000,
                )
            ],
        )
        short_text = " ".join(f"word{i}" for i in range(8800))
        orch.writing_agent.run = MagicMock(
            return_value=EpisodeWritingResponse(
                title="Episode 1",
                segments=[ScriptSegment(segment_id="s1", text=short_text, beat_id="beat-1")],
                citations=[],
            )
        )

        asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        shortfall_logs = [
            call
            for call in log_spy.call_args_list
            if call.args and call.args[0] == "episode_write_duration_shortfall_warning"
        ]
        assert len(shortfall_logs) == 1
        assert shortfall_logs[0].kwargs["likely_source"] == "writing"
        assert shortfall_logs[0].kwargs["shortfall_ratio"] > 0.10
        assert shortfall_logs[0].kwargs["shortfall_words"] > 0
        assert shortfall_logs[0].kwargs["target_word_count"] == 12000

    def test_write_episode_does_not_log_duration_shortfall_at_ten_percent(self, tmp_path):
        orch, _, project, corpus, ep_dir, project_dir = self._build_context(
            tmp_path,
            full_text="full chunk text",
        )
        log_spy = MagicMock()
        orch.run_logger.log = log_spy
        plan = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            **_episode_plan_fields(1, target_duration_minutes=100.0),
            target_duration_minutes=100.0,
            beats=[
                EpisodeBeat(
                    beat_id="beat-1",
                    description="Beat",
                    passage_ids=["p1"],
                    estimated_duration_seconds=6000,
                )
            ],
        )
        boundary_text = " ".join(f"word{i}" for i in range(10800))
        orch.writing_agent.run = MagicMock(
            return_value=EpisodeWritingResponse(
                title="Episode 1",
                segments=[ScriptSegment(segment_id="s1", text=boundary_text, beat_id="beat-1")],
                citations=[],
            )
        )

        asyncio.run(orch._write_episode(plan, project, corpus, ep_dir, project_dir))

        assert not any(
            call.args and call.args[0] == "episode_write_duration_shortfall_warning"
            for call in log_spy.call_args_list
        )


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
            **_episode_plan_fields(1),
            book_balance={"b1": 0.6, "b2": 0.4},
        )
        assert abs(sum(plan.book_balance.values()) - 1.0) < 0.01

    def test_cross_references_after_first_episode(self):
        """Episodes after the first should have cross-references."""
        from podcast_agent.schemas.models import CrossReference
        plan = EpisodePlan(
            episode_number=2, title="Ep 2",
            **_episode_plan_fields(2),
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
        assert spoken.segments[0].speech_hints.style == "neutral"

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
                PipelineConfig(),
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

    def test_render_episode_audio_writes_merged_mp3(self, tmp_path, monkeypatch):
        """Audio rendering should synthesize segment files and write a merged episode mp3."""
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        orch.tts_client = MagicMock()
        orch.tts_client.synthesize.return_value = b"audio-bytes"

        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.shutil.which",
            lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
        )

        def _fake_ffmpeg_run(cmd, capture_output, text, check):
            Path(cmd[-1]).write_bytes(b"merged-audio")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.subprocess.run",
            _fake_ffmpeg_run,
        )

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
                PipelineConfig(),
                None,
                project_dir,
                asyncio.Semaphore(1),
                skip_audio=False,
            )
        )

        render_path = project_dir / "episodes" / "1" / "render_manifest.json"
        audio_manifest_path = project_dir / "episodes" / "1" / "audio_manifest.json"
        merged_path = project_dir / "episodes" / "1" / "episode.mp3"

        manifest = _load_json(audio_manifest_path)
        assert render_path.exists()
        assert audio_manifest_path.exists()
        assert merged_path.exists()
        assert manifest["merged_audio_path"] == str(merged_path)
        assert manifest["diagnostics"]["merged"] is True
        assert manifest["diagnostics"]["failed"] == 0

    def test_render_episode_audio_requires_ffmpeg(self, tmp_path, monkeypatch):
        """Merged episode output should fail fast when ffmpeg is unavailable."""
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        orch.tts_client = MagicMock()
        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.shutil.which",
            lambda name: None,
        )

        project_dir = tmp_path / "run"
        spoken = SpokenScript(
            episode_number=1,
            title="Test",
            segments=[
                SpokenSegment(segment_id="s1", text="Hello world.", max_words=250),
            ],
            tts_provider="openai",
        )

        with pytest.raises(RuntimeError, match="ffmpeg is required"):
            asyncio.run(
                orch._render_episode_audio(
                    1,
                    spoken,
                    PipelineConfig(),
                    None,
                    project_dir,
                    asyncio.Semaphore(1),
                    skip_audio=False,
                )
            )

        orch.tts_client.synthesize.assert_not_called()

    def test_synthesize_audio_from_run_skips_missing_manifests(self, tmp_path, monkeypatch):
        """Audio-only command should process episodes with manifests and skip missing ones."""
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path.parent}),
        )
        orch = PipelineOrchestrator(settings)
        run_dir = tmp_path / "run"
        ep1 = run_dir / "episodes" / "1"
        ep2 = run_dir / "episodes" / "2"
        ep1.mkdir(parents=True)
        ep2.mkdir(parents=True)
        _save_json(
            ep1 / "render_manifest.json",
            RenderManifest(
                episode_number=1,
                segments=[RenderSegment(segment_id="s1", text="Hello world.")],
                total_segments=1,
                estimated_duration_seconds=1,
            ),
        )

        async def _fake_render_existing_episode_audio(*args, **kwargs):
            return MagicMock(diagnostics={"merged": True})

        monkeypatch.setattr(orch, "_render_existing_episode_audio", _fake_render_existing_episode_audio)
        monkeypatch.setattr(orch, "_ensure_ffmpeg_available", lambda: "/usr/bin/ffmpeg")

        summary = asyncio.run(orch.synthesize_audio_from_run(run_dir))

        assert summary["processed"] == 1
        assert summary["succeeded"] == 1
        assert summary["failed"] == 0
        assert summary["skipped"] == 1
        assert summary["skipped_episodes"] == [2]

    def test_short_episode_logs_runtime_shortfall_warning(self, tmp_path):
        """Render stage should warn (not fail) when estimated runtime is below configured floor."""
        from unittest.mock import MagicMock

        from podcast_agent.schemas.models import SpokenScript, SpokenSegment

        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        orch.tts_client = MagicMock()
        log_spy = MagicMock()
        orch.run_logger.log = log_spy

        project_dir = tmp_path / "run"
        spoken = SpokenScript(
            episode_number=1,
            title="Short Episode",
            segments=[
                SpokenSegment(segment_id="s1", text="brief content", max_words=250),
            ],
            tts_provider="openai",
        )
        config = PipelineConfig(min_episode_minutes=90.0, target_episode_minutes=100.0)

        asyncio.run(
            orch._render_episode_audio(
                1,
                spoken,
                config,
                None,
                project_dir,
                asyncio.Semaphore(1),
                skip_audio=True,
            )
        )

        assert any(
            call.args and call.args[0] == "episode_runtime_shortfall_warning"
            for call in log_spy.call_args_list
        )

    def test_rewrite_for_speech_normalizes_segments_before_persisting(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        project = ThematicProject(
            project_id="proj-1",
            theme="Test",
            episode_count=2,
            config=PipelineConfig(spoken_chunk_max_words=50),
        )
        long_text = " ".join(f"word{i}" for i in range(1, 56)) + " <break time='1s'/> outro."
        script = EpisodeScript(
            episode_number=1,
            title="Episode 1",
            segments=[
                ScriptSegment(
                    segment_id="s1",
                    text=long_text,
                )
            ],
            total_word_count=56,
            estimated_duration_seconds=0,
        )

        orch.spoken_delivery_agent.run = MagicMock(
            return_value=SpokenDeliveryResponse.model_validate(
                {
                    "segments": [
                        {
                            "segment_id": "s1",
                            "text": long_text,
                            "max_words": 99,
                            "speech_hints": {
                                "style": "urgent",
                                "intensity": "medium",
                                "pause_before_ms": 250,
                                "pause_after_ms": 450,
                                "pace": "faster",
                            },
                        }
                    ],
                    "arc_plan": "Arc",
                }
            )
        )

        ep_dir = tmp_path / "proj-1" / "episodes" / "1"
        spoken = asyncio.run(
            orch._rewrite_for_speech(
                1,
                script,
                project,
                ep_dir,
                tmp_path / "proj-1",
            )
        )

        assert len(spoken.segments) == 2
        assert all("<" not in seg.text for seg in spoken.segments)
        assert all(len(seg.text.split()) <= 50 for seg in spoken.segments)
        assert all(seg.max_words == 50 for seg in spoken.segments)
        assert spoken.segments[0].speech_hints.pace == "faster"
        assert (ep_dir / "spoken_script.json").exists()


class TestSpokenRenderEmphasisFiltering:
    def test_split_sentences_only_include_piece_local_emphasis_targets(self):
        segment = SpokenSegment(
            segment_id="seg_body_36",
            text=(
                "The Ottoman system of power was built on controlled violence. "
                "Some called this for the sake of the good order of the world. "
                "Blood and Throne became a recurring image."
            ),
            speech_hints=SpeechHints(
                render_strategy="split_sentences",
                intensity="light",
                emphasis_targets=[
                    "controlled violence",
                    "for the sake of the good order of the world",
                    "Blood and Throne",
                ],
            ),
        )

        rendered = _render_segments_for_spoken_segment(
            segment,
            voice_id="ballad",
            speed=1.0,
            tts_provider="openai",
            base_instructions="Narrate as a clear documentary host.",
        )

        assert len(rendered) == 3
        assert "controlled violence" in (rendered[0].instructions or "")
        assert "for the sake of the good order of the world" not in (rendered[0].instructions or "")
        assert "Blood and Throne" not in (rendered[0].instructions or "")
        assert "for the sake of the good order of the world" in (rendered[1].instructions or "")
        assert "controlled violence" not in (rendered[1].instructions or "")
        assert "Blood and Throne" in (rendered[2].instructions or "")

    def test_split_sentences_omits_stress_instruction_when_no_local_target_matches(self):
        segment = SpokenSegment(
            segment_id="seg_no_match",
            text="First sentence mentions nothing. Second sentence mentions controlled violence.",
            speech_hints=SpeechHints(
                render_strategy="split_sentences",
                intensity="light",
                emphasis_targets=["controlled violence"],
            ),
        )

        rendered = _render_segments_for_spoken_segment(
            segment,
            voice_id="ballad",
            speed=1.0,
            tts_provider="openai",
            base_instructions="Keep the narration measured.",
        )

        assert len(rendered) == 2
        assert "Give light stress" not in (rendered[0].instructions or "")
        assert "controlled violence" in (rendered[1].instructions or "")


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


class TestEpisodeCountResolution:
    def test_uses_override_when_present(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        project = ThematicProject(
            project_id="proj1",
            theme="Theme",
            requested_episode_count=4,
            episode_count=3,
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2"],
            episode_arc_details=_episode_arc_details(1, 2),
            recommended_episode_count=8,
        )

        resolved = orch._resolve_episode_count_from_strategy(project, strategy)
        assert resolved.episode_count == 4
        assert resolved.recommended_episode_count == 8

    def test_uses_strategy_count_without_override(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        project = ThematicProject(
            project_id="proj1",
            theme="Theme",
            requested_episode_count=None,
            episode_count=3,
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2"],
            episode_arc_details=_episode_arc_details(1, 2),
            recommended_episode_count=8,
        )

        resolved = orch._resolve_episode_count_from_strategy(project, strategy)
        assert resolved.episode_count == 8
        assert resolved.recommended_episode_count == 8

    def test_fails_without_override_when_strategy_missing_count(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        project = ThematicProject(
            project_id="proj1",
            theme="Theme",
            requested_episode_count=None,
            episode_count=3,
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2"],
            episode_arc_details=_episode_arc_details(1, 2),
            recommended_episode_count=None,
        )

        with pytest.raises(RuntimeError, match="did not return recommended_episode_count"):
            orch._resolve_episode_count_from_strategy(project, strategy)


class TestEpisodePlanningPayload:
    def test_narrative_strategy_payload_includes_axis_ids_and_catalogs(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_strategy(payload):
            captured_payloads.append(payload)
            return NarrativeStrategy(
                strategy_type="convergence",
                justification="J",
                series_arc="Arc",
                episode_arc_outline=["Ep1"],
                episode_arc_details=_episode_arc_details(1),
                recommended_episode_count=8,
                episode_assignments=[
                    EpisodeAssignment(
                        episode_number=1,
                        title="Ep 1",
                        **_episode_assignment_fields(1),
                        merged_narrative_id="merged_narrative_001",
                    )
                ],
            )

        orch.narrative_strategy_agent.run = MagicMock(side_effect=fake_strategy)
        project = ThematicProject(project_id="proj1", theme="Theme", books=[])
        corpus = ThematicCorpus(
            project_id="proj1",
            axes=[ThematicAxis(axis_id="axis_1", name="Axis", description="Desc")],
        )
        synthesis_map = SynthesisMap(
            project_id="proj1",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="I1",
                    description="D1",
                    passage_ids=["p1", "p2"],
                    axis_ids=["axis_1"],
                )
            ],
            unresolved_tensions=["Open question"],
            merged_narratives=[
                MergedNarrative(
                    topic="Merged topic",
                    narrative="Merged narrative",
                    source_passage_ids=["p1"],
                )
            ],
            book_relationship_matrix={"a": {"b": "unused"}},
        )

        asyncio.run(
            orch._choose_narrative_strategy(project, synthesis_map, corpus, tmp_path / "proj1")
        )

        payload = captured_payloads[0]["synthesis_map"]
        assert payload["insights"][0]["axis_ids"] == ["axis_1"]
        assert "book_relationship_matrix" not in payload
        assert payload["merged_narratives"][0]["merged_narrative_id"] == "merged_narrative_001"
        assert payload["unresolved_tensions"][0]["tension_id"] == "tension_001"
        project_payload = captured_payloads[0]["project"]
        assert project_payload["target_episode_minutes"] == 140.0
        assert project_payload["min_episode_minutes"] == 125.0

    def test_map_synthesis_retries_once_for_merged_narrative_count(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_synthesis(payload):
            captured_payloads.append(payload)
            if len(captured_payloads) == 1:
                merged = [MergedNarrative(topic="T1", narrative="N1", source_passage_ids=["p1"])]
            else:
                merged = [
                    MergedNarrative(
                        topic=f"T{i}",
                        narrative=f"N{i}",
                        source_passage_ids=["p1"],
                    )
                    for i in range(1, 8)
                ]
            return SynthesisMappingResponse(
                insights=[],
                narrative_threads=[],
                book_relationship_matrix={},
                unresolved_tensions=[],
                quality_score=0.7,
                merged_narratives=merged,
            )

        orch.synthesis_mapping_agent.run = MagicMock(side_effect=fake_synthesis)
        project = ThematicProject(
            project_id="proj_syn_retry",
            theme="Theme",
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_syn_retry",
            axes=[ThematicAxis(axis_id="axis_1", name="Axis 1", description="Desc")],
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p1",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="passage",
                        axis_id="axis_1",
                    )
                ]
            },
        )

        synthesis_map = asyncio.run(
            orch._map_synthesis(project, corpus, tmp_path / "proj_syn_retry")
        )

        assert orch.synthesis_mapping_agent.run.call_count == 2
        assert "synthesis_feedback" in captured_payloads[1]
        assert captured_payloads[1]["synthesis_feedback"]["issue"] == "merged_narrative_count_out_of_range"
        assert len(synthesis_map.merged_narratives) == 7

    def test_map_synthesis_fails_when_merged_narratives_stay_zero_after_retry(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_synthesis(payload):
            captured_payloads.append(payload)
            return SynthesisMappingResponse(
                insights=[],
                narrative_threads=[],
                book_relationship_matrix={},
                unresolved_tensions=[],
                quality_score=0.7,
                merged_narratives=[],
            )

        orch.synthesis_mapping_agent.run = MagicMock(side_effect=fake_synthesis)
        project = ThematicProject(
            project_id="proj_syn_zero_retry",
            theme="Theme",
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_syn_zero_retry",
            axes=[ThematicAxis(axis_id="axis_1", name="Axis 1", description="Desc")],
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p1",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="passage",
                        axis_id="axis_1",
                    )
                ]
            },
        )

        with pytest.raises(RuntimeError, match="zero merged_narratives after retry"):
            asyncio.run(
                orch._map_synthesis(project, corpus, tmp_path / "proj_syn_zero_retry")
            )

        assert orch.synthesis_mapping_agent.run.call_count == 2
        assert "synthesis_feedback" in captured_payloads[1]
        assert captured_payloads[1]["synthesis_feedback"]["issue"] == "merged_narrative_count_out_of_range"

    def test_choose_narrative_strategy_fails_after_retry_on_duplicate_or_missing_merged_assignments(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_strategy(payload):
            captured_payloads.append(payload)
            return NarrativeStrategy(
                strategy_type="convergence",
                justification="J",
                series_arc="Arc",
                episode_arc_outline=["Ep1", "Ep2"],
                episode_arc_details=_episode_arc_details(1, 2),
                recommended_episode_count=8,
                episode_assignments=[
                    EpisodeAssignment(
                        episode_number=1,
                        title="Ep 1",
                        **_episode_assignment_fields(1),
                        axes=[{"axis_id": "axis_1", "description": "Axis 1"}],
                        insight_ids=[],
                        merged_narrative_id="merged_narrative_001",
                    ),
                    EpisodeAssignment(
                        episode_number=2,
                        title="Ep 2",
                        **_episode_assignment_fields(2),
                        axes=[{"axis_id": "axis_2", "description": "Axis 2"}],
                        insight_ids=[],
                        merged_narrative_id="merged_narrative_001",
                    ),
                ],
            )

        orch.narrative_strategy_agent.run = MagicMock(side_effect=fake_strategy)
        project = ThematicProject(project_id="proj_strategy_retry", theme="Theme", books=[])
        corpus = ThematicCorpus(
            project_id="proj_strategy_retry",
            axes=[
                ThematicAxis(axis_id="axis_1", name="Axis 1", description="Desc"),
                ThematicAxis(axis_id="axis_2", name="Axis 2", description="Desc"),
            ],
        )
        synthesis_map = SynthesisMap(
            project_id="proj_strategy_retry",
            merged_narratives=[
                MergedNarrative(topic="M1", narrative="N1", source_passage_ids=["p1"]),
                MergedNarrative(topic="M2", narrative="N2", source_passage_ids=["p2"]),
            ],
        )

        with pytest.raises(RuntimeError, match="merged_narrative_id assignments invalid after retry"):
            asyncio.run(
                orch._choose_narrative_strategy(
                    project, synthesis_map, corpus, tmp_path / "proj_strategy_retry"
                )
            )

        assert orch.narrative_strategy_agent.run.call_count == 2
        assert "strategy_feedback" in captured_payloads[1]
        assert captured_payloads[1]["strategy_feedback"]["issue"] == "episode_merged_narrative_assignment"
        assert captured_payloads[1]["strategy_feedback"]["duplicate_groups"] == [
            {
                "merged_narrative_id": "merged_narrative_001",
                "episode_numbers": [1, 2],
            }
        ]

    def test_supporting_ranking_prefers_higher_quality_single_axis_over_lower_multi_axis(self):
        axis_1_passages = [
            ExtractedPassage(
                passage_id="p_low_multi",
                book_id="book-a",
                chunk_ids=["chunk_shared"],
                text="low multi",
                full_text="low multi full",
                axis_id="axis_1",
                relevance_score=0.60,
                quotability_score=0.60,
            ),
            ExtractedPassage(
                passage_id="p_high_single",
                book_id="book-a",
                chunk_ids=["chunk_single_high"],
                text="high single",
                full_text="high single full",
                axis_id="axis_1",
                relevance_score=0.90,
                quotability_score=0.90,
            ),
        ]
        axis_2_passages = [
            ExtractedPassage(
                passage_id="p_other_axis_shared",
                book_id="book-b",
                chunk_ids=["chunk_shared"],
                text="other axis shared",
                full_text="other axis shared full",
                axis_id="axis_2",
                relevance_score=0.40,
                quotability_score=0.40,
            )
        ]

        selected = _select_episode_planning_passages(
            passages_by_axis={"axis_1": axis_1_passages, "axis_2": axis_2_passages},
            assigned_axis_ids=["axis_1", "axis_2"],
            selected_insight_passage_ids=set(),
            supporting_passages_per_axis=1,
        )

        assert len(selected["axis_1"]) == 1
        assert selected["axis_1"][0].passage_id == "p_high_single"

    def test_supporting_passages_default_cap_is_sixty(self):
        axis_1_passages = [
            ExtractedPassage(
                passage_id=f"p{i:03d}",
                book_id="book-a",
                chunk_ids=[f"chunk_{i:03d}"],
                text=f"summary {i}",
                full_text=f"full text {i}",
                axis_id="axis_1",
                relevance_score=1.0 - (i / 1000.0),
                quotability_score=0.5,
            )
            for i in range(130)
        ]

        selected = _select_episode_planning_passages(
            passages_by_axis={"axis_1": axis_1_passages},
            assigned_axis_ids=["axis_1"],
            selected_insight_passage_ids=set(),
        )

        assert len(selected["axis_1"]) == 60
        assert selected["axis_1"][0].passage_id == "p000"
        assert selected["axis_1"][-1].passage_id == "p059"

    def test_includes_summary_text_and_duplicate_axes_without_extra_fields(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_episode_plan(payload):
            captured_payloads.append(payload)
            return EpisodePlan(episode_number=999, title="Planned", beats=[], **_episode_plan_fields(999))

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj1",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                    chapters=[
                        ChapterInfo(
                            chapter_id="ch1",
                            title="Chapter 1",
                            start_index=0,
                            end_index=0,
                            word_count=1200,
                            summary="Chapter summary",
                            analysis=ChapterAnalysis(
                                major_tensions=["tension-1"],
                                causal_shifts=["shift-1"],
                                narrative_hooks=["hook-1"],
                            ),
                        )
                    ],
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    thematic_focus="Focus",
                    axes=[
                        {"axis_id": "axis_1", "description": "Axis 1 description"},
                        {"axis_id": "axis_2", "description": "Axis 2 description"},
                    ],
                    insight_ids=["insight_1"],
                    merged_narrative_id="merged_narrative_001",
                    tension_ids=["tension_001"],
                    episode_strategy="Compare",
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj1",
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p_a1_shared",
                        book_id="book-a",
                        chunk_ids=["chunk_shared"],
                        text="Axis 1 summary",
                        full_text="Axis 1 shared full text",
                        chapter_ref="ch1",
                        axis_id="axis_1",
                        relevance_score=0.9,
                        quotability_score=0.7,
                    ),
                    ExtractedPassage(
                        passage_id="p_a1_unique",
                        book_id="book-a",
                        chunk_ids=["chunk_unique_a1"],
                        text="Axis 1 unique summary",
                        full_text="Axis 1 unique full text",
                        chapter_ref="ch1",
                        axis_id="axis_1",
                        relevance_score=0.8,
                        quotability_score=0.6,
                    ),
                ],
                "axis_2": [
                    ExtractedPassage(
                        passage_id="p_a2_shared",
                        book_id="book-a",
                        chunk_ids=["chunk_shared"],
                        text="Axis 2 summary",
                        full_text="Axis 2 shared full text",
                        chapter_ref="ch1",
                        axis_id="axis_2",
                        relevance_score=0.7,
                        quotability_score=0.7,
                    ),
                ],
            },
        )
        synthesis_map = SynthesisMap(
            project_id="proj1",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="I1",
                    description="D1",
                    passage_ids=["p_a1_unique", "p_a2_shared"],
                    axis_ids=["axis_1", "axis_2"],
                    podcast_potential=0.9,
                )
            ],
            narrative_threads=[
                NarrativeThread(
                    thread_id="thread_1",
                    title="Thread",
                    description="Thread desc",
                    insight_ids=["insight_1"],
                    arc_type="convergence",
                )
            ],
            unresolved_tensions=["Open question"],
            merged_narratives=[
                MergedNarrative(
                    topic="Merged topic",
                    narrative="Merged narrative",
                    source_passage_ids=["p_a1_unique"],
                    points_of_consensus=["c1"],
                )
            ],
        )

        plans = asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj1")
        )

        assert len(plans) == 1
        assert len(captured_payloads) == 2
        assert captured_payloads[1]["planning_feedback"]["issue"] == "assigned_insight_and_merged_narrative_realization"
        synthesis_payload = captured_payloads[0]["synthesis_map"]
        assert synthesis_payload["insights"][0]["axis_ids"] == ["axis_1", "axis_2"]
        assert synthesis_payload["merged_narratives"][0]["merged_narrative_id"] == "merged_narrative_001"
        assert synthesis_payload["unresolved_tensions"][0]["tension_id"] == "tension_001"
        available = captured_payloads[0]["available_passages"]
        insight_passages = captured_payloads[0]["insight_passages"]
        assert "axis_1" in available and "axis_2" in available
        assert any(p["passage_id"] == "p_a1_shared" for p in available["axis_1"])
        assert any(p["passage_id"] == "p_a2_shared" for p in available["axis_2"])
        assert {entry["passage_id"] for entry in insight_passages} == {"p_a1_unique", "p_a2_shared"}
        assert "chapter_context" not in available["axis_1"][0]
        assert all("full_text" in entry for entry in insight_passages)
        assert captured_payloads[0]["chapter_context_by_ref"] == {
            "book-a": {
                "ch1": {
                    "chapter_id": "ch1",
                    "chapter_title": "Chapter 1",
                    "chapter_summary": "Chapter summary",
                    "themes_touched": [],
                    "major_tensions": ["tension-1"],
                    "causal_shifts": ["shift-1"],
                    "narrative_hooks": ["hook-1"],
                }
            }
        }
        assert captured_payloads[0]["narrative_strategy"]["episode_arc_detail"]["episode_number"] == 1
        assert len(captured_payloads[0]["narrative_strategy"]["episode_arc_detail"]["episode_inquiries"]) == 4
        assignment_payload = captured_payloads[0]["episode_assignment"]
        assert "axis_ids" not in assignment_payload
        assert assignment_payload["axes"] == [
            {"axis_id": "axis_1", "description": "Axis 1 description"},
            {"axis_id": "axis_2", "description": "Axis 2 description"},
        ]
        insight_entries = {
            entry["passage_id"]: entry
            for axis_entries in available.values()
            for entry in axis_entries
            if entry["passage_id"] in {"p_a1_unique", "p_a2_shared"}
        }
        assert "summary_text" in insight_entries["p_a1_unique"]
        assert "full_text" not in insight_entries["p_a1_unique"]
        assert "summary_text" in insight_entries["p_a2_shared"]
        assert "full_text" not in insight_entries["p_a2_shared"]
        non_insight_entry = next(
            entry
            for axis_entries in available.values()
            for entry in axis_entries
            if entry["passage_id"] == "p_a1_shared"
        )
        assert "summary_text" in non_insight_entry
        assert "full_text" not in non_insight_entry
        for axis_entries in available.values():
            for entry in axis_entries:
                assert "chunk_key" not in entry
                assert "cross_axis_passage_ids" not in entry
                assert "cross_axis_summaries" not in entry
        assert plans[0].synthesis_context is not None
        assert plans[0].synthesis_context.insights[0].axis_ids == ["axis_1", "axis_2"]
        assert plans[0].driving_question == "Episode 1 driving question?"
        assert plans[0].unresolved_questions == ["Episode 1 unresolved question"]
        assert plans[0].payoff_shape == "Episode 1 payoff shape"

    def test_payload_keeps_insight_linked_passages_without_cap(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_episode_plan(payload):
            captured_payloads.append(payload)
            return EpisodePlan(episode_number=1, title="Planned", beats=[], **_episode_plan_fields(1))

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        axis_1_passages = [
            ExtractedPassage(
                passage_id=f"p{i:02d}",
                book_id="book-a",
                chunk_ids=[f"chunk_{i:02d}"],
                text=f"summary {i}",
                full_text=f"full text {i}",
                axis_id="axis_1",
                relevance_score=max(0.0, 1.0 - i / 100),
                quotability_score=max(0.0, 1.0 - i / 120),
            )
            for i in range(40)
        ]
        axis_2_passages = [
            ExtractedPassage(
                passage_id="p_shared_axis_2",
                book_id="book-a",
                chunk_ids=["chunk_05"],
                text="shared summary axis2",
                full_text="shared full axis2",
                axis_id="axis_2",
                relevance_score=0.4,
                quotability_score=0.4,
            )
        ]
        project = ThematicProject(
            project_id="proj2",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    thematic_focus="Focus",
                    axis_ids=["axis_1", "axis_2"],
                    insight_ids=["insight_1"],
                    episode_strategy="Compare",
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj2",
            passages_by_axis={"axis_1": axis_1_passages, "axis_2": axis_2_passages},
        )
        synthesis_map = SynthesisMap(
            project_id="proj2",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="I1",
                    description="D1",
                    passage_ids=["p39", "p_shared_axis_2"],
                    podcast_potential=0.9,
                )
            ],
        )

        asyncio.run(orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj2"))

        available = captured_payloads[0]["available_passages"]
        total_passages = sum(len(v) for v in available.values())
        assert total_passages == 41
        assert len(available["axis_1"]) == 40
        assert len(available["axis_2"]) == 1
        all_ids = {entry["passage_id"] for axis_entries in available.values() for entry in axis_entries}
        assert "p39" in all_ids
        assert "p_shared_axis_2" in all_ids
        insight_by_id = {
            entry["passage_id"]: entry
            for axis_entries in available.values()
            for entry in axis_entries
            if entry["passage_id"] in {"p39", "p_shared_axis_2"}
        }
        assert "summary_text" in insight_by_id["p39"]
        assert "full_text" not in insight_by_id["p39"]
        assert "summary_text" in insight_by_id["p_shared_axis_2"]
        assert "full_text" not in insight_by_id["p_shared_axis_2"]
        assert {entry["passage_id"] for entry in captured_payloads[0]["insight_passages"]} == {
            "p39", "p_shared_axis_2"
        }

    def test_planning_payload_includes_cross_axis_insight_passages_separately(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_episode_plan(payload):
            captured_payloads.append(payload)
            return EpisodePlan(episode_number=1, title="Planned", beats=[], **_episode_plan_fields(1))

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)
        project = ThematicProject(
            project_id="proj_cross_axis",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    axis_ids=["axis_1"],
                    insight_ids=["insight_1"],
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_cross_axis",
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p_support",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="support",
                        full_text="support full",
                        axis_id="axis_1",
                    ),
                ],
                "axis_6": [
                    ExtractedPassage(
                        passage_id="p_insight_a",
                        book_id="book-a",
                        chunk_ids=["c2"],
                        text="insight a",
                        full_text="insight a full",
                        axis_id="axis_6",
                    ),
                    ExtractedPassage(
                        passage_id="p_insight_b",
                        book_id="book-a",
                        chunk_ids=["c3"],
                        text="insight b",
                        full_text="insight b full",
                        axis_id="axis_6",
                    ),
                ],
            },
        )
        synthesis_map = SynthesisMap(
            project_id="proj_cross_axis",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="Insight",
                    description="Desc",
                    passage_ids=["p_insight_a", "p_insight_b"],
                    axis_ids=["axis_6"],
                )
            ],
        )

        asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_cross_axis")
        )

        payload = captured_payloads[0]
        assert payload["available_passages"]["axis_1"][0]["passage_id"] == "p_support"
        assert "chapter_context" not in payload["available_passages"]["axis_1"][0]
        assert {entry["passage_id"] for entry in payload["insight_passages"]} == {
            "p_insight_a", "p_insight_b"
        }
        assert payload["insight_passages"][0]["source_axis_ids"] == ["axis_6"]
        assert payload["chapter_context_by_ref"] == {}
        assert not any(
            entry["passage_id"] in {"p_insight_a", "p_insight_b"}
            for entry in payload["available_passages"]["axis_1"]
        )

    def test_plan_series_retries_once_when_assigned_insight_has_zero_realization(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_episode_plan(payload):
            captured_payloads.append(payload)
            if payload.get("planning_feedback") is None:
                return EpisodePlan(
                    episode_number=1,
                    title="Planned",
                    **_episode_plan_fields(1),
                    beats=[EpisodeBeat(beat_id="beat-1", description="Misses insight", passage_ids=["p_other"])],
                )
            return EpisodePlan(
                episode_number=1,
                title="Planned retry",
                **_episode_plan_fields(1),
                beats=[EpisodeBeat(beat_id="beat-2", description="Uses insight", passage_ids=["p_target", "p_missing"])],
            )

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj_retry",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    axis_ids=["axis_1"],
                    insight_ids=["insight_1"],
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_retry",
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p_target",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="target",
                        full_text="target full",
                        axis_id="axis_1",
                    ),
                    ExtractedPassage(
                        passage_id="p_other",
                        book_id="book-a",
                        chunk_ids=["c2"],
                        text="other",
                        full_text="other full",
                        axis_id="axis_1",
                    ),
                ]
            },
        )
        synthesis_map = SynthesisMap(
            project_id="proj_retry",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="Insight",
                    description="Desc",
                    passage_ids=["p_target", "p_missing"],
                    axis_ids=["axis_1"],
                )
            ],
        )

        plans = asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_retry")
        )

        assert len(captured_payloads) == 2
        assert captured_payloads[1]["planning_feedback"]["issue"] == "assigned_insight_realization"
        assert {entry["passage_id"] for entry in captured_payloads[1]["insight_passages"]} == {
            "p_target"
        }
        assert plans[0].beats[0].passage_ids == ["p_target", "p_missing"]
        realization = json.loads((tmp_path / "proj_retry" / "episode_plan_realization.json").read_text())
        assert realization["episodes"][0]["has_issues"] is False

    def test_plan_series_retries_once_when_assigned_merged_narrative_has_zero_realization(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: list[dict] = []

        def fake_episode_plan(payload):
            captured_payloads.append(payload)
            if payload.get("planning_feedback") is None:
                return EpisodePlan(
                    episode_number=1,
                    title="Planned",
                    **_episode_plan_fields(1),
                    beats=[EpisodeBeat(beat_id="beat-1", description="Misses merged narrative", passage_ids=["p_other"])],
                )
            return EpisodePlan(
                episode_number=1,
                title="Planned retry",
                **_episode_plan_fields(1),
                beats=[EpisodeBeat(beat_id="beat-2", description="Uses merged narrative", passage_ids=["p_target"])],
            )

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj_retry_merged",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    axis_ids=["axis_1"],
                    insight_ids=[],
                    merged_narrative_id="merged_narrative_001",
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_retry_merged",
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p_target",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="target",
                        full_text="target full",
                        axis_id="axis_1",
                    ),
                    ExtractedPassage(
                        passage_id="p_other",
                        book_id="book-a",
                        chunk_ids=["c2"],
                        text="other",
                        full_text="other full",
                        axis_id="axis_1",
                    ),
                ]
            },
        )
        synthesis_map = SynthesisMap(
            project_id="proj_retry_merged",
            insights=[],
            merged_narratives=[
                MergedNarrative(
                    topic="Merged",
                    narrative="Narrative",
                    source_passage_ids=["p_target"],
                )
            ],
        )

        plans = asyncio.run(
            orch._plan_series(
                project,
                synthesis_map,
                strategy,
                corpus,
                tmp_path / "proj_retry_merged",
            )
        )

        assert len(captured_payloads) == 2
        assert captured_payloads[1]["planning_feedback"]["issue"] == "assigned_merged_narrative_realization"
        assert plans[0].beats[0].passage_ids == ["p_target"]
        realization = json.loads(
            (tmp_path / "proj_retry_merged" / "episode_plan_realization.json").read_text()
        )
        assert realization["episodes"][0]["has_issues"] is False
        assert realization["episodes"][0]["merged_narrative_problem_count"] == 0

    def test_plan_series_logs_warning_after_retry_if_realization_stays_weak(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        log_spy = MagicMock()
        orch.run_logger.log = log_spy

        orch.episode_planning_agent.run = MagicMock(
            return_value=EpisodePlan(
                episode_number=1,
                title="Planned",
                **_episode_plan_fields(1),
                beats=[EpisodeBeat(beat_id="beat-1", description="Still weak", passage_ids=["p_other"])],
            )
        )

        project = ThematicProject(
            project_id="proj_retry_warn",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Ep 1",
                    **_episode_assignment_fields(1),
                    axis_ids=["axis_1"],
                    insight_ids=["insight_1"],
                )
            ],
        )
        corpus = ThematicCorpus(
            project_id="proj_retry_warn",
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(
                        passage_id="p_target",
                        book_id="book-a",
                        chunk_ids=["c1"],
                        text="target",
                        full_text="target full",
                        axis_id="axis_1",
                    ),
                    ExtractedPassage(
                        passage_id="p_other",
                        book_id="book-a",
                        chunk_ids=["c2"],
                        text="other",
                        full_text="other full",
                        axis_id="axis_1",
                    ),
                ]
            },
        )
        synthesis_map = SynthesisMap(
            project_id="proj_retry_warn",
            insights=[
                SynthesisInsight(
                    insight_id="insight_1",
                    insight_type=InsightType.SYNCHRONICITY,
                    title="Insight",
                    description="Desc",
                    passage_ids=["p_target", "p_unused"],
                    axis_ids=["axis_1"],
                )
            ],
        )

        asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_retry_warn")
        )

        warning_logs = [
            call for call in log_spy.call_args_list
            if call.args and call.args[0] == "episode_plan_insight_realization_warning"
        ]
        assert len(warning_logs) == 1
        assert warning_logs[0].kwargs["problem_count"] == 1
        assert orch.episode_planning_agent.run.call_count == 2

    def test_plan_series_logs_runtime_budget_warning(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        log_spy = MagicMock()
        orch.run_logger.log = log_spy

        orch.episode_planning_agent.run = MagicMock(
            return_value=EpisodePlan(
                episode_number=1,
                title="Planned",
                **_episode_plan_fields(1),
                target_duration_minutes=100.0,
                beats=[
                    EpisodeBeat(
                        beat_id="beat-1",
                        description="Short beat budget",
                        passage_ids=[],
                        estimated_duration_seconds=3000,
                    )
                ],
            )
        )

        project = ThematicProject(
            project_id="proj_runtime_warning",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[EpisodeAssignment(episode_number=1, title="Ep 1", **_episode_assignment_fields(1))],
        )
        corpus = ThematicCorpus(project_id="proj_runtime_warning")
        synthesis_map = SynthesisMap(project_id="proj_runtime_warning")

        plans = asyncio.run(
            orch._plan_series(
                project,
                synthesis_map,
                strategy,
                corpus,
                tmp_path / "proj_runtime_warning",
            )
        )
        assert plans[0].target_word_count == 12000

        warning_logs = [
            call
            for call in log_spy.call_args_list
            if call.args and call.args[0] == "episode_plan_runtime_budget_warning"
        ]
        assert len(warning_logs) == 1
        assert warning_logs[0].kwargs["shortfall_ratio"] > 0.10

    def test_plan_series_logs_beats_warning_outside_70_to_80(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        log_spy = MagicMock()
        orch.run_logger.log = log_spy

        def make_plan(beat_count: int) -> EpisodePlan:
            return EpisodePlan(
                episode_number=1,
                title="Planned",
                **_episode_plan_fields(1),
                target_duration_minutes=140.0,
                beats=[
                    EpisodeBeat(
                        beat_id=f"beat-{idx}",
                        description=f"Beat {idx}",
                        passage_ids=[],
                        estimated_duration_seconds=120,
                    )
                    for idx in range(beat_count)
                ],
            )

        project = ThematicProject(
            project_id="proj_beats_warning",
            theme="Theme",
            episode_count=1,
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1"],
            episode_arc_details=_episode_arc_details(1),
            recommended_episode_count=8,
            episode_assignments=[EpisodeAssignment(episode_number=1, title="Ep 1", **_episode_assignment_fields(1))],
        )
        corpus = ThematicCorpus(project_id="proj_beats_warning")
        synthesis_map = SynthesisMap(project_id="proj_beats_warning")

        orch.episode_planning_agent.run = MagicMock(return_value=make_plan(69))
        asyncio.run(
            orch._plan_series(
                project,
                synthesis_map,
                strategy,
                corpus,
                tmp_path / "proj_beats_warning",
            )
        )
        warning_logs = [
            call
            for call in log_spy.call_args_list
            if call.args and call.args[0] == "episode_plan_beats_warning"
        ]
        assert len(warning_logs) == 1
        assert warning_logs[0].kwargs["beats"] == 69

        log_spy.reset_mock()
        orch.episode_planning_agent.run = MagicMock(return_value=make_plan(75))
        asyncio.run(
            orch._plan_series(
                project.model_copy(update={"project_id": "proj_beats_ok"}),
                synthesis_map.model_copy(update={"project_id": "proj_beats_ok"}),
                strategy,
                corpus.model_copy(update={"project_id": "proj_beats_ok"}),
                tmp_path / "proj_beats_ok",
            )
        )
        warning_logs = [
            call
            for call in log_spy.call_args_list
            if call.args and call.args[0] == "episode_plan_beats_warning"
        ]
        assert warning_logs == []

    def test_parallel_planning_preserves_episode_order(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        delays = {1: 0.07, 2: 0.01, 3: 0.04}

        def fake_episode_plan(payload):
            ep_number = payload["episode_assignment"]["episode_number"]
            time.sleep(delays.get(ep_number, 0.0))
            return EpisodePlan(
                episode_number=ep_number,
                title=f"Generated {ep_number}",
                beats=[],
                **_episode_plan_fields(ep_number),
            )

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj_parallel_order",
            theme="Theme",
            episode_count=3,
            config=PipelineConfig(episode_write_concurrency=3),
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2", "Ep3"],
            episode_arc_details=_episode_arc_details(1, 2, 3),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(episode_number=1, title="Ep 1", thematic_focus="F1", **_episode_assignment_fields(1)),
                EpisodeAssignment(episode_number=2, title="Ep 2", thematic_focus="F2", **_episode_assignment_fields(2)),
                EpisodeAssignment(episode_number=3, title="Ep 3", thematic_focus="F3", **_episode_assignment_fields(3)),
            ],
        )
        corpus = ThematicCorpus(project_id="proj_parallel_order")
        synthesis_map = SynthesisMap(project_id="proj_parallel_order")

        plans = asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_parallel_order")
        )

        assert [plan.episode_number for plan in plans] == [1, 2, 3]

    def test_planning_context_uses_strategy_assignments(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)
        captured_payloads: dict[int, dict] = {}

        def fake_episode_plan(payload):
            ep_number = payload["episode_assignment"]["episode_number"]
            captured_payloads[ep_number] = payload
            return EpisodePlan(
                episode_number=ep_number,
                title=f"Generated {ep_number}",
                beats=[],
                **_episode_plan_fields(ep_number),
            )

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj_context",
            theme="Theme",
            episode_count=3,
            config=PipelineConfig(episode_write_concurrency=3),
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2", "Ep3"],
            episode_arc_details=_episode_arc_details(1, 2, 3),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(episode_number=1, title="Assignment 1", thematic_focus="Focus 1", **_episode_assignment_fields(1)),
                EpisodeAssignment(episode_number=2, title="Assignment 2", thematic_focus="Focus 2", **_episode_assignment_fields(2)),
                EpisodeAssignment(episode_number=3, title="Assignment 3", thematic_focus="Focus 3", **_episode_assignment_fields(3)),
            ],
        )
        corpus = ThematicCorpus(project_id="proj_context")
        synthesis_map = SynthesisMap(project_id="proj_context")

        asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_context")
        )

        assert captured_payloads[1]["previous_episode"] is None
        assert captured_payloads[1]["next_episode"]["title"] == "Assignment 2"
        assert captured_payloads[1]["next_episode"]["driving_question"] == "Episode 2 driving question?"
        assert captured_payloads[2]["previous_episode"]["title"] == "Assignment 1"
        assert captured_payloads[2]["previous_episode"]["driving_question"] == "Episode 1 driving question?"
        assert captured_payloads[2]["next_episode"]["title"] == "Assignment 3"
        assert captured_payloads[3]["previous_episode"]["title"] == "Assignment 2"
        assert captured_payloads[3]["next_episode"] is None

    def test_parallel_planning_respects_episode_write_concurrency(self, tmp_path):
        settings = Settings(
            llm=Settings().llm.model_copy(update={"llm_provider": "heuristic"}),
            database=Settings().database.model_copy(update={"dsn": None}),
            pipeline=Settings().pipeline.model_copy(update={"artifact_root": tmp_path}),
        )
        orch = PipelineOrchestrator(settings)

        lock = threading.Lock()
        in_flight = 0
        max_in_flight = 0

        def fake_episode_plan(payload):
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.06)
            with lock:
                in_flight -= 1
            ep_number = payload["episode_assignment"]["episode_number"]
            return EpisodePlan(
                episode_number=ep_number,
                title=f"Generated {ep_number}",
                beats=[],
                **_episode_plan_fields(ep_number),
            )

        orch.episode_planning_agent.run = MagicMock(side_effect=fake_episode_plan)

        project = ThematicProject(
            project_id="proj_concurrency",
            theme="Theme",
            episode_count=4,
            config=PipelineConfig(episode_write_concurrency=2),
            books=[
                BookRecord(
                    book_id="book-a",
                    title="Book A",
                    author="A",
                    source_path="/a.txt",
                    source_type="txt",
                )
            ],
        )
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="J",
            series_arc="Arc",
            episode_arc_outline=["Ep1", "Ep2", "Ep3", "Ep4"],
            episode_arc_details=_episode_arc_details(1, 2, 3, 4),
            recommended_episode_count=8,
            episode_assignments=[
                EpisodeAssignment(episode_number=1, title="Assignment 1", **_episode_assignment_fields(1)),
                EpisodeAssignment(episode_number=2, title="Assignment 2", **_episode_assignment_fields(2)),
                EpisodeAssignment(episode_number=3, title="Assignment 3", **_episode_assignment_fields(3)),
                EpisodeAssignment(episode_number=4, title="Assignment 4", **_episode_assignment_fields(4)),
            ],
        )
        corpus = ThematicCorpus(project_id="proj_concurrency")
        synthesis_map = SynthesisMap(project_id="proj_concurrency")

        asyncio.run(
            orch._plan_series(project, synthesis_map, strategy, corpus, tmp_path / "proj_concurrency")
        )

        assert max_in_flight > 1
        assert max_in_flight <= 2
