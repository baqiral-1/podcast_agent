from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _split_sentences,
    _trim_candidate_texts_by_bm25_query_text,
)
from podcast_agent.retrieval.vector_store import RetrievalHit
from podcast_agent.schemas.models import (
    BookRecord,
    ChapterInfo,
    PipelineConfig,
    ThematicAxis,
    ThematicProject,
)


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def test_trim_candidate_texts_by_bm25_query_text_still_prefers_explicit_query_terms():
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "Keepalpha sentence one. Keepalpha sentence two. "
                "Dropgamma sentence three. Dropgamma sentence four."
            ),
        }
    ]

    _trim_candidate_texts_by_bm25_query_text(
        "keepalpha",
        candidates,
        keep_fraction=0.5,
    )

    assert _split_sentences(candidates[0]["text"]) == [
        "Keepalpha sentence one.",
        "Keepalpha sentence two.",
    ]


def test_extract_passages_uses_global_bm25_admission_without_book_quotas(monkeypatch, tmp_path):
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(enabled=True),
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

    axis = ThematicAxis(
        axis_id="axis_1",
        name="Siege and mutiny",
        description="Rangoon exile and the fall of Delhi",
        theme_importance_score=0.95,
        guiding_questions=["Which chunks are most directly about Rangoon exile?"],
        keywords=["rangoon", "exile", "delhi"],
        relevance_by_book={"b1": 0.1, "b2": 1.0},
    )

    hits_by_book = {
        "b1": [
            RetrievalHit(
                chunk_id="b1-c1",
                book_id="b1",
                chapter_id="ch1",
                text="Rangoon exile explained in detail with Delhi collapse.",
                score=0.40,
                metadata={"chapter_id": "ch1"},
            ),
            RetrievalHit(
                chunk_id="b1-c2",
                book_id="b1",
                chapter_id="ch1",
                text="Delhi siege and Rangoon exile become the core memory.",
                score=0.41,
                metadata={"chapter_id": "ch1"},
            ),
        ],
        "b2": [
            RetrievalHit(
                chunk_id="b2-c1",
                book_id="b2",
                chapter_id="ch1",
                text="General administration and taxes with little on exile.",
                score=0.10,
                metadata={"chapter_id": "ch1"},
            ),
            RetrievalHit(
                chunk_id="b2-c2",
                book_id="b2",
                chapter_id="ch1",
                text="Generic court overview with little on Delhi or Rangoon.",
                score=0.11,
                metadata={"chapter_id": "ch1"},
            ),
        ],
    }
    orchestrator.retrieval.retrieve_for_axis = lambda **_kwargs: hits_by_book  # type: ignore[method-assign]

    def fake_passage_extraction_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.passage_extraction_agent.response_model.model_validate(
            {
                "passages": [
                    {
                        "passage_id": candidate["passage_id"],
                        "relevance_score": 0.9,
                        "quotability_score": 0.8,
                        "synthesis_tags": ["exemplifies"],
                    }
                    for candidate in payload["candidate_passages"]
                ],
                "cross_book_pairs": [],
            }
        )

    orchestrator.passage_extraction_agent.run = fake_passage_extraction_run

    project = ThematicProject(
        project_id="proj",
        theme="Late Mughals",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author 1",
                source_path="/tmp/b1.txt",
                source_type="txt",
                chapters=[
                    ChapterInfo(
                        chapter_id="ch1",
                        title="Chapter 1",
                        start_index=0,
                        end_index=10,
                        word_count=500,
                    )
                ],
                chunk_count=2,
                total_words=1000,
            ),
            BookRecord(
                book_id="b2",
                title="Book 2",
                author="Author 2",
                source_path="/tmp/b2.txt",
                source_type="txt",
                chapters=[
                    ChapterInfo(
                        chapter_id="ch1",
                        title="Chapter 1",
                        start_index=0,
                        end_index=10,
                        word_count=500,
                    )
                ],
                chunk_count=2,
                total_words=1000,
            ),
        ],
        config=PipelineConfig(
            axis_candidate_target_total=2,
            pre_axis_total_budget=2,
            pre_axis_floor=0,
            passage_retrieval_percentage=1.0,
            passage_retrieval_min_per_book=2,
            passage_retrieval_max_per_book=2,
            passage_extraction_concurrency=1,
        ),
    )

    corpus = asyncio.run(orchestrator._extract_passages(project, [axis], tmp_path))

    assert [p.book_id for p in corpus.passages_by_axis["axis_1"]] == ["b1", "b1"]
    payload = captured["payload"]
    assert [item["book_id"] for item in payload["candidate_passages"]] == ["b1", "b1"]

    retrieval_artifact = json.loads(
        (
            tmp_path / "stage_artifacts" / "passage_extraction" / "retrieval_candidates_axis_1.json"
        ).read_text()
    )
    assert retrieval_artifact["allocation_policy"] == "global_bm25_okapi_top_n"
    used_candidates = [
        candidate
        for book in retrieval_artifact["books"]
        for candidate in book["candidates"]
        if candidate["used"]
    ]
    assert {candidate["selection_phase"] for candidate in used_candidates} == {"global_bm25"}
    assert all("bm25_score" in candidate for candidate in used_candidates)
