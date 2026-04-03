"""Unit tests for the retrieval layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_agent.config import Settings
from podcast_agent.retrieval.search import RetrievalService
from podcast_agent.retrieval.vector_store import PGVectorRetrieval, RetrievalHit
from podcast_agent.schemas.models import ThematicAxis


class TestRetrievalService:
    def _make_service(self):
        settings = Settings(
            database=Settings().database.model_copy(update={"dsn": None}),
        )
        vs = PGVectorRetrieval(settings)
        return RetrievalService(settings, vs)

    def test_build_axis_query(self):
        service = self._make_service()
        axis = ThematicAxis(
            axis_id="ax1",
            name="Cognitive Biases",
            description="How cognitive biases affect decisions.",
            guiding_questions=["What biases exist?", "How do they manifest?"],
            keywords=["bias", "heuristic", "judgment"],
        )
        query = service._build_axis_query(axis)
        assert "Cognitive Biases" in query
        assert "How cognitive biases" in query
        assert "What biases exist?" in query
        assert "bias" in query

    def test_retrieve_for_axis_no_database(self):
        service = self._make_service()
        axis = ThematicAxis(
            axis_id="ax1", name="Test",
            description="Test axis",
        )
        result = service.retrieve_for_axis(
            axis=axis, project_id="proj1",
            book_ids=["b1", "b2"], k_per_book=10,
        )
        # With no database, returns empty
        assert result == {"b1": [], "b2": []}


class TestPGVectorRetrieval:
    def test_disabled_without_dsn(self):
        settings = Settings(
            database=Settings().database.model_copy(update={"dsn": None}),
        )
        vs = PGVectorRetrieval(settings)
        assert not vs.enabled

    def test_index_chunks_noop_when_disabled(self):
        settings = Settings(
            database=Settings().database.model_copy(update={"dsn": None}),
        )
        vs = PGVectorRetrieval(settings)
        # Should not raise
        vs.index_chunks([], "proj1")

    def test_similarity_search_returns_empty_when_disabled(self):
        settings = Settings(
            database=Settings().database.model_copy(update={"dsn": None}),
        )
        vs = PGVectorRetrieval(settings)
        results = vs.similarity_search(
            query="test", k=10, project_id="proj1",
        )
        assert results == []

    def test_logs_embeddings_config(self):
        class _Logger:
            def __init__(self):
                self.events = []

            def log(self, event_type, **payload):
                self.events.append((event_type, payload))

        settings = Settings(
            database=Settings().database.model_copy(update={"dsn": None}),
            embeddings=Settings().embeddings.model_copy(update={"provider": "deterministic"}),
        )
        logger = _Logger()
        vs = PGVectorRetrieval(settings, run_logger=logger)
        vs._build_embeddings(settings.embeddings)
        assert any(event[0] == "embeddings_config" for event in logger.events)


class TestRetrievalHit:
    def test_creation(self):
        hit = RetrievalHit(
            chunk_id="c1", book_id="b1", chapter_id="ch1",
            text="Some chunk text", score=0.95,
            metadata={"author": "Author A"},
        )
        assert hit.chunk_id == "c1"
        assert hit.score == 0.95
        assert hit.metadata["author"] == "Author A"
