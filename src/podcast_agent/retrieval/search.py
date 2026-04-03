"""Retrieval service for the multi-book thematic pipeline."""

from __future__ import annotations

from podcast_agent.config import Settings
from podcast_agent.retrieval.vector_store import PGVectorRetrieval, RetrievalHit
from podcast_agent.schemas.models import ThematicAxis


class RetrievalService:
    """Cross-book retrieval abstraction used by extraction and writing stages."""

    def __init__(self, settings: Settings, vector_store: PGVectorRetrieval) -> None:
        self.settings = settings
        self.vector_store = vector_store

    def retrieve_for_axis(
        self,
        *,
        axis: ThematicAxis,
        project_id: str,
        book_ids: list[str],
        k_per_book: int,
    ) -> dict[str, list[RetrievalHit]]:
        """Retrieve top-K chunks per book for a given thematic axis."""
        query = self._build_axis_query(axis)
        return self.vector_store.cross_book_search(
            query=query,
            k_per_book=k_per_book,
            project_id=project_id,
            book_ids=book_ids,
        )

    def retrieve_for_query(
        self,
        *,
        query: str,
        project_id: str,
        k: int,
        book_id: str | None = None,
    ) -> list[RetrievalHit]:
        """General-purpose similarity search."""
        oversample = max(1, self.settings.retrieval.oversample_factor)
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            project_id=project_id,
            book_id=book_id,
            oversample=oversample,
        )

    def _build_axis_query(self, axis: ThematicAxis) -> str:
        parts = [axis.name, axis.description]
        parts.extend(axis.guiding_questions)
        parts.extend(axis.keywords)
        return "\n".join(part for part in parts if part).strip()
