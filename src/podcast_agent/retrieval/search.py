"""Retrieval service over the repository layer."""

from __future__ import annotations

from podcast_agent.db.repository import Repository
from podcast_agent.schemas.models import RetrievalHit


class RetrievalService:
    """Thin retrieval abstraction used by writer and validator stages."""

    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def fetch_for_episode(self, book_id: str, chunk_ids: list[str]) -> list[RetrievalHit]:
        """Fetch retrieval hits in source order for the requested chunks."""

        return self.repository.fetch_chunks(book_id=book_id, chunk_ids=chunk_ids)
