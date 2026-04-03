"""Embedding helpers for similarity retrieval."""

from __future__ import annotations

from typing import Iterable

from langchain_core.embeddings import Embeddings

from podcast_agent.retrieval.embeddings import embed_text


class DeterministicEmbeddings(Embeddings):
    """Deterministic embeddings for local/dev use."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(text, dimensions=self.dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text, dimensions=self.dimensions)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [embed_text(text, dimensions=self.dimensions) for text in texts]

