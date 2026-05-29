"""Sentence-embedding helper shared by the semantic tic detector and the
oral-rewriter validation pass.

Wraps ``langchain_openai.OpenAIEmbeddings`` when an API key is available;
falls back to ``DeterministicEmbeddings`` (a hash-based stand-in) otherwise.
Fallback embeddings are NOT useful for semantic similarity — callers should
treat them as a degraded-mode signal and combine with regex/keyword passes.

The wrapper exposes a single ``TextEmbedder`` class with two methods:

- ``embed(texts)`` returns an ``np.ndarray`` of shape ``(len(texts), dim)``.
- ``cosine(a, b)`` returns the cosine similarity between two 1-D vectors.

A module-level cached singleton ``get_text_embedder()`` is provided so
multiple call sites (style_audit linting, oral-rewriter validation) share one
embedder.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_OPENAI_DIM = 256
_DEFAULT_FALLBACK_DIM = 64

_SINGLETON: "TextEmbedder | None" = None


class TextEmbedder:
    """Small wrapper that hides the choice of embedding provider."""

    def __init__(
        self,
        *,
        primary: object | None,
        fallback: object,
        using_fallback: bool,
        dim: int,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self.using_fallback = using_fallback
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        provider = self._fallback if self.using_fallback else self._primary
        vectors = provider.embed_documents(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if not denom:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def cosine_matrix(
        queries: np.ndarray, references: np.ndarray
    ) -> np.ndarray:
        """Pairwise cosine similarity. Returns ``(len(queries), len(references))``."""
        if queries.size == 0 or references.size == 0:
            return np.zeros((queries.shape[0], references.shape[0]), dtype=np.float32)
        q_norm = queries / np.clip(
            np.linalg.norm(queries, axis=1, keepdims=True), 1e-9, None
        )
        r_norm = references / np.clip(
            np.linalg.norm(references, axis=1, keepdims=True), 1e-9, None
        )
        return q_norm @ r_norm.T


def _build_openai_embedder():
    """Construct the OpenAIEmbeddings wrapper or return None on failure."""
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "langchain_openai not installed; semantic embeddings will use the "
            "deterministic fallback."
        )
        return None
    try:
        return OpenAIEmbeddings(
            model=_DEFAULT_OPENAI_MODEL,
            dimensions=_DEFAULT_OPENAI_DIM,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "Failed to construct OpenAIEmbeddings (%s); falling back to deterministic.",
            exc,
        )
        return None


def get_text_embedder() -> TextEmbedder:
    """Return the module-level cached embedder (build lazily on first use)."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    primary = None
    if os.getenv("OPENAI_API_KEY"):
        primary = _build_openai_embedder()
    using_fallback = primary is None
    # Lazy import to avoid circular dependency with the retrieval package
    # (which itself imports langchain.embeddings).
    from podcast_agent.langchain.embeddings import DeterministicEmbeddings

    fallback = DeterministicEmbeddings(_DEFAULT_FALLBACK_DIM)
    dim = _DEFAULT_FALLBACK_DIM if using_fallback else _DEFAULT_OPENAI_DIM
    if using_fallback:
        logger.info(
            "TextEmbedder: using DeterministicEmbeddings fallback (dim=%d); "
            "semantic similarity will be degraded.",
            dim,
        )
    _SINGLETON = TextEmbedder(
        primary=primary,
        fallback=fallback,
        using_fallback=using_fallback,
        dim=dim,
    )
    return _SINGLETON


def reset_text_embedder_for_tests() -> None:
    """Test helper: drop the cached singleton so a new env can be picked up."""
    global _SINGLETON
    _SINGLETON = None
