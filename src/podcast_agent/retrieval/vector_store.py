"""PGVector-backed similarity retrieval for multi-book projects."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from podcast_agent.config import EmbeddingsConfig, RetrievalConfig, Settings
from podcast_agent.langchain.embeddings import DeterministicEmbeddings
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import TextChunk

try:
    from langchain_postgres import PGVector
except ImportError:
    from langchain_postgres.vectorstores import PGVector


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    book_id: str
    chapter_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class PGVectorRetrieval:
    """Similarity retriever backed by PGVector with multi-book metadata filtering."""

    def __init__(self, settings: Settings, run_logger: RunLogger | None = None) -> None:
        self.settings = settings
        self.run_logger = run_logger
        self.enabled = bool(settings.database.dsn)
        self._vector_store = None
        if self.enabled:
            self._vector_store = self._build_vector_store()

    def _build_vector_store(self) -> Any:
        embeddings = self._build_embeddings(self.settings.embeddings)
        return PGVector(
            connection=self.settings.database.dsn,
            collection_name=self.settings.retrieval.collection_name,
            embeddings=embeddings,
            use_jsonb=True,
        )

    def _build_embeddings(self, config: EmbeddingsConfig):
        provider = config.provider.strip().lower()
        self._log_embeddings_config(config, provider)
        if provider in {"local", "deterministic"}:
            return DeterministicEmbeddings(self.settings.pipeline.embedding_dimensions)
        if provider != "openai":
            raise ValueError(f"Unsupported embeddings provider '{config.provider}'.")
        if not self.settings.llm.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "openai_api_key": self.settings.llm.api_key,
            "openai_api_base": self.settings.llm.base_url,
            "request_timeout": config.timeout_seconds,
            "chunk_size": config.batch_size,
        }
        if config.dimensions is not None:
            kwargs["dimensions"] = config.dimensions
        return OpenAIEmbeddings(**kwargs)

    def _log_embeddings_config(self, config: EmbeddingsConfig, provider: str) -> None:
        if self.run_logger is None:
            return
        self.run_logger.log(
            "embeddings_config",
            provider=provider,
            model_name=config.model_name,
            base_url=self.settings.llm.base_url,
            dimensions=config.dimensions,
            chunk_size=config.batch_size,
        )

    def index_chunks(self, chunks: list[TextChunk], project_id: str) -> None:
        """Index chunks for a single book within a project."""
        if not self.enabled or self._vector_store is None:
            return
        if not chunks:
            return

        book_id = chunks[0].book_id
        # Clean up existing chunks for this book in this project
        try:
            self._vector_store.delete(filter={"book_id": book_id, "project_id": project_id})
        except Exception:
            pass

        documents: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "book_id": chunk.book_id,
                        "chapter_id": chunk.chapter_id,
                        "project_id": project_id,
                        "chunk_id": chunk.chunk_id,
                        "position": chunk.position,
                        **chunk.metadata,
                    },
                )
            )
            ids.append(chunk.chunk_id)
        duplicate_ids = {cid for cid, count in Counter(ids).items() if count > 1}
        if duplicate_ids:
            if self.run_logger is not None:
                self.run_logger.log(
                    "embeddings_duplicate_ids",
                    project_id=project_id,
                    book_id=book_id,
                    chunk_count=len(chunks),
                    duplicate_count=len(duplicate_ids),
                    sample_ids=sorted(list(duplicate_ids))[:5],
                )
            raise ValueError(f"Duplicate chunk IDs detected for book {book_id}.")
        try:
            self._vector_store.add_documents(documents, ids=ids)
        except Exception as exc:
            if self.run_logger is not None:
                self.run_logger.log(
                    "embeddings_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    project_id=project_id,
                    book_id=book_id,
                    chunk_count=len(chunks),
                    provider=self.settings.embeddings.provider,
                    model_name=self.settings.embeddings.model_name,
                    base_url=self.settings.llm.base_url,
                    dimensions=self.settings.embeddings.dimensions,
                    chunk_size=self.settings.embeddings.batch_size,
                )
            raise

    def similarity_search(
        self,
        *,
        query: str,
        k: int,
        project_id: str,
        book_id: str | None = None,
        oversample: int = 1,
    ) -> list[RetrievalHit]:
        """Search for similar chunks, optionally filtered by book_id."""
        if not self.enabled or self._vector_store is None:
            return []

        max_k = max(k, k * oversample)
        filter_dict: dict[str, Any] = {"project_id": project_id}
        if book_id is not None:
            filter_dict["book_id"] = book_id

        results = self._vector_store.similarity_search_with_score(
            query,
            k=max_k,
            filter=filter_dict,
        )

        hits: list[RetrievalHit] = []
        for doc, score in results:
            metadata = doc.metadata or {}
            hits.append(
                RetrievalHit(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    book_id=str(metadata.get("book_id", "")),
                    chapter_id=str(metadata.get("chapter_id", "")),
                    text=doc.page_content,
                    score=float(score) if score is not None else 1.0,
                    metadata=metadata,
                )
            )
            if len(hits) >= k:
                break
        return hits

    def cross_book_search(
        self,
        *,
        query: str,
        k_per_book: int,
        project_id: str,
        book_ids: list[str],
    ) -> dict[str, list[RetrievalHit]]:
        """Search across multiple books, returning k results per book."""
        results: dict[str, list[RetrievalHit]] = {}
        oversample = max(1, self.settings.retrieval.oversample_factor)
        for bid in book_ids:
            results[bid] = self.similarity_search(
                query=query,
                k=k_per_book,
                project_id=project_id,
                book_id=bid,
                oversample=oversample,
            )
        return results
