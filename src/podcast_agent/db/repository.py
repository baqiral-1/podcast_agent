"""Repository implementations for books, chunks, and stage artifacts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
import threading
from typing import Any

import psycopg

from podcast_agent.schemas.models import BookIngestionResult, BookStructure, RetrievalHit


class Repository(ABC):
    """Persistence contract used by the pipeline."""

    @abstractmethod
    def save_book(self, ingestion: BookIngestionResult) -> None:
        """Persist a book record."""

    @abstractmethod
    def save_structure(self, structure: BookStructure) -> None:
        """Persist chapters and chunks."""

    @abstractmethod
    def save_embeddings(self, book_id: str, embeddings: dict[str, list[float]]) -> None:
        """Persist vector embeddings for chunks."""

    @abstractmethod
    def fetch_chunks(self, book_id: str, chunk_ids: list[str] | None = None) -> list[RetrievalHit]:
        """Retrieve chunk content for downstream agents."""


class InMemoryRepository(Repository):
    """Simple repository used for tests and local dry runs."""

    def __init__(self) -> None:
        self.books: dict[str, BookIngestionResult] = {}
        self.structures: dict[str, BookStructure] = {}
        self.embeddings: dict[str, dict[str, list[float]]] = {}
        self._lock = threading.Lock()

    def save_book(self, ingestion: BookIngestionResult) -> None:
        with self._lock:
            self.books[ingestion.book_id] = ingestion

    def save_structure(self, structure: BookStructure) -> None:
        with self._lock:
            self.structures[structure.book_id] = structure

    def save_embeddings(self, book_id: str, embeddings: dict[str, list[float]]) -> None:
        with self._lock:
            self.embeddings[book_id] = embeddings

    def fetch_chunks(self, book_id: str, chunk_ids: list[str] | None = None) -> list[RetrievalHit]:
        structure = self.structures[book_id]
        chapter_lookup = {chapter.chapter_id: chapter for chapter in structure.chapters}
        hits = []
        for chunk in structure.chunks:
            if chunk_ids and chunk.chunk_id not in chunk_ids:
                continue
            chapter = chapter_lookup[chunk.chapter_id]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    chapter_id=chunk.chapter_id,
                    chapter_title=chapter.title,
                    score=1.0,
                    text=chunk.text,
                )
            )
        return hits


class PostgresRepository(Repository):
    """PostgreSQL-backed repository implementation."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def save_book(self, ingestion: BookIngestionResult) -> None:
        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                insert into books (book_id, title, author, source_path, source_type, raw_text, ingested_at)
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (book_id) do update set
                  title = excluded.title,
                  author = excluded.author,
                  source_path = excluded.source_path,
                  source_type = excluded.source_type,
                  raw_text = excluded.raw_text,
                  ingested_at = excluded.ingested_at
                """,
                (
                    ingestion.book_id,
                    ingestion.title,
                    ingestion.author,
                    ingestion.source_path,
                    ingestion.source_type.value,
                    ingestion.raw_text,
                    ingestion.ingested_at,
                ),
            )

    def save_structure(self, structure: BookStructure) -> None:
        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            for chapter in structure.chapters:
                cursor.execute(
                    """
                    insert into chapters (chapter_id, book_id, chapter_number, title, summary, chunk_ids)
                    values (%s, %s, %s, %s, %s, %s::jsonb)
                    on conflict (chapter_id) do update set
                      chapter_number = excluded.chapter_number,
                      title = excluded.title,
                      summary = excluded.summary,
                      chunk_ids = excluded.chunk_ids
                    """,
                    (
                        chapter.chapter_id,
                        structure.book_id,
                        chapter.chapter_number,
                        chapter.title,
                        chapter.summary,
                        json.dumps(chapter.chunk_ids),
                    ),
                )
            for chunk in structure.chunks:
                cursor.execute(
                    """
                    insert into chunks (
                      chunk_id, book_id, chapter_id, chapter_title, chapter_number, sequence,
                      text, start_word, end_word, source_offsets, themes
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    on conflict (chunk_id) do update set
                      chapter_title = excluded.chapter_title,
                      chapter_number = excluded.chapter_number,
                      sequence = excluded.sequence,
                      text = excluded.text,
                      start_word = excluded.start_word,
                      end_word = excluded.end_word,
                      source_offsets = excluded.source_offsets,
                      themes = excluded.themes
                    """,
                    (
                        chunk.chunk_id,
                        structure.book_id,
                        chunk.chapter_id,
                        chunk.chapter_title,
                        chunk.chapter_number,
                        chunk.sequence,
                        chunk.text,
                        chunk.start_word,
                        chunk.end_word,
                        json.dumps(list(chunk.source_offsets)),
                        json.dumps(chunk.themes),
                    ),
                )

    def save_embeddings(self, book_id: str, embeddings: dict[str, list[float]]) -> None:
        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            for chunk_id, vector in embeddings.items():
                cursor.execute(
                    """
                    insert into chunk_embeddings (chunk_id, book_id, embedding)
                    values (%s, %s, %s::vector)
                    on conflict (chunk_id) do update set embedding = excluded.embedding
                    """,
                    (chunk_id, book_id, _format_pgvector(vector)),
                )

    def fetch_chunks(self, book_id: str, chunk_ids: list[str] | None = None) -> list[RetrievalHit]:
        clause = ""
        parameters: list[Any] = [book_id]
        if chunk_ids:
            clause = "and chunk_id = any(%s)"
            parameters.append(chunk_ids)
        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                select chunk_id, chapter_id, chapter_title, text
                from chunks
                where book_id = %s {clause}
                order by chapter_number, sequence
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [
            RetrievalHit(
                chunk_id=row[0],
                chapter_id=row[1],
                chapter_title=row[2],
                score=1.0,
                text=row[3],
            )
            for row in rows
        ]


class ArtifactStore:
    """Filesystem persistence for stage outputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, run_id: str, name: str, payload: dict[str, Any]) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def write_bytes(self, run_id: str, name: str, payload: bytes) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / name
        target.write_bytes(payload)
        return target


def _format_pgvector(vector: list[float]) -> str:
    """Serialize a vector in pgvector's accepted text format."""

    return "[" + ",".join(str(value) for value in vector) + "]"
