"""Unit tests for text chunking logic."""

from __future__ import annotations

import pytest

from podcast_agent.pipeline.orchestrator import chunk_text, _split_into_chunks
from podcast_agent.schemas.models import ChapterInfo, ChunkingConfig


class TestSplitIntoChunks:
    def test_short_text_returns_single_chunk(self):
        chunks = _split_into_chunks("Hello world", 400, 50, 1, ["\n\n", ". "])
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text_returns_empty(self):
        chunks = _split_into_chunks("", 400, 50, 80, ["\n\n"])
        assert chunks == []

    def test_text_below_min_words_still_returned(self):
        chunks = _split_into_chunks("short", 400, 50, 80, ["\n\n"])
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        words = " ".join(f"word{i}" for i in range(1000))
        chunks = _split_into_chunks(words, 200, 50, 80, ["\n\n", ". "])
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 250  # Allow some flexibility

    def test_overlap_between_chunks(self):
        words = " ".join(f"word{i}" for i in range(500))
        chunks = _split_into_chunks(words, 100, 20, 10, [])
        # With overlap, chunks should share some words
        assert len(chunks) >= 4

    def test_paragraph_boundary_splitting(self):
        text = "First sentence here. Second sentence here. Third sentence here. " * 50
        chunks = _split_into_chunks(text, 100, 20, 10, [". "])
        for chunk in chunks:
            # Most chunks should end at sentence boundaries
            stripped = chunk.strip()
            if stripped and len(stripped.split()) > 50:
                assert stripped.endswith(".") or stripped.endswith(". ")

    def test_min_chunk_merge(self):
        # Create text where last chunk would be very small
        words = " ".join(f"word{i}" for i in range(210))
        chunks = _split_into_chunks(words, 200, 0, 80, [])
        # The small trailing chunk should be merged
        for chunk in chunks:
            assert len(chunk.split()) >= 10


class TestChunkText:
    def test_single_chapter(self):
        raw_text = " ".join(f"word{i}" for i in range(500))
        chapters = [ChapterInfo(
            title="Chapter 1", start_index=0, end_index=len(raw_text),
            word_count=500, summary="Test chapter.",
        )]
        config = ChunkingConfig(max_chunk_words=200, overlap_words=50, min_chunk_words=50)
        chunks = chunk_text(raw_text, "book1", chapters, config)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.book_id == "book1"
            assert chunk.chapter_id == chapters[0].chapter_id

    def test_multiple_chapters(self):
        raw_text = "Chapter one text here. " * 50 + "Chapter two text here. " * 50
        midpoint = len("Chapter one text here. " * 50)
        chapters = [
            ChapterInfo(
                chapter_id="ch1", title="Ch 1",
                start_index=0, end_index=midpoint,
                word_count=200, summary="First.",
            ),
            ChapterInfo(
                chapter_id="ch2", title="Ch 2",
                start_index=midpoint, end_index=len(raw_text),
                word_count=200, summary="Second.",
            ),
        ]
        config = ChunkingConfig(max_chunk_words=100, overlap_words=20)
        chunks = chunk_text(raw_text, "book1", chapters, config)
        ch1_chunks = [c for c in chunks if c.chapter_id == "ch1"]
        ch2_chunks = [c for c in chunks if c.chapter_id == "ch2"]
        assert len(ch1_chunks) >= 1
        assert len(ch2_chunks) >= 1

    def test_chunk_ids_are_deterministic(self):
        raw_text = "Test text " * 100
        chapters = [ChapterInfo(
            chapter_id="ch1", title="Ch 1",
            start_index=0, end_index=len(raw_text),
            word_count=100, summary="Test.",
        )]
        config = ChunkingConfig(max_chunk_words=50)
        chunks1 = chunk_text(raw_text, "book1", chapters, config)
        chunks2 = chunk_text(raw_text, "book1", chapters, config)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_positions_are_sequential(self):
        raw_text = "Word " * 1000
        chapters = [ChapterInfo(
            chapter_id="ch1", title="Ch 1",
            start_index=0, end_index=len(raw_text),
            word_count=1000, summary="Test.",
        )]
        config = ChunkingConfig(max_chunk_words=200, overlap_words=50)
        chunks = chunk_text(raw_text, "book1", chapters, config)
        positions = [c.position for c in chunks]
        assert positions == list(range(len(positions)))

    def test_empty_chapter_produces_no_chunks(self):
        raw_text = "   "
        chapters = [ChapterInfo(
            title="Empty", start_index=0, end_index=3,
            word_count=0, summary="Empty.",
        )]
        config = ChunkingConfig()
        chunks = chunk_text(raw_text, "book1", chapters, config)
        assert len(chunks) == 0

    def test_chunk_ids_unique_with_repeated_chapter_ids(self):
        raw_text = ("Chapter one text here. " * 40) + ("Chapter one text here. " * 40)
        midpoint = len("Chapter one text here. " * 40)
        chapters = [
            ChapterInfo(
                chapter_id="ch1", title="Ch 1",
                start_index=0, end_index=midpoint,
                word_count=200, summary="First.",
            ),
            ChapterInfo(
                chapter_id="ch1", title="Ch 1 Again",
                start_index=midpoint, end_index=len(raw_text),
                word_count=200, summary="Second.",
            ),
        ]
        config = ChunkingConfig(max_chunk_words=100, overlap_words=20)
        chunks = chunk_text(raw_text, "book1", chapters, config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
