"""Unit tests for source ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_agent.ingestion import read_source_text, normalize_source_text, extract_chapters_from_source


class TestReadSourceText:
    def test_read_text_file(self, tmp_path):
        text_file = tmp_path / "book.txt"
        text_file.write_text("Chapter 1\n\nSome content here.", encoding="utf-8")
        result = read_source_text(text_file)
        assert "Chapter 1" in result
        assert "Some content here" in result

    def test_read_markdown_file(self, tmp_path):
        md_file = tmp_path / "book.md"
        md_file.write_text("# Chapter 1\n\nSome markdown content.", encoding="utf-8")
        result = read_source_text(md_file)
        assert "Chapter 1" in result

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            read_source_text(Path("/nonexistent/file.txt"))


class TestNormalizeSourceText:
    def test_collapses_whitespace(self):
        result = normalize_source_text("Too   many    spaces")
        assert "  " not in result

    def test_handles_line_breaks(self):
        result = normalize_source_text("Line one\r\nLine two\r\n")
        assert "\r" not in result

    def test_handles_hyphenation(self):
        result = normalize_source_text("break-\ning word")
        assert "breaking" in result

    def test_preserves_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two."
        result = normalize_source_text(text)
        assert "\n\n" in result


class TestExtractChaptersFromSource:
    def test_extracts_chapters_from_headings(self):
        raw_text = (
            "Chapter 1\n\n" + ("word " * 1000) +
            "\n\nChapter 2\n\n" + ("word " * 1000)
        )
        chapters = extract_chapters_from_source(raw_text)
        assert len(chapters) == 2
        assert chapters[0].title.lower().startswith("chapter 1")
        assert chapters[1].title.lower().startswith("chapter 2")

    def test_extracts_markdown_headings(self):
        raw_text = (
            "# Chapter 1\n\n" + ("word " * 1000) +
            "\n\n## Chapter 2\n\n" + ("word " * 1000)
        )
        chapters = extract_chapters_from_source(raw_text)
        assert len(chapters) == 2

    def test_raises_when_no_headings(self):
        raw_text = "No headings here, just text."
        with pytest.raises(ValueError, match="No chapter headings"):
            extract_chapters_from_source(raw_text)

    def test_raises_on_short_chapter(self):
        raw_text = "Chapter 1\n\nShort text."
        with pytest.raises(ValueError, match="minimum is 1000"):
            extract_chapters_from_source(raw_text)
