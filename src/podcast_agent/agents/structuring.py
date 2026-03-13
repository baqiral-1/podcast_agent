"""Structuring agent for canonical book organization."""

from __future__ import annotations

import re
from collections import Counter

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import (
    BookChapter,
    BookChunk,
    BookIngestionResult,
    BookStructure,
    StructuredChapter,
    StructuredChunkDraft,
)

STOP_WORDS = {
    "about",
    "after",
    "because",
    "before",
    "their",
    "there",
    "which",
    "would",
    "could",
    "through",
    "between",
}


class StructuringAgent(Agent):
    """Agent that normalizes raw book text into chapters and chunks."""

    schema_name = "structured_chapter"
    instructions = "Normalize this chapter into canonical chunks."
    response_model = StructuredChapter

    def __init__(
        self,
        llm,
        max_chunk_words: int = 180,
        chunk_overlap_words: int = 30,
        max_structuring_chapter_words: int = 2500,
        structuring_window_words: int = 1800,
        structuring_window_overlap_words: int = 150,
    ) -> None:
        super().__init__(llm)
        self.max_chunk_words = max_chunk_words
        self.chunk_overlap_words = chunk_overlap_words
        self.max_structuring_chapter_words = max_structuring_chapter_words
        self.structuring_window_words = structuring_window_words
        self.structuring_window_overlap_words = structuring_window_overlap_words

    def build_payload(self, chapter_number: int, chapter_title: str, chapter_body: str) -> dict:
        draft = _build_structured_chapter_draft(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_body=chapter_body,
            max_chunk_words=self.max_chunk_words,
            overlap_words=self.chunk_overlap_words,
            body_start_word=0,
            body_start_char=0,
        )
        return {"draft": draft.model_dump(mode="python")}

    def build_window_payload(
        self,
        chapter_number: int,
        chapter_title: str,
        chapter_body: str,
        window_start_word: int,
        window_end_word: int,
    ) -> dict:
        window_text, window_start_char, window_end_char = _slice_words(
            chapter_body,
            window_start_word,
            window_end_word,
        )
        draft = _build_structured_chapter_draft(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_body=window_text,
            max_chunk_words=self.max_chunk_words,
            overlap_words=self.chunk_overlap_words,
            body_start_word=window_start_word,
            body_start_char=window_start_char,
        )
        return {
            "draft": draft.model_dump(mode="python"),
            "window": {
                "start_word": window_start_word,
                "end_word": window_end_word,
                "source_offsets": [window_start_char, window_end_char],
            },
        }

    def structure(self, ingestion: BookIngestionResult) -> BookStructure:
        """Generate the canonical structure for a book."""

        chapters = []
        chunks = []
        sections = _split_into_chapters(ingestion.raw_text)
        for chapter_number, section in enumerate(sections, start=1):
            chapter_id = f"{ingestion.book_id}-chapter-{chapter_number}"
            chapter_title, chapter_body = _extract_title_and_body(section, chapter_number)
            structured_chapter = self._structure_chapter(chapter_number, chapter_title, chapter_body)
            chapter_chunk_ids = []
            for local_sequence, chunk in enumerate(structured_chapter.chunks, start=1):
                chunk_id = f"{chapter_id}-chunk-{local_sequence}"
                chapter_chunk_ids.append(chunk_id)
                chunks.append(
                    BookChunk(
                        chunk_id=chunk_id,
                        chapter_id=chapter_id,
                        chapter_title=structured_chapter.title,
                        chapter_number=chapter_number,
                        sequence=local_sequence,
                        text=chunk.text,
                        start_word=chunk.start_word,
                        end_word=chunk.end_word,
                        source_offsets=chunk.source_offsets,
                        themes=chunk.themes,
                    )
                )
            chapters.append(
                BookChapter(
                    chapter_id=chapter_id,
                    chapter_number=chapter_number,
                    title=structured_chapter.title,
                    summary=structured_chapter.summary,
                    chunk_ids=chapter_chunk_ids,
                )
            )
        return BookStructure(
            book_id=ingestion.book_id,
            title=ingestion.title,
            chapters=chapters,
            chunks=chunks,
        )

    def _structure_chapter(
        self,
        chapter_number: int,
        chapter_title: str,
        chapter_body: str,
    ) -> StructuredChapter:
        chapter_words = len(chapter_body.split())
        if chapter_words <= self.max_structuring_chapter_words:
            return self.run(self.build_payload(chapter_number, chapter_title, chapter_body))

        windows = _window_ranges(
            total_words=chapter_words,
            window_words=self.structuring_window_words,
            overlap_words=self.structuring_window_overlap_words,
        )
        window_outputs = []
        for window_index, (start_word, end_word) in enumerate(windows, start=1):
            payload = self.build_window_payload(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chapter_body=chapter_body,
                window_start_word=start_word,
                window_end_word=end_word,
            )
            result = self.run(payload)
            window_outputs.append((window_index, result))
        return _merge_structured_chapter_windows(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_body=chapter_body,
            windows=window_outputs,
        )


def _split_into_chapters(text: str) -> list[str]:
    pattern = re.compile(r"(?im)^(chapter\s+\d+[:\s-].*|chapter\s+\d+.*)$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    sections = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _extract_title_and_body(section: str, chapter_number: int) -> tuple[str, str]:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if not lines:
        return f"Chapter {chapter_number}", section
    title = lines[0]
    body = "\n".join(lines[1:]).strip() or title
    return title, body


def _chunk_text(text: str, max_words: int, overlap_words: int) -> list[dict[str, int | str]]:
    words = text.split()
    if not words:
        return [{"text": "", "start_word": 0, "end_word": 0, "start_char": 0, "end_char": 0}]
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        start_char = max(0, text.find(chunk_words[0]))
        end_char = start_char + len(chunk_text)
        chunks.append(
            {
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _build_structured_chapter_draft(
    chapter_number: int,
    chapter_title: str,
    chapter_body: str,
    max_chunk_words: int,
    overlap_words: int,
    body_start_word: int,
    body_start_char: int,
) -> StructuredChapter:
    chunk_drafts = []
    for chunk in _chunk_text(
        chapter_body,
        max_words=max_chunk_words,
        overlap_words=overlap_words,
    ):
        chunk_drafts.append(
            StructuredChunkDraft(
                text=chunk["text"],
                start_word=body_start_word + int(chunk["start_word"]),
                end_word=body_start_word + int(chunk["end_word"]),
                source_offsets=[
                    body_start_char + int(chunk["start_char"]),
                    body_start_char + int(chunk["end_char"]),
                ],
                themes=_infer_themes(chunk["text"]),
            )
        )
    summary = chapter_body.split(".")[0].strip() or chapter_title
    return StructuredChapter(
        chapter_number=chapter_number,
        title=chapter_title,
        summary=summary,
        chunks=chunk_drafts,
    )


def _window_ranges(total_words: int, window_words: int, overlap_words: int) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    while start < total_words:
        end = min(total_words, start + window_words)
        ranges.append((start, end))
        if end == total_words:
            break
        start = max(0, end - overlap_words)
    return ranges


def _slice_words(text: str, start_word: int, end_word: int) -> tuple[str, int, int]:
    words = text.split()
    selected_words = words[start_word:end_word]
    if not selected_words:
        return "", 0, 0
    selected_text = " ".join(selected_words)
    start_char = text.find(selected_words[0])
    end_char = start_char + len(selected_text)
    return selected_text, start_char, end_char


def _merge_structured_chapter_windows(
    chapter_number: int,
    chapter_title: str,
    chapter_body: str,
    windows: list[tuple[int, StructuredChapter]],
) -> StructuredChapter:
    seen_positions = set()
    merged_chunks = []
    for _, structured_window in windows:
        for chunk in structured_window.chunks:
            key = (chunk.start_word, chunk.end_word, chunk.text)
            if key in seen_positions:
                continue
            seen_positions.add(key)
            merged_chunks.append(chunk)
    merged_chunks.sort(key=lambda chunk: (chunk.start_word, chunk.end_word))
    summary = chapter_body.split(".")[0].strip() or chapter_title
    return StructuredChapter(
        chapter_number=chapter_number,
        title=chapter_title,
        summary=summary,
        chunks=merged_chunks,
    )


def _infer_themes(text: str) -> list[str]:
    terms = [
        token.lower().strip(" ,.;:!?()[]{}\"'")
        for token in text.split()
        if len(token) >= 6
    ]
    filtered = [term for term in terms if term.isalpha() and term not in STOP_WORDS]
    return [theme for theme, _ in Counter(filtered).most_common(4)]
