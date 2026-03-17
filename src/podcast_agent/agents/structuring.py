"""Structuring agent for canonical book organization."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError

from podcast_agent.agents.base import Agent
from podcast_agent.ingestion import normalize_source_text
from podcast_agent.llm.base import LLMContentFilterError
from podcast_agent.schemas.models import (
    BookChapter,
    BookChunk,
    BookIngestionResult,
    BookStructure,
    SourceType,
    StructuredChapter,
    StructuredChapterPlan,
    StructuredChunkDraft,
)
from podcast_agent.utils import (
    split_into_chapters,
    split_into_detected_headings,
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
    instructions = (
        "Normalize this chapter into canonical chunk metadata. "
        "Return only chapter title, summary, and chunk boundaries with themes. "
        "Do not quote or reproduce chunk text from the source."
    )
    response_model = StructuredChapterPlan

    def __init__(
        self,
        llm,
        max_chunk_words: int = 180,
        chunk_overlap_words: int = 30,
        max_structuring_chapter_words: int = 2500,
        max_structuring_llm_chapter_words: int = 75000,
        structuring_parallelism: int = 10,
        structuring_window_words: int = 1800,
        structuring_window_overlap_words: int = 150,
    ) -> None:
        super().__init__(llm)
        self.max_chunk_words = max_chunk_words
        self.chunk_overlap_words = chunk_overlap_words
        self.max_structuring_chapter_words = max_structuring_chapter_words
        self.max_structuring_llm_chapter_words = max_structuring_llm_chapter_words
        self.structuring_parallelism = structuring_parallelism
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

    def structure(
        self,
        ingestion: BookIngestionResult,
        *,
        start_chapter: str | None = None,
        end_chapter: str | None = None,
    ) -> BookStructure:
        """Generate the canonical structure for a book."""

        chapters = []
        chunks = []
        normalized_text = (
            normalize_source_text(ingestion.raw_text)
            if ingestion.source_type == SourceType.PDF
            else ingestion.raw_text
        )
        heading_sections = split_into_detected_headings(normalized_text)
        if len(heading_sections) == 1 and len(heading_sections[0].body.split()) > self.max_structuring_chapter_words:
            self._log(
                "structuring_sectioning_risk",
                source_type=ingestion.source_type.value,
                section_count=1,
                title=heading_sections[0].title,
                word_count=len(heading_sections[0].body.split()),
                message="Detected a single oversized section before chapter structuring.",
            )
        sections = split_into_chapters(normalized_text)
        if sections and all(section.title.startswith("Section ") for section in sections):
            self._log(
                "structuring_sectioning_fallback",
                source_type=ingestion.source_type.value,
                section_count=len(sections),
                message="No reliable chapter headings detected; using deterministic fallback sectioning.",
            )
        start_index = 0
        end_index = len(sections) - 1
        if start_chapter is not None:
            start_index = _find_section_index(sections, start_chapter, "start")
        if end_chapter is not None:
            end_index = _find_section_index(sections, end_chapter, "end")
        if end_index < start_index:
            raise ValueError(
                f"End chapter '{end_chapter}' appears before start chapter '{start_chapter}'."
            )
        selected_sections = sections[start_index : end_index + 1]
        if start_chapter is not None or end_chapter is not None:
            self._log(
                "structuring_chapter_range_applied",
                requested_start_chapter=start_chapter,
                matched_start_chapter=selected_sections[0].title if selected_sections else None,
                requested_end_chapter=end_chapter,
                matched_end_chapter=selected_sections[-1].title if selected_sections else None,
                selected_section_count=len(selected_sections),
            )
        chapter_inputs = []
        for chapter_number, section in enumerate(selected_sections, start=1):
            chapter_inputs.append((chapter_number, section.title, section.body))
        self._log(
            "structuring_schedule",
            chapter_count=len(chapter_inputs),
            concurrency=self.structuring_parallelism,
        )
        structured_results: dict[int, StructuredChapter | None] = {}
        with ThreadPoolExecutor(max_workers=self.structuring_parallelism) as executor:
            future_map = {
                executor.submit(self._structure_chapter, chapter_number, chapter_title, chapter_body): (
                    chapter_number,
                    chapter_title,
                )
                for chapter_number, chapter_title, chapter_body in chapter_inputs
            }
            for future in as_completed(future_map):
                chapter_number, chapter_title = future_map[future]
                try:
                    structured_results[chapter_number] = future.result()
                except Exception as exc:
                    self._log(
                        "structuring_chapter_failed",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise RuntimeError(
                        f"Structuring failed for chapter {chapter_number} ({chapter_title}): {exc}"
                    ) from exc
                if structured_results[chapter_number] is None:
                    self._log(
                        "structuring_chapter_skipped",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                    )
                else:
                    self._log(
                        "structuring_chapter_completed",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                    )
        for chapter_number, _, _ in chapter_inputs:
            structured_chapter = structured_results[chapter_number]
            if structured_chapter is None:
                continue
            chapter_id = f"{ingestion.book_id}-chapter-{chapter_number}"
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
    ) -> StructuredChapter | None:
        self._log(
            "structuring_chapter_started",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_word_count=len(chapter_body.split()),
        )
        chapter_words = len(chapter_body.split())
        if chapter_words > self.max_structuring_llm_chapter_words:
            self._log(
                "structuring_chapter_skipped_llm",
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chapter_word_count=chapter_words,
                max_llm_chapter_words=self.max_structuring_llm_chapter_words,
            )
            return None
        if chapter_words <= self.max_structuring_chapter_words:
            draft = _build_structured_chapter_draft(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chapter_body=chapter_body,
                max_chunk_words=self.max_chunk_words,
                overlap_words=self.chunk_overlap_words,
                body_start_word=0,
                body_start_char=0,
            )
            return self._structure_payload_with_retry(
                payload={"draft": draft.model_dump(mode="python")},
                draft=draft,
                source_text=chapter_body,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                body_start_word=0,
                body_start_char=0,
                context_label="chapter",
            )

        windows = _window_ranges(
            total_words=chapter_words,
            window_words=self.structuring_window_words,
            overlap_words=self.structuring_window_overlap_words,
        )
        window_outputs = []
        self._log(
            "structuring_window_fallback",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            window_count=len(windows),
        )
        window_parallelism = min(len(windows), max(1, min(self.structuring_parallelism, 3)))
        self._log(
            "structuring_window_schedule",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            window_count=len(windows),
            concurrency=window_parallelism,
        )
        with ThreadPoolExecutor(max_workers=window_parallelism) as executor:
            future_map = {
                executor.submit(
                    self._structure_window,
                    chapter_number,
                    chapter_title,
                    chapter_body,
                    window_index,
                    start_word,
                    end_word,
                ): window_index
                for window_index, (start_word, end_word) in enumerate(windows, start=1)
            }
            for future in as_completed(future_map):
                window_outputs.append((future_map[future], future.result()))
        window_outputs.sort(key=lambda item: item[0])
        return _merge_structured_chapter_windows(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_body=chapter_body,
            windows=window_outputs,
        )

    def _structure_window(
        self,
        chapter_number: int,
        chapter_title: str,
        chapter_body: str,
        window_index: int,
        start_word: int,
        end_word: int,
    ) -> StructuredChapter:
        payload = self.build_window_payload(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_body=chapter_body,
            window_start_word=start_word,
            window_end_word=end_word,
        )
        draft = StructuredChapter.model_validate(payload["draft"])
        window_text, window_start_char, _ = _slice_words(
            chapter_body,
            start_word,
            end_word,
        )
        return self._structure_payload_with_retry(
            payload=payload,
            draft=draft,
            source_text=window_text,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            body_start_word=start_word,
            body_start_char=window_start_char,
            context_label=f"window-{window_index}",
        )

    def _log(self, event_type: str, **payload: object) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is not None:
            run_logger.log(event_type, **payload)

    def _structure_payload_with_retry(
        self,
        *,
        payload: dict,
        draft: StructuredChapter,
        source_text: str,
        chapter_number: int,
        chapter_title: str,
        body_start_word: int,
        body_start_char: int,
        context_label: str,
    ) -> StructuredChapter:
        retry_instructions = (
            f"{self.instructions} "
            "Do not repeat source prose. Return valid JSON only with chapter metadata and chunk boundaries."
        )
        attempts = [
            ("initial", self.instructions),
            ("retry", retry_instructions),
        ]
        last_error: Exception | None = None
        for attempt_label, instructions in attempts:
            try:
                plan = self.llm.generate_json(
                    schema_name=self.schema_name,
                    instructions=instructions,
                    payload=payload,
                    response_model=self.response_model,
                )
                return _rebuild_structured_chapter(
                    plan=plan,
                    source_text=source_text,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    fallback_summary=draft.summary,
                    fallback_chunks=draft.chunks,
                    body_start_word=body_start_word,
                    body_start_char=body_start_char,
                )
            except (JSONDecodeError, LLMContentFilterError, ValueError) as exc:
                last_error = exc
                event_type = "structuring_llm_content_filter" if isinstance(exc, LLMContentFilterError) else "structuring_llm_retry"
                self._log(
                    event_type,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    context=context_label,
                    attempt=attempt_label,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if attempt_label == "retry":
                    break
        self._log(
            "structuring_llm_fallback",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            context=context_label,
            fallback_reason=type(last_error).__name__ if last_error is not None else "unknown",
        )
        return draft


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
            key = (chunk.start_word, chunk.end_word, tuple(chunk.source_offsets))
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


def _find_section_index(sections: list[object], requested_title: str, bound_name: str) -> int:
    normalized_requested_title = requested_title.strip().casefold()
    for index, section in enumerate(sections):
        section_title = getattr(section, "title", "").strip().casefold()
        if section_title == normalized_requested_title:
            return index
    available_titles = ", ".join(getattr(section, "title", "") for section in sections)
    raise ValueError(
        f"Unable to find {bound_name} chapter '{requested_title}'. "
        f"Available detected chapters: {available_titles}"
    )


def _rebuild_structured_chapter(
    *,
    plan: StructuredChapterPlan,
    source_text: str,
    chapter_number: int,
    chapter_title: str,
    fallback_summary: str,
    fallback_chunks: list[StructuredChunkDraft],
    body_start_word: int,
    body_start_char: int,
) -> StructuredChapter:
    word_count = len(source_text.split())
    previous_start = body_start_word
    previous_end = body_start_word
    rebuilt_chunks: list[StructuredChunkDraft] = []
    for chunk in plan.chunks:
        if chunk.start_word < body_start_word or chunk.end_word > body_start_word + word_count:
            raise ValueError("LLM returned chunk boundaries outside the available source window.")
        if chunk.start_word >= chunk.end_word:
            raise ValueError("LLM returned an empty or inverted chunk range.")
        if chunk.start_word < previous_start or chunk.end_word < previous_end:
            raise ValueError("LLM returned chunk ranges out of order.")
        local_start_word = chunk.start_word - body_start_word
        local_end_word = chunk.end_word - body_start_word
        chunk_text, start_char, end_char = _slice_words(source_text, local_start_word, local_end_word)
        if not chunk_text.strip():
            raise ValueError("LLM returned a chunk range that produced empty source text.")
        rebuilt_chunks.append(
            StructuredChunkDraft(
                text=chunk_text,
                start_word=chunk.start_word,
                end_word=chunk.end_word,
                source_offsets=[body_start_char + start_char, body_start_char + end_char],
                themes=chunk.themes or _infer_themes(chunk_text),
            )
        )
        previous_start = chunk.start_word
        previous_end = chunk.end_word
    if not rebuilt_chunks:
        rebuilt_chunks = fallback_chunks
    return StructuredChapter(
        chapter_number=plan.chapter_number or chapter_number,
        title=(plan.title or chapter_title).strip(),
        summary=(plan.summary or fallback_summary).strip(),
        chunks=rebuilt_chunks,
    )


def _infer_themes(text: str) -> list[str]:
    terms = [
        token.lower().strip(" ,.;:!?()[]{}\"'")
        for token in text.split()
        if len(token) >= 6
    ]
    filtered = [term for term in terms if term.isalpha() and term not in STOP_WORDS]
    return [theme for theme, _ in Counter(filtered).most_common(4)]
