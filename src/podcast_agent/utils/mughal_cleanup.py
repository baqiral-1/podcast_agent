"""One-off cleanup helpers for newly downloaded Mughal history books."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfReader

from podcast_agent.utils.book_cleaning import (
    _body_has_prose,
    _body_start_index,
    _drop_noise_lines,
    _drop_preface_toc_blocks,
    _is_noise_line,
    _looks_like_multiline_numbered_heading,
    _looks_like_sentence,
    _looks_like_title_line,
    _next_nonblank_index,
    _normalize_lines,
    derive_output_filename,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKIPPABLE_SECTION_RE = re.compile(
    r"^(preface|acknowledg(?:e)?ments|abbreviations?|plates?|maps?|endpapers|"
    r"dedication|copyright|title page|cover|cast of characters|"
    r"list of illustrations|illustrations|list of maps|members of .+|"
    r"the publishers and the author wish to thank)\b",
    re.IGNORECASE,
)
_TERMINAL_SECTION_RE = re.compile(
    r"^(notes|source notes|references|bibliography|glossary|index|"
    r"image credits|photo credits|illustration credits|permissions|"
    r"appendix(?:\b| [a-z0-9]+)|appendices)\b",
    re.IGNORECASE,
)
_NARRATIVE_NAMED_SECTION_RE = re.compile(
    r"^(introduction|prologue|conclusion|epilogue|afterword|postscript)\b",
    re.IGNORECASE,
)
_SCAN_ATTRIBUTION_RE = re.compile(
    r"(digitized by|original from|internet archive|archive\.org|funding from|"
    r"google|university of california|kahle/austin foundation)",
    re.IGNORECASE,
)
_INLINE_ATTRIBUTION_RE = re.compile(
    r"^(note: this .{0,80} recited|english rendering by|translated by)\b",
    re.IGNORECASE,
)
_PAGE_FRONT_MATTER_RE = re.compile(
    r"^(isbn\b|this edition\b|first published\b|published by\b|printed at\b|"
    r"all rights reserved\b|contents\b|table of contents\b|list of plates\b|"
    r"list of illustrations\b|frontispiece\b)",
    re.IGNORECASE,
)
_CHAPTER_RENDER_RE = re.compile(r"^Chapter \d+:$", re.MULTILINE)
_CONCLUSION_SECTION_RE = re.compile(r"^(conclusion|epilogue|afterword|postscript)\b", re.IGNORECASE)
_FRONT_MATTER_PARAGRAPH_RE = re.compile(
    r"(oxford university press|cambridge university press|all rights reserved|retrieval system|"
    r"permission granted|permission of|notes on contributors|the moral rights of the author|"
    r"copyright|rights department|editor and the publisher|reproduce the essays|generous permission)",
    re.IGNORECASE,
)
_SHORT_CITATION_PARAGRAPH_RE = re.compile(
    r"^(p\.|pp\.|vol\.|no\.|ibid\.|seen\.|sect\.|journal of|university of)\b",
    re.IGNORECASE,
)
_CHAPTER_WORD_RE = re.compile(
    r"^(chapter\s+)?(?P<number>\d{1,2}|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
    re.IGNORECASE,
)
_TOC_ENTRY_RE = re.compile(
    r"^(?:(?:chapter|chap\.)\s+)?(?P<number>\d{1,2}|[ivxlcdm]+)[.)]?\s+(?P<title>.+?)\s+(?P<page>\d{1,3})$",
    re.IGNORECASE,
)
_TOC_TITLE_RE = re.compile(
    r"^(?:(?:chapter|chap\.)\s+)?(?P<number>\d{1,2}|[ivxlcdm]+)[.)]?\s+(?P<title>.+?)$",
    re.IGNORECASE,
)
_REFERENCE_TAIL_RE = re.compile(
    r"(epitaph|inscription|printed books|folio|calcutta|bib\. ind\.|vol\.\s*[ivxlcdm0-9]|"
    r"translation\)|op\. cit\.|same to same|letters\)|deposi?t|home poll\.|confidential|"
    r"education,|commerce|general \(old\)|funeral ceremony|gazetteer|8vo\.|4to\.|12mo\.)",
    re.IGNORECASE,
)
_GENEALOGICAL_FRAGMENT_RE = re.compile(
    r"\b(mother|daughter|son|begam|rajah|nawab bai|was born|born on|died at|died there)\b",
    re.IGNORECASE,
)
_OPENING_ARTIFACT_RE = re.compile(r"^\[\s+")
_SECTION_SUBHEADING_RE = re.compile(r"^sec(?:tion)?\.?\s*[ivxlcdm0-9]+(?:\s*[-.:|]|$)", re.IGNORECASE)
_EMBEDDED_TERMINAL_MARKER_RE = re.compile(
    r"(?P<prefix>.*?)(?:[.!?]\s+|\s+)(?P<marker>NOTES|BIBLIOGRAPHY|REFERENCES|INDEX)\b",
    re.DOTALL,
)


@dataclass(frozen=True)
class MughalCleanResult:
    source_path: Path
    output_path: Path
    chapter_count: int
    word_count: int
    extraction_method: str


@dataclass(frozen=True)
class MughalExtractedText:
    raw_text: str
    page_texts: list[str]
    extraction_method: str


@dataclass(frozen=True)
class _TocEntry:
    number: int
    title: str
    page_number: int | None


@dataclass(frozen=True)
class _ChapterStart:
    page_index: int
    body_start_index: int
    number_hint: int | None = None


def extract_mughal_book_text(path: Path) -> tuple[str, str]:
    """Extract raw book text from a PDF with OCR fallback."""

    extracted = extract_mughal_book_source(path)
    return extracted.raw_text, extracted.extraction_method


def extract_mughal_book_source(path: Path) -> MughalExtractedText:
    """Extract raw book text and per-page text from a PDF with OCR fallback."""

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported source format: {path}")

    with TemporaryDirectory(prefix="mughal-clean-", dir=_REPO_ROOT) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        working_path = path
        methods: list[str] = []

        if _is_pdf_encrypted(path):
            working_path = _decrypt_pdf(path, tmp_dir)
            methods.append("qpdf")

        text = _extract_with_pdftotext(working_path, tmp_dir)
        if text.strip():
            methods.append("pdftotext")
            return MughalExtractedText(
                raw_text=text,
                page_texts=_split_text_pages(text),
                extraction_method="+".join(methods),
            )

        page_texts = _extract_pages_with_pypdf(working_path)
        text = "\n\n".join(page_texts)
        if text.strip():
            methods.append("pypdf")
            return MughalExtractedText(
                raw_text=text,
                page_texts=page_texts,
                extraction_method="+".join(methods),
            )

        page_texts = _extract_pages_with_ocr(working_path, tmp_dir)
        text = "\n\n".join(page_texts)
        if text.strip():
            methods.append("ocr")
            return MughalExtractedText(
                raw_text=text,
                page_texts=page_texts,
                extraction_method="+".join(methods),
            )

    raise ValueError(f"Unable to extract text from PDF: {path}")


def clean_mughal_book_text(raw_text: str) -> str:
    """Reduce raw extracted text to chapter-only narrative prose."""

    normalized_lines = _normalize_lines(raw_text)
    filtered_lines = _drop_noise_lines(normalized_lines)
    filtered_lines = _drop_scan_attributions(filtered_lines)
    filtered_lines = _drop_preface_toc_blocks(filtered_lines)
    sections = _extract_sections(filtered_lines)
    if not sections:
        raise ValueError("No chapter-like sections found in extracted text.")

    rendered_bodies: list[str] = []
    for _, body in sections:
        cleaned_body = _collapse_section_body(body)
        if not cleaned_body:
            continue
        rendered_bodies.append(cleaned_body)

    return _render_chapter_bodies(rendered_bodies)


def clean_mughal_book_file(source_path: Path, output_dir: Path) -> MughalCleanResult:
    """Extract, clean, and write a single Mughal-history book."""

    extracted = extract_mughal_book_source(source_path)
    cleaned_text = clean_mughal_pages(extracted.page_texts, fallback_text=extracted.raw_text)
    chapter_count = len(_CHAPTER_RENDER_RE.findall(cleaned_text))
    word_count = len(re.findall(r"\b\w+\b", cleaned_text))
    if word_count < 2000:
        raise ValueError(
            f"Cleaned output is too sparse to trust ({word_count} words extracted after cleanup)."
        )

    output_path = output_dir / derive_output_filename(source_path)
    output_path.write_text(cleaned_text, encoding="utf-8")
    return MughalCleanResult(
        source_path=source_path,
        output_path=output_path,
        chapter_count=chapter_count,
        word_count=word_count,
        extraction_method=extracted.extraction_method,
    )


def clean_mughal_pages(page_texts: list[str], fallback_text: str | None = None) -> str:
    """Reduce extracted page texts to chapter-only narrative prose."""

    page_infos = [_prepare_page(page_text) for page_text in page_texts]
    chapter_starts = _find_chapter_starts(page_infos)
    if chapter_starts:
        cleaned_text = _render_page_chapters(page_infos, chapter_starts)
        if cleaned_text:
            return cleaned_text

    if _has_running_headers(page_infos):
        return clean_mughal_ocr_pages(page_texts)

    if fallback_text is None:
        fallback_text = "\n\n".join(page_texts)
    return clean_mughal_book_text(fallback_text)


def clean_mughal_ocr_pages(page_texts: list[str]) -> str:
    """Reduce OCR page texts to chapter-only narrative prose."""

    page_infos = [_prepare_page(page_text) for page_text in page_texts]
    header_counts = Counter(info.header_key for info in page_infos if info.header_key)
    repeated_headers = {header for header, count in header_counts.items() if count >= 2}
    if not repeated_headers:
        return clean_mughal_book_text("\n\n".join(page_texts))

    book_header = header_counts.most_common(1)[0][0]
    current_title: str | None = None
    current_pages: list[list[str]] = []
    chapters: list[list[str]] = []
    started = False

    for info in page_infos:
        if info.is_terminal:
            if started:
                break
            continue
        if info.is_skippable:
            continue
        if not info.body_lines or not _section_has_narrative(info.body_lines):
            continue

        chapter_header = info.header_key if info.header_key in repeated_headers and info.header_key != book_header else None
        if not started and chapter_header is None:
            continue
        if chapter_header and chapter_header != current_title:
            if current_pages:
                chapters.append(_join_page_lines(current_pages))
            current_title = chapter_header
            current_pages = [info.body_lines]
            started = True
            continue

        if started:
            current_pages.append(info.body_lines)

    if current_pages:
        chapters.append(_join_page_lines(current_pages))

    rendered_bodies: list[str] = []
    for chapter_lines in chapters:
        cleaned_body = _collapse_section_body(chapter_lines)
        if not cleaned_body:
            continue
        rendered_bodies.append(cleaned_body)

    return _render_chapter_bodies(rendered_bodies)


def _extract_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    heading_indexes = [index for index in range(len(lines)) if _is_section_heading_at(lines, index)]
    if not heading_indexes:
        return []

    sections: list[tuple[str, list[str]]] = []
    started = False
    for position, heading_index in enumerate(heading_indexes):
        heading = lines[heading_index]
        next_heading_index = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
        body_start_index = _section_body_start(lines, heading_index, next_heading_index)
        body_lines = lines[body_start_index:next_heading_index]

        if _TERMINAL_SECTION_RE.match(heading):
            if started:
                break
            continue
        if _SKIPPABLE_SECTION_RE.match(heading):
            continue
        if not _is_narrative_section_heading(lines, heading_index):
            continue
        if not _section_has_narrative(body_lines):
            continue

        started = True
        sections.append((heading, body_lines))

    return sections


def _render_chapter_bodies(bodies: list[str]) -> str:
    finalized_bodies = _finalize_chapter_bodies(bodies)
    if not finalized_bodies:
        raise ValueError("Chapter extraction produced no usable narrative text.")

    rendered_sections = [
        f"Chapter {index}:\n{body}" for index, body in enumerate(finalized_bodies, start=1)
    ]
    return "\n\n".join(rendered_sections).strip() + "\n"


def _finalize_chapter_bodies(bodies: list[str]) -> list[str]:
    finalized: list[str] = []
    for index, body in enumerate(bodies):
        normalized = _normalize_chapter_body(body)
        if not normalized:
            continue
        if index == 0 and _looks_like_spurious_opening_section(normalized):
            continue
        finalized.append(normalized)

    while finalized and _looks_like_spurious_trailing_section(finalized[-1]):
        finalized.pop()
    return finalized


def _normalize_chapter_body(body: str) -> str:
    paragraphs = [_truncate_embedded_terminal_block(paragraph) for paragraph in body.split("\n\n")]
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    if not paragraphs:
        return ""

    paragraphs[0] = _normalize_opening_paragraph(paragraphs[0])
    while paragraphs and _looks_like_spurious_opening_paragraph(paragraphs[0]):
        paragraphs.pop(0)
        if paragraphs:
            paragraphs[0] = _normalize_opening_paragraph(paragraphs[0])

    while paragraphs and _looks_like_spurious_trailing_paragraph(paragraphs[-1]):
        paragraphs.pop()
    tail_start = _find_note_heavy_tail_start(paragraphs)
    if tail_start is not None:
        paragraphs = paragraphs[:tail_start]

    return "\n\n".join(paragraphs).strip()


def _normalize_opening_paragraph(paragraph: str) -> str:
    normalized = paragraph.strip()
    normalized = _OPENING_ARTIFACT_RE.sub("", normalized)
    normalized = re.sub(r"^[A-Z][a-z]+ (?=imperial Government\b)", "", normalized)
    normalized = normalized.lstrip(" -–—")
    if normalized.startswith("imperial "):
        normalized = "Imperial " + normalized[len("imperial ") :]
    if normalized.startswith("Imperial Government which"):
        normalized = "The " + normalized[0].lower() + normalized[1:]
    if re.match(r"^\d{4}\b", normalized):
        normalized = f"In {normalized}"
    return normalized.strip()


def _truncate_embedded_terminal_block(paragraph: str) -> str:
    stripped = paragraph.strip()
    match = _EMBEDDED_TERMINAL_MARKER_RE.search(stripped)
    if match is None:
        return paragraph
    prefix = stripped[: match.start("marker")].rstrip()
    return prefix if len(prefix.split()) >= 10 else ""


def _find_chapter_starts(page_infos: list[_PreparedPage]) -> list[_ChapterStart]:
    explicit_starts: list[_ChapterStart] = []
    for page_index, page_info in enumerate(page_infos):
        start = _detect_explicit_chapter_start(page_info)
        if start is None:
            continue
        explicit_starts.append(
            _ChapterStart(
                page_index=page_index,
                body_start_index=start.body_start_index,
                number_hint=start.number_hint,
            )
        )
    if len(explicit_starts) >= 2:
        return _dedupe_chapter_starts(explicit_starts)

    toc_entries = _extract_toc_entries(page_infos)
    if not toc_entries:
        return _dedupe_chapter_starts(explicit_starts)

    page_offset = _infer_page_number_offset(page_infos)
    toc_starts = _locate_toc_chapter_starts(page_infos, toc_entries, page_offset)
    if explicit_starts:
        first_explicit_page = min(start.page_index for start in explicit_starts)
        toc_starts = [start for start in toc_starts if start.page_index >= first_explicit_page]
    combined = explicit_starts + toc_starts
    return _dedupe_chapter_starts(combined)


def _locate_toc_chapter_starts(
    page_infos: list[_PreparedPage],
    toc_entries: list[_TocEntry],
    page_offset: int | None,
) -> list[_ChapterStart]:
    starts: list[_ChapterStart] = []
    for entry in toc_entries:
        predicted_index = (
            entry.page_number + page_offset - 1
            if entry.page_number is not None and page_offset is not None
            else None
        )
        start = _best_toc_start(page_infos, entry, predicted_index)
        if start is not None:
            starts.append(start)
    return starts


def _best_toc_start(
    page_infos: list[_PreparedPage],
    entry: _TocEntry,
    predicted_index: int | None,
) -> _ChapterStart | None:
    if predicted_index is None:
        search_range = range(len(page_infos))
    else:
        low = max(0, predicted_index - 2)
        high = min(len(page_infos), predicted_index + 3)
        search_range = range(low, high)

    best_score = 0
    best_start: _ChapterStart | None = None
    for page_index in search_range:
        page_info = page_infos[page_index]
        if page_info.is_skippable or page_info.is_terminal:
            continue
        score, body_start_index = _score_toc_start_candidate(page_info, entry, page_index, predicted_index)
        if score > best_score and body_start_index is not None:
            best_score = score
            best_start = _ChapterStart(
                page_index=page_index,
                body_start_index=body_start_index,
                number_hint=entry.number,
            )

    return best_start if best_score >= 3 else None


def _score_toc_start_candidate(
    page_info: _PreparedPage,
    entry: _TocEntry,
    page_index: int,
    predicted_index: int | None,
) -> tuple[int, int | None]:
    lines = page_info.body_lines
    if not lines:
        return 0, None

    explicit_start = _detect_explicit_chapter_start(page_info)
    if explicit_start is not None and explicit_start.number_hint == entry.number:
        return 8, explicit_start.body_start_index

    top_text = _normalized_for_match(" ".join(line for line in lines[:6] if line))
    title_score = _title_match_score(top_text, entry.title)
    if title_score:
        body_start_index = _find_body_start_after_top_titles(lines)
        return 5 + title_score, body_start_index

    visible_page = _extract_visible_page_number(lines)
    score = 0
    if entry.page_number is not None and visible_page == entry.page_number:
        score += 2
    if predicted_index is not None and abs(page_index - predicted_index) <= 1:
        score += 1
    if score and _section_has_narrative(lines):
        return score + 1, _find_body_start_after_top_titles(lines)
    return 0, None


def _extract_toc_entries(page_infos: list[_PreparedPage]) -> list[_TocEntry]:
    entries_by_number: dict[int, _TocEntry] = {}
    for page_info in page_infos[:40]:
        lines = page_info.raw_lines
        if not lines:
            continue
        page_candidates: list[_TocEntry] = []
        for line in lines:
            if not line:
                continue
            compact = re.sub(r"\.{2,}", " ", line)
            compact = re.sub(r"\s+", " ", compact).strip(" .\t")
            match = _TOC_ENTRY_RE.match(compact)
            page_number: int | None = None
            if match is None:
                match = _TOC_TITLE_RE.match(compact)
            else:
                page_number = int(match.group("page"))
            if match is None:
                continue
            title = match.group("title").strip(" .")
            if len(title.split()) < 2:
                continue
            number = _parse_chapter_number(match.group("number"))
            if number is None:
                continue
            page_candidates.append(_TocEntry(number=number, title=title, page_number=page_number))

        if not (
            page_candidates
            and (
                _page_looks_like_toc(lines)
                or _PAGE_FRONT_MATTER_RE.match(lines[0])
                or len(page_candidates) >= 2
            )
        ):
            continue

        for entry in page_candidates:
            if entry.number in entries_by_number:
                if entries_by_number[entry.number].page_number is None and entry.page_number is not None:
                    entries_by_number[entry.number] = entry
                continue
            entries_by_number[entry.number] = entry
    return [entries_by_number[number] for number in sorted(entries_by_number)]


def _infer_page_number_offset(page_infos: list[_PreparedPage]) -> int | None:
    offsets: Counter[int] = Counter()
    for page_index, page_info in enumerate(page_infos, start=1):
        if page_info.is_skippable or page_info.is_terminal:
            continue
        visible_page = _extract_visible_page_number(page_info.body_lines)
        if visible_page is None or visible_page <= 0:
            continue
        offsets[page_index - visible_page] += 1

    if not offsets:
        return None
    offset, count = offsets.most_common(1)[0]
    return offset if count >= 3 else None


def _dedupe_chapter_starts(starts: list[_ChapterStart]) -> list[_ChapterStart]:
    deduped: list[_ChapterStart] = []
    for start in sorted(starts, key=lambda item: item.page_index):
        if deduped and deduped[-1].page_index == start.page_index:
            existing = deduped[-1]
            if existing.number_hint is None and start.number_hint is not None:
                deduped[-1] = start
            continue
        deduped.append(start)
    return deduped


def _render_page_chapters(page_infos: list[_PreparedPage], chapter_starts: list[_ChapterStart]) -> str:
    rendered_bodies: list[str] = []
    for index, start in enumerate(chapter_starts):
        end_page = chapter_starts[index + 1].page_index if index + 1 < len(chapter_starts) else len(page_infos)
        chapter_lines: list[str] = []
        for page_index in range(start.page_index, end_page):
            page_info = page_infos[page_index]
            if page_info.is_terminal and page_index > start.page_index:
                break
            lines = page_info.body_lines[start.body_start_index:] if page_index == start.page_index else page_info.body_lines
            if page_index > start.page_index and _starts_terminal_named_section(lines):
                break
            filtered_lines = _drop_page_artifacts(lines)
            if not filtered_lines:
                continue
            if chapter_lines and chapter_lines[-1]:
                chapter_lines.append("")
            chapter_lines.extend(filtered_lines)

        cleaned_body = _collapse_section_body(chapter_lines)
        if not cleaned_body:
            continue
        rendered_bodies.append(cleaned_body)

    if not rendered_bodies:
        return ""
    return _render_chapter_bodies(rendered_bodies)


def _drop_scan_attributions(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        if _SCAN_ATTRIBUTION_RE.search(line):
            continue
        cleaned.append(line)
        previous_blank = False
    return cleaned


def _is_section_heading_at(lines: list[str], index: int) -> bool:
    line = lines[index]
    if not line:
        return False
    return bool(
        _looks_like_multiline_numbered_heading(lines, index)
        or _SKIPPABLE_SECTION_RE.match(line)
        or _TERMINAL_SECTION_RE.match(line)
        or _NARRATIVE_NAMED_SECTION_RE.match(line)
    )


def _is_narrative_section_heading(lines: list[str], index: int) -> bool:
    line = lines[index]
    return bool(
        _looks_like_multiline_numbered_heading(lines, index) or _NARRATIVE_NAMED_SECTION_RE.match(line)
    )


def _section_body_start(lines: list[str], heading_index: int, next_heading_index: int) -> int:
    if _looks_like_multiline_numbered_heading(lines, heading_index):
        return _body_start_index(lines, heading_index, next_heading_index)

    body_start_index = heading_index + 1
    title_index = _next_nonblank_index(lines, body_start_index, next_heading_index)
    if title_index is not None and _looks_like_title_line(lines[title_index]):
        return title_index + 1
    return body_start_index


def _section_has_narrative(body_lines: list[str]) -> bool:
    candidates = [line for line in body_lines if line and not _line_should_be_dropped(line)]
    if not candidates:
        return False
    return _body_has_prose(candidates, min_total_words=6)


def _collapse_section_body(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current_lines: list[str] = []
    for line in lines:
        if not line:
            _flush_paragraph(current_lines, paragraphs)
            current_lines = []
            continue
        if _line_should_be_dropped(line):
            continue
        current_lines.append(line)

    _flush_paragraph(current_lines, paragraphs)
    paragraphs = _trim_edge_paragraphs(paragraphs)
    return "\n\n".join(paragraphs).strip()


def _flush_paragraph(current_lines: list[str], paragraphs: list[str]) -> None:
    if not current_lines:
        return
    paragraph = " ".join(current_lines).strip()
    paragraph = _truncate_embedded_terminal_block(paragraph).strip()
    if not paragraph:
        return
    if _paragraph_should_be_dropped(paragraph):
        return
    paragraphs.append(paragraph)


def _trim_edge_paragraphs(paragraphs: list[str]) -> list[str]:
    trimmed = list(paragraphs)
    while trimmed and (_FRONT_MATTER_PARAGRAPH_RE.search(trimmed[0]) or _looks_like_toc_paragraph(trimmed[0])):
        trimmed.pop(0)
    note_block_start: int | None = None
    note_count = 0
    for index in range(len(trimmed) - 1, -1, -1):
        if _looks_like_trailing_note_paragraph(trimmed[index]):
            note_block_start = index
            note_count += 1
            continue
        break
    if note_block_start is not None and note_count >= 3:
        trimmed = trimmed[:note_block_start]
    while trimmed and _looks_like_trailing_note_paragraph(trimmed[-1]):
        trimmed.pop()
    return trimmed


def _line_should_be_dropped(line: str) -> bool:
    if _is_noise_line(line):
        return True
    if _looks_like_title_line(line):
        return True
    if _SCAN_ATTRIBUTION_RE.search(line) or _INLINE_ATTRIBUTION_RE.match(line):
        return True
    if _is_non_latin_artifact(line):
        return True
    return False


def _paragraph_should_be_dropped(paragraph: str) -> bool:
    if _SCAN_ATTRIBUTION_RE.search(paragraph) or _INLINE_ATTRIBUTION_RE.match(paragraph):
        return True
    if _FRONT_MATTER_PARAGRAPH_RE.search(paragraph):
        return True
    if _looks_like_note_paragraph(paragraph):
        return True
    if _looks_like_toc_paragraph(paragraph):
        return True
    if _SECTION_SUBHEADING_RE.match(paragraph) and len(paragraph.split()) <= 20:
        return True
    if len(paragraph.split()) <= 10 and _SHORT_CITATION_PARAGRAPH_RE.match(paragraph):
        return True
    if _is_non_latin_artifact(paragraph):
        return True
    if not _looks_like_sentence(paragraph) and len(paragraph.split()) <= 6:
        return True
    return False


def _is_non_latin_artifact(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin_letters = sum(1 for char in letters if "LATIN" in unicodedata.name(char, ""))
    return latin_letters / len(letters) < 0.35


def _is_pdf_encrypted(path: Path) -> bool:
    reader = PdfReader(str(path))
    return bool(reader.is_encrypted)


def _decrypt_pdf(path: Path, tmp_dir: Path) -> Path:
    decrypted_path = tmp_dir / f"{path.stem}.decrypted.pdf"
    result = subprocess.run(
        ["qpdf", "--password=", "--decrypt", str(path), str(decrypted_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not decrypted_path.exists():
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"qpdf failed for '{path.name}': {message}")
    return decrypted_path


def _extract_with_pdftotext(path: Path, tmp_dir: Path) -> str:
    txt_path = tmp_dir / f"{path.stem}.pdftotext.txt"
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(path), str(txt_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8", errors="ignore")


def _extract_pages_with_pypdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        return []
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _extract_pages_with_ocr(path: Path, tmp_dir: Path) -> list[str]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    max_workers = min(4, page_count)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda page_number: _ocr_pdf_page(path, tmp_dir, page_number), range(1, page_count + 1)))


def _ocr_pdf_page(path: Path, tmp_dir: Path, page_number: int) -> str:
    image_base = tmp_dir / f"ocr_page_{page_number:04d}"
    image_path = image_base.with_suffix(".jpg")
    render = subprocess.run(
        [
            "pdftoppm",
            "-singlefile",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            "150",
            "-jpeg",
            str(path),
            str(image_base),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if render.returncode != 0 or not image_path.exists():
        message = (render.stderr or render.stdout).strip()
        raise RuntimeError(f"pdftoppm failed for '{path.name}' page {page_number}: {message}")

    ocr = subprocess.run(
        ["tesseract", str(image_path), "stdout", "--psm", "6"],
        capture_output=True,
        check=False,
        text=True,
    )
    image_path.unlink(missing_ok=True)
    if ocr.returncode != 0:
        message = (ocr.stderr or ocr.stdout).strip()
        raise RuntimeError(f"tesseract failed for '{path.name}' page {page_number}: {message}")
    return ocr.stdout.strip()


@dataclass(frozen=True)
class _PreparedPage:
    raw_lines: list[str]
    header_key: str | None
    body_lines: list[str]
    is_skippable: bool
    is_terminal: bool
    word_count: int


def _prepare_page(page_text: str) -> _PreparedPage:
    lines = _drop_scan_attributions(_drop_noise_lines(_normalize_lines(page_text)))
    first_line = next((line for line in lines if line), "")
    header_key = _page_header_key(first_line)
    body_lines = list(lines)

    if header_key and body_lines and _page_header_key(body_lines[0]) == header_key:
        body_lines = body_lines[1:]
        while body_lines and not body_lines[0]:
            body_lines = body_lines[1:]

    first_body_line = next((line for line in body_lines if line), "")
    return _PreparedPage(
        raw_lines=lines,
        header_key=header_key,
        body_lines=body_lines,
        is_skippable=bool(
            first_body_line
            and (
                _SKIPPABLE_SECTION_RE.match(first_body_line)
                or _PAGE_FRONT_MATTER_RE.match(first_body_line)
                or _page_looks_like_toc(body_lines)
            )
        ),
        is_terminal=bool(
            first_body_line
            and (_TERMINAL_SECTION_RE.match(first_body_line) or _CONCLUSION_SECTION_RE.match(first_body_line))
        ),
        word_count=len(re.findall(r"\b\w+\b", " ".join(body_lines))),
    )


def _page_header_key(line: str) -> str | None:
    if not line:
        return None
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return None
    uppercase_letters = sum(1 for char in letters if char.isupper())
    if uppercase_letters / len(letters) < 0.75:
        return None
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", line)
    if not 2 <= len(words) <= 10:
        return None
    if _SKIPPABLE_SECTION_RE.match(line) or _TERMINAL_SECTION_RE.match(line):
        return None
    if _INLINE_ATTRIBUTION_RE.match(line) or _SCAN_ATTRIBUTION_RE.search(line):
        return None
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]+", " ", line.upper())).strip()


def _join_page_lines(page_groups: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for group in page_groups:
        if lines and lines[-1]:
            lines.append("")
        lines.extend(group)
    return lines


def _page_looks_like_toc(lines: list[str]) -> bool:
    nonblank = [line for line in lines if line][:12]
    if len(nonblank) < 4:
        return False
    toc_like = 0
    for line in nonblank:
        compact = re.sub(r"\s+", " ", line).strip()
        if re.fullmatch(r"[IVXLC]+\.?\s+.+\s+\d{1,3}", compact, flags=re.IGNORECASE):
            toc_like += 1
            continue
        if re.fullmatch(r"(appendix|chapter)\s+[a-z0-9ivxlc]+.*\d{1,3}", compact, flags=re.IGNORECASE):
            toc_like += 1
            continue
    return toc_like >= 3


def _has_running_headers(page_infos: list[_PreparedPage]) -> bool:
    header_counts = Counter(info.header_key for info in page_infos if info.header_key)
    return any(count >= 2 for count in header_counts.values())


def _detect_explicit_chapter_start(page_info: _PreparedPage) -> _ChapterStart | None:
    if page_info.is_skippable or page_info.is_terminal:
        return None

    lines = page_info.body_lines
    nonblank_positions = [index for index, line in enumerate(lines[:8]) if line]
    if not nonblank_positions:
        return None

    for first_index in nonblank_positions[:5]:
        first_line = lines[first_index]
        if _starts_terminal_named_section(lines[first_index:]):
            return None

        if first_line.lower().startswith("chapter"):
            chapter_number = _parse_chapter_number(first_line)
            body_start_index = _find_body_start_after_heading(lines, first_index)
            if chapter_number is not None and body_start_index is not None:
                return _ChapterStart(
                    page_index=-1,
                    body_start_index=body_start_index,
                    number_hint=chapter_number,
                )

        if not _is_standalone_chapter_marker(first_line):
            continue
        chapter_number = _parse_chapter_number(first_line)
        if chapter_number is None:
            continue
        if not _looks_like_chapter_banner(lines, first_index):
            continue
        body_start_index = _find_body_start_after_heading(lines, first_index)
        if body_start_index is None:
            continue
        return _ChapterStart(page_index=-1, body_start_index=body_start_index, number_hint=chapter_number)
    return None


def _looks_like_chapter_banner(lines: list[str], first_index: int) -> bool:
    if first_index + 1 >= len(lines):
        return False
    title_candidates = 0
    for line in lines[first_index + 1 : first_index + 4]:
        if not line:
            break
        if _looks_like_sentence(line):
            break
        if _looks_like_title_line(line) or line == line.upper():
            title_candidates += 1
        else:
            break
    return title_candidates >= 2


def _is_standalone_chapter_marker(line: str) -> bool:
    compact = re.sub(r"[\W_]+", " ", line).strip()
    words = compact.split()
    if len(words) != 1:
        return False
    return _parse_chapter_number(words[0]) is not None


def _find_body_start_after_heading(lines: list[str], first_index: int) -> int | None:
    body_start_index = _find_body_start_after_top_titles(lines[first_index:])
    if body_start_index is None:
        return None
    return first_index + body_start_index


def _find_body_start_after_top_titles(lines: list[str]) -> int | None:
    nonblank_seen = 0
    for index, line in enumerate(lines):
        if not line:
            if nonblank_seen >= 1:
                continue
            continue
        nonblank_seen += 1
        if _looks_like_sentence(line):
            return index
        if nonblank_seen >= 4:
            return index
    return None


def _drop_page_artifacts(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and not trimmed[0]:
        trimmed = trimmed[1:]
    while trimmed and not trimmed[-1]:
        trimmed = trimmed[:-1]

    if trimmed and _is_visible_page_number_line(trimmed[0]):
        trimmed = trimmed[1:]
    if trimmed and _is_visible_page_number_line(trimmed[-1]):
        trimmed = trimmed[:-1]

    cleaned: list[str] = []
    for line in trimmed:
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if _page_header_key(line):
            continue
        cleaned.append(line)
    return cleaned


def _starts_terminal_named_section(lines: list[str]) -> bool:
    first_line = next((line for line in lines if line), "")
    return bool(first_line and (_TERMINAL_SECTION_RE.match(first_line) or _CONCLUSION_SECTION_RE.match(first_line)))


def _parse_chapter_number(text: str) -> int | None:
    match = _CHAPTER_WORD_RE.match(text.strip())
    if not match:
        return None
    token = match.group("number").lower()
    if token.isdigit():
        return int(token)
    if re.fullmatch(r"[ivxlcdm]+", token):
        return _roman_to_int(token)
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    return words.get(token)


def _roman_to_int(text: str) -> int | None:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for char in reversed(text.lower()):
        value = values.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _extract_visible_page_number(lines: list[str]) -> int | None:
    for line in lines[:3] + lines[-3:]:
        compact = re.sub(r"[^\d]", "", line)
        if compact and compact == line.strip():
            number = int(compact)
            if 0 < number < 1000:
                return number
    return None


def _is_visible_page_number_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return bool(re.fullmatch(r"\d{1,3}", compact))


def _normalized_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _title_match_score(top_text: str, title: str) -> int:
    title_words = [word for word in _normalized_for_match(title).split() if len(word) > 2]
    if len(title_words) < 2:
        return 0
    prefix = title_words[: min(4, len(title_words))]
    matched = sum(1 for word in prefix if word in top_text)
    return matched if matched >= max(2, len(prefix) - 1) else 0


def _looks_like_note_paragraph(paragraph: str) -> bool:
    if not re.match(r"^\d+\.\s+", paragraph):
        return False
    if _contains_citation_marker(paragraph):
        return True
    return paragraph.count(",") >= 2 and len(re.findall(r"\d+", paragraph)) >= 2


def _looks_like_toc_paragraph(paragraph: str) -> bool:
    if "notes on contributors" in paragraph.lower():
        return True
    return len(re.findall(r"\b\d{1,2}\s+[A-Z][A-Za-z'’-]+", paragraph)) >= 3


def _looks_like_trailing_note_paragraph(paragraph: str) -> bool:
    if _looks_like_note_paragraph(paragraph):
        return True
    if len(paragraph.split()) <= 14 and _SHORT_CITATION_PARAGRAPH_RE.match(paragraph):
        return True
    if re.match(r"^\d{4}[),.\s]", paragraph):
        return True
    if re.match(r"^\d+\.\s+", paragraph):
        return True
    if _contains_citation_marker(paragraph) and (
        len(paragraph.split()) <= 45 or re.search(r"\b(17|18|19)\d{2}\b", paragraph)
    ):
        return True
    if _REFERENCE_TAIL_RE.search(paragraph) and len(paragraph.split()) <= 90:
        return True
    return False


def _contains_citation_marker(paragraph: str) -> bool:
    return bool(
        re.search(
            r"\b(IOR|IOL|WBSA|BRP|BRC|Ibid\.|vol\.|pp?\.|fol\.|journal|to council|university of)\b",
            paragraph,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_spurious_opening_paragraph(paragraph: str) -> bool:
    if _looks_like_toc_paragraph(paragraph) or _FRONT_MATTER_PARAGRAPH_RE.search(paragraph):
        return True
    if _GENEALOGICAL_FRAGMENT_RE.search(paragraph) and len(paragraph.split()) <= 140:
        return True
    return False


def _looks_like_spurious_trailing_paragraph(paragraph: str) -> bool:
    if _looks_like_trailing_note_paragraph(paragraph):
        return True
    if _REFERENCE_TAIL_RE.search(paragraph):
        return True
    if re.search(r"\b(tomb|churchyard|monument|inscription|epitaph)\b", paragraph, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(london|calcutta|lahore|benares),\s*(17|18|19)\d{2}\b", paragraph, flags=re.IGNORECASE):
        return True
    if len(paragraph.split()) <= 20 and ("Like us" in paragraph or "human cares and woes" in paragraph):
        return True
    if paragraph.count("\n") >= 2 and len(paragraph.split()) <= 60:
        return True
    return False


def _looks_like_spurious_opening_section(body: str) -> bool:
    paragraphs = [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return True
    word_count = len(body.split())
    if word_count > 900:
        return False
    return all(_looks_like_spurious_opening_paragraph(paragraph) for paragraph in paragraphs[: min(3, len(paragraphs))])


def _looks_like_spurious_trailing_section(body: str) -> bool:
    paragraphs = [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return True
    word_count = len(body.split())
    if word_count > 900:
        return False
    tail = paragraphs[-min(3, len(paragraphs)) :]
    return all(_looks_like_spurious_trailing_paragraph(paragraph) for paragraph in tail)


def _find_note_heavy_tail_start(paragraphs: list[str]) -> int | None:
    for start in range(max(0, len(paragraphs) - 10), len(paragraphs)):
        tail = paragraphs[start:]
        if len(tail) < 2:
            continue
        flagged = sum(1 for paragraph in tail if _looks_like_spurious_trailing_paragraph(paragraph))
        if flagged >= 3 and flagged * 2 >= len(tail):
            return start
        if flagged >= 2 and flagged >= len(tail) - 1:
            return start
    return None


def _split_text_pages(text: str) -> list[str]:
    if "\x0c" in text:
        return [page.strip() for page in text.split("\x0c")]
    return [text]
