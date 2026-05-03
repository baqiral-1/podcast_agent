#!/usr/bin/env python3
"""One-off cleaner for the Babur Mughal download."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from podcast_agent.utils.book_cleaning import derive_output_filename
from podcast_agent.utils.mughal_cleanup import (
    _FRONT_MATTER_PARAGRAPH_RE,
    _INLINE_ATTRIBUTION_RE,
    _SCAN_ATTRIBUTION_RE,
    _SECTION_SUBHEADING_RE,
    _SHORT_CITATION_PARAGRAPH_RE,
    _drop_page_artifacts,
    _is_non_latin_artifact,
    _looks_like_note_paragraph,
    _looks_like_sentence,
    _prepare_page,
    _truncate_embedded_terminal_block,
    extract_mughal_book_source,
)


DEFAULT_SOURCE = Path.home() / "Downloads" / "dokumen.pub_babur-timurid-prince-and-mughal-emperor-1483-1530-9781108470070-9781107107267.pdf"
_CITATION_RE = re.compile(
    r"\b(BN[–-]M|Beveridge|ibid\.?|trans\.|ed\.|vol\.|pp?\.|fs?\.|University Press|"
    r"Cambridge:|Leiden|Boston:|New York:|Princeton:|Kyoto:|London:|Delhi:|"
    r"Calcutta:|Journal of|Bibliography)\b",
    re.IGNORECASE,
)
_FOOTNOTE_AFTER_WORD_RE = re.compile(r"(?<=[A-Za-z)\]\"'’”])\d{1,3}(?=(?:\s|$))")
_FOOTNOTE_AFTER_PUNCT_RE = re.compile(r"(?<!\d\.)(?<=[.!?])\d{1,3}(?=(?:\s|$))")
_FOOTNOTE_AFTER_YEAR_RE = re.compile(r"(?<=\d{4}\.)\d{1,3}(?=(?:\s|$))")
_VISIBLE_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
_PART_HEADING_RE = re.compile(r"^part\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six)$", re.IGNORECASE)
_DISPLAY_TITLE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9'’\"“”()/:,&.-]*$")
_EXTRACTION_SCAR_REPLACEMENTS = {
    "GrecoIslamic": "Greco-Islamic",
    "PersoIslamic": "Perso-Islamic",
    "TurcoMongol": "Turco-Mongol",
    "antiUzbek": "anti-Uzbek",
    "eastsoutheast": "east-southeast",
    "fatherin-law": "father-in-law",
    "goodhumoured": "good-humoured",
    "late fifteenthcentury": "late fifteenth-century",
    "nobleTimurid": "noble Timurid",
    "nineteenthcentury": "nineteenth-century",
    "sixteenthcentury": "sixteenth-century",
    "thirtytwo": "thirty-two",
    "wellmannered": "well-mannered",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _first_sentence_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line and _looks_like_sentence(line):
            return index
    return None


def _looks_reference_heavy(lines: list[str]) -> bool:
    text = " ".join(lines)
    marker_count = len(_CITATION_RE.findall(text))
    numbered_lines = sum(1 for line in lines if re.match(r"^\d{1,3}[.)]?\s", line))
    short_lines = sum(1 for line in lines if len(line.split()) <= 16)
    return marker_count >= 3 or (marker_count >= 2 and numbered_lines >= 2) or (
        "Endnotes" in text and short_lines >= 2
    )


def _looks_like_display_heading(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    if _PART_HEADING_RE.fullmatch(compact):
        return True
    tokens = re.findall(r"[A-Za-z0-9'’\"“”()/:,&.-]+", compact)
    if not 1 <= len(tokens) <= 8:
        return False
    if _looks_like_sentence(compact) or re.search(r"[.!?]$", compact):
        return False
    lower_connectors = {"a", "an", "and", "as", "at", "for", "from", "in", "of", "on", "the", "to"}
    for token in tokens:
        if token.isdigit() or token.lower() in lower_connectors:
            continue
        if not _DISPLAY_TITLE_TOKEN_RE.fullmatch(token):
            return False
    return True


def _paragraph_should_drop(paragraph: str) -> bool:
    if _SCAN_ATTRIBUTION_RE.search(paragraph) or _INLINE_ATTRIBUTION_RE.match(paragraph):
        return True
    if _FRONT_MATTER_PARAGRAPH_RE.search(paragraph):
        return True
    if _looks_like_note_paragraph(paragraph):
        return True
    if _SECTION_SUBHEADING_RE.match(paragraph) and len(paragraph.split()) <= 20:
        return True
    if len(paragraph.split()) <= 10 and _SHORT_CITATION_PARAGRAPH_RE.match(paragraph):
        return True
    if _is_non_latin_artifact(paragraph):
        return True
    if _looks_like_display_heading(paragraph):
        return True
    return False


def _line_should_drop(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if _VISIBLE_PAGE_NUMBER_RE.fullmatch(compact):
        return True
    if _SCAN_ATTRIBUTION_RE.search(line) or _INLINE_ATTRIBUTION_RE.match(line):
        return True
    if _is_non_latin_artifact(line):
        return True
    return False


def _strip_footnote_markers(text: str) -> str:
    text = _FOOTNOTE_AFTER_WORD_RE.sub("", text)
    text = _FOOTNOTE_AFTER_PUNCT_RE.sub("", text)
    return _FOOTNOTE_AFTER_YEAR_RE.sub("", text)


def _build_repeated_headings(prepared_pages: list[object]) -> set[str]:
    counts: dict[str, int] = {}
    for page in prepared_pages:
        for line in page.body_lines:
            compact = line.strip()
            if not compact:
                continue
            if len(compact) > 80 or len(compact.split()) > 8:
                continue
            if _looks_like_sentence(compact):
                continue
            counts[compact] = counts.get(compact, 0) + 1
    return {line for line, count in counts.items() if count >= 3}


def _looks_like_inline_heading(line: str, next_nonblank: str, repeated_headings: set[str]) -> bool:
    compact = line.strip()
    if _looks_like_sentence(compact):
        return False
    if re.search(r"[.!?]", compact):
        return False
    if not next_nonblank or not _looks_like_sentence(next_nonblank):
        return False
    word_count = len(compact.split())
    return 1 < word_count <= 5


def _merge_false_breaks(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for paragraph in paragraphs:
        if (
            merged
            and paragraph
            and not re.search(r"[.!?][\"'’”\]]?$", merged[-1])
        ):
            merged[-1] = f"{merged[-1]} {paragraph}".strip()
            continue
        merged.append(paragraph)
    return merged


def _collapse_lines(lines: list[str], repeated_headings: set[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        paragraph = " ".join(current).strip()
        paragraph = _truncate_embedded_terminal_block(paragraph).strip()
        paragraph = _strip_footnote_markers(paragraph)
        if paragraph and not _paragraph_should_drop(paragraph):
            paragraphs.append(paragraph)
        current = []

    for index, line in enumerate(lines):
        if not line:
            flush()
            continue
        if _line_should_drop(line):
            continue
        if line.strip() in repeated_headings:
            continue
        next_nonblank = next((item for item in lines[index + 1 :] if item), "")
        if _looks_like_inline_heading(line, next_nonblank, repeated_headings):
            flush()
            continue
        current.append(line)
    flush()
    paragraphs = _merge_false_breaks(paragraphs)
    return "\n\n".join(paragraphs).strip()


def _render_chapters(bodies: list[str]) -> str:
    chapters = [body for body in bodies if body]
    return "\n\n".join(f"Chapter {index}:\n{body}" for index, body in enumerate(chapters, start=1)).strip() + "\n"


def _postprocess_extraction_scars(text: str) -> str:
    for source, target in _EXTRACTION_SCAR_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_mughal_book_source(args.source)
    raw_pages = [[line.strip() for line in page.splitlines() if line.strip()] for page in extracted.page_texts]
    prepared_pages = [_prepare_page(page) for page in extracted.page_texts]
    repeated_headings = _build_repeated_headings(prepared_pages)

    starts: list[tuple[int, int]] = []
    for page_index, lines in enumerate(raw_pages):
        if len(lines) < 2 or not re.fullmatch(r"[1-5]", lines[0]) or len(lines[1].split()) < 2:
            continue
        body_start = _first_sentence_index(prepared_pages[page_index].body_lines)
        if body_start is not None:
            starts.append((page_index, body_start))

    bodies: list[str] = []
    for index, (start_page, body_start) in enumerate(starts):
        end_page = starts[index + 1][0] if index + 1 < len(starts) else len(prepared_pages)
        chapter_lines: list[str] = []
        for page_index in range(start_page, end_page):
            lines = (
                prepared_pages[page_index].body_lines[body_start:]
                if page_index == start_page
                else prepared_pages[page_index].body_lines
            )
            if page_index > start_page and _looks_reference_heavy(lines):
                break
            filtered = _drop_page_artifacts(lines)
            if not filtered:
                continue
            if chapter_lines and chapter_lines[-1]:
                chapter_lines.append("")
            chapter_lines.extend(filtered)
        body = _collapse_lines(chapter_lines, repeated_headings)
        if body:
            bodies.append(body)

    cleaned_text = _postprocess_extraction_scars(_render_chapters(bodies))
    output_path = args.output_dir / derive_output_filename(args.source)
    output_path.write_text(cleaned_text, encoding="utf-8")
    print(
        f"{args.source.name} -> {output_path} | method={extracted.extraction_method} "
        f"chapters={len(bodies)} words={len(cleaned_text.split())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
