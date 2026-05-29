#!/usr/bin/env python3
"""One-off cleaner for The Broken Script download."""

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


DEFAULT_SOURCE = Path.home() / "Downloads" / "_OceanofPDF.com_THE_BROKEN_SCRIPT_-_Swapna_Liddle.pdf"
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
_ACKNOWLEDGEMENT_START_RE = re.compile(
    r"^(This study of Delhi began when I enrolled|I first became interested in research into)",
    re.IGNORECASE,
)
_TERMINAL_SECTION_RE = re.compile(r"^(conclusion|epilogue|afterword|postscript)$", re.IGNORECASE)
_PART_HEADING_RE = re.compile(
    r"^part\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six)$", re.IGNORECASE
)
_DISPLAY_TITLE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9'’\"“”()/:,&.-]*$")
_TABLE_ROW_RE = re.compile(r"^\d[\d,]*(?:\s+\d[\d,]*){2,}$")
_TABLE_HEADER_RE = re.compile(r"^(year|demand in rs|collection in rs)$", re.IGNORECASE)
_DISPLAY_TAIL_RE = re.compile(
    r":\s*\d{1,3}\s+(year|demand in rs|collection in rs)\b.*$", re.IGNORECASE
)
_DISPLAY_TAIL_NO_FOOTNOTE_RE = re.compile(
    r":\s*(year|demand in rs|collection in rs)\b.*$", re.IGNORECASE
)
_CAPTION_LIKE_RE = re.compile(r"^[A-Z][A-Za-z'’ -]+,\s*(?:[A-Z][a-z]+\s+)?\d{4}$")
_EXTRACTION_SCAR_REPLACEMENTS = {
    "AngloMaratha": "Anglo-Maratha",
    "NorthWestern": "North-Western",
    "farflung": "far-flung",
    "onesided": "one-sided",
    "twentysix": "twenty-six",
    "VicePresident": "Vice-President",
    "wellknown": "well-known",
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
    return (
        marker_count >= 3
        or (marker_count >= 2 and numbered_lines >= 2)
        or ("Endnotes" in text and short_lines >= 2)
    )


def _looks_like_display_heading(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    if _PART_HEADING_RE.fullmatch(compact) or _TABLE_HEADER_RE.fullmatch(compact):
        return True
    if _CAPTION_LIKE_RE.fullmatch(compact):
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
    if _TABLE_ROW_RE.fullmatch(" ".join(paragraph.split())):
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
    if _TABLE_ROW_RE.fullmatch(" ".join(line.split())):
        return True
    return False


def _strip_footnote_markers(text: str) -> str:
    text = _FOOTNOTE_AFTER_WORD_RE.sub("", text)
    text = _FOOTNOTE_AFTER_PUNCT_RE.sub("", text)
    return _FOOTNOTE_AFTER_YEAR_RE.sub("", text)


def _strip_display_tail(text: str) -> str:
    text = _DISPLAY_TAIL_RE.sub("", text)
    return _DISPLAY_TAIL_NO_FOOTNOTE_RE.sub("", text)


def _trim_terminal_paragraphs(paragraphs: list[str]) -> list[str]:
    for index, paragraph in enumerate(paragraphs):
        if _ACKNOWLEDGEMENT_START_RE.match(paragraph):
            return paragraphs[:index]
    return paragraphs


def _merge_false_breaks(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for paragraph in paragraphs:
        if merged and paragraph and not re.search(r"[.!?][\"'’”\]]?$", merged[-1]):
            merged[-1] = f"{merged[-1]} {paragraph}".strip()
            continue
        merged.append(paragraph)
    return merged


def _collapse_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        paragraph = " ".join(current).strip()
        paragraph = _truncate_embedded_terminal_block(paragraph).strip()
        paragraph = _strip_footnote_markers(paragraph)
        paragraph = _strip_display_tail(paragraph).strip()
        if paragraph and not _paragraph_should_drop(paragraph):
            paragraphs.append(paragraph)
        current = []

    for line in lines:
        if not line:
            flush()
            continue
        if _TERMINAL_SECTION_RE.match(line.strip()):
            flush()
            break
        if _line_should_drop(line):
            continue
        current.append(line)
    flush()
    paragraphs = _trim_terminal_paragraphs(paragraphs)
    paragraphs = _merge_false_breaks(paragraphs)
    return "\n\n".join(paragraphs).strip()


def _render_chapters(bodies: list[str]) -> str:
    chapters = [body for body in bodies if body]
    return (
        "\n\n".join(
            f"Chapter {index}:\n{body}" for index, body in enumerate(chapters, start=1)
        ).strip()
        + "\n"
    )


def _postprocess_extraction_scars(text: str) -> str:
    for source, target in _EXTRACTION_SCAR_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_mughal_book_source(args.source)
    raw_pages = [
        [line.strip() for line in page.splitlines() if line.strip()]
        for page in extracted.page_texts
    ]
    prepared_pages = [_prepare_page(page) for page in extracted.page_texts]

    starts: list[tuple[int, int, int]] = []
    for page_index, lines in enumerate(raw_pages):
        if not lines or not re.fullmatch(r"\d{1,2}", lines[0]):
            continue
        chapter_number = int(lines[0])
        if not 1 <= chapter_number <= 57:
            continue
        actual_page = page_index
        if prepared_pages[page_index].body_lines:
            body_start = _first_sentence_index(prepared_pages[page_index].body_lines)
        else:
            actual_page = page_index + 1
            if actual_page >= len(prepared_pages):
                continue
            body_start = _first_sentence_index(prepared_pages[actual_page].body_lines)
        starts.append((chapter_number, actual_page, 0 if body_start is None else body_start))

    starts.sort()
    bodies: list[str] = []
    for index, (_, start_page, body_start) in enumerate(starts):
        end_page = starts[index + 1][1] if index + 1 < len(starts) else len(prepared_pages)
        chapter_lines: list[str] = []
        for page_index in range(start_page, end_page):
            lines = (
                prepared_pages[page_index].body_lines[body_start:]
                if page_index == start_page
                else prepared_pages[page_index].body_lines
            )
            filtered = _drop_page_artifacts(lines)
            if not filtered:
                continue
            if _looks_reference_heavy(filtered):
                break
            if chapter_lines and chapter_lines[-1]:
                chapter_lines.append("")
            chapter_lines.extend(filtered)
        body = _collapse_lines(chapter_lines)
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
