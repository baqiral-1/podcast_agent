#!/usr/bin/env python3
"""Repair chapter openings from first-page extraction for the May 2 batch."""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from podcast_agent.utils.book_cleaning import derive_output_filename


HELPER_PATH = Path(__file__).with_name("oneoff_clean_may2_download_batch.py")
HELPER_SPEC = importlib.util.spec_from_file_location("oneoff_clean_may2_download_batch", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper module from {HELPER_PATH}")
HELPER_MODULE = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPER_MODULE
HELPER_SPEC.loader.exec_module(HELPER_MODULE)

CHAPTER_RE = re.compile(r"^Chapter (?P<number>\d+)$", re.MULTILINE)
DEFAULT_OUTPUT_DIR = Path("sample_books") / "temp_clean"
LEADING_WORD_LIMIT = 180
MAX_SOURCE_PAGES = 3
MAX_BODY_OFFSET = 80
MIN_OVERLAP = 12

PRINTED_PAGE_RE = HELPER_MODULE.PRINTED_PAGE_RE
ROMAN_PAGE_RE = HELPER_MODULE.ROMAN_PAGE_RE
WATERMARK_RE = HELPER_MODULE.WATERMARK_RE
_book_title_for = HELPER_MODULE._book_title_for
_chapter_selector_for = HELPER_MODULE._chapter_selector_for
_looks_like_heading_phrase = HELPER_MODULE._looks_like_heading_phrase
_recent_pdf_paths = HELPER_MODULE._recent_pdf_paths
canonicalize = HELPER_MODULE.canonicalize
extract_lines = HELPER_MODULE.extract_lines
flatten_outline = HELPER_MODULE.flatten_outline
title_fragments = HELPER_MODULE.title_fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _split_chapters(text: str) -> list[str]:
    matches = list(CHAPTER_RE.finditer(text))
    bodies: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        bodies.append(text[start:end].strip())
    return bodies


def _render_chapters(bodies: list[str]) -> str:
    rendered = [f"Chapter {index}\n\n{body.strip()}" for index, body in enumerate(bodies, start=1)]
    return "\n\n".join(rendered).strip() + "\n"


def _tokenize(text: str) -> list[str]:
    return text.split()


def _normalized_tokens(words: list[str]) -> list[str]:
    return [canonicalize(word) for word in words]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _normalize_opening_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = WATERMARK_RE.sub("", text)
    text = text.replace("–", "-").replace("—", "-")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\b([A-Z])\s+([A-Z])\b", r"\1\2", text)
        text = re.sub(r"\b([A-Z])\s+([A-Z][A-Za-z'’-]+)\b", r"\1\2", text)
    text = re.sub(r"([A-Z]{3,})([a-z]{2,})", r"\1 \2", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r",(?=[A-Za-z])", ", ", text)
    text = re.sub(r"\.(?=[A-Za-z][a-z])", ". ", text)
    text = re.sub(r"([:;!?])(?=[A-Za-z\"'])", r"\1 ", text)
    return " ".join(text.split()).strip()


def _clean_raw_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in lines:
        line = _normalize_opening_text(raw.strip())
        if not line:
            continue
        if PRINTED_PAGE_RE.fullmatch(line) or ROMAN_PAGE_RE.fullmatch(line):
            if len(line) == 1 and line.isupper():
                cleaned.append(line)
                continue
            continue
        cleaned.append(line)
    return cleaned


def _title_markers(chapter_title: str, book_title: str) -> set[str]:
    markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    markers.add(canonicalize(book_title))
    return {marker for marker in markers if marker}


def _is_heading_line(line: str, markers: set[str]) -> bool:
    normalized = canonicalize(line)
    compact = canonicalize(line.replace(" ", ""))
    if not normalized:
        return True
    if normalized in markers or compact in markers:
        return True
    if re.fullmatch(r"\d+\.", line) or re.fullmatch(r"\d+", line):
        return True
    if re.fullmatch(r"\d{4}[-]\d{4}", line):
        return True
    if line.startswith(("CHAPTER ", "PROLOGUE", "INTRODUCTION", "EPILOGUE")):
        return True
    if _looks_like_heading_phrase(line) and _word_count(line) <= 12:
        return True
    return False


def _drop_leading_epigraph(lines: list[str]) -> list[str]:
    last_citation = -1
    for index, line in enumerate(lines[:8]):
        if line.startswith(("-", "—", "–")) or _word_count(line) <= 7:
            last_citation = index
    if last_citation == -1:
        return lines
    start = last_citation + 1
    while start < min(len(lines), 10) and re.match(r"^[a-z]", lines[start]):
        start += 1
    for index in range(start, min(len(lines), 10)):
        if _word_count(lines[index]) >= 8 and re.match(r'^[A-Z"“\']', lines[index]):
            return lines[index:]
    return lines[start:] if start < len(lines) else lines


def _chapter_opening_source(reader: PdfReader, path: Path, bookmark, book_title: str) -> str:
    last_page = min(bookmark.page + MAX_SOURCE_PAGES - 1, len(reader.pages) - 1)
    raw_lines = extract_lines(reader, bookmark.page, last_page)
    lines = _clean_raw_lines(raw_lines)
    if not lines:
        return ""

    name = path.name.casefold()
    markers = _title_markers(bookmark.title, book_title)

    if "the_indian_mutiny" in name:
        seen_title = False
        filtered: list[str] = []
        for line in lines:
            normalized = canonicalize(line)
            compact = canonicalize(line.replace(" ", ""))
            if not seen_title:
                if normalized in markers or compact in markers:
                    seen_title = True
                continue
            filtered.append(line)
        lines = filtered
    elif "directorate_s" in name:
        dropcap = ""
        if lines and re.fullmatch(r"[A-Z]", lines[0]):
            dropcap = lines.pop(0)
        while lines and _is_heading_line(lines[0], markers):
            lines.pop(0)
        if dropcap and lines:
            lines[0] = f"{dropcap}{lines[0].lstrip()}"
    elif "plan_of_attack" in name:
        normalized_title = _normalize_opening_text(bookmark.title)
        normalized_title = re.sub(r"^\d+\s+", "", normalized_title).strip()
        while lines and _is_heading_line(lines[0], markers):
            lines.pop(0)
        if lines:
            lines[0] = normalized_title
    else:
        while lines and _is_heading_line(lines[0], markers):
            lines.pop(0)

    if "imperial_life_in_the_emerald_city" in name and len(lines) >= 2:
        if _word_count(lines[0]) <= 6 and lines[0].endswith(("!", "?", ".")):
            lines.pop(0)

    if "pity_the_nation" in name or "the_black_banners" in name:
        lines = _drop_leading_epigraph(lines)

    words = _tokenize(" ".join(lines))
    return " ".join(words[:LEADING_WORD_LIMIT]).strip()


def _merge_opening(source_opening: str, body: str) -> str:
    source_words = _tokenize(source_opening)
    body_words = _tokenize(body)
    if len(source_words) < 20 or len(body_words) < 40:
        return body

    source_norm = _normalized_tokens(source_words)
    body_norm = _normalized_tokens(body_words)
    max_overlap = min(120, len(source_words), len(body_words))

    body_window = body_norm[: min(len(body_norm), 260)]
    matcher = difflib.SequenceMatcher(a=source_norm, b=body_window, autojunk=False)
    best = max(matcher.get_matching_blocks(), key=lambda block: block.size)
    if best.size >= 20 and best.a <= 8 and best.b <= MAX_BODY_OFFSET:
        merged_words = source_words + body_words[best.b + best.size :]
        return " ".join(merged_words).strip()

    for overlap in range(max_overlap, MIN_OVERLAP - 1, -1):
        limit = min(MAX_BODY_OFFSET, len(body_words) - overlap)
        for offset in range(limit + 1):
            if source_norm[:overlap] != body_norm[offset : offset + overlap]:
                continue
            source_index = overlap
            body_index = offset + overlap
            while (
                source_index < len(source_words)
                and body_index < len(body_words)
                and source_norm[source_index] == body_norm[body_index]
            ):
                source_index += 1
                body_index += 1
            merged_words = source_words + body_words[body_index:]
            return " ".join(merged_words).strip()

    best_offset: int | None = None
    best_overlap = 0
    for overlap in range(max_overlap, MIN_OVERLAP - 1, -1):
        limit = min(MAX_BODY_OFFSET, len(body_words) - overlap)
        for offset in range(limit + 1):
            if source_norm[-overlap:] == body_norm[offset : offset + overlap]:
                best_offset = offset
                best_overlap = overlap
                break
        if best_offset is not None:
            break

    if best_offset is None:
        return body

    merged_words = source_words + body_words[best_offset + best_overlap :]
    return " ".join(merged_words).strip()


def main() -> int:
    args = parse_args()
    for pdf_path in _recent_pdf_paths():
        if "india_wins_freedom" in pdf_path.name.casefold():
            continue
        output_path = args.output_dir / derive_output_filename(pdf_path)
        if not output_path.exists():
            continue

        text = output_path.read_text(encoding="utf-8")
        chapter_bodies = _split_chapters(text)
        reader = PdfReader(str(pdf_path))
        chapter_selector = _chapter_selector_for(pdf_path)
        selected = [bookmark for bookmark in flatten_outline(pdf_path) if bookmark.page is not None and chapter_selector(bookmark)]
        if len(selected) != len(chapter_bodies):
            continue

        book_title = _book_title_for(pdf_path)
        updated_bodies: list[str] = []
        for bookmark, body in zip(selected, chapter_bodies, strict=True):
            source_opening = _chapter_opening_source(reader, pdf_path, bookmark, book_title)
            updated_bodies.append(_merge_opening(source_opening, body))

        output_path.write_text(_render_chapters(updated_bodies), encoding="utf-8")
        print(f"{pdf_path.name} -> {output_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
