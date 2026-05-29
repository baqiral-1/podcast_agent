#!/usr/bin/env python3
"""Repair outline-book chapter openings with OCR from each chapter's first page."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

from podcast_agent.utils.book_cleaning import derive_output_filename

HELPER_PATH = Path(__file__).with_name("oneoff_clean_may2_download_batch.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "oneoff_clean_may2_download_batch", HELPER_PATH
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper module from {HELPER_PATH}")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPER_MODULE
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)

BACK_MATTER_TITLE_RE = _HELPER_MODULE.BACK_MATTER_TITLE_RE
PRINTED_PAGE_RE = _HELPER_MODULE.PRINTED_PAGE_RE
ROMAN_PAGE_RE = _HELPER_MODULE.ROMAN_PAGE_RE
WATERMARK_RE = _HELPER_MODULE.WATERMARK_RE
_book_title_for = _HELPER_MODULE._book_title_for
_chapter_selector_for = _HELPER_MODULE._chapter_selector_for
_normalize_ocr_text = _HELPER_MODULE._normalize_ocr_text
_ocr_page = _HELPER_MODULE._ocr_page
_recent_pdf_paths = _HELPER_MODULE._recent_pdf_paths
canonicalize = _HELPER_MODULE.canonicalize
flatten_outline = _HELPER_MODULE.flatten_outline
title_fragments = _HELPER_MODULE.title_fragments


CHAPTER_RE = re.compile(r"^Chapter (?P<number>\d+)$", re.MULTILINE)
DEFAULT_OUTPUT_DIR = Path("sample_books") / "temp_clean"
OCR_WORD_LIMIT = 180
MIN_OVERLAP = 12


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
    return (
        "\n\n".join(
            f"Chapter {index}\n\n{body.strip()}" for index, body in enumerate(bodies, start=1)
        ).strip()
        + "\n"
    )


def _tokenize(text: str) -> list[str]:
    return text.split()


def _normalized_tokens(words: list[str]) -> list[str]:
    return [canonicalize(word) for word in words]


def _ocr_opening_text(path: Path, page_number: int, chapter_title: str, book_title: str) -> str:
    try:
        raw_text = _normalize_ocr_text(_ocr_page(path, page_number))
    except Exception:
        return ""
    markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    book_marker = canonicalize(book_title)
    kept: list[str] = []
    for raw_line in raw_text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        line = WATERMARK_RE.sub("", line).strip()
        if not line:
            continue
        canon = canonicalize(line)
        if not canon or canon == book_marker or canon in markers:
            continue
        if BACK_MATTER_TITLE_RE.match(line):
            break
        if PRINTED_PAGE_RE.fullmatch(line) or ROMAN_PAGE_RE.fullmatch(line):
            continue
        if re.fullmatch(r"[A-Z][A-Z'’\- ]{1,80}", line) and len(line.split()) <= 10:
            continue
        kept.append(line)
    words = _tokenize(" ".join(part for part in kept if part))
    return " ".join(words[:OCR_WORD_LIMIT]).strip()


def _merge_opening(ocr_opening: str, body: str) -> str:
    ocr_words = _tokenize(ocr_opening)
    body_words = _tokenize(body)
    if len(ocr_words) < 20 or len(body_words) < 40:
        return body
    ocr_norm = _normalized_tokens(ocr_words)
    body_norm = _normalized_tokens(body_words)
    max_overlap = min(80, len(ocr_words), len(body_words))
    overlap = 0
    for size in range(max_overlap, MIN_OVERLAP - 1, -1):
        if ocr_norm[-size:] == body_norm[:size]:
            overlap = size
            break
    if overlap < MIN_OVERLAP:
        return body
    merged_words = ocr_words + body_words[overlap:]
    return " ".join(merged_words).strip()


def main() -> int:
    args = parse_args()
    for pdf_path in _recent_pdf_paths():
        if "India_Wins_Freedom" in pdf_path.name:
            continue
        output_path = args.output_dir / derive_output_filename(pdf_path)
        if not output_path.exists():
            continue
        text = output_path.read_text(encoding="utf-8")
        chapter_bodies = _split_chapters(text)
        outline = [bookmark for bookmark in flatten_outline(pdf_path) if bookmark.page is not None]
        chapter_selector = _chapter_selector_for(pdf_path)
        selected = [bookmark for bookmark in outline if chapter_selector(bookmark)]
        if len(selected) != len(chapter_bodies):
            continue
        book_title = _book_title_for(pdf_path)
        updated_bodies: list[str] = []
        for bookmark, body in zip(selected, chapter_bodies, strict=True):
            ocr_opening = _ocr_opening_text(pdf_path, bookmark.page + 1, bookmark.title, book_title)
            updated_bodies.append(_merge_opening(ocr_opening, body))
        output_path.write_text(_render_chapters(updated_bodies), encoding="utf-8")
        print(f"{pdf_path.name} -> {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
