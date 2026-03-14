"""Source ingestion helpers for supported book formats."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


_OCR_SPACED_HEADING_RE = re.compile(r"^(?:[A-Za-z]\s+){3,}[A-Za-z]$")
_PAGE_MARKER_RE = re.compile(r"^\[Page \d+\]$")
_CHAPTER_TOKEN_RE = re.compile(r"^CHAPTER((?:\d+)|(?:[IVXLCDM]+))$", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(
    r"^(chapter\s+(?:\d+|[ivxlcdm]+)|prologue|introduction|foreword|preface|appendix(?:\s+[a-z0-9]+)?|epilogue|afterword|conclusion)\b",
    re.IGNORECASE,
)


def read_source_text(path: Path) -> str:
    """Read normalized source text from a supported input file."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def extract_pdf_text(path: Path) -> str:
    """Extract page-tagged text from a PDF and fail loudly on unreadable input."""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - exercised via tests with fake reader
        raise RuntimeError(f"Unable to read PDF '{path}': {exc}") from exc
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF files are not supported: '{path}'.")

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized_text = normalize_source_text(text)
        pages.append(f"[Page {page_number}]\n{normalized_text}".strip())
    return "\n\n".join(page for page in pages if page).strip()


def normalize_source_text(text: str) -> str:
    """Normalize OCR-style spacing and broken line wraps without erasing paragraphs."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
    blocks = re.split(r"\n\s*\n", normalized)

    normalized_blocks: list[str] = []
    for block in blocks:
        lines = [_normalize_line(line) for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1 or any(_PAGE_MARKER_RE.match(line) for line in lines):
            normalized_blocks.extend(lines)
            continue
        if _looks_like_heading(lines[0]):
            normalized_blocks.append(lines[0])
            remaining_lines = lines[1:]
            if remaining_lines and len(remaining_lines[0].split()) <= 12:
                normalized_blocks.append(remaining_lines[0])
                remaining_lines = remaining_lines[1:]
            if remaining_lines:
                normalized_blocks.append(" ".join(remaining_lines))
            continue
        normalized_blocks.append(" ".join(lines))
    return "\n\n".join(normalized_blocks).strip()


def _normalize_line(line: str) -> str:
    stripped = re.sub(r"\s+", " ", line.strip())
    if _OCR_SPACED_HEADING_RE.fullmatch(stripped):
        compact = stripped.replace(" ", "")
        chapter_match = _CHAPTER_TOKEN_RE.fullmatch(compact)
        if chapter_match:
            return f"CHAPTER {chapter_match.group(1)}"
        return compact
    return stripped


def _looks_like_heading(line: str) -> bool:
    return bool(_SECTION_HEADING_RE.match(line))
