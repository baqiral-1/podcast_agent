from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from pypdf import PdfReader


LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}
TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]+")
PRINTER_COUNT_RE = re.compile(r"^(?:\d+\s*){8,}[A-Za-z0-9]*$")
PAGE_TOKEN_RE = re.compile(r"^\d{1,4}[A-Za-z]{0,4}$")
ROMAN_RE = re.compile(r"^[ivxlcdmIVXLCDM]+$")
INLINE_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-z\)\]”’.,;:])(\d{1,3})(?=(?:\s|[.,;:!?]))")
NUMBER_CONTINUATION_RE = re.compile(
    r"^(?:million|billion|percent|villagers|nomads|residents|births|people|days|hours|kilometers|civil servants|votes)\b",
    re.I,
)


@dataclass(frozen=True)
class Bookmark:
    depth: int
    page: int
    title: str


@dataclass(frozen=True)
class BookSpec:
    pdf_path: Path
    output_path: Path
    book_title: str
    chapter_selector: Callable[[Bookmark], bool]
    last_page_inclusive: int
    postprocess: Callable[[str], str] | None = None


def flatten_outline(pdf_path: Path) -> list[Bookmark]:
    reader = PdfReader(str(pdf_path))
    bookmarks: list[Bookmark] = []

    def walk(items: Iterable[object], depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = getattr(item, "title", "") or ""
            if not title:
                continue
            page = reader.get_destination_page_number(item)
            bookmarks.append(Bookmark(depth=depth, page=page, title=title))

    walk(reader.outline)
    return bookmarks


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for source, target in LIGATURES.items():
        text = text.replace(source, target)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("– ", "–").replace("— ", "—")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def canonicalize(text: str) -> str:
    lowered = normalize_text(text).casefold()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def title_fragments(title: str) -> set[str]:
    fragments = {title.strip()}
    stripped = re.sub(r"^[Cc]hapter\s+\d+[:.]?\s*", "", title).strip()
    stripped = re.sub(r"^\d+\s+", "", stripped).strip()
    if stripped:
        fragments.add(stripped)
    if ":" in stripped:
        left, right = [part.strip() for part in stripped.split(":", 1)]
        if left:
            fragments.add(left)
        if right:
            fragments.add(right)
    words = stripped.split()
    for index in range(1, len(words)):
        tail = " ".join(words[index:]).strip()
        if tail and len(tail.split()) <= 5:
            fragments.add(tail)
    return {fragment for fragment in fragments if fragment}


def attribution_like(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if len(text.split()) > 12:
        return False
    if text.startswith(("The ", '"', "'")):
        return True
    if "," in text:
        return True
    words = TITLE_WORD_RE.findall(text)
    if not words:
        return False
    return sum(1 for word in words if word[:1].isupper()) >= max(1, len(words) - 1)


def is_titleish_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if len(text.split()) > 12:
        return False
    if text.lower().startswith(("chapter ", "introduction")):
        return True
    if re.search(r"[.!?]$", text):
        return False
    words = TITLE_WORD_RE.findall(text)
    if not words:
        return False
    upperish = sum(1 for word in words if word[:1].isupper())
    return upperish >= max(1, len(words) - 1)


def is_subheading_like(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text or len(text.split()) > 14:
        return False
    if re.search(r"[.!?]$", text):
        return False
    words = TITLE_WORD_RE.findall(text)
    if not words:
        return False
    if len(words) == 1 and len(words[0]) == 1:
        return False
    uppercase_ratio = sum(1 for word in words if word[:1].isupper()) / len(words)
    return uppercase_ratio >= 0.8


def extract_lines(reader: PdfReader, start_page: int, end_page: int) -> list[str]:
    lines: list[str] = []
    for page_index in range(start_page, end_page + 1):
        text = normalize_text(reader.pages[page_index].extract_text() or "")
        lines.extend(text.splitlines())
        lines.append("")
    return lines


def clean_lines(lines: list[str], *, book_title: str, chapter_title: str) -> list[str]:
    chapter_markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    book_marker = canonicalize(book_title)
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if PAGE_TOKEN_RE.fullmatch(line):
            next_line = lines[index].strip() if index < len(lines) else ""
            if next_line and NUMBER_CONTINUATION_RE.match(next_line):
                lines[index] = f"{line} {next_line}"
                continue
        if PRINTER_COUNT_RE.fullmatch(line):
            continue
        if PAGE_TOKEN_RE.fullmatch(line) or ROMAN_RE.fullmatch(line):
            continue
        if re.fullmatch(r"\d+\s+[A-Z][A-Z '\-]+", line):
            if book_marker in canonicalize(line):
                continue
        canon = canonicalize(line)
        if not canon:
            continue
        if canon == "text":
            continue
        if canon == book_marker or canon in chapter_markers:
            continue
        if is_titleish_line(line) and any(marker.startswith(canon) or canon.startswith(marker) for marker in chapter_markers):
            continue
        if canon.startswith("oceanofpdfcom"):
            continue
        if re.fullmatch(r"\d{1,2}\s+.+", line) and len(line.split()) <= 6 and book_marker in canon:
            continue
        if re.fullmatch(r"\d{1,2}\s+[A-Z].*", line):
            line = re.sub(r"^\d{1,2}\s+", "", line)
        if len(line) == 1 and line.isupper() and index < len(lines):
            next_line = lines[index].strip()
            if next_line and not PRINTER_COUNT_RE.fullmatch(next_line):
                lines[index] = f"{line}{next_line}"
                continue
        cleaned.append(line)
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def strip_leading_titles_and_epigraph(lines: list[str]) -> list[str]:
    remaining = lines[:]
    while remaining and canonicalize(remaining[0]) in {"text", "chapter"}:
        remaining.pop(0)
        while remaining and not remaining[0]:
            remaining.pop(0)
    while remaining and is_titleish_line(remaining[0]):
        remaining.pop(0)
        while remaining and not remaining[0]:
            remaining.pop(0)

    probe = [line for line in remaining[:18] if line]
    total_words = 0
    for index, line in enumerate(probe):
        total_words += len(line.split())
        if attribution_like(line) and 0 < index <= 12 and total_words <= 130:
            trimmed = probe[index + 1 :]
            remaining = trimmed + remaining[len(probe) :]
            break
        if len(line.split()) > 18 and not line.isupper():
            break
    while remaining:
        line = remaining[0].strip()
        if not line:
            remaining.pop(0)
            continue
        if re.fullmatch(r"[A-Z][A-Z'’\- ]{2,}", line) and len(line.split()) <= 8:
            remaining.pop(0)
            continue
        break
    return remaining


def join_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("-") and re.match(r"^[A-Za-z]", line):
            text = text[:-1] + line
        else:
            text = f"{text} {line}"
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)
    text = re.sub(r"([A-Za-z])\(\s*", r"\1 (", text)
    text = re.sub(r"([A-Za-z])\[(\w)", r"\1 [\2", text)
    text = re.sub(r"([A-Za-z])\s+([–—-])\s+([A-Za-z])", r"\1\2\3", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(\d)\. (?=(?:million|billion|percent|thousand)\b)", r"\1 ", text)
    text = INLINE_FOOTNOTE_RE.sub("", text)
    text = re.sub(r"(?<=[.!?])\s+\d{1,3}(?=\s+[A-Z])", "", text)
    return text.strip()


def paragraphize(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
            continue
        if current:
            paragraphs.append(join_lines(current))
            current = []
    if current:
        paragraphs.append(join_lines(current))
    return [paragraph for paragraph in paragraphs if paragraph]


def strip_subheading_prefix(paragraph: str) -> str:
    words = paragraph.split()
    upper_bound = min(len(words) - 3, 18)
    for split_at in range(2, upper_bound + 1):
        prefix = " ".join(words[:split_at]).strip(" :")
        suffix = " ".join(words[split_at:]).strip()
        if not suffix:
            continue
        if not is_subheading_like(prefix):
            continue
        if re.match(r'^[A-Z"\'(][^\n]{20,}', suffix):
            return suffix
    return paragraph


def is_map_like(paragraph: str) -> bool:
    words = paragraph.split()
    if len(words) < 12:
        return False
    lower = paragraph.casefold()
    upper_tokens = sum(
        1
        for word in words
        if re.fullmatch(r"[A-Z][A-Z'’\-]*", word.strip(".,;:()[]"))
    )
    if "500 miles" in lower or "1000 kms" in lower:
        return True
    if upper_tokens >= 10 and any(token in lower for token in (" sea ", " empire", " gulf", " miles", " kms")):
        return True
    return False


def clean_chapter(reader: PdfReader, spec: BookSpec, bookmark: Bookmark, end_page: int) -> str:
    lines = extract_lines(reader, bookmark.page, end_page)
    lines = clean_lines(lines, book_title=spec.book_title, chapter_title=bookmark.title)
    lines = strip_leading_titles_and_epigraph(lines)
    paragraphs = paragraphize(lines)
    cleaned: list[str] = []
    markers = {canonicalize(fragment) for fragment in title_fragments(bookmark.title)}
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        paragraph = strip_subheading_prefix(paragraph)
        if canonicalize(paragraph) in markers:
            continue
        if canonicalize(paragraph) == canonicalize(spec.book_title):
            continue
        if is_map_like(paragraph):
            continue
        if is_subheading_like(paragraph):
            continue
        cleaned.append(paragraph)
    text = "\n\n".join(cleaned).strip()
    if spec.postprocess:
        text = spec.postprocess(text)
    return text.strip()


def render_book(chapters: list[str]) -> str:
    rendered: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        body = chapter.strip()
        if not body:
            continue
        rendered.append(f"Chapter {index}:\n{body}")
    return "\n\n".join(rendered).strip() + "\n"


def clean_book(spec: BookSpec) -> str:
    reader = PdfReader(str(spec.pdf_path))
    bookmarks = [bookmark for bookmark in flatten_outline(spec.pdf_path) if spec.chapter_selector(bookmark)]
    chapters: list[str] = []
    for index, bookmark in enumerate(bookmarks):
        next_page = bookmarks[index + 1].page - 1 if index + 1 < len(bookmarks) else spec.last_page_inclusive
        cleaned = clean_chapter(reader, spec, bookmark, next_page)
        if cleaned:
            chapters.append(cleaned)
    rendered = render_book(chapters)
    spec.output_path.write_text(rendered, encoding="utf-8")
    return rendered


def sample_segments(text: str, *, seed: int = 7, count: int = 20, words_per_segment: int = 200) -> list[str]:
    words = text.split()
    if len(words) <= words_per_segment:
        return [" ".join(words)]
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(len(words) - words_per_segment + 1), k=count))
    return [" ".join(words[start : start + words_per_segment]) for start in starts]
