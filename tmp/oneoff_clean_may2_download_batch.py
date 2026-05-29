#!/usr/bin/env python3
"""One-off cleaner for the May 2 Downloads book batch."""

from __future__ import annotations

import argparse
import importlib.util
from functools import lru_cache
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from podcast_agent.utils.book_cleaning import derive_output_filename


DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_OUTPUT_DIR = Path("sample_books") / "temp_clean"
REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_HELPER_PATH = Path(__file__).with_name("oneoff_iran_book_clean_common_may1.py")
ROMAN_PAGE_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
CHAPTER_LINE_RE = re.compile(r"^(?P<number>\d{1,2})[\])\.]?\s+(?P<title>.+?)\s+\d{1,3}$")
PRINTED_PAGE_RE = re.compile(r"^\d{1,3}$")
INDIA_WINS_PAGE_OFFSET = 12


@dataclass(frozen=True)
class OcrChapter:
    number: int
    title: str
    printed_page: int


_COMMON_SPEC = importlib.util.spec_from_file_location(
    "oneoff_iran_book_clean_common_may1", COMMON_HELPER_PATH
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper module from {COMMON_HELPER_PATH}")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules[_COMMON_SPEC.name] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

BookSpec = _COMMON_MODULE.BookSpec
Bookmark = _COMMON_MODULE.Bookmark
canonicalize = _COMMON_MODULE.canonicalize
clean_book = _COMMON_MODULE.clean_book
clean_lines = _COMMON_MODULE.clean_lines
extract_lines = _COMMON_MODULE.extract_lines
flatten_outline = _COMMON_MODULE.flatten_outline
is_map_like = _COMMON_MODULE.is_map_like
is_subheading_like = _COMMON_MODULE.is_subheading_like
paragraphize = _COMMON_MODULE.paragraphize
title_fragments = _COMMON_MODULE.title_fragments

SPACED_HEADING_RE = re.compile(r"^(?:[A-Z]\s+){2,}[A-Z]$")
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z'’\- ]{1,60}$")
WATERMARK_RE = re.compile(r"\s*OceanofPDF\s*\.?\s*com\b", re.IGNORECASE)
BACK_MATTER_TITLE_RE = re.compile(
    r"^(notes|bibliography|index|about the author|author|copyright|acknowledg(?:e)?ments|"
    r"glossary|chronology|chronology of events|list of illustrations|list of abbreviations|"
    r"illustrations|photos|photographs|photo credits|permissions|appendixes?|"
    r"principal characters|key documents|praise for\b|otherfm|a note about the author|"
    r"sequence of events|table of contents|contents|also by)\b",
    re.IGNORECASE,
)
SMALL_TITLE_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _recent_pdf_paths(limit: int = 11) -> list[Path]:
    return sorted(
        [
            path
            for path in DOWNLOADS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def _word_number_title(title: str) -> bool:
    return bool(
        re.match(
            r"^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
            r"sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]one|twenty[- ]two|twenty[- ]three|"
            r"twenty[- ]four|twenty[- ]five|twenty[- ]six|twenty[- ]seven|twenty[- ]eight|"
            r"twenty[- ]nine|thirty):",
            title,
            flags=re.IGNORECASE,
        )
    )


def _chapter_selector_for(path: Path):
    name = path.name.casefold()

    if "imperial_life_in_the_emerald_city" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(
                title.lower() == "prologue"
                or re.fullmatch(r"chapter \d+", title, flags=re.IGNORECASE)
                or title.startswith("The Green Zone, Scene ")
            )

        return selector

    if "directorate_s" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(title.upper() == "INTRODUCTION" or re.match(r"^[A-Z]+:", title))

        return selector

    if "the_black_banners" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(
                title.lower() in {"prologue", "postscript", "conclusion"}
                or re.match(r"^\d+\.", title)
            )

        return selector

    if "one_palestine_complete" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(title.startswith("Introduction") or re.match(r"^\d+\.", title))

        return selector

    if "the_arabs" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(title == "Introduction" or title.startswith("CHAPTER "))

        return selector

    if "the_iron_wall" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(
                title.startswith("Prologue")
                or title.startswith("Epilogue")
                or re.match(r"^\d+\.", title)
            )

        return selector

    if "pity_the_nation" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(re.match(r"^\d+\s", title))

        return selector

    if "india_at_war" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(title == "Prologue" or re.match(r"^\d+\s", title))

        return selector

    if "the_indian_mutiny" in name:

        def selector(bookmark: Bookmark) -> bool:
            title = bookmark.title.strip()
            return bool(title.startswith("Prologue") or re.match(r"^\d+\.", title))

        return selector

    if "plan_of_attack" in name:

        def selector(bookmark: Bookmark) -> bool:
            return bool(re.match(r"^\d+\s", bookmark.title.strip()))

        return selector

    raise ValueError(f"No selector configured for {path.name}")


def _book_title_for(path: Path) -> str:
    title = derive_output_filename(path).removesuffix(".cleaned.txt").replace("_", " ")
    return title.title()


def _strip_watermark(text: str) -> str:
    return WATERMARK_RE.sub("", text).replace("\x00", "").strip()


def _looks_like_heading_phrase(text: str) -> bool:
    stripped = _strip_watermark(text)
    if not stripped or re.search(r"[.!?]$", stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", stripped)
    if not words or len(words) > 12:
        return False
    significant = [word for word in words if word.casefold() not in SMALL_TITLE_WORDS]
    if not significant:
        return False
    return all(word[:1].isupper() or word.isupper() for word in significant)


def _repair_opening_lines(lines: list[str], chapter_title: str) -> list[str]:
    repaired = lines[:]
    chapter_markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    if len(repaired) < 3:
        return repaired
    first = repaired[0].strip()
    second = repaired[1].strip()
    if len(first) == 1 and first.isupper() and canonicalize(second) in chapter_markers:
        for index in range(2, len(repaired)):
            candidate = repaired[index].strip()
            if candidate:
                repaired[index] = f"{first}{candidate}"
                del repaired[0:2]
                break
    return repaired


def _strip_leading_paragraph_prefix(paragraph: str, chapter_title: str) -> str:
    text = _strip_watermark(paragraph)
    if not text:
        return ""
    markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    spaced_match = re.match(r"^(?P<prefix>(?:[A-Za-z]\s+){2,}[A-Za-z])\s+(?P<rest>.+)$", text)
    if spaced_match:
        compact = spaced_match.group("prefix").replace(" ", "")
        if canonicalize(compact) in markers or canonicalize(compact) in {
            "prologue",
            "introduction",
            "epilogue",
        }:
            text = spaced_match.group("rest").strip()
    words = text.split()
    upper_bound = min(len(words) - 3, 12)
    for split_at in range(2, upper_bound + 1):
        prefix = " ".join(words[:split_at]).strip(" :")
        suffix = " ".join(words[split_at:]).strip()
        if not suffix:
            continue
        if canonicalize(prefix) in markers:
            return suffix
        if split_at > 6:
            continue
        if re.search(r"[,\"']", prefix) or any(char.isdigit() for char in prefix):
            continue
        if not re.search(r"[a-z]", prefix):
            continue
        if _looks_like_heading_phrase(prefix) and re.match(r'^[A-Z"\'(]', suffix):
            return suffix
    return text


def _skip_full_paragraph(paragraph: str) -> bool:
    text = _strip_watermark(paragraph)
    if not text:
        return True
    if BACK_MATTER_TITLE_RE.match(text):
        return True
    if _looks_like_heading_phrase(text) and len(text.split()) <= 10:
        return True
    return False


def _clean_outline_paragraphs(
    path: Path, chapter_title: str, lines: list[str], book_title: str
) -> list[str]:
    cleaned_lines = [_strip_watermark(line) if line else "" for line in lines]
    paragraphs = paragraphize(cleaned_lines)
    title_markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    book_marker = canonicalize(book_title)
    rendered: list[str] = []
    seen_plates = False
    for index, paragraph in enumerate(paragraphs):
        paragraph = _strip_watermark(paragraph)
        if not paragraph:
            continue
        if path.name.casefold().find("india_at_war") != -1 and paragraph.startswith("Plates "):
            seen_plates = True
            break
        if seen_plates:
            break
        if BACK_MATTER_TITLE_RE.match(paragraph):
            break
        normalized = canonicalize(paragraph)
        if normalized in title_markers or normalized == book_marker:
            continue
        if is_map_like(paragraph) or is_subheading_like(paragraph):
            continue
        paragraph = _strip_leading_paragraph_prefix(paragraph, chapter_title).strip()
        if not paragraph:
            continue
        if canonicalize(paragraph) in title_markers:
            continue
        if _skip_full_paragraph(paragraph):
            continue
        rendered.append(paragraph)
    return rendered


def _clean_outline_book(path: Path, output_dir: Path) -> Path:
    reader = PdfReader(str(path))
    outline = [bookmark for bookmark in flatten_outline(path) if bookmark.page is not None]
    chapter_selector = _chapter_selector_for(path)
    selected = [bookmark for bookmark in outline if chapter_selector(bookmark)]
    if not selected:
        raise ValueError(f"No selected outline chapters for {path.name}")
    last_selected_page = selected[-1].page
    trailing_back_matter_pages = [
        bookmark.page
        for bookmark in outline
        if bookmark.page > last_selected_page and BACK_MATTER_TITLE_RE.match(bookmark.title.strip())
    ]
    last_page_inclusive = (
        min(trailing_back_matter_pages) - 1 if trailing_back_matter_pages else len(reader.pages) - 1
    )
    spec = BookSpec(
        pdf_path=path,
        output_path=output_dir / derive_output_filename(path),
        book_title=_book_title_for(path),
        chapter_selector=chapter_selector,
        last_page_inclusive=last_page_inclusive,
    )
    rendered_chapters: list[str] = []
    for index, bookmark in enumerate(selected):
        next_page = (
            selected[index + 1].page - 1 if index + 1 < len(selected) else last_page_inclusive
        )
        lines = extract_lines(reader, bookmark.page, next_page)
        lines = _repair_opening_lines(lines, bookmark.title)
        lines = clean_lines(lines, book_title=spec.book_title, chapter_title=bookmark.title)
        lines = _strip_leading_outline_artifacts(lines, bookmark.title)
        cleaned_paragraphs = _clean_outline_paragraphs(path, bookmark.title, lines, spec.book_title)
        body = "\n\n".join(cleaned_paragraphs).strip()
        if body:
            rendered_chapters.append(f"Chapter {len(rendered_chapters) + 1}\n\n{body}")

    rendered = "\n\n".join(rendered_chapters).strip() + "\n"
    spec.output_path.write_text(rendered.replace("\x00", ""), encoding="utf-8")
    return spec.output_path


def _strip_leading_outline_artifacts(lines: list[str], chapter_title: str) -> list[str]:
    remaining = lines[:]
    markers = {canonicalize(fragment) for fragment in title_fragments(chapter_title)}
    while remaining:
        line = _strip_watermark(remaining[0])
        if not line:
            remaining.pop(0)
            continue
        normalized = canonicalize(line)
        if normalized in markers:
            remaining.pop(0)
            continue
        compact = line.replace(" ", "")
        if canonicalize(compact) in markers:
            remaining.pop(0)
            continue
        if SPACED_HEADING_RE.fullmatch(line) or (
            UPPER_HEADING_RE.fullmatch(line) and len(line.split()) <= 8
        ):
            remaining.pop(0)
            continue
        break
    return remaining


@lru_cache(maxsize=None)
def _ocr_page(path: Path, page_number: int) -> str:
    with tempfile.TemporaryDirectory(prefix="india-wins-ocr-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        image_base = tmp_dir / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                "200",
                "-png",
                str(path),
                str(image_base),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        image_path = image_base.with_name(f"{image_base.name}-{page_number:03d}.png")
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "6"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def _normalize_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_india_wins_chapters(path: Path) -> list[OcrChapter]:
    contents_text = _normalize_ocr_text(_ocr_page(path, 7))
    chapters: list[OcrChapter] = []
    for raw_line in contents_text.splitlines():
        line = " ".join(raw_line.split())
        match = CHAPTER_LINE_RE.match(line)
        if match is None:
            continue
        title = match.group("title").replace("Warin", "War in").replace("Congressin", "Congress in")
        title = title.replace("TheInterim", "The Interim").replace(
            "TheEndofaDream", "The End of a Dream"
        )
        title = title.replace("The SimlaConference", "The Simla Conference")
        chapters.append(
            OcrChapter(
                number=int(match.group("number")),
                title=title.strip(),
                printed_page=int(line.rsplit(" ", 1)[1]),
            )
        )
    if len(chapters) != 16:
        raise ValueError(
            f"Expected 16 chapter entries for India Wins Freedom, found {len(chapters)}"
        )
    return chapters


def _clean_india_wins_freedom(path: Path, output_dir: Path) -> Path:
    chapters = _extract_india_wins_chapters(path)
    pdf_starts = [chapter.printed_page + INDIA_WINS_PAGE_OFFSET for chapter in chapters]

    rendered: list[str] = []
    for index, start_page in enumerate(pdf_starts):
        end_page = pdf_starts[index + 1] - 1 if index + 1 < len(pdf_starts) else 249
        page_chunks: list[str] = []
        for page_number in range(start_page, end_page + 1):
            page_text = _normalize_ocr_text(_ocr_page(path, page_number))
            lines = [" ".join(line.split()) for line in page_text.splitlines()]
            cleaned_lines: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if cleaned_lines and cleaned_lines[-1] != "":
                        cleaned_lines.append("")
                    continue
                lower = stripped.casefold()
                if "pdfbooksfree" in lower or "orient longman" in lower:
                    continue
                if stripped == str(chapters[index].printed_page):
                    continue
                if PRINTED_PAGE_RE.fullmatch(stripped) or ROMAN_PAGE_RE.fullmatch(stripped):
                    continue
                if chapters[index].title.casefold().replace(" ", "") in lower.replace(" ", ""):
                    continue
                if lower in {"prospectus", "preface to the 1959 edition"}:
                    continue
                cleaned_lines.append(stripped)

            paragraphs: list[str] = []
            current: list[str] = []
            for line in cleaned_lines:
                if line:
                    current.append(line)
                    continue
                if current:
                    paragraphs.append(" ".join(current).strip())
                    current = []
            if current:
                paragraphs.append(" ".join(current).strip())

            for paragraph in paragraphs:
                if (
                    re.fullmatch(r"[A-Z][A-Za-z .,'’\-]{0,50}", paragraph)
                    and len(paragraph.split()) <= 8
                ):
                    continue
                page_chunks.append(paragraph)

        body = "\n\n".join(chunk for chunk in page_chunks if len(chunk.split()) >= 6).strip()
        rendered.append(f"Chapter {index + 1}\n\n{body}")

    output_path = output_dir / derive_output_filename(path)
    output_path.write_text("\n\n".join(rendered).strip() + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_pdfs = _recent_pdf_paths()
    for path in recent_pdfs:
        if "India_Wins_Freedom" in path.name:
            output_path = _clean_india_wins_freedom(path, args.output_dir)
        else:
            output_path = _clean_outline_book(path, args.output_dir)
        print(f"{path.name} -> {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
