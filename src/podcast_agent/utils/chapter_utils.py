"""Deterministic chapter and section detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re


_PAGE_MARKER_RE = re.compile(r"^\[Page \d+\]$", re.IGNORECASE)
_CHAPTER_HEADING_RE = re.compile(
    r"^(chapter)\s+((?:\d+)|(?:[ivxlcdm]+))\b(?:\s*[:.\-]\s*|\s+)?(.*)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^((?:\d+)|(?:[ivxlcdm]+))[.\-:]\s+(.+)$", re.IGNORECASE)
_NAMED_SECTION_RE = re.compile(
    r"^(prologue|introduction|foreword|preface|appendix(?:\s+[a-z0-9]+)?|epilogue|afterword|conclusion)\b"
    r"(?:\s*[:.\-]\s*|\s+)?(.*)$",
    re.IGNORECASE,
)
_BAD_HEADING_FRAGMENT_RE = re.compile(r"\b(contents|acknowledgements|notes|index)\b", re.IGNORECASE)
_TOC_NAMED_ENTRY_RE = re.compile(
    r"(?i)(prologue|introduction|foreword|preface|appendix(?:\s+[a-z0-9]+)?|epilogue|afterword|conclusion):\s*(.+?)"
    r"(?=\s+(?:\d{1,2}|[ivxlcdm]+)\.\s+|\s+(?:prologue|introduction|foreword|preface|appendix(?:\s+[a-z0-9]+)?|epilogue|afterword|conclusion):|$)"
)
_TOC_NUMBERED_ENTRY_RE = re.compile(
    r"(?i)((?:\d{1,2})|(?:[ivxlcdm]+))\.\s+(.+?)"
    r"(?=\s+(?:\d{1,2}|[ivxlcdm]+)\.\s+|\s+(?:prologue|introduction|foreword|preface|appendix(?:\s+[a-z0-9]+)?|epilogue|afterword|conclusion):|$)"
)


@dataclass(frozen=True)
class SectionInput:
    """Deterministic pre-LLM section extracted from the raw source."""

    title: str
    body: str


@dataclass(frozen=True)
class TocEntry:
    """Title entry parsed from a contents page."""

    title: str


def split_into_chapters(text: str) -> list[SectionInput]:
    heading_sections = split_into_detected_headings(text)
    trimmed_text = "\n".join(_trim_back_matter([line.strip() for line in text.splitlines()])).strip()
    if not heading_sections:
        toc_sections = _split_using_table_of_contents(text)
        if toc_sections:
            return toc_sections
        return _fallback_sections(trimmed_text)
    toc_sections = _split_using_table_of_contents(text)
    if toc_sections:
        return toc_sections
    if requires_fallback_sectioning(heading_sections):
        if toc_sections:
            return toc_sections
        return _fallback_sections(trimmed_text)
    return heading_sections


def split_into_detected_headings(text: str) -> list[SectionInput]:
    lines = [line.strip() for line in text.splitlines()]
    lines = _trim_back_matter(lines)
    headings: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        if not line or _PAGE_MARKER_RE.fullmatch(line):
            continue
        matched = _match_section_heading(lines, index)
        if matched is None:
            continue
        title, consumed_lines = matched
        if headings and headings[-1][0] == index:
            continue
        headings.append((index, title, consumed_lines))

    if not headings:
        return []

    sections: list[SectionInput] = []
    for heading_index, (start_line, title, consumed_lines) in enumerate(headings):
        body_start = start_line + consumed_lines
        body_end = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        body_lines = [
            line
            for line in lines[body_start:body_end]
            if line and not _PAGE_MARKER_RE.fullmatch(line)
        ]
        body = "\n".join(body_lines).strip() or title
        sections.append(SectionInput(title=title, body=body))
    return sections


def requires_fallback_sectioning(sections: list[SectionInput]) -> bool:
    if len(sections) > 3:
        return False
    word_counts = [len(section.body.split()) for section in sections]
    total_words = sum(word_counts)
    if total_words == 0:
        return False
    largest_section = max(word_counts)
    return largest_section / total_words >= 0.8


def _split_using_table_of_contents(text: str) -> list[SectionInput]:
    lines = [line.strip() for line in text.splitlines()]
    toc_entries, body_start = _extract_toc_entries(lines)
    if len(toc_entries) < 3:
        return []

    matched_positions: list[tuple[int, str]] = []
    search_start = body_start
    for entry in toc_entries:
        matched_index = _find_title_position(lines, entry.title, search_start)
        if matched_index is None:
            continue
        matched_positions.append((matched_index, entry.title))
        search_start = matched_index + 1

    minimum_matches = max(3, len(toc_entries) // 2)
    if len(matched_positions) < minimum_matches:
        return []

    sections: list[SectionInput] = []
    for index, (start_line, title) in enumerate(matched_positions):
        end_line = matched_positions[index + 1][0] if index + 1 < len(matched_positions) else len(lines)
        body_lines = [
            line
            for line in lines[start_line:end_line]
            if line and not _PAGE_MARKER_RE.fullmatch(line)
        ]
        body = "\n".join(body_lines).strip() or title
        sections.append(SectionInput(title=title, body=body))
    return sections


def _extract_toc_entries(lines: list[str]) -> tuple[list[TocEntry], int]:
    toc_start = next((index for index, line in enumerate(lines) if "contents" in line.lower()), -1)
    if toc_start == -1:
        return ([], 0)

    page_breaks = [
        index
        for index in range(toc_start + 1, len(lines))
        if _PAGE_MARKER_RE.fullmatch(lines[index])
    ]
    candidate_ends = page_breaks[:3] + [min(len(lines), toc_start + 60)]

    best_entries: list[TocEntry] = []
    best_end = toc_start
    for toc_end in candidate_ends:
        entries = _parse_toc_entries(lines[toc_start:toc_end])
        if len(entries) > len(best_entries):
            best_entries = entries
            best_end = toc_end
    return (best_entries, best_end)


def _parse_toc_entries(lines: list[str]) -> list[TocEntry]:
    toc_text = " ".join(line for line in lines if line and not _PAGE_MARKER_RE.fullmatch(line))
    toc_text = re.sub(r"\s+", " ", toc_text)
    toc_text = re.sub(r"(?i)^contents\s*", "", toc_text)
    toc_text = re.sub(r"(?i)\b(?:acknowledgements|notes|index)\b.*$", "", toc_text)
    toc_text = re.sub(r"P\s*ART\s+[A-Z ]+\s+[–-]\s+[A-Z'’\-. ]+(?=(?:\s+\d{1,2}\.\s+|\s+Epilogue:|$))", " ", toc_text)

    matches: list[tuple[int, TocEntry]] = []
    for match in _TOC_NAMED_ENTRY_RE.finditer(toc_text):
        label = match.group(1).strip().title()
        title = _clean_toc_title(match.group(2))
        if title:
            matches.append((match.start(), TocEntry(title=f"{label}: {title}")))
    for match in _TOC_NUMBERED_ENTRY_RE.finditer(toc_text):
        chapter_token = match.group(1)
        title = _clean_toc_title(match.group(2))
        if title:
            normalized_token = str(int(chapter_token)) if chapter_token.isdigit() else chapter_token.upper()
            matches.append((match.start(), TocEntry(title=f"Chapter {normalized_token}: {title}")))
    matches.sort(key=lambda item: item[0])
    return [entry for _, entry in matches]


def _clean_toc_title(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .:-")


def _find_title_position(lines: list[str], title: str, start_index: int) -> int | None:
    match_text = title.split(": ", 1)[1] if ": " in title else title
    title_key = _normalized_match_text(match_text)
    title_words = [_normalized_match_text(word) for word in match_text.split() if len(_normalized_match_text(word)) >= 4]
    for index in range(start_index, len(lines)):
        if not lines[index] or _PAGE_MARKER_RE.fullmatch(lines[index]):
            continue
        window_lines = [
            line
            for line in lines[index : index + 4]
            if line and not _PAGE_MARKER_RE.fullmatch(line)
        ]
        heading_window = " ".join(window_lines[:3])
        heading_key = _normalized_match_text(heading_window)
        if title_key and heading_key.startswith(title_key):
            return index
        if title_key and title_key in heading_key and _looks_like_title_window(window_lines, match_text):
            return index
        if title_words and all(word in heading_key for word in title_words[: min(3, len(title_words))]) and _looks_like_title_window(window_lines, match_text):
            return index
    return None


def _normalized_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _looks_like_title_window(lines: list[str], title: str) -> bool:
    if not lines:
        return False
    title_words = [word.lower() for word in re.findall(r"[A-Za-z]+", title)]
    if lines[0].lower().startswith(("prologue", "epilogue")):
        return True
    short_lines = [line for line in lines[:2] if len(line.split()) <= 8]
    if short_lines and any(_uppercase_ratio(line) >= 0.55 for line in short_lines):
        return True
    if title_words and all(word in lines[0].lower().replace(" ", "") for word in title_words[:1]):
        return len(lines[0].split()) <= 8
    return False


def _uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = [character for character in letters if character.isupper()]
    return len(uppercase_letters) / len(letters)


def _match_section_heading(lines: list[str], index: int) -> tuple[str, int] | None:
    line = lines[index].strip()
    if not line:
        return None

    chapter_match = _CHAPTER_HEADING_RE.fullmatch(line)
    if chapter_match:
        chapter_number = chapter_match.group(2)
        if not _is_plausible_chapter_number(chapter_number):
            return None
        label = f"{chapter_match.group(1).title()} {chapter_number.upper()}"
        remainder = chapter_match.group(3).strip()
        if remainder:
            if not _is_plausible_heading_remainder(remainder):
                return None
            return (f"{label}: {remainder}", 1)
        subtitle = _heading_subtitle(lines, index + 1)
        return (f"{label}: {subtitle}" if subtitle else label, 2 if subtitle else 1)

    numbered_match = _NUMBERED_HEADING_RE.fullmatch(line)
    if numbered_match:
        remainder = numbered_match.group(2).strip()
        if not _is_plausible_chapter_number(numbered_match.group(1)):
            return None
        if not _is_plausible_heading_remainder(remainder):
            return None
        return (f"Chapter {numbered_match.group(1).upper()}: {remainder}", 1)

    named_match = _NAMED_SECTION_RE.fullmatch(line)
    if named_match:
        label = named_match.group(1).strip().title()
        remainder = named_match.group(2).strip()
        if remainder:
            if not _is_plausible_heading_remainder(remainder):
                return None
            return (f"{label}: {remainder}", 1)
        subtitle = _heading_subtitle(lines, index + 1)
        return (f"{label}: {subtitle}" if subtitle else label, 2 if subtitle else 1)
    return None


def _heading_subtitle(lines: list[str], index: int) -> str | None:
    while index < len(lines):
        candidate = lines[index].strip()
        if not candidate or _PAGE_MARKER_RE.fullmatch(candidate):
            index += 1
            continue
        if (
            _CHAPTER_HEADING_RE.fullmatch(candidate)
            or _NUMBERED_HEADING_RE.fullmatch(candidate)
            or _NAMED_SECTION_RE.fullmatch(candidate)
        ):
            return None
        if len(candidate.split()) > 12:
            return None
        return candidate
    return None


def _is_plausible_chapter_number(token: str) -> bool:
    if token.isdigit():
        return 0 < int(token) <= 200
    return True


def _is_plausible_heading_remainder(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 12:
        return False
    if _BAD_HEADING_FRAGMENT_RE.search(text):
        return False
    digit_tokens = sum(any(character.isdigit() for character in word) for word in words)
    if digit_tokens > 2:
        return False
    if text.count(":") > 1 or text.count(".") > 2 or text.count(";") > 0:
        return False
    return True


def _fallback_sections(text: str, target_words: int = 1800) -> list[SectionInput]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    sections: list[SectionInput] = []
    current_paragraphs: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        if _PAGE_MARKER_RE.fullmatch(paragraph):
            continue
        paragraph_words = len(paragraph.split())
        if current_paragraphs and current_words + paragraph_words > target_words:
            body = "\n\n".join(current_paragraphs).strip()
            sections.append(SectionInput(title=f"Section {len(sections) + 1}", body=body))
            current_paragraphs = []
            current_words = 0
        current_paragraphs.append(paragraph)
        current_words += paragraph_words

    if current_paragraphs:
        body = "\n\n".join(current_paragraphs).strip()
        sections.append(SectionInput(title=f"Section {len(sections) + 1}", body=body))
    return sections or [SectionInput(title="Section 1", body=text.strip())]


def _trim_back_matter(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if _is_back_matter_boundary(line):
            return lines[:index]
    return lines


def _is_back_matter_boundary(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if lowered in {"acknowledgements", "notes", "index"}:
        return True
    return lowered.startswith("notes ") and "prologue" in lowered
