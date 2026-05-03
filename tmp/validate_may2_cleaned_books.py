#!/usr/bin/env python3
"""Emit deterministic validation reports for the May 2 cleaned book batch."""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path


CHAPTER_RE = re.compile(r"^Chapter (?P<number>\d+)$", re.MULTILINE)
WORD_RE = re.compile(r"\S+")
DEFAULT_INPUT_DIR = Path("sample_books") / "temp_clean"
DEFAULT_REPORT_DIR = Path("tmp") / "may2_clean_validation"
SAMPLE_COUNT = 50
SAMPLE_WORDS = 500
EDGE_WORDS = 200
CHAPTER_EDGE_WORDS = 120
BASE_SEED = 20260502


@dataclass(frozen=True)
class Chapter:
    number: int
    body: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--sample-words", type=int, default=SAMPLE_WORDS)
    return parser.parse_args()


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _slice_words(words: list[str], start: int, size: int) -> str:
    return " ".join(words[start : start + size]).strip()


def _parse_chapters(text: str) -> list[Chapter]:
    matches = list(CHAPTER_RE.finditer(text))
    chapters: list[Chapter] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chapters.append(Chapter(number=int(match.group("number")), body=body))
    return chapters


def _sample_starts(total_words: int, sample_words: int, sample_count: int, seed: int) -> list[int]:
    if total_words <= sample_words:
        return [0]
    rng = random.Random(seed)
    upper = total_words - sample_words
    starts = sorted({rng.randint(0, upper) for _ in range(sample_count * 3)})
    if len(starts) < sample_count:
        starts.extend(range(0, upper + 1, max(1, upper // sample_count or 1)))
    return sorted(starts[:sample_count])


def _render_report(path: Path, sample_count: int, sample_words: int) -> str:
    text = path.read_text(encoding="utf-8")
    chapters = _parse_chapters(text)
    all_words = _words(text)
    seed = BASE_SEED + sum(ord(char) for char in path.name)
    lines: list[str] = [
        f"FILE: {path.name}",
        f"TOTAL_WORDS: {len(all_words)}",
        f"CHAPTERS: {len(chapters)}",
        "",
        "BOOK_BEGINNING:",
        _slice_words(all_words, 0, EDGE_WORDS),
        "",
        "BOOK_ENDING:",
        _slice_words(all_words, max(0, len(all_words) - EDGE_WORDS), EDGE_WORDS),
        "",
        "CHAPTER_STARTS:",
    ]
    for chapter in chapters:
        chapter_words = _words(chapter.body)
        lines.extend(
            [
                f"Chapter {chapter.number} START:",
                _slice_words(chapter_words, 0, CHAPTER_EDGE_WORDS),
                "",
                f"Chapter {chapter.number} END:",
                _slice_words(chapter_words, max(0, len(chapter_words) - CHAPTER_EDGE_WORDS), CHAPTER_EDGE_WORDS),
                "",
            ]
        )
    lines.append("RANDOM_PASSAGES:")
    for index, start in enumerate(_sample_starts(len(all_words), sample_words, sample_count, seed), start=1):
        lines.extend(
            [
                f"Sample {index} START_WORD {start}:",
                _slice_words(all_words, start, sample_words),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input_dir.glob("*.txt")):
        report_path = args.report_dir / f"{path.stem}.report.txt"
        report_path.write_text(_render_report(path, args.sample_count, args.sample_words), encoding="utf-8")
        print(f"{path.name} -> {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
