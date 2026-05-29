#!/usr/bin/env python3
"""Print deterministic 200-word validation samples from cleaned book files."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path


_CHAPTER_RE = re.compile(r"(?m)^Chapter (\d+):\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--words", type=int, default=200)
    return parser.parse_args()


def _chapter_token_windows(text: str, words: int) -> list[tuple[int, int, list[str]]]:
    parts = _CHAPTER_RE.split(text)
    windows: list[tuple[int, int, list[str]]] = []
    token_cursor = 0
    for index in range(1, len(parts), 2):
        chapter_number = int(parts[index])
        body_tokens = re.findall(r"\S+", parts[index + 1])
        if len(body_tokens) < words:
            token_cursor += len(body_tokens)
            continue
        windows.append((chapter_number, token_cursor, body_tokens))
        token_cursor += len(body_tokens)
    return windows


def sample_text(path: Path, segments: int, words: int) -> str:
    text = path.read_text(encoding="utf-8")
    chapter_windows = _chapter_token_windows(text, words)
    if not chapter_windows:
        raise ValueError(f"{path} has no chapter body with at least {words} words")

    seed = int(hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    picks: list[tuple[int, int, str]] = []
    for _ in range(segments):
        chapter_number, token_cursor, body_tokens = rng.choice(chapter_windows)
        max_start = len(body_tokens) - words
        start = rng.randint(0, max_start)
        picks.append(
            (chapter_number, token_cursor + start, " ".join(body_tokens[start : start + words]))
        )

    picks.sort(key=lambda item: item[1])
    rendered: list[str] = []
    for index, (chapter_number, start_word, chunk) in enumerate(picks, start=1):
        rendered.append(
            f"== {path.name} segment {index} chapter={chapter_number} start_word={start_word + 1} ==\n{chunk}\n"
        )
    return "\n".join(rendered)


def main() -> int:
    args = parse_args()
    for path in args.paths:
        print(sample_text(path, args.segments, args.words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
