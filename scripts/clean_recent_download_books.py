#!/usr/bin/env python3
"""Clean the newest downloaded books into chapter-only text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from podcast_agent.utils.book_cleaning import clean_book_file, derive_output_filename, list_recent_book_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the newest downloaded book files into sample_books chapter text."
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("/Users/baqir/Downloads"),
        help="Directory to scan for downloaded book files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/baqir/Python/podcast_agent/sample_books"),
        help="Directory that receives cleaned .txt outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=13,
        help="Number of newest book-like files to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = list_recent_book_files(args.downloads_dir, limit=args.limit)
    if not sources:
        print("No supported book files found.")
        return 1

    failures: list[tuple[Path, str]] = []
    for source_path in sources:
        output_path = args.output_dir / derive_output_filename(source_path)
        try:
            result = clean_book_file(source_path, args.output_dir)
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            failures.append((source_path, str(exc)))
            continue
        print(
            f"{source_path.name} -> {result.output_path.name} | "
            f"chapters={result.chapter_count} words={result.word_count}"
        )

    if failures:
        print("\nFailures:", file=sys.stderr)
        for source_path, message in failures:
            print(f"- {source_path.name}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
