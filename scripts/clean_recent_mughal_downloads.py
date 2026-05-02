#!/usr/bin/env python3
"""Clean the five newest Mughal-history downloads into chapter-only text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from podcast_agent.utils.mughal_cleanup import clean_mughal_book_file


TARGET_FILENAMES = (
    "dokumen.pub_twilight-of-the-mughuls-studies-in-late-mughul-delhi-0685500209-9780685500200.pdf",
    "indian-society-and-the-making-of-the-british-empire.pdf",
    "dokumen.pub_the-eighteenth-century-in-indian-history-evolution-or-revolution-9780195678147-0195678141.pdf",
    "dokumen.pub_delhi-between-two-empires-1803-1930-society-government-and-urban-growth.pdf",
    "Later of Mughals-Vol.1.pdf",
)
SECOND_PASS_FILENAMES = (
    "indian-society-and-the-making-of-the-british-empire.pdf",
    "dokumen.pub_the-eighteenth-century-in-indian-history-evolution-or-revolution-9780195678147-0195678141.pdf",
    "dokumen.pub_delhi-between-two-empires-1803-1930-society-government-and-urban-growth.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the current Mughal-history Downloads set into sample_books/mughal."
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Directory containing the source PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_books") / "mughal",
        help="Directory that receives cleaned .txt outputs.",
    )
    parser.add_argument(
        "--scope",
        choices=("second-pass", "all"),
        default="second-pass",
        help="Which locked Mughal input set to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    target_filenames = SECOND_PASS_FILENAMES if args.scope == "second-pass" else TARGET_FILENAMES

    for filename in target_filenames:
        source_path = args.downloads_dir / filename
        if not source_path.exists():
            failures.append((filename, "source file does not exist"))
            continue

        try:
            result = clean_mughal_book_file(source_path, args.output_dir)
        except Exception as exc:
            failures.append((filename, str(exc)))
            continue

        print(
            f"{source_path.name} -> {result.output_path.name} | "
            f"method={result.extraction_method} chapters={result.chapter_count} words={result.word_count}",
            flush=True,
        )

    if failures:
        print("\nFailures:", file=sys.stderr)
        for filename, message in failures:
            print(f"- {filename}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
