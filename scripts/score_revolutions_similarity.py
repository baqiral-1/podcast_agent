#!/usr/bin/env python3
"""Score completed runs against a benchmark excerpt using only episode_script artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podcast_agent.eval.revolutions_similarity import score_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare run episode_script artifacts to a benchmark excerpt using a reproducible "
            "feature-based similarity score."
        )
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        required=True,
        help="Text file containing the benchmark excerpt.",
    )
    parser.add_argument(
        "--include-intro-section",
        action="store_true",
        help="Include prose_sections[0] when scoring. Default ignores the intro section.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more completed run directories under runs/<project-id>.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark_text = args.benchmark_file.read_text(encoding="utf-8")
    payload = {
        "benchmark_file": str(args.benchmark_file),
        "ignore_intro_section": not args.include_intro_section,
        "runs": [
            score_run_dir(
                run_dir,
                benchmark_text,
                ignore_intro_section=not args.include_intro_section,
            ).to_dict()
            for run_dir in args.run_dirs
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
