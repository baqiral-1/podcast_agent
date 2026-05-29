#!/usr/bin/env python3
"""Generate a standalone blind A/B script comparison page for two run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from podcast_agent.eval.blind_script_comparison import (
    DEFAULT_COUNTS_BY_EPISODE,
    write_blind_comparison_outputs,
)


def _parse_counts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:  # pragma: no cover - argparse surfaces the message
        raise argparse.ArgumentTypeError(
            "counts must be a comma-separated list of integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("counts must include at least one integer")
    return values


def _default_title(run_dirs: tuple[Path, Path]) -> str:
    left, right = (run_dir.name.removeprefix("iranian_revolution_") for run_dir in run_dirs)
    return f"Blind Comparison: {left} vs {right}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone HTML page for blind side-by-side script comparisons "
            "between exactly two completed run directories."
        )
    )
    parser.add_argument(
        "--title",
        help="Optional page title. Defaults to a title derived from the selected run dirs.",
    )
    parser.add_argument(
        "--subtitle",
        help="Optional subtitle shown beneath the page title.",
    )
    parser.add_argument(
        "--comparison-prompt",
        help="Optional rating prompt shown above every A/B comparison.",
    )
    parser.add_argument(
        "--counts-by-episode",
        type=_parse_counts,
        default=DEFAULT_COUNTS_BY_EPISODE,
        help=(
            "Comma-separated comparison counts for each shared episode. "
            "Defaults to 13,13,12,12 for 50 total comparisons."
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        required=True,
        help="Where to write the standalone comparison HTML page.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional companion JSON payload path. Defaults to the HTML path with a .json suffix.",
    )
    parser.add_argument(
        "run_dirs",
        nargs=2,
        type=Path,
        help="Exactly two completed run directories under runs/<project-id>.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = tuple(args.run_dirs)
    output_json = args.output_json or args.output_html.with_suffix(".json")
    write_blind_comparison_outputs(
        run_dirs=run_dirs,
        output_html=args.output_html,
        output_json=output_json,
        title=args.title or _default_title(run_dirs),
        subtitle=args.subtitle,
        comparison_prompt=args.comparison_prompt,
        counts_by_episode=args.counts_by_episode,
    )


if __name__ == "__main__":
    main()
