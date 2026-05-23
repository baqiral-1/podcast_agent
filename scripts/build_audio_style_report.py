#!/usr/bin/env python3
"""Generate a standalone HTML audio-style report for one or more completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podcast_agent.eval.audio_style_report import (
    DEFAULT_SCRIPT_ARTIFACTS,
    build_audio_style_payload,
    render_audio_style_html,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rubric-based HTML report that scores completed run scripts for "
            "audio fitness and target-style similarity."
        )
    )
    parser.add_argument(
        "--title",
        help="Optional page title. Defaults to a title derived from the selected run dirs.",
    )
    parser.add_argument(
        "--target-snippet-file",
        type=Path,
        help="Optional custom text file used as the target spoken-host style reference.",
    )
    parser.add_argument(
        "--tss-v2-snippet-file",
        type=Path,
        help="Optional override for the bundled TSSv2 comparison sample.",
    )
    parser.add_argument(
        "--tss-v2-label",
        help="Optional label for the TSSv2 comparison sample. Defaults to the bundled label or the file stem.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        required=True,
        help="Where to write the standalone report HTML page.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional companion JSON payload path. Defaults to the HTML path with a .json suffix.",
    )
    parser.add_argument(
        "--script-artifact",
        dest="script_artifacts",
        action="append",
        help=(
            "Preferred script artifact to evaluate. Repeat to provide fallbacks. "
            "Defaults to episode_script.json."
        ),
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more completed run directories under runs/<project-id>.",
    )
    return parser.parse_args()


def _default_title(run_dirs: list[Path]) -> str:
    names = ", ".join(run_dir.name for run_dir in run_dirs)
    return f"Audio Style Report: {names}"


def main() -> None:
    args = _parse_args()
    script_artifacts = tuple(args.script_artifacts or DEFAULT_SCRIPT_ARTIFACTS)
    target_snippet = (
        args.target_snippet_file.read_text(encoding="utf-8")
        if args.target_snippet_file is not None
        else None
    )
    comparison_target_snippet = (
        args.tss_v2_snippet_file.read_text(encoding="utf-8")
        if args.tss_v2_snippet_file is not None
        else None
    )
    comparison_target_label = (
        args.tss_v2_label
        or (args.tss_v2_snippet_file.stem if args.tss_v2_snippet_file is not None else None)
    )
    payload = build_audio_style_payload(
        args.run_dirs,
        title=args.title or _default_title(args.run_dirs),
        target_snippet=target_snippet,
        comparison_target_snippet=comparison_target_snippet,
        comparison_target_label=comparison_target_label,
        script_artifacts=script_artifacts,
    )
    html = render_audio_style_html(payload)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")

    output_json = args.output_json or args.output_html.with_suffix(".json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
