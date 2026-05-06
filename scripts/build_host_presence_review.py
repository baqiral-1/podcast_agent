#!/usr/bin/env python3
"""Generate a standalone host-presence RLHF review page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podcast_agent.eval.host_presence_review import build_review_payload, render_review_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone HTML page for rating host-presence snippets across one or "
            "more completed run directories."
        )
    )
    parser.add_argument(
        "--title",
        help="Optional page title. Defaults to a title derived from the selected run dirs.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        required=True,
        help="Where to write the standalone review HTML page.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional companion JSON payload path.",
    )
    parser.add_argument(
        "--script-artifact",
        dest="script_artifacts",
        action="append",
        help=(
            "Preferred script artifact to extract snippets from. Repeat to provide fallbacks. "
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


def main() -> None:
    args = _parse_args()
    script_artifacts = tuple(args.script_artifacts or ["episode_script.json"])
    payload = build_review_payload(
        args.run_dirs,
        title=args.title,
        script_artifacts=script_artifacts,
    )
    html = render_review_html(payload)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
