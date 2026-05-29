#!/usr/bin/env python3
"""Generate a standalone scene-type audit page for a completed run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from podcast_agent.eval.scene_type_audit import (
    DEFAULT_SCENE_AUDIT_ARTIFACTS,
    build_scene_type_audit_payload,
    render_scene_type_audit_html,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scene-type-audit.html for a completed run directory."
    )
    parser.add_argument("run_dir", help="Path to runs/<project-id> directory")
    parser.add_argument(
        "--output",
        help="Optional output HTML path. Defaults to <run_dir>/scene-type-audit.html.",
    )
    parser.add_argument(
        "--script-artifact",
        dest="script_artifacts",
        action="append",
        help=(
            "Preferred script artifact to audit. Repeat to provide fallbacks. "
            "Defaults to style_audited_script.json then episode_script.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    payload = build_scene_type_audit_payload(
        run_dir,
        script_artifacts=tuple(args.script_artifacts or DEFAULT_SCENE_AUDIT_ARTIFACTS),
    )
    output_path = Path(args.output).resolve() if args.output else run_dir / "scene-type-audit.html"
    output_path.write_text(render_scene_type_audit_html(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
