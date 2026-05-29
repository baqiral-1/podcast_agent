#!/usr/bin/env python3
"""Generate a standalone actor-explanation audit page for a run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from podcast_agent.eval.actor_explanation_audit import (
    build_actor_explanation_audit_payload,
    render_actor_explanation_audit_html,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate actor-explanation-audit.html for a completed run directory."
    )
    parser.add_argument("run_dir", help="Path to runs/<project-id> directory")
    parser.add_argument(
        "--output",
        help="Optional output HTML path. Defaults to <run_dir>/actor-explanation-audit.html.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    payload = build_actor_explanation_audit_payload(run_dir)
    output_path = (
        Path(args.output).resolve() if args.output else run_dir / "actor-explanation-audit.html"
    )
    output_path.write_text(
        render_actor_explanation_audit_html(payload),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
