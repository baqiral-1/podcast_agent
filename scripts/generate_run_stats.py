#!/usr/bin/env python3
"""Generate a lightweight standalone HTML statistics page from a pipeline run directory."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

CORE_ARTIFACTS = (
    "thematic_project.json",
    "thematic_axes.json",
    "thematic_corpus.json",
    "synthesis_primitives.json",
    "synthesis_map.json",
    "narrative_strategy.json",
    "series_plan.json",
    "run.log",
)


class RunStatsError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run-stats.html for a completed run directory.")
    parser.add_argument("run_dir", help="Path to runs/<project-id> directory")
    parser.add_argument("--output", help="Optional output HTML path. Defaults to <run_dir>/run-stats.html.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _check_run_dir(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        raise RunStatsError(f"Run directory does not exist: {run_dir}")
    missing = [name for name in CORE_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise RunStatsError(f"Run directory is missing required artifacts: {', '.join(missing)}")


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _render_episode_cards(run_dir: Path, strategy: dict[str, Any]) -> str:
    cards: list[str] = []
    for episode in strategy.get("episodes", []):
        ep_dir = run_dir / "episodes" / str(episode.get("episode_number"))
        script = _load_optional_json(ep_dir / "episode_script.json") or {}
        spoken = _load_optional_json(ep_dir / "spoken_script.json") or {}
        episode_spine = episode.get("episode_spine", {})
        core_primitive_ids = episode_spine.get("core_primitive_ids", [])
        cards.append(
            """
            <section class='card'>
              <h3>{title}</h3>
              <p><strong>Driving question:</strong> {question}</p>
              <p><strong>Core primitives:</strong> {core_primitive_count}</p>
              <p><strong>Script sections:</strong> {section_count}</p>
              <p><strong>Spoken sections:</strong> {spoken_count}</p>
            </section>
            """.format(
                title=_escape(episode.get("title", f"Episode {episode.get('episode_number', '?')}")),
                question=_escape(episode.get("driving_question", "")),
                core_primitive_count=len(core_primitive_ids),
                section_count=len(script.get("prose_sections", [])),
                spoken_count=len(spoken.get("sections", [])),
            )
        )
    return "\n".join(cards)


def _render_html(run_dir: Path) -> str:
    project = _load_json(run_dir / "thematic_project.json")
    primitives = _load_json(run_dir / "synthesis_primitives.json")
    synthesis_map = _load_json(run_dir / "synthesis_map.json")
    strategy = _load_json(run_dir / "narrative_strategy.json")
    series_plan = _load_json(run_dir / "series_plan.json")
    episodes = strategy.get("episodes", [])
    planned_episodes = series_plan.get("episodes", []) if isinstance(series_plan, dict) else []
    primitives_by_family = (
        primitives.get("primitives_by_family", {})
        if isinstance(primitives, dict)
        else {}
    )
    primitive_count_rows = "".join(
        f"<p>{_escape(family)}: {len(items)}</p>"
        for family, items in primitives_by_family.items()
    )
    if not primitive_count_rows:
        primitive_count_rows = "<p>No family primitives found.</p>"

    return f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Run Stats</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 900px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #ddd; padding: 1rem; border-radius: 10px; background: #faf8f2; }}
    code {{ background: #f0eee8; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{_escape(project.get('theme', 'Run Stats'))}</h1>
  <p><strong>Project:</strong> <code>{_escape(project.get('project_id'))}</code></p>
  <p><strong>Status:</strong> {_escape(project.get('status'))}</p>
  <div class='grid'>
    <section class='card'><h2>Primitives</h2>{primitive_count_rows}</section>
    <section class='card'><h2>Synthesis Map</h2><p>Quality score: {_escape(synthesis_map.get('quality_score'))}</p></section>
    <section class='card'><h2>Strategy</h2><p>Type: {_escape(strategy.get('strategy_type'))}</p><p>Episodes: {len(episodes)}</p></section>
    <section class='card'><h2>Planning</h2><p>Planned episodes: {len(planned_episodes)}</p></section>
  </div>
  <h2>Episodes</h2>
  {_render_episode_cards(run_dir, strategy)}
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    _check_run_dir(run_dir)
    output_path = Path(args.output).resolve() if args.output else run_dir / "run-stats.html"
    output_path.write_text(_render_html(run_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
