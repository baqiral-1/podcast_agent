"""Recover a scene_discovery artifact from a run.log when the stage crashed.

Use when scene_discovery returned a schema-valid response (e.g. 147 candidates)
but the count guard fired and the retry never completed. The attempt-1
`llm_response` event in run.log carries the full response we'd otherwise lose.

Usage:
    python scripts/recover_scene_discovery_from_log.py <run_dir>

Writes <run_dir>/scene_discovery.json and <run_dir>/scene_discovery_diagnostics.json
so resume_from_narrative_strategy_stage can pick up from there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_agent.pipeline.orchestrator import (
    _build_scene_discovery_diagnostics,
    _save_json,
)
from podcast_agent.schemas.models import (
    PodcastMode,
    SceneDiscoveryArtifact,
)


def _find_scene_discovery_attempt(log_path: Path, attempt: int = 1) -> dict:
    """Return the parsed payload.response for the first matching llm_response."""
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("event_type") != "llm_response":
                continue
            payload = evt.get("payload") or {}
            if payload.get("schema_name") != "scene_discovery":
                continue
            # llm_response payload doesn't always carry attempt; fall back to
            # ordering — the first matching response is attempt 1.
            if payload.get("attempt") not in (None, attempt):
                continue
            return payload["response"]
    raise RuntimeError(
        f"No scene_discovery llm_response found in {log_path} for attempt={attempt}."
    )


def _resolve_mode(run_dir: Path) -> PodcastMode:
    project_path = run_dir / "thematic_project.json"
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    config = payload.get("config") or {}
    mode_str = config.get("podcast_mode") or "full"
    return PodcastMode(mode_str)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(argv[1]).resolve()
    log_path = run_dir / "run.log"
    if not log_path.exists():
        raise SystemExit(f"run.log not found at {log_path}")

    raw_response = _find_scene_discovery_attempt(log_path, attempt=1)
    artifact = SceneDiscoveryArtifact.model_validate(raw_response)
    mode = _resolve_mode(run_dir)
    diagnostics = _build_scene_discovery_diagnostics(artifact=artifact, mode=mode)

    artifact_path = run_dir / "scene_discovery.json"
    diagnostics_path = run_dir / "scene_discovery_diagnostics.json"
    _save_json(artifact_path, artifact)
    _save_json(diagnostics_path, diagnostics)

    print(
        f"Recovered {len(artifact.candidates)} candidates → {artifact_path}\n"
        f"Diagnostics → {diagnostics_path}\n"
        f"Warnings: {diagnostics.get('warning_count', 0)}"
    )
    for w in diagnostics.get("warnings", []):
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
