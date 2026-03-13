"""Run-scoped logging for prompts, commands, and stage events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """Append-only logger that writes one JSON event per line."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.run_id: str | None = None
        self.log_path: Path | None = None
        self._pending_events: list[dict[str, Any]] = []

    def bind_run(self, run_id: str) -> None:
        """Bind the logger to a concrete run directory and flush buffered events."""

        self.run_id = run_id
        run_dir = self.artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "run.log"
        for event in self._pending_events:
            self._write_event(event)
        self._pending_events.clear()

    def log(self, event_type: str, **payload: Any) -> None:
        """Record an event immediately or buffer it until the run is initialized."""

        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        if self.log_path is None:
            self._pending_events.append(event)
            return
        self._write_event(event)

    def _write_event(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            raise RuntimeError("Run log path is not initialized.")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str))
            handle.write("\n")
