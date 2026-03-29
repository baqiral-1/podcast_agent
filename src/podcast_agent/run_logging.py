"""Run-scoped logging for prompts, commands, and stage events."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


class RunLogger:
    """Append-only logger that writes one JSON event per line."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.run_id: str | None = None
        self.log_path: Path | None = None
        self._pending_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._context: ContextVar[dict[str, Any]] = ContextVar("run_logger_context", default={})

    def bind_run(self, run_id: str) -> None:
        """Bind the logger to a concrete run directory and flush buffered events."""

        with self._lock:
            self.run_id = run_id
            run_dir = self.artifact_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = run_dir / "run.log"
            pending_events = list(self._pending_events)
            self._pending_events.clear()
            for event in pending_events:
                self._write_event_locked(event)

    def log(self, event_type: str, **payload: Any) -> None:
        """Record an event immediately or buffer it until the run is initialized."""

        context_payload = self._context.get()
        resolved_payload = {
            **context_payload,
            **payload,
        }
        resolved_payload.setdefault("book_id", None)
        resolved_payload.setdefault("book_title", None)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": resolved_payload,
        }
        with self._lock:
            if self.log_path is None:
                self._pending_events.append(event)
                return
            self._write_event_locked(event)

    @contextmanager
    def context(self, **payload: Any):
        """Temporarily attach default payload fields to subsequent log events."""

        merged = {**self._context.get(), **payload}
        token = self._context.set(merged)
        try:
            yield
        finally:
            self._context.reset(token)

    def _write_event_locked(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            raise RuntimeError("Run log path is not initialized.")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str))
            handle.write("\n")


def submit_with_context(executor, fn: Callable[..., Any], *args: Any, **kwargs: Any):
    """Submit work to an executor while preserving the current contextvars state."""

    context = copy_context()

    def run_in_context() -> Any:
        return context.run(fn, *args, **kwargs)

    return executor.submit(run_in_context)
