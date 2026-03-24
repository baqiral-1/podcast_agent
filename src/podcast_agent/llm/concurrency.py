"""Process-wide concurrency gate for LLM calls."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator

_lock = threading.Lock()
_default_semaphore: threading.BoundedSemaphore | None = threading.BoundedSemaphore(6)
_schema_semaphores: dict[str, threading.BoundedSemaphore | None] = {}


def configure_llm_semaphore(
    default_limit: int, *, per_schema: dict[str, int] | None = None
) -> None:
    """Configure the global semaphore sizes for LLM calls."""

    if default_limit < 1:
        raise ValueError("LLM semaphore limit must be >= 1.")
    per_schema = per_schema or {}
    for schema_name, limit in per_schema.items():
        if limit < 1:
            raise ValueError(f"LLM semaphore limit for '{schema_name}' must be >= 1.")
    global _default_semaphore, _schema_semaphores
    with _lock:
        _default_semaphore = threading.BoundedSemaphore(default_limit)
        _schema_semaphores = {
            schema_name: threading.BoundedSemaphore(limit)
            for schema_name, limit in per_schema.items()
        }


@contextmanager
def llm_semaphore() -> Iterator[None]:
    """Context manager that enforces the default LLM concurrency cap."""

    semaphore = _default_semaphore
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


@contextmanager
def llm_semaphore_for(schema_name: str) -> Iterator[None]:
    """Context manager that enforces the LLM concurrency cap for a schema."""

    semaphore = _schema_semaphores.get(schema_name, _default_semaphore)
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
