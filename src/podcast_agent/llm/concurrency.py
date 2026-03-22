"""Process-wide concurrency gate for LLM calls."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator

_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = threading.BoundedSemaphore(3)


def configure_llm_semaphore(limit: int) -> None:
    """Configure the global semaphore size for LLM calls."""

    if limit < 1:
        raise ValueError("LLM semaphore limit must be >= 1.")
    global _semaphore
    with _lock:
        _semaphore = threading.BoundedSemaphore(limit)


@contextmanager
def llm_semaphore() -> Iterator[None]:
    """Context manager that enforces the global LLM concurrency cap."""

    semaphore = _semaphore
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
