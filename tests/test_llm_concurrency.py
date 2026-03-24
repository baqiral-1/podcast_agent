"""LLM concurrency gate tests."""

from __future__ import annotations

import threading
import time

from podcast_agent.llm.concurrency import configure_llm_semaphore, llm_semaphore_for


def _max_concurrency(schema_name: str, thread_count: int = 6) -> int:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal active, max_active
        with llm_semaphore_for(schema_name):
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return max_active


def test_llm_semaphore_caps_default_concurrency() -> None:
    configure_llm_semaphore(2)
    try:
        assert _max_concurrency("series_plan") == 2
    finally:
        configure_llm_semaphore(3)


def test_llm_semaphore_respects_schema_overrides() -> None:
    configure_llm_semaphore(1, per_schema={"structured_chapter": 2})
    try:
        assert _max_concurrency("series_plan") == 1
        assert _max_concurrency("structured_chapter") == 2
    finally:
        configure_llm_semaphore(3)
