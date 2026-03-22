"""LLM concurrency gate tests."""

from __future__ import annotations

import threading
import time

from podcast_agent.llm.concurrency import configure_llm_semaphore, llm_semaphore


def test_llm_semaphore_caps_concurrency() -> None:
    configure_llm_semaphore(2)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal active, max_active
        with llm_semaphore():
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert max_active == 2
    finally:
        configure_llm_semaphore(3)
