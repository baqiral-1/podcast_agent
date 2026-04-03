"""LangChain cache configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.globals import set_llm_cache

from podcast_agent.config import LangChainConfig


def _build_cache(config: LangChainConfig) -> Any:
    backend = config.cache_backend.strip().lower()
    if backend in {"none", "off", "disabled"}:
        return None
    if backend in {"memory", "inmemory"}:
        from langchain_community.cache import InMemoryCache

        return InMemoryCache()
    if backend in {"sqlite", "sqlite3"}:
        from langchain_community.cache import SQLiteCache

        path = Path(config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteCache(database_path=str(path))
    if backend == "redis":
        from langchain_community.cache import RedisCache

        if not config.cache_redis_url:
            raise ValueError("LANGCHAIN_CACHE_REDIS_URL is required for redis cache backend.")
        return RedisCache(redis_url=config.cache_redis_url)
    raise ValueError(f"Unsupported LangChain cache backend '{config.cache_backend}'.")


def configure_llm_cache(config: LangChainConfig) -> None:
    if not config.cache_enabled:
        set_llm_cache(None)
        return
    cache = _build_cache(config)
    set_llm_cache(cache)

