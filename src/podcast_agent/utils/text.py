"""Text helpers shared across the podcast agent."""

from __future__ import annotations


def truncate_words(text: str, max_words: int) -> str:
    """Return text truncated to a maximum word count."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()
