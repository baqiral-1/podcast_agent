"""Deterministic lightweight embeddings for local development."""

from __future__ import annotations

import hashlib


def embed_text(text: str, dimensions: int = 8) -> list[float]:
    """Create a stable pseudo-embedding from text."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        byte = digest[index]
        values.append(round((byte / 255.0) * 2.0 - 1.0, 6))
    return values
