"""Helpers for compact analysis and planning payloads."""

from __future__ import annotations

import json

from podcast_agent.schemas.models import BookAnalysis, BookStructure


def build_structure_summary(structure: BookStructure) -> dict:
    """Return a compact summary of the structured book for analysis and planning."""

    chunk_lookup = {
        chunk.chunk_id: {
            "chunk_id": chunk.chunk_id,
            "chapter_id": chunk.chapter_id,
            "chapter_title": chunk.chapter_title,
            "chapter_number": chunk.chapter_number,
            "sequence": chunk.sequence,
            "word_count": len(chunk.text.split()),
            "themes": chunk.themes,
            "excerpt": _chunk_excerpt(chunk.text),
        }
        for chunk in structure.chunks
    }
    chapters = []
    for chapter in structure.chapters:
        chapter_chunk_ids = list(chapter.chunk_ids)
        chapter_chunks = [chunk_lookup[chunk_id] for chunk_id in chapter_chunk_ids if chunk_id in chunk_lookup]
        chapter_themes = _ordered_unique(
            theme
            for chunk in chapter_chunks
            for theme in chunk.get("themes", [])
        )
        chapters.append(
            {
                "chapter_id": chapter.chapter_id,
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "summary": chapter.summary,
                "chunk_ids": chapter_chunk_ids,
                "chunk_count": len(chapter_chunk_ids),
                "word_count": sum(chunk["word_count"] for chunk in chapter_chunks),
                "themes": chapter_themes[:6],
            }
        )
    return {
        "book_id": structure.book_id,
        "title": structure.title,
        "chapters": chapters,
        "chunks": list(chunk_lookup.values()),
    }


def build_analysis_summary(analysis: BookAnalysis) -> dict:
    """Return a compact analysis payload for episode planning."""

    return {
        "book_id": analysis.book_id,
        "themes": analysis.themes,
        "continuity_arcs": [arc.model_dump(mode="python") for arc in analysis.continuity_arcs],
        "notable_claims": analysis.notable_claims[:12],
        "episode_clusters": [cluster.model_dump(mode="python") for cluster in analysis.episode_clusters],
    }


def payload_size_bytes(payload: dict) -> int:
    """Estimate the serialized payload size in bytes."""

    return len(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))


def _chunk_excerpt(text: str, word_limit: int = 24) -> str:
    words = text.split()
    return " ".join(words[:word_limit])


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
