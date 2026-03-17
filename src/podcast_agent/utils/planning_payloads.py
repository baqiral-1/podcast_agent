"""Helpers for compact analysis and planning payloads."""

from __future__ import annotations

import json

from podcast_agent.schemas.models import BookAnalysis, BookStructure


def build_structure_summary(structure: BookStructure) -> dict:
    """Return a compact summary of the structured book for analysis and planning."""
    chapter_chunks = _chapter_chunk_map(structure)
    chapters = []
    for chapter in structure.chapters:
        chapter_chunk_ids = list(chapter.chunk_ids)
        chapter_chunk_summaries = list(chapter_chunks.get(chapter.chapter_id, []))
        chapter_themes = _ordered_unique(
            theme
            for chunk in chapter_chunk_summaries
            for theme in chunk.get("themes", [])
        )
        sections = _build_section_summaries(chapter_chunk_summaries)
        chapters.append(
            {
                "chapter_id": chapter.chapter_id,
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "summary": chapter.summary,
                "chunk_count": len(chapter_chunk_ids),
                "word_count": sum(chunk["word_count"] for chunk in chapter_chunk_summaries),
                "themes": chapter_themes[:6],
                "sections": sections,
            }
        )
    return {
        "book_id": structure.book_id,
        "title": structure.title,
        "chapters": chapters,
    }


def build_analysis_summary(analysis: BookAnalysis) -> dict:
    """Return a compact analysis payload for episode planning."""

    return {
        "book_id": analysis.book_id,
        "themes": analysis.themes,
        "continuity_arcs": [arc.model_dump(mode="python") for arc in analysis.continuity_arcs],
        "notable_claims": analysis.notable_claims[:12],
        "episode_clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "label": cluster.label,
                "rationale": cluster.rationale,
                "chapter_ids": cluster.chapter_ids,
                "themes": cluster.themes,
            }
            for cluster in analysis.episode_clusters
        ],
    }


def payload_size_bytes(payload: dict) -> int:
    """Estimate the serialized payload size in bytes."""

    return len(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _chapter_chunk_map(structure: BookStructure) -> dict[str, list[dict]]:
    chunk_map: dict[str, list[dict]] = {}
    for chunk in structure.chunks:
        chunk_map.setdefault(chunk.chapter_id, []).append(
            {
                "chunk_id": chunk.chunk_id,
                "sequence": chunk.sequence,
                "word_count": len(chunk.text.split()),
                "themes": chunk.themes,
            }
        )
    for chapter_id in chunk_map:
        chunk_map[chapter_id].sort(key=lambda item: item["sequence"])
    return chunk_map


def _build_section_summaries(
    chunks: list[dict],
    *,
    target_words: int = 900,
    max_chunks: int = 6,
) -> list[dict]:
    if not chunks:
        return []
    sections: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for chunk in chunks:
        if current and (current_words >= target_words or len(current) >= max_chunks):
            sections.append(current)
            current = []
            current_words = 0
        current.append(chunk)
        current_words += chunk["word_count"]
    if current:
        sections.append(current)
    summaries: list[dict] = []
    for index, section_chunks in enumerate(sections, start=1):
        section_themes = _ordered_unique(
            theme for chunk in section_chunks for theme in chunk.get("themes", [])
        )
        summaries.append(
            {
                "section_id": f"section-{index}",
                "start_chunk_id": section_chunks[0]["chunk_id"],
                "end_chunk_id": section_chunks[-1]["chunk_id"],
                "chunk_count": len(section_chunks),
                "word_count": sum(chunk["word_count"] for chunk in section_chunks),
                "themes": section_themes[:4],
            }
        )
    return summaries
