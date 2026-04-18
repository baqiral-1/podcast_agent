"""Multi-book thematic podcast pipeline orchestrator.

Implements the four-phase pipeline:
  Phase 1: Ingest & Index (parallel per book)
  Phase 2: Thematic Intelligence (sequential cross-book)
  Phase 3: Episode Production (parallel per episode)
  Phase 4: Audio Rendering (parallel per episode)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.synthesis_consolidation import SynthesisConsolidationAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.config import Settings
from podcast_agent.ingestion import read_source_text, extract_chapters_from_source
from podcast_agent.langchain.llm import build_llm_client
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.concurrency import configure_llm_semaphore
from podcast_agent.retrieval.search import RetrievalService
from podcast_agent.retrieval.vector_store import PGVectorRetrieval
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    AudioManifest,
    AudioSegmentResult,
    ActorMetadata,
    BookRecord,
    ChapterInfo,
    ChunkingConfig,
    CoverageStats,
    EpisodeCandidateCluster,
    EpisodePlan,
    EpisodeScript,
    ExtractedPassage,
    GroundingReport,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProseSection,
    ProjectStatus,
    RenderManifest,
    RenderSegment,
    RepairResult,
    ScriptTransition,
    SceneCard,
    SegmentDiff,
    SpokenSection,
    SpokenScript,
    SpokenTransition,
    SYNTHESIS_PRIMITIVE_FAMILIES,
    SynthesisConsolidationResult,
    SynthesisMap,
    SynthesisPrimitive,
    SynthesisPrimitivesArtifact,
    SynthesisTag,
    TextChunk,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)
from podcast_agent.tts.openai_compatible import build_tts_client
from podcast_agent.utils.actor_metadata import (
    clean_axis_actor_ids,
    clean_scene_actor_links,
    clean_strategy_actor_links,
    clean_synthesis_primitive_actor_links,
    compact_actor_registry,
    compact_actor_metadata,
    sanitize_actor_metadata_payload,
    select_actor_metadata_subset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact persistence helpers
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(path: Path, model_type: type[Any]) -> Any:
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    return model_type.model_validate(payload)


def _input_hash(*args: Any) -> str:
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SPOKEN_TAG_RE = re.compile(r"<[^>]+>")
_RUNTIME_UNDERSHOOT_WARNING_RATIO = 0.10
_WHITESPACE_RE = re.compile(r"\s+")
_WRITING_SOURCE_MODE_FULL_CHUNK = "full_chunk"
_PRIMITIVE_REUSE_WARNING_THRESHOLD = 4
_INSIGHT_REF_RE = re.compile(r"\bins_\d+\b")
_CROSS_REFERENCE_MIN_COVERAGE = 0.5
_BOOK_BALANCE_MAX_ABS_DRIFT = 0.15
_SPOKEN_RATE_MULTIPLIER = {
    "slower": 0.94,
    "normal": 1.0,
    "faster": 1.08,
}
_MERGED_EPISODE_FILENAME = "episode.mp3"
_SYNTHESIS_MERGED_NARRATIVE_MIN = 7
_SYNTHESIS_MERGED_NARRATIVE_MAX = 9


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _split_contiguous_windows(items: list[Any], window_count: int) -> list[list[Any]]:
    if not items:
        return []
    effective_windows = max(1, min(window_count, len(items)))
    base = len(items) // effective_windows
    remainder = len(items) % effective_windows
    windows: list[list[Any]] = []
    start = 0
    for idx in range(effective_windows):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        windows.append(items[start:end])
        start = end
    return windows


def _split_weighted_contiguous_windows(
    items: list[Any],
    weights: list[float],
    window_count: int,
) -> list[list[Any]]:
    if not items:
        return []
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")

    effective_windows = max(1, min(window_count, len(items)))
    if effective_windows == 1:
        return [list(items)]

    normalized_weights = [max(float(weight), 0.0) for weight in weights]
    remaining_total = sum(normalized_weights)
    windows: list[list[Any]] = []
    start = 0

    for idx in range(effective_windows):
        if idx == effective_windows - 1:
            windows.append(items[start:])
            break
        windows_left = effective_windows - idx
        max_end = len(items) - (windows_left - 1)
        target_weight = remaining_total / windows_left if windows_left > 0 else remaining_total
        accumulated = 0.0
        end = start
        while end < max_end:
            accumulated += normalized_weights[end]
            end += 1
            if accumulated >= target_weight:
                break
        if end <= start:
            end = start + 1
        windows.append(items[start:end])
        remaining_total -= sum(normalized_weights[start:end])
        start = end
    return windows


def _extract_insight_refs(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    refs: list[str] = []
    for match in _INSIGHT_REF_RE.findall(text):
        if match in seen:
            continue
        seen.add(match)
        refs.append(match)
    return refs


def _normalize_beat_insight_linkage(
    beats: list[EpisodeBeat],
) -> tuple[list[EpisodeBeat], dict[str, Any]]:
    adjusted: list[EpisodeBeat] = []
    beats_with_refs = 0
    missing_total = 0
    injected_total = 0
    for beat in beats:
        referenced_ids = _extract_insight_refs(beat.description)
        if referenced_ids:
            beats_with_refs += 1
        existing_ids: list[str] = []
        seen_existing: set[str] = set()
        for insight_id in beat.insight_ids:
            if not insight_id or insight_id in seen_existing:
                continue
            seen_existing.add(insight_id)
            existing_ids.append(insight_id)
        missing_ids = [insight_id for insight_id in referenced_ids if insight_id not in seen_existing]
        missing_total += len(missing_ids)
        if missing_ids:
            injected_total += len(missing_ids)
            adjusted.append(beat.model_copy(update={"insight_ids": [*existing_ids, *missing_ids]}))
        elif len(existing_ids) != len(beat.insight_ids):
            adjusted.append(beat.model_copy(update={"insight_ids": existing_ids}))
        else:
            adjusted.append(beat)
    return adjusted, {
        "beat_count": len(beats),
        "beats_with_description_insight_refs": beats_with_refs,
        "missing_references": missing_total,
        "injected_references": injected_total,
    }


def _build_window_synthesis_context(
    plan: EpisodePlan,
    window_beats: list[EpisodeBeat],
    window_scene_cards: list[Any] | None = None,
) -> EpisodeSynthesisContext | None:
    if plan.synthesis_context is None:
        return None
    window_insight_ids = {
        insight_id
        for beat in window_beats
        for insight_id in beat.insight_ids
        if insight_id
    }
    if window_scene_cards:
        window_insight_ids.update(
            insight_id
            for scene in window_scene_cards
            for insight_id in scene.insight_ids
            if insight_id
        )
    window_insights = [
        insight
        for insight in plan.synthesis_context.insights
        if insight.insight_id in window_insight_ids
    ]
    return plan.synthesis_context.model_copy(
        update={
            "insights": window_insights,
            # Keep all narrative threads in each window for continuity.
            "narrative_threads": list(plan.synthesis_context.narrative_threads),
        }
    )


def _build_writing_windows(plan: EpisodePlan, window_count: int) -> list[dict[str, Any]]:
    if (
        plan.narrative_spine is None
        or not plan.narrative_spine.spine_segments
        or not plan.scene_cards
    ):
        return [
            {
                "spine_segments": (
                    list(plan.narrative_spine.spine_segments)
                    if plan.narrative_spine is not None
                    else []
                ),
                "attribution_moments": (
                    list(plan.narrative_spine.attribution_moments)
                    if plan.narrative_spine is not None
                    else []
                ),
                "scene_cards": list(plan.scene_cards),
                "anchor_scene_ids": list(plan.anchor_scene_ids),
                "beats": window_beats,
            }
            for window_beats in _split_contiguous_windows(list(plan.beats), window_count)
        ]

    beats_by_scene_id: dict[str, list[EpisodeBeat]] = {
        scene.scene_id: []
        for scene in plan.scene_cards
    }
    for beat in plan.beats:
        if beat.scene_id in beats_by_scene_id:
            beats_by_scene_id[beat.scene_id].append(beat)

    scenes_by_spine_id: dict[str, list[Any]] = {}
    for scene in plan.scene_cards:
        scenes_by_spine_id.setdefault(scene.spine_segment_id, []).append(scene)

    active_spine_segments: list[dict[str, Any]] = []
    for spine_segment in plan.narrative_spine.spine_segments:
        window_scene_cards = scenes_by_spine_id.get(spine_segment.spine_segment_id, [])
        if not window_scene_cards:
            continue
        window_beats = [
            beat
            for scene in window_scene_cards
            for beat in beats_by_scene_id.get(scene.scene_id, [])
        ]
        weight = sum(float(beat.estimated_duration_seconds) for beat in window_beats)
        if weight <= 0:
            weight = sum(float(scene.estimated_duration_seconds) for scene in window_scene_cards)
        active_spine_segments.append(
            {
                "spine_segment": spine_segment,
                "scene_cards": window_scene_cards,
                "beats": window_beats,
                "weight": max(weight, 1.0),
            }
        )

    if not active_spine_segments:
        return [
            {
                "spine_segments": list(plan.narrative_spine.spine_segments),
                "attribution_moments": list(plan.narrative_spine.attribution_moments),
                "scene_cards": list(plan.scene_cards),
                "anchor_scene_ids": list(plan.anchor_scene_ids),
                "beats": window_beats,
            }
            for window_beats in _split_contiguous_windows(list(plan.beats), window_count)
        ]

    spine_windows = _split_weighted_contiguous_windows(
        active_spine_segments,
        [float(item["weight"]) for item in active_spine_segments],
        window_count,
    )
    window_specs: list[dict[str, Any]] = []
    for spine_window in spine_windows:
        window_spine_segments = [item["spine_segment"] for item in spine_window]
        window_scene_cards = [
            scene
            for item in spine_window
            for scene in item["scene_cards"]
        ]
        window_scene_ids = {scene.scene_id for scene in window_scene_cards}
        window_beats = [
            beat
            for beat in plan.beats
            if beat.scene_id in window_scene_ids
        ]
        window_spine_segment_ids = {
            segment.spine_segment_id for segment in window_spine_segments
        }
        window_specs.append(
            {
                "spine_segments": window_spine_segments,
                "attribution_moments": [
                    moment
                    for moment in plan.narrative_spine.attribution_moments
                    if moment.insert_after_segment_id in window_spine_segment_ids
                ],
                "scene_cards": window_scene_cards,
                "anchor_scene_ids": [
                    scene_id
                    for scene_id in plan.anchor_scene_ids
                    if scene_id in window_scene_ids
                ],
                "beats": window_beats,
            }
        )
    return window_specs


def _bm25_score(
    tokens: list[str],
    query_terms: dict[str, int],
    idf: dict[str, float],
    avg_len: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not tokens:
        return 0.0
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    doc_len = len(tokens)
    score = 0.0
    for term, qf in query_terms.items():
        if term not in tf:
            continue
        freq = tf[term]
        denom = freq + k1 * (1 - b + b * (doc_len / avg_len))
        score += idf.get(term, 0.0) * ((freq * (k1 + 1)) / denom) * qf
    return score


def _trim_candidate_texts_by_bm25_query_text(
    query_text: str,
    candidates: list[dict],
    *,
    keep_fraction: float = 1 / 3,
    keep_fraction_by_passage_id: dict[str, float] | None = None,
) -> None:
    if not candidates:
        return
    keep_fraction = _clamp(float(keep_fraction), 0.0, 1.0)
    if keep_fraction <= 0:
        return
    query_text = query_text.strip()
    if not query_text:
        return
    query_terms: dict[str, int] = {}
    for term in _tokenize(query_text):
        query_terms[term] = query_terms.get(term, 0) + 1
    if not query_terms:
        return

    sentence_tokens: list[list[str]] = []
    sentences_by_candidate: list[list[str]] = []
    for cand in candidates:
        sentences = _split_sentences(cand.get("text", ""))
        sentences_by_candidate.append(sentences)
        for sentence in sentences:
            sentence_tokens.append(_tokenize(sentence))

    if not sentence_tokens:
        return

    df: dict[str, int] = {}
    for tokens in sentence_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    total_sentences = len(sentence_tokens)
    idf = {
        term: math.log(1 + (total_sentences - count + 0.5) / (count + 0.5))
        for term, count in df.items()
    }
    avg_len = sum(len(tokens) for tokens in sentence_tokens) / total_sentences
    if avg_len <= 0:
        return

    for cand, sentences in zip(candidates, sentences_by_candidate, strict=False):
        if not sentences:
            continue
        passage_id = str(cand.get("passage_id", ""))
        passage_keep_fraction = keep_fraction
        if keep_fraction_by_passage_id is not None and passage_id:
            passage_keep_fraction = keep_fraction_by_passage_id.get(passage_id, keep_fraction)
        passage_keep_fraction = _clamp(float(passage_keep_fraction), 0.0, 1.0)
        if passage_keep_fraction <= 0:
            continue
        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            tokens = _tokenize(sentence)
            score = _bm25_score(tokens, query_terms, idf, avg_len)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top_n = max(1, math.ceil(len(sentences) * passage_keep_fraction))
        selected = sorted(scored[:top_n], key=lambda item: item[1])
        trimmed = " ".join(sentence for _, _, sentence in selected).strip()
        if trimmed:
            cand["text"] = trimmed


def _trim_candidate_texts_by_bm25(
    axis: ThematicAxis,
    candidates: list[dict],
    *,
    keep_fraction: float = 1 / 3,
    keep_fraction_by_passage_id: dict[str, float] | None = None,
) -> None:
    query_parts = [axis.name, axis.description]
    query_parts.extend(axis.guiding_questions)
    query_parts.extend(axis.keywords)
    query_text = " ".join(part for part in query_parts if part).strip()
    _trim_candidate_texts_by_bm25_query_text(
        query_text,
        candidates,
        keep_fraction=keep_fraction,
        keep_fraction_by_passage_id=keep_fraction_by_passage_id,
    )


def _resolve_synthesis_bm25_keep_fraction_by_passage(
    passages: list[ExtractedPassage],
) -> tuple[dict[str, float], dict[str, int]]:
    if not passages:
        return {}, {
            "top_10_passages": 0,
            "next_20_passages": 0,
            "next_70_passages": 0,
        }

    ranked_passages = sorted(
        passages,
        key=lambda passage: (
            -passage.relevance_score,
            -passage.quotability_score,
            passage.passage_id,
        ),
    )
    passage_count = len(ranked_passages)
    top_10_count = min(passage_count, max(0, math.ceil(passage_count * 0.10)))
    next_20_count = min(
        max(0, passage_count - top_10_count),
        max(0, math.ceil(passage_count * 0.20)),
    )
    keep_fraction_by_passage_id: dict[str, float] = {}
    for idx, passage in enumerate(ranked_passages):
        if idx < top_10_count:
            keep_fraction_by_passage_id[passage.passage_id] = 0.4
            continue
        if idx < top_10_count + next_20_count:
            keep_fraction_by_passage_id[passage.passage_id] = 0.33
            continue
        keep_fraction_by_passage_id[passage.passage_id] = 0.25
    return keep_fraction_by_passage_id, {
        "top_10_passages": top_10_count,
        "next_20_passages": next_20_count,
        "next_70_passages": max(0, passage_count - top_10_count - next_20_count),
    }


def _compute_passage_retrieval_budget(
    *,
    chunk_count: int,
    percentage: float,
    min_per_book: int,
    max_per_book: int,
) -> dict[str, int]:
    percentage_budget = int(round(max(0, chunk_count) * percentage))
    budget = min(max_per_book, max(min_per_book, percentage_budget))
    return {
        "chunk_count": max(0, chunk_count),
        "percentage_budget": percentage_budget,
        "per_book_budget": budget,
    }


def _compute_weighted_admitted_budgets(
    *,
    book_ids: list[str],
    axis_total_budget: int,
    relevance_by_book: dict[str, float],
    floor_per_book: int = 2,
    relevance_power: float = 1.2,
) -> dict[str, int]:
    if not book_ids:
        return {}

    total_budget = max(0, axis_total_budget)
    if total_budget <= 0:
        return {book_id: 0 for book_id in book_ids}

    book_count = len(book_ids)
    effective_floor = min(max(0, floor_per_book), total_budget // book_count)
    budgets = {book_id: effective_floor for book_id in book_ids}
    remaining_budget = total_budget - (effective_floor * book_count)
    if remaining_budget <= 0:
        return budgets

    weights: dict[str, float] = {}
    for book_id in book_ids:
        relevance = max(0.0, float(relevance_by_book.get(book_id, 0.0)))
        relevance_factor = relevance ** relevance_power if relevance_power > 0 else 1.0
        weights[book_id] = relevance_factor

    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {book_id: 1.0 for book_id in book_ids}
        total_weight = float(len(book_ids))

    fractional_allocations: list[tuple[float, int, str]] = []
    allocated = 0
    for idx, book_id in enumerate(book_ids):
        share = remaining_budget * (weights[book_id] / total_weight)
        extra = int(math.floor(share))
        budgets[book_id] += extra
        allocated += extra
        fractional_allocations.append((share - extra, idx, book_id))

    leftover = remaining_budget - allocated
    fractional_allocations.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, _, book_id in fractional_allocations[:leftover]:
        budgets[book_id] += 1

    return budgets


def _compute_weighted_axis_budgets(
    *,
    axis_ids: list[str],
    total_budget: int,
    weight_by_axis: dict[str, float],
    floor_per_axis: int = 0,
    cap_per_axis: int | None = None,
) -> dict[str, int]:
    if not axis_ids:
        return {}

    budget = max(0, int(total_budget))
    if budget <= 0:
        return {axis_id: 0 for axis_id in axis_ids}

    axis_count = len(axis_ids)
    effective_floor = min(max(0, floor_per_axis), budget // axis_count)
    budgets = {axis_id: effective_floor for axis_id in axis_ids}
    remaining = budget - (effective_floor * axis_count)
    if remaining <= 0:
        return budgets

    for _ in range(max(1, axis_count * 2)):
        if remaining <= 0:
            break
        available_ids = [
            axis_id
            for axis_id in axis_ids
            if cap_per_axis is None or budgets[axis_id] < cap_per_axis
        ]
        if not available_ids:
            break

        weights = {
            axis_id: max(0.0, float(weight_by_axis.get(axis_id, 0.0)))
            for axis_id in available_ids
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            weights = {axis_id: 1.0 for axis_id in available_ids}
            total_weight = float(len(available_ids))

        fractional: list[tuple[float, int, str]] = []
        allocated = 0
        for idx, axis_id in enumerate(available_ids):
            raw_share = remaining * (weights[axis_id] / total_weight)
            extra = int(math.floor(raw_share))
            if cap_per_axis is not None:
                extra = min(extra, max(0, cap_per_axis - budgets[axis_id]))
            if extra > 0:
                budgets[axis_id] += extra
                allocated += extra
            fractional.append((raw_share - extra, idx, axis_id))

        remaining -= allocated
        if remaining <= 0:
            break

        fractional.sort(key=lambda item: (-item[0], item[1], item[2]))
        distributed = 0
        for _, _, axis_id in fractional:
            if remaining <= 0:
                break
            if cap_per_axis is not None and budgets[axis_id] >= cap_per_axis:
                continue
            budgets[axis_id] += 1
            remaining -= 1
            distributed += 1
        if distributed == 0:
            break

    return budgets


def _normalize_axis_importance_weights(
    *,
    axes: list[ThematicAxis],
    power: float,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
) -> dict[str, float]:
    if not axes:
        return {}
    lo = min(min_weight, max_weight)
    hi = max(min_weight, max_weight)
    exponent = max(0.0, power)
    raw_by_axis = {
        axis.axis_id: _clamp(float(axis.theme_importance_score), 0.0, 1.0)
        for axis in axes
    }
    raw_values = list(raw_by_axis.values())
    raw_min = min(raw_values)
    raw_max = max(raw_values)
    if math.isclose(raw_min, raw_max):
        return {axis.axis_id: hi for axis in axes}

    span = raw_max - raw_min
    if span <= 0:
        return {axis.axis_id: hi for axis in axes}

    normalized: dict[str, float] = {}
    for axis in axes:
        scaled = (raw_by_axis[axis.axis_id] - raw_min) / span
        shaped = scaled ** exponent if exponent > 0 else 1.0
        normalized[axis.axis_id] = lo + ((hi - lo) * shaped)
    return normalized


def _build_axis_budget_by_importance(
    *,
    axes: list[ThematicAxis],
    total_budget: int,
    floor_per_axis: int,
    importance_power: float,
    cap_per_axis: int | None = None,
) -> dict[str, int]:
    axis_ids = [axis.axis_id for axis in axes]
    weights = _normalize_axis_importance_weights(
        axes=axes,
        power=importance_power,
        min_weight=0.5,
        max_weight=1.0,
    )
    return _compute_weighted_axis_budgets(
        axis_ids=axis_ids,
        total_budget=total_budget,
        weight_by_axis=weights,
        floor_per_axis=floor_per_axis,
        cap_per_axis=cap_per_axis,
    )


def _compute_stage_axis_target_count(
    *,
    axis_total: int,
    percentage: float,
    minimum: int,
    maximum: int,
) -> int:
    target = int(round(max(0, axis_total) * _clamp(percentage, 0.0, 1.0)))
    target = max(minimum, target)
    target = min(maximum, target)
    return max(0, min(target, max(0, axis_total)))


def _passage_similarity_tokens(passage: ExtractedPassage, *, max_chars: int = 1200) -> set[str]:
    text = (passage.trimmed_text or passage.full_text or passage.text or "")[:max_chars]
    return set(_tokenize(text))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    intersection = len(left & right)
    if intersection <= 0:
        return 0.0
    return intersection / max(1, len(left | right))


def _select_mmr_passages(
    *,
    passages: list[ExtractedPassage],
    top_k: int,
    base_score_fn: Callable[[ExtractedPassage], float],
    lambda_weight: float,
    source_group_fn: Callable[[ExtractedPassage], str | None] | None = None,
    source_penalty_weight: float = 0.0,
) -> list[ExtractedPassage]:
    if not passages:
        return []
    top_n = max(1, min(top_k, len(passages)))
    if top_n >= len(passages):
        return sorted(
            passages,
            key=lambda p: (-base_score_fn(p), -p.relevance_score, -p.quotability_score, p.passage_id),
        )

    lambda_weight = _clamp(lambda_weight, 0.0, 1.0)
    source_penalty_weight = _clamp(source_penalty_weight, 0.0, 1.0)
    token_sets = [_passage_similarity_tokens(passage) for passage in passages]
    base_scores = [base_score_fn(passage) for passage in passages]
    source_groups = [
        source_group_fn(passage)
        if source_group_fn is not None
        else None
        for passage in passages
    ]
    similarity_matrix: list[list[float]] = [[0.0] * len(passages) for _ in passages]
    for idx, left in enumerate(token_sets):
        for jdx in range(idx + 1, len(passages)):
            similarity = _jaccard_similarity(left, token_sets[jdx])
            similarity_matrix[idx][jdx] = similarity
            similarity_matrix[jdx][idx] = similarity

    candidates = list(range(len(passages)))
    max_similarity_to_selected = [0.0] * len(passages)
    selected_source_groups: set[str] = set()
    selected_indices: list[int] = []
    for _ in range(top_n):
        best_idx = max(
            candidates,
            key=lambda idx: (
                (
                    lambda_weight * base_scores[idx]
                ) - (
                    (1.0 - lambda_weight)
                    * max(
                        max_similarity_to_selected[idx],
                        source_penalty_weight
                        if source_groups[idx] is not None and source_groups[idx] in selected_source_groups
                        else 0.0,
                    )
                ),
                base_scores[idx],
                passages[idx].relevance_score,
                passages[idx].quotability_score,
                -idx,
            ),
        )
        selected_indices.append(best_idx)
        candidates.remove(best_idx)
        if source_groups[best_idx] is not None:
            selected_source_groups.add(source_groups[best_idx])
        row = similarity_matrix[best_idx]
        for idx in candidates:
            if row[idx] > max_similarity_to_selected[idx]:
                max_similarity_to_selected[idx] = row[idx]
    return [passages[idx] for idx in selected_indices]


def _resolve_writing_passage_text(passage: ExtractedPassage) -> str:
    full_text = passage.full_text.strip()
    if full_text:
        return full_text
    return passage.text


def _build_compact_chapter_projection(chapter: ChapterInfo) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chapter_id": chapter.chapter_id,
        "title": chapter.title,
    }
    analysis = chapter.analysis
    payload["analysis"] = analysis.model_dump(mode="json") if analysis is not None else None
    return payload


def _build_chapter_context(chapter: ChapterInfo | None) -> dict[str, Any] | None:
    if chapter is None:
        return None
    context: dict[str, Any] = {
        "chapter_id": chapter.chapter_id,
        "chapter_title": chapter.title,
    }
    analysis = chapter.analysis
    context["chapter_analysis"] = (
        analysis.model_dump(mode="json") if analysis is not None else None
    )
    return context


def _build_chapter_lookup(books: list[BookRecord]) -> dict[tuple[str, str], ChapterInfo]:
    lookup: dict[tuple[str, str], ChapterInfo] = {}
    for book in books:
        for chapter in book.chapters:
            lookup[(book.book_id, chapter.chapter_id)] = chapter
    return lookup


def _build_flat_planning_passage_payload(
    *,
    assigned_axis_ids: list[str],
    passages_by_axis: dict[str, list[ExtractedPassage]],
    selected_insight_passage_ids: set[str],
    extra_insight_passages: list[ExtractedPassage] | None = None,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for axis_id in assigned_axis_ids:
        for passage in passages_by_axis.get(axis_id, []):
            if passage.passage_id not in by_id:
                payload: dict[str, Any] = {
                    "passage_id": passage.passage_id,
                    "book_id": passage.book_id,
                    "chapter_ref": passage.chapter_ref,
                    "relevance_score": passage.relevance_score,
                    "quotability_score": passage.quotability_score,
                }
                if passage.passage_id in selected_insight_passage_ids:
                    payload["full_text"] = _resolve_writing_passage_text(passage)
                else:
                    payload["summary_text"] = passage.trimmed_text.strip() or passage.text
                by_id[passage.passage_id] = payload
                ordered_ids.append(passage.passage_id)
                continue
            existing = by_id[passage.passage_id]
            existing["relevance_score"] = max(
                float(existing["relevance_score"]),
                passage.relevance_score,
            )
            existing["quotability_score"] = max(
                float(existing["quotability_score"]),
                passage.quotability_score,
            )
    for passage in extra_insight_passages or []:
        if passage.passage_id in by_id:
            continue
        by_id[passage.passage_id] = {
            "passage_id": passage.passage_id,
            "book_id": passage.book_id,
            "chapter_ref": passage.chapter_ref,
            "relevance_score": passage.relevance_score,
            "quotability_score": passage.quotability_score,
            "full_text": _resolve_writing_passage_text(passage),
        }
        ordered_ids.append(passage.passage_id)
    return [by_id[passage_id] for passage_id in ordered_ids]


def _rank_synthesis_axis_passages(
    passages: list[ExtractedPassage],
    *,
    cross_pair_ids: set[str],
) -> list[ExtractedPassage]:
    return sorted(
        passages,
        key=lambda p: (
            -(1 if p.passage_id in cross_pair_ids else 0),
            -p.relevance_score,
            -p.quotability_score,
            p.passage_id,
        ),
    )


def _allocate_synthesis_passages_by_axis(
    *,
    axes: list[ThematicAxis],
    passages_by_axis: dict[str, list[ExtractedPassage]],
    total_cap: int,
    importance_power: float,
    cross_pair_ids: set[str],
) -> tuple[dict[str, list[ExtractedPassage]], dict[str, Any]]:
    if not axes:
        return {}, {
            "input_total": 0,
            "output_total": 0,
            "round_robin_fill_count": 0,
            "total_cap": max(0, int(total_cap)),
            "axis_order": [],
            "axis_quota_by_axis": {},
            "axis_weight_by_axis": {},
        }
    cap = max(0, int(total_cap))
    axis_order = sorted(axes, key=lambda axis: (-axis.theme_importance_score, axis.axis_id))
    axis_ids = [axis.axis_id for axis in axis_order]
    weights = _normalize_axis_importance_weights(
        axes=axis_order,
        power=importance_power,
        min_weight=0.5,
        max_weight=1.0,
    )
    quotas = _compute_weighted_axis_budgets(
        axis_ids=axis_ids,
        total_budget=cap,
        weight_by_axis=weights,
        floor_per_axis=0,
        cap_per_axis=None,
    )
    ranked_by_axis = {
        axis_id: _rank_synthesis_axis_passages(
            passages_by_axis.get(axis_id, []),
            cross_pair_ids=cross_pair_ids,
        )
        for axis_id in axis_ids
    }

    def _source_key(passage: ExtractedPassage) -> tuple[str, tuple[str, ...]] | tuple[str, str]:
        if passage.chunk_ids:
            return (passage.book_id, tuple(passage.chunk_ids))
        return ("passage", passage.passage_id)

    selected_by_axis: dict[str, list[ExtractedPassage]] = {axis_id: [] for axis_id in axis_ids}
    used_source_keys: set[tuple[str, tuple[str, ...]] | tuple[str, str]] = set()
    for axis_id in axis_ids:
        quota = max(0, int(quotas.get(axis_id, 0)))
        if quota <= 0:
            continue
        for passage in ranked_by_axis.get(axis_id, []):
            if len(selected_by_axis[axis_id]) >= quota:
                break
            source_key = _source_key(passage)
            if source_key in used_source_keys:
                continue
            selected_by_axis[axis_id].append(passage)
            used_source_keys.add(source_key)

    remaining_slots = max(0, cap - len(used_source_keys))
    round_robin_fill_count = 0
    next_index_by_axis = {
        axis_id: len(selected_by_axis[axis_id])
        for axis_id in axis_ids
    }
    while remaining_slots > 0:
        added = False
        for axis_id in axis_ids:
            ranked = ranked_by_axis.get(axis_id, [])
            idx = next_index_by_axis.get(axis_id, 0)
            while idx < len(ranked) and _source_key(ranked[idx]) in used_source_keys:
                idx += 1
            next_index_by_axis[axis_id] = idx
            if idx >= len(ranked):
                continue
            passage = ranked[idx]
            selected_by_axis[axis_id].append(passage)
            used_source_keys.add(_source_key(passage))
            next_index_by_axis[axis_id] = idx + 1
            round_robin_fill_count += 1
            remaining_slots -= 1
            added = True
            if remaining_slots <= 0:
                break
        if not added:
            break

    input_total = sum(len(items) for items in passages_by_axis.values())
    output_total = sum(len(items) for items in selected_by_axis.values())
    return selected_by_axis, {
        "input_total": input_total,
        "output_total": output_total,
        "round_robin_fill_count": round_robin_fill_count,
        "total_cap": cap,
        "axis_order": axis_ids,
        "axis_quota_by_axis": quotas,
        "axis_weight_by_axis": {axis_id: round(float(weights.get(axis_id, 0.0)), 6) for axis_id in axis_ids},
    }


def _select_episode_planning_passages(
    *,
    passages_by_axis: dict[str, list[ExtractedPassage]],
    assigned_axis_ids: list[str],
    selected_insight_passage_ids: set[str],
    supporting_passages_per_axis: int = 60,
    supporting_passages_per_axis_by_axis: dict[str, int] | None = None,
    use_mmr: bool = False,
    mmr_lambda: float = 0.65,
) -> dict[str, list[ExtractedPassage]]:
    if not assigned_axis_ids:
        return {axis_id: [] for axis_id in assigned_axis_ids}

    def _chunk_key(passage: ExtractedPassage) -> tuple[str, ...]:
        if passage.chunk_ids:
            return tuple(passage.chunk_ids)
        return ("passage", passage.passage_id)

    chunk_axes: dict[tuple[str, ...], set[str]] = {}
    for axis_id in assigned_axis_ids:
        for passage in passages_by_axis.get(axis_id, []):
            key = _chunk_key(passage)
            chunk_axes.setdefault(key, set()).add(axis_id)

    selected_by_axis: dict[str, list[ExtractedPassage]] = {axis_id: [] for axis_id in assigned_axis_ids}
    for axis_id in assigned_axis_ids:
        insight_passages: list[ExtractedPassage] = []
        supporting_pool: list[ExtractedPassage] = []
        for passage in passages_by_axis.get(axis_id, []):
            if passage.passage_id in selected_insight_passage_ids:
                insight_passages.append(passage)
                continue
            supporting_pool.append(passage)
        insight_passages.sort(
            key=lambda p: (-p.relevance_score, -p.quotability_score, p.passage_id)
        )
        target_supporting_count = supporting_passages_per_axis
        if supporting_passages_per_axis_by_axis is not None:
            target_supporting_count = supporting_passages_per_axis_by_axis.get(
                axis_id, supporting_passages_per_axis
            )
        supporting_top_n = max(0, min(target_supporting_count, len(supporting_pool)))

        def _planning_score(passage: ExtractedPassage) -> float:
            key = _chunk_key(passage)
            is_multi_axis = len(chunk_axes.get(key, set())) > 1
            return (
                (0.65 * passage.relevance_score)
                + (0.35 * passage.quotability_score)
                + (0.04 if is_multi_axis else 0.0)
            )

        if use_mmr and supporting_top_n > 0:
            supporting_passages = _select_mmr_passages(
                passages=supporting_pool,
                top_k=supporting_top_n,
                base_score_fn=_planning_score,
                lambda_weight=mmr_lambda,
            )
        else:
            supporting_ranked = sorted(
                supporting_pool,
                key=lambda passage: (
                    -_planning_score(passage),
                    -passage.relevance_score,
                    -passage.quotability_score,
                    passage.passage_id,
                ),
            )
            supporting_passages = supporting_ranked[:supporting_top_n]
        selected_by_axis[axis_id] = insight_passages + supporting_passages
    return selected_by_axis


def _build_merged_narrative_catalog(synthesis_map: SynthesisMap) -> list[dict[str, Any]]:
    return [
        {
            "merged_narrative_id": f"merged_narrative_{idx + 1:03d}",
            "topic": merged.topic,
            "narrative": merged.narrative,
            "source_passage_ids": list(merged.source_passage_ids),
            "points_of_consensus": list(merged.points_of_consensus),
            "points_of_disagreement": list(merged.points_of_disagreement),
        }
        for idx, merged in enumerate(synthesis_map.merged_narratives)
    ]


def _build_tension_catalog(synthesis_map: SynthesisMap) -> list[dict[str, Any]]:
    return [
        {
            "tension_id": f"tension_{idx + 1:03d}",
            "question": question,
        }
        for idx, question in enumerate(synthesis_map.unresolved_tensions)
    ]


def _evaluate_synthesis_merged_narrative_count(
    synthesis_map: SynthesisMap,
    *,
    minimum: int = _SYNTHESIS_MERGED_NARRATIVE_MIN,
    maximum: int = _SYNTHESIS_MERGED_NARRATIVE_MAX,
) -> dict[str, Any]:
    count = len(synthesis_map.merged_narratives)
    return {
        "count": count,
        "minimum": minimum,
        "maximum": maximum,
        "is_in_range": minimum <= count <= maximum,
    }


def _build_synthesis_feedback_for_merged_narrative_count(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue": "merged_narrative_count_out_of_range",
        "observed_count": report["count"],
        "target_min": report["minimum"],
        "target_max": report["maximum"],
        "instruction": (
            "Regenerate the synthesis map so merged_narratives contains between "
            f"{report['minimum']} and {report['maximum']} items."
        ),
    }


def _evaluate_strategy_merged_narrative_assignments(
    *,
    strategy: NarrativeStrategy,
    merged_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_ids = {item["merged_narrative_id"] for item in merged_catalog}
    merged_available = bool(valid_ids)
    episodes: list[dict[str, Any]] = []
    episode_numbers_by_id: dict[str, list[int]] = {}
    for assignment in strategy.episode_assignments:
        assigned_id = (assignment.merged_narrative_id or "").strip() or None
        expected_count = 1 if merged_available else 0
        if merged_available:
            is_ok = assigned_id in valid_ids
            invalid_assigned_id = assigned_id if assigned_id and assigned_id not in valid_ids else None
            if assigned_id in valid_ids:
                episode_numbers_by_id.setdefault(assigned_id, []).append(assignment.episode_number)
        else:
            is_ok = assigned_id is None
            invalid_assigned_id = assigned_id
        episodes.append(
            {
                "episode_number": assignment.episode_number,
                "title": assignment.title,
                "assigned_id": assigned_id,
                "invalid_assigned_id": invalid_assigned_id,
                "expected_count": expected_count,
                "status": "ok" if is_ok else "invalid",
            }
        )

    duplicate_groups = [
        {
            "merged_narrative_id": merged_narrative_id,
            "episode_numbers": sorted(episode_numbers),
        }
        for merged_narrative_id, episode_numbers in sorted(episode_numbers_by_id.items())
        if len(episode_numbers) > 1
    ]
    duplicate_id_set = {item["merged_narrative_id"] for item in duplicate_groups}
    if duplicate_id_set:
        for item in episodes:
            if item["assigned_id"] not in duplicate_id_set:
                continue
            item["status"] = "duplicate"
            item["duplicate_episode_numbers"] = next(
                group["episode_numbers"]
                for group in duplicate_groups
                if group["merged_narrative_id"] == item["assigned_id"]
            )
    invalid_episodes = [item for item in episodes if item["status"] != "ok"]
    return {
        "merged_available": merged_available,
        "valid_ids": sorted(valid_ids),
        "has_issues": bool(invalid_episodes),
        "problem_count": len(invalid_episodes),
        "duplicate_groups": duplicate_groups,
        "episodes": episodes,
    }


def _build_strategy_feedback_for_merged_narratives(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue": "episode_merged_narrative_assignment",
        "merged_available": report["merged_available"],
        "valid_merged_narrative_ids": report["valid_ids"],
        "problem_episodes": [
            {
                "episode_number": item["episode_number"],
                "assigned_id": item["assigned_id"],
                "invalid_assigned_id": item["invalid_assigned_id"],
                "expected_count": item["expected_count"],
                "status": item["status"],
                "duplicate_episode_numbers": item.get("duplicate_episode_numbers", []),
            }
            for item in report["episodes"]
            if item["status"] != "ok"
        ],
        "duplicate_groups": report.get("duplicate_groups", []),
        "instruction": (
            "Revise episode_assignments so each episode has exactly one valid merged_narrative_id and "
            "no merged_narrative_id is reused across episodes."
        ),
    }


def _build_episode_synthesis_context(
    *,
    assignment: EpisodeAssignment,
    selected_insights: list[Any],
    synthesis_map: SynthesisMap,
    merged_catalog: list[dict[str, Any]],
    tension_catalog: list[dict[str, Any]],
) -> EpisodeSynthesisContext:
    selected_merged = [
        EpisodeMergedNarrativeRef(
            merged_narrative_id=item["merged_narrative_id"],
            topic=item["topic"],
            narrative=item["narrative"],
            source_passage_ids=item["source_passage_ids"],
        )
        for item in merged_catalog
        if item["merged_narrative_id"] == assignment.merged_narrative_id
    ]
    selected_tensions = [
        EpisodeSynthesisTension(
            tension_id=item["tension_id"],
            question=item["question"],
        )
        for item in tension_catalog
        if item["tension_id"] in assignment.tension_ids
    ]
    selected_threads = [
        thread
        for thread in synthesis_map.narrative_threads
        if any(insight_id in assignment.insight_ids for insight_id in thread.insight_ids)
    ]
    return EpisodeSynthesisContext(
        insights=selected_insights,
        narrative_threads=selected_threads,
        merged_narratives=selected_merged,
        unresolved_tensions=selected_tensions,
        quality_score=synthesis_map.quality_score,
    )


def _evaluate_episode_plan_insight_realization(
    *,
    assignment: EpisodeAssignment,
    selected_insights: list[Any],
    plan: EpisodePlan,
) -> dict[str, Any]:
    planned_passage_ids = {
        passage_id
        for beat in plan.beats
        for passage_id in beat.passage_ids
    }
    results: list[dict[str, Any]] = []
    for insight in selected_insights:
        realized_passage_ids = sorted(planned_passage_ids & set(insight.passage_ids))
        realized_count = len(realized_passage_ids)
        expected_min = min(2, len(insight.passage_ids))
        if realized_count == 0:
            status = "zero"
        elif realized_count < expected_min:
            status = "weak"
        else:
            status = "ok"
        results.append(
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "status": status,
                "realized_count": realized_count,
                "expected_min": expected_min,
                "assigned_passage_count": len(insight.passage_ids),
                "realized_passage_ids": realized_passage_ids,
                "missing_passage_ids": sorted(set(insight.passage_ids) - planned_passage_ids),
            }
        )
    weak_or_zero = [item for item in results if item["status"] in {"weak", "zero"}]
    return {
        "episode_number": assignment.episode_number,
        "title": assignment.title,
        "insight_ids": list(assignment.insight_ids),
        "has_issues": bool(weak_or_zero),
        "problem_count": len(weak_or_zero),
        "insights": results,
    }


def _evaluate_episode_plan_merged_narrative_realization(
    *,
    assignment: EpisodeAssignment,
    synthesis_context: EpisodeSynthesisContext,
    plan: EpisodePlan,
) -> dict[str, Any]:
    planned_passage_ids = {
        passage_id
        for beat in plan.beats
        for passage_id in beat.passage_ids
    }
    results: list[dict[str, Any]] = []
    for merged in synthesis_context.merged_narratives:
        source_ids = set(merged.source_passage_ids)
        realized_passage_ids = sorted(planned_passage_ids & source_ids)
        realized_count = len(realized_passage_ids)
        expected_min = 1 if source_ids else 0
        if expected_min == 0:
            status = "not_applicable"
        elif realized_count == 0:
            status = "zero"
        else:
            status = "ok"
        results.append(
            {
                "merged_narrative_id": merged.merged_narrative_id,
                "topic": merged.topic,
                "status": status,
                "realized_count": realized_count,
                "expected_min": expected_min,
                "assigned_passage_count": len(source_ids),
                "realized_passage_ids": realized_passage_ids,
                "missing_passage_ids": sorted(source_ids - planned_passage_ids),
            }
        )
    zero_only = [item for item in results if item["status"] == "zero"]
    return {
        "episode_number": assignment.episode_number,
        "title": assignment.title,
        "merged_narrative_id": assignment.merged_narrative_id,
        "has_issues": bool(zero_only),
        "problem_count": len(zero_only),
        "merged_narratives": results,
    }


def _evaluate_episode_plan_realization(
    *,
    assignment: EpisodeAssignment,
    selected_insights: list[Any],
    synthesis_context: EpisodeSynthesisContext,
    plan: EpisodePlan,
) -> dict[str, Any]:
    insight_realization = _evaluate_episode_plan_insight_realization(
        assignment=assignment,
        selected_insights=selected_insights,
        plan=plan,
    )
    merged_realization = _evaluate_episode_plan_merged_narrative_realization(
        assignment=assignment,
        synthesis_context=synthesis_context,
        plan=plan,
    )
    return {
        "episode_number": assignment.episode_number,
        "title": assignment.title,
        "insight_ids": insight_realization["insight_ids"],
        "has_issues": insight_realization["has_issues"] or merged_realization["has_issues"],
        "problem_count": insight_realization["problem_count"] + merged_realization["problem_count"],
        "insight_problem_count": insight_realization["problem_count"],
        "merged_narrative_problem_count": merged_realization["problem_count"],
        "insights": insight_realization["insights"],
        "merged_narratives": merged_realization["merged_narratives"],
    }


def _build_planning_feedback(realization: dict[str, Any]) -> dict[str, Any]:
    problem_insights = [
        {
            "insight_id": item["insight_id"],
            "title": item["title"],
            "status": item["status"],
            "expected_min": item["expected_min"],
            "realized_count": item["realized_count"],
            "missing_passage_ids": item["missing_passage_ids"],
        }
        for item in realization["insights"]
        if item["status"] in {"weak", "zero"}
    ]
    problem_merged_narratives = [
        {
            "merged_narrative_id": item["merged_narrative_id"],
            "topic": item["topic"],
            "status": item["status"],
            "expected_min": item["expected_min"],
            "realized_count": item["realized_count"],
            "missing_passage_ids": item["missing_passage_ids"],
        }
        for item in realization.get("merged_narratives", [])
        if item["status"] in {"weak", "zero"}
    ]
    if problem_insights and problem_merged_narratives:
        issue = "assigned_insight_and_merged_narrative_realization"
        instruction = (
            "Revise the episode plan so every assigned insight and assigned merged narrative is "
            "materially realized in beats using the provided passage_ids."
        )
    elif problem_merged_narratives:
        issue = "assigned_merged_narrative_realization"
        instruction = (
            "Revise the episode plan so each assigned merged narrative is materially realized "
            "using its source_passage_ids."
        )
    else:
        issue = "assigned_insight_realization"
        instruction = (
            "Revise the episode plan so every assigned insight is materially realized in beats "
            "using the assigned insight passage_ids."
        )
    return {
        "issue": issue,
        "episode_number": realization["episode_number"],
        "problem_insights": problem_insights,
        "problem_merged_narratives": problem_merged_narratives,
        "instruction": instruction,
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _resolve_axis_relevance(axis: ThematicAxis, book_ids: list[str]) -> dict[str, float]:
    relevance: dict[str, float] = {}
    total = 0.0
    for book_id in book_ids:
        score = float(axis.relevance_by_book.get(book_id, 0.0))
        score = max(0.0, score)
        relevance[book_id] = score
        total += score
    if total <= 0:
        return {book_id: 1.0 for book_id in book_ids}
    return relevance


def _resolve_book_size_shares(
    books: list[BookRecord],
    *,
    basis: str = "total_words",
) -> dict[str, float]:
    if not books:
        return {}
    if basis != "total_words":
        raise ValueError(f"Unsupported retrieval size basis '{basis}'")
    raw_values = {
        book.book_id: max(0.0, float(book.total_words))
        for book in books
    }
    total = sum(raw_values.values())
    if total <= 0:
        even = 1.0 / len(books)
        return {book.book_id: even for book in books}
    return {
        book.book_id: (raw_values.get(book.book_id, 0.0) / total)
        for book in books
    }


def _compute_passage_utilization(
    *,
    corpus: ThematicCorpus,
    episode_plans: list[EpisodePlan],
    episode_scripts: list[EpisodeScript],
    books: list[BookRecord],
) -> dict[str, Any]:
    retained_ids: set[str] = set()
    passage_by_id: dict[str, ExtractedPassage] = {}
    for axis_id, passages in corpus.passages_by_axis.items():
        for passage in passages:
            retained_ids.add(passage.passage_id)
            passage_by_id[passage.passage_id] = passage

    planned_ids: set[str] = set()
    for plan in episode_plans:
        for beat in plan.beats:
            for passage_id in beat.passage_ids:
                if passage_id in retained_ids:
                    planned_ids.add(passage_id)

    cited_ids: set[str] = set()
    for script in episode_scripts:
        for citation in script.citations:
            if citation.passage_id in retained_ids:
                cited_ids.add(citation.passage_id)
        for segment in script.segments:
            for citation in segment.citations:
                if citation.passage_id in retained_ids:
                    cited_ids.add(citation.passage_id)

    def _ratio(count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round(count / total, 4)

    per_axis: dict[str, Any] = {}
    for axis_id, passages in corpus.passages_by_axis.items():
        axis_retained = {p.passage_id for p in passages}
        per_axis[axis_id] = {
            "retained_count": len(axis_retained),
            "planned_count": len(axis_retained & planned_ids),
            "cited_count": len(axis_retained & cited_ids),
            "plan_utilization_ratio": _ratio(len(axis_retained & planned_ids), len(axis_retained)),
            "citation_utilization_ratio": _ratio(len(axis_retained & cited_ids), len(axis_retained)),
        }

    per_book: dict[str, Any] = {}
    for book in books:
        book_retained = {
            pid for pid in retained_ids
            if passage_by_id.get(pid) is not None and passage_by_id[pid].book_id == book.book_id
        }
        per_book[book.book_id] = {
            "title": book.title,
            "retained_count": len(book_retained),
            "planned_count": len(book_retained & planned_ids),
            "cited_count": len(book_retained & cited_ids),
            "plan_utilization_ratio": _ratio(len(book_retained & planned_ids), len(book_retained)),
            "citation_utilization_ratio": _ratio(len(book_retained & cited_ids), len(book_retained)),
        }

    return {
        "summary": {
            "retained_count": len(retained_ids),
            "planned_count": len(planned_ids),
            "cited_count": len(cited_ids),
            "plan_utilization_ratio": _ratio(len(planned_ids), len(retained_ids)),
            "citation_utilization_ratio": _ratio(len(cited_ids), len(retained_ids)),
        },
        "per_axis": per_axis,
        "per_book": per_book,
    }


def _evaluate_episode_script_plan_alignment(
    *,
    plan: EpisodePlan,
    script: EpisodeScript,
    min_cross_reference_coverage: float = _CROSS_REFERENCE_MIN_COVERAGE,
    max_book_balance_abs_drift: float = _BOOK_BALANCE_MAX_ABS_DRIFT,
) -> dict[str, Any]:
    beat_passages_by_id = {
        beat.beat_id: set(beat.passage_ids)
        for beat in plan.beats
    }
    beat_scene_by_id = {
        beat.beat_id: beat.scene_id
        for beat in plan.beats
        if beat.scene_id
    }

    observed_passage_ids: set[str] = set()
    observed_scene_ids: list[str] = []
    for citation in script.citations:
        observed_passage_ids.add(citation.passage_id)
    for segment in script.segments:
        if segment.beat_id and segment.beat_id in beat_passages_by_id:
            observed_passage_ids.update(beat_passages_by_id[segment.beat_id])
        segment_scene_id = segment.scene_id or (
            beat_scene_by_id.get(segment.beat_id) if segment.beat_id else None
        )
        if segment_scene_id:
            observed_scene_ids.append(segment_scene_id)
        for citation in segment.citations:
            observed_passage_ids.add(citation.passage_id)

    insights = list(plan.synthesis_context.insights) if plan.synthesis_context else []
    insight_results: list[dict[str, Any]] = []
    for insight in insights:
        assigned_passage_ids = set(insight.passage_ids)
        realized_passage_ids = sorted(observed_passage_ids & assigned_passage_ids)
        realized_count = len(realized_passage_ids)
        expected_min = min(2, len(assigned_passage_ids))
        if expected_min == 0:
            status = "not_applicable"
        elif realized_count == 0:
            status = "zero"
        elif realized_count < expected_min:
            status = "weak"
        else:
            status = "ok"
        insight_results.append(
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "status": status,
                "realized_count": realized_count,
                "expected_min": expected_min,
                "assigned_passage_count": len(assigned_passage_ids),
                "realized_passage_ids": realized_passage_ids,
                "missing_passage_ids": sorted(assigned_passage_ids - observed_passage_ids),
            }
        )
    insight_issues = [
        item for item in insight_results
        if item["status"] in {"weak", "zero"}
    ]

    planned_pairs = {
        tuple(sorted((item.from_book_id, item.to_book_id)))
        for item in plan.cross_references
        if item.from_book_id and item.to_book_id and item.from_book_id != item.to_book_id
    }
    observed_pairs: set[tuple[str, str]] = set()
    for segment in script.segments:
        books = set(segment.source_book_ids)
        books.update(citation.book_id for citation in segment.citations)
        ordered_books = sorted(book for book in books if book)
        for idx in range(len(ordered_books)):
            for jdx in range(idx + 1, len(ordered_books)):
                observed_pairs.add((ordered_books[idx], ordered_books[jdx]))
    global_citation_books = sorted(
        {
            citation.book_id
            for citation in script.citations
            if citation.book_id
        }
    )
    for idx in range(len(global_citation_books)):
        for jdx in range(idx + 1, len(global_citation_books)):
            observed_pairs.add((global_citation_books[idx], global_citation_books[jdx]))

    covered_pairs = sorted(planned_pairs & observed_pairs)
    planned_pair_count = len(planned_pairs)
    coverage_ratio = (
        len(covered_pairs) / planned_pair_count
        if planned_pair_count > 0
        else 1.0
    )
    cross_reference_has_issues = (
        planned_pair_count > 0 and coverage_ratio < min_cross_reference_coverage
    )

    signal_counts: dict[str, int] = {}

    def _add_book_signal(book_id: str) -> None:
        if not book_id:
            return
        signal_counts[book_id] = signal_counts.get(book_id, 0) + 1

    for citation in script.citations:
        _add_book_signal(citation.book_id)
    for segment in script.segments:
        segment_books = set(segment.source_book_ids)
        segment_books.update(citation.book_id for citation in segment.citations)
        for book_id in segment_books:
            _add_book_signal(book_id)

    total_signals = sum(signal_counts.values())
    observed_balance = {
        book_id: (count / total_signals)
        for book_id, count in signal_counts.items()
        if total_signals > 0
    }
    drift_by_book = {}
    max_abs_drift = 0.0
    for book_id, planned_share in plan.book_balance.items():
        observed_share = observed_balance.get(book_id, 0.0)
        drift = abs(observed_share - float(planned_share))
        drift_by_book[book_id] = round(drift, 4)
        max_abs_drift = max(max_abs_drift, drift)

    insufficient_book_signal = bool(plan.book_balance) and total_signals == 0
    book_balance_has_issues = insufficient_book_signal or (
        bool(plan.book_balance)
        and total_signals > 0
        and max_abs_drift > max_book_balance_abs_drift
    )

    planned_scene_ids = [scene.scene_id for scene in plan.scene_cards]
    planned_scene_id_set = set(planned_scene_ids)
    observed_scene_id_set = {scene_id for scene_id in observed_scene_ids if scene_id}
    covered_scene_ids = [
        scene_id for scene_id in planned_scene_ids
        if scene_id in observed_scene_id_set
    ]
    collapsed_observed_scene_order: list[str] = []
    previous_scene_id: str | None = None
    for scene_id in observed_scene_ids:
        if scene_id == previous_scene_id:
            continue
        collapsed_observed_scene_order.append(scene_id)
        previous_scene_id = scene_id
    unexpected_scene_ids = [
        scene_id for scene_id in collapsed_observed_scene_order
        if scene_id not in planned_scene_id_set
    ]
    planned_scene_index = {
        scene_id: idx for idx, scene_id in enumerate(planned_scene_ids)
    }
    observed_scene_indices = [
        planned_scene_index[scene_id]
        for scene_id in collapsed_observed_scene_order
        if scene_id in planned_scene_index
    ]
    scene_order_preserved = observed_scene_indices == sorted(observed_scene_indices)
    scene_coverage_ratio = (
        len(covered_scene_ids) / len(planned_scene_ids)
        if planned_scene_ids
        else 1.0
    )
    anchor_scene_ids = [scene_id for scene_id in plan.anchor_scene_ids if scene_id]
    missing_anchor_scene_ids = [
        scene_id for scene_id in anchor_scene_ids
        if scene_id not in observed_scene_id_set
    ]
    scene_structure_has_issues = bool(planned_scene_ids) and (
        len(covered_scene_ids) < len(planned_scene_ids)
        or bool(unexpected_scene_ids)
        or not scene_order_preserved
        or bool(missing_anchor_scene_ids)
    )

    sections = {
        "insight_realization": {
            "has_issues": bool(insight_issues),
            "problem_count": len(insight_issues),
            "insights": insight_results,
            "observed_passage_count": len(observed_passage_ids),
        },
        "scene_structure": {
            "has_issues": scene_structure_has_issues,
            "planned_scene_count": len(planned_scene_ids),
            "covered_scene_count": len(covered_scene_ids),
            "coverage_ratio": round(scene_coverage_ratio, 4),
            "anchor_scene_count": len(anchor_scene_ids),
            "covered_anchor_scene_count": (
                len(anchor_scene_ids) - len(missing_anchor_scene_ids)
            ),
            "missing_scene_ids": [
                scene_id for scene_id in planned_scene_ids
                if scene_id not in observed_scene_id_set
            ],
            "missing_anchor_scene_ids": missing_anchor_scene_ids,
            "unexpected_scene_ids": unexpected_scene_ids,
            "order_preserved": scene_order_preserved,
        },
        "cross_references": {
            "has_issues": cross_reference_has_issues,
            "planned_pair_count": planned_pair_count,
            "covered_pair_count": len(covered_pairs),
            "coverage_ratio": round(coverage_ratio, 4),
            "minimum_coverage": min_cross_reference_coverage,
            "missing_pairs": [
                {"book_ids": [a, b]}
                for a, b in sorted(planned_pairs - observed_pairs)
            ],
        },
        "book_balance": {
            "has_issues": book_balance_has_issues,
            "insufficient_signal": insufficient_book_signal,
            "total_signals": total_signals,
            "signal_counts": signal_counts,
            "planned_balance": {
                book_id: round(float(share), 4)
                for book_id, share in plan.book_balance.items()
            },
            "observed_balance": {
                book_id: round(share, 4)
                for book_id, share in observed_balance.items()
            },
            "drift_by_book": drift_by_book,
            "max_abs_drift": round(max_abs_drift, 4),
            "max_allowed_abs_drift": max_book_balance_abs_drift,
        },
    }

    problem_count = sum(1 for section in sections.values() if section["has_issues"])
    return {
        "episode_number": plan.episode_number,
        "title": plan.title,
        "has_issues": problem_count > 0,
        "problem_count": problem_count,
        "thresholds": {
            "cross_reference_min_coverage": min_cross_reference_coverage,
            "book_balance_max_abs_drift": max_book_balance_abs_drift,
        },
        **sections,
    }


# ---------------------------------------------------------------------------
# Stage logging context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _stage_log(run_logger: RunLogger, stage_name: str, project_dir: Path, **input_summary):
    """Log stage start/end with timing and save input/output artifacts."""
    stage_dir = project_dir / "stage_artifacts" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    _save_json(stage_dir / "input.json", input_summary)

    start = time.monotonic()
    run_logger.log("stage_start", stage=stage_name, input_summary=input_summary)

    result_holder: dict[str, Any] = {}
    error_info: dict[str, Any] | None = None
    try:
        yield result_holder
    except Exception as exc:
        error_info = {"error_type": type(exc).__name__, "error_message": str(exc)}
        raise
    finally:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        output_summary = result_holder.get("output_summary", {})
        _save_json(stage_dir / "output.json", output_summary)
        event_payload: dict[str, Any] = {
            "stage": stage_name,
            "elapsed_ms": elapsed_ms,
            "output_summary": output_summary,
            "artifact_dir": str(stage_dir),
        }
        if error_info:
            event_payload["error"] = error_info
            run_logger.log("stage_error", **event_payload)
        else:
            run_logger.log("stage_end", **event_payload)


# ---------------------------------------------------------------------------
# Text chunking (Stage 3 — no LLM)
# ---------------------------------------------------------------------------


def chunk_text(
    raw_text: str,
    book_id: str,
    chapters: list[ChapterInfo],
    config: ChunkingConfig,
) -> list[TextChunk]:
    """Split chapter text into overlapping chunks."""
    chunks: list[TextChunk] = []
    global_index = 0
    for chapter in chapters:
        chapter_text = raw_text[chapter.start_index : chapter.end_index]
        chapter_chunks = _split_into_chunks(
            chapter_text,
            config.max_chunk_words,
            config.overlap_words,
            config.min_chunk_words,
            config.split_on,
        )
        for position, text_str in enumerate(chapter_chunks):
            word_count = len(text_str.split())
            chunks.append(
                TextChunk(
                    chunk_id=f"{book_id}-{chapter.chapter_id}-chunk-{global_index}",
                    book_id=book_id,
                    chapter_id=chapter.chapter_id,
                    text=text_str,
                    word_count=word_count,
                    position=position,
                    metadata={"author": "", "title": ""},
                )
            )
            global_index += 1
    return chunks


def _split_into_chunks(
    text: str,
    max_words: int,
    overlap_words: int,
    min_words: int,
    split_on: list[str],
) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if len(words) >= min_words else ([text] if text.strip() else [])

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        if end < len(words):
            best_split = -1
            for boundary in split_on:
                idx = chunk_str.rfind(boundary)
                if idx > len(chunk_str) // 2:
                    best_split = max(best_split, idx)
            if best_split > 0:
                chunk_str = chunk_str[: best_split + len(split_on[0])]
                end = start + len(chunk_str.split())

        if chunks and len(chunk_str.split()) < min_words:
            chunks[-1] = chunks[-1] + " " + chunk_str
            break

        chunks.append(chunk_str.strip())
        start = max(start + 1, end - overlap_words)

    return [c for c in chunks if c.strip()]


def _concat_file_entry(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", r"'\''")
    return f"file '{escaped}'\n"


def _sanitize_spoken_text(text: str) -> str:
    cleaned = _SPOKEN_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _resolve_spoken_render_speed(speech_rate: str, base_speed: float) -> float:
    multiplier = _SPOKEN_RATE_MULTIPLIER.get(speech_rate, 1.0)
    resolved = round(base_speed * multiplier, 2)
    return min(4.0, max(0.1, resolved))


def _supports_segment_tts_instructions(tts_provider: str) -> bool:
    return tts_provider.strip().lower() in {"openai", "openai-compatible"}


def _normalize_tts_instruction_text(text: str | None, *, max_chars: int | None = None) -> str | None:
    if not text:
        return None
    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if not normalized:
        return None
    if max_chars is not None and len(normalized) > max_chars:
        return normalized[: max_chars - 3].rstrip() + "..."
    return normalized


def _normalize_pronunciation_match_text(text: str) -> str:
    """Normalize text for punctuation/diacritic-tolerant phrase matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = without_marks.lower()
    # Keep alphanumerics only and collapse punctuation/spacing differences.
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return _WHITESPACE_RE.sub(" ", collapsed).strip()


def _segment_contains_pronunciation_term(segment_text: str, term: str) -> bool:
    normalized_segment = _normalize_pronunciation_match_text(segment_text)
    normalized_term = _normalize_pronunciation_match_text(term)
    if not normalized_segment or not normalized_term:
        return False
    return f" {normalized_term} " in f" {normalized_segment} "


def _segment_hint_degradations(hints: SpeechHints, tts_provider: str) -> list[str]:
    if _supports_segment_tts_instructions(tts_provider):
        return []
    degradations: list[str] = []
    if hints.style != "neutral" or hints.intensity != "none" or hints.render_strategy == "slow_clause":
        degradations.append("segment_instructions_not_supported")
    if hints.pronunciation_hints:
        degradations.append("pronunciation_hints_not_supported")
    if hints.emphasis_targets and hints.render_strategy == "plain":
        degradations.append("phrase_emphasis_requires_prompt_steering")
    return degradations


def _split_render_text(text: str, hints: SpeechHints) -> list[tuple[str | None, str]]:
    clean_text = text.strip()
    if not clean_text:
        return []
    if hints.render_strategy == "split_sentences":
        sentences = _split_sentences(clean_text)
        if len(sentences) > 1:
            return [(None, sentence) for sentence in sentences]
    if hints.render_strategy == "isolate_phrase":
        for phrase in hints.emphasis_targets:
            match = re.search(re.escape(phrase), clean_text, flags=re.IGNORECASE)
            if match is None:
                continue
            parts: list[tuple[str | None, str]] = []
            before = clean_text[:match.start()].strip()
            focus = clean_text[match.start():match.end()].strip()
            after = clean_text[match.end():].strip()
            if before:
                parts.append((None, before))
            if focus:
                parts.append((focus, focus))
            if after:
                parts.append((None, after))
            if parts:
                return parts
    return [(None, clean_text)]


def _build_segment_tts_instructions(
    hints: SpeechHints,
    *,
    base_instructions: str | None,
    focus_phrase: str | None = None,
    emphasis_targets: list[str] | None = None,
    pronunciation_hints: list[Any] | None = None,
) -> str | None:
    parts: list[str] = []
    normalized_base = _normalize_tts_instruction_text(base_instructions)
    if normalized_base:
        parts.append(normalized_base)

    overlay: list[str] = []
    if hints.style != "neutral":
        overlay.append(f"Keep the delivery {hints.style}.")
    if hints.intensity != "none":
        overlay.append(f"Use {hints.intensity} vocal emphasis where it feels natural.")
    if hints.render_strategy == "slow_clause":
        overlay.append("Linger slightly on the most reflective clause without sounding theatrical.")

    targets = emphasis_targets if emphasis_targets is not None else (
        [focus_phrase] if focus_phrase else hints.emphasis_targets[:3]
    )
    if targets:
        stress = hints.intensity if hints.intensity != "none" else "light"
        overlay.append(
            f"Give {stress} stress to these phrases when natural: {', '.join(repr(target) for target in targets)}."
        )
    effective_pronunciation_hints = (
        pronunciation_hints
        if pronunciation_hints is not None
        else hints.pronunciation_hints
    )
    if effective_pronunciation_hints:
        pronunciation_text = "; ".join(
            f"{hint.text} as {hint.spoken_as}"
            for hint in effective_pronunciation_hints[:4]
        )
        overlay.append(f"Use these pronunciations: {pronunciation_text}.")

    if overlay:
        parts.append("Segment guidance: " + " ".join(overlay))
    return _normalize_tts_instruction_text("\n\n".join(parts))


def _render_segments_for_spoken_segment(
    segment: SpokenSegment,
    *,
    voice_id: str,
    speed: float,
    tts_provider: str,
    base_instructions: str | None,
) -> list[RenderSegment]:
    hints = segment.speech_hints
    render_speed = _resolve_spoken_render_speed(hints.pace, speed)
    if hints.render_strategy == "slow_clause":
        render_speed = _resolve_spoken_render_speed("slower", render_speed)

    text_parts = _split_render_text(segment.text, hints)
    degradations = _segment_hint_degradations(hints, tts_provider)
    supports_instructions = _supports_segment_tts_instructions(tts_provider)
    render_segments: list[RenderSegment] = []

    for idx, (focus_phrase, part_text) in enumerate(text_parts, start=1):
        piece_text = part_text.strip()
        if not piece_text:
            continue
        matched_emphasis_targets: list[str]
        if focus_phrase:
            matched_emphasis_targets = [focus_phrase]
        else:
            matched_emphasis_targets = []
            for phrase in hints.emphasis_targets:
                candidate = phrase.strip()
                if not candidate:
                    continue
                if re.search(re.escape(candidate), piece_text, flags=re.IGNORECASE):
                    matched_emphasis_targets.append(candidate)
                if len(matched_emphasis_targets) >= 3:
                    break
        matched_pronunciation_hints = [
            hint
            for hint in hints.pronunciation_hints
            if _segment_contains_pronunciation_term(piece_text, hint.text)
        ]
        is_single_part = len(text_parts) == 1
        render_segments.append(
            RenderSegment(
                segment_id=segment.segment_id if is_single_part else f"{segment.segment_id}_{idx}",
                text=piece_text,
                voice_id=voice_id,
                speed=render_speed,
                pause_before_ms=hints.pause_before_ms if idx == 1 else 0,
                pause_after_ms=hints.pause_after_ms if idx == len(text_parts) else 0,
                instructions=(
                    _build_segment_tts_instructions(
                        hints,
                        base_instructions=base_instructions,
                        focus_phrase=focus_phrase,
                        emphasis_targets=matched_emphasis_targets,
                        pronunciation_hints=matched_pronunciation_hints,
                    )
                    if supports_instructions
                    else None
                ),
                hint_degradations=degradations,
            )
        )
    return render_segments


def _normalize_spoken_segments(
    segments: list[SpokenSegment],
    max_words_per_segment: int,
) -> list[SpokenSegment]:
    normalized: list[SpokenSegment] = []
    for seg in segments:
        max_words = max(1, min(seg.max_words, max_words_per_segment))
        cleaned_text = _sanitize_spoken_text(seg.text)
        if not cleaned_text:
            continue

        chunks = _split_into_chunks(
            cleaned_text,
            max_words=max_words,
            overlap_words=0,
            min_words=1,
            split_on=[". ", "? ", "! ", "; ", ", "],
        )
        if not chunks:
            chunks = [cleaned_text]

        for idx, chunk in enumerate(chunks, start=1):
            segment_id = seg.segment_id if idx == 1 else f"{seg.segment_id}_{idx}"
            normalized.append(
                seg.model_copy(
                    update={
                        "segment_id": segment_id,
                        "text": chunk,
                        "max_words": max_words,
                    }
                )
            )
    return normalized


# ---------------------------------------------------------------------------
# Render manifest construction (Stage 15 — no LLM)
# ---------------------------------------------------------------------------


def build_render_manifest(
    spoken_script: SpokenScript,
    voice_id: str = "ballad",
    speed: float = 1.0,
    words_per_minute: int = 130,
    base_instructions: str | None = None,
) -> RenderManifest:
    segments: list[RenderSegment] = []
    tts_provider = spoken_script.tts_provider
    framing = spoken_script.framing
    framing_texts = [
        ("framing_opening_image", framing.opening_image, 0, 900),
        ("framing_threat", framing.threat_or_unresolved_action, 200, 800),
        ("framing_question", framing.opening_question, 200, 1000),
    ]
    if framing.recap:
        framing_texts.insert(0, ("framing_recap", framing.recap, 500, 900))
    if framing.preview:
        framing_texts.append(("framing_preview", framing.preview, 900, 0))
    for segment_id, text, pause_before, pause_after in framing_texts:
        if not text.strip():
            continue
        segments.append(
            RenderSegment(
                segment_id=segment_id,
                text=text,
                voice_id=voice_id,
                speed=speed,
                pause_before_ms=pause_before,
                pause_after_ms=pause_after,
            )
        )

    for idx, section in enumerate(spoken_script.sections):
        segments.extend(
            _render_segments_for_spoken_segment(
                SimpleNamespace(
                    segment_id=section.section_id,
                    text=section.text,
                    speech_hints=section.speech_hints,
                ),
                voice_id=voice_id,
                speed=speed,
                tts_provider=tts_provider,
                base_instructions=base_instructions,
            )
        )
        if idx < len(spoken_script.transitions):
            transition = spoken_script.transitions[idx]
            segments.extend(
                _render_segments_for_spoken_segment(
                    SimpleNamespace(
                        segment_id=transition.transition_id,
                        text=transition.text,
                        speech_hints=transition.speech_hints,
                    ),
                    voice_id=voice_id,
                    speed=speed,
                    tts_provider=tts_provider,
                    base_instructions=base_instructions,
                )
            )

    total_words = sum(len(seg.text.split()) for seg in segments)
    estimated_seconds = int(total_words / words_per_minute * 60)

    return RenderManifest(
        episode_number=spoken_script.episode_number,
        segments=segments,
        total_segments=len(segments),
        estimated_duration_seconds=estimated_seconds,
    )


def _script_text_units(script: EpisodeScript) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for section in script.prose_sections:
        units.append((section.section_id, section.text))
    for transition in script.transitions:
        units.append((transition.transition_id, transition.text))
    return units


def _script_total_word_count(script: EpisodeScript) -> int:
    return sum(len(text.split()) for _, text in _script_text_units(script))


def _estimate_duration_seconds_from_words(word_count: int, words_per_minute: float) -> int:
    if words_per_minute <= 0:
        return 0
    return int((float(word_count) / float(words_per_minute)) * 60)


def _compute_scene_word_count_targets(
    scene_cards: list[SceneCard],
    episode_target_word_count: int,
    words_per_minute: float,
) -> dict[str, int]:
    if not scene_cards:
        return {}
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be positive")

    invalid_scene_ids = [
        scene.scene_id
        for scene in scene_cards
        if int(scene.estimated_duration_seconds) <= 0
    ]
    if invalid_scene_ids:
        raise ValueError(
            "scene cards must define positive estimated_duration_seconds: "
            + ", ".join(invalid_scene_ids)
        )

    raw_targets = [
        (float(scene.estimated_duration_seconds) * float(words_per_minute)) / 60.0
        for scene in scene_cards
    ]
    floor_targets = [int(math.floor(target)) for target in raw_targets]
    target_total = int(round(sum(raw_targets)))
    remainder = max(0, target_total - sum(floor_targets))

    ranked_indices = sorted(
        range(len(raw_targets)),
        key=lambda idx: (raw_targets[idx] - floor_targets[idx], -idx),
        reverse=True,
    )
    for idx in ranked_indices[:remainder]:
        floor_targets[idx] += 1

    return {
        scene.scene_id: floor_targets[idx]
        for idx, scene in enumerate(scene_cards)
    }


def _cluster_batch_group_sizes(cluster_count: int) -> list[int]:
    if cluster_count <= 0:
        return []
    if cluster_count <= 2:
        return [1] * cluster_count

    batch_count = min(2, cluster_count)
    base = cluster_count // batch_count
    remainder = cluster_count % batch_count
    return [base + (1 if idx < remainder else 0) for idx in range(batch_count)]


def _build_cluster_scene_batches(scene_cards: list[SceneCard]) -> list[list[SceneCard]]:
    if not scene_cards:
        return []

    ordered_cluster_ids: list[str] = []
    seen_cluster_ids: set[str] = set()
    for scene in scene_cards:
        if scene.card_kind != "normal":
            continue
        cluster_id = scene.dominant_cluster_occurrence_id
        if not cluster_id or cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)
        ordered_cluster_ids.append(cluster_id)

    if not ordered_cluster_ids:
        return [list(scene_cards)]

    group_sizes = _cluster_batch_group_sizes(len(ordered_cluster_ids))
    cluster_groups: list[list[str]] = []
    start = 0
    for size in group_sizes:
        end = start + size
        cluster_groups.append(ordered_cluster_ids[start:end])
        start = end

    cluster_to_batch_index: dict[str, int] = {}
    for batch_index, cluster_group in enumerate(cluster_groups):
        for cluster_id in cluster_group:
            cluster_to_batch_index[cluster_id] = batch_index

    batches: list[list[SceneCard]] = [[] for _ in cluster_groups]
    last_batch_index = 0
    for scene in scene_cards:
        batch_index = last_batch_index
        if scene.card_kind == "normal":
            cluster_id = scene.dominant_cluster_occurrence_id
            if cluster_id is not None:
                batch_index = cluster_to_batch_index.get(cluster_id, last_batch_index)
        else:
            if scene.bridge_to_occurrence_id is not None:
                batch_index = cluster_to_batch_index.get(scene.bridge_to_occurrence_id, last_batch_index)
        batches[batch_index].append(scene)
        last_batch_index = batch_index

    return [batch for batch in batches if batch]


def _build_passage_lookup(corpus: ThematicCorpus) -> dict[str, ExtractedPassage]:
    passage_lookup: dict[str, ExtractedPassage] = {}
    for axis_passages in corpus.passages_by_axis.values():
        for passage in axis_passages:
            passage_lookup[passage.passage_id] = passage
    return passage_lookup


def _primitive_passage_ids(primitive: SynthesisPrimitive) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for passage_id in [*primitive.core_passage_ids, *primitive.support_passage_ids]:
        if not passage_id or passage_id in seen:
            continue
        seen.add(passage_id)
        ordered.append(passage_id)
    return ordered


def _flatten_synthesis_primitives(synthesis_map: SynthesisMap) -> dict[str, SynthesisPrimitive]:
    flattened: dict[str, SynthesisPrimitive] = {}
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        for item in synthesis_map.primitives_by_family.get(family, []):
            flattened[item.id] = item
    return flattened


def _build_episode_synthesis_map_payload(
    synthesis_map: SynthesisMap,
    cluster_ids: list[str],
    cluster_lookup: dict[str, EpisodeCandidateCluster],
) -> tuple[dict[str, Any], list[str]]:
    selected_clusters: list[EpisodeCandidateCluster] = []
    seen_cluster_ids: set[str] = set()
    primitive_ids: list[str] = []
    seen_primitive_ids: set[str] = set()

    for cluster_id in cluster_ids:
        if cluster_id in seen_cluster_ids:
            continue
        cluster = cluster_lookup.get(cluster_id)
        if cluster is None:
            raise RuntimeError(f"Unknown cluster_id in strategy: {cluster_id}")
        seen_cluster_ids.add(cluster_id)
        selected_clusters.append(cluster)
        for member_id in cluster.member_ids:
            if member_id in seen_primitive_ids:
                continue
            seen_primitive_ids.add(member_id)
            primitive_ids.append(member_id)

    primitive_id_set = set(primitive_ids)
    primitives_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        primitives_by_family[family] = [
            item.model_dump(mode="json")
            for item in synthesis_map.primitives_by_family.get(family, [])
            if item.id in primitive_id_set
        ]
    payload = {
        "project_id": synthesis_map.project_id,
        "episode_candidate_clusters": [
            cluster.model_dump(mode="json") for cluster in selected_clusters
        ],
        "primitives_by_family": primitives_by_family,
        "quality_score": synthesis_map.quality_score,
        "quality_notes": list(synthesis_map.quality_notes),
    }
    return payload, primitive_ids


def _reconstruct_synthesis_map(
    *,
    project_id: str,
    primitives: SynthesisPrimitivesArtifact,
    consolidation: SynthesisConsolidationResult,
) -> SynthesisMap:
    def _select_family_items(*, ids: list[str], items_by_id: dict[str, SynthesisPrimitive], family_name: str) -> list[SynthesisPrimitive]:
        selected: list[SynthesisPrimitive] = []
        seen: set[str] = set()
        missing: list[str] = []
        for primitive_id in ids:
            if primitive_id in seen:
                continue
            seen.add(primitive_id)
            item = items_by_id.get(primitive_id)
            if item is None:
                missing.append(primitive_id)
                continue
            selected.append(item)
        if missing:
            raise RuntimeError(
                f"synthesis_consolidation returned unknown {family_name}: {sorted(missing)}"
            )
        return selected

    primitives_by_family: dict[str, list[SynthesisPrimitive]] = {}
    primitive_ids_by_family = consolidation.primitive_ids_by_family
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        items = primitives.primitives_by_family.get(family, [])
        items_by_id = {item.id: item for item in items}
        primitives_by_family[family] = _select_family_items(
            ids=primitive_ids_by_family.get(family, []),
            items_by_id=items_by_id,
            family_name=f"{family}_ids",
        )

    return SynthesisMap(
        project_id=project_id,
        episode_candidate_clusters=list(consolidation.episode_candidate_clusters),
        primitives_by_family=primitives_by_family,
        quality_score=consolidation.quality_score,
        quality_notes=list(consolidation.quality_notes),
    )


def _merge_actor_metric_dicts(metrics_items: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metrics in metrics_items:
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, int | float):
                merged[key] = merged.get(key, 0) + value
    return merged


def _confidence_counts(actor_metadata: ActorMetadata) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for actor in actor_metadata.actors:
        counts[actor.evidence_confidence] = counts.get(actor.evidence_confidence, 0) + 1
    return counts


def _actor_linkage_counts(
    *,
    synthesis_map: SynthesisMap,
    strategy: NarrativeStrategy,
    episode_plans: list[EpisodePlan],
) -> dict[str, int]:
    primitives = [
        primitive
        for family in SYNTHESIS_PRIMITIVE_FAMILIES
        for primitive in synthesis_map.primitives_by_family.get(family, [])
    ]
    clusters = synthesis_map.episode_candidate_clusters
    scenes = [scene for plan in episode_plans for scene in plan.scene_cards]
    return {
        "primitive_count": len(primitives),
        "actor_linked_primitive_count": sum(1 for primitive in primitives if primitive.actor_ids),
        "cluster_count": len(clusters),
        "actor_linked_cluster_count": sum(1 for cluster in clusters if cluster.actor_ids),
        "episode_count": len(strategy.episodes),
        "actor_linked_episode_count": sum(1 for episode in strategy.episodes if episode.actor_arc_directives),
        "scene_count": len(scenes),
        "actor_linked_scene_count": sum(
            1
            for scene in scenes
            if any(actor.actor_id for actor in scene.actors)
        ),
    }


def _preview_ids(ids: list[str], *, limit: int = 8) -> str:
    if not ids:
        return ""
    preview = ids[:limit]
    suffix = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ", ".join(preview) + suffix


def _build_scene_card_count_warnings(
    *,
    scene_card_count: int,
    scene_card_target_min: int,
    scene_card_target_max: int,
) -> list[str]:
    warnings: list[str] = []
    if scene_card_count < scene_card_target_min:
        warnings.append(
            "scene_card_count_below_target: "
            f"{scene_card_count} < {scene_card_target_min} (target range {scene_card_target_min}-{scene_card_target_max})"
        )
    elif scene_card_count > scene_card_target_max:
        warnings.append(
            "scene_card_count_above_target: "
            f"{scene_card_count} > {scene_card_target_max} (target range {scene_card_target_min}-{scene_card_target_max})"
        )
    return warnings


def _build_scene_card_primitive_warnings(
    *,
    scene_cards: list[SceneCard],
    primitive_pool_ids: set[str],
    primitive_min: int,
    primitive_max: int,
) -> list[str]:
    warnings: list[str] = []
    normal_cards = [scene for scene in scene_cards if scene.card_kind == "normal"]
    out_of_bounds_cards = [
        scene.scene_id
        for scene in normal_cards
        if len(scene.primitive_ids) < primitive_min or len(scene.primitive_ids) > primitive_max
    ]
    if out_of_bounds_cards:
        warnings.append(
            "normal_scene_primitive_density_out_of_range: "
            f"{len(out_of_bounds_cards)} scenes outside {primitive_min}-{primitive_max} primitives "
            f"({_preview_ids(out_of_bounds_cards)})"
        )

    unknown_primitive_ids = sorted(
        {
            primitive_id
            for scene in scene_cards
            for primitive_id in scene.primitive_ids
            if primitive_id not in primitive_pool_ids
        }
    )
    if unknown_primitive_ids:
        warnings.append(
            "scene_card_unknown_primitive_ids: "
            f"{len(unknown_primitive_ids)} unknown ids ({_preview_ids(unknown_primitive_ids)})"
        )

    if normal_cards:
        min_distinct_needed = int(math.ceil(len(normal_cards) / max(1, primitive_max)))
        if primitive_pool_ids and len(primitive_pool_ids) < min_distinct_needed:
            warnings.append(
                "primitive_pool_too_small_for_mapping_target: "
                f"pool={len(primitive_pool_ids)} distinct primitives for {len(normal_cards)} normal scenes "
                f"(>= {min_distinct_needed} suggested for {primitive_min}-{primitive_max} mapping)"
            )
        primitive_use_counts: dict[str, int] = {}
        for scene in normal_cards:
            for primitive_id in scene.primitive_ids:
                primitive_use_counts[primitive_id] = primitive_use_counts.get(primitive_id, 0) + 1
        heavily_reused = sorted(
            primitive_id
            for primitive_id, count in primitive_use_counts.items()
            if count > _PRIMITIVE_REUSE_WARNING_THRESHOLD
        )
        if heavily_reused:
            warnings.append(
                "primitive_reuse_heavy: "
                f"{len(heavily_reused)} primitives reused in more than {_PRIMITIVE_REUSE_WARNING_THRESHOLD} scenes "
                f"({_preview_ids(heavily_reused)})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Orchestrates the four-phase multi-book thematic podcast pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.run_logger = RunLogger(self.settings.pipeline.artifact_root)
        self.llm: LLMClient = build_llm_client(self.settings)
        self.vector_store = PGVectorRetrieval(self.settings, run_logger=self.run_logger)
        self.retrieval = RetrievalService(self.settings, self.vector_store)
        self.tts_client = build_tts_client(self.settings)
        self.llm.set_run_logger(self.run_logger)
        self.tts_client.set_run_logger(self.run_logger)

        # Configure per-schema concurrency semaphores
        per_schema: dict[str, int] = {}
        for schema_name, agent_cfg in self.settings.llm.agent_configs.items():
            if agent_cfg.concurrency_limit is not None:
                per_schema[schema_name] = agent_cfg.concurrency_limit
        configure_llm_semaphore(
            default_limit=self.settings.pipeline.llm_global_max_concurrency,
            per_schema=per_schema,
        )

        # Build agents with per-schema retry counts
        def _retries(name: str) -> int:
            return self.settings.llm.resolve_max_retry_attempts(name)

        self.chapter_summary_agent = ChapterSummaryAgent(self.llm, max_retry_attempts=_retries("chapter_summary"))
        self.book_summary_agent = BookSummaryAgent(self.llm, max_retry_attempts=_retries("book_summary"))
        self.theme_decomposition_agent = ThemeDecompositionAgent(self.llm, max_retry_attempts=_retries("theme_decomposition"))
        self.passage_extraction_agent = PassageExtractionAgent(self.llm, max_retry_attempts=_retries("passage_extraction"))
        self.synthesis_primitives_agent = SynthesisPrimitivesAgent(self.llm, max_retry_attempts=_retries("synthesis_primitives"))
        self.synthesis_consolidation_agent = SynthesisConsolidationAgent(self.llm, max_retry_attempts=_retries("synthesis_consolidation"))
        self.narrative_strategy_agent = NarrativeStrategyAgent(self.llm, max_retry_attempts=_retries("narrative_strategy"))
        self.episode_planning_agent = EpisodePlanningAgent(self.llm, max_retry_attempts=_retries("episode_planning"))
        self.writing_agent = WritingAgent(self.llm, max_retry_attempts=_retries("episode_writing"))
        self.writing_agent_no_citations = WritingAgentNoCitations(
            self.llm,
            max_retry_attempts=_retries("episode_writing"),
        )
        self.grounding_agent = GroundingValidationAgent(self.llm, max_retry_attempts=_retries("grounding_validation"))
        self.repair_agent = RepairAgent(self.llm, max_retry_attempts=_retries("repair"))
        self.spoken_delivery_agent = SpokenDeliveryAgent(self.llm, max_retry_attempts=_retries("spoken_delivery"))

    def _bind_run_logger(self, project_dir: Path) -> None:
        if self.run_logger.artifact_root != project_dir.parent:
            self.run_logger = RunLogger(project_dir.parent)
            self.llm.set_run_logger(self.run_logger)
            self.tts_client.set_run_logger(self.run_logger)
        self.run_logger.bind_run(project_dir.name)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    async def run_multi_book_podcast(
        self,
        source_paths: list[str],
        theme: str,
        episode_count: int | None,
        config: PipelineConfig | None = None,
        theme_elaboration: str | None = None,
        sub_themes: list[str] | None = None,
        titles: list[str] | None = None,
        authors: list[str] | None = None,
        project_id: str | None = None,
    ) -> ThematicProject:
        pipeline_config = config or PipelineConfig()
        project_id = project_id or uuid4().hex
        project_dir = self.settings.pipeline.artifact_root / project_id

        self._bind_run_logger(project_dir)
        database_configured = bool(self.settings.database.dsn)
        retrieval_enabled = self.vector_store.enabled
        self.run_logger.log(
            "pipeline_start",
            theme=theme,
            sub_themes=sub_themes or [],
            sub_theme_count=len(sub_themes or []),
            episode_count=episode_count,
            requested_episode_count=episode_count,
            book_count=len(source_paths),
            skip_grounding=pipeline_config.skip_grounding,
            skip_spoken_delivery=pipeline_config.skip_spoken_delivery,
            database_configured=database_configured,
            retrieval_enabled=retrieval_enabled,
            retrieval_collection=self.settings.retrieval.collection_name,
        )
        self.run_logger.log(
            "retrieval_status",
            database_configured=database_configured,
            retrieval_enabled=retrieval_enabled,
            retrieval_collection=self.settings.retrieval.collection_name,
        )

        project = ThematicProject(
            project_id=project_id,
            theme=theme,
            theme_elaboration=theme_elaboration,
            sub_themes=sub_themes or [],
            requested_episode_count=episode_count,
            episode_count=episode_count or 3,
            config=pipeline_config,
            status=ProjectStatus.INGESTING,
        )

        # Phase 1: Ingest & Index (parallel per book)
        logger.info("Phase 1: Ingest & Index (%d books)", len(source_paths))
        book_tasks = []
        for i, path in enumerate(source_paths):
            title = titles[i] if titles and i < len(titles) else Path(path).stem
            author = authors[i] if authors and i < len(authors) else "Unknown"
            book_tasks.append(
                self._ingest_and_index_book(
                    path,
                    title,
                    author,
                    project_id,
                    project_dir,
                    pipeline_config,
                    theme=theme,
                    sub_themes=sub_themes,
                    theme_elaboration=theme_elaboration,
                )
            )
        book_results = await asyncio.gather(*book_tasks, return_exceptions=True)

        successful_books: list[BookRecord] = []
        for i, result in enumerate(book_results):
            if isinstance(result, Exception):
                logger.error("Book %d failed: %s", i, result)
                self.run_logger.log("book_ingest_failed", index=i, error=str(result))
            else:
                successful_books.append(result)

        if len(successful_books) < 2:
            project = project.model_copy(update={"status": ProjectStatus.FAILED})
            _save_json(project_dir / "thematic_project.json", project)
            raise RuntimeError(
                f"Only {len(successful_books)} books ingested successfully. Minimum 2 required."
            )

        project = project.model_copy(update={
            "books": successful_books,
            "status": ProjectStatus.ANALYZING,
        })
        _save_json(project_dir / "thematic_project.json", project)
        self.run_logger.log(
            "convergence_barrier",
            successful_books=len(successful_books),
            total_words=sum(b.total_words for b in successful_books),
        )

        # Phase 2: Thematic Intelligence (sequential)
        logger.info("Phase 2: Thematic Intelligence")

        axes, actor_metadata, actor_metrics = await self._decompose_theme(project, project_dir)
        corpus = await self._extract_passages(project, axes, project_dir)
        synthesis_map, synthesis_actor_metrics = await self._map_synthesis(
            project, corpus, project_dir, actor_metadata,
        )
        actor_metrics["synthesis_primitives"] = synthesis_actor_metrics.get("primitives", {})
        actor_metrics["synthesis_consolidation"] = synthesis_actor_metrics.get("consolidation", {})

        if synthesis_map.quality_score < pipeline_config.synthesis_quality_threshold:
            logger.warning(
                "Synthesis quality %.2f below threshold %.2f. "
                "Books may lack thematic overlap for strong synthesis.",
                synthesis_map.quality_score, pipeline_config.synthesis_quality_threshold,
            )
            self.run_logger.log(
                "synthesis_quality_warning",
                score=synthesis_map.quality_score,
                threshold=pipeline_config.synthesis_quality_threshold,
            )

        strategy, strategy_actor_metrics = await self._choose_narrative_strategy(
            project, synthesis_map, corpus, project_dir, actor_metadata,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        project = self._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        episode_plans, planning_actor_metrics = await self._plan_series(
            project, synthesis_map, strategy, corpus, project_dir, actor_metadata,
        )
        actor_metrics["episode_planning"] = planning_actor_metrics

        # Phase 3: Episode Production (parallel per episode)
        logger.info("Phase 3: Episode Production (%d episodes)", len(episode_plans))
        project = project.model_copy(update={"status": ProjectStatus.PRODUCING})

        sem = asyncio.Semaphore(pipeline_config.episode_write_concurrency)
        ep_tasks = [
            self._produce_episode(plan, project, corpus, actor_metadata, project_dir, sem)
            for plan in episode_plans
        ]
        ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)

        spoken_scripts: list[tuple[int, SpokenScript]] = []
        for result in ep_results:
            if isinstance(result, Exception):
                logger.error("Episode production failed: %s", result)
            else:
                spoken_scripts.append(result)
        spoken_scripts.sort(key=lambda x: x[0])
        self._write_passage_utilization(
            project=project,
            corpus=corpus,
            episode_plans=episode_plans,
            project_dir=project_dir,
            episode_numbers=[episode_number for episode_number, _ in spoken_scripts],
        )
        actor_metrics["writing"] = self._build_writing_actor_metrics(project_dir, spoken_scripts)
        self._write_actor_metadata_metrics(
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_plans=episode_plans,
            metrics=actor_metrics,
        )

        # Phase 4: Audio Rendering (parallel per episode)
        logger.info("Phase 4: Audio Rendering")
        audio_sem = asyncio.Semaphore(pipeline_config.tts_concurrency)
        audio_tasks = [
            self._render_episode_audio(
                ep_num,
                spoken,
                project.config,
                project_dir,
                audio_sem,
                skip_audio=pipeline_config.skip_audio,
            )
            for ep_num, spoken in spoken_scripts
        ]
        await asyncio.gather(*audio_tasks, return_exceptions=True)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
        self.run_logger.log("pipeline_complete", project_id=project_id)
        logger.info("Pipeline complete. Artifacts at %s", project_dir)

        return project

    async def synthesize_audio_from_run(self, run_dir: Path) -> dict[str, Any]:
        project_dir = run_dir.resolve()
        if not project_dir.exists() or not project_dir.is_dir():
            raise RuntimeError(f"Run directory not found: {project_dir}")

        episodes_dir = project_dir / "episodes"
        if not episodes_dir.exists() or not episodes_dir.is_dir():
            raise RuntimeError(f"Run directory does not contain episodes/: {project_dir}")

        self._bind_run_logger(project_dir)

        episode_dirs = sorted(
            [path for path in episodes_dir.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name),
        )
        if not episode_dirs:
            raise RuntimeError(f"No episode directories found under {episodes_dir}")

        manifests: list[tuple[int, Path]] = []
        skipped_episodes: list[int] = []
        for episode_dir in episode_dirs:
            episode_number = int(episode_dir.name)
            manifest_path = episode_dir / "render_manifest.json"
            if manifest_path.exists():
                manifests.append((episode_number, manifest_path))
            else:
                skipped_episodes.append(episode_number)

        if not manifests:
            raise RuntimeError(f"No render_manifest.json files found under {episodes_dir}")

        self._ensure_ffmpeg_available()
        semaphore = asyncio.Semaphore(self.settings.pipeline.tts_concurrency)
        tasks = [
            self._render_existing_episode_audio(
                episode_number, manifest_path, project_dir, semaphore,
            )
            for episode_number, manifest_path in manifests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        succeeded = 0
        failed = 0
        failures: list[str] = []
        for episode_number, result in zip((ep for ep, _ in manifests), results, strict=True):
            if isinstance(result, Exception):
                failed += 1
                failures.append(f"episode {episode_number}: {result}")
            else:
                succeeded += 1

        summary = {
            "run_dir": str(project_dir),
            "processed": len(manifests),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": len(skipped_episodes),
            "skipped_episodes": skipped_episodes,
            "failures": failures,
        }
        self.run_logger.log("audio_resynthesis_complete", **summary)
        return summary

    # -----------------------------------------------------------------------
    # Phase 1: Ingest & Index
    # -----------------------------------------------------------------------

    async def _ingest_and_index_book(
        self,
        source_path: str,
        title: str,
        author: str,
        project_id: str,
        project_dir: Path,
        config: PipelineConfig,
        *,
        theme: str,
        sub_themes: list[str] | None = None,
        theme_elaboration: str | None = None,
    ) -> BookRecord:
        path = Path(source_path)
        book_id = uuid4().hex
        book_dir = project_dir / "books" / book_id

        async with _stage_log(
            self.run_logger, f"ingest_book_{book_id[:8]}", project_dir,
            book_id=book_id, title=title, path=source_path,
        ) as ctx:
            # Stage 1: Read source
            raw_text = await asyncio.to_thread(read_source_text, path)
            total_words = len(raw_text.split())
            source_type = path.suffix.lower().lstrip(".")
            if source_type not in ("pdf", "txt", "md"):
                source_type = "txt"

            book_record = BookRecord(
                book_id=book_id, title=title, author=author,
                source_path=source_path, source_type=source_type,
                total_words=total_words,
            )

            book_dir.mkdir(parents=True, exist_ok=True)
            (book_dir / "raw_text.txt").write_text(raw_text, encoding="utf-8")

            # Stage 2: Structure chapters
            chapters = await self._structure_chapters(
                book_record,
                raw_text,
                project_dir,
                theme=theme,
                sub_themes=sub_themes,
                theme_elaboration=theme_elaboration,
            )
            book_record = book_record.model_copy(update={"chapters": chapters})
            _save_json(book_dir / "book_record.json", book_record)

            # Stage 3: Chunk text
            chunking_config = ChunkingConfig(
                max_chunk_words=config.chunk_max_words,
                overlap_words=config.chunk_overlap_words,
            )
            chunks = chunk_text(raw_text, book_id, chapters, chunking_config)
            for c in chunks:
                c.metadata["author"] = author
                c.metadata["title"] = title
            book_record = book_record.model_copy(update={"chunk_count": len(chunks)})
            _save_json(book_dir / "book_record.json", book_record)

            # Stage 4: Embed & Store
            await asyncio.to_thread(self.vector_store.index_chunks, chunks, project_id)

            ctx["output_summary"] = {
                "book_id": book_id, "title": title,
                "chapters": len(chapters), "chunks": len(chunks), "words": total_words,
            }
            return book_record

    async def _structure_chapters(
        self,
        book_record: BookRecord,
        raw_text: str,
        project_dir: Path,
        *,
        theme: str,
        sub_themes: list[str] | None = None,
        theme_elaboration: str | None = None,
    ) -> list[ChapterInfo]:
        async with _stage_log(
            self.run_logger, f"structure_{book_record.book_id[:8]}", project_dir,
            book_id=book_record.book_id, text_length=len(raw_text),
        ) as ctx:
            chapters = extract_chapters_from_source(raw_text)
            summary_tasks = []
            for chapter in chapters:
                chapter_text = raw_text[chapter.start_index : chapter.end_index]
                payload = self.chapter_summary_agent.build_payload(
                    theme=theme,
                    sub_themes=sub_themes,
                    theme_elaboration=theme_elaboration,
                    book_id=book_record.book_id,
                    title=book_record.title,
                    author=book_record.author,
                    chapter_title=chapter.title,
                    chapter_text=chapter_text,
                )
                summary_tasks.append(asyncio.to_thread(self.chapter_summary_agent.run, payload))

            summaries = await asyncio.gather(*summary_tasks)
            updated: list[ChapterInfo] = []
            for chapter, summary in zip(chapters, summaries, strict=True):
                updated.append(chapter.model_copy(update={
                    "analysis": summary.analysis,
                }))
            chapters = updated

            ctx["output_summary"] = {
                "chapter_count": len(chapters), "windows_processed": 0,
            }
            return chapters

    # -----------------------------------------------------------------------
    # Phase 2: Thematic Intelligence
    # -----------------------------------------------------------------------

    async def _decompose_theme(
        self, project: ThematicProject, project_dir: Path,
    ) -> tuple[list[ThematicAxis], ActorMetadata, dict[str, Any]]:
        async with _stage_log(
            self.run_logger, "theme_decomposition", project_dir,
            theme=project.theme, sub_themes=project.sub_themes, book_count=len(project.books),
        ) as ctx:
            summary_payloads: list[tuple[str, dict[str, Any]]] = []
            for book in project.books:
                chapter_info = [
                    _build_compact_chapter_projection(ch)
                    for ch in book.chapters
                ]
                summary_payloads.append((
                    book.book_id,
                    self.book_summary_agent.build_payload(
                        theme=project.theme,
                        sub_themes=project.sub_themes,
                        theme_elaboration=project.theme_elaboration,
                        book_id=book.book_id,
                        title=book.title,
                        author=book.author,
                        chapters=chapter_info,
                    ),
                ))

            summary_results = await asyncio.gather(*[
                asyncio.to_thread(self.book_summary_agent.run, payload)
                for _, payload in summary_payloads
            ])
            book_summaries = {
                book_id: result.summary
                for (book_id, _), result in zip(summary_payloads, summary_results, strict=True)
            }

            payload = self.theme_decomposition_agent.build_payload(
                theme=project.theme,
                sub_themes=project.sub_themes,
                theme_elaboration=project.theme_elaboration,
                books=project.books,
                book_summaries=book_summaries,
            )
            expected_book_ids = [book.book_id for book in project.books]
            max_attempts = self.theme_decomposition_agent.max_retry_attempts
            axes: list[ThematicAxis] = []
            actor_metadata = ActorMetadata(project_id=project.project_id)
            actor_metrics: dict[str, Any] = {}
            for attempt in range(1, max_attempts + 1):
                result = await asyncio.to_thread(self.theme_decomposition_agent.run, payload)
                axes = result.axes
                actor_metadata, actor_metrics = sanitize_actor_metadata_payload(
                    result.actor_metadata,
                    project_id=project.project_id,
                )

                missing_by_axis = []
                for axis in axes:
                    provided_book_ids = set(axis.relevance_by_book.keys())
                    missing_book_ids = [
                        book_id for book_id in expected_book_ids
                        if book_id not in provided_book_ids
                    ]
                    if missing_book_ids:
                        missing_by_axis.append({
                            "axis_id": axis.axis_id,
                            "axis_name": axis.name,
                            "missing_book_ids": missing_book_ids,
                        })

                if not missing_by_axis and actor_metadata.actors:
                    break

                validation_issues = []
                if missing_by_axis:
                    validation_issues.append("missing_relevance_by_book")
                if not actor_metadata.actors:
                    validation_issues.append("missing_valid_actor_metadata")
                self.run_logger.log(
                    "theme_decomposition_semantic_validation_failed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    axis_count=len(axes),
                    missing_axis_count=len(missing_by_axis),
                    missing_by_axis=missing_by_axis,
                    actor_count=len(actor_metadata.actors),
                    actor_metrics=actor_metrics,
                    validation_issues=validation_issues,
                )
                if attempt < max_attempts:
                    backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                    time.sleep(backoff)
                    continue
                if missing_by_axis:
                    raise RuntimeError(
                        "Theme decomposition omitted input books in relevance_by_book for one or "
                        f"more axes after {max_attempts} attempts."
                    )
                raise RuntimeError(
                    "Theme decomposition did not produce any valid actor_metadata actors "
                    f"after {max_attempts} attempts."
                )

            valid_axes = [
                a for a in axes
                if sum(1 for s in a.relevance_by_book.values() if s >= 0.3) >= 2
            ]

            if len(valid_axes) < project.config.min_axes:
                valid_axis_ids = {axis.axis_id for axis in valid_axes}
                fallback_axes = [axis for axis in axes if axis.axis_id not in valid_axis_ids]
                padded_axes = valid_axes + fallback_axes
                logger.warning(
                    "Only %d valid axes (min %d). Padding with %d fallback axes from %d total.",
                    len(valid_axes),
                    project.config.min_axes,
                    max(0, project.config.min_axes - len(valid_axes)),
                    len(axes),
                )
                valid_axes = padded_axes

            valid_axes = valid_axes[:project.config.max_axes]
            valid_axes, axis_actor_metrics = clean_axis_actor_ids(valid_axes, actor_metadata)
            actor_metrics.update(axis_actor_metrics)
            _save_json(project_dir / "thematic_axes.json",
                        {"axes": [a.model_dump(mode="json") for a in valid_axes]})
            _save_json(project_dir / "actor_metadata.json", actor_metadata)

            ctx["output_summary"] = {
                "book_summary_count": len(book_summaries),
                "total_axes_generated": len(axes),
                "valid_axes": len(valid_axes),
                "axis_names": [a.name for a in valid_axes],
                "actor_count": len(actor_metadata.actors),
                "relationship_count": len(actor_metadata.relationships),
            }
            return valid_axes, actor_metadata, actor_metrics

    async def _extract_passages(
        self, project: ThematicProject, axes: list[ThematicAxis], project_dir: Path,
    ) -> ThematicCorpus:
        async with _stage_log(
            self.run_logger, "passage_extraction", project_dir,
            axis_count=len(axes), book_count=len(project.books),
        ) as ctx:
            if not self.vector_store.enabled:
                self.run_logger.log(
                    "retrieval_disabled",
                    reason="DATABASE_URL not set",
                    message="Vector retrieval is disabled; passage extraction will yield zero candidates.",
                )
            book_ids = [b.book_id for b in project.books]
            book_by_id = {book.book_id: book for book in project.books}
            retrieval_depth_by_book = {
                book_id: _compute_passage_retrieval_budget(
                    chunk_count=book_by_id.get(book_id).chunk_count if book_by_id.get(book_id) else 0,
                    percentage=project.config.passage_retrieval_percentage,
                    min_per_book=project.config.passage_retrieval_min_per_book,
                    max_per_book=project.config.passage_retrieval_max_per_book,
                )
                for book_id in book_ids
            }
            book_size_share_by_book = _resolve_book_size_shares(
                project.books,
            )
            axis_candidate_budget_target = max(1, project.config.axis_candidate_target_total)
            pre_axis_weight_by_axis = _normalize_axis_importance_weights(
                axes=axes,
                power=project.config.pre_axis_relevance_power,
                min_weight=0.5,
                max_weight=1.0,
            )
            pre_axis_budget_by_axis = _build_axis_budget_by_importance(
                axes=axes,
                total_budget=project.config.pre_axis_total_budget,
                floor_per_axis=project.config.pre_axis_floor,
                importance_power=project.config.pre_axis_relevance_power,
            )
            chapter_word_count_by_book = {
                book.book_id: {
                    chapter.chapter_id: max(0, chapter.word_count)
                    for chapter in book.chapters
                }
                for book in project.books
            }
            max_log_per_book = max(
                100,
                max((info["per_book_budget"] for info in retrieval_depth_by_book.values()), default=0),
            )
            all_passages_by_axis: dict[str, list[ExtractedPassage]] = {}
            all_cross_pairs: list[PassagePair] = []
            candidate_counts_by_axis: dict[str, int] = {}
            axis_policy_by_axis: dict[str, dict[str, Any]] = {}

            axis_priority_order = sorted(
                axes,
                key=lambda axis: (
                    -axis.theme_importance_score,
                    axis.axis_id,
                ),
            )
            axis_priority_rank_by_id = {
                axis.axis_id: rank
                for rank, axis in enumerate(axis_priority_order, start=1)
            }
            cross_axis_reuse_counts: dict[tuple[str, str], int] = {}
            cross_axis_distinct_floor_ratio = 0.8
            cross_axis_distinct_boost = 2.0
            AxisExtractionResult = tuple[
                str,
                list[ExtractedPassage],
                list[PassagePair],
                int,
                dict[str, Any],
                dict[str, Any],
            ]
            DeferredAxisWork = tuple[
                str,
                Callable[[], AxisExtractionResult] | None,
                AxisExtractionResult | None,
            ]

            def _score_axis_candidates(
                *,
                axis: ThematicAxis,
                payload: dict[str, Any],
                candidates: list[dict[str, Any]],
                prompt_candidates: list[dict[str, Any]],
                candidate_full_text_by_id: dict[str, str],
                retrieval_log: dict[str, Any],
                admitted_by_book: dict[str, int],
                axis_candidate_budget_effective: int,
            ) -> AxisExtractionResult:
                candidate_by_id = {c["passage_id"]: c for c in candidates}
                candidate_ids = list(candidate_by_id.keys())
                candidate_count = len(candidate_ids)
                max_attempts = self.passage_extraction_agent.max_retry_attempts
                result = None
                for attempt in range(1, max_attempts + 1):
                    result = self.passage_extraction_agent.run(payload)
                    result_ids = [p.passage_id for p in result.passages]
                    id_counts: dict[str, int] = {}
                    for pid in result_ids:
                        id_counts[pid] = id_counts.get(pid, 0) + 1
                    duplicate_ids = [pid for pid, count in id_counts.items() if count > 1]
                    extra_ids = [pid for pid in id_counts if pid not in candidate_by_id]
                    unique_ids = [pid for pid in candidate_ids if pid in id_counts]
                    unique_count = len(unique_ids)
                    coverage_ratio = unique_count / max(1, candidate_count)
                    missing_ids = [pid for pid in candidate_ids if pid not in id_counts]
                    if duplicate_ids or extra_ids or missing_ids:
                        self.run_logger.log(
                            "passage_extraction_id_mismatch",
                            axis_id=axis.axis_id,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            missing_ids=missing_ids,
                            extra_ids=extra_ids,
                            duplicate_ids=duplicate_ids,
                        )
                    if coverage_ratio >= 0.60:
                        break
                    self.run_logger.log(
                        "passage_extraction_low_coverage",
                        axis_id=axis.axis_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        unique_count=unique_count,
                        candidate_count=candidate_count,
                        coverage_ratio=round(coverage_ratio, 3),
                    )
                    if attempt < max_attempts:
                        backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                        time.sleep(backoff)
                        continue
                    raise RuntimeError(
                        "Passage extraction returned fewer than 60% of candidate passages for axis "
                        f"{axis.axis_id} after {max_attempts} attempts."
                    )
                assert result is not None

                scores_by_id = {}
                for score in result.passages:
                    if score.passage_id in candidate_by_id and score.passage_id not in scores_by_id:
                        scores_by_id[score.passage_id] = score
                trimmed_text_by_id = {
                    candidate["passage_id"]: candidate["text"]
                    for candidate in prompt_candidates
                }
                rehydrated_passages = []
                for candidate in candidates:
                    score = scores_by_id.get(candidate["passage_id"])
                    if score is None:
                        continue
                    trimmed_text = trimmed_text_by_id.get(candidate["passage_id"], candidate["text"])
                    rehydrated_passages.append(
                        ExtractedPassage(
                            passage_id=score.passage_id,
                            book_id=candidate["book_id"],
                            chunk_ids=candidate["chunk_ids"],
                            text=trimmed_text,
                            trimmed_text=trimmed_text,
                            full_text=candidate_full_text_by_id.get(candidate["passage_id"], ""),
                            chapter_ref=candidate.get("chapter_ref", ""),
                            axis_id=candidate.get("axis_id", axis.axis_id),
                            secondary_axes=candidate.get("secondary_axes", []),
                            relevance_score=score.relevance_score,
                            quotability_score=score.quotability_score,
                            synthesis_tags=score.synthesis_tags,
                        )
                    )

                relationship_filtered_pairs = [
                    pair
                    for pair in result.cross_book_pairs
                    if pair.axis_id == axis.axis_id
                    and pair.relationship
                    not in {SynthesisTag.AGREES_WITH, SynthesisTag.EXTENDS}
                ]
                passage_book_by_id = {p.passage_id: p.book_id for p in rehydrated_passages}
                validated_pairs: list[PassagePair] = []
                dropped_missing_id_pairs: list[dict[str, str]] = []
                dropped_same_book_pairs: list[dict[str, str]] = []
                for pair in relationship_filtered_pairs:
                    book_a = passage_book_by_id.get(pair.passage_a_id)
                    book_b = passage_book_by_id.get(pair.passage_b_id)
                    if book_a is None or book_b is None:
                        dropped_missing_id_pairs.append(
                            {
                                "passage_a_id": pair.passage_a_id,
                                "passage_b_id": pair.passage_b_id,
                            }
                        )
                        continue
                    if book_a == book_b:
                        dropped_same_book_pairs.append(
                            {
                                "passage_a_id": pair.passage_a_id,
                                "passage_b_id": pair.passage_b_id,
                                "book_id": book_a,
                            }
                        )
                        continue
                    validated_pairs.append(pair)

                if dropped_missing_id_pairs or dropped_same_book_pairs:
                    self.run_logger.log(
                        "passage_extraction_invalid_cross_book_pairs",
                        axis_id=axis.axis_id,
                        candidate_pair_count=len(relationship_filtered_pairs),
                        dropped_missing_id_count=len(dropped_missing_id_pairs),
                        dropped_same_book_count=len(dropped_same_book_pairs),
                        dropped_missing_id_pairs=dropped_missing_id_pairs,
                        dropped_same_book_pairs=dropped_same_book_pairs,
                    )

                validated_pairs.sort(key=lambda p: p.strength, reverse=True)
                retained_pairs = validated_pairs[:5]
                cross_pair_validation = {
                    "candidate_pair_count": len(relationship_filtered_pairs),
                    "valid_pair_count": len(validated_pairs),
                    "retained_pair_count": len(retained_pairs),
                    "dropped_missing_id_count": len(dropped_missing_id_pairs),
                    "dropped_same_book_count": len(dropped_same_book_pairs),
                }
                retained_passages = sorted(
                    rehydrated_passages,
                    key=lambda p: (
                        -p.relevance_score,
                        -p.quotability_score,
                        p.passage_id,
                    ),
                )
                axis_policy = {
                    "passage_selection_policy": {
                        "strategy": "retain_all_scored_passages",
                        "sort_order": "relevance_desc_then_quotability_desc",
                    },
                    "allocation_policy": retrieval_log["allocation_policy"],
                    "retrieval_relevance_power": retrieval_log["retrieval_relevance_power"],
                    "axis_candidate_budget": axis_candidate_budget_effective,
                    "pre_axis_budget": retrieval_log["pre_axis_budget"],
                    "axis_importance_score": axis.theme_importance_score,
                    "axis_importance_weight": retrieval_log["axis_importance_weight"],
                    "per_book_budget": retrieval_log["per_book_budget"],
                    "admitted_by_book": admitted_by_book,
                    "retained_count": len(retained_passages),
                }
                return axis.axis_id, retained_passages, retained_pairs, candidate_count, cross_pair_validation, axis_policy

            def _process_axis(
                axis: ThematicAxis,
                *,
                axis_priority_rank: int,
            ) -> DeferredAxisWork:
                hits_by_book = self.retrieval.retrieve_for_axis(
                    axis=axis, project_id=project.project_id,
                    book_ids=book_ids,
                    k_per_book=max_log_per_book,
                )

                axis_candidate_budget_effective = min(
                    max(1, pre_axis_budget_by_axis.get(axis.axis_id, axis_candidate_budget_target)),
                    sum(len(hits_by_book.get(bid, [])) for bid in book_ids),
                )
                soft_threshold = project.config.retrieval_soft_threshold
                chapter_penalty_weight = project.config.chapter_penalty_weight
                floor_per_book = project.config.admission_floor_per_book
                cross_axis_reuse_penalty = project.config.pre_axis_cross_axis_reuse_penalty
                cross_axis_distinct_floor_target = 0
                if cross_axis_reuse_penalty > 0.0:
                    cross_axis_distinct_floor_target = min(
                        axis_candidate_budget_effective,
                        int(math.ceil(project.config.pre_axis_floor * cross_axis_distinct_floor_ratio)),
                    )
                adaptive_relevance_powers = {
                    "default": project.config.retrieval_relevance_power,
                    "risky": max(0.0, project.config.retrieval_relevance_power - 0.2),
                }
                retrieval_log: dict[str, Any] = {
                    "axis_id": axis.axis_id,
                    "axis_name": axis.name,
                    "axis_description": axis.description,
                    "max_log_per_book": max_log_per_book,
                    "axis_candidate_budget_target": axis_candidate_budget_target,
                    "axis_candidate_budget_effective": axis_candidate_budget_effective,
                    "axis_candidate_budget": axis_candidate_budget_effective,
                    "pre_axis_budget": pre_axis_budget_by_axis.get(axis.axis_id, axis_candidate_budget_target),
                    "axis_importance_score": axis.theme_importance_score,
                    "axis_importance_weight": pre_axis_weight_by_axis.get(axis.axis_id, 1.0),
                    "pre_axis_total_budget": project.config.pre_axis_total_budget,
                    "budget_strategy": "axis_importance_weighted_pre_budget",
                    "passage_retrieval_percentage": project.config.passage_retrieval_percentage,
                    "passage_retrieval_min_per_book": project.config.passage_retrieval_min_per_book,
                    "passage_retrieval_max_per_book": project.config.passage_retrieval_max_per_book,
                    "allocation_policy": "floor_2_adaptive_relevance_power_spillover",
                    "allocation_floor": floor_per_book,
                    "retrieval_relevance_power": adaptive_relevance_powers["default"],
                    "soft_threshold": soft_threshold,
                    "chapter_penalty_weight": chapter_penalty_weight,
                    "axis_priority_rank": axis_priority_rank,
                    "axis_priority_basis": "theme_importance_score_desc",
                    "pre_axis_cross_axis_reuse_penalty": cross_axis_reuse_penalty,
                    "cross_axis_distinct_floor_ratio": cross_axis_distinct_floor_ratio,
                    "cross_axis_distinct_floor_target": cross_axis_distinct_floor_target,
                    "book_size_share_by_book": book_size_share_by_book,
                    "books": [],
                }
                relevance_by_book = _resolve_axis_relevance(axis, book_ids)
                retrieval_log["relevance_by_book"] = relevance_by_book

                rows_by_book: dict[str, list[dict[str, Any]]] = {}
                for bid, hits in hits_by_book.items():
                    book = next((b for b in project.books if b.book_id == bid), None)
                    book_entry = {
                        "book_id": bid,
                        "title": book.title if book else "Unknown",
                        "author": book.author if book else "Unknown",
                        "chunk_count": retrieval_depth_by_book.get(bid, {}).get("chunk_count", 0),
                        "percentage_budget": retrieval_depth_by_book.get(bid, {}).get("percentage_budget", 0),
                        "retrieval_depth_budget": retrieval_depth_by_book.get(bid, {}).get("per_book_budget", 0),
                        "admission_quota": 0,
                        "candidates": [],
                    }
                    for rank, hit in enumerate(hits, start=1):
                        row = {
                            "book_id": bid,
                            "rank": rank,
                            "priority": 0.0,
                            "retrieval_confidence": 0.0,
                            "selection_phase": None,
                            "chapter_penalty": 0.0,
                            "selection_score": 0.0,
                            "hit": hit,
                            "title": book.title if book else "Unknown",
                            "author": book.author if book else "Unknown",
                        }
                        rows_by_book.setdefault(bid, []).append(row)
                    retrieval_log["books"].append(book_entry)

                retrieval_signal_by_book: dict[str, float] = {}
                for bid in book_ids:
                    rows = rows_by_book.get(bid, [])
                    if not rows:
                        retrieval_signal_by_book[bid] = 0.0
                        continue
                    scores = [float(row["hit"].score) for row in rows]
                    min_score = min(scores)
                    max_score = max(scores)
                    denom = max_score - min_score
                    confidences: list[float] = []
                    for row in rows:
                        raw_score = float(row["hit"].score)
                        if denom <= 0:
                            confidence = 1.0
                        else:
                            confidence = _clamp((max_score - raw_score) / denom, 0.0, 1.0)
                        row["retrieval_confidence"] = confidence
                        row["priority"] = confidence
                        confidences.append(confidence)
                    confidences.sort(reverse=True)
                    top_n = min(10, len(confidences))
                    retrieval_signal_by_book[bid] = (
                        sum(confidences[:top_n]) / max(1, top_n)
                    )

                retrieval_log["retrieval_signal_by_book"] = retrieval_signal_by_book
                retrieval_log["blended_score_by_book"] = relevance_by_book

                provisional_quota_by_book = _compute_weighted_admitted_budgets(
                    book_ids=book_ids,
                    axis_total_budget=axis_candidate_budget_effective,
                    relevance_by_book=relevance_by_book,
                    floor_per_book=floor_per_book,
                    relevance_power=adaptive_relevance_powers["default"],
                )
                provisional_total = max(1, sum(provisional_quota_by_book.values()))
                provisional_top2_share = (
                    sum(sorted(provisional_quota_by_book.values(), reverse=True)[:2]) / provisional_total
                )
                above_threshold_available_by_book = {
                    bid: sum(
                        1
                        for row in rows_by_book.get(bid, [])
                        if row["retrieval_confidence"] >= soft_threshold
                    )
                    for bid in book_ids
                }
                provisional_predicted_backfill = sum(
                    max(
                        0,
                        provisional_quota_by_book.get(bid, 0)
                        - above_threshold_available_by_book.get(bid, 0),
                    )
                    for bid in book_ids
                )
                provisional_predicted_backfill_share = (
                    provisional_predicted_backfill / max(1, axis_candidate_budget_effective)
                )
                concentration_trigger = provisional_top2_share >= 0.60
                backfill_trigger = provisional_predicted_backfill_share >= 0.20
                relevance_power = (
                    adaptive_relevance_powers["risky"]
                    if concentration_trigger or backfill_trigger
                    else adaptive_relevance_powers["default"]
                )

                admitted_quota_by_book = _compute_weighted_admitted_budgets(
                    book_ids=book_ids,
                    axis_total_budget=axis_candidate_budget_effective,
                    relevance_by_book=relevance_by_book,
                    floor_per_book=floor_per_book,
                    relevance_power=relevance_power,
                )
                retrieval_log["admission_quota_by_book"] = admitted_quota_by_book
                retrieval_log["adaptive_policy"] = {
                    "provisional_relevance_power": adaptive_relevance_powers["default"],
                    "provisional_top2_share": round(provisional_top2_share, 6),
                    "provisional_predicted_backfill": provisional_predicted_backfill,
                    "provisional_predicted_backfill_share": round(provisional_predicted_backfill_share, 6),
                    "concentration_trigger": concentration_trigger,
                    "backfill_trigger": backfill_trigger,
                    "selected_relevance_power": relevance_power,
                    "risky_relevance_power": adaptive_relevance_powers["risky"],
                }
                relevance_factor_by_book = {
                    bid: max(0.0, float(relevance_by_book.get(bid, 0.0))) ** relevance_power
                    if relevance_power > 0
                    else 1.0
                    for bid in book_ids
                }
                raw_weight_by_book = {
                    bid: relevance_factor_by_book[bid]
                    for bid in book_ids
                }
                quota_total = max(1, sum(admitted_quota_by_book.values()))
                retrieval_log["relevance_factor_by_book"] = relevance_factor_by_book
                retrieval_log["raw_weight_by_book"] = raw_weight_by_book
                retrieval_log["quota_share_by_book"] = {
                    bid: round(admitted_quota_by_book.get(bid, 0) / quota_total, 6)
                    for bid in book_ids
                }
                retrieval_log["retrieval_relevance_power"] = relevance_power
                retrieval_log["allocation_policy"] = (
                    f"floor_{floor_per_book}_adaptive_relevance_pow_{relevance_power}_spillover"
                )

                admitted_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                selected_rows: list[dict[str, Any]] = []
                selected_above_threshold_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                selected_spillover_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                selected_backfill_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                book_entry_by_id = {entry["book_id"]: entry for entry in retrieval_log["books"]}
                chapter_targets_by_book: dict[str, dict[str, float]] = {}
                selected_by_chapter_by_book: dict[str, dict[str, int]] = {}
                high_pool_by_book: dict[str, list[dict[str, Any]]] = {}
                backfill_pool_by_book: dict[str, list[dict[str, Any]]] = {}

                for bid in book_ids:
                    rows = list(rows_by_book.get(bid, []))
                    quota = admitted_quota_by_book.get(bid, 0)
                    book_entry = book_entry_by_id.get(bid)
                    if book_entry is None:
                        continue
                    book_entry["admission_quota"] = quota
                    book_entry["eligible_total_count"] = len(rows)
                    chapter_targets: dict[str, float] = {}
                    if rows and quota > 0:
                        chapter_ids = sorted({
                            str(row["hit"].chapter_id or "")
                            for row in rows
                        })
                        chapter_word_counts = chapter_word_count_by_book.get(bid, {})
                        total_visible_chapter_words = sum(
                            max(0, chapter_word_counts.get(chapter_id, 0))
                            for chapter_id in chapter_ids
                        )
                        if total_visible_chapter_words > 0:
                            for chapter_id in chapter_ids:
                                chapter_words = max(0, chapter_word_counts.get(chapter_id, 0))
                                chapter_targets[chapter_id] = (
                                    quota * (chapter_words / total_visible_chapter_words)
                                )
                        elif chapter_ids:
                            equal_target = quota / len(chapter_ids)
                            chapter_targets = {
                                chapter_id: equal_target for chapter_id in chapter_ids
                            }
                    chapter_targets_by_book[bid] = chapter_targets
                    selected_by_chapter_by_book[bid] = {}
                    high_pool = [
                        row for row in rows
                        if row["retrieval_confidence"] >= soft_threshold
                    ]
                    backfill_pool = [
                        row for row in rows
                        if row["retrieval_confidence"] < soft_threshold
                    ]
                    high_pool_by_book[bid] = high_pool
                    backfill_pool_by_book[bid] = backfill_pool
                    book_entry["eligible_above_threshold_count"] = len(high_pool)

                unseen_selected_count = 0
                use_cross_axis_reuse_penalty = cross_axis_reuse_penalty > 0.0

                def _chunk_reuse_key(row: dict[str, Any]) -> tuple[str, str]:
                    return (str(row["book_id"]), str(row["hit"].chunk_id))

                def _is_unseen_chunk(row: dict[str, Any]) -> bool:
                    key = _chunk_reuse_key(row)
                    return cross_axis_reuse_counts.get(key, 0) == 0

                def _score_row(
                    row: dict[str, Any],
                    *,
                    bid: str,
                    prefer_unseen: bool = False,
                    unseen_exists: bool = False,
                ) -> tuple[tuple[float, float, float, int, str], float]:
                    chapter_targets = chapter_targets_by_book.get(bid, {})
                    selected_by_chapter = selected_by_chapter_by_book.get(bid, {})
                    chapter_id = str(row["hit"].chapter_id or "")
                    chapter_target = chapter_targets.get(chapter_id, 0.0)
                    denom = max(1.0, chapter_target)
                    over_target = max(
                        0.0,
                        ((selected_by_chapter.get(chapter_id, 0) + 1) - chapter_target) / denom,
                    )
                    chapter_penalty = chapter_penalty_weight * over_target
                    base_selection_score = row["retrieval_confidence"] - chapter_penalty

                    if not use_cross_axis_reuse_penalty:
                        key = (
                            base_selection_score,
                            base_selection_score,
                            row["retrieval_confidence"],
                            -int(row["rank"]),
                            str(row["hit"].chunk_id),
                        )
                        return key, chapter_penalty

                    reuse_key = _chunk_reuse_key(row)
                    reuse_count = cross_axis_reuse_counts.get(reuse_key, 0)
                    penalty_weight = cross_axis_reuse_penalty
                    if prefer_unseen and unseen_exists:
                        penalty_weight *= cross_axis_distinct_boost
                    adjusted_score = base_selection_score - (penalty_weight * math.log1p(reuse_count))
                    if prefer_unseen and unseen_exists and reuse_count > 0:
                        adjusted_score -= 1.0
                    key = (
                        adjusted_score,
                        base_selection_score,
                        row["retrieval_confidence"],
                        -int(row["rank"]),
                        str(row["hit"].chunk_id),
                    )
                    return key, chapter_penalty

                def _pop_best_row(
                    pool: list[dict[str, Any]],
                    *,
                    bid: str,
                    prefer_unseen: bool = False,
                ) -> dict[str, Any]:
                    nonlocal unseen_selected_count
                    best_idx = 0
                    best_key: tuple[float, float, float, int, str] | None = None
                    best_penalty = 0.0
                    unseen_exists = bool(prefer_unseen and any(_is_unseen_chunk(row) for row in pool))
                    for idx, row in enumerate(pool):
                        key, chapter_penalty = _score_row(
                            row,
                            bid=bid,
                            prefer_unseen=prefer_unseen,
                            unseen_exists=unseen_exists,
                        )
                        if best_key is None or key > best_key:
                            best_idx = idx
                            best_key = key
                            best_penalty = chapter_penalty
                    best_row = pool.pop(best_idx)
                    chapter_id = str(best_row["hit"].chapter_id or "")
                    selected_by_chapter = selected_by_chapter_by_book.get(bid, {})
                    selected_by_chapter[chapter_id] = selected_by_chapter.get(chapter_id, 0) + 1
                    best_row["chapter_penalty"] = best_penalty
                    best_row["selection_score"] = (
                        best_key[0]
                        if best_key is not None
                        else best_row["retrieval_confidence"]
                    )
                    if use_cross_axis_reuse_penalty:
                        reuse_key = _chunk_reuse_key(best_row)
                        if cross_axis_reuse_counts.get(reuse_key, 0) == 0:
                            unseen_selected_count += 1
                        cross_axis_reuse_counts[reuse_key] = cross_axis_reuse_counts.get(reuse_key, 0) + 1
                    return best_row

                def _peek_best_row_key(
                    pool: list[dict[str, Any]],
                    *,
                    bid: str,
                    prefer_unseen: bool = False,
                ) -> tuple[float, float, float, int, str] | None:
                    if not pool:
                        return None
                    best_key: tuple[float, float, float, int, str] | None = None
                    unseen_exists = bool(prefer_unseen and any(_is_unseen_chunk(row) for row in pool))
                    for row in pool:
                        key, _ = _score_row(
                            row,
                            bid=bid,
                            prefer_unseen=prefer_unseen,
                            unseen_exists=unseen_exists,
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                    return best_key

                # Phase 1: per-book above-threshold selection up to quota.
                for bid in book_ids:
                    quota = admitted_quota_by_book.get(bid, 0)
                    if quota <= 0:
                        continue
                    high_pool = high_pool_by_book.get(bid, [])
                    while admitted_by_book.get(bid, 0) < quota and high_pool:
                        require_unseen = unseen_selected_count < cross_axis_distinct_floor_target
                        row = _pop_best_row(
                            high_pool,
                            bid=bid,
                            prefer_unseen=require_unseen,
                        )
                        row["selection_phase"] = "above_threshold"
                        selected_rows.append(row)
                        admitted_by_book[bid] = admitted_by_book.get(bid, 0) + 1
                        selected_above_threshold_by_book[bid] = (
                            selected_above_threshold_by_book.get(bid, 0) + 1
                        )

                # Phase 2: spillover from unused above-threshold rows across books.
                total_deficit = sum(
                    max(0, admitted_quota_by_book.get(bid, 0) - admitted_by_book.get(bid, 0))
                    for bid in book_ids
                )
                remaining_spillover = total_deficit
                while remaining_spillover > 0:
                    spillover_frontier: list[tuple[tuple[float, float, float, int, str], str]] = []
                    require_unseen = unseen_selected_count < cross_axis_distinct_floor_target
                    for bid in book_ids:
                        best_key = _peek_best_row_key(
                            high_pool_by_book.get(bid, []),
                            bid=bid,
                            prefer_unseen=require_unseen,
                        )
                        if best_key is None:
                            continue
                        spillover_frontier.append((best_key, bid))
                    if not spillover_frontier:
                        break
                    spillover_frontier.sort(
                        key=lambda item: (item[0], item[1]),
                        reverse=True,
                    )
                    took_any = False
                    for _, bid in spillover_frontier:
                        if remaining_spillover <= 0:
                            break
                        high_pool = high_pool_by_book.get(bid, [])
                        if not high_pool:
                            continue
                        require_unseen = unseen_selected_count < cross_axis_distinct_floor_target
                        row = _pop_best_row(
                            high_pool,
                            bid=bid,
                            prefer_unseen=require_unseen,
                        )
                        row["selection_phase"] = "spillover"
                        selected_rows.append(row)
                        admitted_by_book[bid] = admitted_by_book.get(bid, 0) + 1
                        selected_spillover_by_book[bid] = selected_spillover_by_book.get(bid, 0) + 1
                        remaining_spillover -= 1
                        took_any = True
                    if not took_any:
                        break

                # Phase 3: global backfill only if deficits remain after spillover.
                remaining_deficit = max(0, axis_candidate_budget_effective - len(selected_rows))
                if remaining_deficit > 0:
                    global_backfill_pool: list[tuple[str, dict[str, Any]]] = []
                    for bid in book_ids:
                        global_backfill_pool.extend((bid, row) for row in backfill_pool_by_book.get(bid, []))
                    for _ in range(remaining_deficit):
                        if not global_backfill_pool:
                            break
                        require_unseen = unseen_selected_count < cross_axis_distinct_floor_target
                        unseen_exists = bool(
                            require_unseen and any(_is_unseen_chunk(row) for _, row in global_backfill_pool)
                        )
                        best_index = 0
                        best_key: tuple[float, float, float, int, str] | None = None
                        best_penalty = 0.0
                        for idx, (bid, row) in enumerate(global_backfill_pool):
                            key, chapter_penalty = _score_row(
                                row,
                                bid=bid,
                                prefer_unseen=require_unseen,
                                unseen_exists=unseen_exists,
                            )
                            if best_key is None or key > best_key:
                                best_index = idx
                                best_key = key
                                best_penalty = chapter_penalty
                        bid, row = global_backfill_pool.pop(best_index)
                        row["selection_phase"] = "backfill"
                        row["chapter_penalty"] = best_penalty
                        row["selection_score"] = (
                            best_key[0]
                            if best_key is not None
                            else row["retrieval_confidence"]
                        )
                        selected_rows.append(row)
                        if use_cross_axis_reuse_penalty:
                            reuse_key = _chunk_reuse_key(row)
                            if cross_axis_reuse_counts.get(reuse_key, 0) == 0:
                                unseen_selected_count += 1
                            cross_axis_reuse_counts[reuse_key] = cross_axis_reuse_counts.get(reuse_key, 0) + 1
                        admitted_by_book[bid] = admitted_by_book.get(bid, 0) + 1
                        selected_backfill_by_book[bid] = selected_backfill_by_book.get(bid, 0) + 1

                retrieval_log["cross_axis_distinct_selected_count"] = unseen_selected_count

                for bid in book_ids:
                    book_entry = book_entry_by_id.get(bid)
                    if book_entry is None:
                        continue
                    quota = admitted_quota_by_book.get(bid, 0)
                    book_entry["selected_above_threshold_count"] = selected_above_threshold_by_book.get(bid, 0)
                    book_entry["selected_spillover_count"] = selected_spillover_by_book.get(bid, 0)
                    book_entry["selected_backfill_count"] = selected_backfill_by_book.get(bid, 0)
                    book_entry["underfill_count"] = max(0, quota - admitted_by_book.get(bid, 0))

                selected_row_ids = {id(row) for row in selected_rows}
                candidates: list[dict[str, Any]] = []
                for bid in book_ids:
                    book_entry = book_entry_by_id.get(bid)
                    if book_entry is None:
                        continue
                    for row in rows_by_book.get(bid, []):
                        hit = row["hit"]
                        used = id(row) in selected_row_ids
                        book_entry["candidates"].append({
                            "rank": row["rank"],
                            "used": used,
                            "global_priority": round(float(row["priority"]), 8),
                            "retrieval_confidence": round(float(row["retrieval_confidence"]), 8),
                            "meets_soft_threshold": bool(row["retrieval_confidence"] >= soft_threshold),
                            "selection_phase": row["selection_phase"] if used else None,
                            "chapter_penalty": round(float(row["chapter_penalty"]), 8) if used else 0.0,
                            "selection_score": round(float(row["selection_score"]), 8) if used else None,
                            "chunk_id": hit.chunk_id,
                            "chapter_id": hit.chapter_id,
                            "score": hit.score,
                            "text": hit.text,
                            "metadata": hit.metadata,
                        })
                        if not used:
                            continue
                        candidates.append({
                            "passage_id": uuid4().hex,
                            "book_id": bid,
                            "chunk_ids": [hit.chunk_id],
                            "text": hit.text,
                            "chapter_ref": hit.metadata.get("chapter_id", ""),
                            "axis_id": axis.axis_id,
                            "author": row["author"],
                            "title": row["title"],
                        })

                retrieval_log["admitted_by_book"] = admitted_by_book
                retrieval_log["per_book_budget"] = {
                    bid: admitted_quota_by_book.get(bid, 0) for bid in book_ids
                }
                retrieval_log["retrieval_depth_by_book"] = {
                    bid: info["per_book_budget"] for bid, info in retrieval_depth_by_book.items()
                }

                _save_json(
                    project_dir
                    / "stage_artifacts"
                    / "passage_extraction"
                    / f"retrieval_candidates_{axis.axis_id}.json",
                    retrieval_log,
                )

                candidate_count = len(candidates)
                if not candidates:
                    empty_policy = {
                        "passage_selection_policy": {
                            "strategy": "retain_all_scored_passages",
                            "sort_order": "relevance_desc_then_quotability_desc",
                        },
                        "allocation_policy": retrieval_log["allocation_policy"],
                        "retrieval_relevance_power": retrieval_log["retrieval_relevance_power"],
                        "axis_candidate_budget": axis_candidate_budget_effective,
                        "pre_axis_budget": retrieval_log["pre_axis_budget"],
                        "axis_importance_score": axis.theme_importance_score,
                        "axis_importance_weight": retrieval_log["axis_importance_weight"],
                        "per_book_budget": retrieval_log["per_book_budget"],
                        "admitted_by_book": admitted_by_book,
                        "retained_count": 0,
                    }
                    empty_result: AxisExtractionResult = (
                        axis.axis_id,
                        [],
                        [],
                        candidate_count,
                        {
                            "candidate_pair_count": 0,
                            "valid_pair_count": 0,
                            "retained_pair_count": 0,
                            "dropped_missing_id_count": 0,
                            "dropped_same_book_count": 0,
                        },
                        empty_policy,
                    )
                    return axis.axis_id, None, empty_result

                candidate_full_text_by_id = {
                    candidate["passage_id"]: candidate["text"]
                    for candidate in candidates
                }
                prompt_candidates = [
                    {
                        "passage_id": candidate["passage_id"],
                        "book_id": candidate["book_id"],
                        "text": candidate["text"],
                    }
                    for candidate in candidates
                ]
                _trim_candidate_texts_by_bm25(
                    axis,
                    prompt_candidates,
                    keep_fraction=0.25,
                )

                payload = self.passage_extraction_agent.build_payload(
                    axis_id=axis.axis_id, axis_name=axis.name,
                    axis_description=axis.description,
                    candidate_passages=prompt_candidates,
                )

                def _deferred_score() -> AxisExtractionResult:
                    return _score_axis_candidates(
                        axis=axis,
                        payload=payload,
                        candidates=candidates,
                        prompt_candidates=prompt_candidates,
                        candidate_full_text_by_id=candidate_full_text_by_id,
                        retrieval_log=retrieval_log,
                        admitted_by_book=admitted_by_book,
                        axis_candidate_budget_effective=axis_candidate_budget_effective,
                    )

                return axis.axis_id, _deferred_score, None

            prepared_work = [
                await asyncio.to_thread(
                    _process_axis,
                    axis,
                    axis_priority_rank=axis_priority_rank_by_id[axis.axis_id],
                )
                for axis in axis_priority_order
            ]

            results_by_axis: dict[str, AxisExtractionResult] = {}
            deferred_work: list[tuple[str, Callable[[], AxisExtractionResult]]] = []
            for axis_id, deferred_fn, ready_result in prepared_work:
                if ready_result is not None:
                    results_by_axis[axis_id] = ready_result
                    continue
                if deferred_fn is not None:
                    deferred_work.append((axis_id, deferred_fn))

            if deferred_work:
                extraction_sem = asyncio.Semaphore(max(1, project.config.passage_extraction_concurrency))

                async def _run_deferred(
                    axis_id: str,
                    worker: Callable[[], AxisExtractionResult],
                ) -> tuple[str, AxisExtractionResult]:
                    async with extraction_sem:
                        result = await asyncio.to_thread(worker)
                        return axis_id, result

                deferred_results = await asyncio.gather(
                    *[
                        _run_deferred(axis_id, worker)
                        for axis_id, worker in deferred_work
                    ]
                )
                for axis_id, result in deferred_results:
                    results_by_axis[axis_id] = result

            results = [
                results_by_axis[axis.axis_id]
                for axis in axis_priority_order
                if axis.axis_id in results_by_axis
            ]

            cross_pair_validation_by_axis: dict[str, dict[str, Any]] = {}
            for axis_id, top_passages, cross_pairs, candidate_count, cross_pair_validation, axis_policy in results:
                candidate_counts_by_axis[axis_id] = candidate_count
                all_passages_by_axis[axis_id] = top_passages
                all_cross_pairs.extend(cross_pairs)
                cross_pair_validation_by_axis[axis_id] = cross_pair_validation
                axis_policy_by_axis[axis_id] = axis_policy

            # ---- Retrieval metrics ----
            retrieval_metrics: dict[str, Any] = {"per_axis": {}, "per_book": {}, "summary": {}}

            for axis in axes:
                axis_passages = all_passages_by_axis.get(axis.axis_id, [])
                relevance_scores = [p.relevance_score for p in axis_passages]
                quotability_scores = [p.quotability_score for p in axis_passages]
                books_represented = list(set(p.book_id for p in axis_passages))
                full_text_count = sum(1 for p in axis_passages if p.full_text.strip())
                trimmed_text_count = sum(1 for p in axis_passages if p.trimmed_text.strip())
                full_trim_ratio = round(
                    full_text_count / max(1, len(axis_passages)),
                    3,
                )

                retrieval_metrics["per_axis"][axis.axis_id] = {
                    "axis_name": axis.name,
                    "candidate_count": candidate_counts_by_axis.get(axis.axis_id, 0),
                    "retained_count": len(axis_passages),
                    "rehydrated_count": len(axis_passages),
                    "full_text_count": full_text_count,
                    "trimmed_text_count": trimmed_text_count,
                    "full_text_coverage_ratio": full_trim_ratio,
                    "selection_policy": axis_policy_by_axis.get(axis.axis_id, {}),
                    "avg_relevance_score": round(sum(relevance_scores) / max(1, len(relevance_scores)), 3),
                    "avg_quotability_score": round(sum(quotability_scores) / max(1, len(quotability_scores)), 3),
                    "relevance_distribution": {
                        "above_0.8": sum(1 for s in relevance_scores if s >= 0.8),
                        "0.5_to_0.8": sum(1 for s in relevance_scores if 0.5 <= s < 0.8),
                        "below_0.5": sum(1 for s in relevance_scores if s < 0.5),
                    },
                    "books_represented": books_represented,
                    "cross_pair_validation": cross_pair_validation_by_axis.get(
                        axis.axis_id,
                        {
                            "candidate_pair_count": 0,
                            "valid_pair_count": 0,
                            "retained_pair_count": 0,
                            "dropped_missing_id_count": 0,
                            "dropped_same_book_count": 0,
                        },
                    ),
                }

            for book in project.books:
                book_passages = [
                    p for passages in all_passages_by_axis.values()
                    for p in passages if p.book_id == book.book_id
                ]
                axis_quota_shares = [
                    float(policy.get("quota_share_by_book", {}).get(book.book_id, 0.0))
                    for policy in axis_policy_by_axis.values()
                ]
                avg_quota_share = (
                    sum(axis_quota_shares) / len(axis_quota_shares)
                    if axis_quota_shares
                    else 0.0
                )
                size_share = float(book_size_share_by_book.get(book.book_id, 0.0))
                retrieval_metrics["per_book"][book.book_id] = {
                    "title": book.title,
                    "total_passages": len(book_passages),
                    "axes_with_passages": sum(
                        1 for passages in all_passages_by_axis.values()
                        if any(p.book_id == book.book_id for p in passages)
                    ),
                    "avg_relevance": round(
                        sum(p.relevance_score for p in book_passages) / max(1, len(book_passages)), 3
                    ),
                    "size_share": round(size_share, 4),
                    "avg_axis_quota_share": round(avg_quota_share, 4),
                    "quota_minus_size_share": round(avg_quota_share - size_share, 4),
                }

            cross_pair_counts: dict[str, int] = {}
            for pair in all_cross_pairs:
                key = pair.relationship.value
                cross_pair_counts[key] = cross_pair_counts.get(key, 0) + 1

            total_passages = sum(len(p) for p in all_passages_by_axis.values())
            retrieval_metrics["summary"] = {
                "total_axes": len(axes),
                "total_passages": total_passages,
                "total_cross_book_pairs": len(all_cross_pairs),
                "cross_book_pair_counts": cross_pair_counts,
            }

            _save_json(project_dir / "retrieval_metrics.json", retrieval_metrics)
            self.run_logger.log("retrieval_metrics", **retrieval_metrics["summary"])

            # ---- Build corpus ----
            book_coverage = {}
            for book in project.books:
                total = sum(
                    len([p for p in passages if p.book_id == book.book_id])
                    for passages in all_passages_by_axis.values()
                )
                axes_covered = sum(
                    1 for passages in all_passages_by_axis.values()
                    if any(p.book_id == book.book_id for p in passages)
                )
                book_coverage[book.book_id] = CoverageStats(
                    total_passages=total, axes_covered=axes_covered,
                    coverage_ratio=axes_covered / max(1, len(axes)),
                )

            corpus = ThematicCorpus(
                project_id=project.project_id, axes=axes,
                passages_by_axis=all_passages_by_axis,
                cross_book_pairs=all_cross_pairs,
                book_coverage=book_coverage,
                total_passages=total_passages,
            )

            _save_json(project_dir / "thematic_corpus.json", corpus)

            ctx["output_summary"] = retrieval_metrics["summary"]
            return corpus

    async def _map_synthesis(
        self,
        project: ThematicProject,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[SynthesisMap, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger, "synthesis_mapping", project_dir,
            axis_count=len(corpus.axes), total_passages=corpus.total_passages,
        ) as ctx:
            selected_axis_count = _compute_stage_axis_target_count(
                axis_total=len(corpus.axes),
                percentage=project.config.synthesis_axis_pct,
                minimum=project.config.synthesis_axis_min,
                maximum=project.config.synthesis_axis_max,
            )
            ranked_axes = sorted(
                corpus.axes,
                key=lambda axis: (-axis.theme_importance_score, axis.axis_id),
            )
            selected_axes = list(ranked_axes[:selected_axis_count])
            selected_axis_by_id = {axis.axis_id: axis for axis in selected_axes}
            selected_axis_ids = {axis.axis_id for axis in selected_axes}
            synthesis_trim_tiers = {
                "top_10_passages": 0,
                "next_20_passages": 0,
                "next_70_passages": 0,
            }
            synthesis_total_cap = max(1, project.config.synthesis_total_passage_cap)
            cross_pair_ids = {
                passage_id
                for pair in corpus.cross_book_pairs
                if pair.axis_id in selected_axis_ids
                for passage_id in (pair.passage_a_id, pair.passage_b_id)
            }
            synthesis_passages_by_axis, cap_report = _allocate_synthesis_passages_by_axis(
                axes=selected_axes,
                passages_by_axis={
                    axis.axis_id: corpus.passages_by_axis.get(axis.axis_id, [])
                    for axis in selected_axes
                },
                total_cap=synthesis_total_cap,
                importance_power=project.config.pre_axis_relevance_power,
                cross_pair_ids=cross_pair_ids,
            )

            axes_summary = [
                {
                    "axis_id": a.axis_id,
                    "name": a.name,
                    "description": a.description,
                    "theme_importance_score": a.theme_importance_score,
                }
                for a in selected_axes
            ]
            passages_summary: dict[str, list[dict[str, Any]]] = {}
            for axis_id, passages in synthesis_passages_by_axis.items():
                axis = selected_axis_by_id.get(axis_id)
                prompt_passages = [
                    {
                        "passage_id": passage.passage_id,
                        "book_id": passage.book_id,
                        "text": passage.text,
                    }
                    for passage in passages
                ]
                if axis is not None:
                    synthesis_keep_fraction_by_passage_id, passage_tier_counts = (
                        _resolve_synthesis_bm25_keep_fraction_by_passage(passages)
                    )
                    for key, value in passage_tier_counts.items():
                        synthesis_trim_tiers[key] = synthesis_trim_tiers.get(key, 0) + int(value)
                    _trim_candidate_texts_by_bm25(
                        axis,
                        prompt_passages,
                        keep_fraction=0.25,
                        keep_fraction_by_passage_id=synthesis_keep_fraction_by_passage_id,
                    )
                trimmed_text_by_id = {
                    item["passage_id"]: item["text"]
                    for item in prompt_passages
                }
                book_groups: dict[str, list[dict[str, Any]]] = {}
                for passage in passages:
                    book_groups.setdefault(passage.book_id, []).append(
                        {
                            "passage_id": passage.passage_id,
                            "text": trimmed_text_by_id.get(passage.passage_id, passage.text),
                        }
                    )
                passages_summary[axis_id] = [
                    {"book_id": book_id, "passages": grouped_passages}
                    for book_id, grouped_passages in book_groups.items()
                ]
            cross_pairs = [
                {
                    "passage_a_id": pp.passage_a_id,
                    "passage_b_id": pp.passage_b_id,
                    "relationship": pp.relationship.value,
                    "strength": pp.strength,
                }
                for pp in corpus.cross_book_pairs
                if pp.axis_id in selected_axis_ids
            ]
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]

            primitives_payload = self.synthesis_primitives_agent.build_payload(
                project_id=project.project_id, axes_summary=axes_summary,
                passages_by_axis=passages_summary, cross_book_pairs=cross_pairs,
                book_metadata=book_metadata,
                actor_metadata=compact_actor_registry(actor_metadata),
            )
            primitives = await asyncio.to_thread(self.synthesis_primitives_agent.run, primitives_payload)
            primitives, primitive_actor_metrics = clean_synthesis_primitive_actor_links(
                primitives,
                actor_metadata,
            )
            _save_json(project_dir / "synthesis_primitives.json", primitives)

            consolidation_payload = self.synthesis_consolidation_agent.build_payload(
                project_id=project.project_id,
                primitives=primitives.model_dump(mode="json"),
                axes_summary=axes_summary,
                book_metadata=book_metadata,
                series_size_hint=project.requested_episode_count,
                actor_metadata=compact_actor_metadata(actor_metadata),
            )
            consolidation = await asyncio.to_thread(
                self.synthesis_consolidation_agent.run,
                consolidation_payload,
            )
            consolidation, consolidation_actor_metrics = self._clean_consolidation_actor_links(
                consolidation,
                actor_metadata,
            )
            synthesis_map = _reconstruct_synthesis_map(
                project_id=project.project_id,
                primitives=primitives,
                consolidation=consolidation,
            )
            _save_json(project_dir / "synthesis_map.json", synthesis_map)

            ctx["output_summary"] = {
                "selected_axes": len(selected_axes),
                "selected_passages": sum(len(items) for items in synthesis_passages_by_axis.values()),
                "synthesis_cap": synthesis_total_cap,
                "synthesis_trim_tiers": synthesis_trim_tiers,
                "cap_report": cap_report,
                "clusters": len(synthesis_map.episode_candidate_clusters),
                "primitive_counts_by_family": {
                    family: len(synthesis_map.primitives_by_family.get(family, []))
                    for family in SYNTHESIS_PRIMITIVE_FAMILIES
                },
                "quality_score": synthesis_map.quality_score,
            }
            return synthesis_map, {
                "primitives": primitive_actor_metrics,
                "consolidation": consolidation_actor_metrics,
            }

    async def _choose_narrative_strategy(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[NarrativeStrategy, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger, "narrative_strategy", project_dir,
            cluster_count=len(synthesis_map.episode_candidate_clusters),
        ) as ctx:
            synthesis_summary = synthesis_map.model_dump(mode="json")
            thematic_axes = []
            for axis in corpus.axes:
                passages = corpus.passages_by_axis.get(axis.axis_id, [])
                thematic_axes.append(
                    {
                        "axis_id": axis.axis_id,
                        "name": axis.name,
                        "description": axis.description,
                        "theme_importance_score": axis.theme_importance_score,
                        "guiding_questions": axis.guiding_questions,
                        "keywords": axis.keywords,
                        "relevance_by_book": axis.relevance_by_book,
                        "passage_count": len(passages),
                        "books_with_passages": sorted({p.book_id for p in passages}),
                    }
                )
            project_metadata = {
                "theme": project.theme,
                "sub_themes": project.sub_themes,
                "book_count": len(project.books),
                "books": [{"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books],
                "target_episode_minutes": project.config.target_episode_minutes,
                "min_episode_minutes": project.config.min_episode_minutes,
            }

            payload = self.narrative_strategy_agent.build_payload(
                synthesis_map=synthesis_summary,
                thematic_axes=thematic_axes,
                project_metadata=project_metadata,
                episode_count=project.requested_episode_count,
                actor_metadata=compact_actor_metadata(actor_metadata),
            )
            strategy = await asyncio.to_thread(self.narrative_strategy_agent.run, payload)
            cleaned_episodes, strategy_actor_metrics = clean_strategy_actor_links(
                strategy.episodes,
                actor_metadata,
            )
            strategy = strategy.model_copy(update={"episodes": cleaned_episodes})
            _save_json(project_dir / "narrative_strategy.json", strategy)

            ctx["output_summary"] = {
                "strategy": strategy.strategy_type,
                "recommended_episode_count": strategy.recommended_episode_count,
                "episodes": len(strategy.episodes),
            }
            return strategy, strategy_actor_metrics

    def _resolve_episode_count_from_strategy(
        self,
        project: ThematicProject,
        strategy: NarrativeStrategy,
    ) -> ThematicProject:
        requested = project.requested_episode_count
        if requested is not None:
            self.run_logger.log(
                "episode_count_decision",
                requested_episode_count=requested,
                recommended_episode_count=strategy.recommended_episode_count,
                effective_episode_count=requested,
                source="override",
            )
            return project.model_copy(
                update={
                    "episode_count": requested,
                    "recommended_episode_count": strategy.recommended_episode_count,
                }
            )

        if strategy.recommended_episode_count is None:
            raise RuntimeError(
                "Narrative strategy did not return recommended_episode_count and no --episodes override was provided."
            )

        self.run_logger.log(
            "episode_count_decision",
            requested_episode_count=None,
            recommended_episode_count=strategy.recommended_episode_count,
            effective_episode_count=strategy.recommended_episode_count,
            source="narrative_strategy",
        )
        return project.model_copy(
            update={
                "episode_count": strategy.recommended_episode_count,
                "recommended_episode_count": strategy.recommended_episode_count,
            }
        )

    def _clean_consolidation_actor_links(
        self,
        consolidation: SynthesisConsolidationResult,
        actor_metadata: ActorMetadata,
    ) -> tuple[SynthesisConsolidationResult, dict[str, Any]]:
        valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
        unknown_actor_ids = 0
        cleaned_clusters: list[EpisodeCandidateCluster] = []
        for cluster in consolidation.episode_candidate_clusters:
            actor_ids: list[str] = []
            seen: set[str] = set()
            for actor_id in cluster.actor_ids:
                if actor_id not in valid_actor_ids:
                    unknown_actor_ids += 1
                    continue
                if actor_id in seen:
                    continue
                seen.add(actor_id)
                actor_ids.append(actor_id)
            primary_actor_id = cluster.primary_actor_id
            if primary_actor_id and primary_actor_id not in valid_actor_ids:
                unknown_actor_ids += 1
                primary_actor_id = None
            if primary_actor_id and primary_actor_id not in actor_ids:
                actor_ids.insert(0, primary_actor_id)
            cleaned_clusters.append(
                cluster.model_copy(
                    update={
                        "actor_ids": actor_ids,
                        "primary_actor_id": primary_actor_id,
                    }
                )
            )
        return consolidation.model_copy(
            update={"episode_candidate_clusters": cleaned_clusters}
        ), {"unknown_actor_ids": unknown_actor_ids}

    async def _plan_series(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[list[EpisodePlan], dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger, "episode_planning", project_dir,
            episode_count=project.episode_count, strategy=strategy.strategy_type,
        ) as ctx:
            episode_map = {
                episode.episode_number: episode
                for episode in strategy.episodes
            }
            missing_episodes = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in episode_map
            ]
            if missing_episodes:
                raise RuntimeError(
                    "Narrative strategy did not assign cluster paths for "
                    f"episodes: {missing_episodes}"
                )
            passage_lookup = _build_passage_lookup(corpus)
            primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            cluster_lookup = {
                cluster.cluster_id: cluster
                for cluster in synthesis_map.episode_candidate_clusters
            }
            project_metadata = {
                "theme": project.theme,
                "sub_themes": project.sub_themes,
                "book_count": len(project.books),
                "books": [
                    {"book_id": b.book_id, "title": b.title, "author": b.author}
                    for b in project.books
                ],
                "target_episode_minutes": project.config.target_episode_minutes,
                "min_episode_minutes": project.config.min_episode_minutes,
            }
            planning_sem = asyncio.Semaphore(max(1, project.config.episode_planning_concurrency))
            ordered_episodes = [
                episode_map[episode_number]
                for episode_number in range(1, project.episode_count + 1)
            ]

            async def _plan_episode(
                episode: Any,
            ) -> tuple[int, EpisodePlan, dict[str, Any]]:
                async with planning_sem:
                    cluster_ids = [occurrence.cluster_id for occurrence in episode.cluster_path]
                    episode_synthesis_map_payload, primitive_ids = _build_episode_synthesis_map_payload(
                        synthesis_map,
                        cluster_ids,
                        cluster_lookup,
                    )
                    episode_actor_ids = {
                        actor.actor_id
                        for actor in episode.actor_arc_directives
                        if actor.actor_id
                    }
                    episode_actor_metadata = select_actor_metadata_subset(
                        actor_metadata,
                        episode_actor_ids,
                    )
                    primary_occurrence_ids = {
                        occurrence.occurrence_id
                        for occurrence in episode.cluster_path
                        if occurrence.usage == "primary"
                    }
                    passage_ids: list[str] = []
                    seen_passage_ids: set[str] = set()
                    for primitive_id in primitive_ids:
                        primitive = primitive_lookup.get(primitive_id)
                        if primitive is None:
                            continue
                        for passage_id in _primitive_passage_ids(primitive):
                            if passage_id in seen_passage_ids or passage_id not in passage_lookup:
                                continue
                            seen_passage_ids.add(passage_id)
                            passage_ids.append(passage_id)
                    available_passages = [
                        {
                            "passage_id": passage_lookup[passage_id].passage_id,
                            "book_id": passage_lookup[passage_id].book_id,
                            "text": _resolve_writing_passage_text(passage_lookup[passage_id]),
                            "chapter_ref": passage_lookup[passage_id].chapter_ref,
                        }
                        for passage_id in passage_ids
                    ]
                    episode_query_parts = [
                        episode.title,
                        episode.driving_question,
                        episode.thematic_focus,
                    ]
                    episode_query_parts.extend(episode.unresolved_questions)
                    episode_query_text = " ".join(
                        part for part in episode_query_parts if part
                    ).strip()
                    _trim_candidate_texts_by_bm25_query_text(
                        episode_query_text,
                        available_passages,
                        keep_fraction=0.5,
                    )
                    payload = self.episode_planning_agent.build_payload(
                        episode=episode.model_dump(mode="json"),
                        synthesis_map=episode_synthesis_map_payload,
                        project_metadata=project_metadata,
                        available_passages=available_passages,
                        actor_metadata=compact_actor_metadata(episode_actor_metadata),
                    )
                    plan_draft = await asyncio.to_thread(self.episode_planning_agent.run, payload)
                    plan_draft = plan_draft.model_copy(
                        update={"actor_arc_directives": episode.actor_arc_directives}
                    )
                    plan_draft, actor_link_metrics = clean_scene_actor_links(
                        plan_draft,
                        episode_actor_metadata,
                    )
                    covered_primary_occurrence_ids = {
                        scene.dominant_cluster_occurrence_id
                        for scene in plan_draft.scene_cards
                        if scene.card_kind == "normal" and scene.dominant_cluster_occurrence_id
                    }
                    missing_primary_occurrence_ids = sorted(
                        primary_occurrence_ids.difference(covered_primary_occurrence_ids)
                    )
                    if missing_primary_occurrence_ids:
                        retry_payload = self.episode_planning_agent.build_payload(
                            episode=episode.model_dump(mode="json"),
                            synthesis_map=episode_synthesis_map_payload,
                            project_metadata=project_metadata,
                            available_passages=available_passages,
                            actor_metadata=compact_actor_metadata(episode_actor_metadata),
                            planning_feedback={
                                "issue": "missing_primary_occurrence_coverage",
                                "missing_primary_occurrence_ids": missing_primary_occurrence_ids,
                            },
                        )
                        plan_draft = await asyncio.to_thread(
                            self.episode_planning_agent.run,
                            retry_payload,
                        )
                        plan_draft = plan_draft.model_copy(
                            update={"actor_arc_directives": episode.actor_arc_directives}
                        )
                        plan_draft, actor_link_metrics = clean_scene_actor_links(
                            plan_draft,
                            episode_actor_metadata,
                        )
                        covered_primary_occurrence_ids = {
                            scene.dominant_cluster_occurrence_id
                            for scene in plan_draft.scene_cards
                            if scene.card_kind == "normal" and scene.dominant_cluster_occurrence_id
                        }
                        missing_primary_occurrence_ids = sorted(
                            primary_occurrence_ids.difference(covered_primary_occurrence_ids)
                        )
                    if missing_primary_occurrence_ids:
                        raise RuntimeError(
                            "Episode planning failed to cover primary occurrences for episode "
                            f"{episode.episode_number}: {missing_primary_occurrence_ids}"
                        )
                    scene_card_count_warnings = _build_scene_card_count_warnings(
                        scene_card_count=len(plan_draft.scene_cards),
                        scene_card_target_min=project.config.scene_card_target_min,
                        scene_card_target_max=project.config.scene_card_target_max,
                    )
                    scene_card_primitive_warnings = _build_scene_card_primitive_warnings(
                        scene_cards=plan_draft.scene_cards,
                        primitive_pool_ids=set(primitive_ids),
                        primitive_min=project.config.scene_card_primitives_min,
                        primitive_max=project.config.scene_card_primitives_max,
                    )
                    planning_warnings = scene_card_count_warnings + scene_card_primitive_warnings
                    for warning in planning_warnings:
                        logger.warning(
                            "episode_planning_warning episode=%s %s",
                            episode.episode_number,
                            warning,
                        )
                    target_word_count = int(
                        round(
                            float(plan_draft.target_duration_minutes)
                            * float(self.settings.pipeline.spoken_words_per_minute)
                        )
                    )
                    plan = EpisodePlan.model_validate(
                        {
                            **plan_draft.model_dump(mode="json"),
                            "target_word_count": target_word_count,
                        }
                    )
                    report = {
                        "episode_number": episode.episode_number,
                        "scene_card_count": len(plan.scene_cards),
                        "scene_card_target_min": project.config.scene_card_target_min,
                        "scene_card_target_max": project.config.scene_card_target_max,
                        "scene_card_target_policy": project.config.scene_card_target_policy,
                        "scene_card_primitives_min": project.config.scene_card_primitives_min,
                        "scene_card_primitives_max": project.config.scene_card_primitives_max,
                        "scene_card_primitive_policy": project.config.scene_card_primitive_policy,
                        "scene_card_count_warnings": scene_card_count_warnings,
                        "scene_card_primitive_warnings": scene_card_primitive_warnings,
                        "scene_card_warning_count": len(planning_warnings),
                        "primary_occurrence_count": len(primary_occurrence_ids),
                        "covered_primary_occurrence_count": len(covered_primary_occurrence_ids),
                        "missing_primary_occurrence_ids": missing_primary_occurrence_ids,
                        "actor_link_metrics": actor_link_metrics,
                    }
                    return episode.episode_number, plan, report

            planning_results = await asyncio.gather(
                *[_plan_episode(episode) for episode in ordered_episodes]
            )
            planning_results.sort(key=lambda item: item[0])
            planned_episodes = [plan for _, plan, _ in planning_results]
            planning_reports = [report for _, _, report in planning_results]
            planning_actor_metrics = _merge_actor_metric_dicts(
                report.get("actor_link_metrics", {})
                for report in planning_reports
            )

            _save_json(
                project_dir / "series_plan.json",
                {"episodes": [episode.model_dump(mode="json") for episode in planned_episodes]},
            )
            _save_json(
                project_dir / "episode_plan_realization.json",
                {"episodes": planning_reports},
            )

            ctx["output_summary"] = {
                "episode_count": len(planned_episodes),
                "titles": [episode.title for episode in planned_episodes],
            }
            return planned_episodes, planning_actor_metrics

    def _write_passage_utilization(
        self,
        *,
        project: ThematicProject,
        corpus: ThematicCorpus,
        episode_plans: list[EpisodePlan],
        project_dir: Path,
        episode_numbers: list[int],
    ) -> None:
        utilized_passage_ids: set[str] = set()
        for episode_number in episode_numbers:
            payload = _load_json(project_dir / "episodes" / str(episode_number) / "episode_script.json")
            if payload is None:
                continue
            try:
                script = EpisodeScript.model_validate(payload)
            except Exception as exc:
                self.run_logger.log(
                    "passage_utilization_script_parse_error",
                    episode_number=episode_number,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                continue
            for section in script.prose_sections:
                utilized_passage_ids.update(citation.passage_id for citation in section.citations)
            for transition in script.transitions:
                utilized_passage_ids.update(citation.passage_id for citation in transition.citations)

        utilization = {
            "summary": {
                "episode_count": len(episode_numbers),
                "planned_episode_count": len(episode_plans),
                "total_passages": corpus.total_passages,
                "utilized_passages": len(utilized_passage_ids),
                "utilization_ratio": (
                    len(utilized_passage_ids) / max(1, corpus.total_passages)
                ),
            }
        }
        _save_json(project_dir / "passage_utilization.json", utilization)
        self.run_logger.log("passage_utilization", **utilization["summary"])

    def _build_writing_actor_metrics(
        self,
        project_dir: Path,
        spoken_scripts: list[tuple[int, SpokenScript]],
    ) -> dict[str, Any]:
        return {
            "completed_episode_count": len(spoken_scripts),
            "unknown_actor_ids": 0,
            "project_dir": str(project_dir),
        }

    def _write_actor_metadata_metrics(
        self,
        *,
        project_dir: Path,
        actor_metadata: ActorMetadata,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        episode_plans: list[EpisodePlan],
        metrics: dict[str, Any],
    ) -> None:
        payload = {
            "actor_count": len(actor_metadata.actors),
            "relationship_count": len(actor_metadata.relationships),
            "unresolved_mentions": len(actor_metadata.unresolved_mentions),
            "confidence_counts": _confidence_counts(actor_metadata),
            "actor_linkage": _actor_linkage_counts(
                synthesis_map=synthesis_map,
                strategy=strategy,
                episode_plans=episode_plans,
            ),
            "stage_metrics": metrics,
        }
        _save_json(project_dir / "actor_metadata_metrics.json", payload)
        self.run_logger.log(
            "actor_metadata_metrics",
            actor_count=payload["actor_count"],
            relationship_count=payload["relationship_count"],
            unresolved_mentions=payload["unresolved_mentions"],
            actor_linkage=payload["actor_linkage"],
        )

    # -----------------------------------------------------------------------
    # Phase 3: Episode Production
    # -----------------------------------------------------------------------

    async def _produce_episode(
        self,
        plan: EpisodePlan,
        project: ThematicProject,
        corpus: ThematicCorpus,
        actor_metadata: ActorMetadata,
        project_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, SpokenScript]:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(plan.episode_number)
            ep_dir.mkdir(parents=True, exist_ok=True)

            script = await self._write_episode(
                plan, project, corpus, ep_dir, project_dir, actor_metadata,
            )

            if not project.config.skip_grounding:
                report = await self._validate_grounding(
                    plan.episode_number, script, corpus, ep_dir, project_dir,
                )
                if report.overall_status != "PASSED":
                    script, report = await self._repair_loop(
                        plan.episode_number, script, report, corpus, ep_dir,
                        project_dir, max_attempts=project.config.max_repair_attempts,
                    )
            else:
                self.run_logger.log("grounding_skipped", episode=plan.episode_number)

            if not project.config.skip_spoken_delivery:
                spoken = await self._rewrite_for_speech(
                    plan.episode_number, script, project, ep_dir, project_dir,
                )
            else:
                spoken = SpokenScript(
                    episode_number=plan.episode_number,
                    title=script.title,
                    framing=script.framing,
                    sections=[
                        SpokenSection(section_id=section.section_id, text=section.text)
                        for section in script.prose_sections
                    ],
                    transitions=[
                        SpokenTransition(transition_id=transition.transition_id, text=transition.text)
                        for transition in script.transitions
                    ],
                    tts_provider=project.config.tts_provider,
                )
                _save_json(ep_dir / "spoken_script.json", spoken)
                self.run_logger.log("spoken_delivery_skipped", episode=plan.episode_number)

            return (plan.episode_number, spoken)

    async def _write_episode(
        self, plan: EpisodePlan, project: ThematicProject,
        corpus: ThematicCorpus, ep_dir: Path, project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> EpisodeScript:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger, f"write_episode_{plan.episode_number}", project_dir,
            episode=plan.episode_number,
            scene_card_count=len(plan.scene_cards),
            writing_source_mode=_WRITING_SOURCE_MODE_FULL_CHUNK,
        ) as ctx:
            passage_lookup = _build_passage_lookup(corpus)
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]
            scene_batches = _build_cluster_scene_batches(plan.scene_cards)
            scene_word_count_targets_lower = _compute_scene_word_count_targets(
                plan.scene_cards,
                plan.target_word_count,
                110.0,
            )
            scene_word_count_targets_higher = _compute_scene_word_count_targets(
                plan.scene_cards,
                plan.target_word_count,
                130.0,
            )
            full_episode_plan_payload = plan.model_dump(mode="json")
            scene_payload_by_id: dict[str, dict[str, Any]] = {}
            for scene_payload in full_episode_plan_payload.get("scene_cards", []):
                scene_id = scene_payload.get("scene_id")
                if not scene_id:
                    continue
                scene_payload.pop("estimated_duration_seconds", None)
                scene_payload["target_word_count_lower"] = int(
                    scene_word_count_targets_lower.get(scene_id, 0)
                )
                scene_payload["target_word_count_higher"] = int(
                    scene_word_count_targets_higher.get(scene_id, 0)
                )
                scene_payload_by_id[scene_id] = scene_payload
            all_sections: list[ProseSection] = []
            all_transitions: list[ScriptTransition] = []
            all_window_maps: list[Any] = []
            writing_agent = (
                self.writing_agent_no_citations
                if project.config.skip_grounding
                else self.writing_agent
            )

            for batch_index, batch_scene_cards in enumerate(scene_batches, start=1):
                active_scene_card_ids = [scene.scene_id for scene in batch_scene_cards]
                batch_scene_payloads = [
                    scene_payload_by_id[scene_id]
                    for scene_id in active_scene_card_ids
                    if scene_id in scene_payload_by_id
                ]
                batch_target_word_count_lower = sum(
                    scene_word_count_targets_lower.get(scene_id, 0)
                    for scene_id in active_scene_card_ids
                )
                batch_target_word_count_higher = sum(
                    scene_word_count_targets_higher.get(scene_id, 0)
                    for scene_id in active_scene_card_ids
                )
                episode_plan_payload = {
                    **full_episode_plan_payload,
                    "scene_cards": batch_scene_payloads,
                    "target_word_count": max(1, int(batch_target_word_count_higher)),
                }
                if batch_scene_payloads and episode_plan_payload.get("framing"):
                    episode_plan_payload["framing"] = {
                        **episode_plan_payload["framing"],
                        "handoff_scene_card_id": batch_scene_payloads[0]["scene_id"],
                    }
                batch_passage_ids: list[str] = []
                seen_passage_ids: set[str] = set()
                for scene in batch_scene_cards:
                    for passage_id in scene.passage_ids:
                        if passage_id in seen_passage_ids or passage_id not in passage_lookup:
                            continue
                        seen_passage_ids.add(passage_id)
                        batch_passage_ids.append(passage_id)
                passages = [
                    {
                        "passage_id": passage_lookup[passage_id].passage_id,
                        "book_id": passage_lookup[passage_id].book_id,
                        "text": _resolve_writing_passage_text(passage_lookup[passage_id]),
                        "chapter_ref": passage_lookup[passage_id].chapter_ref,
                    }
                    for passage_id in batch_passage_ids
                ]
                batch_actor_ids = {
                    actor.actor_id
                    for scene in batch_scene_cards
                    for actor in scene.actors
                    if actor.actor_id
                }
                if not batch_actor_ids:
                    batch_actor_ids.update(
                        actor.actor_id
                        for actor in plan.actor_arc_directives
                        if actor.actor_id
                    )
                batch_actor_metadata = select_actor_metadata_subset(
                    actor_metadata,
                    batch_actor_ids,
                )
                payload = writing_agent.build_payload(
                    episode_number=plan.episode_number,
                    batch_id=f"batch_{batch_index}",
                    episode_plan=episode_plan_payload,
                    active_scene_card_ids=active_scene_card_ids,
                    passages=passages,
                    book_metadata=book_metadata,
                    batch_target_word_count_lower=batch_target_word_count_lower,
                    batch_target_word_count_higher=batch_target_word_count_higher,
                    skip_grounding=project.config.skip_grounding,
                    actor_metadata=compact_actor_metadata(batch_actor_metadata),
                )
                result = await asyncio.to_thread(writing_agent.run, payload)
                if project.config.skip_grounding:
                    normalized_sections = [
                        ProseSection.model_validate({
                            **section.model_dump(mode="json"),
                            "citations": [],
                        })
                        for section in result.prose_sections
                    ]
                    normalized_transitions = [
                        ScriptTransition.model_validate({
                            **transition.model_dump(mode="json"),
                            "citations": [],
                        })
                        for transition in result.transitions
                    ]
                else:
                    normalized_sections = [
                        ProseSection.model_validate(section.model_dump(mode="json"))
                        for section in result.prose_sections
                    ]
                    normalized_transitions = [
                        ScriptTransition.model_validate(transition.model_dump(mode="json"))
                        for transition in result.transitions
                    ]
                all_sections.extend(normalized_sections)
                all_transitions.extend(normalized_transitions)
                all_window_maps.extend(result.window_map)

            script = EpisodeScript(
                episode_number=plan.episode_number,
                title=plan.title,
                framing=plan.framing,
                prose_sections=all_sections,
                transitions=all_transitions,
                window_map=all_window_maps,
                total_word_count=_script_total_word_count(
                    EpisodeScript(
                        episode_number=plan.episode_number,
                        title=plan.title,
                        framing=plan.framing,
                        prose_sections=all_sections,
                        transitions=all_transitions,
                        window_map=all_window_maps,
                    )
                ),
                estimated_duration_seconds=0,
            )
            script = script.model_copy(
                update={
                    "estimated_duration_seconds": _estimate_duration_seconds_from_words(
                        script.total_word_count,
                        float(self.settings.pipeline.spoken_words_per_minute),
                    )
                }
            )
            _save_json(ep_dir / "episode_script.json", script)

            ctx["output_summary"] = {
                "words": script.total_word_count,
                "sections": len(script.prose_sections),
                "transitions": len(script.transitions),
                "window_count": len(scene_batches),
            }
            return script

    async def _validate_grounding(
        self, episode_number: int, script: EpisodeScript,
        corpus: ThematicCorpus, ep_dir: Path, project_dir: Path,
    ) -> GroundingReport:
        async with _stage_log(
            self.run_logger, f"grounding_{episode_number}", project_dir,
            episode=episode_number, text_unit_count=len(script.prose_sections) + len(script.transitions),
        ) as ctx:
            passage_lookup: dict[str, dict] = {}
            for axis_passages in corpus.passages_by_axis.values():
                for p in axis_passages:
                    passage_lookup[p.passage_id] = {
                        "passage_id": p.passage_id,
                        "book_id": p.book_id,
                        "text": _resolve_writing_passage_text(p),
                    }

            payload = self.grounding_agent.build_payload(
                episode_number=episode_number,
                script=script.model_dump(mode="json"),
                passages=passage_lookup,
            )
            report = await asyncio.to_thread(self.grounding_agent.run, payload)
            _save_json(ep_dir / "grounding_report.json", report)

            ctx["output_summary"] = {
                "status": report.overall_status,
                "grounding_score": report.grounding_score,
                "attribution_accuracy": report.attribution_accuracy,
                "claim_count": len(report.claim_assessments),
                "fairness_flags": len(report.fairness_flags),
            }
            return report

    async def _repair_loop(
        self, episode_number: int, script: EpisodeScript,
        report: GroundingReport, corpus: ThematicCorpus,
        ep_dir: Path, project_dir: Path, max_attempts: int = 3,
    ) -> tuple[EpisodeScript, GroundingReport]:
        current_script = script
        current_report = report

        for attempt in range(1, max_attempts + 1):
            if current_report.overall_status == "PASSED":
                break

            failing_claims = [
                ca for ca in current_report.claim_assessments
                if ca.status in ("UNSUPPORTED", "FABRICATED")
            ]
            if not failing_claims and not current_report.fairness_flags:
                break

            async with _stage_log(
                self.run_logger, f"repair_{episode_number}_attempt_{attempt}", project_dir,
                episode=episode_number, attempt=attempt,
                failing_claims=len(failing_claims),
            ) as ctx:
                passage_lookup: dict[str, dict] = {}
                for axis_passages in corpus.passages_by_axis.values():
                    for p in axis_passages:
                        passage_lookup[p.passage_id] = {
                            "passage_id": p.passage_id,
                            "book_id": p.book_id,
                            "text": _resolve_writing_passage_text(p),
                        }

                failing_unit_ids = {claim.text_unit_id for claim in failing_claims}
                failing_unit_ids.update(flag.text_unit_id for flag in current_report.fairness_flags)
                failing_sections = [
                    section.model_dump(mode="json")
                    for section in current_script.prose_sections
                    if section.section_id in failing_unit_ids
                ]
                failing_transitions = [
                    transition.model_dump(mode="json")
                    for transition in current_script.transitions
                    if transition.transition_id in failing_unit_ids
                ]
                failure_reasons = [
                    {
                        "text_unit_id": c.text_unit_id,
                        "claim_text": c.claim_text,
                        "status": c.status,
                        "explanation": c.explanation,
                    }
                    for c in failing_claims
                ]

                payload = self.repair_agent.build_payload(
                    failing_sections=failing_sections,
                    failing_transitions=failing_transitions,
                    failure_reasons=failure_reasons,
                    passages=passage_lookup,
                )
                result = await asyncio.to_thread(self.repair_agent.run, payload)

                repaired_sections = {
                    section.section_id: section
                    for section in result.repaired_sections
                }
                repaired_transitions = {
                    transition.transition_id: transition
                    for transition in result.repaired_transitions
                }
                new_sections = []
                new_transitions = []
                diffs: list[SegmentDiff] = []
                for section in current_script.prose_sections:
                    if section.section_id in repaired_sections:
                        repaired = repaired_sections[section.section_id]
                        diffs.append(
                            SegmentDiff(
                                text_unit_id=section.section_id,
                                before=section.text,
                                after=repaired.text,
                            )
                        )
                        new_sections.append(repaired)
                    else:
                        new_sections.append(section)
                for transition in current_script.transitions:
                    if transition.transition_id in repaired_transitions:
                        repaired = repaired_transitions[transition.transition_id]
                        diffs.append(
                            SegmentDiff(
                                text_unit_id=transition.transition_id,
                                before=transition.text,
                                after=repaired.text,
                            )
                        )
                        new_transitions.append(repaired)
                    else:
                        new_transitions.append(transition)

                new_script = current_script.model_copy(
                    update={
                        "prose_sections": new_sections,
                        "transitions": new_transitions,
                    }
                )
                new_script = new_script.model_copy(
                    update={
                        "total_word_count": _script_total_word_count(new_script),
                        "estimated_duration_seconds": _estimate_duration_seconds_from_words(
                            _script_total_word_count(new_script),
                            float(self.settings.pipeline.spoken_words_per_minute),
                        ),
                    }
                )
                new_report = await self._validate_grounding(
                    episode_number, new_script, corpus, ep_dir, project_dir,
                )

                remaining = len([
                    ca for ca in new_report.claim_assessments
                    if ca.status in ("UNSUPPORTED", "FABRICATED")
                ])
                status = (
                    "RESOLVED" if new_report.overall_status == "PASSED"
                    else "IMPROVED" if new_report.grounding_score > current_report.grounding_score
                    else "NO_PROGRESS"
                )

                repair_result = RepairResult(
                    attempt_number=attempt,
                    original_script=current_script,
                    repaired_script=new_script,
                    claims_repaired=len(diffs),
                    remaining_failures=remaining,
                    diffs=diffs,
                    status=status,
                )
                _save_json(ep_dir / f"repair_attempt_{attempt}.json", repair_result)

                current_script = new_script
                current_report = new_report

                ctx["output_summary"] = {
                    "status": status,
                    "claims_repaired": len(diffs),
                    "remaining_failures": remaining,
                }

                if status == "NO_PROGRESS":
                    break

        _save_json(ep_dir / "episode_script.json", current_script)
        return current_script, current_report

    async def _rewrite_for_speech(
        self, episode_number: int, script: EpisodeScript,
        project: ThematicProject, ep_dir: Path, project_dir: Path,
    ) -> SpokenScript:
        async with _stage_log(
            self.run_logger, f"spoken_delivery_{episode_number}", project_dir,
            episode=episode_number, section_count=len(script.prose_sections),
        ) as ctx:
            payload = self.spoken_delivery_agent.build_payload(
                episode_number=episode_number,
                script=script.model_dump(mode="json"),
                max_words_per_segment=project.config.spoken_chunk_max_words,
                tts_provider=project.config.tts_provider,
            )
            result = await asyncio.to_thread(self.spoken_delivery_agent.run, payload)

            spoken = SpokenScript(
                episode_number=episode_number,
                title=script.title,
                framing=script.framing,
                sections=result.sections,
                transitions=result.transitions,
                tts_provider=project.config.tts_provider,
            )
            _save_json(ep_dir / "spoken_script.json", spoken)

            ctx["output_summary"] = {
                "sections": len(spoken.sections),
                "transitions": len(spoken.transitions),
            }
            return spoken

    # -----------------------------------------------------------------------
    # Phase 4: Audio Rendering
    # -----------------------------------------------------------------------

    def _ensure_ffmpeg_available(self) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg is required to merge episode audio into mp3 output. "
                "Install ffmpeg and ensure it is available on PATH."
            )
        return ffmpeg_path

    def _merge_audio_segments(self, segment_paths: list[Path], output_path: Path) -> None:
        ffmpeg_path = self._ensure_ffmpeg_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            concat_file = Path(handle.name)
            for segment_path in segment_paths:
                handle.write(_concat_file_entry(segment_path))

        try:
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-vn",
                    "-acodec",
                    "mp3",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            concat_file.unlink(missing_ok=True)

        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "ffmpeg failed to merge episode audio"
                + (f": {error_text}" if error_text else ".")
            )

    async def _render_existing_episode_audio(
        self,
        episode_number: int,
        manifest_path: Path,
        project_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> AudioManifest:
        async with semaphore:
            ep_dir = manifest_path.parent
            manifest = _load_model(manifest_path, RenderManifest)
            async with _stage_log(
                self.run_logger,
                f"audio_resynthesis_{episode_number}",
                project_dir,
                episode=episode_number,
                segment_count=len(manifest.segments),
                source_manifest=str(manifest_path),
            ) as ctx:
                audio_manifest = await self._synthesize_audio_manifest(
                    episode_number=episode_number,
                    manifest=manifest,
                    ep_dir=ep_dir,
                )
                ctx["output_summary"] = audio_manifest.diagnostics
                return audio_manifest

    async def _synthesize_audio_manifest(
        self,
        *,
        episode_number: int,
        manifest: RenderManifest,
        ep_dir: Path,
    ) -> AudioManifest:
        self._ensure_ffmpeg_available()
        audio_dir = ep_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_segments: list[AudioSegmentResult] = []
        retry_count = 0
        successful_paths: list[Path] = []
        merged_path = ep_dir / _MERGED_EPISODE_FILENAME
        merged_path.unlink(missing_ok=True)

        for seg in manifest.segments:
            if not seg.text.strip():
                continue
            for attempt in range(self.settings.pipeline.audio_retry_attempts + 1):
                try:
                    audio_bytes = await asyncio.to_thread(
                        self.tts_client.synthesize,
                        seg.text,
                        seg.voice_id,
                        self.settings.tts.audio_format,
                        instructions=seg.instructions,
                        speed=seg.speed,
                    )
                    audio_path = audio_dir / f"{seg.segment_id}.{self.settings.tts.audio_format}"
                    audio_path.write_bytes(audio_bytes)
                    successful_paths.append(audio_path)
                    audio_segments.append(
                        AudioSegmentResult(
                            segment_id=seg.segment_id,
                            audio_path=str(audio_path),
                            success=True,
                        )
                    )
                    break
                except Exception as exc:
                    retry_count += 1
                    self.run_logger.log(
                        "tts_retry",
                        episode=episode_number,
                        segment_id=seg.segment_id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    if attempt == self.settings.pipeline.audio_retry_attempts:
                        logger.error("TTS failed for segment %s: %s", seg.segment_id, exc)
                        audio_segments.append(
                            AudioSegmentResult(
                                segment_id=seg.segment_id,
                                audio_path="",
                                success=False,
                                error=str(exc),
                            )
                        )

        diagnostics: dict[str, Any] = {
            "total_segments": len(manifest.segments),
            "successful": sum(1 for s in audio_segments if s.success),
            "failed": sum(1 for s in audio_segments if not s.success),
            "retries": retry_count,
            "merged": False,
        }
        merged_audio_path: str | None = None

        if diagnostics["failed"] == 0:
            if not successful_paths:
                diagnostics["merge_error"] = "No audio segments were rendered."
            else:
                try:
                    await asyncio.to_thread(
                        self._merge_audio_segments,
                        successful_paths,
                        merged_path,
                    )
                    merged_audio_path = str(merged_path)
                    diagnostics["merged"] = True
                except Exception as exc:
                    diagnostics["merge_error"] = str(exc)

        audio_manifest = AudioManifest(
            episode_number=episode_number,
            audio_segments=audio_segments,
            merged_audio_path=merged_audio_path,
            total_duration_seconds=float(manifest.estimated_duration_seconds),
            diagnostics=diagnostics,
        )
        _save_json(ep_dir / "audio_manifest.json", audio_manifest)

        if diagnostics["failed"] > 0:
            raise RuntimeError(
                f"Audio rendering failed for {diagnostics['failed']} segment(s)."
            )
        if not diagnostics["merged"]:
            raise RuntimeError(diagnostics.get("merge_error", "Audio merge failed."))
        return audio_manifest

    async def _render_episode_audio(
        self, episode_number: int, spoken: SpokenScript,
        config: PipelineConfig,
        project_dir: Path,
        semaphore: asyncio.Semaphore, *,
        skip_audio: bool,
    ) -> AudioManifest:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(episode_number)

            async with _stage_log(
                self.run_logger, f"audio_{episode_number}", project_dir,
                episode=episode_number, segment_count=len(spoken.sections) + len(spoken.transitions),
            ) as ctx:
                expected_transitions = max(0, len(spoken.sections) - 1)
                actual_transitions = len(spoken.transitions)
                if actual_transitions != expected_transitions:
                    self.run_logger.log(
                        "spoken_transition_mismatch_warning",
                        episode=episode_number,
                        section_count=len(spoken.sections),
                        transition_count=actual_transitions,
                        expected_transition_count=expected_transitions,
                    )
                manifest = build_render_manifest(
                    spoken,
                    voice_id=self.settings.tts.voice,
                    speed=self.settings.tts.speed,
                    words_per_minute=self.settings.pipeline.spoken_words_per_minute,
                    base_instructions=self.settings.tts.instructions,
                )
                _save_json(ep_dir / "render_manifest.json", manifest)
                for seg in manifest.segments:
                    if not seg.hint_degradations:
                        continue
                    self.run_logger.log(
                        "tts_hint_degradation",
                        episode=episode_number,
                        segment_id=seg.segment_id,
                        degradations=seg.hint_degradations,
                    )
                estimated_minutes = manifest.estimated_duration_seconds / 60.0
                min_minutes = float(config.min_episode_minutes)
                target_minutes = float(config.target_episode_minutes)
                if estimated_minutes < min_minutes:
                    self.run_logger.log(
                        "episode_runtime_shortfall_warning",
                        episode=episode_number,
                        estimated_duration_minutes=estimated_minutes,
                        shortfall_minutes=(min_minutes - estimated_minutes),
                        min_episode_minutes=min_minutes,
                        target_episode_minutes=target_minutes,
                        policy=config.duration_shortfall_policy,
                    )

                if skip_audio:
                    ctx["output_summary"] = {
                        "skipped": True,
                        "total_segments": len(manifest.segments),
                    }
                    return AudioManifest(
                        episode_number=episode_number,
                        audio_segments=[],
                        diagnostics={
                            "skipped": True,
                            "total_segments": len(manifest.segments),
                        },
                    )
                audio_manifest = await self._synthesize_audio_manifest(
                    episode_number=episode_number,
                    manifest=manifest,
                    ep_dir=ep_dir,
                )
                ctx["output_summary"] = audio_manifest.diagnostics
                return audio_manifest
