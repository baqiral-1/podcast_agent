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
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.framing import EpisodeFramingAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.source_weaving import SourceWeavingAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.structuring import StructuringAgent
from podcast_agent.agents.synthesis_mapping import SynthesisMappingAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent
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
    BookRecord,
    ChapterInfo,
    ChunkingConfig,
    CoverageStats,
    EpisodeAssignment,
    EpisodeFraming,
    EpisodePlan,
    EpisodeScript,
    EpisodeSynthesisContext,
    EpisodeSynthesisTension,
    EpisodeMergedNarrativeRef,
    ExtractedPassage,
    GroundingReport,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProjectStatus,
    RenderManifest,
    RenderSegment,
    RepairResult,
    SegmentDiff,
    ScriptSegment,
    SpeechHints,
    SpokenScript,
    SpokenSegment,
    SynthesisMap,
    SynthesisTag,
    TextChunk,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)
from podcast_agent.tts.openai_compatible import build_tts_client

logger = logging.getLogger(__name__)


class StructuringStageError(RuntimeError):
    """Raised when chapter structuring fails for a book during ingestion."""

    def __init__(self, message: str, *, book_id: str, title: str, source_path: str) -> None:
        super().__init__(message)
        self.book_id = book_id
        self.title = title
        self.source_path = source_path


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


def _input_hash(*args: Any) -> str:
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SPOKEN_TAG_RE = re.compile(r"<[^>]+>")
_RUNTIME_UNDERSHOOT_WARNING_RATIO = 0.10
_WHITESPACE_RE = re.compile(r"\s+")
_WRITING_SOURCE_MODE_FULL_CHUNK = "full_chunk"
_SPOKEN_RATE_MULTIPLIER = {
    "slower": 0.94,
    "normal": 1.0,
    "faster": 1.08,
}


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


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


def _trim_candidate_texts_by_bm25(axis: ThematicAxis, candidates: list[dict]) -> None:
    if not candidates:
        return
    query_parts = [axis.name, axis.description]
    query_parts.extend(axis.guiding_questions)
    query_parts.extend(axis.keywords)
    query_text = " ".join(part for part in query_parts if part).strip()
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
        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            tokens = _tokenize(sentence)
            score = _bm25_score(tokens, query_terms, idf, avg_len)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top_n = max(1, math.ceil(len(sentences) / 4))
        selected = sorted(scored[:top_n], key=lambda item: item[1])
        trimmed = " ".join(sentence for _, _, sentence in selected).strip()
        if trimmed:
            cand["text"] = trimmed


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
    floor_per_book: int = 5,
    allocation_power: float = 2.0,
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
        score = max(0.0, float(relevance_by_book.get(book_id, 0.0)))
        weights[book_id] = score ** allocation_power

    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {book_id: 1.0 for book_id in book_ids}
        total_weight = float(book_count)

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


def _resolve_writing_passage_text(passage: ExtractedPassage) -> str:
    full_text = passage.full_text.strip()
    if full_text:
        return full_text
    return passage.text


def _select_top_passages_for_synthesis(
    passages: list[ExtractedPassage],
    *,
    top_k: int = 30,
) -> list[ExtractedPassage]:
    if not passages:
        return []
    top_n = max(1, min(top_k, len(passages)))
    ranked = sorted(
        passages,
        key=lambda p: (-p.relevance_score, -p.quotability_score, p.passage_id),
    )
    return ranked[:top_n]


def _select_synthesis_passages(
    passages: list[ExtractedPassage],
    cross_pair_ids: set[str],
) -> list[ExtractedPassage]:
    selected = _select_top_passages_for_synthesis(passages)
    selected_by_id = {p.passage_id: p for p in selected}
    additions = [
        p for p in passages
        if p.passage_id in cross_pair_ids and p.passage_id not in selected_by_id
    ]
    if additions:
        additions.sort(key=lambda p: p.passage_id)
        selected.extend(additions)
    return selected


def _select_episode_planning_passages(
    *,
    passages_by_axis: dict[str, list[ExtractedPassage]],
    assigned_axis_ids: list[str],
    selected_insight_passage_ids: set[str],
    supporting_passages_per_axis: int = 100,
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
        supporting_ranked: list[tuple[float, float, float, str, ExtractedPassage]] = []
        for passage in passages_by_axis.get(axis_id, []):
            if passage.passage_id in selected_insight_passage_ids:
                insight_passages.append(passage)
                continue
            key = _chunk_key(passage)
            is_multi_axis = len(chunk_axes.get(key, set())) > 1
            weighted_score = (
                (0.65 * passage.relevance_score)
                + (0.35 * passage.quotability_score)
                + (0.04 if is_multi_axis else 0.0)
            )
            supporting_ranked.append(
                (
                    -weighted_score,
                    -passage.relevance_score,
                    -passage.quotability_score,
                    passage.passage_id,
                    passage,
                )
            )
        insight_passages.sort(
            key=lambda p: (-p.relevance_score, -p.quotability_score, p.passage_id)
        )
        supporting_ranked.sort()
        supporting_passages = [
            passage
            for _, _, _, _, passage in supporting_ranked[:supporting_passages_per_axis]
        ]
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
        if item["merged_narrative_id"] in assignment.merged_narrative_ids
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


def _build_planning_feedback(realization: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue": "assigned_insight_realization",
        "episode_number": realization["episode_number"],
        "problem_insights": [
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
        ],
        "instruction": (
            "Revise the episode plan so every assigned insight is materially realized in beats "
            "using the assigned insight passage_ids."
        ),
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


def _compute_adaptive_rerank_target(
    *,
    candidate_count: int,
    rehydrated_count: int,
    valid_cross_pair_count: int,
    book_count: int,
    rerank_top_k: int,
) -> dict[str, Any]:
    if rehydrated_count <= 0:
        return {
            "base_total": rerank_top_k * max(1, book_count),
            "richness_factor": 0.5,
            "pair_density": 0.0,
            "pair_factor": 1.0,
            "raw_target_total": 0,
            "max_target_total": 0,
            "target_total": 0,
            "cap_applied": False,
        }

    base_total = max(1, rerank_top_k * max(1, book_count))
    richness_factor = _clamp(candidate_count / base_total, 0.5, 2.0)
    pair_density = valid_cross_pair_count / max(1, candidate_count)
    pair_factor = 1.0 + min(0.3, pair_density)
    raw_target = int(round(base_total * richness_factor * pair_factor))
    max_target_total = max(1, int(round(base_total * 1.5)))
    capped_target = min(raw_target, max_target_total)
    target_total = max(1, min(capped_target, rehydrated_count))
    return {
        "base_total": base_total,
        "richness_factor": round(richness_factor, 4),
        "pair_density": round(pair_density, 4),
        "pair_factor": round(pair_factor, 4),
        "raw_target_total": raw_target,
        "max_target_total": max_target_total,
        "target_total": target_total,
        "cap_applied": raw_target > max_target_total,
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


def _sanitize_spoken_text(text: str) -> str:
    cleaned = _SPOKEN_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _resolve_spoken_render_speed(speech_rate: str, base_speed: float) -> float:
    multiplier = _SPOKEN_RATE_MULTIPLIER.get(speech_rate, 1.0)
    resolved = round(base_speed * multiplier, 2)
    return min(4.0, max(0.1, resolved))


def _supports_segment_tts_instructions(tts_provider: str) -> bool:
    return tts_provider.strip().lower() in {"openai", "openai-compatible"}


def _normalize_tts_instruction_text(text: str | None, *, max_chars: int = 500) -> str | None:
    if not text:
        return None
    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


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
    if hints.pronunciation_hints:
        pronunciation_text = "; ".join(
            f"{hint.text} as {hint.spoken_as}"
            for hint in hints.pronunciation_hints[:4]
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
    framing: EpisodeFraming | None,
    voice_id: str = "ballad",
    speed: float = 1.0,
    words_per_minute: int = 130,
    base_instructions: str | None = None,
) -> RenderManifest:
    segments: list[RenderSegment] = []
    tts_provider = spoken_script.tts_provider

    if framing and framing.cold_open:
        segments.append(RenderSegment(
            text=framing.cold_open, voice_id=voice_id, speed=speed,
            pause_after_ms=1500,
        ))

    if framing and framing.recap:
        segments.append(RenderSegment(
            text=framing.recap, voice_id=voice_id, speed=speed,
            pause_before_ms=500, pause_after_ms=1000,
        ))

    for seg in spoken_script.segments:
        segments.extend(
            _render_segments_for_spoken_segment(
                seg,
                voice_id=voice_id,
                speed=speed,
                tts_provider=tts_provider,
                base_instructions=base_instructions,
            )
        )

    if framing and framing.preview:
        segments.append(RenderSegment(
            text=framing.preview, voice_id=voice_id, speed=speed,
            pause_before_ms=1000,
        ))

    total_words = sum(len(seg.text.split()) for seg in segments)
    estimated_seconds = int(total_words / words_per_minute * 60)

    return RenderManifest(
        episode_number=spoken_script.episode_number,
        segments=segments,
        total_segments=len(segments),
        estimated_duration_seconds=estimated_seconds,
    )


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

        self.structuring_agent = StructuringAgent(self.llm, max_retry_attempts=_retries("structuring"))
        self.chapter_summary_agent = ChapterSummaryAgent(self.llm, max_retry_attempts=_retries("chapter_summary"))
        self.book_summary_agent = BookSummaryAgent(self.llm, max_retry_attempts=_retries("book_summary"))
        self.theme_decomposition_agent = ThemeDecompositionAgent(self.llm, max_retry_attempts=_retries("theme_decomposition"))
        self.passage_extraction_agent = PassageExtractionAgent(self.llm, max_retry_attempts=_retries("passage_extraction"))
        self.synthesis_mapping_agent = SynthesisMappingAgent(self.llm, max_retry_attempts=_retries("synthesis_mapping"))
        self.narrative_strategy_agent = NarrativeStrategyAgent(self.llm, max_retry_attempts=_retries("narrative_strategy"))
        self.episode_planning_agent = EpisodePlanningAgent(self.llm, max_retry_attempts=_retries("episode_planning"))
        self.writing_agent = WritingAgent(self.llm, max_retry_attempts=_retries("episode_writing"))
        self.source_weaving_agent = SourceWeavingAgent(self.llm, max_retry_attempts=_retries("source_weaving"))
        self.grounding_agent = GroundingValidationAgent(self.llm, max_retry_attempts=_retries("grounding_validation"))
        self.repair_agent = RepairAgent(self.llm, max_retry_attempts=_retries("repair"))
        self.spoken_delivery_agent = SpokenDeliveryAgent(self.llm, max_retry_attempts=_retries("spoken_delivery"))
        self.framing_agent = EpisodeFramingAgent(self.llm, max_retry_attempts=_retries("episode_framing"))

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

        self.run_logger.bind_run(project_id)
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
                    path, title, author, project_id, project_dir, pipeline_config,
                )
            )
        book_results = await asyncio.gather(*book_tasks, return_exceptions=True)

        successful_books: list[BookRecord] = []
        structuring_failures: list[StructuringStageError] = []
        for i, result in enumerate(book_results):
            if isinstance(result, Exception):
                logger.error("Book %d failed: %s", i, result)
                self.run_logger.log("book_ingest_failed", index=i, error=str(result))
                if isinstance(result, StructuringStageError):
                    structuring_failures.append(result)
            else:
                successful_books.append(result)

        if structuring_failures:
            project = project.model_copy(update={"status": ProjectStatus.FAILED})
            _save_json(project_dir / "thematic_project.json", project)
            self.run_logger.log(
                "structuring_failure_barrier",
                failure_count=len(structuring_failures),
                failed_books=[
                    {
                        "book_id": failure.book_id,
                        "title": failure.title,
                        "source_path": failure.source_path,
                        "error": str(failure),
                    }
                    for failure in structuring_failures
                ],
            )
            raise RuntimeError(
                f"Structuring failed for {len(structuring_failures)} book(s). "
                "Aborting before Phase 2."
            )

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

        axes = await self._decompose_theme(project, project_dir)
        corpus = await self._extract_passages(project, axes, project_dir)
        synthesis_map = await self._map_synthesis(project, corpus, project_dir)

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

        strategy = await self._choose_narrative_strategy(project, synthesis_map, corpus, project_dir)
        project = self._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        episode_plans = await self._plan_series(
            project, synthesis_map, strategy, corpus, project_dir,
        )

        # Phase 3: Episode Production (parallel per episode)
        logger.info("Phase 3: Episode Production (%d episodes)", len(episode_plans))
        project = project.model_copy(update={"status": ProjectStatus.PRODUCING})

        sem = asyncio.Semaphore(pipeline_config.episode_write_concurrency)
        ep_tasks = [
            self._produce_episode(plan, project, corpus, project_dir, sem)
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

        # Framing (sequential)
        framings: dict[int, EpisodeFraming] = {}
        for i, (ep_num, spoken) in enumerate(spoken_scripts):
            prev_summary = spoken_scripts[i - 1][1].arc_plan if i > 0 else None
            next_summary = None
            if i < len(spoken_scripts) - 1:
                next_idx = spoken_scripts[i + 1][0] - 1
                if next_idx < len(episode_plans):
                    next_summary = episode_plans[next_idx].thematic_focus
            framing = await self._frame_episode(
                ep_num, len(spoken_scripts), spoken, prev_summary, next_summary,
                project, project_dir,
            )
            framings[ep_num] = framing

        # Phase 4: Audio Rendering (parallel per episode)
        logger.info("Phase 4: Audio Rendering")
        audio_sem = asyncio.Semaphore(pipeline_config.tts_concurrency)
        audio_tasks = [
            self._render_episode_audio(
                ep_num,
                spoken,
                project.config,
                framings.get(ep_num),
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
            try:
                chapters = await self._structure_chapters(book_record, raw_text, project_dir)
            except Exception as exc:
                raise StructuringStageError(
                    f"Structuring failed for book '{title}' ({source_path}): {exc}",
                    book_id=book_id,
                    title=title,
                    source_path=source_path,
                ) from exc
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
        self, book_record: BookRecord, raw_text: str, project_dir: Path,
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
                updated.append(chapter.model_copy(update={"summary": summary.summary}))
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
    ) -> list[ThematicAxis]:
        async with _stage_log(
            self.run_logger, "theme_decomposition", project_dir,
            theme=project.theme, sub_themes=project.sub_themes, book_count=len(project.books),
        ) as ctx:
            summary_payloads: list[tuple[str, dict[str, Any]]] = []
            for book in project.books:
                chapter_info = [
                    {"title": ch.title, "summary": ch.summary}
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
            result = await asyncio.to_thread(self.theme_decomposition_agent.run, payload)
            axes = result.axes

            valid_axes = [
                a for a in axes
                if sum(1 for s in a.relevance_by_book.values() if s >= 0.3) >= 2
            ]

            if len(valid_axes) < project.config.min_axes:
                logger.warning(
                    "Only %d valid axes (min %d). Using all %d.",
                    len(valid_axes), project.config.min_axes, len(axes),
                )
                valid_axes = axes[:project.config.max_axes]

            valid_axes = valid_axes[:project.config.max_axes]
            _save_json(project_dir / "thematic_axes.json",
                        {"axes": [a.model_dump(mode="json") for a in valid_axes]})

            ctx["output_summary"] = {
                "book_summary_count": len(book_summaries),
                "total_axes_generated": len(axes),
                "valid_axes": len(valid_axes),
                "axis_names": [a.name for a in valid_axes],
            }
            return valid_axes

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
            axis_candidate_budget = len(book_ids) * project.config.passage_retrieval_max_per_book
            max_log_per_book = max(
                200,
                max((info["per_book_budget"] for info in retrieval_depth_by_book.values()), default=0),
            )
            all_passages_by_axis: dict[str, list[ExtractedPassage]] = {}
            all_cross_pairs: list[PassagePair] = []
            candidate_counts_by_axis: dict[str, int] = {}
            axis_policy_by_axis: dict[str, dict[str, Any]] = {}

            sem = asyncio.Semaphore(project.config.passage_extraction_concurrency)

            def _process_axis(
                axis: ThematicAxis,
            ) -> tuple[str, list[ExtractedPassage], list[PassagePair], int, dict[str, Any], dict[str, Any]]:
                hits_by_book = self.retrieval.retrieve_for_axis(
                    axis=axis, project_id=project.project_id,
                    book_ids=book_ids,
                    k_per_book=max_log_per_book,
                )

                retrieval_log: dict[str, Any] = {
                    "axis_id": axis.axis_id,
                    "axis_name": axis.name,
                    "axis_description": axis.description,
                    "max_log_per_book": max_log_per_book,
                    "axis_candidate_budget": axis_candidate_budget,
                    "budget_strategy": "hybrid_percentage_floor_cap",
                    "passage_retrieval_percentage": project.config.passage_retrieval_percentage,
                    "passage_retrieval_min_per_book": project.config.passage_retrieval_min_per_book,
                    "passage_retrieval_max_per_book": project.config.passage_retrieval_max_per_book,
                    "legacy_passages_per_axis_per_book": project.config.passages_per_axis_per_book,
                    "allocation_policy": "floor_5_quadratic_global_rank",
                    "allocation_power": 2.0,
                    "allocation_floor": 5,
                    "books": [],
                }
                relevance_by_book = _resolve_axis_relevance(axis, book_ids)
                retrieval_log["relevance_by_book"] = relevance_by_book
                admitted_quota_by_book = _compute_weighted_admitted_budgets(
                    book_ids=book_ids,
                    axis_total_budget=axis_candidate_budget,
                    relevance_by_book=relevance_by_book,
                    floor_per_book=5,
                    allocation_power=retrieval_log["allocation_power"],
                )
                retrieval_log["admission_quota_by_book"] = admitted_quota_by_book

                ranked_rows: list[dict[str, Any]] = []
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
                        "admission_quota": admitted_quota_by_book.get(bid, 0),
                        "candidates": [],
                    }
                    for rank, hit in enumerate(hits, start=1):
                        priority = (relevance_by_book.get(bid, 0.0) ** 2) / max(1, rank)
                        row = {
                            "book_id": bid,
                            "rank": rank,
                            "priority": priority,
                            "hit": hit,
                            "title": book.title if book else "Unknown",
                            "author": book.author if book else "Unknown",
                        }
                        ranked_rows.append(row)
                        rows_by_book.setdefault(bid, []).append(row)
                    retrieval_log["books"].append(book_entry)

                ranked_rows.sort(
                    key=lambda row: (
                        -row["priority"],
                        row["rank"],
                        str(row["hit"].chunk_id),
                    )
                )
                admitted_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                selected_rows: list[dict[str, Any]] = []
                for row in ranked_rows:
                    bid = row["book_id"]
                    per_book_budget = admitted_quota_by_book.get(bid, 0)
                    if admitted_by_book.get(bid, 0) >= per_book_budget:
                        continue
                    selected_rows.append(row)
                    admitted_by_book[bid] = admitted_by_book.get(bid, 0) + 1
                    if len(selected_rows) >= axis_candidate_budget:
                        break
                selected_row_ids = {id(row) for row in selected_rows}
                candidates: list[dict[str, Any]] = []

                book_entry_by_id = {entry["book_id"]: entry for entry in retrieval_log["books"]}
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
                    empty_policy = _compute_adaptive_rerank_target(
                        candidate_count=candidate_count,
                        rehydrated_count=0,
                        valid_cross_pair_count=0,
                        book_count=len(book_ids),
                        rerank_top_k=project.config.rerank_top_k,
                    )
                    empty_policy.update({
                        "allocation_policy": retrieval_log["allocation_policy"],
                        "allocation_power": retrieval_log["allocation_power"],
                        "axis_candidate_budget": axis_candidate_budget,
                        "per_book_budget": retrieval_log["per_book_budget"],
                        "admitted_by_book": admitted_by_book,
                    })
                    return axis.axis_id, [], [], candidate_count, {
                        "candidate_pair_count": 0,
                        "valid_pair_count": 0,
                        "retained_pair_count": 0,
                        "dropped_missing_id_count": 0,
                        "dropped_same_book_count": 0,
                    }, empty_policy

                candidate_full_text_by_id = {
                    candidate["passage_id"]: candidate["text"]
                    for candidate in candidates
                }
                _trim_candidate_texts_by_bm25(axis, candidates)

                payload = self.passage_extraction_agent.build_payload(
                    axis_id=axis.axis_id, axis_name=axis.name,
                    axis_description=axis.description,
                    candidate_passages=candidates,
                )
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
                rehydrated_passages = []
                for candidate in candidates:
                    score = scores_by_id.get(candidate["passage_id"])
                    if score is None:
                        continue
                    trimmed_text = candidate["text"]
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
                rerank_policy = _compute_adaptive_rerank_target(
                    candidate_count=candidate_count,
                    rehydrated_count=len(rehydrated_passages),
                    valid_cross_pair_count=len(validated_pairs),
                    book_count=len(book_ids),
                    rerank_top_k=project.config.rerank_top_k,
                )
                ranked_rehydrated = sorted(
                    rehydrated_passages,
                    key=lambda p: (-p.relevance_score, -p.quotability_score, p.passage_id),
                )
                target_total = rerank_policy["target_total"]
                top_passages = ranked_rehydrated[:target_total]
                rerank_policy.update({
                    "allocation_policy": retrieval_log["allocation_policy"],
                    "allocation_power": retrieval_log["allocation_power"],
                    "axis_candidate_budget": axis_candidate_budget,
                    "per_book_budget": retrieval_log["per_book_budget"],
                    "admitted_by_book": admitted_by_book,
                })
                return axis.axis_id, top_passages, retained_pairs, candidate_count, cross_pair_validation, rerank_policy

            async def _process_axis_async(axis: ThematicAxis):
                async with sem:
                    return await asyncio.to_thread(_process_axis, axis)

            results = await asyncio.gather(*[_process_axis_async(axis) for axis in axes])

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
                    "post_rerank_count": len(axis_passages),
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
        self, project: ThematicProject, corpus: ThematicCorpus, project_dir: Path,
    ) -> SynthesisMap:
        async with _stage_log(
            self.run_logger, "synthesis_mapping", project_dir,
            axis_count=len(corpus.axes), total_passages=corpus.total_passages,
        ) as ctx:
            cross_pair_ids = {
                pid
                for pair in corpus.cross_book_pairs
                for pid in (pair.passage_a_id, pair.passage_b_id)
            }
            axes_summary = [
                {"axis_id": a.axis_id, "name": a.name, "description": a.description}
                for a in corpus.axes
            ]
            passages_summary: dict[str, list[dict]] = {}
            synthesis_passage_total = 0
            for axis_id, passages in corpus.passages_by_axis.items():
                selected_passages = _select_synthesis_passages(passages, cross_pair_ids)
                synthesis_passage_total += len(selected_passages)
                passages_summary[axis_id] = [
                    {
                        "passage_id": p.passage_id, "book_id": p.book_id,
                        "text": p.text[:500],
                        "relevance_score": p.relevance_score,
                        "synthesis_tags": [t.value for t in p.synthesis_tags],
                    }
                    for p in selected_passages
                ]

            cross_pairs = [
                {
                    "passage_a_id": pp.passage_a_id,
                    "passage_b_id": pp.passage_b_id,
                    "relationship": pp.relationship.value,
                    "strength": pp.strength,
                }
                for pp in corpus.cross_book_pairs
            ]
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]

            payload = self.synthesis_mapping_agent.build_payload(
                project_id=project.project_id, axes_summary=axes_summary,
                passages_by_axis=passages_summary, cross_book_pairs=cross_pairs,
                book_metadata=book_metadata,
            )
            result = await asyncio.to_thread(self.synthesis_mapping_agent.run, payload)

            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                insights=result.insights,
                narrative_threads=result.narrative_threads,
                book_relationship_matrix=result.book_relationship_matrix,
                unresolved_tensions=result.unresolved_tensions,
                quality_score=result.quality_score,
                merged_narratives=result.merged_narratives,
            )
            _save_json(project_dir / "synthesis_map.json", synthesis_map)

            ctx["output_summary"] = {
                "insights": len(synthesis_map.insights),
                "threads": len(synthesis_map.narrative_threads),
                "quality_score": synthesis_map.quality_score,
                "synthesis_passages": synthesis_passage_total,
            }
            return synthesis_map

    async def _choose_narrative_strategy(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        corpus: ThematicCorpus,
        project_dir: Path,
    ) -> NarrativeStrategy:
        async with _stage_log(
            self.run_logger, "narrative_strategy", project_dir,
            insight_count=len(synthesis_map.insights),
        ) as ctx:
            merged_catalog = _build_merged_narrative_catalog(synthesis_map)
            tension_catalog = _build_tension_catalog(synthesis_map)
            synthesis_summary = {
                "insights": [
                    {
                        "insight_id": i.insight_id,
                        "insight_type": i.insight_type.value,
                        "title": i.title,
                        "description": i.description,
                        "passage_ids": i.passage_ids,
                        "axis_ids": i.axis_ids,
                        "podcast_potential": i.podcast_potential,
                        "treatment": i.treatment,
                    }
                    for i in synthesis_map.insights
                ],
                "narrative_threads": [
                    {
                        "thread_id": t.thread_id,
                        "title": t.title,
                        "description": t.description,
                        "insight_ids": t.insight_ids,
                        "arc_type": t.arc_type,
                    }
                    for t in synthesis_map.narrative_threads
                ],
                "unresolved_tensions": tension_catalog,
                "merged_narratives": merged_catalog,
                "quality_score": synthesis_map.quality_score,
                "insight_count": len(synthesis_map.insights),
                "thread_count": len(synthesis_map.narrative_threads),
            }
            thematic_axes = []
            for axis in corpus.axes:
                passages = corpus.passages_by_axis.get(axis.axis_id, [])
                thematic_axes.append(
                    {
                        "axis_id": axis.axis_id,
                        "name": axis.name,
                        "description": axis.description,
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
            }

            payload = self.narrative_strategy_agent.build_payload(
                synthesis_map=synthesis_summary,
                thematic_axes=thematic_axes,
                project_metadata=project_metadata,
                episode_count=project.requested_episode_count,
            )
            strategy = await asyncio.to_thread(self.narrative_strategy_agent.run, payload)
            _save_json(project_dir / "narrative_strategy.json", strategy)

            ctx["output_summary"] = {
                "strategy": strategy.strategy_type,
                "recommended_episode_count": strategy.recommended_episode_count,
                "episode_assignments": len(strategy.episode_assignments),
            }
            return strategy

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

    async def _plan_series(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        corpus: ThematicCorpus,
        project_dir: Path,
    ) -> list[EpisodePlan]:
        async with _stage_log(
            self.run_logger, "episode_planning", project_dir,
            episode_count=project.episode_count, strategy=strategy.strategy_type,
        ) as ctx:
            assignment_map: dict[int, EpisodeAssignment] = {
                assignment.episode_number: assignment
                for assignment in strategy.episode_assignments
            }
            missing_assignments = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in assignment_map
            ]
            if missing_assignments:
                raise RuntimeError(
                    "Narrative strategy did not assign axis_ids/insight_ids for "
                    f"episodes: {missing_assignments}"
                )
            project_metadata = {
                "theme": project.theme,
                "sub_themes": project.sub_themes,
                "book_count": len(project.books),
                "books": [
                    {"book_id": b.book_id, "title": b.title, "author": b.author}
                    for b in project.books
                ],
                "attribution_budget": project.config.attribution_budget,
            }
            insights_by_id = {insight.insight_id: insight for insight in synthesis_map.insights}
            merged_catalog = _build_merged_narrative_catalog(synthesis_map)
            tension_catalog = _build_tension_catalog(synthesis_map)
            adjusted_episodes: list[EpisodePlan] = []
            realization_reports: list[dict[str, Any]] = []
            ordered_assignments = [
                assignment_map[episode_number]
                for episode_number in range(1, project.episode_count + 1)
            ]
            planning_requests: list[tuple[int, EpisodeAssignment, list[Any], EpisodeSynthesisContext, dict[str, Any]]] = []
            for idx, assignment in enumerate(ordered_assignments):
                selected_insights = [
                    insights_by_id[insight_id]
                    for insight_id in assignment.insight_ids
                    if insight_id in insights_by_id
                ]
                selected_insight_passage_ids = {
                    passage_id
                    for insight in selected_insights
                    for passage_id in insight.passage_ids
                }
                synthesis_context = _build_episode_synthesis_context(
                    assignment=assignment,
                    selected_insights=selected_insights,
                    synthesis_map=synthesis_map,
                    merged_catalog=merged_catalog,
                    tension_catalog=tension_catalog,
                )
                synthesis_subset = synthesis_context.model_dump(mode="json")
                selected_passages_by_axis = _select_episode_planning_passages(
                    passages_by_axis=corpus.passages_by_axis,
                    assigned_axis_ids=assignment.axis_ids,
                    selected_insight_passage_ids=selected_insight_passage_ids,
                )
                passages_summary = {
                    axis_id: [
                        ({
                            "passage_id": p.passage_id,
                            "book_id": p.book_id,
                            "relevance_score": p.relevance_score,
                            "quotability_score": p.quotability_score,
                        } | (
                            {"full_text": _resolve_writing_passage_text(p)}
                            if p.passage_id in selected_insight_passage_ids
                            else {"summary_text": p.text}
                        ))
                        for p in selected_passages_by_axis.get(axis_id, [])
                    ]
                    for axis_id in assignment.axis_ids
                }
                previous_episode = None
                if idx > 0:
                    prev = ordered_assignments[idx - 1]
                    previous_episode = {
                        "episode_number": prev.episode_number,
                        "title": prev.title,
                        "thematic_focus": prev.thematic_focus,
                    }
                next_episode = None
                if idx + 1 < len(ordered_assignments):
                    nxt = ordered_assignments[idx + 1]
                    next_episode = {
                        "episode_number": nxt.episode_number,
                        "title": nxt.title,
                        "thematic_focus": nxt.thematic_focus,
                    }
                payload = self.episode_planning_agent.build_payload(
                    episode_assignment=assignment.model_dump(mode="json"),
                    narrative_strategy=strategy.model_dump(mode="json"),
                    synthesis_map=synthesis_subset,
                    project_metadata=project_metadata,
                    available_passages=passages_summary,
                    previous_episode=previous_episode,
                    next_episode=next_episode,
                )
                planning_requests.append((idx, assignment, selected_insights, synthesis_context, payload))

            planning_sem = asyncio.Semaphore(project.config.episode_write_concurrency)

            async def _plan_single_episode(
                idx: int,
                assignment: EpisodeAssignment,
                selected_insights: list[Any],
                synthesis_context: EpisodeSynthesisContext,
                payload: dict[str, Any],
            ) -> tuple[int, EpisodeAssignment, EpisodePlan, dict[str, Any]]:
                async with planning_sem:
                    episode = await asyncio.to_thread(self.episode_planning_agent.run, payload)
                    realization = _evaluate_episode_plan_insight_realization(
                        assignment=assignment,
                        selected_insights=selected_insights,
                        plan=episode,
                    )
                    if realization["has_issues"]:
                        retry_payload = self.episode_planning_agent.build_payload(
                            episode_assignment=assignment.model_dump(mode="json"),
                            narrative_strategy=strategy.model_dump(mode="json"),
                            synthesis_map=synthesis_context.model_dump(mode="json"),
                            project_metadata=project_metadata,
                            available_passages=payload["available_passages"],
                            previous_episode=payload["previous_episode"],
                            next_episode=payload["next_episode"],
                            planning_feedback=_build_planning_feedback(realization),
                        )
                        episode = await asyncio.to_thread(self.episode_planning_agent.run, retry_payload)
                        realization = _evaluate_episode_plan_insight_realization(
                            assignment=assignment,
                            selected_insights=selected_insights,
                            plan=episode,
                        )
                return idx, assignment, episode, realization

            planning_results = await asyncio.gather(*[
                _plan_single_episode(idx, assignment, selected_insights, synthesis_context, payload)
                for idx, assignment, selected_insights, synthesis_context, payload in planning_requests
            ])
            planning_results.sort(key=lambda row: row[0])

            for _, assignment, episode, realization in planning_results:
                selected_insights = [
                    insights_by_id[insight_id]
                    for insight_id in assignment.insight_ids
                    if insight_id in insights_by_id
                ]
                synthesis_context = _build_episode_synthesis_context(
                    assignment=assignment,
                    selected_insights=selected_insights,
                    synthesis_map=synthesis_map,
                    merged_catalog=merged_catalog,
                    tension_catalog=tension_catalog,
                )
                episode = episode.model_copy(
                    update={
                        "episode_number": assignment.episode_number,
                        "title": episode.title or assignment.title,
                        "thematic_focus": episode.thematic_focus or assignment.thematic_focus,
                        "axis_ids": assignment.axis_ids,
                        "insight_ids": assignment.insight_ids,
                        "synthesis_context": synthesis_context,
                        "episode_strategy": episode.episode_strategy or assignment.episode_strategy,
                    }
                )
                realization_reports.append(
                    {
                        **realization,
                        "title": episode.title,
                    }
                )
                if realization["has_issues"]:
                    self.run_logger.log(
                        "episode_plan_insight_realization_warning",
                        episode=episode.episode_number,
                        problem_count=realization["problem_count"],
                        insights=[
                            {
                                "insight_id": item["insight_id"],
                                "status": item["status"],
                                "realized_count": item["realized_count"],
                                "expected_min": item["expected_min"],
                            }
                            for item in realization["insights"]
                            if item["status"] in {"weak", "zero"}
                        ],
                    )
                beats = list(episode.beats)
                total_beats = len(beats)
                if total_beats == 0:
                    adjusted_episodes.append(episode)
                    continue
                target_minutes = float(project.config.target_episode_minutes)
                min_minutes = float(project.config.min_episode_minutes)
                if float(episode.target_duration_minutes) < min_minutes:
                    self.run_logger.log(
                        "episode_plan_duration_warning",
                        episode=episode.episode_number,
                        target_duration_minutes=episode.target_duration_minutes,
                        configured_target_minutes=target_minutes,
                        configured_min_minutes=min_minutes,
                    )
                planned_beat_duration_minutes = (
                    sum(float(beat.estimated_duration_seconds) for beat in beats) / 60.0
                )
                target_duration_minutes = float(episode.target_duration_minutes)
                plan_shortfall_minutes = target_duration_minutes - planned_beat_duration_minutes
                plan_shortfall_ratio = (
                    plan_shortfall_minutes / target_duration_minutes
                    if target_duration_minutes > 0
                    else 0.0
                )
                if plan_shortfall_ratio > _RUNTIME_UNDERSHOOT_WARNING_RATIO:
                    self.run_logger.log(
                        "episode_plan_runtime_budget_warning",
                        episode=episode.episode_number,
                        target_duration_minutes=target_duration_minutes,
                        planned_beat_duration_minutes=planned_beat_duration_minutes,
                        shortfall_minutes=plan_shortfall_minutes,
                        shortfall_ratio=plan_shortfall_ratio,
                    )
                if not 40 <= total_beats <= 45:
                    self.run_logger.log(
                        "episode_plan_beats_warning",
                        episode=episode.episode_number,
                        beats=total_beats,
                    )
                budget = episode.attribution_budget or project.config.attribution_budget
                max_attributed = int(total_beats * budget)
                attributed_indices = [
                    i for i, beat in enumerate(beats)
                    if beat.attribution_level in ("light", "full")
                ]
                if len(attributed_indices) <= max_attributed:
                    adjusted_episodes.append(episode)
                    continue
                ordered_indices = [
                    i for i, beat in enumerate(beats) if beat.attribution_level == "full"
                ] + [
                    i for i, beat in enumerate(beats) if beat.attribution_level == "light"
                ]
                keep_set = set(ordered_indices[:max_attributed])
                adjusted_beats = []
                dropped_indices: list[int] = []
                for idx, beat in enumerate(beats):
                    if idx in keep_set:
                        adjusted_beats.append(beat)
                        continue
                    if beat.attribution_level in ("light", "full"):
                        dropped_indices.append(idx)
                        adjusted_beats.append(
                            beat.model_copy(update={"attribution_level": "none"})
                        )
                    else:
                        adjusted_beats.append(beat)
                episode = episode.model_copy(update={"beats": adjusted_beats})
                self.run_logger.log(
                    "attribution_budget_enforced",
                    episode=episode.episode_number,
                    attribution_budget=budget,
                    total_beats=total_beats,
                    max_attributed=max_attributed,
                    attributed_before=len(attributed_indices),
                    attributed_after=max_attributed,
                    adjusted_indices=dropped_indices,
                )
                adjusted_episodes.append(episode)

            _save_json(
                project_dir / "series_plan.json",
                {"episodes": [e.model_dump(mode="json") for e in adjusted_episodes]},
            )
            _save_json(
                project_dir / "episode_plan_realization.json",
                {"episodes": realization_reports},
            )

            ctx["output_summary"] = {
                "episode_count": len(adjusted_episodes),
                "titles": [e.title for e in adjusted_episodes],
            }
            return adjusted_episodes

    def _write_passage_utilization(
        self,
        *,
        project: ThematicProject,
        corpus: ThematicCorpus,
        episode_plans: list[EpisodePlan],
        project_dir: Path,
        episode_numbers: list[int],
    ) -> None:
        episode_scripts: list[EpisodeScript] = []
        for episode_number in episode_numbers:
            script_path = project_dir / "episodes" / str(episode_number) / "episode_script.json"
            payload = _load_json(script_path)
            if payload is None:
                continue
            try:
                episode_scripts.append(EpisodeScript.model_validate(payload))
            except Exception as exc:
                self.run_logger.log(
                    "passage_utilization_script_parse_error",
                    episode_number=episode_number,
                    path=str(script_path),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        utilization = _compute_passage_utilization(
            corpus=corpus,
            episode_plans=episode_plans,
            episode_scripts=episode_scripts,
            books=project.books,
        )
        _save_json(project_dir / "passage_utilization.json", utilization)
        self.run_logger.log("passage_utilization", **utilization["summary"])

    # -----------------------------------------------------------------------
    # Phase 3: Episode Production
    # -----------------------------------------------------------------------

    async def _produce_episode(
        self,
        plan: EpisodePlan,
        project: ThematicProject,
        corpus: ThematicCorpus,
        project_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, SpokenScript]:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(plan.episode_number)
            ep_dir.mkdir(parents=True, exist_ok=True)

            script = await self._write_episode(plan, project, corpus, ep_dir, project_dir)

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
                raw_segments = [
                    SpokenSegment(
                        segment_id=seg.segment_id,
                        text=seg.text,
                        max_words=project.config.spoken_chunk_max_words,
                    )
                    for seg in script.segments
                ]
                spoken = SpokenScript(
                    episode_number=plan.episode_number,
                    title=script.title,
                    segments=_normalize_spoken_segments(
                        raw_segments,
                        project.config.spoken_chunk_max_words,
                    ),
                    arc_plan=None,
                    tts_provider=project.config.tts_provider,
                )
                _save_json(ep_dir / "spoken_script.json", spoken)
                self.run_logger.log("spoken_delivery_skipped", episode=plan.episode_number)

            return (plan.episode_number, spoken)

    async def _write_episode(
        self, plan: EpisodePlan, project: ThematicProject,
        corpus: ThematicCorpus, ep_dir: Path, project_dir: Path,
    ) -> EpisodeScript:
        async with _stage_log(
            self.run_logger, f"write_episode_{plan.episode_number}", project_dir,
            episode=plan.episode_number,
            beat_count=len(plan.beats),
            writing_source_mode=_WRITING_SOURCE_MODE_FULL_CHUNK,
        ) as ctx:
            passage_ids = set()
            for beat in plan.beats:
                passage_ids.update(beat.passage_ids)

            passages = []
            for axis_passages in corpus.passages_by_axis.values():
                for p in axis_passages:
                    if p.passage_id in passage_ids:
                        passages.append({
                            "passage_id": p.passage_id, "book_id": p.book_id,
                            "text": _resolve_writing_passage_text(p),
                            "chapter_ref": p.chapter_ref,
                            "synthesis_tags": [t.value for t in p.synthesis_tags],
                        })

            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]

            payload = self.writing_agent.build_payload(
                episode_number=plan.episode_number,
                episode_plan=plan.model_dump(mode="json"),
                passages=passages, book_metadata=book_metadata,
                max_author_names_per_episode=project.config.max_author_names_per_episode,
                prefer_indirect_attribution=project.config.prefer_indirect_attribution,
                skip_grounding=project.config.skip_grounding,
            )
            payload["writing_source_mode"] = _WRITING_SOURCE_MODE_FULL_CHUNK
            result = await asyncio.to_thread(self.writing_agent.run, payload)

            if project.config.skip_grounding:
                normalized_segments = []
                for seg in result.segments:
                    segment_kwargs = {
                        "text": seg.text,
                        "segment_type": seg.segment_type,
                        "beat_id": seg.beat_id,
                        "source_book_ids": seg.source_book_ids,
                        "attribution_level": seg.attribution_level,
                        "citations": [],
                    }
                    if seg.segment_id:
                        segment_kwargs["segment_id"] = seg.segment_id
                    normalized_segments.append(ScriptSegment(**segment_kwargs))
                result_segments = normalized_segments
                result_citations = []
            else:
                result_segments = result.segments
                result_citations = result.citations

            total_words = sum(len(seg.text.split()) for seg in result_segments)
            script = EpisodeScript(
                episode_number=plan.episode_number, title=result.title,
                segments=result_segments, total_word_count=total_words,
                estimated_duration_seconds=int(
                    total_words / self.settings.pipeline.spoken_words_per_minute * 60
                ),
                citations=result_citations,
            )
            _save_json(ep_dir / "episode_script.json", script)
            planned_beat_duration_minutes = (
                sum(float(beat.estimated_duration_seconds) for beat in plan.beats) / 60.0
            )
            target_duration_minutes = float(plan.target_duration_minutes)
            written_duration_minutes = script.estimated_duration_seconds / 60.0
            write_shortfall_minutes = target_duration_minutes - written_duration_minutes
            write_shortfall_ratio = (
                write_shortfall_minutes / target_duration_minutes
                if target_duration_minutes > 0
                else 0.0
            )
            if write_shortfall_ratio > _RUNTIME_UNDERSHOOT_WARNING_RATIO:
                target_floor_minutes = target_duration_minutes * (
                    1.0 - _RUNTIME_UNDERSHOOT_WARNING_RATIO
                )
                likely_source = (
                    "planning"
                    if planned_beat_duration_minutes < target_floor_minutes
                    else "writing"
                )
                self.run_logger.log(
                    "episode_write_duration_shortfall_warning",
                    episode=plan.episode_number,
                    target_duration_minutes=target_duration_minutes,
                    planned_beat_duration_minutes=planned_beat_duration_minutes,
                    written_duration_minutes=written_duration_minutes,
                    shortfall_minutes=write_shortfall_minutes,
                    shortfall_ratio=write_shortfall_ratio,
                    likely_source=likely_source,
                )

            ctx["output_summary"] = {
                "words": total_words, "segments": len(result.segments),
                "citations": len(result_citations),
                "writing_source_mode": _WRITING_SOURCE_MODE_FULL_CHUNK,
                "target_duration_minutes": target_duration_minutes,
                "planned_beat_duration_minutes": planned_beat_duration_minutes,
                "written_duration_minutes": written_duration_minutes,
                "write_shortfall_minutes": write_shortfall_minutes,
                "write_shortfall_ratio": write_shortfall_ratio,
            }
            return script

    async def _validate_grounding(
        self, episode_number: int, script: EpisodeScript,
        corpus: ThematicCorpus, ep_dir: Path, project_dir: Path,
    ) -> GroundingReport:
        async with _stage_log(
            self.run_logger, f"grounding_{episode_number}", project_dir,
            episode=episode_number, segment_count=len(script.segments),
        ) as ctx:
            passage_lookup: dict[str, dict] = {}
            for axis_passages in corpus.passages_by_axis.values():
                for p in axis_passages:
                    passage_lookup[p.passage_id] = {
                        "passage_id": p.passage_id,
                        "book_id": p.book_id, "text": p.text,
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
                            "book_id": p.book_id, "text": p.text,
                        }

                failing_segments = [
                    seg.model_dump(mode="json") for seg in current_script.segments
                    if any(
                        c.cited_passage_id in [cit.passage_id for cit in seg.citations]
                        for c in failing_claims
                    )
                ]
                failure_reasons = [
                    {"claim_text": c.claim_text, "status": c.status, "explanation": c.explanation}
                    for c in failing_claims
                ]

                payload = self.repair_agent.build_payload(
                    failing_segments=failing_segments,
                    failure_reasons=failure_reasons,
                    passages=passage_lookup,
                )
                result = await asyncio.to_thread(self.repair_agent.run, payload)

                repaired_map = {seg.segment_id: seg for seg in result.repaired_segments}
                new_segments = []
                diffs: list[SegmentDiff] = []
                for seg in current_script.segments:
                    if seg.segment_id in repaired_map:
                        new_seg = repaired_map[seg.segment_id]
                        diffs.append(SegmentDiff(
                            segment_id=seg.segment_id,
                            before=seg.text, after=new_seg.text,
                        ))
                        new_segments.append(new_seg)
                    else:
                        new_segments.append(seg)

                total_words = sum(len(s.text.split()) for s in new_segments)
                new_script = current_script.model_copy(update={
                    "segments": new_segments,
                    "total_word_count": total_words,
                })
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
            episode=episode_number, segment_count=len(script.segments),
        ) as ctx:
            script_segments = [seg.model_dump(mode="json") for seg in script.segments]

            payload = self.spoken_delivery_agent.build_payload(
                episode_number=episode_number,
                script_segments=script_segments,
                max_words_per_segment=project.config.spoken_chunk_max_words,
                tts_provider=project.config.tts_provider,
            )
            result = await asyncio.to_thread(self.spoken_delivery_agent.run, payload)
            normalized_segments = _normalize_spoken_segments(
                result.segments,
                project.config.spoken_chunk_max_words,
            )

            spoken = SpokenScript(
                episode_number=episode_number, title=script.title,
                segments=normalized_segments, arc_plan=result.arc_plan,
                tts_provider=project.config.tts_provider,
            )
            _save_json(ep_dir / "spoken_script.json", spoken)

            ctx["output_summary"] = {"segment_count": len(spoken.segments)}
            return spoken

    # -----------------------------------------------------------------------
    # Framing (sequential)
    # -----------------------------------------------------------------------

    async def _frame_episode(
        self, episode_number: int, total_episodes: int,
        spoken: SpokenScript, previous_summary: str | None,
        next_summary: str | None, project: ThematicProject,
        project_dir: Path,
    ) -> EpisodeFraming:
        ep_dir = project_dir / "episodes" / str(episode_number)

        async with _stage_log(
            self.run_logger, f"framing_{episode_number}", project_dir,
            episode=episode_number, total_episodes=total_episodes,
        ) as ctx:
            current_summary = spoken.arc_plan or spoken.title
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]

            payload = self.framing_agent.build_payload(
                episode_number=episode_number,
                total_episodes=total_episodes,
                current_episode_summary=current_summary,
                previous_episode_summary=previous_summary,
                next_episode_summary=next_summary,
                book_metadata=book_metadata,
            )
            framing = await asyncio.to_thread(self.framing_agent.run, payload)
            _save_json(ep_dir / "episode_framing.json", framing)

            ctx["output_summary"] = {
                "has_recap": framing.recap is not None,
                "has_preview": framing.preview is not None,
                "has_cold_open": framing.cold_open is not None,
            }
            return framing

    # -----------------------------------------------------------------------
    # Phase 4: Audio Rendering
    # -----------------------------------------------------------------------

    async def _render_episode_audio(
        self, episode_number: int, spoken: SpokenScript,
        config: PipelineConfig,
        framing: EpisodeFraming | None, project_dir: Path,
        semaphore: asyncio.Semaphore, *,
        skip_audio: bool,
    ) -> AudioManifest:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(episode_number)

            async with _stage_log(
                self.run_logger, f"audio_{episode_number}", project_dir,
                episode=episode_number, segment_count=len(spoken.segments),
            ) as ctx:
                manifest = build_render_manifest(
                    spoken, framing,
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

                audio_dir = ep_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)

                audio_segments: list[AudioSegmentResult] = []
                retry_count = 0

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
                            audio_segments.append(AudioSegmentResult(
                                segment_id=seg.segment_id,
                                audio_path=str(audio_path), success=True,
                            ))
                            break
                        except Exception as exc:
                            retry_count += 1
                            self.run_logger.log(
                                "tts_retry", episode=episode_number,
                                segment_id=seg.segment_id, attempt=attempt + 1,
                                error=str(exc),
                            )
                            if attempt == self.settings.pipeline.audio_retry_attempts:
                                logger.error("TTS failed for segment %s: %s", seg.segment_id, exc)
                                audio_segments.append(AudioSegmentResult(
                                    segment_id=seg.segment_id, audio_path="",
                                    success=False, error=str(exc),
                                ))

                audio_manifest = AudioManifest(
                    episode_number=episode_number,
                    audio_segments=audio_segments,
                    diagnostics={
                        "total_segments": len(manifest.segments),
                        "successful": sum(1 for s in audio_segments if s.success),
                        "failed": sum(1 for s in audio_segments if not s.success),
                        "retries": retry_count,
                    },
                )
                _save_json(ep_dir / "audio_manifest.json", audio_manifest)

                ctx["output_summary"] = audio_manifest.diagnostics
                return audio_manifest
