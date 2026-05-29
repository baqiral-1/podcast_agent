"""Multi-book thematic podcast pipeline orchestrator.

Implements the four-phase pipeline:
  Phase 1: Ingest & Index (parallel per book)
  Phase 2: Thematic Intelligence (sequential cross-book)
  Phase 3: Episode Production (parallel per episode)
  Phase 4: Audio Rendering (parallel per episode)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
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
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy_enrichment import (
    NarrativeStrategyEnrichmentAgent,
)
from podcast_agent.agents.narrative_strategy_skeleton import (
    NarrativeStrategySkeletonAgent,
)
from podcast_agent.agents.narrative_state_reconciler import (
    NarrativeStateReconcilerAgent,
)
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_function_tagging import (
    PrimitiveFunctionTaggingAgent,
)
from podcast_agent.agents.quality_judge import QualityJudgeAgent
from podcast_agent.agents.scene_discovery import SceneDiscoveryAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.style_audit import StyleAuditAgent
from podcast_agent.agents.excerpt_extraction import ExcerptExtractionAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.pipeline.excerpt_verification import verify_excerpt
from podcast_agent.pipeline.style_audit_linting import (
    compute_style_audit_lint_flags,
)
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent

# Grounding stage removed; QualityJudgeAgent (above) is the new
# post-writing gate (Change 3).
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.config import Settings
from podcast_agent.ingestion import read_source_text, extract_chapters_from_source
from podcast_agent.langchain.llm import build_llm_client
from podcast_agent.langchain.runnables import ComplianceViolationError
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.concurrency import configure_llm_semaphore
from podcast_agent.narrative_state import fold_planned_pre_states
from podcast_agent.retrieval.search import RetrievalService
from podcast_agent.retrieval.vector_store import PGVectorRetrieval
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    AudioManifest,
    AudioSegmentResult,
    ActorExplanationRealization,
    ActorMetadata,
    BaseSynthesisPrimitive,
    BookRecord,
    Citation,
    ChapterInfo,
    ChunkingConfig,
    COMPARATIVE_ASIDE_TOLERANCE,
    ContinuityCarryItem,
    CoverageStats,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeSpine,
    EpisodeScript,
    ExtractedPassage,
    EpisodeQualityScore,
    MustLandFacts,
    NarrativeStrategy,
    NarrativeStrategyEnrichment,
    NarrativeStrategySkeleton,
    NarrativeState,
    NarrativeStateReconciliation,
    PassagePair,
    PipelineConfig,
    ProseSection,
    ProjectStatus,
    PrimitiveFunctionTaggingArtifact,
    PrimitiveFunctionTaggingOverlayArtifact,
    PrimitiveEnrichmentOverlay,
    PRIMITIVE_SUBSTRATES,
    PrimitiveSubstrate,
    ProgressionStage,
    RenderManifest,
    RenderSegment,
    resolve_pipeline_config_for_mode,
    SceneCard,
    SceneCardDraft,
    SceneDiscoveryArtifact,
    EpisodePlanDraft,
    SceneJob,
    ThreadSectionPresence,
    ThreadFallbackMode,
    SeriesNarratorProfile,
    SeriesActorExplanationItem,
    SeriesExplanationItem,
    SectionSonicBeat,
    SectionSonicObligation,
    SectionSonicPlan,
    SonicCue,
    SpeechHints,
    StyleAuditResponse,
    SpokenSection,
    SpokenScript,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitiveBase,
    SynthesisPrimitivesArtifact,
    SynthesisTag,
    TextChunk,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
    PodcastMode,
    primitive_substrate_target_ranges_for_mode,
    apply_primitive_enrichment_overlay,
    effective_narrator_allowed_moves,
    scene_discovery_candidate_range_for_mode,
    scene_job_budget_for_mode,
    ExcerptArtifact,
    ExcerptRecord,
    WithholdUntil,
    WordCountPriority,
)
from podcast_agent.tts.openai_compatible import (
    build_tts_client,
    supports_openai_tts_instructions,
)
from podcast_agent.utils.actor_metadata import (
    clean_axis_actor_ids,
    clean_excerpt_actor_links,
    clean_narrative_strategy_actor_links,
    clean_scene_actor_links,
    clean_synthesis_primitive_actor_links,
    compact_actor_registry,
    compact_actor_metadata,
    collect_actor_ids_for_primitives,
    normalize_actor_name,
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


def _architecture_section_sonic_plan_by_id(
    architecture: EpisodeArchitecture | None,
) -> dict[str, SectionSonicPlan]:
    if architecture is None:
        return {}
    return {
        section.section_id: section.section_sonic_plan
        for section in architecture.sections
        if section.section_sonic_plan is not None
    }


def _attach_section_sonic_plan_to_prose_sections(
    sections: list[ProseSection],
    architecture: EpisodeArchitecture | None,
) -> list[ProseSection]:
    section_sonic_plan_by_id = _architecture_section_sonic_plan_by_id(architecture)
    if not section_sonic_plan_by_id:
        return sections
    return [
        section.model_copy(
            update={"section_sonic_plan": section_sonic_plan_by_id.get(section.section_id)}
        )
        for section in sections
    ]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(path: Path, model_type: type[Any]) -> Any:
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    return model_type.model_validate(payload)


def _persist_failed_project_state(
    *,
    project: ThematicProject,
    project_dir: Path,
    run_logger: RunLogger,
    exc: Exception,
) -> ThematicProject:
    failed_project = project.model_copy(update={"status": ProjectStatus.FAILED})
    _save_json(project_dir / "thematic_project.json", failed_project)
    run_logger.log(
        "pipeline_error",
        project_id=failed_project.project_id,
        status=failed_project.status.value,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )
    return failed_project


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


def _primitive_counts_by_substrate(
    artifact: SynthesisPrimitivesArtifact | PrimitiveFunctionTaggingArtifact | SynthesisMap,
) -> dict[str, int]:
    counts = {substrate: 0 for substrate in PRIMITIVE_SUBSTRATES}
    for primitive in artifact.primitives:
        counts[primitive.substrate.value] = counts.get(primitive.substrate.value, 0) + 1
    return counts


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _spoken_delivery_batch_section_id(index: int) -> str:
    return f"spoken_batch_{index:02d}"


def _section_word_count(section: ProseSection) -> int:
    return len(section.text.split())


def _text_word_count(value: str) -> int:
    return len((value or "").split())


def _project_scene_word_count(
    scene: SceneCardDraft | SceneCard,
    *,
    words_per_minute: float,
) -> int:
    return max(
        1,
        int(round((float(scene.estimated_duration_seconds) * float(words_per_minute)) / 60.0)),
    )


def _score_spoken_delivery_batch_partition(
    sections: list[ProseSection],
    cut_points: tuple[int, ...],
) -> tuple[float, int, tuple[int, ...]]:
    boundaries = (0, *cut_points, len(sections))
    batch_totals = [
        sum(_section_word_count(section) for section in sections[start:end])
        for start, end in zip(boundaries, boundaries[1:])
    ]
    target = sum(batch_totals) / len(batch_totals)
    worst_deviation = max(abs(total - target) for total in batch_totals)
    spread = max(batch_totals) - min(batch_totals)
    return (worst_deviation, spread, cut_points)


def _build_tts_provider_capabilities(tts_provider: str) -> dict:
    """Capability advertisement for the oral rewriter (Change 4).

    Only ``openai-compatible`` (specifically ``gpt-4o-mini-tts``) accepts
    free-form per-segment ``instructions``. Kokoro and the stock OpenAI
    ``tts-1`` / ``tts-1-hd`` models don't. The agent uses this signal to
    decide whether to emit ``SpokenSegment.delivery_instructions`` strings.
    """
    provider = (tts_provider or "").lower()
    supports_instructions = provider in {"openai-compatible", "openai"}
    voice_catalog: list[str] = []
    if provider in {"openai-compatible", "openai"}:
        voice_catalog = [
            "alloy",
            "ash",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ]
    elif provider == "kokoro":
        voice_catalog = ["af_heart"]
    return {
        "provider": provider,
        "supports_per_segment_instructions": supports_instructions,
        "voice_catalog": voice_catalog,
    }


def _build_quotability_marks_for_batch(
    *,
    prose_batch: list[ProseSection],
    script: EpisodeScript,
    plan: "EpisodePlan | None",
) -> list[dict]:
    """Surface verbatim excerpts the oral rewriter may voice in actor voice.

    A ``quotability_mark`` is emitted for every scene-attached excerpt in
    the batch's sections. The spoken-delivery agent decides which to
    render in actor voice via the ``excerpt_type`` allow-list and the
    non-empty ``verbatim_excerpt`` requirement inside its prompt; this
    helper does not filter on the score.
    """
    if plan is None:
        return []
    section_ids_in_batch = {ps.section_id for ps in prose_batch}
    _excerpt_lookup: dict[str, dict] = {}
    # plan.framing / scene_cards may carry excerpt_ids; look them up against
    # the corpus excerpts via plan.scene_cards[].excerpt_ids if present.
    scene_excerpt_pairs: list[tuple[str, str]] = []
    for scene in getattr(plan, "scene_cards", []):
        if scene.section_id not in section_ids_in_batch:
            continue
        for excerpt_id in getattr(scene, "excerpt_ids", []) or []:
            scene_excerpt_pairs.append((scene.section_id, excerpt_id))
    if not scene_excerpt_pairs:
        return []
    # The full excerpt records live on the corpus; we accept that the
    # caller may not have those here in build-time scope. The agent will
    # still receive the excerpt_id, allowing it to coordinate marks even
    # without verbatim text.
    marks: list[dict] = []
    for section_id, excerpt_id in scene_excerpt_pairs:
        marks.append(
            {
                "section_id": section_id,
                "excerpt_id": excerpt_id,
            }
        )
    return marks


def _build_spoken_delivery_batches(
    sections: list[ProseSection],
    *,
    max_batches: int = 4,
) -> list[list[ProseSection]]:
    if not sections:
        return []
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    batch_count = min(max_batches, len(sections))
    if batch_count == 1:
        return [list(sections)]

    best_cut_points: tuple[int, ...] | None = None
    best_score: tuple[float, int, tuple[int, ...]] | None = None
    section_count = len(sections)

    candidate_cut_sets = combinations(range(1, section_count), batch_count - 1)

    for cut_points in candidate_cut_sets:
        score = _score_spoken_delivery_batch_partition(sections, cut_points)
        if best_score is None or score < best_score:
            best_score = score
            best_cut_points = cut_points

    if best_cut_points is None:
        return [list(sections)]

    boundaries = (0, *best_cut_points, len(sections))
    return [list(sections[start:end]) for start, end in zip(boundaries, boundaries[1:])]


_SENTENCE_TAIL_RE = re.compile(r'.*?[.!?](?:["\')\]]+)?(?=\s|$)', re.DOTALL)
_WRITING_WINDOW_MIN_SHARE = 0.30


def _partition_window_totals(
    prefix_totals: list[int],
    cut_points: tuple[int, ...],
) -> list[int]:
    boundaries = (0, *cut_points, len(prefix_totals) - 1)
    return [
        prefix_totals[end] - prefix_totals[start] for start, end in zip(boundaries, boundaries[1:])
    ]


def _score_episode_writing_window_partition(
    prefix_totals: list[int],
    cut_points: tuple[int, ...],
) -> tuple[float, int, tuple[int, ...]]:
    window_totals = _partition_window_totals(prefix_totals, cut_points)
    target = sum(window_totals) / len(window_totals)
    worst_deviation = max(abs(total - target) for total in window_totals)
    spread = max(window_totals) - min(window_totals)
    return (worst_deviation, spread, cut_points)


def _extract_previous_spoken_tail(
    text: str,
    *,
    max_sentences: int = 4,
    max_words: int = 90,
) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None

    sentences = [
        match.group(0).strip()
        for match in _SENTENCE_TAIL_RE.finditer(normalized)
        if match.group(0).strip()
    ]
    if not sentences:
        return None

    tail_sentences = sentences[-max_sentences:]
    selected: list[str] = []
    word_total = 0
    for sentence in reversed(tail_sentences):
        sentence_words = len(sentence.split())
        if sentence_words > max_words:
            continue
        if selected and word_total + sentence_words > max_words:
            continue
        selected.append(sentence)
        word_total += sentence_words

    if not selected:
        return None
    return " ".join(reversed(selected))


def _split_episode_writing_windows(
    *,
    plan: EpisodePlan,
    architecture: EpisodeArchitecture,
    scene_word_count_targets_lower: dict[str, int],
    scene_word_count_targets_higher: dict[str, int],
    max_windows: int = 2,
) -> list[list[SceneCard]]:
    scene_cards = list(plan.scene_cards)
    if not scene_cards:
        return []
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    if len(scene_cards) < 2 or max_windows == 1:
        return [scene_cards]

    midpoint_targets = {
        scene.scene_id: max(
            1,
            int(
                round(
                    (
                        scene_word_count_targets_lower.get(scene.scene_id, 0)
                        + scene_word_count_targets_higher.get(scene.scene_id, 0)
                    )
                    / 2.0
                )
            ),
        )
        for scene in scene_cards
    }
    total_target_words = sum(midpoint_targets.values())
    scene_section_ids = {scene.section_id for scene in scene_cards}
    section_order = [
        section.section_id
        for section in architecture.sections
        if section.section_id in scene_section_ids
    ]
    ordered_section_ids = {section_id: idx for idx, section_id in enumerate(section_order)}

    section_boundaries = [
        boundary_index
        for boundary_index in range(1, len(scene_cards))
        if scene_cards[boundary_index - 1].section_id != scene_cards[boundary_index].section_id
        and ordered_section_ids.get(scene_cards[boundary_index - 1].section_id, -1)
        < ordered_section_ids.get(scene_cards[boundary_index].section_id, -1)
    ]

    if not section_boundaries:
        return [scene_cards]

    if total_target_words > 0:
        prefix_targets = [0]
        for scene in scene_cards:
            prefix_targets.append(prefix_targets[-1] + midpoint_targets[scene.scene_id])
    else:
        prefix_targets = list(range(len(scene_cards) + 1))
        total_target_words = len(scene_cards)

    max_window_count = min(max_windows, len(section_boundaries) + 1, len(scene_cards))
    for window_count in range(max_window_count, 1, -1):
        min_window_target_words = total_target_words * (
            (_WRITING_WINDOW_MIN_SHARE * 2.0) / window_count
        )
        best_cut_points: tuple[int, ...] | None = None
        best_score: tuple[float, int, tuple[int, ...]] | None = None
        for cut_points in combinations(section_boundaries, window_count - 1):
            if any(
                window_total < min_window_target_words
                for window_total in _partition_window_totals(prefix_targets, cut_points)
            ):
                continue
            score = _score_episode_writing_window_partition(prefix_targets, cut_points)
            if best_score is None or score < best_score:
                best_score = score
                best_cut_points = cut_points
        if best_cut_points is None:
            continue
        boundaries = (0, *best_cut_points, len(scene_cards))
        return [scene_cards[start:end] for start, end in zip(boundaries, boundaries[1:])]

    return [scene_cards]


def _build_window_architecture_payload(
    *,
    architecture: EpisodeArchitecture,
    window_scene_cards: list[SceneCard],
    section_authorial_passages_by_section_id: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    section_id_set = {scene.section_id for scene in window_scene_cards}
    filtered_sections = []
    for section in architecture.sections:
        if section.section_id not in section_id_set:
            continue
        section_payload = section.model_dump(mode="json")
        if section_authorial_passages_by_section_id is not None:
            section_payload["authorial_passages"] = list(
                section_authorial_passages_by_section_id.get(section.section_id, [])
            )
        filtered_sections.append(section_payload)

    architecture_payload = {
        "episode_number": architecture.episode_number,
        "major_turn_section_id": architecture.major_turn_section_id,
        "sections": filtered_sections,
    }
    if filtered_sections and architecture.major_turn_section_id not in section_id_set:
        architecture_payload["major_turn_section_id"] = filtered_sections[-1]["section_id"]
    return architecture_payload


def _build_writer_strategy_episode_payload(
    strategy_episode: StrategyEpisode,
) -> dict[str, Any]:
    spine = strategy_episode.episode_spine
    return {
        "episode_number": strategy_episode.episode_number,
        "title": strategy_episode.title,
        "episode_spine": {
            "listener_problem": spine.listener_problem,
            "episode_answer": spine.episode_answer,
            "pressure_line": spine.pressure_line,
            "core_primitive_ids": list(spine.core_primitive_ids),
        },
        "narrative_agenda": strategy_episode.narrative_agenda.model_dump(mode="json"),
    }


def _build_window_plan_payload(
    *,
    plan: EpisodePlan,
    window_scene_cards: list[SceneCard],
    scene_payload_by_id: dict[str, dict[str, Any]],
    target_word_count: int,
) -> dict[str, Any]:
    return {
        "episode_number": plan.episode_number,
        "framing": plan.framing.model_dump(mode="json"),
        "scene_cards": [
            scene_payload_by_id[scene.scene_id]
            for scene in window_scene_cards
            if scene.scene_id in scene_payload_by_id
        ],
        "dropped_support_primitive_reasons": dict(plan.dropped_support_primitive_reasons),
        "target_word_count": max(1, int(target_word_count)),
    }


def _normalize_scene_fact_tiers(must_land_facts: Any) -> dict[str, list[str]]:
    if isinstance(must_land_facts, MustLandFacts):
        return must_land_facts.model_dump(mode="json")
    if isinstance(must_land_facts, list):
        return {
            "required": [str(fact).strip() for fact in must_land_facts if str(fact).strip()],
            "strongly_preferred": [],
            "if_room": [],
        }
    if isinstance(must_land_facts, dict):
        required = [
            str(fact).strip()
            for fact in list(must_land_facts.get("required", []) or [])
            if str(fact).strip()
        ]
        strongly_preferred = [
            str(fact).strip()
            for fact in list(must_land_facts.get("strongly_preferred", []) or [])
            if str(fact).strip()
        ]
        if_room = [
            str(fact).strip()
            for fact in list(must_land_facts.get("if_room", []) or [])
            if str(fact).strip()
        ]
        return {
            "required": required,
            "strongly_preferred": strongly_preferred,
            "if_room": if_room,
        }
    return {
        "required": [],
        "strongly_preferred": [],
        "if_room": [],
    }


def _ordered_scene_fact_values(must_land_facts: Any) -> list[str]:
    fact_tiers = _normalize_scene_fact_tiers(must_land_facts)
    return [
        *fact_tiers["required"],
        *fact_tiers["strongly_preferred"],
        *fact_tiers["if_room"],
    ]


def _first_required_scene_fact(must_land_facts: Any) -> str:
    fact_tiers = _normalize_scene_fact_tiers(must_land_facts)
    return fact_tiers["required"][0] if fact_tiers["required"] else ""


def _scene_fact_total_count(must_land_facts: Any) -> int:
    return len(_ordered_scene_fact_values(must_land_facts))


def _withhold_until_payload(withhold_until: Any) -> dict[str, Any] | None:
    if isinstance(withhold_until, WithholdUntil):
        return withhold_until.model_dump(mode="json")
    if isinstance(withhold_until, str):
        normalized = withhold_until.strip()
        if not normalized:
            return None
        if normalized.startswith("scene_"):
            return {
                "subject": "withheld material",
                "reveal_phase": "open",
                "reveal_scene_id": normalized,
                "surrogate_label": "",
            }
        return {
            "subject": normalized,
            "reveal_phase": "close",
            "reveal_scene_id": None,
            "surrogate_label": "",
        }
    if isinstance(withhold_until, dict):
        return {
            "subject": str(withhold_until.get("subject", "") or "").strip(),
            "reveal_phase": str(withhold_until.get("reveal_phase", "") or "").strip(),
            "reveal_scene_id": str(withhold_until.get("reveal_scene_id", "") or "").strip() or None,
            "surrogate_label": str(withhold_until.get("surrogate_label", "") or "").strip(),
        }
    return None


def _build_field_semantics_payload() -> dict[str, Any]:
    return {
        "must_land_facts": {
            "required": "Load-bearing facts the scene must land to succeed.",
            "strongly_preferred": "Important facts to include if the scene has room after the required tier lands.",
            "if_room": "Contextual facts that may be omitted under length pressure.",
        },
        "withhold_until": {
            "subject": "What is being withheld from the listener.",
            "reveal_scene_id": "Optional later scene where the withheld subject may first be revealed.",
            "reveal_phase": "Scene phase where the reveal should land: open, pivot, or close.",
            "surrogate_label": "Optional temporary label to use before the reveal.",
        },
        "word_count_priority": {
            "default": "Use the widened default scene range.",
            "tight": "Use the narrower scene range only when pacing discipline matters more than breathing room.",
        },
    }


def _resolve_authorial_passages(
    *,
    architecture: EpisodeArchitecture,
    plan: EpisodePlan | EpisodePlanDraft,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    scene_authorial_passages_by_scene_id: dict[str, list[dict[str, Any]]] = {
        scene.scene_id: [] for scene in plan.scene_cards
    }
    section_authorial_passages_by_section_id: dict[str, list[dict[str, Any]]] = {}
    scenes_by_section_id: dict[str, list[SceneCard | SceneCardDraft]] = {}
    for scene in plan.scene_cards:
        scenes_by_section_id.setdefault(scene.section_id, []).append(scene)

    for section in architecture.sections:
        section_scene_cards = scenes_by_section_id.get(section.section_id, [])
        section_authorial_by_id = {
            passage.authorial_passage_id: passage for passage in section.authorial_passages
        }
        authorial_scene_id_by_passage_id: dict[str, str] = {}
        for scene in section_scene_cards:
            for authorial_passage_id in scene.authorial_passage_ids:
                passage = section_authorial_by_id.get(authorial_passage_id)
                if passage is None:
                    continue
                scene_payload = passage.model_dump(mode="json")
                scene_payload["scene_id"] = scene.scene_id
                scene_authorial_passages_by_scene_id.setdefault(scene.scene_id, []).append(
                    scene_payload
                )
                authorial_scene_id_by_passage_id[authorial_passage_id] = scene.scene_id
        section_authorial_passages_by_section_id[section.section_id] = []
        for passage in section.authorial_passages:
            payload = passage.model_dump(mode="json")
            payload["scene_id"] = authorial_scene_id_by_passage_id.get(passage.authorial_passage_id)
            section_authorial_passages_by_section_id[section.section_id].append(payload)

    return (
        scene_authorial_passages_by_scene_id,
        section_authorial_passages_by_section_id,
    )


def _build_writer_scene_brief(scene_payload: dict[str, Any]) -> dict[str, Any]:
    actors = []
    for actor in list(scene_payload.get("actors", []) or []):
        if not isinstance(actor, dict):
            continue
        actors.append(
            {
                "name": actor.get("name", ""),
                "actor_id": actor.get("actor_id"),
                "affiliation": actor.get("affiliation"),
                "presence": actor.get("presence", "secondary"),
                "explanation_stage": actor.get("explanation_stage"),
                "background_depth": actor.get("background_depth"),
                "role_label": actor.get("role_label", ""),
                "source_primitive_ids": list(actor.get("source_primitive_ids", []) or []),
                "source_passage_ids": list(actor.get("source_passage_ids", []) or []),
                "intro_facts": list(actor.get("intro_facts", []) or []),
                "why_now": actor.get("why_now", ""),
                "preferred_plain_gloss": actor.get("preferred_plain_gloss", ""),
            }
        )

    beat_change = str(scene_payload.get("beat_change", "") or "").strip()
    must_land_facts = _normalize_scene_fact_tiers(scene_payload.get("must_land_facts"))
    why_now = beat_change or _first_required_scene_fact(must_land_facts)
    return {
        "scene_id": scene_payload.get("scene_id"),
        "section_id": scene_payload.get("section_id"),
        "title": scene_payload.get("title", ""),
        "scene_role": scene_payload.get("scene_role", ""),
        "scene_job": scene_payload.get("scene_job", ""),
        "entry_image": scene_payload.get("entry_image", ""),
        "observable_detail": scene_payload.get("observable_detail", ""),
        "audible_detail": scene_payload.get("audible_detail", ""),
        "beat_change": beat_change,
        "must_land_facts": must_land_facts,
        "why_now": why_now,
        "timeframe": scene_payload.get("timeframe"),
        "location": scene_payload.get("location"),
        "actors": actors,
        "primitive_ids": list(scene_payload.get("primitive_ids", []) or []),
        "passage_ids": list(scene_payload.get("passage_ids", []) or []),
        "authorial_passage_ids": list(scene_payload.get("authorial_passage_ids", []) or []),
        "authorial_passages": list(scene_payload.get("authorial_passages", []) or []),
        "word_count_priority": scene_payload.get(
            "word_count_priority", WordCountPriority.DEFAULT.value
        ),
        "target_word_count_lower": int(scene_payload.get("target_word_count_lower", 0) or 0),
        "target_word_count_higher": int(scene_payload.get("target_word_count_higher", 0) or 0),
        "withhold_until": _withhold_until_payload(scene_payload.get("withhold_until")),
        "host_moves": scene_payload.get("host_moves")
        or {
            "open": [],
            "pivot": [],
            "close": [],
        },
    }


def _nonempty_runtime_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _compact_runtime_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    compacted: list[str] = []
    for value in values:
        text = _nonempty_runtime_text(value)
        if text:
            compacted.append(text)
    return compacted


def _primitive_substrate_fields_payload(
    primitive: SynthesisPrimitiveBase,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    substrate = primitive.substrate
    if substrate == PrimitiveSubstrate.EVENTS:
        if event_type := _nonempty_runtime_text(getattr(primitive, "event_type", None)):
            payload["event_type"] = event_type
        if what_happened := _nonempty_runtime_text(getattr(primitive, "what_happened", None)):
            payload["what_happened"] = what_happened
        if event_result := _nonempty_runtime_text(getattr(primitive, "event_result", None)):
            payload["event_result"] = event_result
        return payload
    if substrate == PrimitiveSubstrate.ACTS:
        if act_type := _nonempty_runtime_text(getattr(primitive, "act_type", None)):
            payload["act_type"] = act_type
        if acting_subject := _nonempty_runtime_text(getattr(primitive, "acting_subject", None)):
            payload["acting_subject"] = acting_subject
        if act_summary := _nonempty_runtime_text(getattr(primitive, "act_summary", None)):
            payload["act_summary"] = act_summary
        if immediate_result := _nonempty_runtime_text(getattr(primitive, "immediate_result", None)):
            payload["immediate_result"] = immediate_result
        return payload
    if substrate == PrimitiveSubstrate.ACTOR_PORTRAITS:
        if focus_actor_id := _nonempty_runtime_text(getattr(primitive, "focus_actor_id", None)):
            payload["focus_actor_id"] = focus_actor_id
        if actor_label := _nonempty_runtime_text(getattr(primitive, "actor_label", None)):
            payload["actor_label"] = actor_label
        if goal_or_project := _nonempty_runtime_text(getattr(primitive, "goal_or_project", None)):
            payload["goal_or_project"] = goal_or_project
        if stakes_or_fears := _nonempty_runtime_text(getattr(primitive, "stakes_or_fears", None)):
            payload["stakes_or_fears"] = stakes_or_fears
        if operating_pressure := _nonempty_runtime_text(
            getattr(primitive, "operating_pressure", None)
        ):
            payload["operating_pressure"] = operating_pressure
        return payload
    if substrate == PrimitiveSubstrate.MECHANISMS:
        if mechanism_name := _nonempty_runtime_text(getattr(primitive, "mechanism_name", None)):
            payload["mechanism_name"] = mechanism_name
        if operating_chain := _compact_runtime_string_list(
            getattr(primitive, "operating_chain", None)
        ):
            payload["operating_chain"] = operating_chain
        if inputs := _compact_runtime_string_list(getattr(primitive, "inputs", None)):
            payload["inputs"] = inputs
        if outputs := _compact_runtime_string_list(getattr(primitive, "outputs", None)):
            payload["outputs"] = outputs
        if failure_mode := _nonempty_runtime_text(getattr(primitive, "failure_mode", None)):
            payload["failure_mode"] = failure_mode
        return payload
    if substrate == PrimitiveSubstrate.CONDITIONS:
        if condition_type := _nonempty_runtime_text(getattr(primitive, "condition_type", None)):
            payload["condition_type"] = condition_type
        if condition_summary := _nonempty_runtime_text(
            getattr(primitive, "condition_summary", None)
        ):
            payload["condition_summary"] = condition_summary
        if active_tension := _nonempty_runtime_text(getattr(primitive, "active_tension", None)):
            payload["active_tension"] = active_tension
        return payload
    if substrate == PrimitiveSubstrate.ARTIFACTS:
        if artifact_type := _nonempty_runtime_text(getattr(primitive, "artifact_type", None)):
            payload["artifact_type"] = artifact_type
        if artifact_label := _nonempty_runtime_text(getattr(primitive, "artifact_label", None)):
            payload["artifact_label"] = artifact_label
        if artifact_detail := _nonempty_runtime_text(getattr(primitive, "artifact_detail", None)):
            payload["artifact_detail"] = artifact_detail
        return payload
    if substrate == PrimitiveSubstrate.READINGS:
        if reading_type := _nonempty_runtime_text(getattr(primitive, "reading_type", None)):
            payload["reading_type"] = reading_type
        if subject_of_reading := _nonempty_runtime_text(
            getattr(primitive, "subject_of_reading", None)
        ):
            payload["subject_of_reading"] = subject_of_reading
        if attributed_to := _nonempty_runtime_text(getattr(primitive, "attributed_to", None)):
            payload["attributed_to"] = attributed_to
        if reading_summary := _nonempty_runtime_text(getattr(primitive, "reading_summary", None)):
            payload["reading_summary"] = reading_summary
        return payload
    return payload


def _primitive_substrate_text_fragments(
    primitive: SynthesisPrimitiveBase,
) -> list[str]:
    fragments: list[str] = []
    for value in _primitive_substrate_fields_payload(primitive).values():
        if isinstance(value, str):
            fragments.append(value)
            continue
        if isinstance(value, list):
            fragments.extend(_compact_runtime_string_list(value))
    return fragments


def _build_scene_primitive_briefs(
    *,
    scene_cards: list[SceneCard],
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
) -> dict[str, list[dict[str, Any]]]:
    briefs: dict[str, list[dict[str, Any]]] = {}
    for scene in scene_cards:
        scene_briefs: list[dict[str, Any]] = []
        for primitive_id in scene.primitive_ids:
            primitive = primitive_lookup.get(primitive_id)
            if primitive is None:
                continue
            scene_briefs.append(primitive.model_dump(mode="json"))
        briefs[scene.scene_id] = scene_briefs
    return briefs


def _build_scene_excerpt_briefs(
    *,
    scene_cards: list[SceneCard],
    excerpt_by_id: dict[str, ExcerptRecord],
    recall_excerpt_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per-scene briefs of the excerpts attached via ``scene.excerpt_ids``.

    ``recall_excerpt_ids`` is the host episode's ``EpisodeSpine.recall_excerpt_ids``
    set: any excerpt id appearing there is a cross-episode callback and gets
    ``is_recall: true`` in its brief so the writer frames it as a return.
    """

    recall_set = recall_excerpt_ids or set()
    briefs: dict[str, list[dict[str, Any]]] = {}
    for scene in scene_cards:
        scene_briefs: list[dict[str, Any]] = []
        for excerpt_id in scene.excerpt_ids:
            excerpt = excerpt_by_id.get(excerpt_id)
            if excerpt is None:
                continue
            scene_briefs.append(
                {
                    "id": excerpt.id,
                    "excerpt_type": excerpt.excerpt_type,
                    "title": excerpt.title,
                    "speaker": excerpt.speaker,
                    "verbatim_excerpt": excerpt.verbatim_excerpt,
                    "plain_gloss": excerpt.plain_gloss,
                    "passage_ids": list(excerpt.passage_ids),
                    "is_recall": excerpt_id in recall_set,
                }
            )
        if scene_briefs:
            briefs[scene.scene_id] = scene_briefs
    return briefs


def _build_host_policy_payload(
    narrator_profile: SeriesNarratorProfile,
    narrative_state_pre: NarrativeState | None = None,
    narrative_state_post: NarrativeState | None = None,
) -> dict[str, Any]:
    allowed_moves = effective_narrator_allowed_moves(narrator_profile.allowed_moves)
    has_persona = narrator_profile.persona is not None
    # Gate persona_aside on an authored persona: a default profile keeps the move
    # in its allowed list for schema stability, but we only license it at runtime
    # when the enrichment stage actually produced a persona.
    if not has_persona:
        allowed_moves = [move for move in allowed_moves if move != "persona_aside"]
    persona_singular_moves = ["persona_aside"] if has_persona else []
    host_posture = (
        narrative_state_post.host.confidence_posture
        if narrative_state_post is not None
        else (
            narrative_state_pre.host.confidence_posture
            if narrative_state_pre is not None
            else "mixed"
        )
    )
    return {
        "presence_mode": narrator_profile.presence_mode,
        "baseline_tone": narrator_profile.baseline_tone,
        "spoken_style_contract": narrator_profile.spoken_style_contract,
        "allowed_moves": allowed_moves,
        "forbidden_moves": list(narrator_profile.forbidden_moves),
        "persona": (
            narrator_profile.persona.model_dump(mode="json")
            if narrator_profile.persona is not None
            else None
        ),
        "target_persona_asides_per_episode": (
            narrator_profile.target_persona_asides_per_episode if has_persona else 0
        ),
        "target_full_phase_scene_coverage_min": (
            narrator_profile.target_full_phase_scene_coverage_min
        ),
        "target_full_phase_scene_coverage_target": (
            narrator_profile.target_full_phase_scene_coverage_target
        ),
        "pronoun_policy": {
            "allow_first_person_singular": True,
            "first_person_singular_allowed_for": [
                "uncertainty",
                "revision",
                "surprise",
                "closing_reflection",
                *persona_singular_moves,
            ],
            "allow_first_person_plural_only_for": [
                "handoff",
                "callback",
                "closing",
                "clarify",
                "reorientation",
                "evaluation",
            ],
            "allow_second_person_guidance": True,
        },
        "authorial_policy": {
            "analysis_mode": narrator_profile.analysis_mode,
            "analysis_density": narrator_profile.analysis_density,
            "quote_gloss_preference": narrator_profile.quote_gloss_preference,
            "clarifier_tolerance": narrator_profile.clarifier_tolerance,
            "comparative_aside_tolerance": COMPARATIVE_ASIDE_TOLERANCE,
            "wit_ceiling": narrator_profile.wit_ceiling,
            "host_confidence_posture": host_posture,
            "target_authorial_passages_per_episode": (
                narrator_profile.target_authorial_passages_per_episode
            ),
        },
    }


def _listener_agenda(strategy_episode: StrategyEpisode) -> Any:
    return strategy_episode.narrative_agenda.listener


def _host_agenda(strategy_episode: StrategyEpisode) -> Any:
    return strategy_episode.narrative_agenda.host


def _strategy_episode_question_texts(strategy_episode: StrategyEpisode) -> list[str]:
    question_texts = [
        move.text
        for move in _listener_agenda(strategy_episode).question_moves
        if move.action in {"open", "advance", "reframe"} and move.text
    ]
    if question_texts:
        return question_texts
    return [
        str(question or "").strip()
        for question in strategy_episode.unresolved_questions
        if str(question or "").strip()
    ]


@dataclass(frozen=True)
class _EpisodePlanningRuntime:
    project_metadata: dict[str, Any]
    host_policy: dict[str, Any]
    continuity_contract_pre: dict[str, Any]
    episode_synthesis_map_payload: dict[str, Any]
    primitive_ids: list[str]
    episode_actor_metadata: ActorMetadata
    compact_episode_actor_metadata: dict[str, Any]
    available_passages: list[dict[str, Any]]
    episode_payload: dict[str, Any]
    episode_excerpts: list[dict[str, Any]] = field(default_factory=list)


def _build_episode_planning_project_metadata(
    project: ThematicProject,
) -> dict[str, Any]:
    return {
        "podcast_mode": project.config.podcast_mode.value,
        "theme": project.theme,
        "sub_themes": project.sub_themes,
        "book_count": len(project.books),
        "books": [
            {"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books
        ],
        "scene_card_target_min": project.config.scene_card_target_min,
        "scene_card_target_max": project.config.scene_card_target_max,
        "min_episode_minutes": project.config.min_episode_minutes,
        "max_episode_minutes": project.config.max_episode_minutes,
    }


def _build_episode_planning_runtime(
    *,
    project: ThematicProject,
    strategy: NarrativeStrategy,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    synthesis_map: SynthesisMap,
    corpus: ThematicCorpus,
    actor_metadata: ActorMetadata,
    narrative_state_pre: NarrativeState | None,
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
    passage_lookup: dict[str, ExtractedPassage],
    excerpt_by_id: dict[str, ExcerptRecord] | None = None,
) -> _EpisodePlanningRuntime:
    excerpt_by_id = excerpt_by_id or {}
    continuity_contract_pre = (
        _build_continuity_contract(
            narrative_state=narrative_state_pre,
            episode_number=strategy_episode.episode_number,
            phase="pre",
        )
        if narrative_state_pre is not None
        else {}
    )
    host_policy = _build_host_policy_payload(
        strategy.narrator_profile,
        narrative_state_pre=narrative_state_pre,
    )
    primitive_ids_by_role = _filter_primitive_ids_by_architecture(
        strategy_episode,
        architecture,
    )
    episode_synthesis_map_payload, primitive_ids = _build_episode_synthesis_map_payload(
        synthesis_map,
        primitive_ids_by_role,
    )
    episode_actor_ids = _collect_episode_actor_ids(
        strategy_episode=strategy_episode,
        primitive_ids=primitive_ids,
        primitive_lookup=primitive_lookup,
    )
    episode_actor_metadata = select_actor_metadata_subset(
        actor_metadata,
        episode_actor_ids,
    )
    passage_ids: list[str] = []
    seen_passage_ids: set[str] = set()
    passage_keep_fraction_by_id: dict[str, float] = {}
    planning_passage_refs = _build_episode_planning_passage_refs(
        primitive_ids_by_role=primitive_ids_by_role,
        primitive_lookup=primitive_lookup,
    )
    for passage_ref in planning_passage_refs:
        passage_id = passage_ref["passage_id"]
        if passage_id not in passage_lookup:
            continue
        keep_fraction = _planning_passage_keep_fraction(
            passage_ref["episode_role"],
            passage_ref["passage_kind"],
        )
        passage_keep_fraction_by_id[passage_id] = max(
            keep_fraction,
            passage_keep_fraction_by_id.get(passage_id, 0.0),
        )
        if passage_id in seen_passage_ids:
            continue
        seen_passage_ids.add(passage_id)
        passage_ids.append(passage_id)
    episode_excerpts = _build_episode_excerpt_payload(
        primitive_ids_by_role.get("excerpt", []),
        excerpt_by_id,
    )
    # Ground every voiced excerpt: pull its source passages into the writer's
    # available-passage set so the verbatim text can be read on the page.
    for excerpt_item in episode_excerpts:
        for passage_id in excerpt_item.get("passage_ids", []):
            if passage_id in passage_lookup and passage_id not in seen_passage_ids:
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
        strategy_episode.title,
        strategy_episode.episode_spine.listener_problem,
        strategy_episode.thematic_focus,
    ]
    episode_query_parts.extend(_strategy_episode_question_texts(strategy_episode))
    episode_query_parts.extend(
        item.get("label", "")
        for item in continuity_contract_pre.get("recap_items", [])
        if isinstance(item, dict)
    )
    episode_query_parts.extend(
        item.get("label", "")
        for item in continuity_contract_pre.get("must_surface_early", [])
        if isinstance(item, dict)
    )
    episode_query_parts.extend(continuity_contract_pre.get("open_question_texts", []))
    episode_query_parts.extend(continuity_contract_pre.get("open_memory_thread_labels", []))
    episode_query_parts.extend(continuity_contract_pre.get("host_open_pressures", []))
    episode_query_text = " ".join(part for part in episode_query_parts if part).strip()
    passage_query_text_by_id = _build_episode_planning_passage_query_texts(
        episode_query_text=episode_query_text,
        passage_refs=planning_passage_refs,
        primitive_lookup=primitive_lookup,
    )
    _trim_candidate_texts_by_bm25_query_text(
        episode_query_text,
        available_passages,
        keep_fraction=0.5,
        keep_fraction_by_passage_id=passage_keep_fraction_by_id,
        query_text_by_passage_id=passage_query_text_by_id,
    )
    return _EpisodePlanningRuntime(
        project_metadata=_build_episode_planning_project_metadata(project),
        host_policy=host_policy,
        continuity_contract_pre=continuity_contract_pre,
        episode_synthesis_map_payload=episode_synthesis_map_payload,
        primitive_ids=primitive_ids,
        episode_actor_metadata=episode_actor_metadata,
        compact_episode_actor_metadata=compact_actor_metadata(episode_actor_metadata),
        available_passages=available_passages,
        episode_payload=architecture.model_dump(mode="json"),
        episode_excerpts=episode_excerpts,
    )


def _postcheck_episode_plan_draft(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    plan_draft: EpisodePlanDraft,
    episode_actor_metadata: ActorMetadata,
    narrator_profile: SeriesNarratorProfile,
    continuity_contract_pre: dict[str, Any],
) -> tuple[EpisodePlanDraft, dict[str, Any], list[str]]:
    plan_draft = _validate_plan_transition(
        strategy_episode=strategy_episode,
        architecture=architecture,
        plan=plan_draft,
    )
    _validate_continuity_recap_requirement(
        episode_number=strategy_episode.episode_number,
        plan=plan_draft,
        continuity_contract_pre=continuity_contract_pre,
    )
    plan_draft, actor_link_metrics = clean_scene_actor_links(
        plan_draft,
        episode_actor_metadata,
        strategy_episode.actor_arc_directives,
    )
    actor_explanation_warnings = _validate_actor_explanation_scene_links(
        architecture=architecture,
        plan=plan_draft,
    )
    retry_host_moves_diagnostics, _retry_host_move_warnings = _build_host_move_plan_diagnostics(
        scene_cards=plan_draft.scene_cards,
        architecture=architecture,
        narrator_profile=narrator_profile,
    )
    if retry_host_moves_diagnostics["disallowed_move_scene_ids"]:
        raise ComplianceViolationError(
            "Episode plan used host move types that are not allowed by the narrator policy.",
            data={
                "issue": "host_move_allowed_move_mismatch",
                "episode_number": architecture.episode_number,
                "scene_ids": retry_host_moves_diagnostics["disallowed_move_scene_ids"],
                "instruction": "Use only move_type values allowed by host_policy.allowed_moves.",
            },
        )
    return plan_draft, actor_link_metrics, actor_explanation_warnings


def _build_initial_narrative_state(project_id: str) -> NarrativeState:
    return NarrativeState(project_id=project_id, next_episode_number=1)


def _build_episode_explanation_registry_payload(
    *,
    strategy_episode: StrategyEpisode | None,
    architecture: EpisodeArchitecture | None,
    series_explanation_registry: list[Any] | None,
) -> list[dict[str, Any]]:
    if not series_explanation_registry:
        return []
    referenced_item_ids: set[str] = set()
    if strategy_episode is not None:
        listener_agenda = _listener_agenda(strategy_episode)
        referenced_item_ids.update(
            item_id
            for item_id in (
                listener_agenda.introduce_explanation_item_ids
                or strategy_episode.authorial_contract.introduce_explanation_item_ids
            )
            if item_id
        )
        referenced_item_ids.update(
            item_id
            for item_id in (
                listener_agenda.remind_explanation_item_ids
                or strategy_episode.authorial_contract.remind_explanation_item_ids
            )
            if item_id
        )
    if architecture is not None:
        for section in architecture.sections:
            referenced_item_ids.update(
                item_id
                for item_id in (explanation.item_id for explanation in section.term_explanations)
                if item_id
            )
    filtered_items: list[dict[str, Any]] = []
    for item in series_explanation_registry:
        item_id = getattr(item, "item_id", None)
        if item_id not in referenced_item_ids:
            continue
        if hasattr(item, "model_dump"):
            filtered_items.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            filtered_items.append(dict(item))
    return filtered_items


def _episode_actor_gloss(actor: Any) -> str:
    text = str(getattr(actor, "description", "") or "").strip()
    if not text:
        return str(getattr(actor, "actor_type", "actor") or "actor").replace("_", " ")
    for delimiter in (".", ";", ":"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0].strip()
            break
    words = text.split()
    if len(words) > 18:
        text = " ".join(words[:18]).rstrip(",")
    return text


def _build_episode_actor_explanation_registry_payload(
    *,
    strategy_episode: StrategyEpisode | None,
    architecture: EpisodeArchitecture | None,
    series_actor_explanation_registry: list[SeriesActorExplanationItem] | None,
    actor_metadata: ActorMetadata | None = None,
) -> list[dict[str, Any]]:
    if not series_actor_explanation_registry:
        return []
    referenced_actor_ids: set[str] = set()
    if strategy_episode is not None:
        listener_agenda = _listener_agenda(strategy_episode)
        referenced_actor_ids.update(
            actor_id
            for actor_id in (
                listener_agenda.introduce_actor_ids
                or strategy_episode.authorial_contract.introduce_actor_ids
            )
            if actor_id
        )
        referenced_actor_ids.update(
            actor_id
            for actor_id in (
                listener_agenda.remind_actor_ids
                or strategy_episode.authorial_contract.remind_actor_ids
            )
            if actor_id
        )
    if architecture is not None:
        for section in architecture.sections:
            referenced_actor_ids.update(
                explanation.actor_id
                for explanation in section.actor_explanations
                if explanation.actor_id
            )
    if not referenced_actor_ids:
        return []
    actor_by_id = {
        actor.actor_id: actor for actor in (actor_metadata.actors if actor_metadata else [])
    }
    filtered_items: list[dict[str, Any]] = []
    for item in series_actor_explanation_registry:
        if item.actor_id not in referenced_actor_ids:
            continue
        payload = item.model_dump(mode="json")
        if not str(payload.get("preferred_plain_gloss", "") or "").strip():
            actor = actor_by_id.get(item.actor_id)
            if actor is not None:
                payload["preferred_plain_gloss"] = _episode_actor_gloss(actor)
        filtered_items.append(payload)
    return filtered_items


def _collect_episode_actor_ids(
    *,
    strategy_episode: StrategyEpisode,
    primitive_ids: list[str],
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
) -> set[str]:
    actor_ids = {
        actor.actor_id for actor in strategy_episode.actor_arc_directives if actor.actor_id
    }
    actor_ids.update(
        actor_id
        for actor_id in (
            _listener_agenda(strategy_episode).introduce_actor_ids
            or strategy_episode.authorial_contract.introduce_actor_ids
        )
        if actor_id
    )
    actor_ids.update(
        actor_id
        for actor_id in (
            _listener_agenda(strategy_episode).remind_actor_ids
            or strategy_episode.authorial_contract.remind_actor_ids
        )
        if actor_id
    )
    selected_primitives = [
        primitive_lookup[primitive_id]
        for primitive_id in primitive_ids
        if primitive_id in primitive_lookup
    ]
    actor_ids.update(collect_actor_ids_for_primitives(selected_primitives))
    return {actor_id for actor_id in actor_ids if actor_id}


def _build_host_moves_by_section(
    plan: EpisodePlan | None,
) -> dict[str, list[dict[str, Any]]]:
    host_moves_by_section: dict[str, list[dict[str, Any]]] = {}
    if plan is None:
        return host_moves_by_section
    for scene in plan.scene_cards:
        host_moves_by_section.setdefault(scene.section_id, []).append(
            {
                "scene_id": scene.scene_id,
                "host_moves": scene.host_moves.model_dump(mode="json"),
            }
        )
    return host_moves_by_section


def _inherited_pressure_by_section(
    architecture: EpisodeArchitecture | None,
) -> dict[str, str]:
    """Map each section to the prior section's ``what_remains_live`` (carry-over)."""

    inherited: dict[str, str] = {}
    if architecture is None:
        return inherited
    prior_live = ""
    for section in architecture.sections:
        inherited[section.section_id] = prior_live
        prior_live = section.section_progression.what_remains_live
    return inherited


def _build_window_passages(
    *,
    window_scene_cards: list[SceneCard],
    passage_lookup: dict[str, ExtractedPassage],
    excerpt_by_id: dict[str, ExcerptRecord] | None = None,
) -> list[dict[str, Any]]:
    excerpt_by_id = excerpt_by_id or {}
    ordered_passage_ids: list[str] = []
    seen_passage_ids: set[str] = set()
    for scene in window_scene_cards:
        scene_passage_ids = list(scene.passage_ids)
        # Pull the source passages behind any voiced excerpt into the window so the
        # verbatim text is grounded on the page.
        for excerpt_id in scene.excerpt_ids:
            excerpt = excerpt_by_id.get(excerpt_id)
            if excerpt is not None:
                scene_passage_ids.extend(excerpt.passage_ids)
        for passage_id in scene_passage_ids:
            if passage_id in seen_passage_ids or passage_id not in passage_lookup:
                continue
            seen_passage_ids.add(passage_id)
            ordered_passage_ids.append(passage_id)

    return [
        {
            "passage_id": passage_lookup[passage_id].passage_id,
            "book_id": passage_lookup[passage_id].book_id,
            "text": _resolve_writing_passage_text(passage_lookup[passage_id]),
            "chapter_ref": passage_lookup[passage_id].chapter_ref,
        }
        for passage_id in ordered_passage_ids
    ]


def _build_window_actor_metadata(
    *,
    window_scene_cards: list[SceneCard],
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    actor_metadata: ActorMetadata,
) -> dict[str, Any]:
    actor_ids = {
        actor.actor_id for scene in window_scene_cards for actor in scene.actors if actor.actor_id
    }
    window_section_ids = {scene.section_id for scene in window_scene_cards}
    actor_ids.update(
        explanation.actor_id
        for section in architecture.sections
        if section.section_id in window_section_ids
        for explanation in section.actor_explanations
        if explanation.actor_id
    )
    if not actor_ids:
        actor_ids.update(
            actor.actor_id for actor in strategy_episode.actor_arc_directives if actor.actor_id
        )
    window_actor_metadata = select_actor_metadata_subset(actor_metadata, actor_ids)
    return compact_actor_metadata(window_actor_metadata)


def _build_prior_window_continuity(
    *,
    completed_scene_cards: list[SceneCard],
    completed_scene_outputs: list[dict[str, Any]],
    strategy_episode: StrategyEpisode,
    remaining_scene_ids: set[str],
) -> dict[str, Any]:
    last_scene = completed_scene_cards[-1]
    continuity_text = "\n\n".join(
        str(scene_output.get("text", "") or "") for scene_output in completed_scene_outputs
    )
    tail_excerpt = _extract_previous_spoken_tail(continuity_text) or ""

    live_unresolved_questions: list[str] = []
    seen_questions: set[str] = set()
    for question in _strategy_episode_question_texts(strategy_episode):
        normalized_question = str(question or "").strip()
        if not normalized_question or normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        live_unresolved_questions.append(normalized_question)

    carry_forward_threads: list[str] = []
    if live_unresolved_questions:
        carry_forward_threads.append(
            f"Keep this tension live without restating it explicitly: {live_unresolved_questions[-1]}"
        )
    withheld_scene_id = (
        last_scene.withhold_until.reveal_scene_id if last_scene.withhold_until is not None else None
    )
    if withheld_scene_id and withheld_scene_id in remaining_scene_ids:
        carry_forward_threads.append(
            f"Do not reveal the withheld material before {withheld_scene_id}."
        )
    if not carry_forward_threads:
        carry_forward_threads.append(
            "Continue from the completed handoff without re-narrating the previous window."
        )

    return {
        "completed_scene_count": len(completed_scene_cards),
        "completed_scene_ids": [scene.scene_id for scene in completed_scene_cards],
        "last_completed_scene": {
            "scene_id": last_scene.scene_id,
            "section_id": last_scene.section_id,
            "scene_role": last_scene.scene_role,
            "scene_job": last_scene.scene_job,
            "spine_relation": "",
            "withhold_until": _withhold_until_payload(last_scene.withhold_until),
            "intended_move": last_scene.beat_change,
        },
        "live_unresolved_questions": live_unresolved_questions,
        "carry_forward_threads": carry_forward_threads[:3],
        "tail_excerpt": tail_excerpt,
    }


def _build_style_audit_sections_payload(
    *,
    script: EpisodeScript,
    architecture: EpisodeArchitecture,
    plan: EpisodePlan | None = None,
    strategy_episode: StrategyEpisode | None = None,
    series_carryover_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the style_audit section payload and the episode-level lint flags.

    Returns ``(sections, lint_flags)``. ``lint_flags`` carries per-section
    tic/thesis-overlap/abstract-noun signals computed in pure Python; each
    section dict in ``sections`` also has its own slice as ``lint_flags`` so
    the style_audit prompt can act on it locally.
    """

    section_meta_by_id = {section.section_id: section for section in architecture.sections}
    section_progression_by_id = {
        section.section_id: section.section_progression.stage.value
        for section in architecture.sections
    }
    section_authorial_passages_by_section_id: dict[str, list[dict[str, Any]]] = {}
    if plan is not None:
        _, section_authorial_passages_by_section_id = _resolve_authorial_passages(
            architecture=architecture,
            plan=plan,
        )
    host_moves_by_section = _build_host_moves_by_section(plan)
    inherited_pressure_by_id = _inherited_pressure_by_section(architecture)
    scene_cards_by_section: dict[str, list[SceneCard]] = {}
    if plan is not None:
        for scene in plan.scene_cards:
            scene_cards_by_section.setdefault(scene.section_id, []).append(scene)

    prose_section_dicts = [
        {"section_id": ps.section_id, "text": ps.text} for ps in script.prose_sections
    ]
    spine = strategy_episode.episode_spine if strategy_episode is not None else None
    from podcast_agent.pipeline.tic_families import detect_tic_hits
    from podcast_agent.pipeline.text_embeddings import get_text_embedder
    from podcast_agent.pipeline.text_utils import split_sentences as _split_sentences

    _audit_embedder = get_text_embedder()

    def _audit_semantic_detector(text: str, section_id: str):
        return detect_tic_hits(
            _split_sentences(text),
            section_id=section_id,
            embedder=_audit_embedder,
        )

    lint_flags = compute_style_audit_lint_flags(
        prose_section_dicts,
        spine_episode_answer=(spine.episode_answer if spine else ""),
        spine_pressure_line=(spine.pressure_line if spine else ""),
        section_progression_by_id=section_progression_by_id,
        semantic_detector=_audit_semantic_detector,
        series_carryover_counts=series_carryover_counts or {},
    )
    by_section_flags = lint_flags["by_section"]

    payload_sections: list[dict[str, Any]] = []
    for prose_section in script.prose_sections:
        meta = section_meta_by_id.get(prose_section.section_id)
        if meta is None:
            continue
        section_scene_cards = scene_cards_by_section.get(prose_section.section_id, [])
        # Aggregate binding must-land facts and citations across the section's
        # scene cards so style_audit has the preservation contract inline with
        # the prose. The aggregation lives in style_audit_linting so the
        # post-audit fact-coverage diagnostic shares one source of truth.
        from podcast_agent.pipeline.style_audit_linting import (
            aggregate_section_must_land_facts,
        )

        (
            section_required_facts,
            section_strongly_preferred_facts,
        ) = aggregate_section_must_land_facts(section_scene_cards)
        section_citations = [
            citation.model_dump(mode="json") for citation in prose_section.citations
        ]
        payload_sections.append(
            {
                "section_id": prose_section.section_id,
                "purpose": meta.purpose.value,
                "open_mode": meta.open_mode,
                "anchor": meta.section_anchor,
                "section_progression": meta.section_progression.model_dump(mode="json"),
                "inherited_pressure": inherited_pressure_by_id.get(prose_section.section_id, ""),
                "scene_card_count": len(section_scene_cards),
                "projected_word_count": sum(
                    _project_scene_word_count(scene, words_per_minute=145.0)
                    for scene in section_scene_cards
                ),
                "structural_card_count": sum(
                    1 for scene in section_scene_cards if _is_structural_scene_card(scene)
                ),
                "must_land_facts": {
                    "required": section_required_facts,
                    "strongly_preferred": section_strongly_preferred_facts,
                },
                "citations": section_citations,
                "host_moves": host_moves_by_section.get(prose_section.section_id, []),
                "key_terms": list(meta.key_terms),
                "authorial_passages": section_authorial_passages_by_section_id.get(
                    prose_section.section_id,
                    [passage.model_dump(mode="json") for passage in meta.authorial_passages],
                ),
                "term_explanations": [
                    explanation.model_dump(mode="json") for explanation in meta.term_explanations
                ],
                "actor_explanations": [
                    explanation.model_dump(mode="json") for explanation in meta.actor_explanations
                ],
                "section_sonic_plan": (
                    meta.section_sonic_plan.model_dump(mode="json")
                    if meta.section_sonic_plan is not None
                    else None
                ),
                "text": prose_section.text,
                "actor_explanation_realizations": [
                    realization.model_dump(mode="json")
                    for realization in prose_section.actor_explanation_realizations
                ],
                "lint_flags": by_section_flags.get(prose_section.section_id, {}),
            }
        )
    return payload_sections, lint_flags


def _build_spoken_delivery_sections_payload(
    *,
    script: EpisodeScript,
    architecture: EpisodeArchitecture | None,
    plan: EpisodePlan | None = None,
) -> list[dict[str, Any]]:
    architecture_section_lookup = {
        section.section_id: section
        for section in (architecture.sections if architecture is not None else [])
    }
    section_authorial_passages_by_section_id: dict[str, list[dict[str, Any]]] = {}
    if architecture is not None and plan is not None:
        _, section_authorial_passages_by_section_id = _resolve_authorial_passages(
            architecture=architecture,
            plan=plan,
        )
    plan_scene_lookup: dict[str, SceneCard] = {
        scene.scene_id: scene for scene in (plan.scene_cards if plan is not None else [])
    }
    host_moves_by_section = _build_host_moves_by_section(plan)
    inherited_pressure_by_id = _inherited_pressure_by_section(architecture)
    payload_sections: list[dict[str, Any]] = []
    for prose_section in script.prose_sections:
        architecture_section = architecture_section_lookup.get(prose_section.section_id)
        scene_cues = [
            SonicCue(
                scene_id=scene.scene_id,
                scene_job=scene.scene_job.value,
                entry_image=scene.entry_image,
                observable_detail=scene.observable_detail,
                audible_detail=scene.audible_detail,
            ).model_dump(mode="json")
            for scene_id in prose_section.scene_card_ids
            if (scene := plan_scene_lookup.get(scene_id)) is not None
        ]
        payload_sections.append(
            {
                "section_id": prose_section.section_id,
                "purpose": architecture_section.purpose.value if architecture_section else "",
                "open_mode": architecture_section.open_mode
                if architecture_section
                else "scene_anchor",
                "anchor": architecture_section.section_anchor if architecture_section else "",
                "section_progression": (
                    architecture_section.section_progression.model_dump(mode="json")
                    if architecture_section is not None
                    else None
                ),
                "inherited_pressure": inherited_pressure_by_id.get(prose_section.section_id, ""),
                "section_sonic_plan": (
                    architecture_section.section_sonic_plan.model_dump(mode="json")
                    if architecture_section is not None
                    and architecture_section.section_sonic_plan is not None
                    else prose_section.section_sonic_plan.model_dump(mode="json")
                    if prose_section.section_sonic_plan is not None
                    else None
                ),
                "movement_goal": prose_section.movement_goal,
                "scene_card_ids": list(prose_section.scene_card_ids),
                "scene_cues": scene_cues,
                "host_moves": host_moves_by_section.get(prose_section.section_id, []),
                "key_terms": (
                    list(architecture_section.key_terms) if architecture_section is not None else []
                ),
                "authorial_passages": (
                    section_authorial_passages_by_section_id.get(
                        prose_section.section_id,
                        [
                            passage.model_dump(mode="json")
                            for passage in architecture_section.authorial_passages
                        ],
                    )
                    if architecture_section is not None
                    else []
                ),
                "term_explanations": (
                    [
                        explanation.model_dump(mode="json")
                        for explanation in architecture_section.term_explanations
                    ]
                    if architecture_section is not None
                    else []
                ),
                "actor_explanations": (
                    [
                        explanation.model_dump(mode="json")
                        for explanation in architecture_section.actor_explanations
                    ]
                    if architecture_section is not None
                    else []
                ),
                "text": prose_section.text,
                "actor_explanation_realizations": [
                    realization.model_dump(mode="json")
                    for realization in prose_section.actor_explanation_realizations
                ],
            }
        )
    return payload_sections


def _build_script_sections_payload(
    *,
    script: EpisodeScript,
    architecture: EpisodeArchitecture | None = None,
) -> list[dict[str, Any]]:
    architecture_section_lookup = {
        section.section_id: section
        for section in (architecture.sections if architecture is not None else [])
    }
    payload_sections: list[dict[str, Any]] = []
    for prose_section in script.prose_sections:
        architecture_section = architecture_section_lookup.get(prose_section.section_id)
        payload_sections.append(
            {
                **prose_section.model_dump(mode="json"),
                "section_sonic_plan": (
                    architecture_section.section_sonic_plan.model_dump(mode="json")
                    if architecture_section is not None
                    and architecture_section.section_sonic_plan is not None
                    else prose_section.section_sonic_plan.model_dump(mode="json")
                    if prose_section.section_sonic_plan is not None
                    else None
                ),
                "key_terms": (
                    list(architecture_section.key_terms) if architecture_section is not None else []
                ),
                "authorial_passages": (
                    [
                        passage.model_dump(mode="json")
                        for passage in architecture_section.authorial_passages
                    ]
                    if architecture_section is not None
                    else []
                ),
                "term_explanations": (
                    [
                        explanation.model_dump(mode="json")
                        for explanation in architecture_section.term_explanations
                    ]
                    if architecture_section is not None
                    else []
                ),
                "actor_explanations": (
                    [
                        explanation.model_dump(mode="json")
                        for explanation in architecture_section.actor_explanations
                    ]
                    if architecture_section is not None
                    else []
                ),
            }
        )
    return payload_sections


def _apply_style_audit_to_script(
    *,
    script: EpisodeScript,
    audit: StyleAuditResponse,
    spoken_words_per_minute: float,
) -> EpisodeScript:
    edits_by_section_id = {section.section_id: section for section in audit.sections}
    new_sections: list[ProseSection] = []
    for section in script.prose_sections:
        edit = edits_by_section_id.get(section.section_id)
        if edit is None:
            new_sections.append(section)
            continue
        new_sections.append(section.model_copy(update={"text": edit.edited_text}))
    updated = script.model_copy(update={"prose_sections": new_sections})
    return updated.model_copy(
        update={
            "total_word_count": _script_total_word_count(updated),
            "estimated_duration_seconds": _estimate_duration_seconds_from_words(
                _script_total_word_count(updated),
                spoken_words_per_minute,
            ),
        }
    )


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


def _bm25_query_terms(query_text: str) -> dict[str, int]:
    query_terms: dict[str, int] = {}
    for term in _tokenize(query_text):
        query_terms[term] = query_terms.get(term, 0) + 1
    return query_terms


def _bm25_idf_and_avg_len(documents: list[list[str]]) -> tuple[dict[str, float], float]:
    if not documents:
        return {}, 0.0
    df: dict[str, int] = {}
    for tokens in documents:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    total_documents = len(documents)
    idf = {
        term: math.log(1 + (total_documents - count + 0.5) / (count + 0.5))
        for term, count in df.items()
    }
    avg_len = sum(len(tokens) for tokens in documents) / total_documents
    return idf, avg_len


@dataclass(frozen=True)
class _PrimitiveEvidenceTrimProfile:
    core_keep_fraction: float
    support_keep_fraction: float
    shared_passage_keep_fraction: float


@dataclass(frozen=True)
class _ScoredSentence:
    score: float
    index: int
    sentence: str
    word_count: int


_FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE = _PrimitiveEvidenceTrimProfile(
    core_keep_fraction=0.15,
    support_keep_fraction=0.075,
    shared_passage_keep_fraction=0.25,
)


def _rank_bm25_sentences(
    sentences: list[str],
    *,
    query_terms: dict[str, int],
    idf: dict[str, float],
    avg_len: float,
) -> list[_ScoredSentence]:
    scored: list[_ScoredSentence] = []
    for idx, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        score = _bm25_score(tokens, query_terms, idf, avg_len)
        scored.append(
            _ScoredSentence(
                score=score,
                index=idx,
                sentence=sentence,
                word_count=len(sentence.split()),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.index))
    return scored


def _select_ranked_sentences(
    scored_sentences: list[_ScoredSentence],
    *,
    word_budget: int,
) -> list[_ScoredSentence]:
    if not scored_sentences or word_budget <= 0:
        return []
    selected: list[_ScoredSentence] = []
    selected_word_count = 0
    for item in scored_sentences:
        if not selected:
            selected.append(item)
            selected_word_count += item.word_count
            continue
        if selected_word_count + item.word_count <= word_budget:
            selected.append(item)
            selected_word_count += item.word_count
    return selected


def _trim_candidate_texts_by_bm25_query_text(
    query_text: str,
    candidates: list[dict],
    *,
    keep_fraction: float = 1 / 3,
    keep_fraction_by_passage_id: dict[str, float] | None = None,
    query_text_by_passage_id: dict[str, str] | None = None,
) -> None:
    if not candidates:
        return
    keep_fraction = _clamp(float(keep_fraction), 0.0, 1.0)
    if keep_fraction <= 0:
        return
    query_text = query_text.strip()
    if not query_text and not query_text_by_passage_id:
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

    idf, avg_len = _bm25_idf_and_avg_len(sentence_tokens)
    if avg_len <= 0:
        return

    default_query_terms = _bm25_query_terms(query_text)

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
        passage_query_text = query_text
        if query_text_by_passage_id is not None and passage_id:
            passage_query_text = query_text_by_passage_id.get(passage_id, query_text)
        query_terms = default_query_terms
        if passage_query_text != query_text:
            query_terms = _bm25_query_terms(passage_query_text)
        if not query_terms:
            continue
        scored = _rank_bm25_sentences(
            sentences,
            query_terms=query_terms,
            idf=idf,
            avg_len=avg_len,
        )
        word_budget = max(1, math.floor(len(cand.get("text", "").split()) * passage_keep_fraction))
        selected = _select_ranked_sentences(scored, word_budget=word_budget)
        selected.sort(key=lambda item: item.index)
        trimmed = " ".join(item.sentence for item in selected).strip()
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
    *,
    top_fraction: float = 0.10,
    mid_fraction: float = 0.20,
    top_keep_fraction: float = 0.35,
    mid_keep_fraction: float = 0.25,
    tail_keep_fraction: float = 0.15,
) -> tuple[dict[str, float], dict[str, int]]:
    if not passages:
        return {}, {
            "top_tier_passages": 0,
            "mid_tier_passages": 0,
            "tail_tier_passages": 0,
        }

    ranked_passages = sorted(
        passages,
        key=lambda passage: (
            -passage.relevance_score,
            passage.passage_id,
        ),
    )
    passage_count = len(ranked_passages)
    top_tier_count = min(passage_count, max(0, math.ceil(passage_count * top_fraction)))
    mid_tier_count = min(
        max(0, passage_count - top_tier_count),
        max(0, math.ceil(passage_count * mid_fraction)),
    )
    keep_fraction_by_passage_id: dict[str, float] = {}
    for idx, passage in enumerate(ranked_passages):
        if idx < top_tier_count:
            keep_fraction_by_passage_id[passage.passage_id] = top_keep_fraction
            continue
        if idx < top_tier_count + mid_tier_count:
            keep_fraction_by_passage_id[passage.passage_id] = mid_keep_fraction
            continue
        keep_fraction_by_passage_id[passage.passage_id] = tail_keep_fraction
    return keep_fraction_by_passage_id, {
        "top_tier_passages": top_tier_count,
        "mid_tier_passages": mid_tier_count,
        "tail_tier_passages": max(
            0,
            passage_count - top_tier_count - mid_tier_count,
        ),
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
        relevance_factor = relevance**relevance_power if relevance_power > 0 else 1.0
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
            axis_id: max(0.0, float(weight_by_axis.get(axis_id, 0.0))) for axis_id in available_ids
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
        axis.axis_id: _clamp(float(axis.theme_importance_score), 0.0, 1.0) for axis in axes
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
        shaped = scaled**exponent if exponent > 0 else 1.0
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
            key=lambda p: (
                -base_score_fn(p),
                -p.relevance_score,
                p.passage_id,
            ),
        )

    lambda_weight = _clamp(lambda_weight, 0.0, 1.0)
    source_penalty_weight = _clamp(source_penalty_weight, 0.0, 1.0)
    token_sets = [_passage_similarity_tokens(passage) for passage in passages]
    base_scores = [base_score_fn(passage) for passage in passages]
    source_groups = [
        source_group_fn(passage) if source_group_fn is not None else None for passage in passages
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
                (lambda_weight * base_scores[idx])
                - (
                    (1.0 - lambda_weight)
                    * max(
                        max_similarity_to_selected[idx],
                        source_penalty_weight
                        if source_groups[idx] is not None
                        and source_groups[idx] in selected_source_groups
                        else 0.0,
                    )
                ),
                base_scores[idx],
                passages[idx].relevance_score,
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


def _build_narrative_strategy_warnings(
    *,
    strategy: NarrativeStrategy,
    requested_episode_count: int | None,
) -> list[str]:
    warnings: list[str] = []
    narrator_profile = strategy.narrator_profile
    if (
        narrator_profile.presence_mode == "visible_host"
        and narrator_profile.baseline_tone == "grave"
    ):
        warnings.append(
            "visible_host_grave_tone_mismatch: visible_host should usually stay plainspoken unless the material truly demands graver surface phrasing."
        )
    if (
        narrator_profile.wit_ceiling in {"dry", "wry"}
        and "light_aside" not in narrator_profile.allowed_moves
    ):
        warnings.append(
            "allowed_moves_missing_light_aside_for_non_none_wit: narrator_profile allows wit but omits light_aside."
        )
    if (
        requested_episode_count is not None
        and requested_episode_count <= 5
        and narrator_profile.target_authorial_passages_per_episode < 6
    ):
        warnings.append(
            "authored_passage_target_low_for_runtime: compressed season still needs a higher authored passage budget for full-length episodes."
        )
    return warnings


def _build_narrative_strategy_actor_arc_diagnostics(
    *,
    strategy: NarrativeStrategy,
    synthesis_map: SynthesisMap,
    scene_discovery: SceneDiscoveryArtifact | None,
    actor_metadata: ActorMetadata,
) -> dict[str, Any]:
    primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
    known_actor_ids = {actor.actor_id for actor in actor_metadata.actors if actor.actor_id}
    scene_candidates = list(scene_discovery.candidates) if scene_discovery is not None else []
    episode_reports: list[dict[str, Any]] = []
    warnings: list[str] = []

    for episode in strategy.episodes:
        assigned_primitive_ids = list(episode.episode_spine.assigned_primitive_ids)
        assigned_primitive_id_set = set(assigned_primitive_ids)
        primitive_actor_ids = sorted(
            {
                actor_id
                for primitive_id in assigned_primitive_ids
                for actor_id in primitive_lookup.get(
                    primitive_id, SimpleNamespace(actor_ids=[])
                ).actor_ids
                if actor_id and actor_id in known_actor_ids
            }
        )
        scene_candidate_actor_ids = sorted(
            {
                actor_id
                for candidate in scene_candidates
                if assigned_primitive_id_set.intersection(candidate.primitive_ids)
                for actor_id in candidate.actor_ids
                if actor_id and actor_id in known_actor_ids
            }
        )
        evidence_actor_ids = sorted(set(primitive_actor_ids) | set(scene_candidate_actor_ids))
        directive_actor_ids = [
            directive.actor_id for directive in episode.actor_arc_directives if directive.actor_id
        ]
        clear_actor_evidence = bool(scene_candidate_actor_ids) or len(primitive_actor_ids) >= 2
        actor_rich_episode = len(evidence_actor_ids) >= 2
        episode_warnings: list[str] = []
        if clear_actor_evidence and not directive_actor_ids:
            episode_warnings.append(
                f"actor_arc_directives_missing_with_actor_evidence: episode_{episode.episode_number}"
            )
        elif actor_rich_episode and len(directive_actor_ids) == 1:
            episode_warnings.append(
                f"actor_arc_directives_thin_for_actor_rich_episode: episode_{episode.episode_number}"
            )
        warnings.extend(episode_warnings)
        episode_reports.append(
            {
                "episode_number": episode.episode_number,
                "actor_directive_count": len(directive_actor_ids),
                "directive_actor_ids": directive_actor_ids,
                "primitive_actor_ids": primitive_actor_ids,
                "scene_candidate_actor_ids": scene_candidate_actor_ids,
                "evidence_actor_ids": evidence_actor_ids,
                "clear_actor_evidence": clear_actor_evidence,
                "actor_rich_episode": actor_rich_episode,
                "warnings": episode_warnings,
            }
        )

    return {
        "episodes": episode_reports,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _build_narration_hook_gloss_warnings(
    primitives: list[SynthesisPrimitiveBase],
) -> list[str]:
    warnings: list[str] = []
    for primitive in primitives:
        narration_hooks = getattr(primitive, "narration_hooks", None)
        if narration_hooks is None:
            continue
        authorial_move = getattr(narration_hooks, "authorial_move", "none") or "none"
        plain_gloss = str(getattr(narration_hooks, "plain_gloss", "") or "").strip()
        if authorial_move != "none" and not plain_gloss:
            warnings.append(
                "enrichment_missing_plain_gloss: "
                f"{primitive.id} ({primitive.substrate.value}) carries authorial_move={authorial_move} without a usable spoken gloss."
            )
    return warnings


def _build_chapter_context(chapter: ChapterInfo | None) -> dict[str, Any] | None:
    if chapter is None:
        return None
    context: dict[str, Any] = {
        "chapter_id": chapter.chapter_id,
        "chapter_title": chapter.title,
    }
    analysis = chapter.analysis
    context["chapter_analysis"] = analysis.model_dump(mode="json") if analysis is not None else None
    return context


def _build_chapter_lookup(
    books: list[BookRecord],
) -> dict[tuple[str, str], ChapterInfo]:
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


def _allocate_excerpt_extraction_passages(
    *,
    selected_axes: Sequence[Any],
    passages_by_axis: Mapping[str, Sequence[ExtractedPassage]],
    passage_cap: int,
    axis_floor: int,
    ceiling_fraction: float,
) -> tuple[
    dict[str, list[ExtractedPassage]],
    dict[str, int],
    dict[str, list[ExtractedPassage]],
    list[str],
]:
    """Allocate a global excerpt-extraction passage cap across the selected axes.

    Per-axis budgets are proportional to ``theme_importance_score`` with a floor
    (``axis_floor``) and a ceiling (``ceil(passage_cap * ceiling_fraction)``).
    Within each axis pool, passages are ranked by ``(-quotability_score,
    -relevance_score, passage_id)``. Axes claim their budget in order of
    descending ``theme_importance_score``; unused slots redistribute to axes
    whose pools still have unclaimed passages, respecting the per-axis ceiling.
    Returns the per-axis pools, per-axis budgets, per-axis claimed passages,
    and the axis claim order.
    """
    passage_cap = max(1, passage_cap)
    axis_floor = max(0, axis_floor)
    axis_ceiling = max(axis_floor, math.ceil(passage_cap * ceiling_fraction))

    axis_pools: dict[str, list[ExtractedPassage]] = {}
    for axis in selected_axes:
        pool = list(passages_by_axis.get(axis.axis_id, []))
        pool.sort(
            key=lambda p: (
                -p.quotability_score,
                -p.relevance_score,
                p.passage_id,
            )
        )
        axis_pools[axis.axis_id] = pool

    axis_weights = {a.axis_id: max(0.0, float(a.theme_importance_score)) for a in selected_axes}
    total_weight = sum(axis_weights.values())
    if total_weight <= 0.0:
        even_share = max(1, passage_cap // max(1, len(selected_axes)))
        axis_budgets = {a.axis_id: even_share for a in selected_axes}
    else:
        axis_budgets = {
            a.axis_id: max(
                axis_floor,
                math.floor(passage_cap * axis_weights[a.axis_id] / total_weight),
            )
            for a in selected_axes
        }
    axis_budgets = {axis_id: min(axis_ceiling, budget) for axis_id, budget in axis_budgets.items()}

    axis_order = [
        a.axis_id
        for a in sorted(
            selected_axes,
            key=lambda a: (-a.theme_importance_score, a.axis_id),
        )
    ]
    claimed_by_axis: dict[str, list[ExtractedPassage]] = {axis_id: [] for axis_id in axis_order}
    claimed_ids: set[str] = set()
    for axis_id in axis_order:
        budget = axis_budgets[axis_id]
        for passage in axis_pools[axis_id]:
            if len(claimed_by_axis[axis_id]) >= budget:
                break
            if passage.passage_id in claimed_ids:
                continue
            claimed_by_axis[axis_id].append(passage)
            claimed_ids.add(passage.passage_id)

    while len(claimed_ids) < passage_cap:
        progress = False
        for axis_id in axis_order:
            if len(claimed_ids) >= passage_cap:
                break
            if len(claimed_by_axis[axis_id]) >= axis_ceiling:
                continue
            for passage in axis_pools[axis_id]:
                if passage.passage_id in claimed_ids:
                    continue
                claimed_by_axis[axis_id].append(passage)
                claimed_ids.add(passage.passage_id)
                progress = True
                break
        if not progress:
            break

    return axis_pools, axis_budgets, claimed_by_axis, axis_order


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
            p.passage_id,
        ),
    )


_CROSS_BOOK_PAIR_RELATIONSHIP_PRIORITY = {
    SynthesisTag.CONTRADICTS: 0,
    SynthesisTag.CONTEXTUALIZES: 1,
    SynthesisTag.EXEMPLIFIES: 2,
}


def _prioritize_cross_book_pairs(pairs: list[PassagePair]) -> list[PassagePair]:
    prioritized_pairs = [
        pair for pair in pairs if pair.relationship in _CROSS_BOOK_PAIR_RELATIONSHIP_PRIORITY
    ]

    def _sort_key(pair: PassagePair) -> tuple[int, float, str, str]:
        passage_a_id, passage_b_id = sorted((pair.passage_a_id, pair.passage_b_id))
        return (
            _CROSS_BOOK_PAIR_RELATIONSHIP_PRIORITY[pair.relationship],
            -pair.strength,
            passage_a_id,
            passage_b_id,
        )

    return sorted(prioritized_pairs, key=_sort_key)


def _combined_synthesis_passage_score(passage: ExtractedPassage) -> float:
    # quotability_score now measures excerpt-presence, not narratability, so it
    # no longer belongs in synthesis ranking. Use relevance only.
    return float(passage.relevance_score)


def _synthesis_source_key(
    passage: ExtractedPassage,
) -> tuple[str, tuple[str, ...]] | tuple[str, str]:
    if passage.chunk_ids:
        return (passage.book_id, tuple(passage.chunk_ids))
    return ("passage", passage.passage_id)


def _compute_synthesis_axis_floor_and_ceiling(
    *,
    total_cap: int,
    axis_count: int,
    floor_budget_fraction: float,
    axis_floor_min: int,
    axis_floor_max: int,
    axis_ceiling_multiplier: float,
) -> tuple[int, int, float]:
    cap = max(0, int(total_cap))
    count = max(0, int(axis_count))
    if cap <= 0 or count <= 0:
        return 0, 0, 0.0
    fair_share = cap / count
    configured_floor = min(
        max(0, axis_floor_max),
        max(max(0, axis_floor_min), math.floor(floor_budget_fraction * cap / count)),
    )
    max_feasible_floor = cap // count
    axis_floor = min(configured_floor, max_feasible_floor)
    axis_ceiling = max(axis_floor, int(round(axis_ceiling_multiplier * fair_share)))
    return axis_floor, axis_ceiling, fair_share


def _allocate_synthesis_passages_by_axis(
    *,
    axes: list[ThematicAxis],
    passages_by_axis: dict[str, list[ExtractedPassage]],
    total_cap: int,
    importance_power: float,
    cross_pair_ids: set[str],
    floor_budget_fraction: float = 0.35,
    axis_floor_min: int = 10,
    axis_floor_max: int = 15,
    axis_ceiling_multiplier: float = 1.4,
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
            "selected_axis_count": 0,
            "fair_share": 0.0,
            "axis_floor": 0,
            "axis_ceiling": 0,
            "per_axis_counts_before_refill": {},
            "per_axis_counts_after_refill": {},
            "selected_passage_ids_by_axis": {},
            "global_refill_added_count": 0,
            "ceiling_blocked_candidate_count": 0,
            "fallback_fill_count": 0,
            "duplicate_source_candidates_skipped": 0,
            "duplicate_passage_ids_detected_final": 0,
            "duplicate_source_keys_detected_final": 0,
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
    axis_floor, axis_ceiling, fair_share = _compute_synthesis_axis_floor_and_ceiling(
        total_cap=cap,
        axis_count=len(axis_ids),
        floor_budget_fraction=floor_budget_fraction,
        axis_floor_min=axis_floor_min,
        axis_floor_max=axis_floor_max,
        axis_ceiling_multiplier=axis_ceiling_multiplier,
    )
    ranked_by_axis = {
        axis_id: _rank_synthesis_axis_passages(
            passages_by_axis.get(axis_id, []),
            cross_pair_ids=cross_pair_ids,
        )
        for axis_id in axis_ids
    }

    selected_by_axis: dict[str, list[ExtractedPassage]] = {axis_id: [] for axis_id in axis_ids}
    used_source_keys: set[tuple[str, tuple[str, ...]] | tuple[str, str]] = set()
    duplicate_source_candidates_skipped = 0
    for axis_id in axis_ids:
        for passage in ranked_by_axis.get(axis_id, []):
            if len(selected_by_axis[axis_id]) >= axis_floor:
                break
            source_key = _synthesis_source_key(passage)
            if source_key in used_source_keys:
                duplicate_source_candidates_skipped += 1
                continue
            selected_by_axis[axis_id].append(passage)
            used_source_keys.add(source_key)

    per_axis_counts_before_refill = {
        axis_id: len(selected_by_axis[axis_id]) for axis_id in axis_ids
    }
    remaining_slots = max(0, cap - sum(len(items) for items in selected_by_axis.values()))
    global_refill_added_count = 0
    ceiling_blocked_candidate_count = 0
    fallback_fill_count = 0
    refill_pool: list[tuple[tuple[Any, ...], str, ExtractedPassage]] = []
    for axis_id in axis_ids:
        for passage in ranked_by_axis.get(axis_id, []):
            source_key = _synthesis_source_key(passage)
            if source_key in used_source_keys:
                continue
            refill_pool.append(
                (
                    (
                        -(1 if passage.passage_id in cross_pair_ids else 0),
                        -_combined_synthesis_passage_score(passage),
                        -passage.relevance_score,
                        passage.passage_id,
                    ),
                    axis_id,
                    passage,
                )
            )
    refill_pool.sort(key=lambda item: (item[0], item[1]))
    for _, axis_id, passage in refill_pool:
        if remaining_slots <= 0:
            break
        if len(selected_by_axis[axis_id]) >= axis_ceiling:
            ceiling_blocked_candidate_count += 1
            continue
        source_key = _synthesis_source_key(passage)
        if source_key in used_source_keys:
            duplicate_source_candidates_skipped += 1
            continue
        selected_by_axis[axis_id].append(passage)
        used_source_keys.add(source_key)
        global_refill_added_count += 1
        remaining_slots -= 1
    if remaining_slots > 0:
        for _, axis_id, passage in refill_pool:
            if remaining_slots <= 0:
                break
            source_key = _synthesis_source_key(passage)
            if source_key in used_source_keys:
                duplicate_source_candidates_skipped += 1
                continue
            selected_by_axis[axis_id].append(passage)
            used_source_keys.add(source_key)
            fallback_fill_count += 1
            remaining_slots -= 1

    passage_axes: dict[str, list[str]] = {}
    source_key_axes: dict[tuple[str, tuple[str, ...]] | tuple[str, str], list[str]] = {}
    for axis_id, passages in selected_by_axis.items():
        for passage in passages:
            passage_axes.setdefault(passage.passage_id, []).append(axis_id)
            source_key_axes.setdefault(_synthesis_source_key(passage), []).append(axis_id)
    duplicate_passage_ids = {
        passage_id: axes_for_passage
        for passage_id, axes_for_passage in passage_axes.items()
        if len(axes_for_passage) > 1
    }
    duplicate_source_keys = {
        repr(source_key): axes_for_source
        for source_key, axes_for_source in source_key_axes.items()
        if len(axes_for_source) > 1
    }
    if duplicate_passage_ids or duplicate_source_keys:
        raise ValueError(
            "Duplicate synthesis passages detected after allocation: "
            f"passage_ids={duplicate_passage_ids}, source_keys={duplicate_source_keys}"
        )

    input_total = sum(len(items) for items in passages_by_axis.values())
    output_total = sum(len(items) for items in selected_by_axis.values())
    return selected_by_axis, {
        "input_total": input_total,
        "output_total": output_total,
        "round_robin_fill_count": 0,
        "total_cap": cap,
        "axis_order": axis_ids,
        "axis_quota_by_axis": {axis_id: axis_floor for axis_id in axis_ids},
        "axis_weight_by_axis": {
            axis_id: round(float(weights.get(axis_id, 0.0)), 6) for axis_id in axis_ids
        },
        "selected_axis_count": len(axis_ids),
        "fair_share": round(float(fair_share), 6),
        "axis_floor": axis_floor,
        "axis_ceiling": axis_ceiling,
        "axis_floor_by_axis": {axis_id: axis_floor for axis_id in axis_ids},
        "axis_ceiling_by_axis": {axis_id: axis_ceiling for axis_id in axis_ids},
        "per_axis_counts_before_refill": per_axis_counts_before_refill,
        "per_axis_counts_after_refill": {
            axis_id: len(selected_by_axis[axis_id]) for axis_id in axis_ids
        },
        "selected_passage_ids_by_axis": {
            axis_id: [passage.passage_id for passage in selected_by_axis[axis_id]]
            for axis_id in axis_ids
        },
        "global_refill_added_count": global_refill_added_count,
        "ceiling_blocked_candidate_count": ceiling_blocked_candidate_count,
        "fallback_fill_count": fallback_fill_count,
        "duplicate_source_candidates_skipped": duplicate_source_candidates_skipped,
        "duplicate_passage_ids_detected_final": 0,
        "duplicate_source_keys_detected_final": 0,
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

    selected_by_axis: dict[str, list[ExtractedPassage]] = {
        axis_id: [] for axis_id in assigned_axis_ids
    }
    for axis_id in assigned_axis_ids:
        insight_passages: list[ExtractedPassage] = []
        supporting_pool: list[ExtractedPassage] = []
        for passage in passages_by_axis.get(axis_id, []):
            if passage.passage_id in selected_insight_passage_ids:
                insight_passages.append(passage)
                continue
            supporting_pool.append(passage)
        insight_passages.sort(key=lambda p: (-p.relevance_score, p.passage_id))
        target_supporting_count = supporting_passages_per_axis
        if supporting_passages_per_axis_by_axis is not None:
            target_supporting_count = supporting_passages_per_axis_by_axis.get(
                axis_id, supporting_passages_per_axis
            )
        supporting_top_n = max(0, min(target_supporting_count, len(supporting_pool)))

        def _planning_score(passage: ExtractedPassage) -> float:
            key = _chunk_key(passage)
            is_multi_axis = len(chunk_axes.get(key, set())) > 1
            # quotability_score now measures excerpt-presence and no longer belongs
            # in planning ranking; relevance + the multi-axis bonus carry it.
            return passage.relevance_score + (0.04 if is_multi_axis else 0.0)

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
                    passage.passage_id,
                ),
            )
            supporting_passages = supporting_ranked[:supporting_top_n]
        selected_by_axis[axis_id] = insight_passages + supporting_passages
    return selected_by_axis


def _build_merged_narrative_catalog(
    synthesis_map: SynthesisMap,
) -> list[dict[str, Any]]:
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


def _build_synthesis_feedback_for_merged_narrative_count(
    report: dict[str, Any],
) -> dict[str, Any]:
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
            invalid_assigned_id = (
                assigned_id if assigned_id and assigned_id not in valid_ids else None
            )
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


def _build_strategy_feedback_for_merged_narratives(
    report: dict[str, Any],
) -> dict[str, Any]:
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


_NARRATIVE_STRATEGY_OMIT = object()
_NARRATIVE_STRATEGY_PRIMITIVE_DROP_FIELDS = {
    "support_passage_ids",
    "affected_actor_ids",
    "actor_tags",
    "narration_hooks",
}
_SCENE_DISCOVERY_PRIMITIVE_BASE_KEEP_FIELDS = {
    "id",
    "substrate",
    "title",
    "core_passage_ids",
    "support_passage_ids",
    "timeframe",
    "geography",
    "actor_ids",
    "functions",
    "cost",
    "complication",
}
_SCENE_DISCOVERY_PRIMITIVE_TYPE_KEEP_FIELDS: dict[str, set[str]] = {
    PrimitiveSubstrate.EVENTS.value: {"event_type", "what_happened", "event_result"},
    PrimitiveSubstrate.ACTS.value: {
        "act_type",
        "acting_subject",
        "act_summary",
        "immediate_result",
    },
    PrimitiveSubstrate.ACTOR_PORTRAITS.value: {
        "focus_actor_id",
        "actor_label",
        "goal_or_project",
        "stakes_or_fears",
        "operating_pressure",
    },
    PrimitiveSubstrate.MECHANISMS.value: {
        "mechanism_name",
        "operating_chain",
        "inputs",
        "outputs",
        "failure_mode",
    },
    PrimitiveSubstrate.CONDITIONS.value: {
        "condition_type",
        "condition_summary",
        "active_tension",
    },
    PrimitiveSubstrate.ARTIFACTS.value: {
        "artifact_type",
        "artifact_label",
        "artifact_detail",
    },
    PrimitiveSubstrate.READINGS.value: {
        "reading_type",
        "subject_of_reading",
        "attributed_to",
        "reading_summary",
    },
}
_SCENE_DISCOVERY_NARRATION_HOOK_KEEP_FIELDS = {
    "plain_gloss",
    "why_it_matters",
    "best_use",
    "natural_host_move",
}
_NARRATIVE_STRATEGY_ACTOR_KEEP_FIELDS = {
    "actor_id",
    "display_name",
    "aliases",
    "actor_type",
    "description",
    "goals_or_motivational_pressures",
    "constraints",
    "stakes",
    "transformations",
    "narrative_tier",
    "series_scope",
    "relevant_episode_numbers",
}
_NARRATIVE_STRATEGY_SCENE_KEEP_FIELDS = {
    "candidate_id",
    "primitive_ids",
    "scene_jobs",
    "scene_sketch",
    "anchor_image",
    "why_sceneable",
    "quote_anchor",
    "actor_ids",
}
_PRIMITIVE_ANNOTATION_ACTOR_KEEP_FIELDS = {
    "actor_id",
    "display_name",
    "aliases",
    "actor_type",
}


def _compact_narrative_strategy_runtime_value(value: Any) -> Any:
    if value is None:
        return _NARRATIVE_STRATEGY_OMIT
    if isinstance(value, str):
        return value if value else _NARRATIVE_STRATEGY_OMIT
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            compacted_item = _compact_narrative_strategy_runtime_value(item)
            if compacted_item is _NARRATIVE_STRATEGY_OMIT:
                continue
            compacted[key] = compacted_item
        return compacted or _NARRATIVE_STRATEGY_OMIT
    if isinstance(value, list):
        compacted_list = [
            compacted_item
            for item in value
            if (compacted_item := _compact_narrative_strategy_runtime_value(item))
            is not _NARRATIVE_STRATEGY_OMIT
        ]
        return compacted_list or _NARRATIVE_STRATEGY_OMIT
    return value


def _compact_narrative_strategy_runtime_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    compacted = _compact_narrative_strategy_runtime_value(payload)
    if compacted is _NARRATIVE_STRATEGY_OMIT:
        return {}
    assert isinstance(compacted, dict)
    return compacted


def _build_narrative_strategy_synthesis_map_payload(
    synthesis_map: PrimitiveFunctionTaggingArtifact | SynthesisMap,
) -> dict[str, Any]:
    primitives = [
        {
            key: value
            for key, value in primitive.model_dump(mode="json").items()
            if key not in _NARRATIVE_STRATEGY_PRIMITIVE_DROP_FIELDS
        }
        for primitive in synthesis_map.primitives
    ]
    return _compact_narrative_strategy_runtime_payload(
        {
            "project_id": synthesis_map.project_id,
            "primitives": primitives,
            "quality_score": synthesis_map.quality_score,
            "quality_notes": list(synthesis_map.quality_notes),
        }
    )


def _build_strategy_excerpt_payload(
    excerpts: ExcerptArtifact | None,
) -> list[dict[str, Any]] | None:
    if excerpts is None or not excerpts.excerpts:
        return None
    payload: list[dict[str, Any]] = []
    for excerpt in excerpts.excerpts:
        item: dict[str, Any] = {
            "id": excerpt.id,
            "excerpt_type": excerpt.excerpt_type,
            "title": excerpt.title,
            "summary": excerpt.summary,
            "quotability": excerpt.quotability,
        }
        if excerpt.verbatim_excerpt.strip():
            item["verbatim_excerpt"] = excerpt.verbatim_excerpt
        if excerpt.speaker.strip():
            item["speaker"] = excerpt.speaker
        if excerpt.timeframe:
            item["timeframe"] = excerpt.timeframe
        if excerpt.actor_ids:
            item["actor_ids"] = list(excerpt.actor_ids)
        payload.append(item)
    return payload


def _build_episode_excerpt_payload(
    excerpt_ids: list[str],
    excerpt_by_id: dict[str, "ExcerptRecord"],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for excerpt_id in excerpt_ids:
        if not excerpt_id or excerpt_id in seen:
            continue
        seen.add(excerpt_id)
        excerpt = excerpt_by_id.get(excerpt_id)
        if excerpt is None:
            continue
        item: dict[str, Any] = {
            "id": excerpt.id,
            "excerpt_type": excerpt.excerpt_type,
            "title": excerpt.title,
            "summary": excerpt.summary,
            "quotability": excerpt.quotability,
            "passage_ids": list(excerpt.passage_ids),
        }
        if excerpt.verbatim_excerpt.strip():
            item["verbatim_excerpt"] = excerpt.verbatim_excerpt
        if excerpt.plain_gloss.strip():
            item["plain_gloss"] = excerpt.plain_gloss
        if excerpt.speaker.strip():
            item["speaker"] = excerpt.speaker
        if excerpt.audience.strip():
            item["audience"] = excerpt.audience
        if excerpt.timeframe:
            item["timeframe"] = excerpt.timeframe
        if excerpt.actor_ids:
            item["actor_ids"] = list(excerpt.actor_ids)
        payload.append(item)
    return payload


def _build_narrative_strategy_project_metadata_payload(
    project: ThematicProject,
) -> dict[str, Any]:
    return _compact_narrative_strategy_runtime_payload(
        {
            "podcast_mode": project.config.podcast_mode.value,
            "theme": project.theme,
            "sub_themes": list(project.sub_themes),
            "book_count": len(project.books),
            "books": [
                {
                    "book_id": book.book_id,
                    "title": book.title,
                    "author": book.author,
                }
                for book in project.books
            ],
            "episode_spine_core_primitive_target_min": (
                project.config.episode_spine_core_primitive_target_min
            ),
            "episode_spine_core_primitive_target_max": (
                project.config.episode_spine_core_primitive_target_max
            ),
            "episode_spine_support_primitive_target_min": (
                project.config.episode_spine_support_primitive_target_min
            ),
            "episode_spine_support_primitive_target_max": (
                project.config.episode_spine_support_primitive_target_max
            ),
            "episode_spine_recall_primitive_target_max": (
                project.config.episode_spine_recall_primitive_target_max
            ),
            "episode_spine_excerpt_target_min": (project.config.episode_spine_excerpt_target_min),
            "episode_spine_excerpt_target_max": (project.config.episode_spine_excerpt_target_max),
            "min_episode_minutes": project.config.min_episode_minutes,
            "max_episode_minutes": project.config.max_episode_minutes,
        }
    )


def _build_narrative_strategy_actor_metadata_payload(
    actor_metadata: ActorMetadata,
) -> dict[str, Any]:
    return _compact_narrative_strategy_runtime_payload(
        {
            "actors": [
                {
                    key: value
                    for key, value in actor.model_dump(mode="json").items()
                    if key in _NARRATIVE_STRATEGY_ACTOR_KEEP_FIELDS
                }
                for actor in actor_metadata.actors
            ],
        }
    )


def _build_human_thread_candidate_index(
    synthesis_map: SynthesisMap | PrimitiveFunctionTaggingArtifact,
    actor_metadata: ActorMetadata,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Project-level ranking of human-thread carrier candidates by cross-section
    coverage (not fame). Includes canonical person-actors and situated label-only
    people from actor_portraits primitives so the skeleton can prefer coverage."""
    primitives = _flatten_base_synthesis_primitives(synthesis_map)
    actors_by_id = {actor.actor_id: actor for actor in actor_metadata.actors}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def _ensure(key, *, kind, label, actor_id, tier, scope):
        existing = candidates.get(key)
        if existing is None:
            existing = {
                "kind": kind,
                "label": label,
                "actor_id": actor_id,
                "narrative_tier": tier,
                "series_scope": scope,
                "_prims": set(),
                "_pass": set(),
            }
            candidates[key] = existing
        return existing

    for pid, prim in primitives.items():
        passage_ids = set(getattr(prim, "passage_ids", []) or [])
        for aid in getattr(prim, "actor_ids", []) or []:
            actor = actors_by_id.get(aid)
            if actor is None or actor.actor_type != "person":
                continue
            cand = _ensure(
                ("canonical", aid),
                kind="canonical",
                label=actor.display_name,
                actor_id=aid,
                tier=actor.narrative_tier.value,
                scope=actor.series_scope.value,
            )
            cand["_prims"].add(pid)
            cand["_pass"].update(passage_ids)
        if getattr(prim, "substrate", None) == PrimitiveSubstrate.ACTOR_PORTRAITS:
            label = (getattr(prim, "actor_label", "") or "").strip()
            focus = getattr(prim, "focus_actor_id", None)
            if label and (focus is None or focus not in actors_by_id):
                cand = _ensure(
                    ("situated", normalize_actor_name(label)),
                    kind="situated",
                    label=label,
                    actor_id=None,
                    tier="supporting",
                    scope="local",
                )
                cand["_prims"].add(pid)
                cand["_pass"].update(passage_ids)

    ranked: list[dict[str, Any]] = []
    for cand in candidates.values():
        passage_count = len(cand["_pass"])
        coverage_score = passage_count + len(cand["_prims"])
        # Down-weight a famous, series-wide major actor whose passage coverage is thin.
        if (
            cand["narrative_tier"] == "major"
            and cand["series_scope"] == "series_wide"
            and passage_count < 3
        ):
            coverage_score -= 2
        ranked.append(
            {
                "kind": cand["kind"],
                "actor_id": cand["actor_id"],
                "label": cand["label"],
                "narrative_tier": cand["narrative_tier"],
                "series_scope": cand["series_scope"],
                "primitive_ids": sorted(cand["_prims"]),
                "passage_count": passage_count,
                "coverage_score": coverage_score,
            }
        )
    ranked.sort(key=lambda item: (-item["coverage_score"], item["label"]))
    return ranked[:top_n]


def _build_narrative_strategy_scene_discovery_payload(
    scene_discovery: SceneDiscoveryArtifact | None,
) -> dict[str, Any] | None:
    if scene_discovery is None:
        return None
    candidates = [
        {
            key: value
            for key, value in candidate.model_dump(mode="json").items()
            if key in _NARRATIVE_STRATEGY_SCENE_KEEP_FIELDS
        }
        for candidate in scene_discovery.candidates
    ]
    return _compact_narrative_strategy_runtime_payload({"candidates": candidates})


def _strategy_selected_primitive_ids_from_skeleton(
    strategy: NarrativeStrategySkeleton,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for episode in strategy.episodes:
        for primitive_id in episode.episode_spine.assigned_primitive_ids:
            if not primitive_id or primitive_id in seen:
                continue
            seen.add(primitive_id)
            ordered.append(primitive_id)
    return ordered


def _build_strategy_selected_synthesis_map_preview(
    *,
    project_id: str,
    synthesis_map: PrimitiveFunctionTaggingArtifact,
    strategy: NarrativeStrategySkeleton,
) -> SynthesisMap:
    selected_ids = _strategy_selected_primitive_ids_from_skeleton(strategy)
    primitive_by_id = _flatten_base_synthesis_primitives(synthesis_map)
    selected_primitives = [
        primitive_by_id[primitive_id]
        for primitive_id in selected_ids
        if primitive_id in primitive_by_id
    ]
    return SynthesisMap(
        project_id=project_id,
        primitives=selected_primitives,
        quality_score=synthesis_map.quality_score,
        quality_notes=list(synthesis_map.quality_notes),
    )


def _episode_scene_candidate_cap_for_mode(mode: PodcastMode) -> int:
    return 8 if mode == PodcastMode.MINIFIED else 13


def _human_thread_candidate_cap_for_mode(mode: PodcastMode) -> int:
    return 12 if mode == PodcastMode.MINIFIED else 40


def _build_episode_scene_candidate_payloads(
    *,
    strategy: NarrativeStrategySkeleton,
    scene_discovery: SceneDiscoveryArtifact | None,
    mode: PodcastMode,
) -> list[dict[str, Any]]:
    if scene_discovery is None:
        return [
            {"episode_number": episode.episode_number, "candidates": []}
            for episode in strategy.episodes
        ]
    role_priority = {
        "opening": 0,
        "turn": 1,
        "answer": 2,
        "build": 4,
    }
    cap = _episode_scene_candidate_cap_for_mode(mode)
    payloads: list[dict[str, Any]] = []
    for episode in strategy.episodes:
        assigned_ids = set(episode.episode_spine.assigned_primitive_ids)
        core_ids = set(episode.episode_spine.core_primitive_ids)
        ranked: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
        for idx, candidate in enumerate(scene_discovery.candidates):
            overlap = assigned_ids.intersection(candidate.primitive_ids)
            if not overlap:
                continue
            core_overlap = len(core_ids.intersection(candidate.primitive_ids))
            role_rank = min(
                role_priority.get(role, len(role_priority)) for role in candidate.scene_jobs
            )
            score = (
                -core_overlap,
                -len(overlap),
                role_rank,
                0 if candidate.quote_anchor.strip() else 1,
                idx,
            )
            ranked.append(
                (
                    score,
                    {
                        key: value
                        for key, value in candidate.model_dump(mode="json").items()
                        if key in _NARRATIVE_STRATEGY_SCENE_KEEP_FIELDS
                    },
                )
            )
        ranked.sort(key=lambda item: item[0])
        payloads.append(
            {
                "episode_number": episode.episode_number,
                "candidates": [payload for _, payload in ranked[:cap]],
            }
        )
    return payloads


def _merge_narrative_strategy_parts(
    *,
    skeleton: NarrativeStrategySkeleton,
    enrichment: NarrativeStrategyEnrichment,
) -> NarrativeStrategy:
    enrichment_by_number = {episode.episode_number: episode for episode in enrichment.episodes}
    skeleton_numbers = [episode.episode_number for episode in skeleton.episodes]
    enrichment_numbers = sorted(enrichment_by_number)
    if sorted(skeleton_numbers) != enrichment_numbers:
        raise RuntimeError(
            "narrative strategy enrichment episode numbers must match skeleton episodes"
        )
    episodes: list[StrategyEpisode] = []
    for skeleton_episode in skeleton.episodes:
        enrichment_episode = enrichment_by_number[skeleton_episode.episode_number]
        episodes.append(
            StrategyEpisode(
                episode_number=skeleton_episode.episode_number,
                title=skeleton_episode.title,
                thematic_focus=skeleton_episode.thematic_focus,
                arc_summary=skeleton_episode.arc_summary,
                unresolved_questions=list(skeleton_episode.unresolved_questions),
                episode_spine=skeleton_episode.episode_spine.model_copy(deep=True),
                actor_arc_directives=[
                    directive.model_copy(deep=True)
                    for directive in enrichment_episode.actor_arc_directives
                ],
                human_thread=(
                    enrichment_episode.human_thread.model_copy(deep=True)
                    if enrichment_episode.human_thread is not None
                    else None
                ),
                narrator_contract=enrichment_episode.narrator_contract.model_copy(deep=True),
                authorial_contract=enrichment_episode.authorial_contract.model_copy(deep=True),
                narrative_agenda=enrichment_episode.narrative_agenda.model_copy(deep=True),
                promised_beats=[
                    beat.model_copy(deep=True) for beat in enrichment_episode.promised_beats
                ],
                negative_scope=skeleton_episode.negative_scope.model_copy(deep=True),
            )
        )
    return NarrativeStrategy(
        strategy_type=skeleton.strategy_type,
        justification=skeleton.justification,
        series_arc=skeleton.series_arc,
        episode_arc_outline=list(skeleton.episode_arc_outline),
        recommended_episode_count=skeleton.recommended_episode_count,
        narrator_profile=enrichment.narrator_profile.model_copy(deep=True),
        series_explanation_registry=[
            item.model_copy(deep=True) for item in enrichment.series_explanation_registry
        ],
        series_actor_explanation_registry=[
            item.model_copy(deep=True) for item in enrichment.series_actor_explanation_registry
        ],
        episodes=episodes,
    )


def _build_scene_discovery_synthesis_map_payload(
    synthesis_map: PrimitiveFunctionTaggingArtifact,
) -> dict[str, Any]:
    primitives: list[dict[str, Any]] = []
    for primitive in synthesis_map.primitives:
        payload = primitive.model_dump(mode="json")
        keep_fields = _SCENE_DISCOVERY_PRIMITIVE_BASE_KEEP_FIELDS | (
            _SCENE_DISCOVERY_PRIMITIVE_TYPE_KEEP_FIELDS.get(
                str(payload.get("substrate", "")),
                set(),
            )
        )
        compacted_primitive = {key: value for key, value in payload.items() if key in keep_fields}
        salience = payload.get("salience")
        if isinstance(salience, dict) and salience.get("score") is not None:
            compacted_primitive["salience"] = {"score": salience["score"]}
        narration_hooks = payload.get("narration_hooks")
        if isinstance(narration_hooks, dict):
            compacted_primitive["narration_hooks"] = {
                key: value
                for key, value in narration_hooks.items()
                if key in _SCENE_DISCOVERY_NARRATION_HOOK_KEEP_FIELDS
            }
        primitives.append(compacted_primitive)
    return _compact_narrative_strategy_runtime_payload(
        {
            "project_id": synthesis_map.project_id,
            "primitives": primitives,
            "quality_score": synthesis_map.quality_score,
            "quality_notes": list(synthesis_map.quality_notes),
        }
    )


def _build_scene_discovery_project_metadata_payload(
    project: ThematicProject,
) -> dict[str, Any]:
    metadata = _build_narrative_strategy_project_metadata_payload(project)
    metadata.update(
        {
            "scene_card_target_min": project.config.scene_card_target_min,
            "scene_card_target_max": project.config.scene_card_target_max,
        }
    )
    return _compact_narrative_strategy_runtime_payload(metadata)


def _build_scene_discovery_passage_list_payload(
    *,
    project: ThematicProject,
    synthesis_map: PrimitiveFunctionTaggingArtifact,
    passage_lookup: dict[str, ExtractedPassage],
) -> list[dict[str, str]]:
    primitive_lookup = {primitive.id: primitive for primitive in synthesis_map.primitives}
    passage_refs: list[dict[str, str]] = []
    core_flags_by_passage_id: dict[str, bool] = {}
    primitive_sets_by_passage_id: dict[str, set[str]] = {}
    for primitive in synthesis_map.primitives:
        for passage_id in primitive.core_passage_ids:
            if not passage_id:
                continue
            passage_refs.append({"passage_id": passage_id, "primitive_id": primitive.id})
            core_flags_by_passage_id[passage_id] = True
            primitive_sets_by_passage_id.setdefault(passage_id, set()).add(primitive.id)
        for passage_id in primitive.support_passage_ids:
            if not passage_id:
                continue
            passage_refs.append({"passage_id": passage_id, "primitive_id": primitive.id})
            core_flags_by_passage_id.setdefault(passage_id, False)
            primitive_sets_by_passage_id.setdefault(passage_id, set()).add(primitive.id)
    query_text = " ".join(
        part.strip()
        for part in [project.theme, *list(project.sub_themes)[:4]]
        if part and part.strip()
    )
    query_text_by_passage_id = _build_episode_planning_passage_query_texts(
        episode_query_text=query_text,
        passage_refs=passage_refs,
        primitive_lookup=primitive_lookup,
    )
    keep_fraction_by_passage_id: dict[str, float] = {}
    passage_list: list[dict[str, str]] = []
    seen_passage_ids: set[str] = set()
    for passage_ref in passage_refs:
        passage_id = passage_ref["passage_id"]
        if passage_id in seen_passage_ids:
            continue
        seen_passage_ids.add(passage_id)
        passage = passage_lookup.get(passage_id)
        if passage is None:
            continue
        primitive_count = len(primitive_sets_by_passage_id.get(passage_id, set()))
        if primitive_count > 1:
            keep_fraction_by_passage_id[passage_id] = 0.16
        elif core_flags_by_passage_id.get(passage_id):
            keep_fraction_by_passage_id[passage_id] = 0.10
        else:
            keep_fraction_by_passage_id[passage_id] = 0.05
        passage_list.append(
            {
                "passage_id": passage_id,
                "text": _resolve_writing_passage_text(passage),
            }
        )
    _trim_candidate_texts_by_bm25_query_text(
        query_text,
        passage_list,
        keep_fraction=0.10,
        keep_fraction_by_passage_id=keep_fraction_by_passage_id,
        query_text_by_passage_id=query_text_by_passage_id,
    )
    return [
        {
            "passage_id": str(item.get("passage_id", "")),
            "text": str(item.get("text", "")).strip(),
        }
        for item in passage_list
        if str(item.get("passage_id", "")).strip() and str(item.get("text", "")).strip()
    ]


def _build_episode_scene_payload(
    *,
    strategy_episode: StrategyEpisode,
    scene_discovery: SceneDiscoveryArtifact | None,
) -> list[dict[str, Any]]:
    if scene_discovery is None:
        return []

    assigned_primitive_ids = set(strategy_episode.episode_spine.assigned_primitive_ids)
    promised_candidate_ids = {
        candidate_id
        for beat in strategy_episode.promised_beats
        for candidate_id in beat.source_candidate_ids
        if candidate_id
    }
    episode_candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for candidate in scene_discovery.candidates:
        if candidate.candidate_id in seen_candidate_ids:
            continue
        if candidate.candidate_id in promised_candidate_ids or (
            assigned_primitive_ids & set(candidate.primitive_ids)
        ):
            episode_candidates.append(candidate.model_dump(mode="json"))
            seen_candidate_ids.add(candidate.candidate_id)
    return episode_candidates


def _build_scene_discovery_diagnostics(
    *,
    artifact: SceneDiscoveryArtifact,
    mode: PodcastMode,
) -> dict[str, Any]:
    target_min, target_max = scene_discovery_candidate_range_for_mode(mode)
    candidate_count = len(artifact.candidates)
    overlap_heavy_candidate_ids: list[str] = []
    thin_evidence_candidate_ids: list[str] = []
    candidate_sets = [
        (candidate.candidate_id, set(candidate.primitive_ids)) for candidate in artifact.candidates
    ]
    for candidate in artifact.candidates:
        if len(candidate.passage_ids) <= 1 and not candidate.quote_anchor.strip():
            thin_evidence_candidate_ids.append(candidate.candidate_id)
    for idx, (candidate_id, primitive_ids) in enumerate(candidate_sets):
        for other_id, other_primitive_ids in candidate_sets[idx + 1 :]:
            if not primitive_ids or not other_primitive_ids:
                continue
            overlap_ratio = len(primitive_ids & other_primitive_ids) / max(
                1, min(len(primitive_ids), len(other_primitive_ids))
            )
            if overlap_ratio >= 0.75:
                overlap_heavy_candidate_ids.extend([candidate_id, other_id])
    overlap_heavy_candidate_ids = sorted(set(overlap_heavy_candidate_ids))
    warnings: list[str] = []
    if candidate_count < target_min:
        warnings.append(f"scene_candidate_count_below_target: {candidate_count} < {target_min}")
    elif candidate_count > target_max:
        warnings.append(f"scene_candidate_count_above_target: {candidate_count} > {target_max}")
    if overlap_heavy_candidate_ids:
        warnings.append(
            f"scene_candidate_overlap_heavy: {_preview_ids(overlap_heavy_candidate_ids)}"
        )
    if thin_evidence_candidate_ids:
        warnings.append(
            "scene_candidate_thin_evidence: "
            f"{_preview_ids(sorted(set(thin_evidence_candidate_ids)))}"
        )
    return {
        "candidate_count": candidate_count,
        "candidate_count_target_min": target_min,
        "candidate_count_target_max": target_max,
        "candidates_with_quote_anchor": sum(
            1 for candidate in artifact.candidates if candidate.quote_anchor.strip()
        ),
        "candidates_with_actor_ids": sum(
            1 for candidate in artifact.candidates if candidate.actor_ids
        ),
        "overlap_heavy_candidate_ids": overlap_heavy_candidate_ids,
        "thin_evidence_candidate_ids": sorted(set(thin_evidence_candidate_ids)),
        "warning_count": len(warnings),
        "warnings": warnings,
    }


_PRIMITIVE_ANNOTATION_BATCH_CONCURRENCY = 8
_PRIMITIVE_ANNOTATION_BATCH_THRESHOLD = 30
_PRIMITIVE_ANNOTATION_SPLIT_BATCH_COUNT = 2


def _flatten_base_synthesis_primitives(
    artifact: SynthesisPrimitivesArtifact | PrimitiveFunctionTaggingArtifact | SynthesisMap,
) -> dict[str, BaseSynthesisPrimitive]:
    return {item.id: item for item in artifact.primitives}


def _split_primitive_annotation_batches(
    primitives: list[BaseSynthesisPrimitive],
) -> list[list[BaseSynthesisPrimitive]]:
    if len(primitives) <= _PRIMITIVE_ANNOTATION_BATCH_THRESHOLD:
        return [primitives]
    midpoint = (len(primitives) + 1) // _PRIMITIVE_ANNOTATION_SPLIT_BATCH_COUNT
    return [primitives[:midpoint], primitives[midpoint:]]


def _collect_strategy_selected_primitive_ids(strategy: NarrativeStrategy) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for episode in strategy.episodes:
        for primitive_id in episode.episode_spine.assigned_primitive_ids:
            if not primitive_id or primitive_id in seen:
                continue
            seen.add(primitive_id)
            ordered.append(primitive_id)
    return ordered


def _collect_strategy_selected_excerpt_ids(strategy: NarrativeStrategy) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for episode in strategy.episodes:
        for excerpt_id in episode.episode_spine.assigned_excerpt_ids:
            if not excerpt_id or excerpt_id in seen:
                continue
            seen.add(excerpt_id)
            ordered.append(excerpt_id)
    return ordered


def _build_passage_lookup(corpus: ThematicCorpus) -> dict[str, ExtractedPassage]:
    lookup: dict[str, ExtractedPassage] = {}
    for passages in corpus.passages_by_axis.values():
        for passage in passages:
            lookup[passage.passage_id] = passage
    return lookup


def _build_primitive_evidence_query(
    substrate: str,
    primitive: BaseSynthesisPrimitive,
    actor_metadata: ActorMetadata,
) -> str:
    actor_name_by_id = {
        actor.actor_id: actor.display_name.strip()
        for actor in actor_metadata.actors
        if actor.display_name.strip()
    }
    parts: list[str] = [substrate.replace("_", " ")]
    parts.append(primitive.title)
    parts.extend(_primitive_substrate_text_fragments(primitive))
    if primitive.timeframe:
        parts.append(primitive.timeframe)
    if primitive.geography:
        parts.append(primitive.geography)
    for actor_id in primitive.actor_ids:
        if actor_name := actor_name_by_id.get(actor_id):
            parts.append(actor_name)
    return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(frozen=True)
class _PrimitivePassageReference:
    primitive_id: str
    role: str
    query_text: str


def _build_function_tagging_passage_list(
    *,
    primitives: list[BaseSynthesisPrimitive],
    passage_lookup: dict[str, ExtractedPassage],
    actor_metadata: ActorMetadata,
    trim_profile: _PrimitiveEvidenceTrimProfile,
) -> list[dict[str, str]]:
    if not primitives:
        return []
    passage_records: dict[str, dict[str, Any]] = {}

    for primitive in primitives:
        query_text = _build_primitive_evidence_query(
            primitive.substrate.value,
            primitive,
            actor_metadata,
        )
        seen_core_ids: set[str] = set()
        seen_support_ids: set[str] = set()
        for passage_id in primitive.core_passage_ids:
            if not passage_id or passage_id in seen_core_ids:
                continue
            seen_core_ids.add(passage_id)
            passage = passage_lookup.get(passage_id)
            if passage is None:
                continue
            full_text = (
                passage.full_text.strip() or passage.trimmed_text.strip() or passage.text.strip()
            )
            if not full_text:
                continue
            record = passage_records.setdefault(
                passage.passage_id,
                {
                    "passage_id": passage.passage_id,
                    "text": full_text,
                    "references": [],
                },
            )
            record["references"].append(
                _PrimitivePassageReference(
                    primitive_id=primitive.id,
                    role="core",
                    query_text=query_text,
                )
            )
        for passage_id in primitive.support_passage_ids:
            if not passage_id or passage_id in seen_support_ids or passage_id in seen_core_ids:
                continue
            seen_support_ids.add(passage_id)
            passage = passage_lookup.get(passage_id)
            if passage is None:
                continue
            full_text = (
                passage.full_text.strip() or passage.trimmed_text.strip() or passage.text.strip()
            )
            if not full_text:
                continue
            record = passage_records.setdefault(
                passage.passage_id,
                {
                    "passage_id": passage.passage_id,
                    "text": full_text,
                    "references": [],
                },
            )
            record["references"].append(
                _PrimitivePassageReference(
                    primitive_id=primitive.id,
                    role="support",
                    query_text=query_text,
                )
            )

    passage_list: list[dict[str, str]] = []
    for passage_id, record in passage_records.items():
        full_text = str(record["text"]).strip()
        sentences = _split_sentences(full_text)
        if not sentences:
            continue
        sentence_tokens = [_tokenize(sentence) for sentence in sentences]
        idf, avg_len = _bm25_idf_and_avg_len(sentence_tokens)
        total_word_count = len(full_text.split())
        if total_word_count <= 0:
            continue

        selected_indices_by_reference: dict[_PrimitivePassageReference, set[int]] = {}
        score_by_reference_and_index: dict[tuple[_PrimitivePassageReference, int], float] = {}
        role_by_index: dict[int, set[str]] = {}
        reference_count_by_index: dict[int, int] = {}

        references: list[_PrimitivePassageReference] = list(record["references"])
        for reference in references:
            keep_fraction = (
                trim_profile.core_keep_fraction
                if reference.role == "core"
                else trim_profile.support_keep_fraction
            )
            query_terms = _bm25_query_terms(reference.query_text)
            scored = _rank_bm25_sentences(
                sentences,
                query_terms=query_terms,
                idf=idf,
                avg_len=avg_len,
            )
            word_budget = max(1, math.floor(total_word_count * keep_fraction))
            selected = _select_ranked_sentences(scored, word_budget=word_budget)
            selected_indices = {item.index for item in selected}
            selected_indices_by_reference[reference] = selected_indices
            for item in scored:
                score_by_reference_and_index[(reference, item.index)] = item.score
            for idx in selected_indices:
                role_by_index.setdefault(idx, set()).add(reference.role)
                reference_count_by_index[idx] = reference_count_by_index.get(idx, 0) + 1

        final_pool_indices: set[int]
        if len(references) == 1:
            final_pool_indices = set(next(iter(selected_indices_by_reference.values())))
        else:
            shared_budget = max(
                1,
                math.floor(total_word_count * trim_profile.shared_passage_keep_fraction),
            )
            reserved_candidates: dict[int, dict[str, Any]] = {}
            for reference in references:
                scored = _rank_bm25_sentences(
                    sentences,
                    query_terms=_bm25_query_terms(reference.query_text),
                    idf=idf,
                    avg_len=avg_len,
                )
                if not scored:
                    continue
                top = scored[0]
                candidate = reserved_candidates.setdefault(
                    top.index,
                    {
                        "refs": set(),
                        "has_core": False,
                        "best_score": top.score,
                    },
                )
                candidate["refs"].add(reference)
                candidate["has_core"] = candidate["has_core"] or reference.role == "core"
                candidate["best_score"] = max(candidate["best_score"], top.score)

            final_pool_indices = set()
            used_words = 0
            covered_refs: set[_PrimitivePassageReference] = set()
            pending_reserved = dict(reserved_candidates)
            while pending_reserved:
                next_idx = None
                next_payload = None
                for idx, payload in sorted(
                    pending_reserved.items(),
                    key=lambda item: (
                        -len(item[1]["refs"] - covered_refs),
                        0 if item[1]["has_core"] else 1,
                        -item[1]["best_score"],
                        item[0],
                    ),
                ):
                    sentence_words = len(sentences[idx].split())
                    if used_words + sentence_words <= shared_budget:
                        next_idx = idx
                        next_payload = payload
                        break
                if next_idx is None or next_payload is None:
                    break
                final_pool_indices.add(next_idx)
                used_words += len(sentences[next_idx].split())
                covered_refs.update(next_payload["refs"])
                pending_reserved.pop(next_idx, None)

            remaining_candidates = sorted(
                [idx for idx in reference_count_by_index if idx not in final_pool_indices],
                key=lambda idx: (
                    -reference_count_by_index.get(idx, 0),
                    0 if "core" in role_by_index.get(idx, set()) else 1,
                    -max(
                        score_by_reference_and_index.get((reference, idx), 0.0)
                        for reference in references
                    ),
                    idx,
                ),
            )
            for idx in remaining_candidates:
                sentence_words = len(sentences[idx].split())
                if used_words + sentence_words > shared_budget:
                    continue
                final_pool_indices.add(idx)
                used_words += sentence_words

            if not final_pool_indices and reserved_candidates:
                best_idx = min(
                    reserved_candidates,
                    key=lambda idx: (
                        0 if reserved_candidates[idx]["has_core"] else 1,
                        -reserved_candidates[idx]["best_score"],
                        idx,
                    ),
                )
                final_pool_indices.add(best_idx)

        trimmed = " ".join(sentences[idx] for idx in sorted(final_pool_indices)).strip()
        if trimmed:
            passage_list.append(
                {
                    "passage_id": passage_id,
                    "text": trimmed,
                }
            )
    return passage_list


def _build_primitive_actor_metadata_payload(
    actor_metadata: ActorMetadata,
) -> dict[str, Any]:
    def _one_line_role(description: str, actor_type: str) -> str:
        text = str(description or "").strip()
        if not text:
            return actor_type.replace("_", " ")
        for delimiter in (".", ";", ":"):
            if delimiter in text:
                text = text.split(delimiter, 1)[0].strip()
                break
        words = text.split()
        if len(words) > 16:
            text = " ".join(words[:16]).rstrip(",")
        return text

    return {
        "actors": [
            {
                **{
                    key: value
                    for key, value in actor.model_dump(mode="json").items()
                    if key in _PRIMITIVE_ANNOTATION_ACTOR_KEEP_FIELDS
                },
                "one_line_role": _one_line_role(actor.description, actor.actor_type),
            }
            for actor in actor_metadata.actors
        ],
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


def _build_plan_transition_feedback(exc: ComplianceViolationError) -> dict[str, Any]:
    data = exc.data or {}
    feedback = {
        "issue": str(data.get("issue", "plan_transition_contract_violation")),
        "episode_number": data.get("episode_number"),
        "instruction": data.get(
            "instruction",
            "Revise the episode plan so it strictly follows the architecture section order, "
            "keeps primitives inside their assigned sections, and covers all required sections and core primitives.",
        ),
    }
    for key in (
        "scene_id",
        "scene_ids",
        "section_id",
        "phase_ids",
        "invalid_section_ids",
        "invalid_scene_primitives",
        "missing_section_ids",
        "missing_core_primitive_ids",
        "invalid_dropped_support_ids",
        "expected_episode_number",
        "actual_episode_number",
        "answer_scene_card_id",
        "close_scene_card_id",
        "expected_section_id",
        "actual_section_id",
        "authorial_passage_id",
        "authorial_passage_ids",
    ):
        value = data.get(key)
        if value:
            feedback[key] = value
    return feedback


def _build_architecture_retry_feedback(exc: Exception) -> dict[str, Any]:
    data = getattr(exc, "data", None) or {}
    issue = str(data.get("issue", "architecture_contract_invalid"))
    if issue == "voice_first_unvoiceable":
        # Structured feedback for the voice_first preconditions: every
        # voice_first section needs at least one attached excerpt with a
        # non-empty verbatim_excerpt. Empty-verbatim excerpts (named-but-
        # not-quoted documents) cannot anchor a quoted opening.
        return {
            "issue": issue,
            "episode_number": data.get("episode_number"),
            "sections": data.get("sections", []),
            "instruction": (
                "Each section with open_mode=voice_first MUST attach at least "
                "one excerpt_id whose excerpt has a non-empty verbatim_excerpt. "
                "Empty-verbatim excerpts (named-but-not-quoted documents) cannot "
                "anchor a voice_first opening. For each failing section, either: "
                "(a) swap the section's excerpt_ids to include an excerpt with "
                "quoted text from the spine's assigned_excerpt_ids, or (b) "
                "downgrade open_mode to scene_anchor and choose a visible anchor "
                "(object, person, dated action, place) instead."
            ),
        }
    # Pydantic field-level validation errors: extract loc/msg/ctx so the
    # agent sees exactly which field failed and by what rule. Without this
    # the agent gets only the generic fallback below and tends to rewrite
    # the same offending field on retry.
    if isinstance(exc, ValidationError):
        lines: list[str] = []
        for err in exc.errors()[:8]:
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = str(err.get("msg", "validation error"))
            ctx = err.get("ctx") or {}
            ctx_bits = ", ".join(f"{k}={v}" for k, v in ctx.items())
            if ctx_bits:
                lines.append(f"- `{loc}`: {msg} ({ctx_bits})")
            else:
                lines.append(f"- `{loc}`: {msg}")
        total = len(exc.errors())
        suffix = f" (+{total - 8} more)" if total > 8 else ""
        return {
            "issue": "schema_validation_failed",
            "episode_number": data.get("episode_number"),
            "instruction": (
                "Pydantic schema validation failed on the previous attempt. "
                "Fix each named field on the next attempt and keep all other "
                "fields unchanged. Pay attention to max_length / min_length / "
                "max_items constraints on the offending fields.\n" + "\n".join(lines) + suffix
            ),
        }
    # Model-validator ValueError messages already include the field name
    # and the rule that failed (e.g. "verdict_landing may only appear in
    # the answer-stage section; found in section_05"). Pass them through
    # verbatim.
    if isinstance(exc, ValueError) and not data:
        return {
            "issue": "model_validator_failed",
            "episode_number": None,
            "instruction": (
                "A model-level validator rejected the previous attempt with "
                f"the following message. Fix exactly this failure and keep "
                f"all other requirements unchanged:\n{exc}"
            ),
        }
    feedback = {
        "issue": issue,
        "episode_number": data.get("episode_number"),
        "instruction": data.get(
            "instruction",
            "Return a schema-valid episode architecture that satisfies section counts, "
            "answer/close stage placement, and promised-beat accounting.",
        ),
    }
    for key in (
        "section_id",
        "section_ids",
        "expected_section_count_range",
        "promised_beat_ids",
    ):
        value = data.get(key)
        if value:
            feedback[key] = value
    return feedback


def _build_attempt_record(
    *,
    attempt: int,
    status: str,
    blocking_issue: str | None = None,
    warnings: list[str] | None = None,
    retry_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warning_list = list(warnings or [])
    return {
        "attempt": attempt,
        "status": status,
        "blocking_issue": blocking_issue,
        "warning_count": len(warning_list),
        "warnings": warning_list,
        "retry_feedback": retry_feedback,
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
    raw_values = {book.book_id: max(0.0, float(book.total_words)) for book in books}
    total = sum(raw_values.values())
    if total <= 0:
        even = 1.0 / len(books)
        return {book.book_id: even for book in books}
    return {book.book_id: (raw_values.get(book.book_id, 0.0) / total) for book in books}


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
            "citation_utilization_ratio": _ratio(
                len(axis_retained & cited_ids), len(axis_retained)
            ),
        }

    per_book: dict[str, Any] = {}
    for book in books:
        book_retained = {
            pid
            for pid in retained_ids
            if passage_by_id.get(pid) is not None and passage_by_id[pid].book_id == book.book_id
        }
        per_book[book.book_id] = {
            "title": book.title,
            "retained_count": len(book_retained),
            "planned_count": len(book_retained & planned_ids),
            "cited_count": len(book_retained & cited_ids),
            "plan_utilization_ratio": _ratio(len(book_retained & planned_ids), len(book_retained)),
            "citation_utilization_ratio": _ratio(
                len(book_retained & cited_ids), len(book_retained)
            ),
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
    beat_passages_by_id = {beat.beat_id: set(beat.passage_ids) for beat in plan.beats}
    beat_scene_by_id = {beat.beat_id: beat.scene_id for beat in plan.beats if beat.scene_id}

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
    insight_issues = [item for item in insight_results if item["status"] in {"weak", "zero"}]

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
        {citation.book_id for citation in script.citations if citation.book_id}
    )
    for idx in range(len(global_citation_books)):
        for jdx in range(idx + 1, len(global_citation_books)):
            observed_pairs.add((global_citation_books[idx], global_citation_books[jdx]))

    covered_pairs = sorted(planned_pairs & observed_pairs)
    planned_pair_count = len(planned_pairs)
    coverage_ratio = len(covered_pairs) / planned_pair_count if planned_pair_count > 0 else 1.0
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
        bool(plan.book_balance) and total_signals > 0 and max_abs_drift > max_book_balance_abs_drift
    )

    planned_scene_ids = [scene.scene_id for scene in plan.scene_cards]
    planned_scene_id_set = set(planned_scene_ids)
    observed_scene_id_set = {scene_id for scene_id in observed_scene_ids if scene_id}
    covered_scene_ids = [
        scene_id for scene_id in planned_scene_ids if scene_id in observed_scene_id_set
    ]
    collapsed_observed_scene_order: list[str] = []
    previous_scene_id: str | None = None
    for scene_id in observed_scene_ids:
        if scene_id == previous_scene_id:
            continue
        collapsed_observed_scene_order.append(scene_id)
        previous_scene_id = scene_id
    unexpected_scene_ids = [
        scene_id
        for scene_id in collapsed_observed_scene_order
        if scene_id not in planned_scene_id_set
    ]
    planned_scene_index = {scene_id: idx for idx, scene_id in enumerate(planned_scene_ids)}
    observed_scene_indices = [
        planned_scene_index[scene_id]
        for scene_id in collapsed_observed_scene_order
        if scene_id in planned_scene_index
    ]
    scene_order_preserved = observed_scene_indices == sorted(observed_scene_indices)
    scene_coverage_ratio = (
        len(covered_scene_ids) / len(planned_scene_ids) if planned_scene_ids else 1.0
    )
    anchor_scene_ids = [scene_id for scene_id in plan.anchor_scene_ids if scene_id]
    missing_anchor_scene_ids = [
        scene_id for scene_id in anchor_scene_ids if scene_id not in observed_scene_id_set
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
            "covered_anchor_scene_count": (len(anchor_scene_ids) - len(missing_anchor_scene_ids)),
            "missing_scene_ids": [
                scene_id for scene_id in planned_scene_ids if scene_id not in observed_scene_id_set
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
                {"book_ids": [a, b]} for a, b in sorted(planned_pairs - observed_pairs)
            ],
        },
        "book_balance": {
            "has_issues": book_balance_has_issues,
            "insufficient_signal": insufficient_book_signal,
            "total_signals": total_signals,
            "signal_counts": signal_counts,
            "planned_balance": {
                book_id: round(float(share), 4) for book_id, share in plan.book_balance.items()
            },
            "observed_balance": {
                book_id: round(share, 4) for book_id, share in observed_balance.items()
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
    *,
    book_slot: int = 0,
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
                    chunk_id=_project_chunk_id(book_slot, global_index),
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


def _base36_encode(number: int, *, width: int) -> str:
    if number < 0:
        raise ValueError("base36 encoding requires a non-negative integer")
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    base = len(alphabet)
    digits = ["0"] * width
    current = number
    for index in range(width - 1, -1, -1):
        current, remainder = divmod(current, base)
        digits[index] = alphabet[remainder]
    if current:
        raise ValueError(f"value {number} exceeds {width}-digit base36 capacity")
    return "".join(digits)


def _project_chunk_id(book_slot: int, chunk_index: int) -> str:
    return f"{_base36_encode(book_slot, width=2)}{_base36_encode(chunk_index + 1, width=4)}"


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


def _supports_segment_tts_instructions(
    tts_provider: str,
    tts_model_name: str | None = None,
) -> bool:
    if tts_provider.strip().lower() not in {"openai", "openai-compatible"}:
        return False
    return supports_openai_tts_instructions(tts_model_name)


def _normalize_tts_instruction_text(
    text: str | None, *, max_chars: int | None = None
) -> str | None:
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


def _segment_hint_degradations(
    hints: SpeechHints,
    tts_provider: str,
    tts_model_name: str | None = None,
) -> list[str]:
    if _supports_segment_tts_instructions(tts_provider, tts_model_name):
        return []
    degradations: list[str] = []
    if (
        hints.style != "neutral"
        or hints.intensity != "none"
        or hints.render_strategy == "slow_clause"
    ):
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
            before = clean_text[: match.start()].strip()
            focus = clean_text[match.start() : match.end()].strip()
            after = clean_text[match.end() :].strip()
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

    targets = (
        emphasis_targets
        if emphasis_targets is not None
        else ([focus_phrase] if focus_phrase else hints.emphasis_targets[:3])
    )
    if targets:
        stress = hints.intensity if hints.intensity != "none" else "light"
        overlay.append(
            f"Give {stress} stress to these phrases when natural: {', '.join(repr(target) for target in targets)}."
        )
    effective_pronunciation_hints = (
        pronunciation_hints if pronunciation_hints is not None else hints.pronunciation_hints
    )
    if effective_pronunciation_hints:
        pronunciation_text = "; ".join(
            f"{hint.text} as {hint.spoken_as}" for hint in effective_pronunciation_hints[:4]
        )
        overlay.append(f"Use these pronunciations: {pronunciation_text}.")

    if overlay:
        parts.append("Segment guidance: " + " ".join(overlay))
    return _normalize_tts_instruction_text("\n\n".join(parts))


def _render_segments_for_spoken_segment(
    segment: Any,
    *,
    voice_id: str,
    speed: float,
    tts_provider: str,
    tts_model_name: str | None,
    base_instructions: str | None,
) -> list[RenderSegment]:
    hints = segment.speech_hints
    render_speed = _resolve_spoken_render_speed(hints.pace, speed)
    if hints.render_strategy == "slow_clause":
        render_speed = _resolve_spoken_render_speed("slower", render_speed)

    text_parts = _split_render_text(segment.text, hints)
    degradations = _segment_hint_degradations(hints, tts_provider, tts_model_name)
    supports_instructions = _supports_segment_tts_instructions(tts_provider, tts_model_name)
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


# ---------------------------------------------------------------------------
# Render manifest construction (Stage 15 — no LLM)
# ---------------------------------------------------------------------------


def _resolve_segment_voice(
    spoken_segment,  # type: SpokenSegment (imported lazily inside the body)
    default_voice_id: str,
    actor_voice_catalog: dict[str, str],
) -> str:
    """Resolve the per-RenderSegment voice id for a SpokenSegment.

    Primary segments use the episode-default voice. Actor segments use the
    voice id mapped to their ``speaker_id`` in ``actor_voice_catalog``,
    falling back to the default with a logged degradation when unknown.
    """
    from podcast_agent.schemas.models import SpokenSpeakerRole

    if spoken_segment.speaker_role == SpokenSpeakerRole.PRIMARY or not spoken_segment.speaker_id:
        return default_voice_id
    resolved = actor_voice_catalog.get(spoken_segment.speaker_id)
    if resolved:
        return resolved
    return default_voice_id


def build_render_manifest(
    spoken_script: SpokenScript,
    voice_id: str = "fable",
    speed: float = 1.0,
    words_per_minute: int = 130,
    base_instructions: str | None = None,
    tts_model_name: str | None = "tts-1-hd",
    actor_voice_catalog: dict[str, str] | None = None,
) -> RenderManifest:
    segments: list[RenderSegment] = []
    tts_provider = spoken_script.tts_provider
    framing = spoken_script.framing
    actor_voice_catalog = actor_voice_catalog or {}
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

    # Change 4: iterate SpokenSegments, not SpokenSections. Each
    # SpokenSegment becomes one or more RenderSegments (via
    # _render_segments_for_spoken_segment when render_strategy splits the
    # text). Per-segment voice override drives actor voicing. Speech hints
    # at the SpokenSegment level take precedence; SpokenSection-level hints
    # fall through so the agent can set tonal posture per section while
    # leaving most segments at defaults.
    for section in spoken_script.sections:
        section_hints = section.speech_hints
        section_has_default_hints = section_hints == SpeechHints()
        for spoken_segment in section.segments:
            seg_voice = _resolve_segment_voice(spoken_segment, voice_id, actor_voice_catalog)
            seg_instructions = spoken_segment.delivery_instructions
            if seg_instructions is None and section.section_delivery_instructions:
                seg_instructions = section.section_delivery_instructions
            effective_base_instructions = (
                seg_instructions if seg_instructions else base_instructions
            )
            # If the SpokenSegment carries default hints AND the parent
            # SpokenSection has non-default hints, inherit the section's
            # hints so per-section pacing / emphasis / pause overrides flow
            # through to the render segment without each segment having to
            # restate them.
            effective_hints = spoken_segment.speech_hints
            if effective_hints == SpeechHints() and not section_has_default_hints:
                effective_hints = section_hints
            segments.extend(
                _render_segments_for_spoken_segment(
                    SimpleNamespace(
                        segment_id=spoken_segment.segment_id,
                        text=spoken_segment.text,
                        speech_hints=effective_hints,
                    ),
                    voice_id=seg_voice,
                    speed=speed,
                    tts_provider=tts_provider,
                    tts_model_name=tts_model_name,
                    base_instructions=effective_base_instructions,
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
        scene.scene_id for scene in scene_cards if int(scene.estimated_duration_seconds) <= 0
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

    return {scene.scene_id: floor_targets[idx] for idx, scene in enumerate(scene_cards)}


def _scene_word_count_band_multipliers(
    scene: SceneCard | SceneCardDraft,
) -> tuple[float, float]:
    if scene.word_count_priority == WordCountPriority.TIGHT:
        return (0.95, 1.05)
    return (0.90, 1.10)


def _compute_scene_word_count_bands(
    scene_cards: list[SceneCard],
    episode_target_word_count: int,
) -> tuple[dict[str, int], dict[str, int]]:
    baseline_lower = _compute_scene_word_count_targets(
        scene_cards,
        episode_target_word_count,
        145.0,
    )
    baseline_higher = _compute_scene_word_count_targets(
        scene_cards,
        episode_target_word_count,
        160.0,
    )
    widened_lower: dict[str, int] = {}
    widened_higher: dict[str, int] = {}
    for scene in scene_cards:
        baseline_center = (
            float(baseline_lower.get(scene.scene_id, 0))
            + float(baseline_higher.get(scene.scene_id, 0))
        ) / 2.0
        lower_mult, higher_mult = _scene_word_count_band_multipliers(scene)
        widened_lower[scene.scene_id] = max(1, int(math.floor(baseline_center * lower_mult)))
        widened_higher[scene.scene_id] = max(
            widened_lower[scene.scene_id],
            int(math.ceil(baseline_center * higher_mult)),
        )
    return widened_lower, widened_higher


_HEAVY_AUTHORIAL_PASSAGE_MODES: frozenset[str] = frozenset(
    {
        "quote_then_gloss",
        "doctrinal_unpack",
        "institutional_clarifier",
        "comparative_aside",
        "verdict_landing",
    }
)
_COMPARATIVE_ASIDE_MIN_WORDS_PER_SENTENCE = 18


_PACK_ROLE_WEIGHTS: dict[str, float] = {
    "core": 1.75,
    "stakes": 0.95,
    "mechanism": 0.90,
    "counterpressure": 0.95,
    "consequence": 0.95,
    "texture": 0.60,
    "recall": 0.65,
}
_PACK_ROLE_SPILLOVER_WEIGHTS: dict[str, float] = {
    "core": 1.15,
    "stakes": 0.25,
    "mechanism": 0.25,
    "counterpressure": 0.20,
    "consequence": 0.20,
    "texture": 0.00,
    "recall": 0.00,
}
_DEFAULT_PACK_WEIGHT = 1.00

_SCENE_ROLE_WEIGHTS: dict[str, float] = {
    "context_setup": 0.95,
    "actor_setup": 1.00,
    "shock": 1.20,
    "action": 1.10,
    "fallout": 1.10,
    "contestation": 1.05,
    "reaction": 1.00,
    "implication": 1.05,
    "reversal": 1.10,
    "reveal": 1.05,
    "stage_choice": 1.05,
    "perspective_shift": 0.95,
}
_PRESENCE_WEIGHTS: dict[str, float] = {
    "primary": 1.30,
    "secondary": 1.00,
    "background": 0.80,
    "none": 0.65,
}
_ARC_BINDING_WEIGHTS: dict[str, float] = {
    "strong": 1.50,
    "standard": 1.20,
    "light": 1.00,
    "none": 0.85,
}

# Bounds are multipliers of avg_sec = target_duration_seconds / n_cards so
# the allocator self-calibrates to episode length and card count.
_BOUNDS_MULT_BY_PRESENCE_BINDING: dict[tuple[str, str], tuple[float, float]] = {
    ("primary", "strong"): (0.90, 1.44),
    ("primary", "standard"): (0.66, 1.20),
    ("primary", "light"): (0.54, 0.99),
    ("primary", "none"): (0.48, 0.84),
    ("secondary", "strong"): (0.60, 1.08),
    ("secondary", "standard"): (0.51, 0.90),
    ("secondary", "light"): (0.42, 0.78),
    ("secondary", "none"): (0.36, 0.66),
    ("background", "strong"): (0.42, 0.78),
    ("background", "standard"): (0.36, 0.66),
    ("background", "light"): (0.33, 0.60),
    ("background", "none"): (0.30, 0.54),
}

# For no-actor cards, bounds key on what the scene is trying to do.
_NO_ACTOR_BOUNDS_MULT_BY_BUCKET: dict[str, tuple[float, float]] = {
    "argument": (0.27, 0.54),
    "mid": (0.45, 0.90),
    "action": (0.60, 1.20),
}
_ARGUMENT_SCENE_JOBS: frozenset[str] = frozenset({"build", "answer", "close"})
_ARGUMENT_ROLES: frozenset[str] = frozenset({"implication", "perspective_shift"})
_ACTION_BUCKET_ROLES: frozenset[str] = frozenset(
    {"shock", "action", "fallout", "reveal", "reversal", "stage_choice"}
)
_STRUCTURAL_SCENE_JOBS: frozenset[str] = frozenset({"turn", "answer", "close"})
_GROUNDING_SCENE_JOBS: frozenset[str] = frozenset({"opening", "build", "turn"})
_GROUNDING_SCENE_ROLES: frozenset[str] = frozenset(
    {"actor_setup", "action", "shock", "reaction", "fallout"}
)

_WRITING_WORD_OVERRUN_WARNING_RATIO = 1.08


def _positive_allocation_weights(weights: list[float]) -> list[float]:
    return [max(float(weight), 0.0001) for weight in weights]


def _bounded_weighted_allocation(
    *,
    total: float,
    weights: list[float],
    bounds: list[tuple[float, float]],
    overflow_weights: list[float] | None = None,
) -> list[float]:
    if not weights:
        return []
    if len(weights) != len(bounds):
        raise ValueError("weights and bounds must have the same length")
    if overflow_weights is not None and len(overflow_weights) != len(weights):
        raise ValueError("overflow_weights and weights must have the same length")

    positive_weights = _positive_allocation_weights(weights)
    positive_overflow_weights = (
        [max(float(weight), 0.0) for weight in overflow_weights]
        if overflow_weights is not None
        else positive_weights
    )
    min_total = sum(lower for lower, _ in bounds)
    max_total = sum(upper for _, upper in bounds)
    if total <= min_total:
        scale = total / min_total if min_total > 0 else 0.0
        return [lower * scale for lower, _ in bounds]
    if total >= max_total:
        allocations = [upper for _, upper in bounds]
        stranded_total = total - max_total
        return _redistribute_stranded_allocation(
            allocations,
            stranded_total,
            positive_weights,
            positive_overflow_weights,
        )

    allocations: list[float | None] = [None] * len(weights)
    active = set(range(len(weights)))
    remaining_total = float(total)

    while active:
        active_weight = sum(positive_weights[idx] for idx in active)
        if active_weight <= 0:
            equal_share = remaining_total / len(active)
            for idx in active:
                allocations[idx] = equal_share
            remaining_total = 0.0
            break

        changed = False
        for idx in list(active):
            lower, upper = bounds[idx]
            share = remaining_total * positive_weights[idx] / active_weight
            if share < lower:
                allocations[idx] = lower
                remaining_total -= lower
                active.remove(idx)
                changed = True
            elif share > upper:
                allocations[idx] = upper
                remaining_total -= upper
                active.remove(idx)
                changed = True

        if not changed:
            for idx in active:
                allocations[idx] = remaining_total * positive_weights[idx] / active_weight
            remaining_total = 0.0
            break

    resolved_allocations = [float(value or 0.0) for value in allocations]
    stranded_total = float(total) - sum(resolved_allocations)
    if abs(stranded_total) > 1e-6:
        return _redistribute_stranded_allocation(
            resolved_allocations,
            stranded_total,
            positive_weights,
            positive_overflow_weights,
        )
    return resolved_allocations


def _redistribute_stranded_allocation(
    allocations: list[float],
    stranded_total: float,
    fallback_weights: list[float],
    overflow_weights: list[float],
) -> list[float]:
    if abs(stranded_total) <= 1e-6:
        return allocations
    redistribution_weights = (
        overflow_weights if stranded_total > 0 and sum(overflow_weights) > 0 else fallback_weights
    )
    redistribution_weight = sum(redistribution_weights)
    if redistribution_weight <= 0:
        equal_share = stranded_total / len(allocations)
        return [allocation + equal_share for allocation in allocations]
    return [
        allocation + stranded_total * redistribution_weights[idx] / redistribution_weight
        for idx, allocation in enumerate(allocations)
    ]


def _round_allocations_to_total(
    allocations: list[float],
    target_total: int,
) -> list[int]:
    if not allocations:
        return []

    floor_values = [max(1, int(math.floor(value))) for value in allocations]
    delta = int(target_total) - sum(floor_values)
    fractions = sorted(
        range(len(allocations)),
        key=lambda idx: (allocations[idx] - math.floor(allocations[idx]), -idx),
        reverse=True,
    )

    if delta > 0:
        base_increment, remainder = divmod(delta, len(fractions))
        if base_increment:
            for idx in fractions:
                floor_values[idx] += base_increment
        for idx in fractions[:remainder]:
            floor_values[idx] += 1
    elif delta < 0:
        for idx in reversed(fractions):
            if delta == 0:
                break
            removable = min(floor_values[idx] - 1, -delta)
            if removable <= 0:
                continue
            floor_values[idx] -= removable
            delta += removable
        if delta != 0:
            raise ValueError("cannot round allocations to target without non-positive durations")

    return floor_values


def _scene_occurrence_id(scene: SceneCardDraft | SceneCard) -> str | None:
    return scene.section_id or scene.scene_id


def _episode_pack_role(strategy_episode: StrategyEpisode, pack_id: str) -> str | None:
    if pack_id in strategy_episode.episode_spine.core_primitive_ids:
        return "core"
    support_role = strategy_episode.episode_spine.support_primitive_roles.get(pack_id)
    if support_role is not None:
        return support_role.value
    if pack_id in strategy_episode.episode_spine.recall_primitive_ids:
        return "recall"
    return None


def _scene_pack_id_for_episode(
    scene: SceneCardDraft | SceneCard,
    strategy_episode: StrategyEpisode,
) -> str | None:
    assigned = set(strategy_episode.episode_spine.assigned_primitive_ids)
    for fact in scene.must_land_facts.ordered_facts():
        if fact in assigned:
            return fact
    return None


def _occurrence_weight(strategy_episode: StrategyEpisode, occurrence_id: str) -> float:
    role = _episode_pack_role(strategy_episode, occurrence_id)
    if role is None:
        return _DEFAULT_PACK_WEIGHT
    return _PACK_ROLE_WEIGHTS.get(role, _DEFAULT_PACK_WEIGHT)


def _occurrence_spillover_weight(
    strategy_episode: StrategyEpisode,
    occurrence_id: str,
) -> float:
    role = _episode_pack_role(strategy_episode, occurrence_id)
    if role is None:
        return 0.0
    return _PACK_ROLE_SPILLOVER_WEIGHTS.get(role, 0.0)


def _scene_max_presence(scene: SceneCardDraft | SceneCard) -> str:
    ranks = {"primary": 3, "secondary": 2, "background": 1}
    best = "none"
    for actor in scene.actors:
        if ranks.get(actor.presence, 0) > ranks.get(best, 0):
            best = actor.presence
    return best


_HOST_MOVE_PHASE_ORDER = ("open", "pivot", "close")


def _iter_host_move_cues(
    scene: SceneCardDraft | SceneCard,
) -> list[tuple[str, Any]]:
    cues: list[tuple[str, Any]] = []
    for phase in _HOST_MOVE_PHASE_ORDER:
        for cue in getattr(scene.host_moves, phase):
            cues.append((phase, cue))
    return cues


def _scene_host_phase_bucket_count(scene: SceneCardDraft | SceneCard) -> int:
    return sum(1 for phase in _HOST_MOVE_PHASE_ORDER if getattr(scene.host_moves, phase))


def _scene_has_full_host_phase_coverage(scene: SceneCardDraft | SceneCard) -> bool:
    return _scene_host_phase_bucket_count(scene) == len(_HOST_MOVE_PHASE_ORDER)


def _scene_host_intensity(scene: SceneCardDraft | SceneCard) -> str:
    move_types = {cue.move_type for _, cue in _iter_host_move_cues(scene)}
    if not move_types:
        return "none"
    if move_types == {"light_aside"}:
        return "light"
    if move_types.issubset({"orient", "clarify", "naming_note", "light_aside"}):
        return "none"
    return "standard"


_HOST_MOVE_META_NOTE_PATTERNS = (
    "hand the listener forward",
    "name the mechanism",
    "call it ",
    "the pattern is",
    "what we are seeing",
    "read this as",
)
_HOST_MOVE_LOW_YIELD_ROLES = {
    "context_setup",
    "actor_setup",
    "action",
    "shock",
    "reaction",
}
_HOST_NOTE_EDITORIAL_CONTROL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(state the through[- ]line|mark the math|land the lens|name the move|name the lens|define it carefully)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btell the listener\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:to watch for|they are about to watch)\b", re.IGNORECASE),
    re.compile(r"\bpause the chronology\b", re.IGNORECASE),
)
_HOST_NOTE_EPISODE_MANAGEMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(rest of the episode|rest of the hour|next section|next scene|what comes next|later in the series)\b",
        re.IGNORECASE,
    ),
)
_HOST_NOTE_ABSTRACT_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthrough[- ]line\b", re.IGNORECASE),
    re.compile(r"\blens\b", re.IGNORECASE),
    re.compile(r"\bmath\b", re.IGNORECASE),
    re.compile(r"\bverdict\b", re.IGNORECASE),
    re.compile(r"\bmechanism\b", re.IGNORECASE),
    re.compile(r"\bstructural (choice|fact|finding)\b", re.IGNORECASE),
)
_HOST_NOTE_ANCHOR_IGNORE_TOKENS = {
    "after",
    "already",
    "before",
    "carefully",
    "carry",
    "chronology",
    "cleanly",
    "define",
    "episode",
    "guide",
    "hold",
    "honestly",
    "hour",
    "land",
    "later",
    "lens",
    "line",
    "listener",
    "mark",
    "math",
    "mechanism",
    "move",
    "name",
    "next",
    "pause",
    "plainly",
    "rest",
    "section",
    "series",
    "show",
    "state",
    "tell",
    "through",
    "use",
    "verdict",
    "walk",
    "watch",
    "what",
}


def _host_move_note_looks_meta(note: str) -> bool:
    normalized = " ".join(str(note or "").lower().split())
    return any(pattern in normalized for pattern in _HOST_MOVE_META_NOTE_PATTERNS)


def _scene_host_anchor_tokens(scene: SceneCardDraft | SceneCard) -> set[str]:
    anchor_texts = [
        scene.title,
        scene.entry_image,
        scene.observable_detail,
        scene.location or "",
        scene.timeframe or "",
        *scene.must_land_facts.ordered_facts(),
        *(actor.name for actor in scene.actors),
    ]
    return _normalize_section_text_tokens(" ".join(text for text in anchor_texts if text))


def _host_move_editorial_scaffolding_flags(
    note: str,
    *,
    anchor_tokens: set[str],
) -> list[str]:
    normalized_note = " ".join(str(note or "").split())
    if not normalized_note:
        return []

    flags: list[str] = []
    if any(pattern.search(normalized_note) for pattern in _HOST_NOTE_EDITORIAL_CONTROL_PATTERNS):
        flags.append("editorial_control_phrase")
    if any(pattern.search(normalized_note) for pattern in _HOST_NOTE_EPISODE_MANAGEMENT_PATTERNS):
        flags.append("episode_management_phrase")
    if any(pattern.search(normalized_note) for pattern in _HOST_NOTE_ABSTRACT_TARGET_PATTERNS):
        flags.append("abstract_target_noun")

    note_tokens = _normalize_section_text_tokens(normalized_note)
    meaningful_note_tokens = note_tokens - _HOST_NOTE_ANCHOR_IGNORE_TOKENS
    if anchor_tokens and not (meaningful_note_tokens & anchor_tokens):
        flags.append("missing_scene_anchor_overlap")
    return flags


def _build_host_move_plan_diagnostics(
    *,
    scene_cards: list[SceneCardDraft | SceneCard],
    architecture: EpisodeArchitecture,
    narrator_profile: SeriesNarratorProfile,
) -> tuple[dict[str, Any], list[str]]:
    counts_by_type: dict[str, int] = {}
    counts_by_phase: dict[str, int] = {phase: 0 for phase in _HOST_MOVE_PHASE_ORDER}
    meta_target_phase_ids: list[str] = []
    editorial_scaffolding_phase_ids: list[str] = []
    first_person_plural_target_scene_ids: list[str] = []
    disallowed_move_scene_ids: list[str] = []
    personalized_scene_ids: list[str] = []
    scenes_with_1_phase: list[str] = []
    scenes_with_2_plus_phases: list[str] = []
    total_host_phase_count = 0
    host_target_too_long_ids: list[str] = []
    host_target_sentence_shaped_ids: list[str] = []
    host_target_verb_led_ids: list[str] = []
    host_target_abstract_ids: list[str] = []
    host_phase_multiple_cues: list[str] = []
    host_phase_overcoverage_unjustified: list[str] = []
    host_target_closure_override_pressure: list[str] = []
    allowed_moves = set(effective_narrator_allowed_moves(narrator_profile.allowed_moves))

    for scene in scene_cards:
        anchor_tokens = _scene_host_anchor_tokens(scene)
        phase_bucket_count = _scene_host_phase_bucket_count(scene)
        if phase_bucket_count == 1:
            scenes_with_1_phase.append(scene.scene_id)
        elif phase_bucket_count >= 2:
            scenes_with_2_plus_phases.append(scene.scene_id)
        if phase_bucket_count > 1 and scene.scene_job == SceneJob.BUILD:
            host_phase_overcoverage_unjustified.append(scene.scene_id)

        scene_is_personalized = False
        scene_has_first_person_plural_target = False
        scene_has_disallowed_move = False
        for phase in _HOST_MOVE_PHASE_ORDER:
            phase_cues = list(getattr(scene.host_moves, phase))
            if len(phase_cues) > 1:
                host_phase_multiple_cues.append(f"{scene.scene_id}:{phase}")
            for cue in phase_cues:
                total_host_phase_count += 1
                counts_by_phase[phase] = counts_by_phase.get(phase, 0) + 1
                counts_by_type[cue.move_type] = counts_by_type.get(cue.move_type, 0) + 1
                normalized_target = " ".join(cue.target.lower().split())
                if normalized_target and re.search(r"\b(we|our|us)\b", normalized_target):
                    scene_has_first_person_plural_target = True
                if cue.address_mode in {"we", "you", "i"}:
                    scene_is_personalized = True
                if cue.move_type not in allowed_moves:
                    scene_has_disallowed_move = True
                if _text_word_count(cue.target) > 6:
                    host_target_too_long_ids.append(f"{scene.scene_id}:{phase}")
                if len(re.findall(r"[.!?]", cue.target)) > 0 or _text_word_count(cue.target) > 4:
                    host_target_sentence_shaped_ids.append(f"{scene.scene_id}:{phase}")
                if re.match(
                    r"^(use|let|enter|keep|tell|state|mark|name|define)\b",
                    normalized_target,
                ):
                    host_target_verb_led_ids.append(f"{scene.scene_id}:{phase}")
                if _host_move_note_looks_meta(cue.target):
                    meta_target_phase_ids.append(f"{scene.scene_id}:{phase}")
                target_flags = _host_move_editorial_scaffolding_flags(
                    cue.target,
                    anchor_tokens=anchor_tokens,
                )
                if target_flags:
                    editorial_scaffolding_phase_ids.append(f"{scene.scene_id}:{phase}")
                if "abstract_target_noun" in target_flags:
                    host_target_abstract_ids.append(f"{scene.scene_id}:{phase}")
                if scene.scene_job == SceneJob.CLOSE and (
                    phase != "close" or cue.move_type in {"clarify", "contrast"}
                ):
                    host_target_closure_override_pressure.append(f"{scene.scene_id}:{phase}")
        if scene_has_first_person_plural_target:
            first_person_plural_target_scene_ids.append(scene.scene_id)
        if scene_is_personalized:
            personalized_scene_ids.append(scene.scene_id)
        if scene_has_disallowed_move:
            disallowed_move_scene_ids.append(scene.scene_id)

    warnings: list[str] = []
    scene_card_count = len(scene_cards)
    if meta_target_phase_ids:
        warnings.append(f"host_phase_meta_targets_detected: {_preview_ids(meta_target_phase_ids)}")
    if editorial_scaffolding_phase_ids:
        warnings.append(
            "host_phase_editorial_scaffolding_detected: "
            f"{_preview_ids(sorted(set(editorial_scaffolding_phase_ids)))}"
        )
    if host_target_too_long_ids:
        warnings.append(
            f"host_target_too_long: {_preview_ids(sorted(set(host_target_too_long_ids)))}"
        )
    if host_target_sentence_shaped_ids:
        warnings.append(
            "host_target_sentence_shaped: "
            f"{_preview_ids(sorted(set(host_target_sentence_shaped_ids)))}"
        )
    if host_target_verb_led_ids:
        warnings.append(
            f"host_target_verb_led: {_preview_ids(sorted(set(host_target_verb_led_ids)))}"
        )
    if host_phase_multiple_cues:
        warnings.append(
            f"host_phase_multiple_cues: {_preview_ids(sorted(set(host_phase_multiple_cues)))}"
        )
    if host_phase_overcoverage_unjustified:
        warnings.append(
            "host_phase_overcoverage_unjustified: "
            f"{_preview_ids(sorted(set(host_phase_overcoverage_unjustified)))}"
        )
    if host_target_abstract_ids:
        warnings.append(
            f"host_target_abstract: {_preview_ids(sorted(set(host_target_abstract_ids)))}"
        )
    if host_target_closure_override_pressure:
        warnings.append(
            "host_target_closure_override_pressure: "
            f"{_preview_ids(sorted(set(host_target_closure_override_pressure)))}"
        )
    if disallowed_move_scene_ids:
        warnings.append(
            f"host_move_allowed_move_mismatch: {_preview_ids(disallowed_move_scene_ids)}"
        )
    orient_callback_count = counts_by_type.get("orient", 0) + counts_by_type.get("callback", 0)
    total_cue_count = sum(counts_by_type.values())
    if total_cue_count and orient_callback_count / total_cue_count > 0.5:
        warnings.append(
            "host_move_monotone_distribution: "
            f"orient={counts_by_type.get('orient', 0)} "
            f"callback={counts_by_type.get('callback', 0)} "
            f"clarify={counts_by_type.get('clarify', 0)} "
            f"contrast={counts_by_type.get('contrast', 0)} "
            f"evaluate={counts_by_type.get('evaluate', 0)} "
            f"naming_note={counts_by_type.get('naming_note', 0)}"
        )
    host_shaped_scene_count = len(scenes_with_1_phase) + len(scenes_with_2_plus_phases)
    if host_shaped_scene_count and len(personalized_scene_ids) < max(
        1, int(host_shaped_scene_count * 0.2)
    ):
        warnings.append(
            "host_move_underpersonalized: "
            f"{len(personalized_scene_ids)}/{host_shaped_scene_count} host-shaped "
            "scenes use explicit `we` or `you` address"
        )

    diagnostics = {
        "scene_card_count": scene_card_count,
        "host_shaped_scene_count": host_shaped_scene_count,
        "total_host_phase_count": total_host_phase_count,
        "counts_by_type": counts_by_type,
        "counts_by_phase": counts_by_phase,
        "scenes_with_1_phase_count": len(scenes_with_1_phase),
        "scenes_with_2_plus_phases_count": len(scenes_with_2_plus_phases),
        "meta_target_phase_ids": meta_target_phase_ids,
        "editorial_scaffolding_phase_ids": sorted(set(editorial_scaffolding_phase_ids)),
        "first_person_plural_target_scene_ids": first_person_plural_target_scene_ids,
        "personalized_scene_ids": personalized_scene_ids,
        "disallowed_move_scene_ids": disallowed_move_scene_ids,
        "host_target_too_long_ids": sorted(set(host_target_too_long_ids)),
        "host_target_sentence_shaped_ids": sorted(set(host_target_sentence_shaped_ids)),
        "host_target_verb_led_ids": sorted(set(host_target_verb_led_ids)),
        "host_phase_multiple_cues": sorted(set(host_phase_multiple_cues)),
        "host_phase_overcoverage_unjustified": sorted(set(host_phase_overcoverage_unjustified)),
        "host_target_abstract_ids": sorted(set(host_target_abstract_ids)),
        "host_target_closure_override_pressure": sorted(set(host_target_closure_override_pressure)),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return diagnostics, warnings


def _section_has_host_state_moves(section: Any) -> bool:
    state_effects = getattr(getattr(section, "section_progression", None), "state_effects", None)
    if state_effects is None:
        return False
    return bool(
        getattr(state_effects, "host_mystery_moves", [])
        or getattr(state_effects, "host_assumption_moves", [])
        or getattr(state_effects, "host_theory_moves", [])
    )


def _build_host_density_diagnostics(
    *,
    scene_cards: list[SceneCardDraft | SceneCard],
    architecture: EpisodeArchitecture,
) -> tuple[dict[str, Any], list[str]]:
    section_by_id = {section.section_id: section for section in architecture.sections}
    verdict_authorial_ids = {
        passage.authorial_passage_id
        for section in architecture.sections
        for passage in section.authorial_passages
        if passage.mode == "verdict_landing"
    }
    build_scene_ids_with_three_populated_phases: list[str] = []
    build_scene_ids_with_two_populated_phases_without_justification: list[str] = []
    explicit_verdict_scene_ids: list[str] = []
    for scene in scene_cards:
        phase_count = _scene_host_phase_bucket_count(scene)
        section = section_by_id.get(scene.section_id)
        if scene.scene_job == SceneJob.BUILD:
            if phase_count >= 3:
                build_scene_ids_with_three_populated_phases.append(scene.scene_id)
            elif phase_count == 2:
                justified = bool(scene.authorial_passage_ids) or (
                    section is not None and _section_has_host_state_moves(section)
                )
                if not justified:
                    build_scene_ids_with_two_populated_phases_without_justification.append(
                        scene.scene_id
                    )
        has_explicit_close_evaluate = any(
            cue.move_type == "evaluate" and cue.surface_mode in {"mixed", "distinct"}
            for cue in scene.host_moves.close
        )
        has_verdict_authorial = bool(verdict_authorial_ids & set(scene.authorial_passage_ids))
        if has_explicit_close_evaluate or has_verdict_authorial:
            explicit_verdict_scene_ids.append(scene.scene_id)
    warnings: list[str] = []
    if build_scene_ids_with_three_populated_phases:
        warnings.append(
            "build_scene_phase_cap_exceeded: "
            f"{_preview_ids(build_scene_ids_with_three_populated_phases)}"
        )
    if build_scene_ids_with_two_populated_phases_without_justification:
        warnings.append(
            "build_scene_phase_overcoverage_without_obligation: "
            f"{_preview_ids(build_scene_ids_with_two_populated_phases_without_justification)}"
        )
    if len(explicit_verdict_scene_ids) > 4:
        warnings.append(
            "explicit_verdict_scene_cap_exceeded: "
            f"count={len(explicit_verdict_scene_ids)} "
            f"ids={_preview_ids(explicit_verdict_scene_ids)}"
        )
    diagnostics = {
        "build_scene_ids_with_three_populated_phases": build_scene_ids_with_three_populated_phases,
        "build_scene_ids_with_two_populated_phases_without_justification": (
            build_scene_ids_with_two_populated_phases_without_justification
        ),
        "explicit_verdict_scene_ids": explicit_verdict_scene_ids,
        "explicit_verdict_scene_count": len(explicit_verdict_scene_ids),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return diagnostics, warnings


def _no_actor_bucket(scene: SceneCardDraft | SceneCard) -> str:
    if scene.scene_job in _ARGUMENT_SCENE_JOBS or scene.scene_role in _ARGUMENT_ROLES:
        return "argument"
    if scene.scene_job == "turn" or scene.scene_role in _ACTION_BUCKET_ROLES:
        return "action"
    return "mid"


def _scene_duration_bounds(
    scene: SceneCardDraft | SceneCard,
    strategy_episode: StrategyEpisode,
    avg_sec: float,
) -> tuple[float, float]:
    presence = _scene_max_presence(scene)
    if presence == "none":
        bucket = _no_actor_bucket(scene)
        lower_mult, upper_mult = _NO_ACTOR_BOUNDS_MULT_BY_BUCKET[bucket]
    else:
        binding = _scene_host_intensity(scene)
        lower_mult, upper_mult = _BOUNDS_MULT_BY_PRESENCE_BINDING.get(
            (presence, binding),
            (0.27, 0.54),
        )
    return lower_mult * avg_sec, upper_mult * avg_sec


def _scene_importance_weight(scene: SceneCardDraft | SceneCard) -> float:
    presence = _scene_max_presence(scene)
    binding = _scene_host_intensity(scene)
    role_weight = _SCENE_ROLE_WEIGHTS.get(scene.scene_role, 1.0)
    presence_weight = _PRESENCE_WEIGHTS[presence]
    if presence == "none":
        return presence_weight * role_weight
    return presence_weight * _ARC_BINDING_WEIGHTS.get(binding, 0.85) * role_weight


def _allocate_scene_durations(
    scene_cards: list[SceneCardDraft | SceneCard],
    strategy_episode: StrategyEpisode,
    target_duration_seconds: int,
) -> list[SceneCard]:
    if not scene_cards:
        return []
    if target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be positive")
    avg_sec = float(target_duration_seconds) / len(scene_cards)

    occurrence_ids: list[str] = []
    seen_occurrence_ids: set[str] = set()
    for scene in scene_cards:
        occurrence_id = _scene_occurrence_id(scene) or "__unassigned__"
        if occurrence_id in seen_occurrence_ids:
            continue
        seen_occurrence_ids.add(occurrence_id)
        occurrence_ids.append(occurrence_id)

    occurrence_weights = [
        _occurrence_weight(strategy_episode, occurrence_id) for occurrence_id in occurrence_ids
    ]
    occurrence_spillover_weights = [
        _occurrence_spillover_weight(strategy_episode, occurrence_id)
        for occurrence_id in occurrence_ids
    ]
    occurrence_bounds: list[tuple[float, float]] = []
    for occurrence_id in occurrence_ids:
        occurrence_scene_cards = [
            scene
            for scene in scene_cards
            if (_scene_occurrence_id(scene) or "__unassigned__") == occurrence_id
        ]
        bounds = [
            _scene_duration_bounds(scene, strategy_episode, avg_sec)
            for scene in occurrence_scene_cards
        ]
        occurrence_bounds.append(
            (
                sum(lower for lower, _ in bounds),
                sum(upper for _, upper in bounds),
            )
        )

    occurrence_allocations = _bounded_weighted_allocation(
        total=float(target_duration_seconds),
        weights=occurrence_weights,
        bounds=occurrence_bounds,
        overflow_weights=occurrence_spillover_weights,
    )
    occurrence_seconds_by_id = dict(zip(occurrence_ids, occurrence_allocations, strict=True))

    raw_scene_seconds_by_index: dict[int, float] = {}
    for occurrence_id in occurrence_ids:
        occurrence_scene_entries = [
            (idx, scene)
            for idx, scene in enumerate(scene_cards)
            if (_scene_occurrence_id(scene) or "__unassigned__") == occurrence_id
        ]
        scene_weights = [_scene_importance_weight(scene) for _, scene in occurrence_scene_entries]
        scene_bounds = [
            _scene_duration_bounds(scene, strategy_episode, avg_sec)
            for _, scene in occurrence_scene_entries
        ]
        occurrence_scene_seconds = _bounded_weighted_allocation(
            total=occurrence_seconds_by_id[occurrence_id],
            weights=scene_weights,
            bounds=scene_bounds,
        )
        for (idx, _), seconds in zip(
            occurrence_scene_entries,
            occurrence_scene_seconds,
            strict=True,
        ):
            raw_scene_seconds_by_index[idx] = seconds

    raw_scene_seconds = [raw_scene_seconds_by_index[idx] for idx in range(len(scene_cards))]

    rounded_scene_seconds = _round_allocations_to_total(
        raw_scene_seconds,
        int(target_duration_seconds),
    )
    if sum(rounded_scene_seconds) != int(target_duration_seconds):
        raise ValueError(
            "allocated scene durations must sum to target_duration_seconds "
            f"({sum(rounded_scene_seconds)} != {int(target_duration_seconds)})"
        )
    return [
        SceneCard.model_validate(
            {
                **scene.model_dump(mode="json"),
                "estimated_duration_seconds": seconds,
            }
        )
        for scene, seconds in zip(scene_cards, rounded_scene_seconds, strict=True)
    ]


def _writing_result_word_count(result: Any) -> int:
    prose_sections = getattr(result, "prose_sections", []) or []
    return sum(len((getattr(section, "text", "") or "").split()) for section in prose_sections)


def _default_writing_movement_goal(
    *,
    architecture: EpisodeArchitecture,
    section_id: str,
) -> str:
    for section in architecture.sections:
        if section.section_id == section_id:
            return section.purpose.value
    return "continue"


def _normalize_writing_section_outputs(
    *,
    result: Any,
    architecture: EpisodeArchitecture,
    scene_cards: list[SceneCard],
    episode_number: int,
    skip_grounding: bool,
) -> list[dict[str, Any]]:
    prose_sections = getattr(result, "prose_sections", []) or []
    expected_sections: list[tuple[str, list[str]]] = []
    current_section_id: str | None = None
    current_scene_ids: list[str] = []
    for scene in scene_cards:
        if current_section_id is not None and scene.section_id != current_section_id:
            expected_sections.append((current_section_id, list(current_scene_ids)))
            current_scene_ids = []
        current_section_id = scene.section_id
        current_scene_ids.append(scene.scene_id)
    if current_section_id is not None:
        expected_sections.append((current_section_id, list(current_scene_ids)))

    normalized: list[dict[str, Any]] = []
    for index, section_output in enumerate(prose_sections):
        expected_section_id = ""
        expected_scene_ids: list[str] = []
        if index < len(expected_sections):
            expected_section_id, expected_scene_ids = expected_sections[index]

        section_id = str(getattr(section_output, "section_id", "") or "").strip()
        if not section_id:
            section_id = expected_section_id
        scene_card_ids = [
            str(scene_id) for scene_id in list(getattr(section_output, "scene_card_ids", []) or [])
        ]
        if not scene_card_ids:
            scene_card_ids = list(expected_scene_ids)
        movement_goal = str(getattr(section_output, "movement_goal", "") or "").strip()
        if not movement_goal:
            movement_goal = _default_writing_movement_goal(
                architecture=architecture,
                section_id=section_id,
            )
        payload = {
            "section_id": section_id,
            "scene_card_ids": scene_card_ids,
            "movement_goal": movement_goal,
            "text": str(section_output.text),
            "source_book_ids": list(getattr(section_output, "source_book_ids", []) or []),
            "actor_explanation_realizations": [
                ActorExplanationRealization.model_validate(
                    realization.model_dump(mode="json")
                    if hasattr(realization, "model_dump")
                    else realization
                )
                for realization in getattr(section_output, "actor_explanation_realizations", [])
                or []
            ],
        }
        if skip_grounding:
            payload["citations"] = []
        else:
            payload["citations"] = [
                Citation.model_validate(citation.model_dump(mode="json"))
                for citation in getattr(section_output, "citations", []) or []
            ]
        normalized.append(payload)
    returned_sections = [
        (payload["section_id"], list(payload["scene_card_ids"])) for payload in normalized
    ]
    if returned_sections != expected_sections:
        raise ComplianceViolationError(
            "Episode writing must return exactly one prose section per planned section window, with exact scene_card_ids in order "
            f"(episode {episode_number}); expected {expected_sections}, received {returned_sections}.",
            data={
                "episode_number": episode_number,
                "expected_sections": expected_sections,
                "returned_sections": returned_sections,
                "failed_part_number": None,
            },
        )
    return normalized


def _build_writing_retry_feedback(exc: ComplianceViolationError) -> str:
    data = exc.data or {}
    expected_sections = data.get("expected_sections")
    returned_sections = data.get("returned_sections")
    feedback_lines = [
        "Retry feedback:",
        str(exc),
    ]
    if expected_sections and returned_sections:
        feedback_lines.extend(
            [
                f"Expected section sequence: {expected_sections}",
                f"Returned section sequence: {returned_sections}",
            ]
        )
    feedback_lines.append(
        "Return exactly one prose section per input section, preserving section_id order and exact scene_card_ids."
    )
    return "\n".join(feedback_lines)


_ENDING_MARKER_RE = re.compile(r"\b(?:in the end|ultimately|finally|the verdict)\b", re.IGNORECASE)
_QUESTION_SENTENCE_RE = re.compile(r"[^?]+\?")


def _build_spine_script_diagnostics(
    *,
    strategy_episode: StrategyEpisode,
    plan: EpisodePlan,
    script: EpisodeScript,
) -> dict[str, Any]:
    expected_scene_ids = [scene.scene_id for scene in plan.scene_cards]
    rendered_scene_ids = [
        scene_id
        for section in script.prose_sections
        for scene_id in section.scene_card_ids
        if scene_id
    ]
    seen_rendered: set[str] = set()
    rendered_scene_order: list[str] = []
    for scene_id in rendered_scene_ids:
        if scene_id in seen_rendered:
            continue
        seen_rendered.add(scene_id)
        rendered_scene_order.append(scene_id)
    scene_order_preserved = rendered_scene_order == expected_scene_ids

    ending_marker_sections = [
        section.section_id
        for section in script.prose_sections
        if _ENDING_MARKER_RE.search(section.text or "")
    ]
    question_sentences = [
        match.group(0).strip()
        for section in script.prose_sections
        for match in _QUESTION_SENTENCE_RE.finditer(section.text or "")
    ]
    driving_problem = strategy_episode.episode_spine.listener_problem.strip().lower()
    new_load_bearing_question_detected = any(
        sentence.strip().lower() != driving_problem for sentence in question_sentences[1:]
    )
    last_section_scene_ids = (
        script.prose_sections[-1].scene_card_ids if script.prose_sections else []
    )
    ending_alignment_pass = bool(
        last_section_scene_ids and last_section_scene_ids[-1] == expected_scene_ids[-1]
    )
    second_ending_detected = len(ending_marker_sections) > 1
    return {
        "scene_order_preserved": scene_order_preserved,
        "ending_alignment_pass": ending_alignment_pass,
        "new_load_bearing_question_detected": new_load_bearing_question_detected,
        "second_ending_detected": second_ending_detected,
        "spine_drift_detected": (
            not scene_order_preserved
            or new_load_bearing_question_detected
            or second_ending_detected
            or not ending_alignment_pass
        ),
        "failure_labels": [
            label
            for label, enabled in [
                ("new_load_bearing_question", new_load_bearing_question_detected),
                ("second_ending", second_ending_detected),
                ("ending_displacement", not ending_alignment_pass),
                (
                    "multi_proposition_episode",
                    new_load_bearing_question_detected and second_ending_detected,
                ),
            ]
            if enabled
        ],
    }


def _validate_writing_context(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    plan: EpisodePlan,
    passage_lookup: dict[str, ExtractedPassage],
) -> None:
    if strategy_episode.episode_number != architecture.episode_number:
        raise RuntimeError(
            "Writing context strategy and architecture episode numbers do not match "
            f"({strategy_episode.episode_number} != {architecture.episode_number})."
        )
    if plan.episode_number != architecture.episode_number:
        raise RuntimeError(
            "Writing context plan and architecture episode numbers do not match "
            f"({plan.episode_number} != {architecture.episode_number})."
        )

    section_ids = {section.section_id for section in architecture.sections}
    for scene in plan.scene_cards:
        if scene.section_id not in section_ids:
            raise RuntimeError(
                "Writing context scene referenced unknown architecture section for "
                f"episode {plan.episode_number}: {scene.scene_id} -> {scene.section_id}"
            )
        unknown_passage_ids = sorted(
            passage_id for passage_id in scene.passage_ids if passage_id not in passage_lookup
        )
        if unknown_passage_ids:
            raise RuntimeError(
                "Writing context scene referenced unknown passage ids for episode "
                f"{plan.episode_number}: {scene.scene_id} -> {_preview_ids(unknown_passage_ids)}"
            )


def _build_passage_lookup(corpus: ThematicCorpus) -> dict[str, ExtractedPassage]:
    passage_lookup: dict[str, ExtractedPassage] = {}
    for axis_passages in corpus.passages_by_axis.values():
        for passage in axis_passages:
            passage_lookup[passage.passage_id] = passage
    return passage_lookup


def _primitive_passage_ids(primitive: SynthesisPrimitiveBase) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for passage_id in [*primitive.core_passage_ids, *primitive.support_passage_ids]:
        if not passage_id or passage_id in seen:
            continue
        seen.add(passage_id)
        ordered.append(passage_id)
    return ordered


def _iter_primitive_passage_refs(
    primitive: SynthesisPrimitiveBase,
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for passage_kind, passage_ids in (
        ("core", primitive.core_passage_ids),
        ("support", primitive.support_passage_ids),
    ):
        for passage_id in passage_ids:
            if not passage_id or passage_id in seen:
                continue
            seen.add(passage_id)
            ordered.append((passage_id, passage_kind))
    return ordered


def _build_episode_planning_passage_refs(
    *,
    primitive_ids_by_role: dict[str, list[str]],
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for episode_role in ("core", "support", "recall"):
        primitive_ids = primitive_ids_by_role.get(episode_role, [])
        for primitive_id in primitive_ids:
            primitive = primitive_lookup.get(primitive_id)
            if primitive is None:
                continue
            for passage_id, passage_kind in _iter_primitive_passage_refs(primitive):
                refs.append(
                    {
                        "passage_id": passage_id,
                        "episode_role": episode_role,
                        "passage_kind": passage_kind,
                        "primitive_id": primitive_id,
                    }
                )
    return refs


def _planning_passage_keep_fraction(episode_role: str, passage_kind: str) -> float:
    if episode_role == "core":
        return 0.50 if passage_kind == "core" else 0.33
    if episode_role in {"support", "recall"}:
        return 0.30 if passage_kind == "core" else 0.25
    raise ValueError(f"Unsupported episode planning role: {episode_role}")


def _build_episode_planning_passage_query_texts(
    *,
    episode_query_text: str,
    passage_refs: list[dict[str, str]],
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
) -> dict[str, str]:
    query_parts_by_passage_id: dict[str, list[str]] = {}
    seen_primitive_ids_by_passage_id: dict[str, set[str]] = {}
    for passage_ref in passage_refs:
        passage_id = passage_ref["passage_id"]
        primitive_id = passage_ref["primitive_id"]
        primitive = primitive_lookup.get(primitive_id)
        if primitive is None:
            continue
        seen_primitive_ids = seen_primitive_ids_by_passage_id.setdefault(passage_id, set())
        if primitive_id in seen_primitive_ids:
            continue
        seen_primitive_ids.add(primitive_id)
        query_parts = query_parts_by_passage_id.setdefault(passage_id, [episode_query_text])
        query_parts.append(primitive.title)
        query_parts.extend(_primitive_substrate_text_fragments(primitive))
        if primitive.timeframe:
            query_parts.append(primitive.timeframe)
        if primitive.geography:
            query_parts.append(primitive.geography)
    return {
        passage_id: " ".join(part.strip() for part in parts if part and part.strip()).strip()
        for passage_id, parts in query_parts_by_passage_id.items()
    }


def _build_episode_architecture_passage_refs(
    *,
    episode_spine: EpisodeSpine,
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    core_refs: list[dict[str, str]] = []
    for primitive_id in episode_spine.core_primitive_ids:
        primitive = primitive_lookup.get(primitive_id)
        if primitive is None:
            continue
        for passage_id in primitive.core_passage_ids:
            if not passage_id:
                continue
            core_refs.append(
                {
                    "passage_id": passage_id,
                    "episode_role": "core",
                    "passage_kind": "core",
                    "primitive_id": primitive_id,
                }
            )

    support_refs = _build_episode_planning_passage_refs(
        primitive_ids_by_role={
            "core": [],
            "support": list(episode_spine.support_primitive_roles.keys()),
            "recall": [],
        },
        primitive_lookup=primitive_lookup,
    )
    return core_refs, support_refs


def _build_episode_architecture_passage_payload(
    *,
    passage_refs: list[dict[str, str]],
    seen_passage_ids: set[str],
    passage_lookup: dict[str, ExtractedPassage],
    query_text: str,
    query_text_by_passage_id: dict[str, str],
    keep_fraction: float,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for passage_ref in passage_refs:
        passage_id = passage_ref["passage_id"]
        if passage_id in seen_passage_ids:
            continue
        passage = passage_lookup.get(passage_id)
        if passage is None:
            continue
        seen_passage_ids.add(passage_id)
        full_text = passage.full_text.strip() or passage.text
        trimmed_passage = {
            "passage_id": passage.passage_id,
            "text": full_text,
        }
        _trim_candidate_texts_by_bm25_query_text(
            query_text,
            [trimmed_passage],
            keep_fraction=keep_fraction,
            query_text_by_passage_id=query_text_by_passage_id,
        )
        summary_text = str(trimmed_passage.get("text", "")).strip() or full_text
        payload.append(
            {
                "passage_id": passage.passage_id,
                "primitive_id": passage_ref["primitive_id"],
                "book_id": passage.book_id,
                "chapter_ref": passage.chapter_ref,
                "summary_text": summary_text,
            }
        )
    return payload


def _ordered_architecture_primitive_ids(
    episode: EpisodeArchitecture,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for section in episode.sections:
        for primitive_id in section.primitive_ids:
            if primitive_id in seen:
                continue
            seen.add(primitive_id)
            ordered.append(primitive_id)
    return ordered


def _ordered_architecture_excerpt_ids(
    episode: EpisodeArchitecture,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for section in episode.sections:
        for excerpt_id in section.excerpt_ids:
            if excerpt_id in seen:
                continue
            seen.add(excerpt_id)
            ordered.append(excerpt_id)
    return ordered


def _strategy_episode_by_number(
    strategy: NarrativeStrategy,
    episode_number: int,
) -> StrategyEpisode:
    for episode in strategy.episodes:
        if episode.episode_number == episode_number:
            return episode
    raise RuntimeError(f"Missing strategy episode for episode_number={episode_number}")


def _validate_architecture_transition(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    excerpt_by_id: dict[str, ExcerptRecord] | None = None,
) -> EpisodeArchitecture:
    excerpt_by_id = excerpt_by_id or {}
    if architecture.episode_number != strategy_episode.episode_number:
        raise RuntimeError(
            "Episode architecture episode_number does not match strategy episode "
            f"({architecture.episode_number} != {strategy_episode.episode_number})."
        )

    assigned_primitive_ids = set(strategy_episode.episode_spine.assigned_primitive_ids)
    section_primitive_ids = set(_ordered_architecture_primitive_ids(architecture))
    invalid_primitive_ids = sorted(section_primitive_ids - assigned_primitive_ids)
    if invalid_primitive_ids:
        raise RuntimeError(
            "Episode architecture referenced primitives outside the strategy spine "
            f"for episode {architecture.episode_number}: {_preview_ids(invalid_primitive_ids)}"
        )

    assigned_excerpt_ids = set(strategy_episode.episode_spine.assigned_excerpt_ids)
    section_excerpt_ids = set(_ordered_architecture_excerpt_ids(architecture))
    invalid_excerpt_ids = sorted(section_excerpt_ids - assigned_excerpt_ids)
    if invalid_excerpt_ids:
        raise RuntimeError(
            "Episode architecture referenced excerpts outside the strategy spine "
            f"for episode {architecture.episode_number}: {_preview_ids(invalid_excerpt_ids)}"
        )

    # Voice_first preconditions: a section opened in `voice_first` mode requires
    # at least one attached excerpt with a non-empty verbatim_excerpt. Empty-
    # verbatim excerpts (e.g. named-but-not-quoted documents like a referenced
    # decree whose words aren't in the source passage) cannot anchor the literal
    # first beat the writer is asked to perform; route them elsewhere. Raised as
    # a ComplianceViolationError so the architecture agent's retry loop picks it
    # up — the model can swap in an excerpt with quoted text OR downgrade the
    # section to scene_anchor. The numerical quotability score is NOT gated on;
    # treat it as a soft ranking signal only.
    if excerpt_by_id:
        unvoiceable_voice_first: list[dict[str, Any]] = []
        for section in architecture.sections:
            if section.open_mode != "voice_first":
                continue
            attempted = []
            voiceable = False
            for xid in section.excerpt_ids:
                excerpt = excerpt_by_id.get(xid)
                if excerpt is None:
                    continue
                attempted.append(
                    {
                        "excerpt_id": xid,
                        "excerpt_type": excerpt.excerpt_type,
                        "has_verbatim": bool(excerpt.verbatim_excerpt.strip()),
                        "quotability": excerpt.quotability,
                    }
                )
                if excerpt.verbatim_excerpt.strip():
                    voiceable = True
                    break
            if not voiceable:
                unvoiceable_voice_first.append(
                    {
                        "section_id": section.section_id,
                        "attempted_excerpts": attempted,
                    }
                )
        if unvoiceable_voice_first:
            section_ids = [item["section_id"] for item in unvoiceable_voice_first]
            raise ComplianceViolationError(
                "Episode architecture has voice_first section(s) without a "
                "voiceable excerpt (non-empty verbatim_excerpt) for episode "
                f"{architecture.episode_number}: {_preview_ids(section_ids)}. "
                "Either swap in an excerpt with quoted text or downgrade those "
                "sections to scene_anchor.",
                data={
                    "issue": "voice_first_unvoiceable",
                    "episode_number": architecture.episode_number,
                    "sections": unvoiceable_voice_first,
                },
            )

    missing_core_primitive_ids = sorted(
        primitive_id
        for primitive_id in strategy_episode.episode_spine.core_primitive_ids
        if primitive_id not in section_primitive_ids
    )
    if missing_core_primitive_ids:
        raise RuntimeError(
            "Episode architecture omitted required core primitives for episode "
            f"{architecture.episode_number}: {_preview_ids(missing_core_primitive_ids)}"
        )

    invalid_recurring_primitive_ids = sorted(
        primitive_id
        for primitive_id in architecture.allowed_recurring_primitive_ids
        if primitive_id not in assigned_primitive_ids
    )
    if invalid_recurring_primitive_ids:
        raise RuntimeError(
            "Episode architecture allowed recurring primitives outside the strategy "
            f"spine for episode {architecture.episode_number}: "
            f"{_preview_ids(invalid_recurring_primitive_ids)}"
        )

    if not architecture.sections:
        raise RuntimeError(
            f"Episode architecture must contain sections for episode {architecture.episode_number}."
        )
    missing_section_anchor_ids = [
        section.section_id
        for section in architecture.sections
        if not section.section_anchor.strip()
    ]
    if missing_section_anchor_ids:
        raise RuntimeError(
            "Episode architecture omitted required section_anchor values for episode "
            f"{architecture.episode_number}: {_preview_ids(missing_section_anchor_ids)}"
        )

    missing_must_stage_beat_ids = [
        section.section_id for section in architecture.sections if not section.must_stage_beats
    ]
    if missing_must_stage_beat_ids:
        raise RuntimeError(
            "Episode architecture omitted required must_stage_beats for episode "
            f"{architecture.episode_number}: {_preview_ids(missing_must_stage_beat_ids)}"
        )

    last_section_purpose = getattr(
        architecture.sections[-1].purpose, "value", architecture.sections[-1].purpose
    )
    if last_section_purpose != "closing":
        raise RuntimeError(
            "Episode architecture must end with a closing section for episode "
            f"{architecture.episode_number}."
        )

    valid_section_ids = {section.section_id for section in architecture.sections}
    promised_beat_ids = [beat.beat_id for beat in strategy_episode.promised_beats]
    if promised_beat_ids:
        decision_by_id = {
            decision.beat_id: decision for decision in architecture.promised_beat_decisions
        }
        missing_decision_ids = sorted(
            beat_id for beat_id in promised_beat_ids if beat_id not in decision_by_id
        )
        if missing_decision_ids:
            raise RuntimeError(
                "Episode architecture omitted promised beat decisions for episode "
                f"{architecture.episode_number}: {_preview_ids(missing_decision_ids)}"
            )
        unknown_decision_ids = sorted(
            beat_id for beat_id in decision_by_id if beat_id not in set(promised_beat_ids)
        )
        if unknown_decision_ids:
            raise RuntimeError(
                "Episode architecture emitted promised beat decisions outside the strategy "
                f"episode for episode {architecture.episode_number}: {_preview_ids(unknown_decision_ids)}"
            )
        invalid_staged_section_ids = sorted(
            decision.section_id or ""
            for decision in architecture.promised_beat_decisions
            if decision.section_id and decision.section_id not in valid_section_ids
        )
        if invalid_staged_section_ids:
            raise RuntimeError(
                "Episode architecture staged promised beats into unknown sections for episode "
                f"{architecture.episode_number}: {_preview_ids(invalid_staged_section_ids)}"
            )
    return architecture


def _filter_primitive_ids_by_architecture(
    strategy_episode: StrategyEpisode,
    episode: EpisodeArchitecture,
) -> dict[str, list[str]]:
    section_primitive_ids = set(_ordered_architecture_primitive_ids(episode))
    section_excerpt_ids = set(_ordered_architecture_excerpt_ids(episode))
    return {
        "core": [
            primitive_id
            for primitive_id in strategy_episode.episode_spine.core_primitive_ids
            if primitive_id in section_primitive_ids
        ],
        "support": [
            primitive_id
            for primitive_id in strategy_episode.episode_spine.support_primitive_roles.keys()
            if primitive_id in section_primitive_ids
        ],
        "recall": [
            primitive_id
            for primitive_id in strategy_episode.episode_spine.recall_primitive_ids
            if primitive_id in section_primitive_ids
        ],
        "excerpt": [
            excerpt_id
            for excerpt_id in strategy_episode.episode_spine.assigned_excerpt_ids
            if excerpt_id in section_excerpt_ids
        ],
    }


def _build_architecture_narrative_state_coverage(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
) -> dict[str, list[str]]:
    listener_agenda = _listener_agenda(strategy_episode)
    host_agenda = _host_agenda(strategy_episode)
    question_ids = {
        move.question_id
        for section in architecture.sections
        for move in section.section_progression.state_effects.question_moves
    }
    memory_thread_ids = {
        move.thread_id
        for section in architecture.sections
        for move in section.section_progression.state_effects.memory_thread_moves
    }
    host_mystery_ids = {
        move.mystery_id
        for section in architecture.sections
        for move in section.section_progression.state_effects.host_mystery_moves
    }
    host_assumption_ids = {
        move.assumption_id
        for section in architecture.sections
        for move in section.section_progression.state_effects.host_assumption_moves
    }
    host_theory_ids = {
        move.theory_id
        for section in architecture.sections
        for move in section.section_progression.state_effects.host_theory_moves
    }
    return {
        "listener_question_move_ids_missing_from_architecture": [
            move.question_id
            for move in listener_agenda.question_moves
            if move.question_id not in question_ids
        ],
        "listener_memory_thread_move_ids_missing_from_architecture": [
            move.thread_id
            for move in listener_agenda.memory_thread_moves
            if move.thread_id not in memory_thread_ids
        ],
        "host_mystery_move_ids_missing_from_architecture": [
            move.mystery_id
            for move in host_agenda.mystery_moves
            if move.mystery_id not in host_mystery_ids
        ],
        "host_assumption_move_ids_missing_from_architecture": [
            move.assumption_id
            for move in host_agenda.assumption_moves
            if move.assumption_id not in host_assumption_ids
        ],
        "host_theory_move_ids_missing_from_architecture": [
            move.theory_id
            for move in host_agenda.theory_moves
            if move.theory_id not in host_theory_ids
        ],
    }


def _section_is_explanation_overloaded(section: Any) -> bool:
    return bool(
        (
            any(
                plan.stage in {"define", "payoff"}
                for plan in getattr(section, "term_explanations", [])
            )
            and getattr(section, "approx_runtime_minutes", 0.0) >= 8.0
        )
        or (
            len(getattr(section, "key_terms", [])) >= 4
            and len(getattr(section, "authorial_passages", [])) <= 1
            and len(getattr(section, "term_explanations", [])) >= 2
        )
    )


def _overloaded_architecture_section_ids(
    architecture: EpisodeArchitecture,
) -> list[str]:
    return [
        section.section_id
        for section in architecture.sections
        if _section_is_explanation_overloaded(section)
    ]


def _build_thread_architecture_warnings(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    answer_section_id: str | None,
) -> list[str]:
    """Non-blocking checks that the binding human thread is carried, grounded, and
    not silently dropped across the architecture's sections."""
    thread = strategy_episode.human_thread
    warnings: list[str] = []
    if thread is None:
        return warnings
    member_ids = {member.member_id for member in thread.members}
    sections = architecture.sections
    bound = [s for s in sections if s.thread_binding is not None]
    if not bound:
        warnings.append("thread_binding_missing_all_sections")
        return warnings
    if len(bound) != len(sections):
        missing = [s.section_id for s in sections if s.thread_binding is None]
        warnings.append(f"thread_dropped_in_section: {_preview_ids(missing)}")
    structural_only = 0
    carried_like = 0
    absent = 0
    pivot_section_ids = {answer_section_id, architecture.major_turn_section_id}
    for section in sections:
        binding = section.thread_binding
        if binding is None:
            continue
        if binding.carrying_member_id is not None and binding.carrying_member_id not in member_ids:
            warnings.append(f"thread_relay_orphan: {section.section_id}")
        presence = binding.presence
        fallback = binding.fallback_mode
        carried_or_relay = (
            presence == ThreadSectionPresence.CARRIED or fallback == ThreadFallbackMode.FAMILY_RELAY
        )
        if carried_or_relay:
            carried_like += 1
        if presence == ThreadSectionPresence.ABSENT:
            absent += 1
        if fallback == ThreadFallbackMode.STRUCTURAL_ONLY:
            structural_only += 1
        if section.section_id in pivot_section_ids and not carried_or_relay:
            warnings.append(f"thread_pivot_not_carried: {section.section_id}")
    if sections and structural_only / len(sections) > 0.4:
        warnings.append("thread_structural_only_streak")
    if sections and absent / len(sections) > 0.3:
        warnings.append("thread_too_many_absent")
    if sections and carried_like / len(sections) < 0.6:
        warnings.append("thread_continuity_share_low")
    return warnings


def _build_architecture_grounding_diagnostics(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
) -> dict[str, Any]:
    overloaded_set = set(_overloaded_architecture_section_ids(architecture))
    directive_actor_ids = sorted(
        {
            directive.actor_id
            for directive in strategy_episode.actor_arc_directives
            if directive.actor_id
        }
    )
    overloaded_runs: list[dict[str, Any]] = []
    current_run: list[Any] = []
    for section in architecture.sections:
        if section.section_id in overloaded_set:
            current_run.append(section)
            continue
        if len(current_run) >= 2:
            overloaded_runs.append(
                _build_overloaded_run_actor_arc_report(
                    current_run=current_run,
                    directive_actor_ids=directive_actor_ids,
                )
            )
        current_run = []
    if len(current_run) >= 2:
        overloaded_runs.append(
            _build_overloaded_run_actor_arc_report(
                current_run=current_run,
                directive_actor_ids=directive_actor_ids,
            )
        )
    warnings: list[str] = []
    for run in overloaded_runs:
        if not run["directive_actor_ids"]:
            warnings.append(
                f"overloaded_run_missing_actor_arc_directive: {_preview_ids(run['section_ids'])}"
            )
        elif not run["has_recurring_actor_arc_realization"]:
            warnings.append(
                f"overloaded_run_missing_actor_arc_realization: {_preview_ids(run['section_ids'])}"
            )
    return {
        "overloaded_runs": overloaded_runs,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _build_overloaded_run_actor_arc_report(
    *,
    current_run: list[Any],
    directive_actor_ids: list[str],
) -> dict[str, Any]:
    actor_counts: dict[str, int] = {}
    for run_section in current_run:
        for plan in run_section.actor_explanations:
            if plan.actor_id in directive_actor_ids:
                actor_counts[plan.actor_id] = actor_counts.get(plan.actor_id, 0) + 1
    realized_directive_actor_ids = sorted(actor_counts)
    recurring_directive_actor_ids = sorted(
        actor_id for actor_id, count in actor_counts.items() if count >= 2
    )
    return {
        "section_ids": [run_section.section_id for run_section in current_run],
        "directive_actor_ids": list(directive_actor_ids),
        "realized_directive_actor_ids": realized_directive_actor_ids,
        "has_recurring_actor_arc_realization": bool(recurring_directive_actor_ids),
    }


def _build_episode_architecture_realization(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    pipeline_config: PipelineConfig,
    narrator_profile: SeriesNarratorProfile | None = None,
    primitive_lookup: dict[str, SynthesisPrimitiveBase] | None = None,
    series_explanation_registry: list[SeriesExplanationItem] | None = None,
) -> dict[str, Any]:
    section_count = len(architecture.sections)
    section_primitive_ids = _ordered_architecture_primitive_ids(architecture)
    section_primitive_id_set = set(section_primitive_ids)
    selected_core_primitive_ids = list(strategy_episode.episode_spine.core_primitive_ids)
    selected_support_primitive_ids = list(
        strategy_episode.episode_spine.support_primitive_roles.keys()
    )
    selected_recall_primitive_ids = list(strategy_episode.episode_spine.recall_primitive_ids)
    missing_core_primitive_ids = [
        primitive_id
        for primitive_id in selected_core_primitive_ids
        if primitive_id not in section_primitive_id_set
    ]
    missing_support_primitive_ids = [
        primitive_id
        for primitive_id in selected_support_primitive_ids
        if primitive_id not in section_primitive_id_set
    ]
    missing_recall_primitive_ids = [
        primitive_id
        for primitive_id in selected_recall_primitive_ids
        if primitive_id not in section_primitive_id_set
    ]
    section_order = {section.section_id: idx for idx, section in enumerate(architecture.sections)}
    answer_section_id = next(
        (
            section.section_id
            for section in architecture.sections
            if section.section_progression.stage == ProgressionStage.ANSWER
        ),
        None,
    )
    promised_beat_ids = [beat.beat_id for beat in strategy_episode.promised_beats]
    promised_beat_decision_counts = {"stage": 0, "defer": 0, "drop": 0}
    promised_beat_unaccounted_ids: list[str] = []
    staged_promised_beats_without_section: list[str] = []
    deferred_promised_beats_without_reason: list[str] = []
    dropped_promised_beats_without_reason: list[str] = []
    negative_scope_violation_examples: list[str] = []
    warnings: list[str] = []
    narrative_state_coverage = _build_architecture_narrative_state_coverage(
        strategy_episode=strategy_episode,
        architecture=architecture,
    )
    warnings.extend(
        _build_architecture_section_count_warnings(
            section_count=section_count,
            section_target_min=pipeline_config.architecture_section_target_min,
            section_target_max=pipeline_config.architecture_section_target_max,
        )
    )
    if missing_support_primitive_ids:
        warnings.append(
            "architecture_missing_support_primitives: "
            f"{_preview_ids(missing_support_primitive_ids)}"
        )
    if missing_recall_primitive_ids:
        warnings.append(
            f"architecture_missing_recall_primitives: {_preview_ids(missing_recall_primitive_ids)}"
        )
    target_authorial_passages = (
        narrator_profile.target_authorial_passages_per_episode
        if narrator_profile is not None
        else None
    )
    authorial_passage_count = sum(
        len(section.authorial_passages) for section in architecture.sections
    )
    if (
        target_authorial_passages is not None
        and authorial_passage_count < target_authorial_passages
    ):
        warnings.append(
            "authored_passage_budget_below_target: "
            f"{authorial_passage_count}/{target_authorial_passages}"
        )

    registry_by_id = {item.item_id: item for item in (series_explanation_registry or [])}
    missing_payoff_item_ids: list[str] = []
    listener_agenda = _listener_agenda(strategy_episode)
    introduced_item_ids = (
        listener_agenda.introduce_explanation_item_ids
        or strategy_episode.authorial_contract.introduce_explanation_item_ids
    )
    reminded_item_ids = (
        listener_agenda.remind_explanation_item_ids
        or strategy_episode.authorial_contract.remind_explanation_item_ids
    )
    for item_id in introduced_item_ids:
        registry_item = registry_by_id.get(item_id)
        if registry_item is None or registry_item.importance != "foundational":
            continue
        stages = {
            plan.stage
            for section in architecture.sections
            for plan in section.term_explanations
            if plan.item_id == item_id
        }
        if "define" in stages and "payoff" not in stages:
            missing_payoff_item_ids.append(item_id)
    if missing_payoff_item_ids:
        warnings.append(
            f"foundational_item_missing_payoff: {_preview_ids(missing_payoff_item_ids)}"
        )

    reminder_redefined_item_ids = [
        item_id
        for item_id in reminded_item_ids
        if any(
            plan.item_id == item_id and plan.stage == "define"
            for section in architecture.sections
            for plan in section.term_explanations
        )
    ]
    if reminder_redefined_item_ids:
        warnings.append(f"reminder_item_redefined: {_preview_ids(reminder_redefined_item_ids)}")

    if primitive_lookup:
        sectioned_substrates = {
            primitive_lookup[primitive_id].substrate.value
            for primitive_id in section_primitive_id_set
            if primitive_id in primitive_lookup
        }
        sectioned_functions = {
            function.value
            for primitive_id in section_primitive_id_set
            if primitive_id in primitive_lookup
            for function in primitive_lookup[primitive_id].functions
        }
        if not (
            {"cost", "stake"} & sectioned_functions
            or {"actor_portraits", "events", "acts"} & sectioned_substrates
        ):
            warnings.append(
                "missing_human_grounding_support: "
                "episode sections omit cost/stake or actor/event/act grounding."
            )

    overloaded_section_ids = _overloaded_architecture_section_ids(architecture)
    if overloaded_section_ids:
        warnings.append(f"explanation_section_overloaded: {_preview_ids(overloaded_section_ids)}")
    grounding_diagnostics = _build_architecture_grounding_diagnostics(
        strategy_episode=strategy_episode,
        architecture=architecture,
    )
    warnings.extend(grounding_diagnostics["warnings"])
    warnings.extend(
        _build_thread_architecture_warnings(
            strategy_episode=strategy_episode,
            architecture=architecture,
            answer_section_id=answer_section_id,
        )
    )
    if not answer_section_id:
        warnings.append("answer_section_missing")
    if answer_section_id and answer_section_id == architecture.major_turn_section_id:
        warnings.append("answer_lands_on_turn")

    # Carry-over continuity: each non-opening section should pick up the prior
    # section's live pressure (section_progression.what_remains_live).
    carryover_dropped_section_ids: list[str] = []
    prior_live_tokens: set[str] = set()
    prior_live_present = False
    for section in architecture.sections:
        if prior_live_present and prior_live_tokens:
            section_tokens = _normalize_section_text_tokens(
                " ".join(
                    [
                        section.section_progression.becomes_obvious,
                        section.section_anchor,
                        *section.must_stage_beats,
                    ]
                )
            )
            if not (prior_live_tokens & section_tokens):
                carryover_dropped_section_ids.append(section.section_id)
        live = section.section_progression.what_remains_live
        prior_live_tokens = _normalize_section_text_tokens(live)
        prior_live_present = bool(live.strip())
    if carryover_dropped_section_ids:
        warnings.append(
            f"progression_carryover_dropped: {_preview_ids(carryover_dropped_section_ids)}"
        )

    decision_by_id = {
        decision.beat_id: decision for decision in architecture.promised_beat_decisions
    }
    for decision in architecture.promised_beat_decisions:
        promised_beat_decision_counts[decision.decision.value] = (
            promised_beat_decision_counts.get(decision.decision.value, 0) + 1
        )
        if decision.decision.value == "stage":
            if not decision.section_id or decision.section_id not in section_order:
                staged_promised_beats_without_section.append(decision.beat_id)
        elif decision.decision.value == "defer":
            if not decision.reason.strip():
                deferred_promised_beats_without_reason.append(decision.beat_id)
        elif decision.decision.value == "drop":
            if not decision.reason.strip():
                dropped_promised_beats_without_reason.append(decision.beat_id)
    promised_beat_unaccounted_ids = [
        beat_id for beat_id in promised_beat_ids if beat_id not in decision_by_id
    ]
    if promised_beat_unaccounted_ids:
        warnings.append(
            f"promised_beat_unaccounted_for: {_preview_ids(promised_beat_unaccounted_ids)}"
        )
    if staged_promised_beats_without_section:
        warnings.append(
            "promised_beat_staged_without_section: "
            f"{_preview_ids(staged_promised_beats_without_section)}"
        )
    if deferred_promised_beats_without_reason:
        warnings.append(
            "promised_beat_deferred_without_reason: "
            f"{_preview_ids(deferred_promised_beats_without_reason)}"
        )
    if dropped_promised_beats_without_reason:
        warnings.append(
            "promised_beat_dropped_without_reason: "
            f"{_preview_ids(dropped_promised_beats_without_reason)}"
        )

    negative_scope_terms = [
        term.strip()
        for term in [
            *strategy_episode.negative_scope.excluded_topics,
            *strategy_episode.negative_scope.tempting_but_out,
        ]
        if term and str(term).strip()
    ]
    if negative_scope_terms:
        for section in architecture.sections:
            section_text = " ".join(
                [
                    section.section_anchor,
                    *section.must_stage_beats,
                    *[passage.claim for passage in section.authorial_passages],
                ]
            ).lower()
            for term in negative_scope_terms:
                term_lower = term.lower()
                if term_lower and term_lower in section_text:
                    negative_scope_violation_examples.append(f"{section.section_id}:{term}")
    if negative_scope_violation_examples:
        warnings.append(
            f"negative_scope_violation: {_preview_ids(negative_scope_violation_examples)}"
        )

    def _section_tokens(section_id: str | None) -> set[str]:
        if not section_id:
            return set()
        section = next(
            (item for item in architecture.sections if item.section_id == section_id),
            None,
        )
        if section is None:
            return set()
        text = " ".join(
            [
                section.section_anchor,
                *section.must_stage_beats,
                *[passage.claim for passage in section.authorial_passages],
            ]
        )
        return _normalize_section_text_tokens(text)

    answer_tokens = _section_tokens(answer_section_id)
    closing_section = architecture.sections[-1] if architecture.sections else None
    closing_tokens = _section_tokens(closing_section.section_id if closing_section else None)
    if answer_tokens and closing_tokens:
        overlap = answer_tokens & closing_tokens
        overlap_ratio = len(overlap) / max(1, min(len(answer_tokens), len(closing_tokens)))
        if len(overlap) >= 4 and overlap_ratio >= 0.6:
            warnings.append("close_restates_answer")

    if closing_section is not None and primitive_lookup is not None:
        prior_primitive_ids = {
            primitive_id
            for section in architecture.sections[:-1]
            for primitive_id in section.primitive_ids
        }
        closing_only_system_primitive_ids = [
            primitive_id
            for primitive_id in closing_section.primitive_ids
            if primitive_id not in prior_primitive_ids
            and primitive_id in primitive_lookup
            and primitive_lookup[primitive_id].substrate.value
            in {"mechanisms", "conditions", "readings"}
        ]
        if closing_only_system_primitive_ids:
            warnings.append(
                "closing_section_new_mechanism_or_counterpressure: "
                f"{_preview_ids(closing_only_system_primitive_ids)}"
            )
    return {
        "episode_number": architecture.episode_number,
        "selected_core_primitive_ids": selected_core_primitive_ids,
        "selected_support_primitive_ids": selected_support_primitive_ids,
        "selected_recall_primitive_ids": selected_recall_primitive_ids,
        "section_primitive_ids": section_primitive_ids,
        "missing_core_primitive_ids": missing_core_primitive_ids,
        "missing_support_primitive_ids": missing_support_primitive_ids,
        "missing_recall_primitive_ids": missing_recall_primitive_ids,
        "answer_section_id": answer_section_id,
        "progression_carryover_dropped_section_ids": carryover_dropped_section_ids,
        "promised_beat_count": len(promised_beat_ids),
        "promised_beat_decision_counts": promised_beat_decision_counts,
        "promised_beat_unaccounted_ids": promised_beat_unaccounted_ids,
        "staged_promised_beats_without_section": staged_promised_beats_without_section,
        "deferred_promised_beats_without_reason": deferred_promised_beats_without_reason,
        "dropped_promised_beats_without_reason": dropped_promised_beats_without_reason,
        "negative_scope_violation_examples": negative_scope_violation_examples,
        "narrative_state_coverage": narrative_state_coverage,
        "target_authorial_passages_per_episode": target_authorial_passages,
        "authorial_passage_count": authorial_passage_count,
        "grounding_diagnostics": grounding_diagnostics,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _validate_plan_transition(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    plan: EpisodePlanDraft | EpisodePlan,
) -> EpisodePlanDraft | EpisodePlan:
    if architecture.episode_number != strategy_episode.episode_number:
        raise ComplianceViolationError(
            "Episode architecture episode_number does not match strategy episode "
            f"({architecture.episode_number} != {strategy_episode.episode_number}).",
            data={
                "issue": "architecture_episode_number_mismatch",
                "episode_number": strategy_episode.episode_number,
                "expected_episode_number": strategy_episode.episode_number,
                "actual_episode_number": architecture.episode_number,
                "instruction": "Use the same episode_number in the architecture as in the strategy episode.",
            },
        )
    if plan.episode_number != architecture.episode_number:
        raise ComplianceViolationError(
            "Episode plan episode_number does not match architecture episode "
            f"({plan.episode_number} != {architecture.episode_number}).",
            data={
                "issue": "plan_episode_number_mismatch",
                "episode_number": architecture.episode_number,
                "expected_episode_number": architecture.episode_number,
                "actual_episode_number": plan.episode_number,
                "instruction": "Return an episode plan whose episode_number matches the architecture episode_number exactly.",
            },
        )

    section_order = {section.section_id: idx for idx, section in enumerate(architecture.sections)}
    scene_counts_by_section = {section.section_id: 0 for section in architecture.sections}
    highest_section_index = -1
    invalid_section_ids: list[str] = []
    for scene in plan.scene_cards:
        section_index = section_order.get(scene.section_id)
        if section_index is None:
            invalid_section_ids.append(scene.section_id)
            continue
        if section_index < highest_section_index:
            raise ComplianceViolationError(
                "Episode plan scene order revisited an earlier section for episode "
                f"{plan.episode_number}: scene {scene.scene_id} returned to {scene.section_id}.",
                data={
                    "issue": "scene_order_revisited_section",
                    "episode_number": plan.episode_number,
                    "scene_id": scene.scene_id,
                    "section_id": scene.section_id,
                    "instruction": "Keep scene order monotonic by architecture section. Once the plan advances to a later section, do not return to an earlier one.",
                },
            )
        highest_section_index = section_index
        scene_counts_by_section[scene.section_id] += 1

    if invalid_section_ids:
        raise ComplianceViolationError(
            "Episode plan referenced unknown architecture sections for episode "
            f"{plan.episode_number}: {_preview_ids(sorted(set(invalid_section_ids)))}",
            data={
                "issue": "unknown_architecture_section",
                "episode_number": plan.episode_number,
                "invalid_section_ids": sorted(set(invalid_section_ids)),
                "instruction": "Use only section_id values that exist in the provided architecture.",
            },
        )
    missing_section_ids = sorted(
        section_id for section_id, count in scene_counts_by_section.items() if count == 0
    )
    if missing_section_ids:
        raise ComplianceViolationError(
            "Episode plan did not cover every architecture section for episode "
            f"{plan.episode_number}: {_preview_ids(missing_section_ids)}",
            data={
                "issue": "missing_architecture_section_coverage",
                "episode_number": plan.episode_number,
                "missing_section_ids": missing_section_ids,
                "instruction": "Return at least one scene for every architecture section.",
            },
        )

    scenes_by_section_id: dict[str, list[SceneCardDraft | SceneCard]] = {}
    for scene in plan.scene_cards:
        scenes_by_section_id.setdefault(scene.section_id, []).append(scene)
    for section in architecture.sections:
        section_authorial_passages = list(section.authorial_passages)
        if not section_authorial_passages:
            unexpected_scene_ids = sorted(
                scene.scene_id
                for scene in scenes_by_section_id.get(section.section_id, [])
                if scene.authorial_passage_ids
            )
            if unexpected_scene_ids:
                raise ComplianceViolationError(
                    "Episode plan assigned authorial passages in a section with no authorial_passages.",
                    data={
                        "issue": "authorial_passage_assignment_without_section_authorial",
                        "episode_number": plan.episode_number,
                        "section_id": section.section_id,
                        "scene_ids": unexpected_scene_ids,
                        "instruction": "Assign authorial_passage_ids only in sections that define authorial_passages.",
                    },
                )
            continue
        section_authorial_by_id = {
            passage.authorial_passage_id: passage for passage in section_authorial_passages
        }
        assignment_counts: dict[str, int] = {
            passage.authorial_passage_id: 0 for passage in section_authorial_passages
        }
        for scene in scenes_by_section_id.get(section.section_id, []):
            for authorial_passage_id in scene.authorial_passage_ids:
                if authorial_passage_id not in section_authorial_by_id:
                    raise ComplianceViolationError(
                        "Episode plan assigned an unknown authorial_passage_id.",
                        data={
                            "issue": "unknown_authorial_passage_id",
                            "episode_number": plan.episode_number,
                            "section_id": section.section_id,
                            "scene_id": scene.scene_id,
                            "authorial_passage_id": authorial_passage_id,
                            "instruction": "Use only authorial_passage_id values defined on the matching architecture section.",
                        },
                    )
                assignment_counts[authorial_passage_id] += 1
        missing_authorial_passage_ids = sorted(
            authorial_passage_id
            for authorial_passage_id, count in assignment_counts.items()
            if count == 0
        )
        duplicated_authorial_passage_ids = sorted(
            authorial_passage_id
            for authorial_passage_id, count in assignment_counts.items()
            if count > 1
        )
        if missing_authorial_passage_ids:
            raise ComplianceViolationError(
                "Episode plan left section authorial passages unassigned.",
                data={
                    "issue": "authorial_passage_unassigned",
                    "episode_number": plan.episode_number,
                    "section_id": section.section_id,
                    "authorial_passage_ids": missing_authorial_passage_ids,
                    "instruction": "Assign every section authorial_passage_id to exactly one scene in the same section.",
                },
            )
        if duplicated_authorial_passage_ids:
            raise ComplianceViolationError(
                "Episode plan assigned the same authorial_passage_id to multiple scenes.",
                data={
                    "issue": "authorial_passage_assigned_multiple_times",
                    "episode_number": plan.episode_number,
                    "section_id": section.section_id,
                    "authorial_passage_ids": duplicated_authorial_passage_ids,
                    "instruction": "Assign each authorial_passage_id to exactly one scene in its section.",
                },
            )

    scene_by_id = {scene.scene_id: scene for scene in plan.scene_cards}
    if not plan.answer_scene_card_id:
        raise ComplianceViolationError(
            f"Episode plan must include answer_scene_card_id for episode {plan.episode_number}.",
            data={
                "issue": "answer_scene_missing",
                "episode_number": plan.episode_number,
                "instruction": "Return exactly one answer scene and set answer_scene_card_id to that scene_id.",
            },
        )

    answer_scene = scene_by_id.get(plan.answer_scene_card_id)
    if answer_scene is None or answer_scene.scene_job != SceneJob.ANSWER:
        raise ComplianceViolationError(
            "Episode plan answer_scene_card_id must reference a scene_job='answer' card.",
            data={
                "issue": "answer_scene_invalid",
                "episode_number": plan.episode_number,
                "answer_scene_card_id": plan.answer_scene_card_id,
                "instruction": "Set answer_scene_card_id to the scene_id of the single answer card.",
            },
        )
    architecture_answer_section_id = next(
        (
            section.section_id
            for section in architecture.sections
            if section.section_progression.stage == ProgressionStage.ANSWER
        ),
        None,
    )
    if architecture_answer_section_id and answer_scene.section_id != architecture_answer_section_id:
        raise ComplianceViolationError(
            "Answer scene does not belong to the answer-stage section.",
            data={
                "issue": "answer_scene_wrong_section",
                "episode_number": plan.episode_number,
                "answer_scene_card_id": plan.answer_scene_card_id,
                "expected_section_id": architecture_answer_section_id,
                "actual_section_id": answer_scene.section_id,
                "instruction": "Place the answer scene inside the section whose progression stage is 'answer'.",
            },
        )

    answer_scene_ids = [
        scene.scene_id for scene in plan.scene_cards if scene.scene_job == SceneJob.ANSWER
    ]
    close_scene_ids = [
        scene.scene_id for scene in plan.scene_cards if scene.scene_job == SceneJob.CLOSE
    ]
    if len(answer_scene_ids) != 1:
        raise ComplianceViolationError(
            "Episode plan must contain exactly one answer scene.",
            data={
                "issue": "answer_scene_count_invalid",
                "episode_number": plan.episode_number,
                "scene_ids": answer_scene_ids,
                "instruction": "Return exactly one scene card with scene_job='answer'.",
            },
        )
    if len(close_scene_ids) != 1:
        raise ComplianceViolationError(
            "Episode plan must contain exactly one close scene.",
            data={
                "issue": "close_scene_count_invalid",
                "episode_number": plan.episode_number,
                "scene_ids": close_scene_ids,
                "instruction": "Return exactly one scene card with scene_job='close'.",
            },
        )

    ordered_scene_ids = [scene.scene_id for scene in plan.scene_cards]
    close_scene_id = close_scene_ids[0]
    if ordered_scene_ids.index(close_scene_id) <= ordered_scene_ids.index(answer_scene.scene_id):
        raise ComplianceViolationError(
            "Episode plan must place the close scene after the answer scene.",
            data={
                "issue": "close_precedes_answer",
                "episode_number": plan.episode_number,
                "close_scene_card_id": close_scene_id,
                "answer_scene_card_id": answer_scene.scene_id,
                "instruction": "Place the close scene after the answer scene.",
            },
        )
    if ordered_scene_ids[-1] != close_scene_id:
        raise ComplianceViolationError(
            "Episode plan must end with the close scene.",
            data={
                "issue": "close_not_last",
                "episode_number": plan.episode_number,
                "close_scene_card_id": close_scene_id,
                "instruction": "Make the close scene the final scene card in the episode.",
            },
        )

    return plan


def _validate_actor_explanation_scene_links(
    *,
    architecture: EpisodeArchitecture,
    plan: EpisodePlanDraft | EpisodePlan,
) -> list[str]:
    scene_cards_by_section: dict[str, list[SceneCardDraft | SceneCard]] = {}
    earliest_scene_by_actor: dict[str, SceneCardDraft | SceneCard] = {}
    for scene in plan.scene_cards:
        scene_cards_by_section.setdefault(scene.section_id, []).append(scene)
        for actor in scene.actors:
            if actor.actor_id and actor.actor_id not in earliest_scene_by_actor:
                earliest_scene_by_actor[actor.actor_id] = scene

    missing_links: list[str] = []
    misplaced_introductions: list[str] = []
    for section in architecture.sections:
        if not section.actor_explanations:
            continue
        section_scenes = scene_cards_by_section.get(section.section_id, [])
        for explanation in section.actor_explanations:
            matched = any(
                actor.actor_id == explanation.actor_id
                and actor.explanation_stage == explanation.stage
                for scene in section_scenes
                for actor in scene.actors
            )
            if not matched:
                missing_links.append(
                    f"{section.section_id}:{explanation.actor_id}:{explanation.stage}"
                )
                continue
            if explanation.stage != "introduce":
                continue
            earliest_scene = earliest_scene_by_actor.get(explanation.actor_id)
            if earliest_scene is None:
                continue
            tagged_earliest_scene = any(
                actor.actor_id == explanation.actor_id and actor.explanation_stage == "introduce"
                for actor in earliest_scene.actors
            )
            if not tagged_earliest_scene:
                misplaced_introductions.append(f"{explanation.actor_id}:{earliest_scene.scene_id}")
    warnings: list[str] = []
    if missing_links:
        warnings.append("missing_actor_explanation_scene_links:" + ",".join(missing_links))
    if misplaced_introductions:
        warnings.append("late_actor_introduction_scene_links:" + ",".join(misplaced_introductions))
    return warnings


def _flatten_synthesis_primitives(
    synthesis_map: SynthesisMap,
) -> dict[str, SynthesisPrimitiveBase]:
    return {item.id: item for item in synthesis_map.primitives}


def _primitive_substrate_lookup(synthesis_map: SynthesisMap) -> dict[str, str]:
    return {item.id: item.substrate.value for item in synthesis_map.primitives}


def _build_episode_synthesis_map_payload(
    synthesis_map: SynthesisMap,
    primitive_ids_by_role: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    primitive_ids: list[str] = []
    seen_primitive_ids: set[str] = set()

    for role in ("core", "support", "recall"):
        for primitive_id in primitive_ids_by_role.get(role, []):
            if primitive_id in seen_primitive_ids:
                continue
            seen_primitive_ids.add(primitive_id)
            primitive_ids.append(primitive_id)

    primitive_id_set = set(primitive_ids)
    primitives = [
        item.model_dump(mode="json")
        for item in synthesis_map.primitives
        if item.id in primitive_id_set
    ]
    payload = {
        "project_id": synthesis_map.project_id,
        "primitive_ids_by_role": {
            role: [
                primitive_id
                for primitive_id in primitive_ids_by_role.get(role, [])
                if primitive_id in primitive_id_set
            ]
            for role in ("core", "support", "recall")
        },
        "primitives": primitives,
        "quality_score": synthesis_map.quality_score,
        "quality_notes": list(synthesis_map.quality_notes),
    }
    return payload, primitive_ids


def _build_episode_architecture_core_passages(
    *,
    driving_question: str,
    thematic_focus: str,
    episode_spine: EpisodeSpine,
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
    passage_lookup: dict[str, ExtractedPassage],
) -> list[dict[str, Any]]:
    core_passages, _support_passages = _build_episode_architecture_passages(
        driving_question=driving_question,
        thematic_focus=thematic_focus,
        episode_spine=episode_spine,
        primitive_lookup=primitive_lookup,
        passage_lookup=passage_lookup,
    )
    return core_passages


def _build_episode_architecture_support_passages(
    *,
    driving_question: str,
    thematic_focus: str,
    episode_spine: EpisodeSpine,
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
    passage_lookup: dict[str, ExtractedPassage],
) -> list[dict[str, Any]]:
    _core_passages, support_passages = _build_episode_architecture_passages(
        driving_question=driving_question,
        thematic_focus=thematic_focus,
        episode_spine=episode_spine,
        primitive_lookup=primitive_lookup,
        passage_lookup=passage_lookup,
    )
    return support_passages


def _build_episode_architecture_passages(
    *,
    driving_question: str,
    thematic_focus: str,
    episode_spine: EpisodeSpine,
    primitive_lookup: dict[str, SynthesisPrimitiveBase],
    passage_lookup: dict[str, ExtractedPassage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_text = " ".join(
        part.strip() for part in (driving_question, thematic_focus) if part and part.strip()
    )
    core_refs, support_refs = _build_episode_architecture_passage_refs(
        episode_spine=episode_spine,
        primitive_lookup=primitive_lookup,
    )
    query_text_by_passage_id = _build_episode_planning_passage_query_texts(
        episode_query_text=query_text,
        passage_refs=[*core_refs, *support_refs],
        primitive_lookup=primitive_lookup,
    )
    seen_passage_ids: set[str] = set()
    core_passages = _build_episode_architecture_passage_payload(
        passage_refs=core_refs,
        seen_passage_ids=seen_passage_ids,
        passage_lookup=passage_lookup,
        query_text=query_text,
        query_text_by_passage_id=query_text_by_passage_id,
        keep_fraction=0.20,
    )
    support_passages = _build_episode_architecture_passage_payload(
        passage_refs=support_refs,
        seen_passage_ids=seen_passage_ids,
        passage_lookup=passage_lookup,
        query_text=query_text,
        query_text_by_passage_id=query_text_by_passage_id,
        keep_fraction=0.10,
    )
    return core_passages, support_passages


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
    primitives = list(synthesis_map.primitives)
    scenes = [scene for plan in episode_plans for scene in plan.scene_cards]
    return {
        "primitive_count": len(primitives),
        "actor_linked_primitive_count": sum(1 for primitive in primitives if primitive.actor_ids),
        "episode_count": len(strategy.episodes),
        "actor_linked_episode_count": sum(
            1 for episode in strategy.episodes if episode.actor_arc_directives
        ),
        "scene_count": len(scenes),
        "actor_linked_scene_count": sum(
            1 for scene in scenes if any(actor.actor_id for actor in scene.actors)
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


def _scene_overlap_token_ratio(
    left_tokens: set[str],
    right_tokens: set[str],
) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _scene_text_tokens(scene: SceneCardDraft | SceneCard) -> set[str]:
    return _normalize_section_text_tokens(
        " ".join(
            [
                scene.beat_change,
                *scene.must_land_facts.ordered_facts(),
                scene.entry_image,
            ]
        )
    )


def _build_scene_job_budget_diagnostics(
    *,
    scene_cards: list[SceneCardDraft | SceneCard],
    scene_job_budget: dict[str, Any],
    answer_scene_card_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    scene_job_counts = _build_scene_job_counts(scene_cards)
    scene_ids_by_job: dict[str, list[str]] = {
        job.value: [scene.scene_id for scene in scene_cards if scene.scene_job == job]
        for job in SceneJob
    }
    warnings: list[str] = []
    total_count = len(scene_cards)
    total_min = int(scene_job_budget.get("total_min", total_count))
    total_max = int(scene_job_budget.get("total_max", total_count))
    if total_count < total_min:
        warnings.append(
            f"scene_budget_total_out_of_range: {total_count} < {total_min} (target range {total_min}-{total_max})"
        )
    elif total_count > total_max:
        warnings.append(
            f"scene_budget_total_out_of_range: {total_count} > {total_max} (target range {total_min}-{total_max})"
        )

    for job_name in ("opening", "build", "turn"):
        count = scene_job_counts.get(job_name, 0)
        lower = int(scene_job_budget.get(f"{job_name}_min", count))
        upper = int(scene_job_budget.get(f"{job_name}_max", count))
        if count < lower or count > upper:
            warnings.append(
                f"scene_budget_{job_name}_out_of_range: {count} (target range {lower}-{upper})"
            )

    for job_name in ("answer", "close"):
        count = scene_job_counts.get(job_name, 0)
        if count == 0:
            warnings.append(f"scene_budget_{job_name}_missing")
        elif count > 1:
            warnings.append(f"scene_budget_{job_name}_duplicate: {count}")

    ordered_scene_ids = [scene.scene_id for scene in scene_cards]
    answer_scene_present = bool(answer_scene_card_id) and answer_scene_card_id in ordered_scene_ids
    close_scene_ids = scene_ids_by_job[SceneJob.CLOSE.value]
    close_scene_card_id = close_scene_ids[0] if close_scene_ids else None
    close_scene_present = close_scene_card_id is not None
    close_follows_answer = True
    if answer_scene_present and close_scene_present:
        close_follows_answer = ordered_scene_ids.index(
            close_scene_card_id
        ) > ordered_scene_ids.index(answer_scene_card_id)  # type: ignore[arg-type]
        if not close_follows_answer:
            warnings.append("scene_budget_close_precedes_answer")
    elif close_scene_present and ordered_scene_ids and ordered_scene_ids[-1] != close_scene_card_id:
        close_follows_answer = False
        warnings.append("scene_budget_close_not_last")

    close_duplicates_answer = False
    if answer_scene_present and close_scene_present:
        answer_scene = next(
            scene for scene in scene_cards if scene.scene_id == answer_scene_card_id
        )
        close_scene = next(scene for scene in scene_cards if scene.scene_id == close_scene_card_id)
        overlap_ratio = _scene_overlap_token_ratio(
            _scene_text_tokens(answer_scene),
            _scene_text_tokens(close_scene),
        )
        close_duplicates_answer = overlap_ratio >= 0.6

    recap_build_scene_count = sum(
        1
        for scene in scene_cards
        if scene.scene_job == SceneJob.BUILD
        and re.search(r"\b(recap|reorient|reset|previously)\b", scene.title.lower())
    )
    max_recap_build_scenes = int(scene_job_budget.get("max_recap_build_scenes", 1))
    if recap_build_scene_count > max_recap_build_scenes:
        warnings.append(
            f"scene_budget_recap_build_out_of_range: {recap_build_scene_count} > {max_recap_build_scenes}"
        )

    diagnostics = {
        "scene_job_budget": scene_job_budget,
        "scene_job_counts": scene_job_counts,
        "scene_ids_by_job": scene_ids_by_job,
        "answer_scene_card_id": answer_scene_card_id,
        "close_scene_card_id": close_scene_card_id,
        "answer_scene_present": answer_scene_present,
        "close_scene_present": close_scene_present,
        "close_follows_answer": close_follows_answer,
        "close_duplicates_answer": close_duplicates_answer,
        "parallel_answer_detected": scene_job_counts.get(SceneJob.ANSWER.value, 0) > 1,
        "recap_build_scene_count": recap_build_scene_count,
        "scene_job_budget_warnings": warnings,
    }
    return diagnostics, warnings


def _build_comparative_aside_scene_warnings(
    *,
    architecture: EpisodeArchitecture,
    scene_cards: list[SceneCardDraft | SceneCard],
) -> list[str]:
    warnings: list[str] = []
    scenes_by_section_id: dict[str, list[SceneCardDraft | SceneCard]] = {}
    for scene in scene_cards:
        scenes_by_section_id.setdefault(scene.section_id, []).append(scene)

    for section in architecture.sections:
        section_authorial_by_id = {
            passage.authorial_passage_id: passage for passage in section.authorial_passages
        }
        for scene in scenes_by_section_id.get(section.section_id, []):
            assigned_passages = [
                section_authorial_by_id[authorial_passage_id]
                for authorial_passage_id in scene.authorial_passage_ids
                if authorial_passage_id in section_authorial_by_id
            ]
            comparative_asides = [
                passage for passage in assigned_passages if passage.mode == "comparative_aside"
            ]
            if not comparative_asides:
                continue

            move_types = {cue.move_type for _phase, cue in _iter_host_move_cues(scene)}
            has_return_move = bool({"callback", "evaluate"} & move_types)
            heavy_non_aside_passages = [
                passage
                for passage in assigned_passages
                if passage.mode in _HEAVY_AUTHORIAL_PASSAGE_MODES
                and passage.mode != "comparative_aside"
            ]
            projected_words = _project_scene_word_count(scene, words_per_minute=145.0)

            for passage in comparative_asides:
                if passage.placement == "close" and passage.budget_sentences <= 3:
                    warnings.append(
                        "comparative_aside_close_underprovisioned: "
                        f"{scene.scene_id}/{passage.authorial_passage_id}"
                    )
                if not has_return_move:
                    warnings.append(
                        "comparative_aside_missing_return_move: "
                        f"{scene.scene_id}/{passage.authorial_passage_id}"
                    )
                if scene.word_count_priority == WordCountPriority.TIGHT:
                    warnings.append(
                        "comparative_aside_tight_priority: "
                        f"{scene.scene_id}/{passage.authorial_passage_id}"
                    )
                if projected_words < (
                    int(passage.budget_sentences) * _COMPARATIVE_ASIDE_MIN_WORDS_PER_SENTENCE
                ):
                    warnings.append(
                        "comparative_aside_scene_too_short: "
                        f"{scene.scene_id}/{passage.authorial_passage_id}"
                    )
            if scene.scene_job == SceneJob.ANSWER and heavy_non_aside_passages:
                warnings.append(
                    "comparative_aside_answer_scene_stacked: "
                    f"{scene.scene_id}/"
                    + ",".join(
                        sorted(passage.authorial_passage_id for passage in heavy_non_aside_passages)
                    )
                )
    return warnings


def _build_architecture_section_count_warnings(
    *,
    section_count: int,
    section_target_min: int,
    section_target_max: int,
) -> list[str]:
    warnings: list[str] = []
    if section_count < section_target_min:
        warnings.append(
            "architecture_section_count_below_target: "
            f"{section_count} < {section_target_min} (target range {section_target_min}-{section_target_max})"
        )
    elif section_count > section_target_max:
        warnings.append(
            "architecture_section_count_above_target: "
            f"{section_count} > {section_target_max} (target range {section_target_min}-{section_target_max})"
        )
    return warnings


def _is_structural_scene_card(scene: SceneCardDraft | SceneCard) -> bool:
    return scene.scene_job in _STRUCTURAL_SCENE_JOBS


def _is_human_grounding_scene_card(scene: SceneCardDraft | SceneCard) -> bool:
    return (
        scene.scene_job in _GROUNDING_SCENE_JOBS
        and scene.scene_role in _GROUNDING_SCENE_ROLES
        and bool(scene.actors)
        and bool((scene.entry_image or "").strip())
        and bool((scene.observable_detail or "").strip())
    )


def _has_audible_anchor(scene: SceneCardDraft | SceneCard) -> bool:
    return bool((scene.audible_detail or "").strip())


_SECTION_SONIC_LEXICON = frozenset(
    {
        "alarm",
        "bell",
        "boots",
        "broadcast",
        "chant",
        "cheer",
        "clatter",
        "crack",
        "crowd",
        "drum",
        "echo",
        "engine",
        "engines",
        "gunfire",
        "gulls",
        "hooves",
        "hum",
        "idle",
        "loudspeaker",
        "march",
        "murmur",
        "phone",
        "prayer",
        "quiet",
        "radio",
        "rattle",
        "rifle",
        "rifles",
        "scream",
        "shout",
        "shutters",
        "silence",
        "singing",
        "sirens",
        "stamp",
        "static",
        "tearing",
        "thud",
        "volley",
        "whisper",
    }
)


def _normalize_sonic_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _normalize_sonic_tokens(text: str) -> list[str]:
    return [token for token in _normalize_sonic_text(text).split() if len(token) >= 3]


def _text_has_sound_lexicon(text: str) -> bool:
    return bool(set(_normalize_sonic_tokens(text)) & _SECTION_SONIC_LEXICON)


def _sonic_phrase_realized(target_text: str, observed_texts: list[str]) -> bool:
    normalized_target = _normalize_sonic_text(target_text)
    if not normalized_target:
        return True
    normalized_observed = [_normalize_sonic_text(text) for text in observed_texts]
    if any(
        normalized_target in observed_text for observed_text in normalized_observed if observed_text
    ):
        return True
    target_tokens = set(_normalize_sonic_tokens(target_text))
    if not target_tokens:
        return True
    observed_tokens: set[str] = set()
    for text in observed_texts:
        observed_tokens.update(_normalize_sonic_tokens(text))
    overlap = len(target_tokens & observed_tokens)
    required_overlap = 1 if len(target_tokens) <= 2 else 2
    return overlap >= required_overlap


def _scene_audible_detail_is_section_copy(
    opening_anchor: str,
    audible_detail: str,
) -> bool:
    normalized_anchor = _normalize_sonic_text(opening_anchor)
    normalized_audible = _normalize_sonic_text(audible_detail)
    if not normalized_anchor or not normalized_audible:
        return False
    if normalized_anchor == normalized_audible:
        return True
    anchor_tokens = _normalize_sonic_tokens(opening_anchor)
    audible_tokens = _normalize_sonic_tokens(audible_detail)
    if not anchor_tokens or not audible_tokens:
        return False
    overlap_ratio = len(set(anchor_tokens) & set(audible_tokens)) / max(
        len(set(anchor_tokens)),
        len(set(audible_tokens)),
    )
    return overlap_ratio >= 0.85 and abs(len(anchor_tokens) - len(audible_tokens)) <= 2


def _find_later_beat_binding_scene_id(
    beat: SectionSonicBeat,
    scene_cards: list[SceneCardDraft | SceneCard],
) -> str | None:
    for scene in scene_cards[1:]:
        observed_texts = [
            scene.audible_detail,
            scene.title,
            scene.beat_change,
            scene.entry_image,
            scene.observable_detail,
        ]
        if _sonic_phrase_realized(beat.cue, observed_texts):
            return scene.scene_id
    return None


def _build_scene_job_counts(
    scene_cards: list[SceneCardDraft | SceneCard],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scene in scene_cards:
        counts[scene.scene_job] = counts.get(scene.scene_job, 0) + 1
    return counts


def _build_structural_card_concreteness_warnings(
    *,
    scene_cards: list[SceneCardDraft | SceneCard],
) -> list[str]:
    missing_entry_image = [
        scene.scene_id
        for scene in scene_cards
        if _is_structural_scene_card(scene) and not (scene.entry_image or "").strip()
    ]
    missing_observable_detail = [
        scene.scene_id
        for scene in scene_cards
        if _is_structural_scene_card(scene) and not (scene.observable_detail or "").strip()
    ]
    warnings: list[str] = []
    if missing_entry_image:
        warnings.append(f"structural_card_missing_entry_image: {_preview_ids(missing_entry_image)}")
    if missing_observable_detail:
        warnings.append(
            f"structural_card_missing_observable_detail: {_preview_ids(missing_observable_detail)}"
        )
    return warnings


def _build_human_grounding_warnings(
    *,
    scene_cards: list[SceneCardDraft | SceneCard],
) -> tuple[dict[str, Any], list[str]]:
    structural_scene_ids = [
        scene.scene_id for scene in scene_cards if _is_structural_scene_card(scene)
    ]
    grounding_scene_ids = [
        scene.scene_id for scene in scene_cards if _is_human_grounding_scene_card(scene)
    ]
    structural_scene_count = len(structural_scene_ids)
    scene_card_count = len(scene_cards)
    structural_scene_ratio = structural_scene_count / max(1, scene_card_count)
    structurally_heavy = structural_scene_count >= 6 or structural_scene_ratio >= 0.30
    warnings: list[str] = []
    if structurally_heavy and not grounding_scene_ids:
        warnings.append(
            "structurally_heavy_episode_missing_human_grounding: "
            f"{structural_scene_count}/{scene_card_count} structural cards and no grounding card"
        )
    diagnostics = {
        "scene_card_count": scene_card_count,
        "structural_scene_count": structural_scene_count,
        "structural_scene_ratio": structural_scene_ratio,
        "structurally_heavy": structurally_heavy,
        "grounding_scene_count": len(grounding_scene_ids),
        "grounding_scene_ids": grounding_scene_ids,
    }
    return diagnostics, warnings


_SPINE_PRIMITIVE_SUBSTRATES = frozenset({"events", "acts"})
_SCENE_DETAIL_PRIMITIVE_SUBSTRATES = frozenset({"events", "artifacts"})
_SYSTEM_CONTEXT_PRIMITIVE_SUBSTRATES = frozenset({"mechanisms", "conditions", "readings"})


def _build_scene_card_family_warnings(
    *,
    strategy_episode: StrategyEpisode,
    primitive_pool_ids: set[str],
    primitive_by_id: dict[str, SynthesisPrimitiveBase],
) -> list[str]:
    primitive_pool = [
        primitive_by_id[primitive_id]
        for primitive_id in primitive_pool_ids
        if primitive_id in primitive_by_id
    ]
    substrates_present = {primitive.substrate.value for primitive in primitive_pool}
    functions_present = {
        function.value for primitive in primitive_pool for function in primitive.functions
    }
    warnings: list[str] = []
    if not primitive_pool:
        return warnings

    if substrates_present.isdisjoint(_SPINE_PRIMITIVE_SUBSTRATES):
        warnings.append(
            "primitive_pool_missing_spine: episode primitive pool lacks event/act grounding"
        )
    if "texture" not in functions_present and substrates_present.isdisjoint(
        _SCENE_DETAIL_PRIMITIVE_SUBSTRATES
    ):
        warnings.append(
            "primitive_pool_missing_scene_or_detail: "
            "episode primitive pool lacks texture-capable concrete primitives"
        )
    if not ({"cost", "stake"} & functions_present) and "actor_portraits" not in substrates_present:
        warnings.append(
            "primitive_pool_missing_human_grounding: "
            "episode primitive pool lacks cost/stake or actor-centered grounding"
        )
    if substrates_present.isdisjoint(_SYSTEM_CONTEXT_PRIMITIVE_SUBSTRATES):
        warnings.append(
            "primitive_pool_missing_system_or_context: "
            "episode primitive pool lacks mechanism/condition/reading support"
        )

    support_primitive_count = len(strategy_episode.episode_spine.support_primitive_roles) + len(
        strategy_episode.episode_spine.recall_primitive_ids
    )
    if support_primitive_count >= 1 and "recurrence" not in functions_present:
        warnings.append(
            "primitive_pool_missing_recurrence: "
            "support-heavy episode primitive pool lacks recurrence-tagged primitives"
        )
    return warnings


def _build_scene_card_primitive_warnings(
    *,
    scene_cards: list[SceneCard],
    primitive_pool_ids: set[str],
    primitive_by_id: dict[str, SynthesisPrimitiveBase],
    primitive_min: int,
    primitive_max: int,
) -> list[str]:
    warnings: list[str] = []
    overloaded_cards = [
        scene.scene_id
        for scene in scene_cards
        if scene.must_land_facts.total_count() > max(5, primitive_max + 3)
    ]
    if overloaded_cards:
        warnings.append(
            "scene_fact_density_out_of_range: "
            f"{len(overloaded_cards)} scenes exceed lean fact density "
            f"({_preview_ids(overloaded_cards)})"
        )
    host_move_scene_ids = [
        scene.scene_id for scene in scene_cards if _scene_host_phase_bucket_count(scene)
    ]
    if not host_move_scene_ids:
        warnings.append("host_voice_absent: no scene cards carry planned host phase shaping")
    return warnings


def _normalize_section_text_tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return {token for token in normalized.split() if len(token) >= 4}


def _continuity_item_sort_key(item: ContinuityCarryItem) -> tuple[int, int, str]:
    return (
        0 if item.priority == "high" else 1,
        0 if item.desired_surface == "recap" else 1,
        item.item_id,
    )


def _continuity_items_to_payload(
    items: list[ContinuityCarryItem],
) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def _build_continuity_contract(
    *,
    narrative_state: NarrativeState | None,
    episode_number: int,
    phase: str,
) -> dict[str, Any]:
    if narrative_state is None:
        return {
            "phase": phase,
            "episode_number": episode_number,
            "last_episode_takeaway": None,
            "recap_items": [],
            "must_surface_early": [],
            "must_leave_live": [],
            "open_question_texts": [],
            "open_memory_thread_labels": [],
            "host_open_pressures": [],
        }

    carry_items = list(narrative_state.listener.carry_forward_memory)
    if phase == "pre":
        recap_items = sorted(carry_items, key=_continuity_item_sort_key)[:2]
        takeaway = narrative_state.listener.last_episode_takeaway
        if not recap_items and takeaway is not None:
            recap_items = [
                ContinuityCarryItem(
                    item_id=f"takeaway_ep_{max(episode_number - 1, 0)}",
                    label=(
                        f"{takeaway.inherited_condition} — and the choice still in play: "
                        f"{takeaway.proximate_contingency}"
                    ),
                    kind="takeaway",
                    source_episode_number=max(episode_number - 1, 0),
                    priority="normal",
                    desired_surface="recap",
                    recommended_action="remind",
                )
            ]
        must_surface_early = [
            item for item in carry_items if item.desired_surface in {"recap", "opening"}
        ]
        must_leave_live: list[ContinuityCarryItem] = []
    else:
        recap_items = []
        must_surface_early = []
        must_leave_live = sorted(carry_items, key=_continuity_item_sort_key)

    open_question_texts = [
        question.text
        for question in narrative_state.listener.questions
        if question.status in {"open", "advanced", "reframed"}
    ]
    open_memory_thread_labels = [
        thread.label
        for thread in narrative_state.listener.memory_threads
        if thread.status in {"open", "refreshed"}
    ]
    host_open_pressures = [
        mystery.text
        for mystery in narrative_state.host.mysteries
        if mystery.status in {"open", "advanced", "reframed"}
    ]
    return {
        "phase": phase,
        "episode_number": episode_number,
        "last_episode_takeaway": (
            narrative_state.listener.last_episode_takeaway.model_dump()
            if narrative_state.listener.last_episode_takeaway is not None
            else None
        ),
        "recap_items": _continuity_items_to_payload(recap_items),
        "must_surface_early": _continuity_items_to_payload(
            sorted(must_surface_early, key=_continuity_item_sort_key)
        ),
        "must_leave_live": _continuity_items_to_payload(must_leave_live),
        "open_question_texts": open_question_texts,
        "open_memory_thread_labels": open_memory_thread_labels,
        "host_open_pressures": host_open_pressures,
    }


def _continuity_item_realized(item: dict[str, Any], observed_texts: list[str]) -> bool:
    label = str(item.get("label", "") or "").strip()
    if not label:
        return True
    normalized_label = " ".join(sorted(_normalize_section_text_tokens(label)))
    normalized_texts = [
        " ".join(sorted(_normalize_section_text_tokens(text))) for text in observed_texts
    ]
    if any(label.lower() in (text or "").lower() for text in observed_texts):
        return True
    label_tokens = _normalize_section_text_tokens(normalized_label)
    if not label_tokens:
        return True
    observed_tokens: set[str] = set()
    for text in normalized_texts:
        observed_tokens.update(_normalize_section_text_tokens(text))
    overlap = label_tokens & observed_tokens
    return len(overlap) >= min(2, len(label_tokens))


def _build_continuity_realization_diagnostics(
    *,
    episode_number: int,
    stage: str,
    framing: dict[str, Any],
    ordered_sections: list[dict[str, Any]],
    continuity_contract_pre: dict[str, Any],
    continuity_contract_post: dict[str, Any],
) -> dict[str, Any]:
    recap_text = str(framing.get("recap", "") or "")
    opening_question = str(framing.get("opening_question", "") or "")
    opening_zone_texts = [recap_text, opening_question]
    if ordered_sections:
        opening_zone_texts.append(str(ordered_sections[0].get("text", "") or ""))
    closing_zone_sections = (
        ordered_sections[-2:] if len(ordered_sections) >= 2 else ordered_sections
    )
    closing_zone_texts = [str(section.get("text", "") or "") for section in closing_zone_sections]
    recap_items = list(continuity_contract_pre.get("recap_items", []) or [])
    must_surface_early = list(continuity_contract_pre.get("must_surface_early", []) or [])
    must_leave_live = list(continuity_contract_post.get("must_leave_live", []) or [])

    per_item_results: list[dict[str, Any]] = []
    realized_item_ids: list[str] = []
    missed_item_ids: list[str] = []
    warning_labels: list[str] = []
    targeted_feedback: list[dict[str, str]] = []

    def _record(item: dict[str, Any], zone: str, observed_texts: list[str]) -> None:
        item_id = str(item.get("item_id", "") or "")
        label = str(item.get("label", "") or "")
        priority = str(item.get("priority", "normal") or "normal")
        realized = _continuity_item_realized(item, observed_texts)
        if realized:
            realized_item_ids.append(item_id)
        else:
            missed_item_ids.append(item_id)
            warning_labels.append(f"{zone}_miss:{item_id}")
            if zone == "recap":
                instruction = f"Restore `{label}` in framing.recap."
            elif zone == "opening":
                instruction = (
                    f"Re-surface `{label}` in the opening zone rather than only later exposition."
                )
            else:
                instruction = f"Leave `{label}` live in the closing or after-pressure zone so the next episode can inherit it."
            targeted_feedback.append(
                {
                    "item_id": item_id,
                    "label": label,
                    "zone": zone,
                    "priority": priority,
                    "instruction": instruction,
                }
            )
        per_item_results.append(
            {
                "item_id": item_id,
                "label": label,
                "zone": zone,
                "priority": priority,
                "realized": realized,
            }
        )

    for item in recap_items:
        _record(item, "recap", [recap_text])
    for item in must_surface_early:
        _record(item, "opening", opening_zone_texts)
    for item in must_leave_live:
        _record(item, "closing", closing_zone_texts)

    return {
        "episode_number": episode_number,
        "stage": stage,
        "expected_recap_items": recap_items,
        "expected_opening_items": must_surface_early,
        "expected_closing_items": must_leave_live,
        "per_item_results": per_item_results,
        "realized_item_ids": sorted(set(filter(None, realized_item_ids))),
        "missed_item_ids": sorted(set(filter(None, missed_item_ids))),
        "warning_labels": warning_labels,
        "targeted_feedback": targeted_feedback,
    }


_HOST_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(follow|notice|hold|remember)\b", re.IGNORECASE),
    re.compile(r"\b(we|our|us|you|your)\b", re.IGNORECASE),
    re.compile(r"\b(in plain english|what this means|in practice)\b", re.IGNORECASE),
    re.compile(r"\b(the problem|the point|which meant|that left)\b", re.IGNORECASE),
)


def _count_host_cue_clusters(text: str) -> int:
    sentences = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    count = 0
    for sentence in sentences:
        if not sentence:
            continue
        if any(pattern.search(sentence) for pattern in _HOST_CUE_PATTERNS):
            count += 1
    return count


def _build_host_move_text_diagnostics(
    *,
    text_by_section_id: dict[str, str],
    plan: EpisodePlan | None,
) -> dict[str, Any]:
    if plan is None:
        return {
            "planned_host_phase_count": 0,
            "approx_realized_host_phase_count": 0,
            "approx_unrealized_phase_ids": [],
            "sections_with_first_person_plural": [],
            "section_host_cue_counts": {},
            "sections_with_host_phase_collapse": [],
            "editorial_host_target_count": 0,
            "sections_with_editorial_host_target_pressure": [],
            "editorial_host_target_examples": [],
            "phase_trace": [],
        }

    phase_trace: list[dict[str, Any]] = []
    approx_realized_phase_ids: list[str] = []
    approx_unrealized_phase_ids: list[str] = []
    sections_with_first_person_plural: set[str] = set()
    phase_counts_by_section: dict[str, int] = {}
    for scene in plan.scene_cards:
        anchor_tokens = _scene_host_anchor_tokens(scene)
        section_text = text_by_section_id.get(scene.section_id, "")
        if re.search(r"\b(we|our|us)\b", section_text.lower()):
            sections_with_first_person_plural.add(scene.section_id)
        section_tokens = _normalize_section_text_tokens(section_text)
        for phase in _HOST_MOVE_PHASE_ORDER:
            phase_cues = getattr(scene.host_moves, phase)
            if not phase_cues:
                continue
            phase_counts_by_section[scene.section_id] = (
                phase_counts_by_section.get(scene.section_id, 0) + 1
            )
            phase_target = " ".join(cue.target for cue in phase_cues if cue.target)
            target_tokens = _normalize_section_text_tokens(phase_target)
            approx_realized = not target_tokens or bool(target_tokens & section_tokens)
            editorial_scaffolding_flags = _host_move_editorial_scaffolding_flags(
                phase_target,
                anchor_tokens=anchor_tokens,
            )
            phase_id = f"{scene.scene_id}:{phase}"
            if approx_realized:
                approx_realized_phase_ids.append(phase_id)
            else:
                approx_unrealized_phase_ids.append(phase_id)
            phase_trace.append(
                {
                    "scene_id": scene.scene_id,
                    "section_id": scene.section_id,
                    "phase": phase,
                    "cue_count": len(phase_cues),
                    "move_types": [cue.move_type for cue in phase_cues],
                    "host_target": phase_target,
                    "address_modes": [cue.address_mode for cue in phase_cues],
                    "approx_realized": approx_realized,
                    "editorial_scaffolding_flags": editorial_scaffolding_flags,
                    "first_person_plural_present": scene.section_id
                    in sections_with_first_person_plural,
                }
            )

    section_host_cue_counts = {
        section_id: _count_host_cue_clusters(text)
        for section_id, text in text_by_section_id.items()
    }
    sections_with_host_phase_collapse = sorted(
        section_id
        for section_id, count in phase_counts_by_section.items()
        if count >= 4 and section_host_cue_counts.get(section_id, 0) <= 2
    )
    editorial_phase_entries = [row for row in phase_trace if row.get("editorial_scaffolding_flags")]
    editorial_host_target_examples = [
        {
            "scene_id": str(row["scene_id"]),
            "section_id": str(row["section_id"]),
            "phase": str(row["phase"]),
            "host_target": str(row["host_target"]),
            "editorial_scaffolding_flags": list(row.get("editorial_scaffolding_flags", [])),
        }
        for row in editorial_phase_entries[:8]
    ]

    return {
        "planned_host_phase_count": len(phase_trace),
        "approx_realized_host_phase_count": len(approx_realized_phase_ids),
        "approx_realized_phase_ids": approx_realized_phase_ids,
        "approx_unrealized_phase_ids": approx_unrealized_phase_ids,
        "sections_with_first_person_plural": sorted(sections_with_first_person_plural),
        "section_host_cue_counts": section_host_cue_counts,
        "sections_with_host_phase_collapse": sections_with_host_phase_collapse,
        "editorial_host_target_count": len(editorial_phase_entries),
        "sections_with_editorial_host_target_pressure": sorted(
            {str(row["section_id"]) for row in editorial_phase_entries}
        ),
        "editorial_host_target_examples": editorial_host_target_examples,
        "phase_trace": phase_trace,
    }


def _section_text_realized(target_text: str, observed_texts: list[str]) -> bool:
    target_tokens = _normalize_section_text_tokens(target_text)
    if not target_tokens:
        return True
    observed_tokens: set[str] = set()
    for text in observed_texts:
        observed_tokens.update(_normalize_section_text_tokens(text))
    return bool(target_tokens & observed_tokens)


def _opening_sentence_window(text: str, *, sentence_limit: int = 2) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if sentence.strip()
    ]
    return " ".join(sentences[:sentence_limit])


def _build_section_sonic_realization_diagnostics(
    *,
    episode_number: int,
    stage: str,
    sections: list[ProseSection | SpokenSection],
) -> dict[str, Any]:
    section_reports: list[dict[str, Any]] = []
    warning_labels: list[str] = []
    targeted_feedback: list[dict[str, str]] = []

    for section in sections:
        section_sonic_plan = section.section_sonic_plan
        if section_sonic_plan is None:
            section_reports.append(
                {
                    "section_id": section.section_id,
                    "has_section_sonic_plan": False,
                    "opening_status": "no_plan",
                    "later_beats": [],
                    "warnings": [],
                }
            )
            continue

        opening_window = _opening_sentence_window(section.text, sentence_limit=2)
        opening_anchor_realized = _sonic_phrase_realized(
            section_sonic_plan.opening_anchor,
            [opening_window],
        )
        if opening_anchor_realized:
            opening_status = "realized"
        elif _text_has_sound_lexicon(opening_window):
            opening_status = "sonic_present_but_anchor_paraphrased"
        else:
            opening_status = "dry"

        section_warnings: list[str] = []
        if opening_status == "dry":
            section_warnings.append(
                f"section_sonic_opening_not_realized_early: {section.section_id}"
            )
            targeted_feedback.append(
                {
                    "section_id": section.section_id,
                    "issue": "opening",
                    "instruction": "Spend the section_sonic_plan opening anchor in the first 1-2 sentences.",
                }
            )
            if section_sonic_plan.obligation == SectionSonicObligation.REQUIRED:
                section_warnings.append(
                    f"section_opening_dry_despite_required_section_sonic_plan: {section.section_id}"
                )

        later_beat_reports: list[dict[str, Any]] = []
        for beat in section_sonic_plan.later_beats:
            realized = _sonic_phrase_realized(beat.cue, [section.text])
            later_beat_reports.append(
                {
                    "moment": beat.moment,
                    "cue": beat.cue,
                    "realized": realized,
                }
            )
            if not realized:
                section_warnings.append(
                    f"section_sonic_later_beat_not_realized: {section.section_id} -> {beat.moment}"
                )
                targeted_feedback.append(
                    {
                        "section_id": section.section_id,
                        "issue": "later_beat",
                        "instruction": f"Realize the planned later sonic beat `{beat.moment}` inside the section body.",
                    }
                )

        warning_labels.extend(section_warnings)
        section_reports.append(
            {
                "section_id": section.section_id,
                "has_section_sonic_plan": True,
                "obligation": section_sonic_plan.obligation.value,
                "opening_anchor": section_sonic_plan.opening_anchor,
                "opening_pressure": section_sonic_plan.opening_pressure,
                "opening_status": opening_status,
                "later_beats": later_beat_reports,
                "warnings": section_warnings,
            }
        )

    return {
        "episode_number": episode_number,
        "stage": stage,
        "section_reports": section_reports,
        "warning_labels": warning_labels,
        "warning_count": len(warning_labels),
        "targeted_feedback": targeted_feedback,
    }


def _build_section_plan_realization(
    *,
    episode: EpisodeArchitecture,
    scene_cards: list[SceneCardDraft | SceneCard],
    words_per_minute: float = 145.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    scenes_by_section: dict[str, list[SceneCardDraft | SceneCard]] = {
        section.section_id: [] for section in episode.sections
    }
    for scene in scene_cards:
        if scene.section_id in scenes_by_section:
            scenes_by_section[scene.section_id].append(scene)

    section_reports: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for section in episode.sections:
        section_scene_cards = scenes_by_section.get(section.section_id, [])
        scene_card_count = len(section_scene_cards)
        projected_word_count = sum(
            _project_scene_word_count(scene, words_per_minute=words_per_minute)
            for scene in section_scene_cards
        )
        structural_card_count = sum(
            1 for scene in section_scene_cards if _is_structural_scene_card(scene)
        )
        used_priority_core_passage_ids: list[str] = []
        scene_realization_texts: list[str] = []
        for scene in section_scene_cards:
            scene_realization_texts.extend(
                [
                    scene.title,
                    scene.entry_image,
                    scene.observable_detail,
                    scene.audible_detail,
                    scene.beat_change,
                    *scene.must_land_facts.ordered_facts(),
                ]
            )
            for passage_id in scene.passage_ids:
                if (
                    passage_id in section.priority_core_passage_ids
                    and passage_id not in used_priority_core_passage_ids
                ):
                    used_priority_core_passage_ids.append(passage_id)

        section_warnings: list[str] = []
        unused_priority_core_passage_ids = [
            passage_id
            for passage_id in section.priority_core_passage_ids
            if passage_id not in used_priority_core_passage_ids
        ]
        if section.priority_core_passage_ids and unused_priority_core_passage_ids:
            section_warnings.append(
                "section_priority_core_passages_unused: "
                f"{section.section_id} -> {_preview_ids(unused_priority_core_passage_ids)}"
            )
        unrealized_must_stage_beats = [
            beat
            for beat in section.must_stage_beats
            if not _section_text_realized(beat, scene_realization_texts)
        ]
        if unrealized_must_stage_beats:
            section_warnings.append(
                "section_must_stage_beats_not_realized: "
                f"{section.section_id} -> {len(unrealized_must_stage_beats)} beats"
            )
        if scene_card_count >= 5:
            section_warnings.append(
                f"section_scene_card_load_high: {section.section_id} -> {scene_card_count} cards"
            )
        if projected_word_count > 2200:
            section_warnings.append(
                "section_projected_word_count_high: "
                f"{section.section_id} -> {projected_word_count} projected words"
            )
        section_sonic_plan = section.section_sonic_plan
        if section_sonic_plan is not None and section_scene_cards:
            first_scene = section_scene_cards[0]
            if not _has_audible_anchor(first_scene):
                warning_name = (
                    "section_sonic_plan_required_missing_first_scene_derivation"
                    if section_sonic_plan.obligation == SectionSonicObligation.REQUIRED
                    else "section_sonic_plan_preferred_but_first_scene_dry"
                )
                section_warnings.append(
                    f"{warning_name}: {section.section_id} -> {first_scene.scene_id}"
                )
            elif _scene_audible_detail_is_section_copy(
                section_sonic_plan.opening_anchor,
                first_scene.audible_detail,
            ):
                section_warnings.append(
                    "scene_audible_detail_verbatim_section_copy: "
                    f"{section.section_id} -> {first_scene.scene_id}"
                )
            for beat in section_sonic_plan.later_beats:
                binding_scene_id = _find_later_beat_binding_scene_id(
                    beat,
                    section_scene_cards,
                )
                if binding_scene_id is None:
                    section_warnings.append(
                        "section_sonic_plan_later_beat_unbound: "
                        f"{section.section_id} -> {beat.moment}"
                    )

        section_reports.append(
            {
                "section_id": section.section_id,
                "section_anchor": section.section_anchor,
                "section_sonic_plan": (
                    section_sonic_plan.model_dump(mode="json")
                    if section_sonic_plan is not None
                    else None
                ),
                "scene_card_count": scene_card_count,
                "projected_word_count": projected_word_count,
                "structural_card_count": structural_card_count,
                "declared_must_stage_beats": list(section.must_stage_beats),
                "unrealized_must_stage_beats": unrealized_must_stage_beats,
                "used_priority_core_passage_ids": used_priority_core_passage_ids,
                "unused_priority_core_passage_ids": unused_priority_core_passage_ids,
                "warning_count": len(section_warnings),
                "warnings": section_warnings,
            }
        )
        all_warnings.extend(section_warnings)
    return section_reports, all_warnings


def _build_state_alignment_diagnostics(
    *,
    architecture: EpisodeArchitecture,
    scene_cards: list[SceneCardDraft | SceneCard],
) -> tuple[dict[str, Any], list[str]]:
    host_change_section_ids = [
        section.section_id
        for section in architecture.sections
        if section.section_progression.state_effects.host_mystery_moves
        or section.section_progression.state_effects.host_assumption_moves
        or section.section_progression.state_effects.host_theory_moves
    ]
    host_change_section_id_set = set(host_change_section_ids)
    scene_cards_by_section: dict[str, list[SceneCardDraft | SceneCard]] = {}
    for scene in scene_cards:
        scene_cards_by_section.setdefault(scene.section_id, []).append(scene)
    section_ids_missing_scene_host_moves = [
        section_id
        for section_id in host_change_section_ids
        if not any(
            _scene_host_phase_bucket_count(scene) > 0
            for scene in scene_cards_by_section.get(section_id, [])
        )
    ]
    scene_ids_with_epistemic_moves_outside_host_change_sections: list[str] = []
    for scene in scene_cards:
        if scene.section_id in host_change_section_id_set:
            continue
        scene_cues = [
            cue for phase in _HOST_MOVE_PHASE_ORDER for cue in getattr(scene.host_moves, phase)
        ]
        if any(cue.move_type in {"uncertainty", "revision", "surprise"} for cue in scene_cues):
            scene_ids_with_epistemic_moves_outside_host_change_sections.append(scene.scene_id)
    warnings: list[str] = []
    if section_ids_missing_scene_host_moves:
        warnings.append(
            "state_alignment_missing_scene_host_moves: "
            f"{_preview_ids(section_ids_missing_scene_host_moves)}"
        )
    if scene_ids_with_epistemic_moves_outside_host_change_sections:
        warnings.append(
            "state_alignment_epistemic_moves_outside_host_change_sections: "
            f"{_preview_ids(scene_ids_with_epistemic_moves_outside_host_change_sections)}"
        )
    diagnostics = {
        "host_change_section_ids": host_change_section_ids,
        "section_ids_missing_scene_host_moves": section_ids_missing_scene_host_moves,
        "scene_ids_with_epistemic_moves_outside_host_change_sections": (
            scene_ids_with_epistemic_moves_outside_host_change_sections
        ),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return diagnostics, warnings


def _validate_continuity_recap_requirement(
    *,
    episode_number: int,
    plan: EpisodePlanDraft,
    continuity_contract_pre: dict[str, Any],
) -> None:
    recap_items = list(continuity_contract_pre.get("recap_items", []) or [])
    recap_text = str(plan.framing.recap or "").strip()
    if episode_number <= 1:
        if recap_text:
            raise ComplianceViolationError(
                "Episode 1 must not emit framing.recap.",
                data={
                    "issue": "continuity_recap_forbidden",
                    "episode_number": episode_number,
                    "instruction": "Set framing.recap to null for episode 1.",
                },
            )
        return
    if recap_items and not recap_text:
        raise ComplianceViolationError(
            "Episode planning omitted framing.recap despite inherited continuity items.",
            data={
                "issue": "continuity_recap_missing",
                "episode_number": episode_number,
                "instruction": "Add a 1-2 sentence framing.recap that recalls inherited continuity burden before the new opening.",
                "missing_item_ids": [str(item.get("item_id", "") or "") for item in recap_items],
            },
        )


def _scene_counts_toward_spine(
    scene: SceneCardDraft | SceneCard,
    strategy_episode: StrategyEpisode,
) -> bool:
    return bool(scene.must_land_facts.ordered_facts() or scene.beat_change.strip())


def _build_thread_plan_warnings(
    *,
    strategy_episode: StrategyEpisode,
    architecture: EpisodeArchitecture,
    scene_cards: list[SceneCardDraft] | list[SceneCard],
) -> list[str]:
    """Planning-stage checks that the carried human thread is actually realized in scenes."""
    thread = strategy_episode.human_thread
    warnings: list[str] = []
    if thread is None:
        return warnings
    members_by_id = {member.member_id: member for member in thread.members}
    cards_by_section: dict[str, list] = {}
    for card in scene_cards:
        cards_by_section.setdefault(card.section_id, []).append(card)
    bound = 0
    carried_or_relay = 0
    for section in architecture.sections:
        binding = section.thread_binding
        if binding is None:
            continue
        bound += 1
        if (
            binding.presence == ThreadSectionPresence.CARRIED
            or binding.fallback_mode == ThreadFallbackMode.FAMILY_RELAY
        ):
            carried_or_relay += 1
        if binding.presence not in (
            ThreadSectionPresence.CARRIED,
            ThreadSectionPresence.PERIPHERAL,
        ):
            continue
        member = members_by_id.get(binding.carrying_member_id or "")
        placed = False
        for card in cards_by_section.get(section.section_id, []):
            for actor in card.actors or []:
                if member is None:
                    continue
                if member.actor_id and actor.actor_id == member.actor_id:
                    placed = True
                elif actor.name and normalize_actor_name(actor.name) == normalize_actor_name(
                    member.display_name
                ):
                    placed = True
                if placed:
                    break
            if placed:
                break
        if not placed:
            warnings.append(f"thread_section_missing_scene_carrier: {section.section_id}")
    if bound and carried_or_relay / bound < 0.6:
        warnings.append("thread_plan_continuity_share_low")
    return warnings


def _build_host_beat_placement_warnings(
    *,
    architecture: EpisodeArchitecture,
    scene_cards: list[SceneCardDraft] | list[SceneCard],
) -> list[str]:
    """Planning-stage checks for architecture-designated host beats and persona variety."""
    warnings: list[str] = []
    designation_ids = {
        designation.host_beat_id
        for section in architecture.sections
        for designation in section.host_beat_designations
    }
    assigned_ids: set[str] = set()
    persona_phases: list[str] = []
    persona_targets: list[str] = []
    for card in scene_cards:
        assigned_ids.update(card.host_beat_ids or [])
        for phase in ("open", "pivot", "close"):
            for cue in getattr(card.host_moves, phase, []) or []:
                if cue.move_type == "persona_aside":
                    persona_phases.append(phase)
                    persona_targets.append(cue.target.strip().lower())
    for beat_id in sorted(designation_ids - assigned_ids):
        warnings.append(f"host_beat_unassigned: {beat_id}")
    if len(persona_phases) >= 3 and len(set(persona_phases)) == 1:
        warnings.append("persona_aside_phase_monotony")
    seen: set[str] = set()
    for target in persona_targets:
        if target and target in seen:
            warnings.append("persona_aside_repeat_target")
            break
        seen.add(target)
    return warnings


def _build_spine_plan_diagnostics(
    *,
    strategy_episode: StrategyEpisode,
    plan: EpisodePlanDraft | EpisodePlan,
    scene_job_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene_cards = list(plan.scene_cards)
    if scene_job_budget is None:
        inferred_mode = PodcastMode.MINIFIED if len(scene_cards) <= 20 else PodcastMode.FULL
        scene_job_budget = scene_job_budget_for_mode(inferred_mode)
    total_scene_count = len(scene_cards)
    spine_scene_cards = [
        scene for scene in scene_cards if _scene_counts_toward_spine(scene, strategy_episode)
    ]
    scene_share = len(spine_scene_cards) / total_scene_count if total_scene_count else 0.0

    total_word_weight = sum(max(1, len(scene.beat_change.split())) for scene in scene_cards)
    spine_word_weight = sum(max(1, len(scene.beat_change.split())) for scene in spine_scene_cards)
    word_share = spine_word_weight / total_word_weight if total_word_weight else 0.0

    support_scene_count = total_scene_count - len(spine_scene_cards)
    last_scene = scene_cards[-1] if scene_cards else None
    ending_alignment_pass = bool(
        last_scene and _scene_counts_toward_spine(last_scene, strategy_episode)
    )
    dropped_support_primitive_reasons = dict(plan.dropped_support_primitive_reasons)
    dropped_support_primitive_ids = list(dropped_support_primitive_reasons)
    support_takeover_detected = support_scene_count > max(
        1, int(math.floor(total_scene_count * 0.35))
    )
    secondary_chain_detected = support_scene_count >= max(2, len(spine_scene_cards))
    scene_job_budget_diagnostics, _scene_job_budget_warnings = _build_scene_job_budget_diagnostics(
        scene_cards=scene_cards,
        scene_job_budget=scene_job_budget,
        answer_scene_card_id=plan.answer_scene_card_id,
    )
    diagnostics = {
        "core_scene_share": round(scene_share, 4),
        "core_word_share": round(word_share, 4),
        "ending_alignment_pass": ending_alignment_pass,
        "secondary_chain_detected": secondary_chain_detected,
        "support_takeover_detected": support_takeover_detected,
        "recall_takeover_detected": False,
        "new_load_bearing_question_detected": False,
        "second_ending_detected": False,
        "host_phase_count": sum(_scene_host_phase_bucket_count(scene) for scene in scene_cards),
        **scene_job_budget_diagnostics,
        "spine_drift_detected": (
            scene_share < 0.60
            or word_share < 0.60
            or support_takeover_detected
            or not ending_alignment_pass
        ),
        "dropped_support_primitive_ids": dropped_support_primitive_ids,
        "dropped_support_primitive_reasons": dropped_support_primitive_reasons,
        "failure_labels": [],
        "scene_trace": [
            {
                "scene_id": scene.scene_id,
                "beat_change": scene.beat_change,
                "host_phase_bucket_count": _scene_host_phase_bucket_count(scene),
                "host_move_types_by_phase": {
                    phase: [cue.move_type for cue in getattr(scene.host_moves, phase)]
                    for phase in _HOST_MOVE_PHASE_ORDER
                    if getattr(scene.host_moves, phase)
                },
                "scene_job": scene.scene_job,
                "counts_toward_spine_dominance": _scene_counts_toward_spine(
                    scene,
                    strategy_episode,
                ),
                "contributed_to_drift": not _scene_counts_toward_spine(
                    scene,
                    strategy_episode,
                ),
            }
            for scene in scene_cards
        ],
    }
    if diagnostics["core_scene_share"] < 0.60 or diagnostics["core_word_share"] < 0.60:
        diagnostics["failure_labels"].append("spine_underweight")
    if diagnostics["secondary_chain_detected"]:
        diagnostics["failure_labels"].append("parallel_full_arc")
    if diagnostics["support_takeover_detected"]:
        diagnostics["failure_labels"].append("support_thread_takeover")
    if not diagnostics["ending_alignment_pass"]:
        diagnostics["failure_labels"].append("ending_displacement")
    if diagnostics["close_duplicates_answer"]:
        diagnostics["failure_labels"].append("close_duplicates_answer")
    return diagnostics


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
        # Cross-episode signal caches (Changes 2 + 3).
        self._latest_lint_flags_by_episode: dict[int, dict] = {}
        self._series_state_lock = asyncio.Lock()

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

        self.chapter_summary_agent = ChapterSummaryAgent(
            self.llm, max_retry_attempts=_retries("chapter_summary")
        )
        self.book_summary_agent = BookSummaryAgent(
            self.llm, max_retry_attempts=_retries("book_summary")
        )
        self.theme_decomposition_agent = ThemeDecompositionAgent(
            self.llm, max_retry_attempts=_retries("theme_decomposition")
        )
        self.passage_extraction_agent = PassageExtractionAgent(
            self.llm, max_retry_attempts=_retries("passage_extraction")
        )
        self.synthesis_primitives_agent = SynthesisPrimitivesAgent(
            self.llm, max_retry_attempts=_retries("primitive_substrate_extraction")
        )
        self.excerpt_extraction_agent = ExcerptExtractionAgent(
            self.llm, max_retry_attempts=_retries("excerpt_extraction")
        )
        self.primitive_function_tagging_agents = {
            substrate: PrimitiveFunctionTaggingAgent(
                self.llm,
                substrate=substrate,
                max_retry_attempts=_retries(f"primitive_function_tagging_{substrate}"),
            )
            for substrate in PRIMITIVE_SUBSTRATES
        }
        self.scene_discovery_agent = SceneDiscoveryAgent(
            self.llm, max_retry_attempts=_retries("scene_discovery")
        )
        self.narrative_strategy_skeleton_agent = NarrativeStrategySkeletonAgent(
            self.llm, max_retry_attempts=_retries("narrative_strategy_skeleton")
        )
        self.narrative_strategy_enrichment_agent = NarrativeStrategyEnrichmentAgent(
            self.llm, max_retry_attempts=_retries("narrative_strategy_enrichment")
        )
        self.episode_architecture_agent = EpisodeArchitectureAgent(
            self.llm, max_retry_attempts=_retries("episode_architecture")
        )
        self.episode_planning_agent = EpisodePlanningAgent(
            self.llm, max_retry_attempts=_retries("episode_planning")
        )
        self.narrative_state_reconciler_agent = NarrativeStateReconcilerAgent(
            self.llm, max_retry_attempts=_retries("narrative_state_reconciliation")
        )
        self.writing_agent = WritingAgent(self.llm, max_retry_attempts=_retries("episode_writing"))
        self.writing_agent_no_citations = WritingAgentNoCitations(
            self.llm,
            max_retry_attempts=_retries("episode_writing"),
        )
        self.quality_judge_agent = QualityJudgeAgent(
            self.llm, max_retry_attempts=_retries("quality_judge")
        )
        self.style_audit_agent = StyleAuditAgent(
            self.llm, max_retry_attempts=_retries("style_audit")
        )
        self.spoken_delivery_agent = SpokenDeliveryAgent(
            self.llm, max_retry_attempts=_retries("spoken_delivery")
        )

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
        pipeline_config = resolve_pipeline_config_for_mode(config or PipelineConfig())
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
            podcast_mode=pipeline_config.podcast_mode.value,
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
        try:
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
                        i,
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

            project = project.model_copy(update={"books": successful_books})
            if len(successful_books) < 2:
                raise RuntimeError(
                    f"Only {len(successful_books)} books ingested successfully. Minimum 2 required."
                )

            project = project.model_copy(update={"status": ProjectStatus.ANALYZING})
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
            (
                (synthesis_primitives, synthesis_actor_metrics),
                excerpts,
            ) = await asyncio.gather(
                self._map_synthesis(project, corpus, project_dir, actor_metadata),
                self._extract_excerpts(project, corpus, project_dir, actor_metadata),
            )
            actor_metrics["synthesis_primitives"] = synthesis_actor_metrics.get("primitives", {})

            if synthesis_primitives.quality_score < pipeline_config.synthesis_quality_threshold:
                logger.warning(
                    "Synthesis quality %.2f below threshold %.2f. "
                    "Books may lack thematic overlap for strong synthesis.",
                    synthesis_primitives.quality_score,
                    pipeline_config.synthesis_quality_threshold,
                )
                self.run_logger.log(
                    "synthesis_quality_warning",
                    score=synthesis_primitives.quality_score,
                    threshold=pipeline_config.synthesis_quality_threshold,
                )

            scene_discovery = await self._discover_scenes(
                project,
                synthesis_primitives,
                corpus,
                project_dir,
                actor_metadata,
            )
            strategy, strategy_actor_metrics = await self._choose_narrative_strategy(
                project,
                synthesis_primitives,
                project_dir,
                scene_discovery,
                actor_metadata,
                excerpts,
            )
            actor_metrics["narrative_strategy"] = strategy_actor_metrics
            synthesis_map = await self._materialize_selected_primitives(
                project,
                synthesis_primitives,
                strategy,
                project_dir,
            )
            retained_excerpts = await self._materialize_selected_excerpts(
                project,
                excerpts,
                strategy,
                project_dir,
            )
            project = self._resolve_episode_count_from_strategy(project, strategy)
            _save_json(project_dir / "thematic_project.json", project)

            project = project.model_copy(update={"status": ProjectStatus.PLANNING})
            (
                episode_architectures,
                episode_plans,
                narrative_state_pre_by_episode,
                narrative_state_post_by_episode,
                planning_state_metrics,
            ) = await self._plan_series_with_narrative_state(
                project,
                synthesis_map,
                strategy,
                corpus,
                project_dir,
                actor_metadata,
                scene_discovery,
                retained_excerpts,
            )
            actor_metrics["episode_architecture"] = planning_state_metrics.get(
                "episode_architecture", {}
            )
            actor_metrics["episode_planning"] = planning_state_metrics.get("episode_planning", {})

            # Phase 3: Episode Production (parallel per episode)
            logger.info("Phase 3: Episode Production (%d episodes)", len(episode_plans))
            project = project.model_copy(update={"status": ProjectStatus.PRODUCING})

            sem = asyncio.Semaphore(max(1, pipeline_config.episode_write_concurrency))
            spoken_sem = asyncio.Semaphore(
                max(
                    1,
                    pipeline_config.spoken_delivery_concurrency
                    or pipeline_config.episode_write_concurrency,
                )
            )
            retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            retained_excerpt_by_id = retained_excerpts.excerpt_by_id()
            strategy_episode_by_number = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            architecture_by_number = {
                episode.episode_number: episode for episode in episode_architectures
            }
            ep_tasks = [
                self._produce_episode(
                    plan,
                    strategy_episode_by_number[plan.episode_number],
                    architecture_by_number[plan.episode_number],
                    project,
                    corpus,
                    actor_metadata,
                    project_dir,
                    _build_host_policy_payload(
                        strategy.narrator_profile,
                        narrative_state_pre=narrative_state_pre_by_episode.get(plan.episode_number),
                        narrative_state_post=narrative_state_post_by_episode.get(
                            plan.episode_number
                        ),
                    ),
                    retained_primitive_lookup,
                    sem,
                    spoken_sem,
                    strategy.series_explanation_registry,
                    narrative_state_pre_by_episode.get(plan.episode_number),
                    narrative_state_post_by_episode.get(plan.episode_number),
                    excerpt_by_id=retained_excerpt_by_id,
                )
                for plan in episode_plans
            ]
            ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)

            spoken_scripts: list[tuple[int, SpokenScript]] = []
            episode_errors = [result for result in ep_results if isinstance(result, Exception)]
            for error in episode_errors:
                logger.error("Episode production failed: %s", error)
            if episode_errors:
                raise RuntimeError(
                    "Episode production failed for "
                    f"{len(episode_errors)} episode(s): {episode_errors[0]}"
                )
            for result in ep_results:
                if not isinstance(result, Exception):
                    spoken_scripts.append(result)
            spoken_scripts.sort(key=lambda x: x[0])
            self._write_passage_utilization(
                project=project,
                corpus=corpus,
                episode_plans=episode_plans,
                project_dir=project_dir,
                episode_numbers=[episode_number for episode_number, _ in spoken_scripts],
            )
            actor_metrics["writing"] = self._build_writing_actor_metrics(
                project_dir, spoken_scripts
            )
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
            audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
            audio_errors = [result for result in audio_results if isinstance(result, Exception)]
            if audio_errors:
                raise RuntimeError(
                    f"Audio rendering failed for {len(audio_errors)} episode(s): {audio_errors[0]}"
                )

            project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
            _save_json(project_dir / "thematic_project.json", project)
            self.run_logger.log("pipeline_complete", project_id=project_id)
            logger.info("Pipeline complete. Artifacts at %s", project_dir)

            return project
        except Exception as exc:
            _persist_failed_project_state(
                project=project,
                project_dir=project_dir,
                run_logger=self.run_logger,
                exc=exc,
            )
            raise

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
                episode_number,
                manifest_path,
                project_dir,
                semaphore,
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
        book_slot: int,
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
            self.run_logger,
            f"ingest_book_{book_id[:8]}",
            project_dir,
            book_id=book_id,
            title=title,
            path=source_path,
        ) as ctx:
            # Stage 1: Read source
            raw_text = await asyncio.to_thread(read_source_text, path)
            total_words = len(raw_text.split())
            source_type = path.suffix.lower().lstrip(".")
            if source_type not in ("pdf", "txt", "md"):
                source_type = "txt"

            book_record = BookRecord(
                book_id=book_id,
                title=title,
                author=author,
                source_path=source_path,
                source_type=source_type,
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
            chunks = chunk_text(
                raw_text,
                book_id,
                chapters,
                chunking_config,
                book_slot=book_slot,
            )
            for c in chunks:
                c.metadata["author"] = author
                c.metadata["title"] = title
            book_record = book_record.model_copy(update={"chunk_count": len(chunks)})
            _save_json(book_dir / "book_record.json", book_record)

            # Stage 4: Embed & Store
            await asyncio.to_thread(self.vector_store.index_chunks, chunks, project_id)

            ctx["output_summary"] = {
                "book_id": book_id,
                "title": title,
                "chapters": len(chapters),
                "chunks": len(chunks),
                "words": total_words,
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
            self.run_logger,
            f"structure_{book_record.book_id[:8]}",
            project_dir,
            book_id=book_record.book_id,
            text_length=len(raw_text),
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
                updated.append(
                    chapter.model_copy(
                        update={
                            "analysis": summary.analysis,
                        }
                    )
                )
            chapters = updated

            ctx["output_summary"] = {
                "chapter_count": len(chapters),
                "windows_processed": 0,
            }
            return chapters

    # -----------------------------------------------------------------------
    # Phase 2: Thematic Intelligence
    # -----------------------------------------------------------------------

    async def _decompose_theme(
        self,
        project: ThematicProject,
        project_dir: Path,
    ) -> tuple[list[ThematicAxis], ActorMetadata, dict[str, Any]]:
        async with _stage_log(
            self.run_logger,
            "theme_decomposition",
            project_dir,
            theme=project.theme,
            sub_themes=project.sub_themes,
            book_count=len(project.books),
        ) as ctx:
            summary_payloads: list[tuple[str, dict[str, Any]]] = []
            for book in project.books:
                chapter_info = [_build_compact_chapter_projection(ch) for ch in book.chapters]
                summary_payloads.append(
                    (
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
                    )
                )

            summary_results = await asyncio.gather(
                *[
                    asyncio.to_thread(self.book_summary_agent.run, payload)
                    for _, payload in summary_payloads
                ]
            )
            book_summaries = {
                book_id: result.summary
                for (book_id, _), result in zip(summary_payloads, summary_results, strict=True)
            }

            payload = self.theme_decomposition_agent.build_payload(
                theme=project.theme,
                sub_themes=project.sub_themes,
                theme_elaboration=project.theme_elaboration,
                books=project.books,
                axis_count_min=project.config.min_axes,
                axis_count_max=project.config.max_axes,
                actor_count_min=(8 if project.config.podcast_mode == PodcastMode.MINIFIED else 16),
                actor_count_max=(24 if project.config.podcast_mode == PodcastMode.MINIFIED else 60),
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
                        book_id for book_id in expected_book_ids if book_id not in provided_book_ids
                    ]
                    if missing_book_ids:
                        missing_by_axis.append(
                            {
                                "axis_id": axis.axis_id,
                                "axis_name": axis.name,
                                "missing_book_ids": missing_book_ids,
                            }
                        )

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
                a for a in axes if sum(1 for s in a.relevance_by_book.values() if s >= 0.3) >= 2
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

            valid_axes = valid_axes[: project.config.max_axes]
            valid_axes, axis_actor_metrics = clean_axis_actor_ids(valid_axes, actor_metadata)
            actor_metrics.update(axis_actor_metrics)
            _save_json(
                project_dir / "thematic_axes.json",
                {"axes": [a.model_dump(mode="json") for a in valid_axes]},
            )
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
        self,
        project: ThematicProject,
        axes: list[ThematicAxis],
        project_dir: Path,
    ) -> ThematicCorpus:
        async with _stage_log(
            self.run_logger,
            "passage_extraction",
            project_dir,
            axis_count=len(axes),
            book_count=len(project.books),
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
                    chunk_count=book_by_id.get(book_id).chunk_count
                    if book_by_id.get(book_id)
                    else 0,
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
            max_log_per_book = max(
                100,
                max(
                    (info["per_book_budget"] for info in retrieval_depth_by_book.values()),
                    default=0,
                ),
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
                axis.axis_id: rank for rank, axis in enumerate(axis_priority_order, start=1)
            }
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
                    candidate["passage_id"]: candidate["text"] for candidate in prompt_candidates
                }
                rehydrated_passages = []
                for candidate in candidates:
                    score = scores_by_id.get(candidate["passage_id"])
                    if score is None:
                        continue
                    trimmed_text = trimmed_text_by_id.get(
                        candidate["passage_id"], candidate["text"]
                    )
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
                    and pair.relationship not in {SynthesisTag.AGREES_WITH, SynthesisTag.EXTENDS}
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

                prioritized_pairs = _prioritize_cross_book_pairs(validated_pairs)
                retained_pairs = prioritized_pairs[:2]
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
                        p.passage_id,
                    ),
                )
                axis_policy = {
                    "passage_selection_policy": {
                        "strategy": "retain_all_scored_passages",
                        "sort_order": "relevance_desc_then_quotability_desc",
                    },
                    "allocation_policy": retrieval_log["allocation_policy"],
                    "bm25_variant": retrieval_log["bm25_variant"],
                    "axis_candidate_budget": axis_candidate_budget_effective,
                    "pre_axis_budget": retrieval_log["pre_axis_budget"],
                    "axis_importance_score": axis.theme_importance_score,
                    "axis_importance_weight": retrieval_log["axis_importance_weight"],
                    "retrieval_depth_by_book": retrieval_log["retrieval_depth_by_book"],
                    "admitted_by_book": admitted_by_book,
                    "retained_count": len(retained_passages),
                }
                return (
                    axis.axis_id,
                    retained_passages,
                    retained_pairs,
                    candidate_count,
                    cross_pair_validation,
                    axis_policy,
                )

            def _process_axis(
                axis: ThematicAxis,
                *,
                axis_priority_rank: int,
            ) -> DeferredAxisWork:
                hits_by_book = self.retrieval.retrieve_for_axis(
                    axis=axis,
                    project_id=project.project_id,
                    book_ids=book_ids,
                    k_per_book=max_log_per_book,
                )

                axis_candidate_budget_effective = min(
                    max(
                        1,
                        pre_axis_budget_by_axis.get(axis.axis_id, axis_candidate_budget_target),
                    ),
                    sum(len(hits_by_book.get(bid, [])) for bid in book_ids),
                )
                retrieval_log: dict[str, Any] = {
                    "axis_id": axis.axis_id,
                    "axis_name": axis.name,
                    "axis_description": axis.description,
                    "max_log_per_book": max_log_per_book,
                    "axis_candidate_budget_target": axis_candidate_budget_target,
                    "axis_candidate_budget_effective": axis_candidate_budget_effective,
                    "axis_candidate_budget": axis_candidate_budget_effective,
                    "pre_axis_budget": pre_axis_budget_by_axis.get(
                        axis.axis_id, axis_candidate_budget_target
                    ),
                    "axis_importance_score": axis.theme_importance_score,
                    "axis_importance_weight": pre_axis_weight_by_axis.get(axis.axis_id, 1.0),
                    "pre_axis_total_budget": project.config.pre_axis_total_budget,
                    "budget_strategy": "axis_importance_weighted_pre_budget",
                    "passage_retrieval_percentage": project.config.passage_retrieval_percentage,
                    "passage_retrieval_min_per_book": project.config.passage_retrieval_min_per_book,
                    "passage_retrieval_max_per_book": project.config.passage_retrieval_max_per_book,
                    "allocation_policy": "global_bm25_okapi_top_n",
                    "bm25_variant": "okapi",
                    "axis_priority_rank": axis_priority_rank,
                    "axis_priority_basis": "theme_importance_score_desc",
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
                        "percentage_budget": retrieval_depth_by_book.get(bid, {}).get(
                            "percentage_budget", 0
                        ),
                        "retrieval_depth_budget": retrieval_depth_by_book.get(bid, {}).get(
                            "per_book_budget", 0
                        ),
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
                    retrieval_signal_by_book[bid] = sum(confidences[:top_n]) / max(1, top_n)

                retrieval_log["retrieval_signal_by_book"] = retrieval_signal_by_book
                retrieval_log["blended_score_by_book"] = relevance_by_book
                query_parts = [axis.name, axis.description]
                query_parts.extend(axis.guiding_questions)
                query_parts.extend(axis.keywords)
                axis_query_text = " ".join(part for part in query_parts if part).strip()
                query_terms = _bm25_query_terms(axis_query_text)

                all_rows = [row for bid in book_ids for row in rows_by_book.get(bid, [])]
                all_tokens = [_tokenize(row["hit"].text) for row in all_rows]
                idf, avg_len = _bm25_idf_and_avg_len(all_tokens)

                book_entry_by_id = {entry["book_id"]: entry for entry in retrieval_log["books"]}
                for bid in book_ids:
                    book_entry = book_entry_by_id.get(bid)
                    if book_entry is None:
                        continue
                    book_entry["eligible_total_count"] = len(rows_by_book.get(bid, []))

                for row, tokens in zip(all_rows, all_tokens, strict=False):
                    bm25_score = 0.0
                    if avg_len > 0 and query_terms:
                        bm25_score = _bm25_score(tokens, query_terms, idf, avg_len)
                    row["bm25_score"] = bm25_score
                    row["priority"] = bm25_score

                selected_rows = sorted(
                    all_rows,
                    key=lambda row: (
                        -float(row.get("bm25_score", 0.0)),
                        int(row["rank"]),
                        str(row["hit"].chunk_id),
                    ),
                )[:axis_candidate_budget_effective]

                admitted_by_book: dict[str, int] = {bid: 0 for bid in book_ids}
                for row in selected_rows:
                    bid = str(row["book_id"])
                    row["selection_phase"] = "global_bm25"
                    row["selection_score"] = float(row.get("bm25_score", 0.0))
                    admitted_by_book[bid] = admitted_by_book.get(bid, 0) + 1

                retrieval_log["bm25_k1"] = 1.5
                retrieval_log["bm25_b"] = 0.75
                retrieval_log["admitted_by_book"] = admitted_by_book

                selected_row_ids = {id(row) for row in selected_rows}
                candidates: list[dict[str, Any]] = []
                for bid in book_ids:
                    book_entry = book_entry_by_id.get(bid)
                    if book_entry is None:
                        continue
                    book_entry["selected_count"] = admitted_by_book.get(bid, 0)
                    for row in rows_by_book.get(bid, []):
                        hit = row["hit"]
                        used = id(row) in selected_row_ids
                        book_entry["candidates"].append(
                            {
                                "rank": row["rank"],
                                "used": used,
                                "global_priority": round(float(row["priority"]), 8),
                                "retrieval_confidence": round(
                                    float(row["retrieval_confidence"]), 8
                                ),
                                "bm25_score": round(float(row.get("bm25_score", 0.0)), 8),
                                "selection_phase": row["selection_phase"] if used else None,
                                "chapter_penalty": 0.0,
                                "selection_score": round(float(row["selection_score"]), 8)
                                if used
                                else None,
                                "chunk_id": hit.chunk_id,
                                "chapter_id": hit.chapter_id,
                                "score": hit.score,
                                "text": hit.text,
                                "metadata": hit.metadata,
                            }
                        )
                        if not used:
                            continue
                        candidates.append(
                            {
                                "passage_id": hit.chunk_id,
                                "book_id": bid,
                                "chunk_ids": [hit.chunk_id],
                                "text": hit.text,
                                "chapter_ref": hit.metadata.get("chapter_id", ""),
                                "axis_id": axis.axis_id,
                                "author": row["author"],
                                "title": row["title"],
                            }
                        )

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
                        "bm25_variant": retrieval_log["bm25_variant"],
                        "axis_candidate_budget": axis_candidate_budget_effective,
                        "pre_axis_budget": retrieval_log["pre_axis_budget"],
                        "axis_importance_score": axis.theme_importance_score,
                        "axis_importance_weight": retrieval_log["axis_importance_weight"],
                        "retrieval_depth_by_book": retrieval_log["retrieval_depth_by_book"],
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
                    candidate["passage_id"]: candidate["text"] for candidate in candidates
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
                    axis_id=axis.axis_id,
                    axis_name=axis.name,
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
                extraction_sem = asyncio.Semaphore(
                    max(1, project.config.passage_extraction_concurrency)
                )

                async def _run_deferred(
                    axis_id: str,
                    worker: Callable[[], AxisExtractionResult],
                ) -> tuple[str, AxisExtractionResult]:
                    async with extraction_sem:
                        result = await asyncio.to_thread(worker)
                        return axis_id, result

                deferred_results = await asyncio.gather(
                    *[_run_deferred(axis_id, worker) for axis_id, worker in deferred_work]
                )
                for axis_id, result in deferred_results:
                    results_by_axis[axis_id] = result

            results = [
                results_by_axis[axis.axis_id]
                for axis in axis_priority_order
                if axis.axis_id in results_by_axis
            ]

            cross_pair_validation_by_axis: dict[str, dict[str, Any]] = {}
            for (
                axis_id,
                top_passages,
                cross_pairs,
                candidate_count,
                cross_pair_validation,
                axis_policy,
            ) in results:
                candidate_counts_by_axis[axis_id] = candidate_count
                all_passages_by_axis[axis_id] = top_passages
                all_cross_pairs.extend(cross_pairs)
                cross_pair_validation_by_axis[axis_id] = cross_pair_validation
                axis_policy_by_axis[axis_id] = axis_policy

            # ---- Retrieval metrics ----
            retrieval_metrics: dict[str, Any] = {
                "per_axis": {},
                "per_book": {},
                "summary": {},
            }

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
                    "avg_relevance_score": round(
                        sum(relevance_scores) / max(1, len(relevance_scores)), 3
                    ),
                    "avg_quotability_score": round(
                        sum(quotability_scores) / max(1, len(quotability_scores)), 3
                    ),
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
                    p
                    for passages in all_passages_by_axis.values()
                    for p in passages
                    if p.book_id == book.book_id
                ]
                axis_admitted_shares = [
                    float(policy.get("admitted_by_book", {}).get(book.book_id, 0))
                    / max(1, int(policy.get("axis_candidate_budget", 0)))
                    for policy in axis_policy_by_axis.values()
                ]
                avg_admitted_share = (
                    sum(axis_admitted_shares) / len(axis_admitted_shares)
                    if axis_admitted_shares
                    else 0.0
                )
                size_share = float(book_size_share_by_book.get(book.book_id, 0.0))
                retrieval_metrics["per_book"][book.book_id] = {
                    "title": book.title,
                    "total_passages": len(book_passages),
                    "axes_with_passages": sum(
                        1
                        for passages in all_passages_by_axis.values()
                        if any(p.book_id == book.book_id for p in passages)
                    ),
                    "avg_relevance": round(
                        sum(p.relevance_score for p in book_passages) / max(1, len(book_passages)),
                        3,
                    ),
                    "size_share": round(size_share, 4),
                    "avg_axis_admitted_share": round(avg_admitted_share, 4),
                    "admitted_minus_size_share": round(avg_admitted_share - size_share, 4),
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
                    1
                    for passages in all_passages_by_axis.values()
                    if any(p.book_id == book.book_id for p in passages)
                )
                book_coverage[book.book_id] = CoverageStats(
                    total_passages=total,
                    axes_covered=axes_covered,
                    coverage_ratio=axes_covered / max(1, len(axes)),
                )

            corpus = ThematicCorpus(
                project_id=project.project_id,
                axes=axes,
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
    ) -> tuple[PrimitiveFunctionTaggingArtifact, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "synthesis_mapping",
            project_dir,
            axis_count=len(corpus.axes),
            total_passages=corpus.total_passages,
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
                "top_tier_passages": 0,
                "mid_tier_passages": 0,
                "tail_tier_passages": 0,
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
                floor_budget_fraction=project.config.synthesis_floor_budget_fraction,
                axis_floor_min=project.config.synthesis_axis_floor_min,
                axis_floor_max=project.config.synthesis_axis_floor_max,
                axis_ceiling_multiplier=project.config.synthesis_axis_ceiling_multiplier,
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
                        "text": passage.full_text.strip() or passage.text,
                    }
                    for passage in passages
                ]
                if axis is not None:
                    synthesis_keep_fraction_by_passage_id, passage_tier_counts = (
                        _resolve_synthesis_bm25_keep_fraction_by_passage(
                            passages,
                            top_fraction=project.config.synthesis_trim_top_fraction,
                            mid_fraction=project.config.synthesis_trim_mid_fraction,
                            top_keep_fraction=project.config.synthesis_trim_top_keep_fraction,
                            mid_keep_fraction=project.config.synthesis_trim_mid_keep_fraction,
                            tail_keep_fraction=project.config.synthesis_trim_tail_keep_fraction,
                        )
                    )
                    for key, value in passage_tier_counts.items():
                        synthesis_trim_tiers[key] = synthesis_trim_tiers.get(key, 0) + int(value)
                    _trim_candidate_texts_by_bm25(
                        axis,
                        prompt_passages,
                        keep_fraction=0.25,
                        keep_fraction_by_passage_id=synthesis_keep_fraction_by_passage_id,
                    )
                trimmed_text_by_id = {item["passage_id"]: item["text"] for item in prompt_passages}
                book_groups: dict[str, list[dict[str, Any]]] = {}
                for passage in passages:
                    book_groups.setdefault(passage.book_id, []).append(
                        {
                            "passage_id": passage.passage_id,
                            "text": trimmed_text_by_id.get(
                                passage.passage_id,
                                passage.full_text.strip() or passage.text,
                            ),
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
                {"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books
            ]

            primitives_payload = self.synthesis_primitives_agent.build_payload(
                project_id=project.project_id,
                podcast_mode=project.config.podcast_mode.value,
                axes_summary=axes_summary,
                passages_by_axis=passages_summary,
                cross_book_pairs=cross_pairs,
                book_metadata=book_metadata,
                primitive_target_ranges=primitive_substrate_target_ranges_for_mode(
                    project.config.podcast_mode
                ),
                actor_metadata=compact_actor_registry(actor_metadata),
            )
            primitives = await asyncio.to_thread(
                self.synthesis_primitives_agent.run, primitives_payload
            )
            primitives, primitive_actor_metrics = clean_synthesis_primitive_actor_links(
                primitives,
                actor_metadata,
            )
            extraction_counts = _primitive_counts_by_substrate(primitives)
            _save_json(project_dir / "substrate_primitives.json", primitives)
            tagged_primitives, tagging_metrics = await self._tag_substrate_primitives(
                project=project,
                corpus=corpus,
                project_dir=project_dir,
                primitives=primitives,
                actor_metadata=actor_metadata,
            )

            ctx["output_summary"] = {
                "selected_axes": len(selected_axes),
                "selected_passages": sum(
                    len(items) for items in synthesis_passages_by_axis.values()
                ),
                "synthesis_cap": synthesis_total_cap,
                "synthesis_trim_tiers": synthesis_trim_tiers,
                "synthesis_trim_keep_fractions": {
                    "top_fraction": project.config.synthesis_trim_top_fraction,
                    "mid_fraction": project.config.synthesis_trim_mid_fraction,
                    "tail_fraction": max(
                        0.0,
                        1.0
                        - project.config.synthesis_trim_top_fraction
                        - project.config.synthesis_trim_mid_fraction,
                    ),
                    "top_keep_fraction": project.config.synthesis_trim_top_keep_fraction,
                    "mid_keep_fraction": project.config.synthesis_trim_mid_keep_fraction,
                    "tail_keep_fraction": project.config.synthesis_trim_tail_keep_fraction,
                },
                "cap_report": cap_report,
                "primitive_counts_by_substrate": extraction_counts,
                "tagged_counts_by_substrate": tagging_metrics["tagged_counts_by_substrate"],
                "primitive_count": len(primitives.primitives),
                "tagged_primitive_count": len(tagged_primitives.primitives),
                "quality_score": tagged_primitives.quality_score,
            }
            return tagged_primitives, {
                "primitives": primitive_actor_metrics,
            }

    async def _tag_substrate_primitives(
        self,
        *,
        project: ThematicProject,
        corpus: ThematicCorpus,
        project_dir: Path,
        primitives: SynthesisPrimitivesArtifact,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[PrimitiveFunctionTaggingArtifact, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        passage_lookup = _build_passage_lookup(corpus)
        tagging_sem = asyncio.Semaphore(_PRIMITIVE_ANNOTATION_BATCH_CONCURRENCY)
        overlay_by_primitive_id: dict[str, PrimitiveEnrichmentOverlay] = {}

        async def _tag_substrate_batch(
            substrate: str,
            substrate_primitives: list[BaseSynthesisPrimitive],
        ) -> PrimitiveFunctionTaggingOverlayArtifact:
            async with tagging_sem:
                batch_actor_ids = collect_actor_ids_for_primitives(substrate_primitives)
                batch_actor_metadata = select_actor_metadata_subset(
                    actor_metadata,
                    batch_actor_ids,
                )
                stage_name = f"primitive_function_tagging_{substrate}"
                async with _stage_log(
                    self.run_logger,
                    stage_name,
                    project_dir,
                    substrate=substrate,
                    primitive_count=len(substrate_primitives),
                ) as batch_ctx:
                    payload = self.primitive_function_tagging_agents[substrate].build_payload(
                        project_id=project.project_id,
                        podcast_mode=project.config.podcast_mode.value,
                        base_primitives=[
                            primitive.model_dump(mode="json") for primitive in substrate_primitives
                        ],
                        passage_list=_build_function_tagging_passage_list(
                            primitives=substrate_primitives,
                            passage_lookup=passage_lookup,
                            actor_metadata=batch_actor_metadata,
                            trim_profile=_FUNCTION_TAGGING_PRIMITIVE_EVIDENCE_TRIM_PROFILE,
                        ),
                        actor_metadata=_build_primitive_actor_metadata_payload(
                            batch_actor_metadata
                        ),
                    )
                    tagged_batch = await asyncio.to_thread(
                        self.primitive_function_tagging_agents[substrate].run,
                        payload,
                    )
                    batch_ctx["output_summary"] = {
                        "substrate": substrate,
                        "primitive_count": len(tagged_batch.overlays_by_id),
                    }
                return tagged_batch

        tagging_tasks = []
        for substrate, substrate_primitives in primitives.primitives_by_substrate().items():
            if not substrate_primitives:
                continue
            tagging_tasks.append(
                _tag_substrate_batch(
                    substrate,
                    substrate_primitives,
                )
            )
        for tagged_batch in await asyncio.gather(*tagging_tasks):
            overlay_by_primitive_id.update(tagged_batch.overlays_by_id)

        tagged_primitives = PrimitiveFunctionTaggingArtifact(
            project_id=project.project_id,
            primitives=[
                apply_primitive_enrichment_overlay(
                    primitive,
                    overlay_by_primitive_id.get(
                        primitive.id,
                        PrimitiveEnrichmentOverlay(),
                    ),
                )
                for primitive in primitives.primitives
            ],
            quality_score=primitives.quality_score,
            quality_notes=list(primitives.quality_notes),
        )
        _save_json(project_dir / "tagged_primitives.json", tagged_primitives)
        return tagged_primitives, {
            "primitive_counts_by_substrate": _primitive_counts_by_substrate(primitives),
            "tagged_counts_by_substrate": _primitive_counts_by_substrate(tagged_primitives),
        }

    async def _extract_excerpts(
        self,
        project: ThematicProject,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> ExcerptArtifact:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "excerpt_extraction",
            project_dir,
            axis_count=len(corpus.axes),
            total_passages=corpus.total_passages,
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

            # Axis-budgeted, quotability-ranked passage selection.
            # quotability_score now measures excerpt-presence (does this passage carry
            # a discrete utterance/document we could surface), so we pick the top-K per
            # axis using budgets proportional to theme_importance_score, with a floor
            # and ceiling to keep coverage broad. Unused budget redistributes to the
            # axes with deeper excerpt-bearing pools.
            axis_pools, axis_budgets, claimed_by_axis, axis_order = (
                _allocate_excerpt_extraction_passages(
                    selected_axes=selected_axes,
                    passages_by_axis=corpus.passages_by_axis,
                    passage_cap=project.config.excerpt_extraction_total_passage_cap,
                    axis_floor=project.config.excerpt_extraction_axis_passage_floor,
                    ceiling_fraction=project.config.excerpt_extraction_axis_passage_ceiling_fraction,
                )
            )
            selected_passages = [
                passage for axis_id in axis_order for passage in claimed_by_axis[axis_id]
            ]

            passages_by_axis: dict[str, list[dict[str, Any]]] = {}
            for passage in selected_passages:
                book_groups = passages_by_axis.setdefault(passage.axis_id, [])
                book_entry = next(
                    (entry for entry in book_groups if entry["book_id"] == passage.book_id),
                    None,
                )
                if book_entry is None:
                    book_entry = {"book_id": passage.book_id, "passages": []}
                    book_groups.append(book_entry)
                book_entry["passages"].append(
                    {
                        "passage_id": passage.passage_id,
                        "text": passage.full_text.strip() or passage.text,
                    }
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
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books
            ]

            payload = self.excerpt_extraction_agent.build_payload(
                project_id=project.project_id,
                podcast_mode=project.config.podcast_mode.value,
                axes_summary=axes_summary,
                passages_by_axis=passages_by_axis,
                book_metadata=book_metadata,
                excerpt_count_min=project.config.excerpt_extraction_count_min,
                excerpt_count_max=project.config.excerpt_extraction_count_max,
                actor_metadata=compact_actor_registry(actor_metadata),
            )
            excerpts = await asyncio.to_thread(self.excerpt_extraction_agent.run, payload)
            excerpts, excerpt_actor_metrics = clean_excerpt_actor_links(excerpts, actor_metadata)

            # Verify each excerpt's verbatim text against its source passages.
            # Drops verbatim to empty when fuzzy-match falls below threshold —
            # a misquoted line under quotation marks is worse than no quote.
            passage_text_by_id: dict[str, str] = {
                passage.passage_id: (passage.full_text.strip() or passage.text)
                for passage in selected_passages
            }
            verbatim_with_text_before = sum(
                1 for e in excerpts.excerpts if e.verbatim_excerpt.strip()
            )
            excerpts = ExcerptArtifact(
                project_id=excerpts.project_id,
                excerpts=[verify_excerpt(e, passage_text_by_id) for e in excerpts.excerpts],
                quality_score=excerpts.quality_score,
                quality_notes=list(excerpts.quality_notes),
            )
            verbatim_with_text_after = sum(
                1 for e in excerpts.excerpts if e.verbatim_excerpt.strip()
            )
            excerpts_verbatim_dropped = verbatim_with_text_before - verbatim_with_text_after
            ratios = [e.verbatim_match_ratio for e in excerpts.excerpts]
            mean_verbatim_match_ratio = round(sum(ratios) / len(ratios), 4) if ratios else 0.0

            _save_json(project_dir / "excerpts.json", excerpts)

            type_counts: dict[str, int] = {}
            for excerpt in excerpts.excerpts:
                type_counts[excerpt.excerpt_type] = type_counts.get(excerpt.excerpt_type, 0) + 1
            ctx["output_summary"] = {
                "selected_axes": len(selected_axes),
                "candidate_passages": sum(len(pool) for pool in axis_pools.values()),
                "selected_passages": len(selected_passages),
                "axis_budgets": dict(axis_budgets),
                "selected_per_axis": {
                    axis_id: len(claimed_by_axis[axis_id]) for axis_id in axis_order
                },
                "excerpt_count": len(excerpts.excerpts),
                "excerpt_counts_by_type": type_counts,
                "excerpts_with_verbatim": verbatim_with_text_after,
                "excerpts_verbatim_dropped": excerpts_verbatim_dropped,
                "mean_verbatim_match_ratio": mean_verbatim_match_ratio,
                "unknown_actor_ids": excerpt_actor_metrics.get("unknown_actor_ids", 0),
                "quality_score": excerpts.quality_score,
            }
            return excerpts

    async def _discover_scenes(
        self,
        project: ThematicProject,
        synthesis_map: PrimitiveFunctionTaggingArtifact,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> SceneDiscoveryArtifact:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        passage_lookup = _build_passage_lookup(corpus)
        passage_list = _build_scene_discovery_passage_list_payload(
            project=project,
            synthesis_map=synthesis_map,
            passage_lookup=passage_lookup,
        )
        async with _stage_log(
            self.run_logger,
            "scene_discovery",
            project_dir,
            primitive_count=len(_flatten_base_synthesis_primitives(synthesis_map)),
            passage_count=len(passage_list),
            actor_count=len(actor_metadata.actors),
        ) as ctx:
            payload = self.scene_discovery_agent.build_payload(
                synthesis_map=_build_scene_discovery_synthesis_map_payload(synthesis_map),
                project_metadata=_build_scene_discovery_project_metadata_payload(project),
                actor_metadata=_build_narrative_strategy_actor_metadata_payload(actor_metadata),
                passage_list=passage_list,
            )
            artifact = await asyncio.to_thread(self.scene_discovery_agent.run, payload)
            diagnostics = _build_scene_discovery_diagnostics(
                artifact=artifact,
                mode=PodcastMode(project.config.podcast_mode),
            )
            _save_json(project_dir / "scene_discovery.json", artifact)
            _save_json(project_dir / "scene_discovery_diagnostics.json", diagnostics)
            ctx["output_summary"] = {
                "candidate_count": len(artifact.candidates),
                "warning_count": diagnostics["warning_count"],
            }
            return artifact

    async def _choose_narrative_strategy(
        self,
        project: ThematicProject,
        synthesis_map: PrimitiveFunctionTaggingArtifact,
        project_dir: Path,
        scene_discovery: SceneDiscoveryArtifact | None = None,
        actor_metadata: ActorMetadata | None = None,
        excerpts: ExcerptArtifact | None = None,
    ) -> tuple[NarrativeStrategy, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "narrative_strategy",
            project_dir,
            primitive_count=len(_flatten_base_synthesis_primitives(synthesis_map)),
        ) as ctx:
            skeleton_payload = _compact_narrative_strategy_runtime_payload(
                self.narrative_strategy_skeleton_agent.build_payload(
                    synthesis_map=_build_narrative_strategy_synthesis_map_payload(synthesis_map),
                    project_metadata=_build_narrative_strategy_project_metadata_payload(project),
                    scene_discovery=_build_narrative_strategy_scene_discovery_payload(
                        scene_discovery
                    ),
                    episode_count=project.requested_episode_count,
                    recommended_episode_count_min=(
                        project.config.narrative_strategy_episode_count_min
                    ),
                    recommended_episode_count_max=(
                        project.config.narrative_strategy_episode_count_max
                    ),
                    excerpts=_build_strategy_excerpt_payload(excerpts),
                )
            )
            strategy_skeleton = await asyncio.to_thread(
                self.narrative_strategy_skeleton_agent.run, skeleton_payload
            )
            _save_json(project_dir / "narrative_strategy_skeleton.json", strategy_skeleton)

            selected_preview = _build_strategy_selected_synthesis_map_preview(
                project_id=project.project_id,
                synthesis_map=synthesis_map,
                strategy=strategy_skeleton,
            )
            try:
                mode = PodcastMode(project.config.podcast_mode.value)
            except ValueError:
                mode = PodcastMode.FULL
            enrichment_payload = _compact_narrative_strategy_runtime_payload(
                self.narrative_strategy_enrichment_agent.build_payload(
                    strategy_skeleton=strategy_skeleton.model_dump(mode="json"),
                    synthesis_map=_build_narrative_strategy_synthesis_map_payload(selected_preview),
                    project_metadata=_build_narrative_strategy_project_metadata_payload(project),
                    episode_scene_candidates=_build_episode_scene_candidate_payloads(
                        strategy=strategy_skeleton,
                        scene_discovery=scene_discovery,
                        mode=mode,
                    ),
                    actor_metadata=_build_narrative_strategy_actor_metadata_payload(actor_metadata),
                    human_thread_candidates=_build_human_thread_candidate_index(
                        synthesis_map,
                        actor_metadata,
                        top_n=_human_thread_candidate_cap_for_mode(project.config.podcast_mode),
                    ),
                )
            )
            strategy_enrichment = await asyncio.to_thread(
                self.narrative_strategy_enrichment_agent.run, enrichment_payload
            )
            _save_json(
                project_dir / "narrative_strategy_enrichment.json",
                strategy_enrichment,
            )
            strategy = _merge_narrative_strategy_parts(
                skeleton=strategy_skeleton,
                enrichment=strategy_enrichment,
            )
            strategy, strategy_actor_metrics = clean_narrative_strategy_actor_links(
                strategy,
                actor_metadata,
            )
            _save_json(project_dir / "narrative_strategy.json", strategy)
            strategy_warnings = _build_narrative_strategy_warnings(
                strategy=strategy,
                requested_episode_count=project.requested_episode_count,
            )
            strategy_actor_arc_diagnostics = _build_narrative_strategy_actor_arc_diagnostics(
                strategy=strategy,
                synthesis_map=synthesis_map,
                scene_discovery=scene_discovery,
                actor_metadata=actor_metadata,
            )
            _save_json(
                project_dir / "narrative_strategy_diagnostics.json",
                strategy_actor_arc_diagnostics,
            )
            combined_strategy_warnings = [
                *strategy_warnings,
                *strategy_actor_arc_diagnostics["warnings"],
            ]
            for warning in combined_strategy_warnings:
                logger.warning("narrative_strategy_warning %s", warning)
                self.run_logger.log("narrative_strategy_warning", warning=warning)

            ctx["output_summary"] = {
                "strategy": strategy.strategy_type,
                "recommended_episode_count": strategy.recommended_episode_count,
                "episodes": len(strategy.episodes),
                "warning_count": len(combined_strategy_warnings),
            }
            return strategy, strategy_actor_metrics

    async def _materialize_selected_primitives(
        self,
        project: ThematicProject,
        synthesis_primitives: PrimitiveFunctionTaggingArtifact,
        strategy: NarrativeStrategy,
        project_dir: Path,
    ) -> SynthesisMap:
        selected_primitive_ids = _collect_strategy_selected_primitive_ids(strategy)
        base_primitive_by_id = _flatten_base_synthesis_primitives(synthesis_primitives)
        selected_base_primitives: list[BaseSynthesisPrimitive] = []
        missing_primitive_ids: list[str] = []
        for primitive_id in selected_primitive_ids:
            primitive = base_primitive_by_id.get(primitive_id)
            if primitive is None:
                missing_primitive_ids.append(primitive_id)
                continue
            selected_base_primitives.append(primitive)
        if missing_primitive_ids:
            raise RuntimeError(
                f"narrative_strategy selected unknown primitive ids: {missing_primitive_ids[:10]}"
            )

        async with _stage_log(
            self.run_logger,
            "selected_primitives",
            project_dir,
            selected_primitive_count=len(selected_base_primitives),
        ) as ctx:
            retained_primitives: list[SynthesisPrimitiveBase] = list(selected_base_primitives)
            gloss_warnings = _build_narration_hook_gloss_warnings(retained_primitives)
            for warning in gloss_warnings:
                logger.warning("%s", warning)

            # `gloss_warnings` are operator diagnostics only — never fold them into
            # `quality_notes`, which is serialized into downstream model prompts
            # (architecture, narrative_strategy). They stay in the logs and in the
            # stage output_summary below.
            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives=retained_primitives,
                quality_score=synthesis_primitives.quality_score,
                quality_notes=list(synthesis_primitives.quality_notes),
            )
            _save_json(project_dir / "retained_primitives.json", synthesis_map)
            ctx["output_summary"] = {
                "selected_primitive_count": len(selected_base_primitives),
                "selected_substrate_count": sum(
                    1
                    for substrate in PRIMITIVE_SUBSTRATES
                    if any(
                        primitive.substrate.value == substrate
                        for primitive in selected_base_primitives
                    )
                ),
                "retained_primitive_count": len(synthesis_map.primitives),
                "warning_count": len(gloss_warnings),
            }
            return synthesis_map

    async def _materialize_selected_excerpts(
        self,
        project: ThematicProject,
        excerpts: ExcerptArtifact | None,
        strategy: NarrativeStrategy,
        project_dir: Path,
    ) -> ExcerptArtifact:
        excerpts = excerpts or ExcerptArtifact(project_id=project.project_id)
        excerpt_by_id = excerpts.excerpt_by_id()

        # Cross-episode recall is strict backward-only: each episode N's
        # `recall_excerpt_ids[i]` must already appear in some prior episode M's
        # primary `excerpt_ids` (M < N). The spine schema cannot enforce this
        # because it cannot see other episodes; we enforce here at retention
        # time so the violation is loud, not silent.
        ordered_episodes = sorted(strategy.episodes, key=lambda ep: ep.episode_number)
        introduced_excerpt_ids: set[str] = set()
        for episode in ordered_episodes:
            spine = episode.episode_spine
            unintroduced = [
                excerpt_id
                for excerpt_id in spine.recall_excerpt_ids
                if excerpt_id not in introduced_excerpt_ids
            ]
            if unintroduced:
                raise RuntimeError(
                    "narrative_strategy episode "
                    f"{episode.episode_number} recall_excerpt_ids reference "
                    "excerpts not assigned to any earlier episode's "
                    f"excerpt_ids: {unintroduced[:10]}"
                )
            for excerpt_id in spine.excerpt_ids:
                introduced_excerpt_ids.add(excerpt_id)

        selected_excerpt_ids = _collect_strategy_selected_excerpt_ids(strategy)
        retained: list[ExcerptRecord] = []
        missing_excerpt_ids: list[str] = []
        for excerpt_id in selected_excerpt_ids:
            excerpt = excerpt_by_id.get(excerpt_id)
            if excerpt is None:
                missing_excerpt_ids.append(excerpt_id)
                continue
            retained.append(excerpt)
        if missing_excerpt_ids:
            raise RuntimeError(
                f"narrative_strategy selected unknown excerpt ids: {missing_excerpt_ids[:10]}"
            )

        async with _stage_log(
            self.run_logger,
            "selected_excerpts",
            project_dir,
            selected_excerpt_count=len(retained),
        ) as ctx:
            retained_excerpts = ExcerptArtifact(
                project_id=project.project_id,
                excerpts=retained,
                quality_score=excerpts.quality_score,
                quality_notes=list(excerpts.quality_notes),
            )
            warnings: list[str] = []
            extracted_count = len(excerpts.excerpts)
            episode_count = max(1, len(strategy.episodes))
            floor = episode_count  # expect roughly >= 1 voiced excerpt per episode
            if extracted_count > 0 and len(retained) == 0:
                warnings.append(
                    "retained_excerpts_below_floor: "
                    f"{extracted_count} excerpts extracted but 0 retained by narrative strategy"
                )
            elif extracted_count > 0 and len(retained) < floor:
                warnings.append(
                    "retained_excerpts_below_floor: "
                    f"only {len(retained)} excerpts retained for {episode_count} episodes "
                    f"(extracted {extracted_count})"
                )
            for episode in strategy.episodes:
                spine = episode.episode_spine
                if not spine.excerpt_ids:
                    warnings.append(
                        "episode_without_excerpt: "
                        f"episode {episode.episode_number} carries no excerpt_ids"
                    )
                    continue
                # Per-episode register diversity: ≥ 3 distinct speakers and
                # ≥ 2 distinct excerpt_types. Same speaker in the same type
                # collapses to one voice slot.
                episode_excerpts = [
                    excerpt_by_id[xid] for xid in spine.excerpt_ids if xid in excerpt_by_id
                ]
                speakers = {e.speaker.strip() for e in episode_excerpts if e.speaker.strip()}
                types = {e.excerpt_type for e in episode_excerpts}
                if len(speakers) < 3 or len(types) < 2:
                    warnings.append(
                        "episode_excerpt_register_thin: "
                        f"episode {episode.episode_number} carries "
                        f"{len(speakers)} distinct speakers and {len(types)} "
                        "distinct excerpt_types (target: >=3 speakers, "
                        ">=2 types)"
                    )
                # Voice_first slots need voiceable lines: at least one excerpt
                # with a non-empty verbatim_excerpt. Pre-warning of the
                # architecture-stage hard error.
                if not any(e.verbatim_excerpt.strip() for e in episode_excerpts):
                    warnings.append(
                        "episode_excerpt_voicefirst_unvoiceable: "
                        f"episode {episode.episode_number} has no excerpt with "
                        "a non-empty verbatim_excerpt; voice_first sections "
                        "cannot be honored"
                    )
            for warning in warnings:
                logger.warning("%s", warning)
                self.run_logger.log("selected_excerpts_warning", warning=warning)

            _save_json(project_dir / "retained_excerpts.json", retained_excerpts)
            type_counts: dict[str, int] = {}
            for excerpt in retained:
                type_counts[excerpt.excerpt_type] = type_counts.get(excerpt.excerpt_type, 0) + 1
            ctx["output_summary"] = {
                "extracted_excerpt_count": extracted_count,
                "retained_excerpt_count": len(retained),
                "retained_excerpt_counts_by_type": type_counts,
                "warning_count": len(warnings),
            }
            return retained_excerpts

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

    async def _build_single_episode_architecture(
        self,
        *,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        strategy_episode: StrategyEpisode,
        corpus: ThematicCorpus,
        actor_metadata: ActorMetadata,
        scene_discovery: SceneDiscoveryArtifact | None,
        narrative_state_pre: NarrativeState,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
    ) -> tuple[EpisodeArchitecture, dict[str, Any], list[dict[str, Any]]]:
        excerpt_by_id = excerpt_by_id or {}
        primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        passage_lookup = _build_passage_lookup(corpus)
        episode_excerpts_payload = _build_episode_excerpt_payload(
            strategy_episode.episode_spine.assigned_excerpt_ids,
            excerpt_by_id,
        )
        primitive_ids_by_role = {
            "core": list(strategy_episode.episode_spine.core_primitive_ids),
            "support": list(strategy_episode.episode_spine.support_primitive_roles.keys()),
            "recall": list(strategy_episode.episode_spine.recall_primitive_ids),
        }
        episode_synthesis_map_payload, primitive_ids = _build_episode_synthesis_map_payload(
            synthesis_map,
            primitive_ids_by_role,
        )
        episode_actor_ids = _collect_episode_actor_ids(
            strategy_episode=strategy_episode,
            primitive_ids=primitive_ids,
            primitive_lookup=primitive_lookup,
        )
        episode_actor_metadata = select_actor_metadata_subset(
            actor_metadata,
            episode_actor_ids,
        )
        project_metadata = {
            "podcast_mode": project.config.podcast_mode.value,
            "theme": project.theme,
            "sub_themes": project.sub_themes,
            "book_count": len(project.books),
            "books": [
                {"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books
            ],
            "architecture_section_target_min": (project.config.architecture_section_target_min),
            "architecture_section_target_max": (project.config.architecture_section_target_max),
            "min_episode_minutes": project.config.min_episode_minutes,
            "max_episode_minutes": project.config.max_episode_minutes,
            # Density budget (Change 1) — threaded into
            # EpisodeArchitectureAgent.validate_result for enforce-mode checks.
            "max_dense_sections_per_episode": (project.config.max_dense_sections_per_episode),
            "dense_section_runtime_min_minutes": (project.config.dense_section_runtime_min_minutes),
            "dense_section_runtime_max_minutes": (project.config.dense_section_runtime_max_minutes),
            "section_runtime_floor_minutes": (project.config.section_runtime_floor_minutes),
            "section_runtime_ceiling_minutes": (project.config.section_runtime_ceiling_minutes),
            "series_runtime_policy": project.config.series_runtime_policy,
        }
        core_passages = _build_episode_architecture_core_passages(
            driving_question=strategy_episode.episode_spine.listener_problem,
            thematic_focus=strategy_episode.thematic_focus,
            episode_spine=strategy_episode.episode_spine,
            primitive_lookup=primitive_lookup,
            passage_lookup=passage_lookup,
        )
        support_passages = _build_episode_architecture_support_passages(
            driving_question=strategy_episode.episode_spine.listener_problem,
            thematic_focus=strategy_episode.thematic_focus,
            episode_spine=strategy_episode.episode_spine,
            primitive_lookup=primitive_lookup,
            passage_lookup=passage_lookup,
        )
        episode_scene_payload = _build_episode_scene_payload(
            strategy_episode=strategy_episode,
            scene_discovery=scene_discovery,
        )
        architecture_feedback: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        max_attempts = self.episode_architecture_agent.max_retry_attempts
        architecture: EpisodeArchitecture | None = None
        for attempt in range(1, max_attempts + 1):
            payload = self.episode_architecture_agent.build_payload(
                episode=strategy_episode.model_dump(mode="json"),
                synthesis_map=episode_synthesis_map_payload,
                project_metadata=project_metadata,
                core_passages=core_passages,
                support_passages=support_passages,
                episode_scenes=episode_scene_payload,
                series_explanation_registry=[
                    item.model_dump(mode="json") for item in strategy.series_explanation_registry
                ],
                series_actor_explanation_registry=[
                    item.model_dump(mode="json")
                    for item in strategy.series_actor_explanation_registry
                ],
                narrator_profile=strategy.narrator_profile.model_dump(mode="json"),
                narrative_state=narrative_state_pre.model_dump(mode="json"),
                actor_metadata=compact_actor_metadata(episode_actor_metadata),
                architecture_feedback=architecture_feedback,
                excerpts=episode_excerpts_payload,
            )
            try:
                architecture = await asyncio.to_thread(self.episode_architecture_agent.run, payload)
                architecture = _validate_architecture_transition(
                    strategy_episode=strategy_episode,
                    architecture=architecture,
                    excerpt_by_id=excerpt_by_id,
                )
            except ComplianceViolationError as exc:
                architecture_feedback = _build_architecture_retry_feedback(exc)
                attempts.append(
                    _build_attempt_record(
                        attempt=attempt,
                        status="rejected_contract",
                        blocking_issue=architecture_feedback["issue"],
                        retry_feedback=architecture_feedback,
                    )
                )
                self.run_logger.log(
                    "episode_architecture_attempt_warning",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    status="rejected_contract",
                    blocking_issue=architecture_feedback["issue"],
                    warning_count=0,
                    warnings=[],
                )
                if attempt >= max_attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                self.run_logger.log(
                    "episode_architecture_retry_scheduled",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    next_attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    issue=architecture_feedback["issue"],
                )
                await asyncio.sleep(backoff)
                continue
            except (ValidationError, ValueError) as exc:
                architecture_feedback = _build_architecture_retry_feedback(exc)
                attempts.append(
                    _build_attempt_record(
                        attempt=attempt,
                        status="rejected_validation",
                        blocking_issue=architecture_feedback["issue"],
                        retry_feedback=architecture_feedback,
                    )
                )
                self.run_logger.log(
                    "episode_architecture_attempt_warning",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    status="rejected_validation",
                    blocking_issue=architecture_feedback["issue"],
                    warning_count=0,
                    warnings=[],
                )
                if attempt >= max_attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                self.run_logger.log(
                    "episode_architecture_retry_scheduled",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    next_attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    issue=architecture_feedback["issue"],
                )
                await asyncio.sleep(backoff)
                continue
            except Exception as exc:
                architecture_feedback = _build_architecture_retry_feedback(exc)
                attempts.append(
                    _build_attempt_record(
                        attempt=attempt,
                        status="rejected_contract",
                        blocking_issue=architecture_feedback["issue"],
                        retry_feedback=architecture_feedback,
                    )
                )
                self.run_logger.log(
                    "episode_architecture_attempt_warning",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    status="rejected_contract",
                    blocking_issue=architecture_feedback["issue"],
                    warning_count=0,
                    warnings=[],
                )
                if attempt >= max_attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                self.run_logger.log(
                    "episode_architecture_retry_scheduled",
                    episode=strategy_episode.episode_number,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    next_attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    issue=architecture_feedback["issue"],
                )
                await asyncio.sleep(backoff)
                continue
            attempts.append(
                _build_attempt_record(
                    attempt=attempt,
                    status="accepted",
                )
            )
            break
        if architecture is None:
            raise RuntimeError(
                f"Episode architecture did not produce an architecture for episode {strategy_episode.episode_number}."
            )
        report = {
            "episode_number": strategy_episode.episode_number,
            "section_count": len(architecture.sections),
            "major_turn_section_id": architecture.major_turn_section_id,
            "core_primitive_count": len(strategy_episode.episode_spine.core_primitive_ids),
            "episode_scene_candidate_count": len(episode_scene_payload),
            "actor_directive_count": len(strategy_episode.actor_arc_directives),
        }
        return architecture, report, attempts

    async def _plan_single_episode(
        self,
        *,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        corpus: ThematicCorpus,
        actor_metadata: ActorMetadata,
        narrative_state_pre: NarrativeState,
    ) -> tuple[EpisodePlan, dict[str, Any], list[dict[str, Any]]]:
        passage_lookup = _build_passage_lookup(corpus)
        primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        planning_runtime = _build_episode_planning_runtime(
            project=project,
            strategy=strategy,
            strategy_episode=strategy_episode,
            architecture=architecture,
            synthesis_map=synthesis_map,
            corpus=corpus,
            actor_metadata=actor_metadata,
            narrative_state_pre=narrative_state_pre,
            primitive_lookup=primitive_lookup,
            passage_lookup=passage_lookup,
        )
        plan_draft: EpisodePlanDraft | None = None
        actor_link_metrics: dict[str, Any] = {}
        actor_explanation_warnings: list[str] = []
        planning_feedback: dict[str, Any] | None = None
        planning_attempts: list[dict[str, Any]] = []
        max_attempts = self.episode_planning_agent.max_retry_attempts
        # Change 1: derive per-section scene targets from the architecture's
        # runtime distribution. The planner gets this on top of the
        # mode-locked scene_job_budget so dense sections can be allocated
        # more scenes than light ones.
        from podcast_agent.schemas.models import derive_section_scene_targets

        mode_scene_job_budget = scene_job_budget_for_mode(project.config.podcast_mode)
        section_scene_targets = derive_section_scene_targets(
            architecture,
            mode=project.config.podcast_mode,
            scene_job_budget=mode_scene_job_budget,
        )
        for attempt in range(1, max_attempts + 1):
            payload = self.episode_planning_agent.build_payload(
                strategy_episode=strategy_episode.model_dump(mode="json"),
                architecture=planning_runtime.episode_payload,
                synthesis_map=planning_runtime.episode_synthesis_map_payload,
                project_metadata=planning_runtime.project_metadata,
                scene_job_budget=mode_scene_job_budget,
                available_passages=planning_runtime.available_passages,
                host_policy=planning_runtime.host_policy,
                narrative_state_pre=narrative_state_pre.model_dump(mode="json"),
                continuity_contract_pre=planning_runtime.continuity_contract_pre,
                actor_metadata=planning_runtime.compact_episode_actor_metadata,
                planning_feedback=planning_feedback,
                field_semantics=_build_field_semantics_payload(),
                section_scene_targets=section_scene_targets,
            )
            try:
                plan_draft = await asyncio.to_thread(self.episode_planning_agent.run, payload)
                (
                    plan_draft,
                    actor_link_metrics,
                    actor_explanation_warnings,
                ) = _postcheck_episode_plan_draft(
                    strategy_episode=strategy_episode,
                    architecture=architecture,
                    plan_draft=plan_draft,
                    episode_actor_metadata=planning_runtime.episode_actor_metadata,
                    narrator_profile=strategy.narrator_profile,
                    continuity_contract_pre=planning_runtime.continuity_contract_pre,
                )
                planning_attempts.append(_build_attempt_record(attempt=attempt, status="accepted"))
                break
            except ComplianceViolationError as exc:
                attempt_feedback = _build_plan_transition_feedback(exc)
                planning_attempts.append(
                    _build_attempt_record(
                        attempt=attempt,
                        status="rejected_contract",
                        blocking_issue=attempt_feedback["issue"],
                        retry_feedback=attempt_feedback,
                    )
                )
                self.run_logger.log(
                    "episode_planning_attempt_warning",
                    episode=architecture.episode_number,
                    attempt=attempt,
                    status="rejected_contract",
                    blocking_issue=attempt_feedback["issue"],
                    warning_count=0,
                    warnings=[],
                )
                if attempt >= max_attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                planning_feedback = attempt_feedback
                self.run_logger.log(
                    "episode_planning_retry_scheduled",
                    episode=architecture.episode_number,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    next_attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    issue=planning_feedback["issue"],
                )
                await asyncio.sleep(backoff)
        if plan_draft is None:
            raise RuntimeError(
                f"Episode planning did not produce a plan draft for episode {architecture.episode_number}."
            )
        spine_diagnostics = _build_spine_plan_diagnostics(
            strategy_episode=strategy_episode,
            plan=plan_draft,
            scene_job_budget=scene_job_budget_for_mode(project.config.podcast_mode),
        )
        host_moves_diagnostics, host_move_warnings = _build_host_move_plan_diagnostics(
            scene_cards=plan_draft.scene_cards,
            architecture=architecture,
            narrator_profile=strategy.narrator_profile,
        )
        host_density_diagnostics, host_density_warnings = _build_host_density_diagnostics(
            scene_cards=plan_draft.scene_cards,
            architecture=architecture,
        )
        scene_card_count_warnings = _build_scene_card_count_warnings(
            scene_card_count=len(plan_draft.scene_cards),
            scene_card_target_min=project.config.scene_card_target_min,
            scene_card_target_max=project.config.scene_card_target_max,
        )
        scene_card_primitive_warnings = _build_scene_card_primitive_warnings(
            scene_cards=plan_draft.scene_cards,
            primitive_pool_ids=set(planning_runtime.primitive_ids),
            primitive_by_id=primitive_lookup,
            primitive_min=project.config.scene_card_primitives_min,
            primitive_max=project.config.scene_card_primitives_max,
        )
        scene_card_family_warnings = _build_scene_card_family_warnings(
            strategy_episode=strategy_episode,
            primitive_pool_ids=set(planning_runtime.primitive_ids),
            primitive_by_id=primitive_lookup,
        )
        section_realization_reports, section_planning_warnings = _build_section_plan_realization(
            episode=architecture,
            scene_cards=plan_draft.scene_cards,
            words_per_minute=float(self.settings.pipeline.spoken_words_per_minute),
        )
        state_alignment_diagnostics, state_alignment_warnings = _build_state_alignment_diagnostics(
            architecture=architecture,
            scene_cards=plan_draft.scene_cards,
        )
        structural_card_concreteness_warnings = _build_structural_card_concreteness_warnings(
            scene_cards=plan_draft.scene_cards,
        )
        human_grounding_diagnostics, human_grounding_warnings = _build_human_grounding_warnings(
            scene_cards=plan_draft.scene_cards,
        )
        comparative_aside_warnings = _build_comparative_aside_scene_warnings(
            architecture=architecture,
            scene_cards=plan_draft.scene_cards,
        )
        thread_plan_warnings = _build_thread_plan_warnings(
            strategy_episode=strategy_episode,
            architecture=architecture,
            scene_cards=plan_draft.scene_cards,
        )
        host_beat_placement_warnings = _build_host_beat_placement_warnings(
            architecture=architecture,
            scene_cards=plan_draft.scene_cards,
        )
        scene_job_counts = _build_scene_job_counts(plan_draft.scene_cards)
        scene_role_counts: dict[str, int] = {}
        for scene in plan_draft.scene_cards:
            scene_role_counts[scene.scene_role] = scene_role_counts.get(scene.scene_role, 0) + 1
        section_load_warnings = [
            warning
            for warning in section_planning_warnings
            if warning.startswith("section_scene_card_load_high")
            or warning.startswith("section_projected_word_count_high")
        ]
        planning_warnings = (
            scene_card_count_warnings
            + scene_card_primitive_warnings
            + scene_card_family_warnings
            + structural_card_concreteness_warnings
            + human_grounding_warnings
            + comparative_aside_warnings
            + actor_explanation_warnings
            + section_planning_warnings
            + state_alignment_warnings
            + list(spine_diagnostics.get("scene_job_budget_warnings", []))
            + host_move_warnings
            + host_density_warnings
            + thread_plan_warnings
            + host_beat_placement_warnings
        )
        for warning in planning_warnings:
            self.run_logger.log(
                "episode_planning_warning",
                episode=architecture.episode_number,
                warning=warning,
            )
            logger.warning(
                "episode_planning_warning episode=%s %s",
                architecture.episode_number,
                warning,
            )
        target_word_count = int(
            round(
                float(architecture.runtime_minutes)
                * float(self.settings.pipeline.spoken_words_per_minute)
            )
        )
        plan_payload = plan_draft.model_dump(mode="json")
        plan = EpisodePlan.model_validate(
            {
                **plan_payload,
                "target_word_count": target_word_count,
            }
        )
        report = {
            "episode_number": architecture.episode_number,
            "scene_card_count": len(plan.scene_cards),
            "scene_card_target_min": project.config.scene_card_target_min,
            "scene_card_target_max": project.config.scene_card_target_max,
            "scene_card_target_policy": project.config.scene_card_target_policy,
            "scene_card_primitives_min": project.config.scene_card_primitives_min,
            "scene_card_primitives_max": project.config.scene_card_primitives_max,
            "scene_card_primitive_policy": project.config.scene_card_primitive_policy,
            "accepted_warning_count": len(planning_warnings),
            "accepted_warnings": planning_warnings,
            "scene_card_count_warnings": scene_card_count_warnings,
            "scene_card_primitive_warnings": scene_card_primitive_warnings,
            "scene_card_family_warnings": scene_card_family_warnings,
            "scene_job_budget_warnings": list(
                spine_diagnostics.get("scene_job_budget_warnings", [])
            ),
            "structural_card_concreteness_warnings": structural_card_concreteness_warnings,
            "human_grounding_warnings": human_grounding_warnings,
            "section_load_warnings": section_load_warnings,
            "section_realization": section_realization_reports,
            "state_alignment": state_alignment_diagnostics,
            "host_density_diagnostics": host_density_diagnostics,
            "scene_card_warning_count": len(planning_warnings),
            "scene_role_counts": scene_role_counts,
            "scene_job_counts": scene_job_counts,
            "human_grounding_diagnostics": human_grounding_diagnostics,
            "section_count": len(architecture.sections),
            "core_primitive_count": len(strategy_episode.episode_spine.core_primitive_ids),
            "covered_core_primitive_count": len(strategy_episode.episode_spine.core_primitive_ids),
            "missing_core_primitive_ids": [],
            "spine_diagnostics": spine_diagnostics,
            "host_moves_diagnostics": host_moves_diagnostics,
            "actor_link_metrics": actor_link_metrics,
            "allocated_duration_seconds": sum(
                scene.estimated_duration_seconds for scene in plan.scene_cards
            ),
        }
        return plan, report, planning_attempts

    async def _reconcile_narrative_state(
        self,
        *,
        project: ThematicProject,
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        narrative_state_pre: NarrativeState,
    ) -> NarrativeStateReconciliation:
        payload = self.narrative_state_reconciler_agent.build_payload(
            episode_number=architecture.episode_number,
            project_id=project.project_id,
            narrative_state_pre=narrative_state_pre.model_dump(mode="json"),
            strategy_episode=strategy_episode.model_dump(mode="json"),
            architecture=architecture.model_dump(mode="json"),
        )
        reconciliation = await asyncio.to_thread(self.narrative_state_reconciler_agent.run, payload)
        return reconciliation

    async def _build_episode_architectures_with_narrative_state(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
        scene_discovery: SceneDiscoveryArtifact | None = None,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
    ) -> tuple[
        list[EpisodeArchitecture],
        dict[int, NarrativeState],
        dict[int, NarrativeState],
        list[NarrativeStateReconciliation],
        dict[str, Any],
    ]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        excerpt_by_id = excerpt_by_id or {}
        async with _stage_log(
            self.run_logger,
            "episode_architecture",
            project_dir,
            episode_count=project.episode_count,
            strategy=strategy.strategy_type,
        ) as ctx:
            strategy_episode_map = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            missing_episodes = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in strategy_episode_map
            ]
            if missing_episodes:
                raise RuntimeError(
                    "Narrative strategy did not assign episode spines for "
                    f"episodes: {missing_episodes}"
                )

            episode_numbers = list(range(1, project.episode_count + 1))

            # Pre-states are a deterministic fold of the planned cross-episode agenda.
            # This removes the serial dependency on the previous episode's reconciled
            # post-state, so every episode's architecture (and reconciliation) is
            # independent and can run in parallel.
            state_pre_by_episode: dict[int, NarrativeState] = fold_planned_pre_states(
                strategy, project.project_id
            )

            arch_sem = asyncio.Semaphore(max(1, project.config.episode_architecture_concurrency))

            async def _architect(
                episode_number: int,
            ) -> tuple[EpisodeArchitecture, dict[str, Any], list[dict[str, Any]]]:
                async with arch_sem:
                    return await self._build_single_episode_architecture(
                        project=project,
                        synthesis_map=synthesis_map,
                        strategy=strategy,
                        strategy_episode=strategy_episode_map[episode_number],
                        corpus=corpus,
                        actor_metadata=actor_metadata,
                        scene_discovery=scene_discovery,
                        narrative_state_pre=state_pre_by_episode[episode_number],
                        excerpt_by_id=excerpt_by_id,
                    )

            architecture_results = await asyncio.gather(*(_architect(n) for n in episode_numbers))
            architecture_by_episode = {
                n: result[0] for n, result in zip(episode_numbers, architecture_results)
            }

            # Change 1: series-level runtime budget validation. When the
            # configured series target is None, we derive it from
            # episode_count * per-episode min/max so a series can drift
            # outside an explicit total without unfounded slack.
            from podcast_agent.schemas.models import (
                validate_series_runtime_budget,
            )

            derived_series_min = (
                project.config.series_runtime_target_min_minutes
                if project.config.series_runtime_target_min_minutes is not None
                else project.config.min_episode_minutes * project.episode_count
            )
            derived_series_max = (
                project.config.series_runtime_target_max_minutes
                if project.config.series_runtime_target_max_minutes is not None
                else project.config.max_episode_minutes * project.episode_count
            )
            try:
                series_runtime_warnings = validate_series_runtime_budget(
                    list(architecture_by_episode.values()),
                    series_min=float(derived_series_min),
                    series_max=float(derived_series_max),
                    policy=project.config.series_runtime_policy,
                )
            except ValueError as exc:
                self.run_logger.log(
                    "series_runtime_violation",
                    message=str(exc),
                    policy=project.config.series_runtime_policy,
                )
                raise
            for warning in series_runtime_warnings:
                self.run_logger.log(
                    "series_runtime_warning",
                    message=warning,
                    policy=project.config.series_runtime_policy,
                )

            # Reconcile each episode independently from its folded pre-state + its own
            # architecture (realized state moves live only on the architecture). Each
            # call is ready as soon as its architecture completes, so this is parallel.
            async def _reconcile(
                episode_number: int,
            ) -> NarrativeStateReconciliation:
                async with arch_sem:
                    return await self._reconcile_narrative_state(
                        project=project,
                        strategy_episode=strategy_episode_map[episode_number],
                        architecture=architecture_by_episode[episode_number],
                        narrative_state_pre=state_pre_by_episode[episode_number],
                    )

            reconciliation_results = await asyncio.gather(*(_reconcile(n) for n in episode_numbers))
            reconciliation_by_episode = dict(zip(episode_numbers, reconciliation_results))

            architectures: list[EpisodeArchitecture] = []
            architecture_reports: list[dict[str, Any]] = []
            architecture_attempt_reports: list[dict[str, Any]] = []
            state_post_by_episode: dict[int, NarrativeState] = {}
            reconciliations: list[NarrativeStateReconciliation] = []

            for episode_number in episode_numbers:
                architecture, architecture_report, architecture_attempts = architecture_results[
                    episode_number - 1
                ]
                reconciliation = reconciliation_by_episode[episode_number]
                state_post_by_episode[episode_number] = reconciliation.state_post.model_copy(
                    deep=True
                )
                architectures.append(architecture)
                architecture_reports.append(architecture_report)
                architecture_attempt_reports.append(
                    {
                        "episode_number": episode_number,
                        "accepted_attempt": next(
                            (
                                item["attempt"]
                                for item in architecture_attempts
                                if item["status"] == "accepted"
                            ),
                            None,
                        ),
                        "attempts": architecture_attempts,
                    }
                )
                reconciliations.append(reconciliation)

                ep_dir = project_dir / "episodes" / str(episode_number)
                _save_json(
                    ep_dir / "narrative_state_pre.json", state_pre_by_episode[episode_number]
                )
                _save_json(
                    ep_dir / "narrative_state_post.json", state_post_by_episode[episode_number]
                )
                _save_json(ep_dir / "narrative_state_reconciliation.json", reconciliation)

            current_state = state_post_by_episode[episode_numbers[-1]]

            realization_reports = [
                _build_episode_architecture_realization(
                    strategy_episode=strategy_episode_map[architecture.episode_number],
                    architecture=architecture,
                    pipeline_config=project.config,
                    narrator_profile=strategy.narrator_profile,
                    primitive_lookup=_flatten_synthesis_primitives(synthesis_map),
                    series_explanation_registry=strategy.series_explanation_registry,
                )
                for architecture in architectures
            ]
            for realization in realization_reports:
                for warning in realization["warnings"]:
                    self.run_logger.log(
                        "episode_architecture_warning",
                        episode=realization["episode_number"],
                        warning=warning,
                    )
                    logger.warning(
                        "episode_architecture_warning episode=%s %s",
                        realization["episode_number"],
                        warning,
                    )
            _save_json(
                project_dir / "episode_architectures.json",
                {"episodes": [episode.model_dump(mode="json") for episode in architectures]},
            )
            _save_json(
                project_dir / "architecture_realization.json",
                {"episodes": realization_reports},
            )
            _save_json(
                project_dir / "episode_architecture_attempts.json",
                {"episodes": architecture_attempt_reports},
            )
            _save_json(
                project_dir / "narrative_state_timeline.json",
                {
                    "episodes": [
                        {
                            "episode_number": episode_number,
                            "pre": state_pre_by_episode[episode_number].model_dump(mode="json"),
                            "post": state_post_by_episode[episode_number].model_dump(mode="json"),
                        }
                        for episode_number in range(1, project.episode_count + 1)
                    ]
                },
            )
            _save_json(project_dir / "narrative_state_latest.json", current_state)
            ctx["output_summary"] = {
                "episode_count": len(architectures),
                "titles": [
                    strategy_episode_map[n].title for n in range(1, project.episode_count + 1)
                ],
                "reconciliation_count": len(reconciliations),
            }
            architecture_actor_metrics = _merge_actor_metric_dicts(architecture_reports)
            return (
                architectures,
                state_pre_by_episode,
                state_post_by_episode,
                reconciliations,
                architecture_actor_metrics,
            )

    async def _plan_series_with_narrative_state(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
        scene_discovery: SceneDiscoveryArtifact | None = None,
        retained_excerpts: ExcerptArtifact | None = None,
    ) -> tuple[
        list[EpisodeArchitecture],
        list[EpisodePlan],
        dict[int, NarrativeState],
        dict[int, NarrativeState],
        dict[str, Any],
    ]:
        excerpt_by_id = retained_excerpts.excerpt_by_id() if retained_excerpts is not None else {}
        (
            architectures,
            state_pre_by_episode,
            state_post_by_episode,
            reconciliations,
            architecture_actor_metrics,
        ) = await self._build_episode_architectures_with_narrative_state(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            excerpt_by_id=excerpt_by_id,
        )
        plans, planning_actor_metrics = await self._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=architectures,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            narrative_state_pre_by_episode=state_pre_by_episode,
            excerpt_by_id=excerpt_by_id,
        )
        return (
            architectures,
            plans,
            state_pre_by_episode,
            state_post_by_episode,
            {
                "episode_architecture": architecture_actor_metrics,
                "episode_planning": planning_actor_metrics,
                "reconciliation_count": len(reconciliations),
            },
        )

    async def _plan_series(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        episode_architectures: list[EpisodeArchitecture],
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
        narrative_state_pre_by_episode: dict[int, NarrativeState] | None = None,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
    ) -> tuple[list[EpisodePlan], dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        excerpt_by_id = excerpt_by_id or {}
        async with _stage_log(
            self.run_logger,
            "episode_planning",
            project_dir,
            episode_count=project.episode_count,
        ) as ctx:
            episode_map = {episode.episode_number: episode for episode in episode_architectures}
            strategy_episode_map = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            missing_episodes = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in episode_map or episode_number not in strategy_episode_map
            ]
            if missing_episodes:
                raise RuntimeError(
                    "Narrative strategy and episode architecture must both cover "
                    f"episodes: {missing_episodes}"
                )
            passage_lookup = _build_passage_lookup(corpus)
            primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            project_metadata = _build_episode_planning_project_metadata(project)
            planning_sem = asyncio.Semaphore(max(1, project.config.episode_planning_concurrency))
            ordered_episodes = [
                (
                    episode_map[episode_number],
                    strategy_episode_map[episode_number],
                )
                for episode_number in range(1, project.episode_count + 1)
            ]

            async def _plan_episode(
                episode: EpisodeArchitecture,
                strategy_episode: StrategyEpisode,
            ) -> tuple[int, EpisodePlan, dict[str, Any]]:
                async with planning_sem:
                    narrative_state_pre = (
                        narrative_state_pre_by_episode.get(episode.episode_number)
                        if narrative_state_pre_by_episode is not None
                        else None
                    )
                    planning_runtime = _build_episode_planning_runtime(
                        project=project,
                        strategy=strategy,
                        strategy_episode=strategy_episode,
                        architecture=episode,
                        synthesis_map=synthesis_map,
                        corpus=corpus,
                        actor_metadata=actor_metadata,
                        narrative_state_pre=narrative_state_pre,
                        primitive_lookup=primitive_lookup,
                        passage_lookup=passage_lookup,
                        excerpt_by_id=excerpt_by_id,
                    )
                    plan_draft: EpisodePlanDraft | None = None
                    actor_link_metrics: dict[str, Any] = {}
                    actor_explanation_warnings: list[str] = []
                    planning_feedback: dict[str, Any] | None = None
                    planning_attempts: list[dict[str, Any]] = []
                    max_attempts = self.episode_planning_agent.max_retry_attempts
                    # Change 1: derive per-section scene targets from
                    # architecture runtime distribution.
                    from podcast_agent.schemas.models import (
                        derive_section_scene_targets as _derive_section_scene_targets,
                    )

                    mode_scene_job_budget = scene_job_budget_for_mode(project.config.podcast_mode)
                    section_scene_targets = _derive_section_scene_targets(
                        episode,
                        mode=project.config.podcast_mode,
                        scene_job_budget=mode_scene_job_budget,
                    )
                    for attempt in range(1, max_attempts + 1):
                        payload = self.episode_planning_agent.build_payload(
                            strategy_episode=strategy_episode.model_dump(mode="json"),
                            architecture=planning_runtime.episode_payload,
                            synthesis_map=planning_runtime.episode_synthesis_map_payload,
                            project_metadata=project_metadata,
                            scene_job_budget=mode_scene_job_budget,
                            available_passages=planning_runtime.available_passages,
                            host_policy=planning_runtime.host_policy,
                            narrative_state_pre=(
                                narrative_state_pre.model_dump(mode="json")
                                if narrative_state_pre is not None
                                else None
                            ),
                            continuity_contract_pre=planning_runtime.continuity_contract_pre,
                            actor_metadata=planning_runtime.compact_episode_actor_metadata,
                            planning_feedback=planning_feedback,
                            field_semantics=_build_field_semantics_payload(),
                            excerpts=planning_runtime.episode_excerpts,
                            section_scene_targets=section_scene_targets,
                        )
                        try:
                            plan_draft = await asyncio.to_thread(
                                self.episode_planning_agent.run,
                                payload,
                            )
                            (
                                plan_draft,
                                actor_link_metrics,
                                actor_explanation_warnings,
                            ) = _postcheck_episode_plan_draft(
                                strategy_episode=strategy_episode,
                                architecture=episode,
                                plan_draft=plan_draft,
                                episode_actor_metadata=planning_runtime.episode_actor_metadata,
                                narrator_profile=strategy.narrator_profile,
                                continuity_contract_pre=planning_runtime.continuity_contract_pre,
                            )
                            planning_attempts.append(
                                _build_attempt_record(
                                    attempt=attempt,
                                    status="accepted",
                                )
                            )
                            break
                        except ComplianceViolationError as exc:
                            attempt_feedback = _build_plan_transition_feedback(exc)
                            planning_attempts.append(
                                _build_attempt_record(
                                    attempt=attempt,
                                    status="rejected_contract",
                                    blocking_issue=attempt_feedback["issue"],
                                    retry_feedback=attempt_feedback,
                                )
                            )
                            self.run_logger.log(
                                "episode_planning_attempt_warning",
                                episode=episode.episode_number,
                                attempt=attempt,
                                status="rejected_contract",
                                blocking_issue=attempt_feedback["issue"],
                                warning_count=0,
                                warnings=[],
                            )
                            if attempt >= max_attempts:
                                raise
                            backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                            planning_feedback = attempt_feedback
                            self.run_logger.log(
                                "episode_planning_retry_scheduled",
                                episode=episode.episode_number,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                next_attempt=attempt + 1,
                                backoff_seconds=backoff,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                issue=planning_feedback["issue"],
                            )
                            await asyncio.sleep(backoff)
                    if plan_draft is None:
                        raise RuntimeError(
                            f"Episode planning did not produce a plan draft for episode {episode.episode_number}."
                        )
                    spine_diagnostics = _build_spine_plan_diagnostics(
                        strategy_episode=strategy_episode,
                        plan=plan_draft,
                        scene_job_budget=scene_job_budget_for_mode(project.config.podcast_mode),
                    )
                    host_moves_diagnostics, host_move_warnings = _build_host_move_plan_diagnostics(
                        scene_cards=plan_draft.scene_cards,
                        architecture=episode,
                        narrator_profile=strategy.narrator_profile,
                    )
                    host_density_diagnostics, host_density_warnings = (
                        _build_host_density_diagnostics(
                            scene_cards=plan_draft.scene_cards,
                            architecture=episode,
                        )
                    )
                    scene_card_count_warnings = _build_scene_card_count_warnings(
                        scene_card_count=len(plan_draft.scene_cards),
                        scene_card_target_min=project.config.scene_card_target_min,
                        scene_card_target_max=project.config.scene_card_target_max,
                    )
                    scene_card_primitive_warnings = _build_scene_card_primitive_warnings(
                        scene_cards=plan_draft.scene_cards,
                        primitive_pool_ids=set(planning_runtime.primitive_ids),
                        primitive_by_id=primitive_lookup,
                        primitive_min=project.config.scene_card_primitives_min,
                        primitive_max=project.config.scene_card_primitives_max,
                    )
                    scene_card_family_warnings = _build_scene_card_family_warnings(
                        strategy_episode=strategy_episode,
                        primitive_pool_ids=set(planning_runtime.primitive_ids),
                        primitive_by_id=primitive_lookup,
                    )
                    section_realization_reports, section_planning_warnings = (
                        _build_section_plan_realization(
                            episode=episode,
                            scene_cards=plan_draft.scene_cards,
                            words_per_minute=float(self.settings.pipeline.spoken_words_per_minute),
                        )
                    )
                    state_alignment_diagnostics, state_alignment_warnings = (
                        _build_state_alignment_diagnostics(
                            architecture=episode,
                            scene_cards=plan_draft.scene_cards,
                        )
                    )
                    structural_card_concreteness_warnings = (
                        _build_structural_card_concreteness_warnings(
                            scene_cards=plan_draft.scene_cards,
                        )
                    )
                    human_grounding_diagnostics, human_grounding_warnings = (
                        _build_human_grounding_warnings(
                            scene_cards=plan_draft.scene_cards,
                        )
                    )
                    comparative_aside_warnings = _build_comparative_aside_scene_warnings(
                        architecture=episode,
                        scene_cards=plan_draft.scene_cards,
                    )
                    scene_job_counts = _build_scene_job_counts(plan_draft.scene_cards)
                    scene_role_counts: dict[str, int] = {}
                    for scene in plan_draft.scene_cards:
                        scene_role_counts[scene.scene_role] = (
                            scene_role_counts.get(scene.scene_role, 0) + 1
                        )
                    section_load_warnings = [
                        warning
                        for warning in section_planning_warnings
                        if warning.startswith("section_scene_card_load_high")
                        or warning.startswith("section_projected_word_count_high")
                    ]
                    planning_warnings = (
                        scene_card_count_warnings
                        + scene_card_primitive_warnings
                        + scene_card_family_warnings
                        + structural_card_concreteness_warnings
                        + human_grounding_warnings
                        + comparative_aside_warnings
                        + actor_explanation_warnings
                        + section_planning_warnings
                        + state_alignment_warnings
                        + list(spine_diagnostics.get("scene_job_budget_warnings", []))
                        + host_move_warnings
                        + host_density_warnings
                    )
                    for warning in planning_warnings:
                        self.run_logger.log(
                            "episode_planning_warning",
                            episode=episode.episode_number,
                            warning=warning,
                        )
                        logger.warning(
                            "episode_planning_warning episode=%s %s",
                            episode.episode_number,
                            warning,
                        )
                    target_word_count = int(
                        round(
                            float(episode.runtime_minutes)
                            * float(self.settings.pipeline.spoken_words_per_minute)
                        )
                    )
                    plan_payload = plan_draft.model_dump(mode="json")
                    plan = EpisodePlan.model_validate(
                        {
                            **plan_payload,
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
                        "accepted_warning_count": len(planning_warnings),
                        "accepted_warnings": planning_warnings,
                        "scene_card_count_warnings": scene_card_count_warnings,
                        "scene_card_primitive_warnings": scene_card_primitive_warnings,
                        "scene_card_family_warnings": scene_card_family_warnings,
                        "scene_job_budget_warnings": list(
                            spine_diagnostics.get("scene_job_budget_warnings", [])
                        ),
                        "structural_card_concreteness_warnings": structural_card_concreteness_warnings,
                        "human_grounding_warnings": human_grounding_warnings,
                        "section_load_warnings": section_load_warnings,
                        "section_realization": section_realization_reports,
                        "state_alignment": state_alignment_diagnostics,
                        "host_density_diagnostics": host_density_diagnostics,
                        "scene_card_warning_count": len(planning_warnings),
                        "scene_role_counts": scene_role_counts,
                        "scene_job_counts": scene_job_counts,
                        "human_grounding_diagnostics": human_grounding_diagnostics,
                        "section_count": len(episode.sections),
                        "core_primitive_count": len(
                            strategy_episode.episode_spine.core_primitive_ids
                        ),
                        "covered_core_primitive_count": len(
                            strategy_episode.episode_spine.core_primitive_ids
                        ),
                        "missing_core_primitive_ids": [],
                        "spine_diagnostics": spine_diagnostics,
                        "host_moves_diagnostics": host_moves_diagnostics,
                        "actor_link_metrics": actor_link_metrics,
                        "allocated_duration_seconds": sum(
                            scene.estimated_duration_seconds for scene in plan.scene_cards
                        ),
                        "planning_attempts": {
                            "episode_number": episode.episode_number,
                            "accepted_attempt": next(
                                (
                                    item["attempt"]
                                    for item in planning_attempts
                                    if item["status"] == "accepted"
                                ),
                                None,
                            ),
                            "attempts": planning_attempts,
                        },
                    }
                    return episode.episode_number, plan, report

            planning_results = await asyncio.gather(
                *[
                    _plan_episode(episode, strategy_episode)
                    for episode, strategy_episode in ordered_episodes
                ]
            )
            planning_results.sort(key=lambda item: item[0])
            planned_episodes = [plan for _, plan, _ in planning_results]
            planning_reports = [report for _, _, report in planning_results]
            planning_actor_metrics = _merge_actor_metric_dicts(
                report.get("actor_link_metrics", {}) for report in planning_reports
            )
            planning_attempt_reports = [
                report.pop("planning_attempts")
                for report in planning_reports
                if "planning_attempts" in report
            ]

            _save_json(
                project_dir / "series_plan.json",
                {"episodes": [episode.model_dump(mode="json") for episode in planned_episodes]},
            )
            _save_json(
                project_dir / "episode_plan_realization.json",
                {"episodes": planning_reports},
            )
            _save_json(
                project_dir / "episode_planning_attempts.json",
                {"episodes": planning_attempt_reports},
            )

            ctx["output_summary"] = {
                "episode_count": len(planned_episodes),
                "titles": [strategy_episode.title for _, strategy_episode in ordered_episodes],
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
            payload = _load_json(
                project_dir / "episodes" / str(episode_number) / "episode_script.json"
            )
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

        utilization = {
            "summary": {
                "episode_count": len(episode_numbers),
                "planned_episode_count": len(episode_plans),
                "total_passages": corpus.total_passages,
                "utilized_passages": len(utilized_passage_ids),
                "utilization_ratio": (len(utilized_passage_ids) / max(1, corpus.total_passages)),
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
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        project: ThematicProject,
        corpus: ThematicCorpus,
        actor_metadata: ActorMetadata,
        project_dir: Path,
        host_policy: dict[str, Any],
        primitive_lookup: dict[str, SynthesisPrimitiveBase],
        semaphore: asyncio.Semaphore,
        spoken_semaphore: asyncio.Semaphore | None = None,
        series_explanation_registry: list[Any] | None = None,
        narrative_state_pre: NarrativeState | None = None,
        narrative_state_post: NarrativeState | None = None,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
    ) -> tuple[int, SpokenScript]:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(plan.episode_number)
            ep_dir.mkdir(parents=True, exist_ok=True)
            continuity_contract_pre = _build_continuity_contract(
                narrative_state=narrative_state_pre,
                episode_number=plan.episode_number,
                phase="pre",
            )
            continuity_contract_post = _build_continuity_contract(
                narrative_state=narrative_state_post,
                episode_number=plan.episode_number,
                phase="post",
            )
            script = await self._write_episode(
                plan,
                strategy_episode,
                architecture,
                project,
                corpus,
                ep_dir,
                project_dir,
                actor_metadata,
                host_policy,
                narrative_state_pre=narrative_state_pre,
                narrative_state_post=narrative_state_post,
                continuity_contract_pre=continuity_contract_pre,
                continuity_contract_post=continuity_contract_post,
                primitive_lookup=primitive_lookup,
                excerpt_by_id=excerpt_by_id,
            )
            spine_diagnostics = _load_json(ep_dir / "spine_diagnostics.json") or {}
            return await self._continue_episode_from_script(
                plan=plan,
                strategy_episode=strategy_episode,
                architecture=architecture,
                script=script,
                project=project,
                corpus=corpus,
                actor_metadata=actor_metadata,
                project_dir=project_dir,
                ep_dir=ep_dir,
                host_policy=host_policy,
                primitive_lookup=primitive_lookup,
                spoken_semaphore=spoken_semaphore or semaphore,
                series_explanation_registry=series_explanation_registry,
                narrative_state_pre=narrative_state_pre,
                narrative_state_post=narrative_state_post,
                excerpt_by_id=excerpt_by_id,
                spine_diagnostics=spine_diagnostics,
                host_moves_diagnostics={},
            )

    async def _continue_episode_from_script(
        self,
        *,
        plan: EpisodePlan,
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        script: EpisodeScript,
        project: ThematicProject,
        corpus: ThematicCorpus,
        actor_metadata: ActorMetadata,
        project_dir: Path,
        ep_dir: Path,
        host_policy: dict[str, Any],
        primitive_lookup: dict[str, SynthesisPrimitiveBase],
        spoken_semaphore: asyncio.Semaphore,
        series_explanation_registry: list[Any] | None = None,
        narrative_state_pre: NarrativeState | None = None,
        narrative_state_post: NarrativeState | None = None,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
        spine_diagnostics: dict[str, Any] | None = None,
        host_moves_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[int, SpokenScript]:
        continuity_contract_pre = _build_continuity_contract(
            narrative_state=narrative_state_pre,
            episode_number=plan.episode_number,
            phase="pre",
        )
        continuity_contract_post = _build_continuity_contract(
            narrative_state=narrative_state_post,
            episode_number=plan.episode_number,
            phase="post",
        )
        spine_diagnostics = dict(spine_diagnostics or {})
        host_moves_diagnostics = dict(host_moves_diagnostics or {})

        # Grounding stage removed in Change 3. The LLM-as-judge runs in
        # its place between writing and style_audit and feeds its
        # judgment into style_audit's payload as remediation direction.
        # Excerpt verbatim verification still runs (in
        # pipeline/excerpt_verification.py) as part of the spoken pass.
        judgment = await self._judge_episode(
            plan=plan,
            script=script,
            architecture=architecture,
            ep_dir=ep_dir,
            project_dir=project_dir,
            spine_diagnostics=spine_diagnostics,
            host_moves_diagnostics=host_moves_diagnostics,
            project_id=project.project_id,
        )

        # Pull the cumulative series tic counts so the lint pass can
        # surface carryover-warning families to the style_audit prompt.
        from podcast_agent.pipeline.series_state import load_series_state

        series_state_pre_audit = load_series_state(project_dir, project.project_id)
        series_carryover_counts = dict(series_state_pre_audit.cumulative_family_counts)
        script, audit_response = await self._style_audit_episode(
            plan.episode_number,
            script,
            architecture,
            ep_dir,
            project_dir,
            plan=plan,
            host_policy=host_policy,
            narrative_state_pre=narrative_state_pre,
            narrative_state_post=narrative_state_post,
            continuity_contract_pre=continuity_contract_pre,
            continuity_contract_post=continuity_contract_post,
            strategy_episode=strategy_episode,
            series_explanation_registry=series_explanation_registry,
            quality_judgment=judgment,
            series_carryover_counts=series_carryover_counts,
            project_id=project.project_id,
        )

        # Redraft loop removed: style_audit now has broad-rewrite scope on
        # Opus and acts on the judge's remediation hints directly. The
        # judge → style_audit → spoken pipeline is one shot per episode.

        if not project.config.skip_spoken_delivery:
            async with spoken_semaphore:
                spoken = await self._rewrite_for_speech(
                    plan.episode_number,
                    script,
                    project,
                    ep_dir,
                    project_dir,
                    architecture=architecture,
                    plan=plan,
                    host_policy=host_policy,
                    narrative_state_pre=narrative_state_pre,
                    narrative_state_post=narrative_state_post,
                    continuity_contract_pre=continuity_contract_pre,
                    continuity_contract_post=continuity_contract_post,
                )
        else:
            spoken = SpokenScript(
                episode_number=plan.episode_number,
                title=script.title,
                framing=script.framing,
                sections=[
                    SpokenSection(
                        section_id=section.section_id,
                        text=section.text,
                        section_sonic_plan=section.section_sonic_plan,
                    )
                    for section in script.prose_sections
                ],
                tts_provider=project.config.tts_provider,
            )
            _save_json(ep_dir / "spoken_script.json", spoken)
            self.run_logger.log("spoken_delivery_skipped", episode=plan.episode_number)

        return (plan.episode_number, spoken)

    async def _write_episode(
        self,
        plan: EpisodePlan,
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        project: ThematicProject,
        corpus: ThematicCorpus,
        ep_dir: Path,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
        host_policy: dict[str, Any] | None = None,
        narrative_state_pre: NarrativeState | None = None,
        narrative_state_post: NarrativeState | None = None,
        continuity_contract_pre: dict[str, Any] | None = None,
        continuity_contract_post: dict[str, Any] | None = None,
        primitive_lookup: dict[str, SynthesisPrimitiveBase] | None = None,
        excerpt_by_id: dict[str, ExcerptRecord] | None = None,
    ) -> EpisodeScript:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        primitive_lookup = primitive_lookup or {}
        excerpt_by_id = excerpt_by_id or {}
        async with _stage_log(
            self.run_logger,
            f"write_episode_{plan.episode_number}",
            project_dir,
            episode=plan.episode_number,
            scene_card_count=len(plan.scene_cards),
            writing_source_mode=_WRITING_SOURCE_MODE_FULL_CHUNK,
        ) as ctx:
            passage_lookup = _build_passage_lookup(corpus)
            _validate_writing_context(
                strategy_episode=strategy_episode,
                architecture=architecture,
                plan=plan,
                passage_lookup=passage_lookup,
            )
            field_semantics = _build_field_semantics_payload()
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author} for b in project.books
            ]
            (
                scene_authorial_passages_by_scene_id,
                section_authorial_passages_by_section_id,
            ) = _resolve_authorial_passages(
                architecture=architecture,
                plan=plan,
            )
            (
                scene_word_count_targets_lower,
                scene_word_count_targets_higher,
            ) = _compute_scene_word_count_bands(
                plan.scene_cards,
                plan.target_word_count,
            )
            full_episode_plan_payload = plan.model_dump(mode="json")
            scene_payload_by_id: dict[str, dict[str, Any]] = {}
            for scene_payload in full_episode_plan_payload.get("scene_cards", []):
                scene_id = scene_payload.get("scene_id")
                if not scene_id:
                    continue
                scene_payload.pop("estimated_duration_seconds", None)
                scene_payload["authorial_passages"] = list(
                    scene_authorial_passages_by_scene_id.get(scene_id, [])
                )
                scene_payload["target_word_count_lower"] = int(
                    scene_word_count_targets_lower.get(scene_id, 0)
                )
                scene_payload["target_word_count_higher"] = int(
                    scene_word_count_targets_higher.get(scene_id, 0)
                )
                scene_payload_by_id[scene_id] = _build_writer_scene_brief(scene_payload)
            writing_agent = (
                self.writing_agent_no_citations
                if project.config.skip_grounding
                else self.writing_agent
            )
            episode_target_word_count_lower = sum(scene_word_count_targets_lower.values())
            episode_target_word_count_higher = sum(scene_word_count_targets_higher.values())
            writing_windows = _split_episode_writing_windows(
                plan=plan,
                architecture=architecture,
                scene_word_count_targets_lower=scene_word_count_targets_lower,
                scene_word_count_targets_higher=scene_word_count_targets_higher,
                max_windows=project.config.episode_writing_batch_count,
            )
            warning_threshold = int(
                math.ceil(
                    float(episode_target_word_count_higher) * _WRITING_WORD_OVERRUN_WARNING_RATIO
                )
            )
            script: EpisodeScript | None = None
            actual_word_count = 0
            max_attempts = writing_agent.max_retry_attempts
            writing_feedback_by_part: dict[int, str] = {}
            for attempt in range(1, max_attempts + 1):
                try:
                    actual_word_count = 0
                    normalized_section_outputs: list[dict[str, Any]] = []
                    prior_window_continuity: dict[str, Any] | None = None
                    for part_number, window_scene_cards in enumerate(writing_windows, start=1):
                        part_target_word_count_lower = sum(
                            scene_word_count_targets_lower.get(scene.scene_id, 0)
                            for scene in window_scene_cards
                        )
                        part_target_word_count_higher = sum(
                            scene_word_count_targets_higher.get(scene.scene_id, 0)
                            for scene in window_scene_cards
                        )
                        part_target_word_count = max(
                            1,
                            int(
                                round(
                                    (part_target_word_count_lower + part_target_word_count_higher)
                                    / 2.0
                                )
                            ),
                        )
                        window_plan_payload = _build_window_plan_payload(
                            plan=plan,
                            window_scene_cards=window_scene_cards,
                            scene_payload_by_id=scene_payload_by_id,
                            target_word_count=part_target_word_count,
                        )
                        window_architecture_payload = _build_window_architecture_payload(
                            architecture=architecture,
                            window_scene_cards=window_scene_cards,
                            section_authorial_passages_by_section_id=section_authorial_passages_by_section_id,
                        )
                        window_passages = _build_window_passages(
                            window_scene_cards=window_scene_cards,
                            passage_lookup=passage_lookup,
                            excerpt_by_id=excerpt_by_id,
                        )
                        window_actor_metadata = _build_window_actor_metadata(
                            window_scene_cards=window_scene_cards,
                            strategy_episode=strategy_episode,
                            architecture=architecture,
                            actor_metadata=actor_metadata,
                        )
                        window_scene_primitive_briefs = _build_scene_primitive_briefs(
                            scene_cards=window_scene_cards,
                            primitive_lookup=primitive_lookup,
                        )
                        window_scene_excerpt_briefs = _build_scene_excerpt_briefs(
                            scene_cards=window_scene_cards,
                            excerpt_by_id=excerpt_by_id,
                            recall_excerpt_ids=set(
                                strategy_episode.episode_spine.recall_excerpt_ids
                            ),
                        )
                        async with _stage_log(
                            self.run_logger,
                            f"write_episode_{plan.episode_number}_part_{part_number}",
                            project_dir,
                            episode=plan.episode_number,
                            part_number=part_number,
                            scene_card_count=len(window_scene_cards),
                            writing_source_mode=_WRITING_SOURCE_MODE_FULL_CHUNK,
                            continuity_scene_count=(
                                prior_window_continuity.get("completed_scene_count", 0)
                                if prior_window_continuity is not None
                                else 0
                            ),
                        ) as part_ctx:
                            payload = writing_agent.build_payload(
                                episode_number=plan.episode_number,
                                strategy_episode=_build_writer_strategy_episode_payload(
                                    strategy_episode
                                ),
                                architecture=window_architecture_payload,
                                episode_plan=window_plan_payload,
                                passages=window_passages,
                                book_metadata=book_metadata,
                                episode_target_word_count_lower=part_target_word_count_lower,
                                episode_target_word_count_higher=part_target_word_count_higher,
                                skip_grounding=project.config.skip_grounding,
                                host_policy=host_policy,
                                narrative_state_pre=(
                                    narrative_state_pre.model_dump(mode="json")
                                    if narrative_state_pre is not None
                                    else None
                                ),
                                narrative_state_post=(
                                    narrative_state_post.model_dump(mode="json")
                                    if narrative_state_post is not None
                                    else None
                                ),
                                continuity_contract_pre=continuity_contract_pre,
                                continuity_contract_post=continuity_contract_post,
                                scene_primitive_briefs=window_scene_primitive_briefs,
                                scene_excerpt_briefs=window_scene_excerpt_briefs,
                                actor_metadata=window_actor_metadata,
                                writing_feedback=writing_feedback_by_part.get(part_number),
                                prior_window_continuity=prior_window_continuity,
                                field_semantics=field_semantics,
                            )
                            result = await asyncio.to_thread(writing_agent.run, payload)
                            part_word_count = _writing_result_word_count(result)
                            actual_word_count += part_word_count
                            try:
                                normalized_window_outputs = _normalize_writing_section_outputs(
                                    result=result,
                                    architecture=architecture,
                                    scene_cards=window_scene_cards,
                                    episode_number=plan.episode_number,
                                    skip_grounding=project.config.skip_grounding,
                                )
                            except ComplianceViolationError as exc:
                                exc.data = {
                                    **(exc.data or {}),
                                    "failed_part_number": part_number,
                                }
                                raise
                            part_ctx["output_summary"] = {
                                "words": part_word_count,
                                "section_count": len(normalized_window_outputs),
                            }
                        normalized_section_outputs.extend(normalized_window_outputs)
                        if part_number < len(writing_windows):
                            remaining_scene_ids = {
                                scene.scene_id
                                for later_window in writing_windows[part_number:]
                                for scene in later_window
                            }
                            prior_window_continuity = _build_prior_window_continuity(
                                completed_scene_cards=window_scene_cards,
                                completed_scene_outputs=normalized_window_outputs,
                                strategy_episode=strategy_episode,
                                remaining_scene_ids=remaining_scene_ids,
                            )
                    normalized_sections = [
                        ProseSection.model_validate(section_output)
                        for section_output in normalized_section_outputs
                    ]
                    normalized_sections = _attach_section_sonic_plan_to_prose_sections(
                        normalized_sections,
                        architecture,
                    )
                    script = EpisodeScript(
                        episode_number=plan.episode_number,
                        title=strategy_episode.title,
                        framing=plan.framing,
                        prose_sections=normalized_sections,
                        total_word_count=_script_total_word_count(
                            EpisodeScript(
                                episode_number=plan.episode_number,
                                title=strategy_episode.title,
                                framing=plan.framing,
                                prose_sections=normalized_sections,
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
                    break
                except ValidationError as exc:
                    contract_exc = ComplianceViolationError(
                        "Episode writing produced invalid section or script content "
                        f"(episode {plan.episode_number}): {exc}",
                        data={"episode_number": plan.episode_number},
                    )
                    if attempt >= max_attempts:
                        raise contract_exc from exc
                    backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                    failed_part_number = int(
                        (contract_exc.data or {}).get("failed_part_number") or len(writing_windows)
                    )
                    writing_feedback_by_part[failed_part_number] = _build_writing_retry_feedback(
                        contract_exc
                    )
                    self.run_logger.log(
                        "episode_writing_retry_scheduled",
                        episode=plan.episode_number,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        next_attempt=attempt + 1,
                        backoff_seconds=backoff,
                        error_type=type(contract_exc).__name__,
                        error_message=str(contract_exc),
                    )
                    await asyncio.sleep(backoff)
                except ComplianceViolationError as exc:
                    if attempt >= max_attempts:
                        raise
                    backoff = min(2 ** (attempt - 1), 16) + (time.monotonic() % 1)
                    failed_part_number = int(
                        (exc.data or {}).get("failed_part_number") or len(writing_windows)
                    )
                    writing_feedback_by_part[failed_part_number] = _build_writing_retry_feedback(
                        exc
                    )
                    self.run_logger.log(
                        "episode_writing_retry_scheduled",
                        episode=plan.episode_number,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        next_attempt=attempt + 1,
                        backoff_seconds=backoff,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    await asyncio.sleep(backoff)
            if script is None:
                raise RuntimeError(
                    f"Episode writing did not produce a script for episode {plan.episode_number}."
                )
            if actual_word_count > warning_threshold:
                self.run_logger.log(
                    "episode_writing_budget_warning",
                    episode=plan.episode_number,
                    actual_word_count=actual_word_count,
                    target_word_count_lower=episode_target_word_count_lower,
                    target_word_count_higher=episode_target_word_count_higher,
                    warning_threshold=warning_threshold,
                    over_high_word_count=(actual_word_count - episode_target_word_count_higher),
                    over_high_ratio=(actual_word_count / max(1, episode_target_word_count_higher)),
                )
            _save_json(ep_dir / "episode_script.json", script)
            spine_script_diagnostics = _build_spine_script_diagnostics(
                strategy_episode=strategy_episode,
                plan=plan,
                script=script,
            )
            _save_json(
                ep_dir / "spine_diagnostics.json",
                spine_script_diagnostics,
            )
            self.run_logger.log(
                "spine_diagnostics",
                episode=plan.episode_number,
                **{
                    key: value
                    for key, value in spine_script_diagnostics.items()
                    if key != "failure_labels"
                },
                failure_labels=spine_script_diagnostics.get("failure_labels", []),
            )

            ctx["output_summary"] = {
                "words": script.total_word_count,
                "sections": len(script.prose_sections),
                "spine_diagnostics": spine_script_diagnostics,
            }
            return script

    async def _judge_episode(
        self,
        *,
        plan: EpisodePlan,
        script: EpisodeScript,
        architecture: EpisodeArchitecture,
        ep_dir: Path,
        project_dir: Path,
        spine_diagnostics: dict[str, Any] | None = None,
        host_moves_diagnostics: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> EpisodeQualityScore:
        """Run the LLM-as-judge on the assembled episode.

        Output is persisted to ``episodes/N/quality_judgment.json`` and the
        EpisodeQualityScore returned to the caller for inclusion in the
        style_audit payload.
        """
        from podcast_agent.pipeline.style_audit_linting import (
            compute_style_audit_lint_flags,
        )
        from podcast_agent.pipeline.tic_families import detect_tic_hits
        from podcast_agent.pipeline.text_embeddings import get_text_embedder
        from podcast_agent.pipeline.text_utils import split_sentences

        embedder = get_text_embedder()

        def _semantic_detector(text: str, section_id: str):
            return detect_tic_hits(
                split_sentences(text),
                section_id=section_id,
                embedder=embedder,
            )

        section_progression_by_id = {
            s.section_id: s.section_progression.stage.value for s in architecture.sections
        }
        lint_flags = compute_style_audit_lint_flags(
            prose_sections=[
                {"section_id": s.section_id, "text": s.text} for s in script.prose_sections
            ],
            spine_episode_answer=(
                plan.strategy_episode.episode_spine.episode_answer
                if hasattr(plan, "strategy_episode")
                else ""
            )
            or "",
            spine_pressure_line=(
                plan.strategy_episode.episode_spine.pressure_line
                if hasattr(plan, "strategy_episode")
                else ""
            )
            or "",
            section_progression_by_id=section_progression_by_id,
            semantic_detector=_semantic_detector,
            series_carryover_counts={},
        )

        architecture_summary = [
            {
                "section_id": s.section_id,
                "purpose": s.purpose.value,
                "stage": s.section_progression.stage.value,
                "is_dense": s.is_dense,
                "approx_runtime_minutes": s.approx_runtime_minutes,
                "must_stage_beats": list(s.must_stage_beats),
            }
            for s in architecture.sections
        ]
        excerpt_staging: list[dict] = []
        for section in script.prose_sections:
            for citation in section.citations:
                excerpt_staging.append(
                    {
                        "section_id": section.section_id,
                        "passage_id": citation.passage_id,
                        "text_span": citation.text_span,
                    }
                )

        if spine_diagnostics is None:
            try:
                spine_diagnostics = _load_json(ep_dir / "spine_diagnostics.json") or {}
            except Exception:
                spine_diagnostics = {}
        if host_moves_diagnostics is None:
            host_moves_diagnostics = {}

        # Load series state so the judge can flag cross-episode tic clusters
        # (cumulative_family_counts) and avoid prescribing structural moves
        # that prior episodes already used (prior_episode_remediation_hints).
        # Both are skipped for episode 1 / when project_id is unavailable.
        series_state_payload: dict | None = None
        prior_hints: list[dict] = []
        if project_id is not None:
            from podcast_agent.pipeline.series_state import load_series_state

            try:
                series_state = load_series_state(project_dir, project_id)
                cumulative = dict(series_state.cumulative_family_counts)
                if cumulative:
                    series_state_payload = {
                        "cumulative_family_counts": cumulative,
                    }
            except Exception:
                series_state_payload = None
        for prior_ep in range(1, plan.episode_number):
            prior_path = project_dir / "episodes" / str(prior_ep) / "quality_judgment.json"
            if not prior_path.exists():
                continue
            try:
                prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for section_score in prior_payload.get("section_scores", []) or []:
                section_id = section_score.get("section_id")
                weakest = section_score.get("weakest_criterion")
                for hint in section_score.get("remediation_hints", []) or []:
                    hint_text = str(hint).strip()
                    if not hint_text:
                        continue
                    prior_hints.append(
                        {
                            "episode_number": prior_ep,
                            "section_id": section_id,
                            "criterion": weakest,
                            "hint": hint_text,
                        }
                    )

        async with _stage_log(
            self.run_logger,
            f"quality_judge_{plan.episode_number}",
            project_dir,
            episode=plan.episode_number,
            section_count=len(script.prose_sections),
        ) as ctx:
            payload = self.quality_judge_agent.build_payload(
                episode_number=plan.episode_number,
                title=script.title,
                framing=script.framing.model_dump(),
                prose_sections=[s.model_dump() for s in script.prose_sections],
                architecture_summary=architecture_summary,
                excerpt_staging=excerpt_staging,
                rubric_thresholds={},
                style_audit_lint_flags=lint_flags,
                spine_diagnostics=spine_diagnostics,
                host_moves_diagnostics=host_moves_diagnostics,
                series_state=series_state_payload,
                prior_episode_remediation_hints=prior_hints or None,
            )
            judgment = await asyncio.to_thread(self.quality_judge_agent.run, payload)
            ctx["output_summary"] = {
                "overall_score": judgment.overall_score,
                "criterion_scores": {
                    cs.criterion.value: cs.score for cs in judgment.criterion_scores
                },
                "weakest_sections": list(judgment.weakest_sections),
            }
        _save_json(ep_dir / "quality_judgment.json", judgment.model_dump())
        # Stash the lint flags so style_audit can reuse them without
        # re-embedding.
        self._latest_lint_flags_by_episode[plan.episode_number] = lint_flags
        return judgment

    async def _style_audit_episode(
        self,
        episode_number: int,
        script: EpisodeScript,
        architecture: EpisodeArchitecture,
        ep_dir: Path,
        project_dir: Path,
        plan: EpisodePlan | None = None,
        host_policy: dict[str, Any] | None = None,
        narrative_state_pre: NarrativeState | None = None,
        narrative_state_post: NarrativeState | None = None,
        continuity_contract_pre: dict[str, Any] | None = None,
        continuity_contract_post: dict[str, Any] | None = None,
        strategy_episode: StrategyEpisode | None = None,
        series_explanation_registry: list[Any] | None = None,
        quality_judgment: "EpisodeQualityScore | None" = None,
        series_carryover_counts: dict[str, int] | None = None,
        project_id: str | None = None,
    ) -> tuple[EpisodeScript, "StyleAuditResponse"]:
        async with _stage_log(
            self.run_logger,
            f"style_audit_{episode_number}",
            project_dir,
            episode=episode_number,
            section_count=len(script.prose_sections),
        ) as ctx:
            style_audit_sections, style_audit_lint_flags = _build_style_audit_sections_payload(
                script=script,
                architecture=architecture,
                plan=plan,
                strategy_episode=strategy_episode,
                series_carryover_counts=series_carryover_counts,
            )
            payload = self.style_audit_agent.build_payload(
                episode_number=episode_number,
                title=script.title,
                sections=style_audit_sections,
                lint_flags=style_audit_lint_flags,
                quality_judgment=(
                    quality_judgment.model_dump(mode="json")
                    if quality_judgment is not None
                    else None
                ),
                series_carryover_counts=series_carryover_counts,
                host_policy=host_policy,
                narrative_state_pre=(
                    narrative_state_pre.model_dump(mode="json")
                    if narrative_state_pre is not None
                    else None
                ),
                narrative_state_post=(
                    narrative_state_post.model_dump(mode="json")
                    if narrative_state_post is not None
                    else None
                ),
                continuity_contract_pre=continuity_contract_pre,
                continuity_contract_post=continuity_contract_post,
                series_explanation_registry=_build_episode_explanation_registry_payload(
                    strategy_episode=strategy_episode,
                    architecture=architecture,
                    series_explanation_registry=series_explanation_registry,
                ),
                field_semantics=_build_field_semantics_payload(),
            )
            audit = await asyncio.to_thread(self.style_audit_agent.run, payload)
            _save_json(ep_dir / "style_audit_result.json", audit)
            audited_script = _apply_style_audit_to_script(
                script=script,
                audit=audit,
                spoken_words_per_minute=float(self.settings.pipeline.spoken_words_per_minute),
            )
            _save_json(ep_dir / "style_audited_script.json", audited_script)

            # Post-audit fact-coverage sweep — pure Python. Reads each
            # section's audited text against the must_land_facts.required
            # aggregated from its scene cards and the original citations
            # from the writer's prose. Writes a diagnostic JSON; logs a
            # structured event on any miss; never gates downstream stages.
            from podcast_agent.pipeline.style_audit_linting import (
                compute_fact_coverage_diagnostics,
            )

            scene_cards_by_section_id: dict[str, list[SceneCard]] = {}
            if plan is not None:
                for scene in plan.scene_cards:
                    scene_cards_by_section_id.setdefault(scene.section_id, []).append(scene)
            original_citations_by_section_id: dict[str, list[Citation]] = {
                section.section_id: list(section.citations) for section in script.prose_sections
            }
            fact_coverage = compute_fact_coverage_diagnostics(
                audited_script=audited_script,
                scene_cards_by_section_id=scene_cards_by_section_id,
                original_citations_by_section_id=(original_citations_by_section_id),
            )
            _save_json(
                ep_dir / "fact_coverage_diagnostics.json",
                fact_coverage,
            )
            for section_report in fact_coverage["sections"]:
                if (
                    section_report["missing_required"]
                    or section_report["missing_citation_passage_ids"]
                ):
                    self.run_logger.log(
                        "style_audit_fact_coverage_miss",
                        episode=episode_number,
                        section_id=section_report["section_id"],
                        missing_required=section_report["missing_required"],
                        missing_citation_passage_ids=section_report["missing_citation_passage_ids"],
                    )

            # Post-audit lint sweep — recompute lint flags on the audited
            # prose using the same detector. The pre-audit lint flags live
            # in ``style_audit_lint_flags``; the new ``post_audit_lint_flags``
            # captures what survived the audit. Both feed the residual-lint
            # diagnostic. The series_state write below reads from POST-audit
            # counts so the next episode's blocklist reflects what shipped.
            from podcast_agent.pipeline.style_audit_linting import (
                compute_style_audit_lint_flags as _post_compute_lint,
            )
            from podcast_agent.pipeline.tic_families import (
                detect_tic_hits as _post_detect_tic_hits,
            )
            from podcast_agent.pipeline.text_embeddings import (
                get_text_embedder as _post_get_text_embedder,
            )
            from podcast_agent.pipeline.text_utils import (
                split_sentences as _post_split_sentences,
            )

            _post_embedder = _post_get_text_embedder()

            def _post_audit_semantic_detector(text: str, section_id: str):
                return _post_detect_tic_hits(
                    _post_split_sentences(text),
                    section_id=section_id,
                    embedder=_post_embedder,
                )

            _audited_section_progression_by_id = {
                s.section_id: s.section_progression.stage.value for s in architecture.sections
            }
            _audited_prose_section_dicts = [
                {"section_id": ps.section_id, "text": ps.text}
                for ps in audited_script.prose_sections
            ]
            _post_spine = strategy_episode.episode_spine if strategy_episode is not None else None
            post_audit_lint_flags = _post_compute_lint(
                _audited_prose_section_dicts,
                spine_episode_answer=(_post_spine.episode_answer if _post_spine else ""),
                spine_pressure_line=(_post_spine.pressure_line if _post_spine else ""),
                section_progression_by_id=_audited_section_progression_by_id,
                semantic_detector=_post_audit_semantic_detector,
                series_carryover_counts=series_carryover_counts or {},
            )
            pre_tic_counts = dict(style_audit_lint_flags.get("tic_counts", {}))
            post_tic_counts = dict(post_audit_lint_flags.get("tic_counts", {}))
            residual_families = sorted(fam for fam, count in post_tic_counts.items() if count >= 2)
            frame_overlap_residual_sections: list[dict[str, Any]] = []
            for section_id, section_flags in post_audit_lint_flags.get("by_section", {}).items():
                if section_flags.get("is_answer_stage"):
                    continue
                opening = float(section_flags.get("opening_thesis_overlap", 0.0) or 0.0)
                closing = float(section_flags.get("closing_thesis_overlap", 0.0) or 0.0)
                if opening >= 0.30 or closing >= 0.30:
                    frame_overlap_residual_sections.append(
                        {
                            "section_id": section_id,
                            "opening_thesis_overlap": opening,
                            "closing_thesis_overlap": closing,
                        }
                    )
            tics_removed = sum(
                max(pre_tic_counts.get(fam, 0) - post_tic_counts.get(fam, 0), 0)
                for fam in set(pre_tic_counts) | set(post_tic_counts)
            )
            tics_remaining = sum(post_tic_counts.values())
            residual_lint_diagnostics = {
                "episode_number": episode_number,
                "pre_audit_tic_counts": pre_tic_counts,
                "post_audit_tic_counts": post_tic_counts,
                "tic_families_residual_at_2_plus": residual_families,
                "frame_overlap_residual_sections": (frame_overlap_residual_sections),
                "delta_summary": {
                    "tics_removed": tics_removed,
                    "tics_remaining": tics_remaining,
                    "frame_overlap_sections_remaining": len(frame_overlap_residual_sections),
                },
            }
            _save_json(
                ep_dir / "residual_lint_diagnostics.json",
                residual_lint_diagnostics,
            )
            if residual_families or frame_overlap_residual_sections:
                self.run_logger.log(
                    "style_audit_residual_lint",
                    episode=episode_number,
                    tic_families_residual_at_2_plus=residual_families,
                    frame_overlap_residual_sections=(frame_overlap_residual_sections),
                )

            host_moves_text_diagnostics = _build_host_move_text_diagnostics(
                text_by_section_id={
                    section.section_id: section.text for section in audited_script.prose_sections
                },
                plan=plan,
            )
            _save_json(
                ep_dir / "host_moves_script_diagnostics.json",
                host_moves_text_diagnostics,
            )
            continuity_script_diagnostics = _build_continuity_realization_diagnostics(
                episode_number=episode_number,
                stage="script",
                framing=audited_script.framing.model_dump(mode="json"),
                ordered_sections=[
                    {"section_id": section.section_id, "text": section.text}
                    for section in audited_script.prose_sections
                ],
                continuity_contract_pre=continuity_contract_pre or {},
                continuity_contract_post=continuity_contract_post or {},
            )
            _save_json(
                ep_dir / "continuity_script_diagnostics.json",
                continuity_script_diagnostics,
            )
            section_sonic_script_diagnostics = _build_section_sonic_realization_diagnostics(
                episode_number=episode_number,
                stage="script",
                sections=list(audited_script.prose_sections),
            )
            _save_json(
                ep_dir / "section_sonic_script_diagnostics.json",
                section_sonic_script_diagnostics,
            )
            if continuity_script_diagnostics["missed_item_ids"]:
                self.run_logger.log(
                    "continuity_script_diagnostics",
                    episode=episode_number,
                    stage="script",
                    missed_item_ids=continuity_script_diagnostics["missed_item_ids"],
                    warning_labels=continuity_script_diagnostics["warning_labels"],
                )
            if section_sonic_script_diagnostics["warning_labels"]:
                self.run_logger.log(
                    "section_sonic_script_diagnostics",
                    episode=episode_number,
                    stage="script",
                    warning_labels=section_sonic_script_diagnostics["warning_labels"],
                )
            ctx["output_summary"] = {
                "sections": len(audit.sections),
                "warnings": len(audit.episode_warnings),
                "word_delta": audited_script.total_word_count - script.total_word_count,
                "planned_host_phases": host_moves_text_diagnostics["planned_host_phase_count"],
                "approx_realized_host_phases": host_moves_text_diagnostics[
                    "approx_realized_host_phase_count"
                ],
                "continuity_misses": len(continuity_script_diagnostics["missed_item_ids"]),
                "section_sonic_warnings": section_sonic_script_diagnostics["warning_count"],
            }
        # Persist this episode's per-family tic counts to series_state.json
        # so later episodes' writing payloads can pick up a blocklist of
        # surface phrases used here.
        if project_id is not None:
            from podcast_agent.pipeline.series_state import (
                load_series_state,
                record_episode_tics,
                save_series_state,
            )
            from podcast_agent.pipeline.style_audit_linting import (
                collect_surface_phrases,
            )

            # Read POST-audit tic counts and surface phrases so the next
            # episode's series_tic_blocklist reflects what actually shipped,
            # not what the writer produced before style_audit's edits.
            family_counts = dict(post_audit_lint_flags.get("tic_counts", {}))
            surface_phrases = collect_surface_phrases(post_audit_lint_flags)
            async with self._series_state_lock:
                series_state = load_series_state(project_dir, project_id)
                series_state = record_episode_tics(
                    series_state,
                    episode_number=episode_number,
                    family_counts=family_counts,
                    surface_phrases=surface_phrases,
                )
                save_series_state(project_dir, series_state)
        return audited_script, audit

    async def _rewrite_for_speech(
        self,
        episode_number: int,
        script: EpisodeScript,
        project: ThematicProject,
        ep_dir: Path,
        project_dir: Path,
        architecture: EpisodeArchitecture | None = None,
        plan: EpisodePlan | None = None,
        host_policy: dict[str, Any] | None = None,
        narrative_state_pre: NarrativeState | None = None,
        narrative_state_post: NarrativeState | None = None,
        continuity_contract_pre: dict[str, Any] | None = None,
        continuity_contract_post: dict[str, Any] | None = None,
    ) -> SpokenScript:
        batches = _build_spoken_delivery_batches(script.prose_sections)
        async with _stage_log(
            self.run_logger,
            f"spoken_delivery_{episode_number}",
            project_dir,
            episode=episode_number,
            section_count=len(script.prose_sections),
            batch_count=len(batches),
        ) as ctx:
            rewritten_sections: list[SpokenSection] = []
            section_payload_by_id = {
                section_payload["section_id"]: section_payload
                for section_payload in _build_spoken_delivery_sections_payload(
                    script=script,
                    architecture=architecture,
                    plan=plan,
                )
            }
            previous_spoken_tail: str | None = None
            for prose_batch in batches:
                batch_sections = [
                    section_payload_by_id[prose_section.section_id]
                    for prose_section in prose_batch
                    if prose_section.section_id in section_payload_by_id
                ]
                script_payload = {
                    "framing": script.framing.model_dump(mode="json"),
                    "prose_sections": batch_sections,
                }
                # Build the new oral-rewriter input payload. The provider
                # capabilities tell the model whether it may emit
                # per-segment delivery_instructions. Quotability marks
                # surface excerpts the agent may voice in actor voice.
                tts_provider_capabilities = _build_tts_provider_capabilities(
                    project.config.tts_provider
                )
                quotability_marks = _build_quotability_marks_for_batch(
                    prose_batch=prose_batch,
                    script=script,
                    plan=plan,
                )
                payload = self.spoken_delivery_agent.build_payload(
                    episode_number=episode_number,
                    script=script_payload,
                    max_words_per_segment=project.config.spoken_chunk_max_words,
                    tts_provider=project.config.tts_provider,
                    tts_provider_capabilities=tts_provider_capabilities,
                    quotability_marks=quotability_marks,
                    host_policy=host_policy,
                    narrative_state_pre=(
                        narrative_state_pre.model_dump(mode="json")
                        if narrative_state_pre is not None
                        else None
                    ),
                    narrative_state_post=(
                        narrative_state_post.model_dump(mode="json")
                        if narrative_state_post is not None
                        else None
                    ),
                    continuity_contract_pre=continuity_contract_pre,
                    continuity_contract_post=continuity_contract_post,
                    previous_spoken_tail=previous_spoken_tail,
                    field_semantics=_build_field_semantics_payload(),
                    actor_voice_catalog=dict(project.config.actor_voice_catalog),
                )
                result = await asyncio.to_thread(self.spoken_delivery_agent.run, payload)
                # The new SpokenDeliveryResponse always returns
                # `sections: list[SpokenSection]`. Re-attach
                # `section_sonic_plan` and `sonic_cues` from the source
                # section payload because those are control metadata the
                # agent is told not to emit.
                for rewritten in result.sections:
                    source_section = section_payload_by_id.get(rewritten.section_id, {})
                    sonic_cues = [
                        SonicCue.model_validate(cue)
                        for cue in source_section.get("scene_cues", []) or []
                    ]
                    section_sonic_plan = (
                        SectionSonicPlan.model_validate(source_section["section_sonic_plan"])
                        if source_section.get("section_sonic_plan") is not None
                        else None
                    )
                    rewritten_sections.append(
                        rewritten.model_copy(
                            update={
                                "section_sonic_plan": section_sonic_plan,
                                "sonic_cues": sonic_cues,
                            }
                        )
                    )
                if result.sections:
                    previous_spoken_tail = _extract_previous_spoken_tail(result.sections[-1].text)

            spoken = SpokenScript(
                episode_number=episode_number,
                title=script.title,
                framing=script.framing,
                sections=rewritten_sections,
                tts_provider=project.config.tts_provider,
            )
            _save_json(ep_dir / "spoken_script.json", spoken)
            spoken_host_moves_diagnostics = _build_host_move_text_diagnostics(
                text_by_section_id={
                    section.section_id: section.text for section in spoken.sections
                },
                plan=plan,
            )
            _save_json(
                ep_dir / "spoken_host_moves_diagnostics.json",
                spoken_host_moves_diagnostics,
            )
            continuity_spoken_diagnostics = _build_continuity_realization_diagnostics(
                episode_number=episode_number,
                stage="spoken",
                framing=spoken.framing.model_dump(mode="json"),
                ordered_sections=[
                    {"section_id": section.section_id, "text": section.text}
                    for section in spoken.sections
                ],
                continuity_contract_pre=continuity_contract_pre or {},
                continuity_contract_post=continuity_contract_post or {},
            )
            _save_json(
                ep_dir / "continuity_spoken_diagnostics.json",
                continuity_spoken_diagnostics,
            )
            section_sonic_spoken_diagnostics = _build_section_sonic_realization_diagnostics(
                episode_number=episode_number,
                stage="spoken",
                sections=list(spoken.sections),
            )
            _save_json(
                ep_dir / "section_sonic_spoken_diagnostics.json",
                section_sonic_spoken_diagnostics,
            )
            if continuity_spoken_diagnostics["missed_item_ids"]:
                self.run_logger.log(
                    "continuity_spoken_diagnostics",
                    episode=episode_number,
                    stage="spoken",
                    missed_item_ids=continuity_spoken_diagnostics["missed_item_ids"],
                    warning_labels=continuity_spoken_diagnostics["warning_labels"],
                )
            if section_sonic_spoken_diagnostics["warning_labels"]:
                self.run_logger.log(
                    "section_sonic_spoken_diagnostics",
                    episode=episode_number,
                    stage="spoken",
                    warning_labels=section_sonic_spoken_diagnostics["warning_labels"],
                )

            ctx["output_summary"] = {
                "sections": len(spoken.sections),
                "batch_count": len(spoken.sections),
                "planned_host_phases": spoken_host_moves_diagnostics["planned_host_phase_count"],
                "approx_realized_host_phases": spoken_host_moves_diagnostics[
                    "approx_realized_host_phase_count"
                ],
                "continuity_misses": len(continuity_spoken_diagnostics["missed_item_ids"]),
                "section_sonic_warnings": section_sonic_spoken_diagnostics["warning_count"],
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
                "ffmpeg failed to merge episode audio" + (f": {error_text}" if error_text else ".")
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
            raise RuntimeError(f"Audio rendering failed for {diagnostics['failed']} segment(s).")
        if not diagnostics["merged"]:
            raise RuntimeError(diagnostics.get("merge_error", "Audio merge failed."))
        return audio_manifest

    async def _render_episode_audio(
        self,
        episode_number: int,
        spoken: SpokenScript,
        config: PipelineConfig,
        project_dir: Path,
        semaphore: asyncio.Semaphore,
        *,
        skip_audio: bool,
    ) -> AudioManifest:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(episode_number)

            async with _stage_log(
                self.run_logger,
                f"audio_{episode_number}",
                project_dir,
                episode=episode_number,
                segment_count=len(spoken.sections),
            ) as ctx:
                manifest = build_render_manifest(
                    spoken,
                    voice_id=self.settings.tts.voice,
                    speed=self.settings.tts.speed,
                    words_per_minute=self.settings.pipeline.spoken_words_per_minute,
                    base_instructions=self.settings.tts.instructions,
                    tts_model_name=self.settings.tts.model_name,
                    actor_voice_catalog=dict(config.actor_voice_catalog),
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
                max_minutes = float(config.max_episode_minutes)
                if estimated_minutes < min_minutes:
                    self.run_logger.log(
                        "episode_runtime_shortfall_warning",
                        episode=episode_number,
                        estimated_duration_minutes=estimated_minutes,
                        shortfall_minutes=(min_minutes - estimated_minutes),
                        min_episode_minutes=min_minutes,
                        max_episode_minutes=max_minutes,
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
