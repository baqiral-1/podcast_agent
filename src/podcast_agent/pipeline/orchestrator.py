"""Multi-book thematic podcast pipeline orchestrator.

Implements the four-phase pipeline:
  Phase 1: Ingest & Index (parallel per book)
  Phase 2: Thematic Intelligence (sequential cross-book)
  Phase 3: Episode Production (parallel per episode)
  Phase 4: Audio Rendering (parallel per episode)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
from typing import Any, Callable, cast
from uuid import uuid4

from pydantic import ValidationError

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_function_tagging import (
    PrimitiveFunctionTaggingAgent,
)
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.spoken_delivery_agent import (
    SpokenDeliveryAgent,
    SpokenDeliveryBatchSection,
)
from podcast_agent.agents.style_audit import StyleAuditAgent
from podcast_agent.agents.synthesis_primitives import SynthesisPrimitivesAgent
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations
from podcast_agent.config import Settings
from podcast_agent.ingestion import read_source_text, extract_chapters_from_source
from podcast_agent.langchain.llm import build_llm_client
from podcast_agent.langchain.runnables import ComplianceViolationError
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.concurrency import configure_llm_semaphore
from podcast_agent.retrieval.search import RetrievalService
from podcast_agent.retrieval.vector_store import PGVectorRetrieval
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    AudioManifest,
    AudioSegmentResult,
    ActPrimitive,
    ActorExplanationRealization,
    ActorPrimitive,
    ActorMetadata,
    ArtifactPrimitive,
    BaseSynthesisPrimitive,
    BookRecord,
    CharacterEnginePrimitive,
    Citation,
    ChapterInfo,
    ChunkingConfig,
    CoalitionFaultLinePrimitive,
    ContestedExplanationPrimitive,
    CoverageStats,
    DecisionPrimitive,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeSpine,
    EpisodeScript,
    EpochalTurnPrimitive,
    EventPrimitive,
    ExtractedArtifactPrimitive,
    ExtractedMechanismPrimitive,
    ExtractedPassage,
    GroundingReport,
    HumanCostPrimitive,
    IronyReversalPrimitive,
    MoralTrapPrimitive,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProseSection,
    ProjectStatus,
    PrimitiveFunctionTaggingArtifact,
    PrimitiveFunctionTaggingOverlayArtifact,
    PrimitiveEnrichmentOverlay,
    PRIMITIVE_SUBSTRATES,
    PrimitiveSubstrate,
    ReadingPrimitive,
    RenderManifest,
    RenderSegment,
    RepairResult,
    resolve_pipeline_config_for_mode,
    SceneCard,
    SceneCardDraft,
    EpisodePlanDraft,
    SeriesNarratorProfile,
    SeriesActorExplanationItem,
    SeriesExplanationItem,
    SegmentDiff,
    SetPieceScenePrimitive,
    StyleAuditResponse,
    SpokenSection,
    SpokenScript,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitiveBase,
    SynthesisPrimitive,
    SynthesisPrimitivesArtifact,
    SynthesisTag,
    SystemsOperatingLogicPrimitive,
    TextChunk,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
    ConditionPrimitive,
    MechanismPrimitive,
    PodcastMode,
    primitive_substrate_target_ranges_for_mode,
    apply_primitive_enrichment_overlay,
    UtterancePrimitive,
)
from podcast_agent.tts.openai_compatible import (
    build_tts_client,
    supports_openai_tts_instructions,
)
from podcast_agent.utils.actor_metadata import (
    clean_axis_actor_ids,
    clean_narrative_strategy_actor_links,
    clean_scene_actor_links,
    clean_synthesis_primitive_actor_links,
    compact_actor_registry,
    compact_actor_metadata,
    collect_actor_ids_for_primitives,
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


def _trim_synthesis_primitives_by_family_caps(
    artifact: SynthesisPrimitivesArtifact,
    *,
    family_max_counts: dict[str, int] | None = None,
) -> SynthesisPrimitivesArtifact:
    capped_counts = family_max_counts or synthesis_primitive_target_max_counts_for_mode(
        "full"
    )
    trimmed_by_family: dict[str, list[BaseSynthesisPrimitive]] = {}
    changed = False
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        items = list(artifact.primitives_by_family.get(family, []))
        family_cap = capped_counts.get(family)
        if family_cap is None or len(items) <= family_cap:
            trimmed_by_family[family] = items
            continue
        changed = True
        ranked_items = sorted(
            items,
            key=lambda primitive: (
                -primitive.narrative_importance_score,
                -len(primitive.core_passage_ids),
                -len(primitive.support_passage_ids),
                primitive.id,
            ),
        )
        trimmed_by_family[family] = ranked_items[:family_cap]
    if not changed:
        return artifact
    return artifact.model_copy(update={"primitives_by_family": trimmed_by_family})


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


def _project_scene_word_count(
    scene: SceneCardDraft | SceneCard,
    *,
    words_per_minute: float,
) -> int:
    return max(
        1,
        int(
            round(
                (float(scene.estimated_duration_seconds) * float(words_per_minute))
                / 60.0
            )
        ),
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
        prefix_totals[end] - prefix_totals[start]
        for start, end in zip(boundaries, boundaries[1:])
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
    ordered_section_ids = {
        section_id: idx for idx, section_id in enumerate(section_order)
    }

    section_boundaries = [
        boundary_index
        for boundary_index in range(1, len(scene_cards))
        if scene_cards[boundary_index - 1].section_id
        != scene_cards[boundary_index].section_id
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
        return [
            scene_cards[start:end]
            for start, end in zip(boundaries, boundaries[1:])
        ]

    return [scene_cards]


def _build_window_architecture_payload(
    *,
    architecture: EpisodeArchitecture,
    window_scene_cards: list[SceneCard],
) -> dict[str, Any]:
    section_id_set = {scene.section_id for scene in window_scene_cards}
    filtered_sections = []
    for section in architecture.sections:
        if section.section_id not in section_id_set:
            continue
        section_payload = section.model_dump(mode="json")
        section_payload["depends_on_section_ids"] = [
            section_id
            for section_id in section.depends_on_section_ids
            if section_id in section_id_set
        ]
        section_payload["sets_up_section_ids"] = [
            section_id
            for section_id in section.sets_up_section_ids
            if section_id in section_id_set
        ]
        filtered_sections.append(section_payload)

    architecture_payload = architecture.model_dump(mode="json")
    architecture_payload["sections"] = filtered_sections
    if filtered_sections and architecture.major_turn_section_id not in section_id_set:
        architecture_payload["major_turn_section_id"] = filtered_sections[-1][
            "section_id"
        ]
    return architecture_payload


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
        "dropped_support_primitive_reasons": dict(
            plan.dropped_support_primitive_reasons
        ),
        "target_word_count": max(1, int(target_word_count)),
    }


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
    must_land_facts = [
        str(fact).strip()
        for fact in list(scene_payload.get("must_land_facts", []) or [])
        if str(fact).strip()
    ]
    why_now = beat_change or (must_land_facts[0] if must_land_facts else "")
    return {
        "scene_id": scene_payload.get("scene_id"),
        "section_id": scene_payload.get("section_id"),
        "title": scene_payload.get("title", ""),
        "scene_role": scene_payload.get("scene_role", ""),
        "scene_function": scene_payload.get("scene_function", ""),
        "entry_image": scene_payload.get("entry_image", ""),
        "observable_detail": scene_payload.get("observable_detail", ""),
        "beat_change": beat_change,
        "must_land_facts": must_land_facts,
        "why_now": why_now,
        "timeframe": scene_payload.get("timeframe"),
        "location": scene_payload.get("location"),
        "actors": actors,
        "primitive_ids": list(scene_payload.get("primitive_ids", []) or []),
        "passage_ids": list(scene_payload.get("passage_ids", []) or []),
        "target_word_count_lower": int(
            scene_payload.get("target_word_count_lower", 0) or 0
        ),
        "target_word_count_higher": int(
            scene_payload.get("target_word_count_higher", 0) or 0
        ),
        "withhold_until": scene_payload.get("withhold_until"),
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
        if what_happened := _nonempty_runtime_text(
            getattr(primitive, "what_happened", None)
        ):
            payload["what_happened"] = what_happened
        if event_result := _nonempty_runtime_text(
            getattr(primitive, "event_result", None)
        ):
            payload["event_result"] = event_result
        return payload
    if substrate == PrimitiveSubstrate.ACTS:
        if act_type := _nonempty_runtime_text(getattr(primitive, "act_type", None)):
            payload["act_type"] = act_type
        if acting_subject := _nonempty_runtime_text(
            getattr(primitive, "acting_subject", None)
        ):
            payload["acting_subject"] = acting_subject
        if act_summary := _nonempty_runtime_text(
            getattr(primitive, "act_summary", None)
        ):
            payload["act_summary"] = act_summary
        if immediate_result := _nonempty_runtime_text(
            getattr(primitive, "immediate_result", None)
        ):
            payload["immediate_result"] = immediate_result
        return payload
    if substrate == PrimitiveSubstrate.UTTERANCES:
        if utterance_type := _nonempty_runtime_text(
            getattr(primitive, "utterance_type", None)
        ):
            payload["utterance_type"] = utterance_type
        if speaker := _nonempty_runtime_text(getattr(primitive, "speaker", None)):
            payload["speaker"] = speaker
        if audience := _nonempty_runtime_text(getattr(primitive, "audience", None)):
            payload["audience"] = audience
        if utterance_summary := _nonempty_runtime_text(
            getattr(primitive, "utterance_summary", None)
        ):
            payload["utterance_summary"] = utterance_summary
        if key_quote := _nonempty_runtime_text(getattr(primitive, "key_quote", None)):
            payload["key_quote"] = key_quote
        return payload
    if substrate == PrimitiveSubstrate.ACTOR_PORTRAITS:
        if focus_actor_id := _nonempty_runtime_text(
            getattr(primitive, "focus_actor_id", None)
        ):
            payload["focus_actor_id"] = focus_actor_id
        if actor_label := _nonempty_runtime_text(
            getattr(primitive, "actor_label", None)
        ):
            payload["actor_label"] = actor_label
        if goal_or_project := _nonempty_runtime_text(
            getattr(primitive, "goal_or_project", None)
        ):
            payload["goal_or_project"] = goal_or_project
        if stakes_or_fears := _nonempty_runtime_text(
            getattr(primitive, "stakes_or_fears", None)
        ):
            payload["stakes_or_fears"] = stakes_or_fears
        if operating_pressure := _nonempty_runtime_text(
            getattr(primitive, "operating_pressure", None)
        ):
            payload["operating_pressure"] = operating_pressure
        return payload
    if substrate == PrimitiveSubstrate.MECHANISMS:
        if mechanism_name := _nonempty_runtime_text(
            getattr(primitive, "mechanism_name", None)
        ):
            payload["mechanism_name"] = mechanism_name
        if operating_chain := _compact_runtime_string_list(
            getattr(primitive, "operating_chain", None)
        ):
            payload["operating_chain"] = operating_chain
        if inputs := _compact_runtime_string_list(getattr(primitive, "inputs", None)):
            payload["inputs"] = inputs
        if outputs := _compact_runtime_string_list(getattr(primitive, "outputs", None)):
            payload["outputs"] = outputs
        if failure_mode := _nonempty_runtime_text(
            getattr(primitive, "failure_mode", None)
        ):
            payload["failure_mode"] = failure_mode
        return payload
    if substrate == PrimitiveSubstrate.CONDITIONS:
        if condition_type := _nonempty_runtime_text(
            getattr(primitive, "condition_type", None)
        ):
            payload["condition_type"] = condition_type
        if condition_summary := _nonempty_runtime_text(
            getattr(primitive, "condition_summary", None)
        ):
            payload["condition_summary"] = condition_summary
        if active_tension := _nonempty_runtime_text(
            getattr(primitive, "active_tension", None)
        ):
            payload["active_tension"] = active_tension
        return payload
    if substrate == PrimitiveSubstrate.ARTIFACTS:
        if artifact_type := _nonempty_runtime_text(
            getattr(primitive, "artifact_type", None)
        ):
            payload["artifact_type"] = artifact_type
        if artifact_label := _nonempty_runtime_text(
            getattr(primitive, "artifact_label", None)
        ):
            payload["artifact_label"] = artifact_label
        if artifact_detail := _nonempty_runtime_text(
            getattr(primitive, "artifact_detail", None)
        ):
            payload["artifact_detail"] = artifact_detail
        return payload
    if substrate == PrimitiveSubstrate.READINGS:
        if reading_type := _nonempty_runtime_text(
            getattr(primitive, "reading_type", None)
        ):
            payload["reading_type"] = reading_type
        if subject_of_reading := _nonempty_runtime_text(
            getattr(primitive, "subject_of_reading", None)
        ):
            payload["subject_of_reading"] = subject_of_reading
        if attributed_to := _nonempty_runtime_text(
            getattr(primitive, "attributed_to", None)
        ):
            payload["attributed_to"] = attributed_to
        if reading_summary := _nonempty_runtime_text(
            getattr(primitive, "reading_summary", None)
        ):
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


def _build_host_policy_payload(
    narrator_profile: SeriesNarratorProfile,
) -> dict[str, Any]:
    return {
        "presence_mode": narrator_profile.presence_mode,
        "baseline_tone": narrator_profile.baseline_tone,
        "allowed_moves": list(narrator_profile.allowed_moves),
        "forbidden_moves": list(narrator_profile.forbidden_moves),
        "target_full_phase_scene_coverage_min": (
            narrator_profile.target_full_phase_scene_coverage_min
        ),
        "target_full_phase_scene_coverage_target": (
            narrator_profile.target_full_phase_scene_coverage_target
        ),
        "target_policy": "full_phase_scene_coverage",
        "scene_shaping_rule": (
            "Host moves should shape each scene's entry, pivot, and residue "
            "without turning the scene into standalone commentary."
        ),
        "pronoun_policy": {
            "allow_first_person_singular": False,
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
        "density_policy": {
            "prefer_full_phase_scene_shaping": True,
            "prefer_host_guidance_on_most_cards": True,
            "allow_light_scenes_to_use_one_or_two_phases": True,
        },
        "authorial_policy": {
            "analysis_mode": narrator_profile.analysis_mode,
            "analysis_density": narrator_profile.analysis_density,
            "quote_gloss_preference": narrator_profile.quote_gloss_preference,
            "clarifier_tolerance": narrator_profile.clarifier_tolerance,
            "comparative_aside_tolerance": narrator_profile.comparative_aside_tolerance,
            "wit_ceiling": narrator_profile.wit_ceiling,
            "target_authorial_passages_per_episode": (
                narrator_profile.target_authorial_passages_per_episode
            ),
            "authorial_passages_are_primary_exposition": True,
            "host_moves_are_primary_scene_shaping": True,
            "host_moves_are_secondary_exposition": True,
        },
    }


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
        referenced_item_ids.update(
            item_id
            for item_id in strategy_episode.authorial_contract.introduce_explanation_item_ids
            if item_id
        )
        referenced_item_ids.update(
            item_id
            for item_id in strategy_episode.authorial_contract.remind_explanation_item_ids
            if item_id
        )
    if architecture is not None:
        for section in architecture.sections:
            referenced_item_ids.update(
                item_id
                for item_id in (
                    explanation.item_id for explanation in section.term_explanations
                )
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
        referenced_actor_ids.update(
            actor_id
            for actor_id in strategy_episode.authorial_contract.introduce_actor_ids
            if actor_id
        )
        referenced_actor_ids.update(
            actor_id
            for actor_id in strategy_episode.authorial_contract.remind_actor_ids
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
        actor.actor_id
        for actor in strategy_episode.actor_arc_directives
        if actor.actor_id
    }
    actor_ids.update(
        actor_id
        for actor_id in strategy_episode.authorial_contract.introduce_actor_ids
        if actor_id
    )
    actor_ids.update(
        actor_id
        for actor_id in strategy_episode.authorial_contract.remind_actor_ids
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


def _build_window_passages(
    *,
    window_scene_cards: list[SceneCard],
    passage_lookup: dict[str, ExtractedPassage],
) -> list[dict[str, Any]]:
    ordered_passage_ids: list[str] = []
    seen_passage_ids: set[str] = set()
    for scene in window_scene_cards:
        for passage_id in scene.passage_ids:
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
        actor.actor_id
        for scene in window_scene_cards
        for actor in scene.actors
        if actor.actor_id
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
            actor.actor_id
            for actor in strategy_episode.actor_arc_directives
            if actor.actor_id
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
        str(scene_output.get("text", "") or "")
        for scene_output in completed_scene_outputs
    )
    tail_excerpt = _extract_previous_spoken_tail(continuity_text) or ""

    live_unresolved_questions: list[str] = []
    seen_questions: set[str] = set()
    for question in strategy_episode.unresolved_questions:
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
    withheld_scene_id = (last_scene.withhold_until or "").strip()
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
            "scene_function": last_scene.scene_function,
            "spine_relation": "",
            "withhold_until": last_scene.withhold_until,
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
) -> list[dict[str, Any]]:
    section_meta_by_id = {
        section.section_id: section for section in architecture.sections
    }
    host_moves_by_section = _build_host_moves_by_section(plan)
    scene_cards_by_section: dict[str, list[SceneCard]] = {}
    if plan is not None:
        for scene in plan.scene_cards:
            scene_cards_by_section.setdefault(scene.section_id, []).append(scene)
    payload_sections: list[dict[str, Any]] = []
    for prose_section in script.prose_sections:
        meta = section_meta_by_id.get(prose_section.section_id)
        if meta is None:
            continue
        section_scene_cards = scene_cards_by_section.get(prose_section.section_id, [])
        payload_sections.append(
            {
                "section_id": prose_section.section_id,
                "purpose": meta.purpose.value,
                "anchor": meta.section_anchor,
                "closure_mode": meta.closure_mode.value if meta.closure_mode else "",
                "scene_card_count": len(section_scene_cards),
                "projected_word_count": sum(
                    _project_scene_word_count(scene, words_per_minute=145.0)
                    for scene in section_scene_cards
                ),
                "structural_card_count": sum(
                    1
                    for scene in section_scene_cards
                    if _is_structural_scene_card(scene)
                ),
                "host_moves": host_moves_by_section.get(prose_section.section_id, []),
                "analysis_goal": meta.analysis_goal,
                "key_terms": list(meta.key_terms),
                "authorial_passages": [
                    passage.model_dump(mode="json")
                    for passage in meta.authorial_passages
                ],
                "term_explanations": [
                    explanation.model_dump(mode="json")
                    for explanation in meta.term_explanations
                ],
                "actor_explanations": [
                    explanation.model_dump(mode="json")
                    for explanation in meta.actor_explanations
                ],
                "host_presence_beats": [
                    beat.model_dump(mode="json") for beat in meta.host_presence_beats
                ],
                "text": prose_section.text,
                "actor_explanation_realizations": [
                    realization.model_dump(mode="json")
                    for realization in prose_section.actor_explanation_realizations
                ],
            }
        )
    return payload_sections


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
    host_moves_by_section = _build_host_moves_by_section(plan)
    payload_sections: list[dict[str, Any]] = []
    for prose_section in script.prose_sections:
        architecture_section = architecture_section_lookup.get(prose_section.section_id)
        payload_sections.append(
            {
                "section_id": prose_section.section_id,
                "purpose": architecture_section.purpose.value
                if architecture_section
                else "",
                "anchor": architecture_section.section_anchor
                if architecture_section
                else "",
                "closure_mode": (
                    architecture_section.closure_mode.value
                    if architecture_section and architecture_section.closure_mode
                    else ""
                ),
                "movement_goal": prose_section.movement_goal,
                "scene_card_ids": list(prose_section.scene_card_ids),
                "host_moves": host_moves_by_section.get(prose_section.section_id, []),
                "analysis_goal": (
                    architecture_section.analysis_goal if architecture_section else ""
                ),
                "key_terms": (
                    list(architecture_section.key_terms)
                    if architecture_section is not None
                    else []
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
                "host_presence_beats": (
                    [
                        beat.model_dump(mode="json")
                        for beat in architecture_section.host_presence_beats
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
                "analysis_goal": (
                    architecture_section.analysis_goal if architecture_section else ""
                ),
                "key_terms": (
                    list(architecture_section.key_terms)
                    if architecture_section is not None
                    else []
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
                "host_presence_beats": (
                    [
                        beat.model_dump(mode="json")
                        for beat in architecture_section.host_presence_beats
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
        missing_ids = [
            insight_id
            for insight_id in referenced_ids
            if insight_id not in seen_existing
        ]
        missing_total += len(missing_ids)
        if missing_ids:
            injected_total += len(missing_ids)
            adjusted.append(
                beat.model_copy(update={"insight_ids": [*existing_ids, *missing_ids]})
            )
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
            passage_keep_fraction = keep_fraction_by_passage_id.get(
                passage_id, keep_fraction
            )
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
        word_budget = max(
            1, math.floor(len(cand.get("text", "").split()) * passage_keep_fraction)
        )
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
            -passage.quotability_score,
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


def _passage_similarity_tokens(
    passage: ExtractedPassage, *, max_chars: int = 1200
) -> set[str]:
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
                -p.quotability_score,
                p.passage_id,
            ),
        )

    lambda_weight = _clamp(lambda_weight, 0.0, 1.0)
    source_penalty_weight = _clamp(source_penalty_weight, 0.0, 1.0)
    token_sets = [_passage_similarity_tokens(passage) for passage in passages]
    base_scores = [base_score_fn(passage) for passage in passages]
    source_groups = [
        source_group_fn(passage) if source_group_fn is not None else None
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
    payload["analysis"] = (
        analysis.model_dump(mode="json") if analysis is not None else None
    )
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
    context["chapter_analysis"] = (
        analysis.model_dump(mode="json") if analysis is not None else None
    )
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
                    payload["summary_text"] = (
                        passage.trimmed_text.strip() or passage.text
                    )
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


_CROSS_BOOK_PAIR_RELATIONSHIP_PRIORITY = {
    SynthesisTag.CONTRADICTS: 0,
    SynthesisTag.CONTEXTUALIZES: 1,
    SynthesisTag.EXEMPLIFIES: 2,
}


def _prioritize_cross_book_pairs(pairs: list[PassagePair]) -> list[PassagePair]:
    prioritized_pairs = [
        pair
        for pair in pairs
        if pair.relationship in _CROSS_BOOK_PAIR_RELATIONSHIP_PRIORITY
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
    return (0.7 * float(passage.relevance_score)) + (
        0.3 * float(passage.quotability_score)
    )


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
    axis_order = sorted(
        axes, key=lambda axis: (-axis.theme_importance_score, axis.axis_id)
    )
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

    selected_by_axis: dict[str, list[ExtractedPassage]] = {
        axis_id: [] for axis_id in axis_ids
    }
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
    remaining_slots = max(
        0, cap - sum(len(items) for items in selected_by_axis.values())
    )
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
                        -passage.quotability_score,
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
            source_key_axes.setdefault(_synthesis_source_key(passage), []).append(
                axis_id
            )
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
                episode_numbers_by_id.setdefault(assigned_id, []).append(
                    assignment.episode_number
                )
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
        for merged_narrative_id, episode_numbers in sorted(
            episode_numbers_by_id.items()
        )
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
    "narrative_importance_score",
}
_PRIMITIVE_ANNOTATION_ACTOR_KEEP_FIELDS = {
    "actor_id",
    "display_name",
    "aliases",
    "actor_type",
}

_CONSOLIDATION_PRIMITIVE_DROP_FIELDS = {
    "family",
    "core_passage_ids",
    "support_passage_ids",
    "actor_tags",
    "institution_tags",
}


def _compact_primitives_for_consolidation(
    artifact: SynthesisPrimitivesArtifact,
) -> dict[str, Any]:
    primitives_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        compacted_items: list[dict[str, Any]] = []
        for primitive in artifact.primitives_by_family.get(family, []):
            payload = {
                key: value
                for key, value in primitive.model_dump(mode="json").items()
                if key not in _CONSOLIDATION_PRIMITIVE_DROP_FIELDS
            }
            if "candidate_readings" in payload:
                payload["candidate_readings"] = [
                    {
                        "label": reading.get("label", ""),
                        "claim": reading.get("claim", reading.get("summary", "")),
                        "emphasizes": reading.get("emphasizes", ""),
                        "downplays": reading.get("downplays", ""),
                    }
                    for reading in payload["candidate_readings"]
                ]
            compacted_items.append(payload)
        primitives_by_family[family] = compacted_items
    return {
        "project_id": artifact.project_id,
        "primitives_by_family": primitives_by_family,
        "quality_score": artifact.quality_score,
        "quality_notes": list(artifact.quality_notes),
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


_ENRICHED_PRIMITIVE_MODEL_BY_FAMILY: dict[str, type[SynthesisPrimitiveBase]] = {
    "epochal_turns": EpochalTurnPrimitive,
    "decisions_and_nondecisions": DecisionPrimitive,
    "set_piece_scenes": SetPieceScenePrimitive,
    "telling_details": SynthesisPrimitive,
    "human_costs": HumanCostPrimitive,
    "character_engines": CharacterEnginePrimitive,
    "coalitions_and_fault_lines": CoalitionFaultLinePrimitive,
    "systems_and_operating_logics": SystemsOperatingLogicPrimitive,
    "misreadings_and_fantasies": SynthesisPrimitive,
    "contested_explanations": ContestedExplanationPrimitive,
    "perspective_windows": SynthesisPrimitive,
    "moral_traps": MoralTrapPrimitive,
    "afterlives": SynthesisPrimitive,
    "recurring_images_and_symbols": SynthesisPrimitive,
    "ironies_and_reversals": IronyReversalPrimitive,
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
                passage.full_text.strip()
                or passage.trimmed_text.strip()
                or passage.text.strip()
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
            if (
                not passage_id
                or passage_id in seen_support_ids
                or passage_id in seen_core_ids
            ):
                continue
            seen_support_ids.add(passage_id)
            passage = passage_lookup.get(passage_id)
            if passage is None:
                continue
            full_text = (
                passage.full_text.strip()
                or passage.trimmed_text.strip()
                or passage.text.strip()
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
                reference_count_by_index[idx] = (
                    reference_count_by_index.get(idx, 0) + 1
                )

        final_pool_indices: set[int]
        if len(references) == 1:
            final_pool_indices = set(next(iter(selected_indices_by_reference.values())))
        else:
            shared_budget = max(
                1,
                math.floor(
                    total_word_count * trim_profile.shared_passage_keep_fraction
                ),
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
                [
                    idx
                    for idx in reference_count_by_index
                    if idx not in final_pool_indices
                ],
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
        if any(
            insight_id in assignment.insight_ids for insight_id in thread.insight_ids
        )
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
        passage_id for beat in plan.beats for passage_id in beat.passage_ids
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
                "missing_passage_ids": sorted(
                    set(insight.passage_ids) - planned_passage_ids
                ),
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
        passage_id for beat in plan.beats for passage_id in beat.passage_ids
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
        "has_issues": insight_realization["has_issues"]
        or merged_realization["has_issues"],
        "problem_count": insight_realization["problem_count"]
        + merged_realization["problem_count"],
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
        "section_id",
        "invalid_section_ids",
        "invalid_scene_primitives",
        "missing_section_ids",
        "missing_core_primitive_ids",
        "invalid_dropped_support_ids",
        "expected_episode_number",
        "actual_episode_number",
    ):
        value = data.get(key)
        if value:
            feedback[key] = value
    return feedback


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _resolve_axis_relevance(
    axis: ThematicAxis, book_ids: list[str]
) -> dict[str, float]:
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
            "plan_utilization_ratio": _ratio(
                len(axis_retained & planned_ids), len(axis_retained)
            ),
            "citation_utilization_ratio": _ratio(
                len(axis_retained & cited_ids), len(axis_retained)
            ),
        }

    per_book: dict[str, Any] = {}
    for book in books:
        book_retained = {
            pid
            for pid in retained_ids
            if passage_by_id.get(pid) is not None
            and passage_by_id[pid].book_id == book.book_id
        }
        per_book[book.book_id] = {
            "title": book.title,
            "retained_count": len(book_retained),
            "planned_count": len(book_retained & planned_ids),
            "cited_count": len(book_retained & cited_ids),
            "plan_utilization_ratio": _ratio(
                len(book_retained & planned_ids), len(book_retained)
            ),
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
    beat_scene_by_id = {
        beat.beat_id: beat.scene_id for beat in plan.beats if beat.scene_id
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
                "missing_passage_ids": sorted(
                    assigned_passage_ids - observed_passage_ids
                ),
            }
        )
    insight_issues = [
        item for item in insight_results if item["status"] in {"weak", "zero"}
    ]

    planned_pairs = {
        tuple(sorted((item.from_book_id, item.to_book_id)))
        for item in plan.cross_references
        if item.from_book_id
        and item.to_book_id
        and item.from_book_id != item.to_book_id
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
    coverage_ratio = (
        len(covered_pairs) / planned_pair_count if planned_pair_count > 0 else 1.0
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
        len(covered_scene_ids) / len(planned_scene_ids) if planned_scene_ids else 1.0
    )
    anchor_scene_ids = [scene_id for scene_id in plan.anchor_scene_ids if scene_id]
    missing_anchor_scene_ids = [
        scene_id
        for scene_id in anchor_scene_ids
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
                scene_id
                for scene_id in planned_scene_ids
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
                {"book_ids": [a, b]} for a, b in sorted(planned_pairs - observed_pairs)
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
async def _stage_log(
    run_logger: RunLogger, stage_name: str, project_dir: Path, **input_summary
):
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
        overlay.append(
            "Linger slightly on the most reflective clause without sounding theatrical."
        )

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
    tts_model_name: str | None,
    base_instructions: str | None,
) -> list[RenderSegment]:
    hints = segment.speech_hints
    render_speed = _resolve_spoken_render_speed(hints.pace, speed)
    if hints.render_strategy == "slow_clause":
        render_speed = _resolve_spoken_render_speed("slower", render_speed)

    text_parts = _split_render_text(segment.text, hints)
    degradations = _segment_hint_degradations(hints, tts_provider, tts_model_name)
    supports_instructions = _supports_segment_tts_instructions(
        tts_provider, tts_model_name
    )
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
                segment_id=segment.segment_id
                if is_single_part
                else f"{segment.segment_id}_{idx}",
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
    voice_id: str = "fable",
    speed: float = 1.0,
    words_per_minute: int = 130,
    base_instructions: str | None = None,
    tts_model_name: str | None = "tts-1-hd",
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

    for section in spoken_script.sections:
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
                tts_model_name=tts_model_name,
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
    return units


def _script_total_word_count(script: EpisodeScript) -> int:
    return sum(len(text.split()) for _, text in _script_text_units(script))


def _estimate_duration_seconds_from_words(
    word_count: int, words_per_minute: float
) -> int:
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

    return {scene.scene_id: floor_targets[idx] for idx, scene in enumerate(scene_cards)}


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
_ARGUMENT_SCENE_FUNCTIONS: frozenset[str] = frozenset(
    {"mechanism", "landing", "callback", "afterlife"}
)
_ARGUMENT_ROLES: frozenset[str] = frozenset({"implication", "perspective_shift"})
_ACTION_BUCKET_ROLES: frozenset[str] = frozenset(
    {"shock", "action", "fallout", "reveal", "reversal", "stage_choice"}
)
_STRUCTURAL_SCENE_FUNCTIONS: frozenset[str] = frozenset(
    {"mechanism", "turn", "landing", "callback", "afterlife"}
)
_GROUNDING_SCENE_FUNCTIONS: frozenset[str] = frozenset({"scene", "hinge"})
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
                allocations[idx] = (
                    remaining_total * positive_weights[idx] / active_weight
                )
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
        overflow_weights
        if stranded_total > 0 and sum(overflow_weights) > 0
        else fallback_weights
    )
    redistribution_weight = sum(redistribution_weights)
    if redistribution_weight <= 0:
        equal_share = stranded_total / len(allocations)
        return [allocation + equal_share for allocation in allocations]
    return [
        allocation
        + stranded_total * redistribution_weights[idx] / redistribution_weight
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
            raise ValueError(
                "cannot round allocations to target without non-positive durations"
            )

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
    for fact in scene.must_land_facts:
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
    return sum(
        1 for phase in _HOST_MOVE_PHASE_ORDER if getattr(scene.host_moves, phase)
    )


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
        *scene.must_land_facts,
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
    if any(
        pattern.search(normalized_note)
        for pattern in _HOST_NOTE_EDITORIAL_CONTROL_PATTERNS
    ):
        flags.append("editorial_control_phrase")
    if any(
        pattern.search(normalized_note)
        for pattern in _HOST_NOTE_EPISODE_MANAGEMENT_PATTERNS
    ):
        flags.append("episode_management_phrase")
    if any(
        pattern.search(normalized_note)
        for pattern in _HOST_NOTE_ABSTRACT_TARGET_PATTERNS
    ):
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
    meta_note_phase_ids: list[str] = []
    editorial_scaffolding_phase_ids: list[str] = []
    first_person_plural_note_scene_ids: list[str] = []
    disallowed_move_scene_ids: list[str] = []
    personalized_scene_ids: list[str] = []
    scenes_with_1_phase: list[str] = []
    scenes_with_2_phases: list[str] = []
    scenes_with_3_phases: list[str] = []
    full_phase_scene_ids: list[str] = []
    total_host_phase_count = 0
    allowed_moves = set(narrator_profile.allowed_moves)

    for scene in scene_cards:
        anchor_tokens = _scene_host_anchor_tokens(scene)
        phase_bucket_count = _scene_host_phase_bucket_count(scene)
        if phase_bucket_count == 1:
            scenes_with_1_phase.append(scene.scene_id)
        elif phase_bucket_count == 2:
            scenes_with_2_phases.append(scene.scene_id)
        elif phase_bucket_count == 3:
            scenes_with_3_phases.append(scene.scene_id)
            full_phase_scene_ids.append(scene.scene_id)

        scene_is_personalized = False
        scene_has_first_person_plural_note = False
        scene_has_disallowed_move = False
        for phase, cue in _iter_host_move_cues(scene):
            total_host_phase_count += 1
            counts_by_phase[phase] = counts_by_phase.get(phase, 0) + 1
            counts_by_type[cue.move_type] = counts_by_type.get(cue.move_type, 0) + 1
            normalized_note = " ".join(cue.note.lower().split())
            if normalized_note and re.search(r"\b(we|our|us)\b", normalized_note):
                scene_has_first_person_plural_note = True
            if cue.address_mode in {"we", "you", "i"}:
                scene_is_personalized = True
            if cue.move_type not in allowed_moves:
                scene_has_disallowed_move = True
            if _host_move_note_looks_meta(cue.note):
                meta_note_phase_ids.append(f"{scene.scene_id}:{phase}")
            if _host_move_editorial_scaffolding_flags(
                cue.note,
                anchor_tokens=anchor_tokens,
            ):
                editorial_scaffolding_phase_ids.append(f"{scene.scene_id}:{phase}")
        if scene_has_first_person_plural_note:
            first_person_plural_note_scene_ids.append(scene.scene_id)
        if scene_is_personalized:
            personalized_scene_ids.append(scene.scene_id)
        if scene_has_disallowed_move:
            disallowed_move_scene_ids.append(scene.scene_id)

    warnings: list[str] = []
    scene_card_count = len(scene_cards)
    full_phase_scene_count = len(full_phase_scene_ids)
    full_phase_scene_coverage = full_phase_scene_count / max(1, scene_card_count)
    if meta_note_phase_ids:
        warnings.append(
            f"host_phase_meta_notes_detected: {_preview_ids(meta_note_phase_ids)}"
        )
    if editorial_scaffolding_phase_ids:
        warnings.append(
            "host_phase_editorial_scaffolding_detected: "
            f"{_preview_ids(sorted(set(editorial_scaffolding_phase_ids)))}"
        )
    if disallowed_move_scene_ids:
        warnings.append(
            "host_move_allowed_move_mismatch: "
            f"{_preview_ids(disallowed_move_scene_ids)}"
        )
    if (
        full_phase_scene_coverage
        < narrator_profile.target_full_phase_scene_coverage_min
    ):
        warnings.append(
            "host_full_phase_coverage_below_min: "
            f"{full_phase_scene_count}/{scene_card_count} "
            f"({full_phase_scene_coverage:.2f} < "
            f"{narrator_profile.target_full_phase_scene_coverage_min:.2f})"
        )
    orient_callback_count = counts_by_type.get("orient", 0) + counts_by_type.get(
        "callback", 0
    )
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
    host_shaped_scene_count = (
        len(scenes_with_1_phase) + len(scenes_with_2_phases) + len(scenes_with_3_phases)
    )
    if host_shaped_scene_count and len(personalized_scene_ids) < max(
        1, int(host_shaped_scene_count * 0.2)
    ):
        warnings.append(
            "host_move_underpersonalized: "
            f"{len(personalized_scene_ids)}/{host_shaped_scene_count} host-shaped "
            "scenes use explicit `we` or `you` address"
        )

    diagnostics = {
        "target_full_phase_scene_coverage_min": (
            narrator_profile.target_full_phase_scene_coverage_min
        ),
        "target_full_phase_scene_coverage_target": (
            narrator_profile.target_full_phase_scene_coverage_target
        ),
        "target_policy": "full_phase_scene_coverage",
        "scene_card_count": scene_card_count,
        "host_shaped_scene_count": host_shaped_scene_count,
        "full_phase_scene_count": full_phase_scene_count,
        "full_phase_scene_coverage": full_phase_scene_coverage,
        "total_host_phase_count": total_host_phase_count,
        "counts_by_type": counts_by_type,
        "counts_by_phase": counts_by_phase,
        "scenes_with_1_phase_count": len(scenes_with_1_phase),
        "scenes_with_2_phases_count": len(scenes_with_2_phases),
        "scenes_with_3_phases_count": len(scenes_with_3_phases),
        "full_phase_scene_ids": full_phase_scene_ids,
        "meta_note_phase_ids": meta_note_phase_ids,
        "editorial_scaffolding_phase_ids": sorted(
            set(editorial_scaffolding_phase_ids)
        ),
        "first_person_plural_note_scene_ids": first_person_plural_note_scene_ids,
        "personalized_scene_ids": personalized_scene_ids,
        "disallowed_move_scene_ids": disallowed_move_scene_ids,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return diagnostics, warnings


def _no_actor_bucket(scene: SceneCardDraft | SceneCard) -> str:
    if (
        scene.scene_function in _ARGUMENT_SCENE_FUNCTIONS
        or scene.scene_role in _ARGUMENT_ROLES
    ):
        return "argument"
    if scene.scene_function == "turn" or scene.scene_role in _ACTION_BUCKET_ROLES:
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
        _occurrence_weight(strategy_episode, occurrence_id)
        for occurrence_id in occurrence_ids
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
    occurrence_seconds_by_id = dict(
        zip(occurrence_ids, occurrence_allocations, strict=True)
    )

    raw_scene_seconds_by_index: dict[int, float] = {}
    for occurrence_id in occurrence_ids:
        occurrence_scene_entries = [
            (idx, scene)
            for idx, scene in enumerate(scene_cards)
            if (_scene_occurrence_id(scene) or "__unassigned__") == occurrence_id
        ]
        scene_weights = [
            _scene_importance_weight(scene) for _, scene in occurrence_scene_entries
        ]
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

    raw_scene_seconds = [
        raw_scene_seconds_by_index[idx] for idx in range(len(scene_cards))
    ]

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
    return sum(
        len((getattr(section, "text", "") or "").split()) for section in prose_sections
    )


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

    returned_sections = [
        (
            str(getattr(section_output, "section_id", "") or ""),
            [
                str(scene_id)
                for scene_id in list(
                    getattr(section_output, "scene_card_ids", []) or []
                )
            ],
        )
        for section_output in prose_sections
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

    normalized: list[dict[str, Any]] = []
    for section_output in prose_sections:
        section_id = str(section_output.section_id)
        payload = {
            "section_id": section_id,
            "scene_card_ids": [
                str(scene_id)
                for scene_id in list(
                    getattr(section_output, "scene_card_ids", []) or []
                )
            ],
            "movement_goal": str(section_output.movement_goal),
            "text": str(section_output.text),
            "source_book_ids": list(
                getattr(section_output, "source_book_ids", []) or []
            ),
            "actor_explanation_realizations": [
                ActorExplanationRealization.model_validate(
                    realization.model_dump(mode="json")
                    if hasattr(realization, "model_dump")
                    else realization
                )
                for realization in getattr(
                    section_output, "actor_explanation_realizations", []
                )
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


_ENDING_MARKER_RE = re.compile(
    r"\b(?:in the end|ultimately|finally|the verdict)\b", re.IGNORECASE
)
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
        sentence.strip().lower() != driving_problem
        for sentence in question_sentences[1:]
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
            passage_id
            for passage_id in scene.passage_ids
            if passage_id not in passage_lookup
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
        seen_primitive_ids = seen_primitive_ids_by_passage_id.setdefault(
            passage_id, set()
        )
        if primitive_id in seen_primitive_ids:
            continue
        seen_primitive_ids.add(primitive_id)
        query_parts = query_parts_by_passage_id.setdefault(
            passage_id, [episode_query_text]
        )
        query_parts.append(primitive.title)
        query_parts.extend(_primitive_substrate_text_fragments(primitive))
        if primitive.timeframe:
            query_parts.append(primitive.timeframe)
        if primitive.geography:
            query_parts.append(primitive.geography)
    return {
        passage_id: " ".join(
            part.strip() for part in parts if part and part.strip()
        ).strip()
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
) -> EpisodeArchitecture:
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
        section.section_id
        for section in architecture.sections
        if not section.must_stage_beats
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
    return architecture


def _filter_primitive_ids_by_architecture(
    strategy_episode: StrategyEpisode,
    episode: EpisodeArchitecture,
) -> dict[str, list[str]]:
    section_primitive_ids = set(_ordered_architecture_primitive_ids(episode))
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
    selected_core_primitive_ids = list(
        strategy_episode.episode_spine.core_primitive_ids
    )
    selected_support_primitive_ids = list(
        strategy_episode.episode_spine.support_primitive_roles.keys()
    )
    selected_recall_primitive_ids = list(
        strategy_episode.episode_spine.recall_primitive_ids
    )
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
    warnings: list[str] = []
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
            "architecture_missing_recall_primitives: "
            f"{_preview_ids(missing_recall_primitive_ids)}"
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

    registry_by_id = {
        item.item_id: item for item in (series_explanation_registry or [])
    }
    missing_payoff_item_ids: list[str] = []
    for item_id in strategy_episode.authorial_contract.introduce_explanation_item_ids:
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
        for item_id in strategy_episode.authorial_contract.remind_explanation_item_ids
        if any(
            plan.item_id == item_id and plan.stage == "define"
            for section in architecture.sections
            for plan in section.term_explanations
        )
    ]
    if reminder_redefined_item_ids:
        warnings.append(
            f"reminder_item_redefined: {_preview_ids(reminder_redefined_item_ids)}"
        )

    host_presence_counts: dict[str, int] = {}
    host_presence_counts_by_section: dict[str, int] = {}
    for section in architecture.sections:
        host_presence_counts_by_section[section.section_id] = len(
            section.host_presence_beats
        )
        for beat in section.host_presence_beats:
            host_presence_counts[beat.kind] = host_presence_counts.get(beat.kind, 0) + 1
    underdense_host_presence_sections = [
        section_id
        for section_id, count in host_presence_counts_by_section.items()
        if count < 3
    ]
    overdense_host_presence_sections = [
        section_id
        for section_id, count in host_presence_counts_by_section.items()
        if count > 6
    ]
    if underdense_host_presence_sections:
        warnings.append(
            "host_presence_density_below_target: "
            f"{_preview_ids(underdense_host_presence_sections)}"
        )
    if overdense_host_presence_sections:
        warnings.append(
            "host_presence_density_above_target: "
            f"{_preview_ids(overdense_host_presence_sections)}"
        )
    callback_orientation_count = host_presence_counts.get(
        "callback", 0
    ) + host_presence_counts.get("orientation", 0)
    active_host_presence_count = sum(host_presence_counts.values())
    if active_host_presence_count and callback_orientation_count >= max(
        2, active_host_presence_count - 1
    ):
        warnings.append(
            "host_presence_skewed_to_orientation_callback: "
            f"orientation={host_presence_counts.get('orientation', 0)} "
            f"callback={host_presence_counts.get('callback', 0)} "
            f"clarify={host_presence_counts.get('clarify', 0)} "
            f"contrast={host_presence_counts.get('contrast', 0)} "
            f"evaluate={host_presence_counts.get('evaluate', 0)}"
        )

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

    overloaded_section_ids = [
        section.section_id
        for section in architecture.sections
        if (
            (
                any(
                    plan.stage in {"define", "payoff"}
                    for plan in section.term_explanations
                )
                and section.approx_runtime_minutes >= 8.0
            )
            or (
                len(section.key_terms) >= 4
                and len(section.authorial_passages) <= 1
                and len(section.term_explanations) >= 2
            )
            or (
                len(section.primitive_ids) >= 4
                and len(section.authorial_passages) <= 1
                and section.analysis_goal
            )
        )
    ]
    if overloaded_section_ids:
        warnings.append(
            f"explanation_section_overloaded: {_preview_ids(overloaded_section_ids)}"
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
        "target_authorial_passages_per_episode": target_authorial_passages,
        "authorial_passage_count": authorial_passage_count,
        "host_presence_counts": host_presence_counts,
        "host_presence_counts_by_section": host_presence_counts_by_section,
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

    section_order = {
        section.section_id: idx for idx, section in enumerate(architecture.sections)
    }
    scene_counts_by_section = {
        section.section_id: 0 for section in architecture.sections
    }
    highest_section_index = -1
    invalid_section_ids: list[str] = []
    multi_turn_sections: list[str] = []
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
        if len(scene.must_land_facts) > 5:
            multi_turn_sections.append(scene.scene_id)

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
        section_id
        for section_id, count in scene_counts_by_section.items()
        if count == 0
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

    if multi_turn_sections:
        raise ComplianceViolationError(
            "Episode plan overloaded scene cards with too many must_land_facts for "
            f"episode {plan.episode_number}: {_preview_ids(sorted(set(multi_turn_sections)))}",
            data={
                "issue": "scene_card_overloaded",
                "episode_number": plan.episode_number,
                "scene_ids": sorted(set(multi_turn_sections)),
                "instruction": "Keep scene cards lean. Split overloaded beats instead of packing more than five required facts into one scene.",
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
                actor.actor_id == explanation.actor_id
                and actor.explanation_stage == "introduce"
                for actor in earliest_scene.actors
            )
            if not tagged_earliest_scene:
                misplaced_introductions.append(
                    f"{explanation.actor_id}:{earliest_scene.scene_id}"
                )
    warnings: list[str] = []
    if missing_links:
        warnings.append(
            "missing_actor_explanation_scene_links:"
            + ",".join(missing_links)
        )
    if misplaced_introductions:
        warnings.append(
            "late_actor_introduction_scene_links:"
            + ",".join(misplaced_introductions)
        )
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
        part.strip()
        for part in (driving_question, thematic_focus)
        if part and part.strip()
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
        "actor_linked_primitive_count": sum(
            1 for primitive in primitives if primitive.actor_ids
        ),
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
    return scene.scene_function in _STRUCTURAL_SCENE_FUNCTIONS


def _is_human_grounding_scene_card(scene: SceneCardDraft | SceneCard) -> bool:
    return (
        scene.scene_function in _GROUNDING_SCENE_FUNCTIONS
        and scene.scene_role in _GROUNDING_SCENE_ROLES
        and bool(scene.actors)
        and bool((scene.entry_image or "").strip())
        and bool((scene.observable_detail or "").strip())
    )


def _build_scene_function_counts(
    scene_cards: list[SceneCardDraft | SceneCard],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scene in scene_cards:
        counts[scene.scene_function] = counts.get(scene.scene_function, 0) + 1
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
        if _is_structural_scene_card(scene)
        and not (scene.observable_detail or "").strip()
    ]
    warnings: list[str] = []
    if missing_entry_image:
        warnings.append(
            f"structural_card_missing_entry_image: {_preview_ids(missing_entry_image)}"
        )
    if missing_observable_detail:
        warnings.append(
            "structural_card_missing_observable_detail: "
            f"{_preview_ids(missing_observable_detail)}"
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
_SCENE_DETAIL_PRIMITIVE_SUBSTRATES = frozenset({"events", "utterances", "artifacts"})
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
        function.value
        for primitive in primitive_pool
        for function in primitive.functions
    }
    warnings: list[str] = []
    if not primitive_pool:
        return warnings

    if substrates_present.isdisjoint(_SPINE_PRIMITIVE_SUBSTRATES):
        warnings.append(
            "primitive_pool_missing_spine: "
            "episode primitive pool lacks event/act grounding"
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

    support_primitive_count = len(
        strategy_episode.episode_spine.support_primitive_roles
    ) + len(strategy_episode.episode_spine.recall_primitive_ids)
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
        if len(scene.must_land_facts) > max(5, primitive_max + 3)
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
        warnings.append(
            "host_voice_absent: no scene cards carry planned host phase shaping"
        )
    return warnings


def _normalize_section_text_tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return {token for token in normalized.split() if len(token) >= 4}


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
            "editorial_host_note_count": 0,
            "sections_with_editorial_host_note_pressure": [],
            "editorial_host_note_examples": [],
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
            phase_note = " ".join(cue.note for cue in phase_cues if cue.note)
            note_tokens = _normalize_section_text_tokens(phase_note)
            approx_realized = not note_tokens or bool(note_tokens & section_tokens)
            editorial_scaffolding_flags = _host_move_editorial_scaffolding_flags(
                phase_note,
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
                    "host_note": phase_note,
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
    editorial_phase_entries = [
        row
        for row in phase_trace
        if row.get("editorial_scaffolding_flags")
    ]
    editorial_host_note_examples = [
        {
            "scene_id": str(row["scene_id"]),
            "section_id": str(row["section_id"]),
            "phase": str(row["phase"]),
            "host_note": str(row["host_note"]),
            "editorial_scaffolding_flags": list(
                row.get("editorial_scaffolding_flags", [])
            ),
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
        "editorial_host_note_count": len(editorial_phase_entries),
        "sections_with_editorial_host_note_pressure": sorted(
            {str(row["section_id"]) for row in editorial_phase_entries}
        ),
        "editorial_host_note_examples": editorial_host_note_examples,
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
                    scene.beat_change,
                    *scene.must_land_facts,
                ]
            )
            for passage_id in scene.passage_ids:
                if (
                    passage_id in section.priority_core_passage_ids
                    and passage_id not in used_priority_core_passage_ids
                ):
                    used_priority_core_passage_ids.append(passage_id)

        section_warnings: list[str] = []
        if section.listener_tension and not _section_text_realized(
            section.listener_tension,
            scene_realization_texts,
        ):
            section_warnings.append(
                f"listener_tension_not_realized: {section.section_id}"
            )
        if section.section_turn and not _section_text_realized(
            section.section_turn,
            scene_realization_texts,
        ):
            section_warnings.append(f"section_turn_not_realized: {section.section_id}")
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
                "section_scene_card_load_high: "
                f"{section.section_id} -> {scene_card_count} cards"
            )
        if projected_word_count > 2200:
            section_warnings.append(
                "section_projected_word_count_high: "
                f"{section.section_id} -> {projected_word_count} projected words"
            )

        section_reports.append(
            {
                "section_id": section.section_id,
                "section_anchor": section.section_anchor,
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


def _scene_counts_toward_spine(
    scene: SceneCardDraft | SceneCard,
    strategy_episode: StrategyEpisode,
) -> bool:
    return bool(scene.must_land_facts or scene.beat_change.strip())


def _build_spine_plan_diagnostics(
    *,
    strategy_episode: StrategyEpisode,
    plan: EpisodePlanDraft | EpisodePlan,
) -> dict[str, Any]:
    scene_cards = list(plan.scene_cards)
    total_scene_count = len(scene_cards)
    spine_scene_cards = [
        scene
        for scene in scene_cards
        if _scene_counts_toward_spine(scene, strategy_episode)
    ]
    scene_share = (
        len(spine_scene_cards) / total_scene_count if total_scene_count else 0.0
    )

    total_word_weight = sum(
        max(1, len(scene.beat_change.split())) for scene in scene_cards
    )
    spine_word_weight = sum(
        max(1, len(scene.beat_change.split())) for scene in spine_scene_cards
    )
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
    diagnostics = {
        "core_scene_share": round(scene_share, 4),
        "core_word_share": round(word_share, 4),
        "ending_alignment_pass": ending_alignment_pass,
        "secondary_chain_detected": secondary_chain_detected,
        "support_takeover_detected": support_takeover_detected,
        "recall_takeover_detected": False,
        "new_load_bearing_question_detected": False,
        "second_ending_detected": False,
        "host_phase_count": sum(
            _scene_host_phase_bucket_count(scene) for scene in scene_cards
        ),
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
        self.primitive_function_tagging_agents = {
            substrate: PrimitiveFunctionTaggingAgent(
                self.llm,
                substrate=substrate,
                max_retry_attempts=_retries(
                    f"primitive_function_tagging_{substrate}"
                ),
            )
            for substrate in PRIMITIVE_SUBSTRATES
        }
        self.narrative_strategy_agent = NarrativeStrategyAgent(
            self.llm, max_retry_attempts=_retries("narrative_strategy")
        )
        self.episode_architecture_agent = EpisodeArchitectureAgent(
            self.llm, max_retry_attempts=_retries("episode_architecture")
        )
        self.episode_planning_agent = EpisodePlanningAgent(
            self.llm, max_retry_attempts=_retries("episode_planning")
        )
        self.writing_agent = WritingAgent(
            self.llm, max_retry_attempts=_retries("episode_writing")
        )
        self.writing_agent_no_citations = WritingAgentNoCitations(
            self.llm,
            max_retry_attempts=_retries("episode_writing"),
        )
        self.grounding_agent = GroundingValidationAgent(
            self.llm, max_retry_attempts=_retries("grounding_validation")
        )
        self.repair_agent = RepairAgent(self.llm, max_retry_attempts=_retries("repair"))
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

            axes, actor_metadata, actor_metrics = await self._decompose_theme(
                project, project_dir
            )
            corpus = await self._extract_passages(project, axes, project_dir)
            synthesis_primitives, synthesis_actor_metrics = await self._map_synthesis(
                project,
                corpus,
                project_dir,
                actor_metadata,
            )
            actor_metrics["synthesis_primitives"] = synthesis_actor_metrics.get(
                "primitives", {}
            )

            if (
                synthesis_primitives.quality_score
                < pipeline_config.synthesis_quality_threshold
            ):
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

            strategy, strategy_actor_metrics = await self._choose_narrative_strategy(
                project,
                synthesis_primitives,
                project_dir,
                actor_metadata,
            )
            actor_metrics["narrative_strategy"] = strategy_actor_metrics
            synthesis_map = await self._materialize_selected_primitives(
                project,
                synthesis_primitives,
                strategy,
                project_dir,
            )
            project = self._resolve_episode_count_from_strategy(project, strategy)
            _save_json(project_dir / "thematic_project.json", project)

            project = project.model_copy(update={"status": ProjectStatus.PLANNING})
            (
                episode_architectures,
                architecture_actor_metrics,
            ) = await self._build_episode_architectures(
                project,
                synthesis_map,
                strategy,
                corpus,
                project_dir,
                actor_metadata,
            )
            actor_metrics["episode_architecture"] = architecture_actor_metrics
            episode_plans, planning_actor_metrics = await self._plan_series(
                project,
                synthesis_map,
                strategy,
                episode_architectures,
                corpus,
                project_dir,
                actor_metadata,
            )
            actor_metrics["episode_planning"] = planning_actor_metrics

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
            host_policy = _build_host_policy_payload(strategy.narrator_profile)
            retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            strategy_episode_by_number = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            architecture_by_number = {
                episode.episode_number: episode
                for episode in episode_architectures
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
                    host_policy,
                    retained_primitive_lookup,
                    sem,
                    spoken_sem,
                    strategy.series_explanation_registry,
                )
                for plan in episode_plans
            ]
            ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)

            spoken_scripts: list[tuple[int, SpokenScript]] = []
            episode_errors = [
                result for result in ep_results if isinstance(result, Exception)
            ]
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
                episode_numbers=[
                    episode_number for episode_number, _ in spoken_scripts
                ],
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
            audio_errors = [
                result for result in audio_results if isinstance(result, Exception)
            ]
            if audio_errors:
                raise RuntimeError(
                    "Audio rendering failed for "
                    f"{len(audio_errors)} episode(s): {audio_errors[0]}"
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
            raise RuntimeError(
                f"Run directory does not contain episodes/: {project_dir}"
            )

        self._bind_run_logger(project_dir)

        episode_dirs = sorted(
            [
                path
                for path in episodes_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
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
            raise RuntimeError(
                f"No render_manifest.json files found under {episodes_dir}"
            )

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
        for episode_number, result in zip(
            (ep for ep, _ in manifests), results, strict=True
        ):
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
                summary_tasks.append(
                    asyncio.to_thread(self.chapter_summary_agent.run, payload)
                )

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
                chapter_info = [
                    _build_compact_chapter_projection(ch) for ch in book.chapters
                ]
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
                for (book_id, _), result in zip(
                    summary_payloads, summary_results, strict=True
                )
            }

            payload = self.theme_decomposition_agent.build_payload(
                theme=project.theme,
                sub_themes=project.sub_themes,
                theme_elaboration=project.theme_elaboration,
                books=project.books,
                axis_count_min=project.config.min_axes,
                axis_count_max=project.config.max_axes,
                actor_count_min=(
                    5 if project.config.podcast_mode == PodcastMode.MINIFIED else 10
                ),
                actor_count_max=(
                    12 if project.config.podcast_mode == PodcastMode.MINIFIED else 40
                ),
                book_summaries=book_summaries,
            )
            expected_book_ids = [book.book_id for book in project.books]
            max_attempts = self.theme_decomposition_agent.max_retry_attempts
            axes: list[ThematicAxis] = []
            actor_metadata = ActorMetadata(project_id=project.project_id)
            actor_metrics: dict[str, Any] = {}
            for attempt in range(1, max_attempts + 1):
                result = await asyncio.to_thread(
                    self.theme_decomposition_agent.run, payload
                )
                axes = result.axes
                actor_metadata, actor_metrics = sanitize_actor_metadata_payload(
                    result.actor_metadata,
                    project_id=project.project_id,
                )

                missing_by_axis = []
                for axis in axes:
                    provided_book_ids = set(axis.relevance_by_book.keys())
                    missing_book_ids = [
                        book_id
                        for book_id in expected_book_ids
                        if book_id not in provided_book_ids
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
                a
                for a in axes
                if sum(1 for s in a.relevance_by_book.values() if s >= 0.3) >= 2
            ]

            if len(valid_axes) < project.config.min_axes:
                valid_axis_ids = {axis.axis_id for axis in valid_axes}
                fallback_axes = [
                    axis for axis in axes if axis.axis_id not in valid_axis_ids
                ]
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
            valid_axes, axis_actor_metrics = clean_axis_actor_ids(
                valid_axes, actor_metadata
            )
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
            axis_candidate_budget_target = max(
                1, project.config.axis_candidate_target_total
            )
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
                    (
                        info["per_book_budget"]
                        for info in retrieval_depth_by_book.values()
                    ),
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
                axis.axis_id: rank
                for rank, axis in enumerate(axis_priority_order, start=1)
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
                    duplicate_ids = [
                        pid for pid, count in id_counts.items() if count > 1
                    ]
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
                    if (
                        score.passage_id in candidate_by_id
                        and score.passage_id not in scores_by_id
                    ):
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
                            full_text=candidate_full_text_by_id.get(
                                candidate["passage_id"], ""
                            ),
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
                passage_book_by_id = {
                    p.passage_id: p.book_id for p in rehydrated_passages
                }
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
                        pre_axis_budget_by_axis.get(
                            axis.axis_id, axis_candidate_budget_target
                        ),
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
                    "axis_importance_weight": pre_axis_weight_by_axis.get(
                        axis.axis_id, 1.0
                    ),
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
                        "chunk_count": retrieval_depth_by_book.get(bid, {}).get(
                            "chunk_count", 0
                        ),
                        "percentage_budget": retrieval_depth_by_book.get(bid, {}).get(
                            "percentage_budget", 0
                        ),
                        "retrieval_depth_budget": retrieval_depth_by_book.get(
                            bid, {}
                        ).get("per_book_budget", 0),
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
                            confidence = _clamp(
                                (max_score - raw_score) / denom, 0.0, 1.0
                            )
                        row["retrieval_confidence"] = confidence
                        row["priority"] = confidence
                        confidences.append(confidence)
                    confidences.sort(reverse=True)
                    top_n = min(10, len(confidences))
                    retrieval_signal_by_book[bid] = sum(confidences[:top_n]) / max(
                        1, top_n
                    )

                retrieval_log["retrieval_signal_by_book"] = retrieval_signal_by_book
                retrieval_log["blended_score_by_book"] = relevance_by_book
                query_parts = [axis.name, axis.description]
                query_parts.extend(axis.guiding_questions)
                query_parts.extend(axis.keywords)
                axis_query_text = " ".join(part for part in query_parts if part).strip()
                query_terms = _bm25_query_terms(axis_query_text)

                all_rows = [
                    row for bid in book_ids for row in rows_by_book.get(bid, [])
                ]
                all_tokens = [_tokenize(row["hit"].text) for row in all_rows]
                idf, avg_len = _bm25_idf_and_avg_len(all_tokens)

                book_entry_by_id = {
                    entry["book_id"]: entry for entry in retrieval_log["books"]
                }
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
                                "bm25_score": round(
                                    float(row.get("bm25_score", 0.0)), 8
                                ),
                                "selection_phase": row["selection_phase"]
                                if used
                                else None,
                                "chapter_penalty": 0.0,
                                "selection_score": round(
                                    float(row["selection_score"]), 8
                                )
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
                    bid: info["per_book_budget"]
                    for bid, info in retrieval_depth_by_book.items()
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
                        "axis_importance_weight": retrieval_log[
                            "axis_importance_weight"
                        ],
                        "retrieval_depth_by_book": retrieval_log[
                            "retrieval_depth_by_book"
                        ],
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
                trimmed_text_count = sum(
                    1 for p in axis_passages if p.trimmed_text.strip()
                )
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
                        "0.5_to_0.8": sum(
                            1 for s in relevance_scores if 0.5 <= s < 0.8
                        ),
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
                        sum(p.relevance_score for p in book_passages)
                        / max(1, len(book_passages)),
                        3,
                    ),
                    "size_share": round(size_share, 4),
                    "avg_axis_admitted_share": round(avg_admitted_share, 4),
                    "admitted_minus_size_share": round(
                        avg_admitted_share - size_share, 4
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
            synthesis_passages_by_axis, cap_report = (
                _allocate_synthesis_passages_by_axis(
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
                        synthesis_trim_tiers[key] = synthesis_trim_tiers.get(
                            key, 0
                        ) + int(value)
                    _trim_candidate_texts_by_bm25(
                        axis,
                        prompt_passages,
                        keep_fraction=0.25,
                        keep_fraction_by_passage_id=synthesis_keep_fraction_by_passage_id,
                    )
                trimmed_text_by_id = {
                    item["passage_id"]: item["text"] for item in prompt_passages
                }
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
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
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
                            primitive.model_dump(mode="json")
                            for primitive in substrate_primitives
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

    async def _choose_narrative_strategy(
        self,
        project: ThematicProject,
        synthesis_map: PrimitiveFunctionTaggingArtifact,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[NarrativeStrategy, dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "narrative_strategy",
            project_dir,
            primitive_count=len(_flatten_base_synthesis_primitives(synthesis_map)),
        ) as ctx:
            payload = _compact_narrative_strategy_runtime_payload(
                self.narrative_strategy_agent.build_payload(
                    synthesis_map=_build_narrative_strategy_synthesis_map_payload(
                        synthesis_map
                    ),
                    project_metadata=_build_narrative_strategy_project_metadata_payload(
                        project
                    ),
                    episode_count=project.requested_episode_count,
                    recommended_episode_count_min=(
                        project.config.narrative_strategy_episode_count_min
                    ),
                    recommended_episode_count_max=(
                        project.config.narrative_strategy_episode_count_max
                    ),
                    actor_metadata=_build_narrative_strategy_actor_metadata_payload(
                        actor_metadata
                    ),
                )
            )
            strategy = await asyncio.to_thread(
                self.narrative_strategy_agent.run, payload
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
            for warning in strategy_warnings:
                logger.warning("narrative_strategy_warning %s", warning)

            ctx["output_summary"] = {
                "strategy": strategy.strategy_type,
                "recommended_episode_count": strategy.recommended_episode_count,
                "episodes": len(strategy.episodes),
                "warning_count": len(strategy_warnings),
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
            retained_primitives: list[SynthesisPrimitiveBase] = list(
                selected_base_primitives
            )
            gloss_warnings = _build_narration_hook_gloss_warnings(retained_primitives)
            for warning in gloss_warnings:
                logger.warning("%s", warning)

            synthesis_map = SynthesisMap(
                project_id=project.project_id,
                primitives=retained_primitives,
                quality_score=synthesis_primitives.quality_score,
                quality_notes=[
                    *list(synthesis_primitives.quality_notes),
                    *gloss_warnings,
                ],
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

    async def _build_episode_architectures(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[list[EpisodeArchitecture], dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "episode_architecture",
            project_dir,
            episode_count=project.episode_count,
            strategy=strategy.strategy_type,
        ) as ctx:
            episode_map = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            missing_episodes = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in episode_map
            ]
            if missing_episodes:
                raise RuntimeError(
                    "Narrative strategy did not assign episode spines for "
                    f"episodes: {missing_episodes}"
                )

            primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            passage_lookup = _build_passage_lookup(corpus)
            project_metadata = {
                "podcast_mode": project.config.podcast_mode.value,
                "theme": project.theme,
                "sub_themes": project.sub_themes,
                "book_count": len(project.books),
                "books": [
                    {"book_id": b.book_id, "title": b.title, "author": b.author}
                    for b in project.books
                ],
                "architecture_section_target_min": (
                    project.config.architecture_section_target_min
                ),
                "architecture_section_target_max": (
                    project.config.architecture_section_target_max
                ),
                "min_episode_minutes": project.config.min_episode_minutes,
                "max_episode_minutes": project.config.max_episode_minutes,
            }
            architecture_sem = asyncio.Semaphore(
                max(1, project.config.episode_architecture_concurrency)
            )
            ordered_episodes = [
                episode_map[episode_number]
                for episode_number in range(1, project.episode_count + 1)
            ]

            async def _build_episode_architecture(
                episode: StrategyEpisode,
            ) -> tuple[int, StrategyEpisode, EpisodeArchitecture, dict[str, Any]]:
                async with architecture_sem:
                    primitive_ids_by_role = {
                        "core": list(episode.episode_spine.core_primitive_ids),
                        "support": list(
                            episode.episode_spine.support_primitive_roles.keys()
                        ),
                        "recall": list(episode.episode_spine.recall_primitive_ids),
                    }
                    episode_synthesis_map_payload, primitive_ids = (
                        _build_episode_synthesis_map_payload(
                            synthesis_map,
                            primitive_ids_by_role,
                        )
                    )
                    episode_actor_ids = _collect_episode_actor_ids(
                        strategy_episode=episode,
                        primitive_ids=primitive_ids,
                        primitive_lookup=primitive_lookup,
                    )
                    episode_actor_metadata = select_actor_metadata_subset(
                        actor_metadata,
                        episode_actor_ids,
                    )
                    core_passages = _build_episode_architecture_core_passages(
                        driving_question=episode.episode_spine.listener_problem,
                        thematic_focus=episode.thematic_focus,
                        episode_spine=episode.episode_spine,
                        primitive_lookup=primitive_lookup,
                        passage_lookup=passage_lookup,
                    )
                    support_passages = _build_episode_architecture_support_passages(
                        driving_question=episode.episode_spine.listener_problem,
                        thematic_focus=episode.thematic_focus,
                        episode_spine=episode.episode_spine,
                        primitive_lookup=primitive_lookup,
                        passage_lookup=passage_lookup,
                    )
                    payload = self.episode_architecture_agent.build_payload(
                        episode=episode.model_dump(mode="json"),
                        synthesis_map=episode_synthesis_map_payload,
                        project_metadata=project_metadata,
                        core_passages=core_passages,
                        support_passages=support_passages,
                        series_explanation_registry=[
                            item.model_dump(mode="json")
                            for item in strategy.series_explanation_registry
                        ],
                        series_actor_explanation_registry=[
                            item.model_dump(mode="json")
                            for item in strategy.series_actor_explanation_registry
                        ],
                        narrator_profile=strategy.narrator_profile.model_dump(
                            mode="json"
                        ),
                        actor_metadata=compact_actor_metadata(episode_actor_metadata),
                    )
                    architecture = await asyncio.to_thread(
                        self.episode_architecture_agent.run, payload
                    )
                    architecture = _validate_architecture_transition(
                        strategy_episode=episode,
                        architecture=architecture,
                    )
                    report = {
                        "episode_number": episode.episode_number,
                        "section_count": len(architecture.sections),
                        "major_turn_section_id": architecture.major_turn_section_id,
                        "core_primitive_count": len(
                            episode.episode_spine.core_primitive_ids
                        ),
                        "actor_directive_count": len(episode.actor_arc_directives),
                    }
                    return episode.episode_number, episode, architecture, report

            architecture_results = await asyncio.gather(
                *[_build_episode_architecture(episode) for episode in ordered_episodes]
            )
            architecture_results.sort(key=lambda item: item[0])
            architectures = [
                architecture for _, _, architecture, _ in architecture_results
            ]
            reports = [report for _, _, _, report in architecture_results]
            realization_reports = [
                _build_episode_architecture_realization(
                    strategy_episode=episode,
                    architecture=architecture,
                    pipeline_config=project.config,
                    narrator_profile=strategy.narrator_profile,
                    primitive_lookup=primitive_lookup,
                    series_explanation_registry=strategy.series_explanation_registry,
                )
                for _, episode, architecture, _ in architecture_results
            ]
            for realization in realization_reports:
                for warning in realization["warnings"]:
                    logger.warning(
                        "episode_architecture_warning episode=%s %s",
                        realization["episode_number"],
                        warning,
                    )

            _save_json(
                project_dir / "episode_architectures.json",
                {
                    "episodes": [
                        episode.model_dump(mode="json") for episode in architectures
                    ]
                },
            )
            _save_json(
                project_dir / "architecture_realization.json",
                {"episodes": realization_reports},
            )
            ctx["output_summary"] = {
                "episode_count": len(architectures),
                "titles": [episode.title for episode in ordered_episodes],
                "warning_count": sum(
                    realization["warning_count"] for realization in realization_reports
                ),
            }
            return architectures, _merge_actor_metric_dicts(reports)

    async def _plan_series(
        self,
        project: ThematicProject,
        synthesis_map: SynthesisMap,
        strategy: NarrativeStrategy,
        episode_architectures: list[EpisodeArchitecture],
        corpus: ThematicCorpus,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
    ) -> tuple[list[EpisodePlan], dict[str, Any]]:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        async with _stage_log(
            self.run_logger,
            "episode_planning",
            project_dir,
            episode_count=project.episode_count,
        ) as ctx:
            episode_map = {
                episode.episode_number: episode for episode in episode_architectures
            }
            strategy_episode_map = {
                episode.episode_number: episode for episode in strategy.episodes
            }
            missing_episodes = [
                episode_number
                for episode_number in range(1, project.episode_count + 1)
                if episode_number not in episode_map
                or episode_number not in strategy_episode_map
            ]
            if missing_episodes:
                raise RuntimeError(
                    "Narrative strategy and episode architecture must both cover "
                    f"episodes: {missing_episodes}"
                )
            passage_lookup = _build_passage_lookup(corpus)
            primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
            project_metadata = {
                "podcast_mode": project.config.podcast_mode.value,
                "theme": project.theme,
                "sub_themes": project.sub_themes,
                "book_count": len(project.books),
                "books": [
                    {"book_id": b.book_id, "title": b.title, "author": b.author}
                    for b in project.books
                ],
                "scene_card_target_min": project.config.scene_card_target_min,
                "scene_card_target_max": project.config.scene_card_target_max,
                "min_episode_minutes": project.config.min_episode_minutes,
                "max_episode_minutes": project.config.max_episode_minutes,
            }
            host_policy = _build_host_policy_payload(strategy.narrator_profile)
            planning_sem = asyncio.Semaphore(
                max(1, project.config.episode_planning_concurrency)
            )
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
                    primitive_ids_by_role = _filter_primitive_ids_by_architecture(
                        strategy_episode,
                        episode,
                    )
                    episode_synthesis_map_payload, primitive_ids = (
                        _build_episode_synthesis_map_payload(
                            synthesis_map,
                            primitive_ids_by_role,
                        )
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
                    available_passages = [
                        {
                            "passage_id": passage_lookup[passage_id].passage_id,
                            "book_id": passage_lookup[passage_id].book_id,
                            "text": _resolve_writing_passage_text(
                                passage_lookup[passage_id]
                            ),
                            "chapter_ref": passage_lookup[passage_id].chapter_ref,
                        }
                        for passage_id in passage_ids
                    ]
                    episode_query_parts = [
                        strategy_episode.title,
                        strategy_episode.episode_spine.listener_problem,
                        strategy_episode.thematic_focus,
                    ]
                    episode_query_parts.extend(strategy_episode.unresolved_questions)
                    episode_query_text = " ".join(
                        part for part in episode_query_parts if part
                    ).strip()
                    passage_query_text_by_id = (
                        _build_episode_planning_passage_query_texts(
                            episode_query_text=episode_query_text,
                            passage_refs=planning_passage_refs,
                            primitive_lookup=primitive_lookup,
                        )
                    )
                    _trim_candidate_texts_by_bm25_query_text(
                        episode_query_text,
                        available_passages,
                        keep_fraction=0.5,
                        keep_fraction_by_passage_id=passage_keep_fraction_by_id,
                        query_text_by_passage_id=passage_query_text_by_id,
                    )
                    episode_payload = episode.model_dump(mode="json")
                    compact_episode_actor_metadata = compact_actor_metadata(
                        episode_actor_metadata
                    )
                    plan_draft: EpisodePlanDraft | None = None
                    actor_link_metrics: dict[str, Any] = {}
                    actor_explanation_warnings: list[str] = []
                    planning_feedback: dict[str, Any] | None = None
                    max_attempts = self.episode_planning_agent.max_retry_attempts
                    for attempt in range(1, max_attempts + 1):
                        payload = self.episode_planning_agent.build_payload(
                            strategy_episode=strategy_episode.model_dump(mode="json"),
                            architecture=episode_payload,
                            synthesis_map=episode_synthesis_map_payload,
                            project_metadata=project_metadata,
                            available_passages=available_passages,
                            host_policy=host_policy,
                            actor_metadata=compact_episode_actor_metadata,
                            planning_feedback=planning_feedback,
                        )
                        try:
                            plan_draft = await asyncio.to_thread(
                                self.episode_planning_agent.run,
                                payload,
                            )
                            plan_draft = _validate_plan_transition(
                                strategy_episode=strategy_episode,
                                architecture=episode,
                                plan=plan_draft,
                            )
                            plan_draft, actor_link_metrics = clean_scene_actor_links(
                                plan_draft,
                                episode_actor_metadata,
                                strategy_episode.actor_arc_directives,
                            )
                            actor_explanation_warnings = (
                                _validate_actor_explanation_scene_links(
                                    architecture=episode,
                                    plan=plan_draft,
                                )
                            )
                            break
                        except ComplianceViolationError as exc:
                            if attempt >= max_attempts:
                                raise
                            backoff = min(2 ** (attempt - 1), 16) + (
                                time.monotonic() % 1
                            )
                            planning_feedback = _build_plan_transition_feedback(exc)
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
                    )
                    host_moves_diagnostics, host_move_warnings = (
                        _build_host_move_plan_diagnostics(
                            scene_cards=plan_draft.scene_cards,
                            architecture=episode,
                            narrator_profile=strategy.narrator_profile,
                        )
                    )
                    scene_card_count_warnings = _build_scene_card_count_warnings(
                        scene_card_count=len(plan_draft.scene_cards),
                        scene_card_target_min=project.config.scene_card_target_min,
                        scene_card_target_max=project.config.scene_card_target_max,
                    )
                    scene_card_primitive_warnings = (
                        _build_scene_card_primitive_warnings(
                            scene_cards=plan_draft.scene_cards,
                            primitive_pool_ids=set(primitive_ids),
                            primitive_by_id=primitive_lookup,
                            primitive_min=project.config.scene_card_primitives_min,
                            primitive_max=project.config.scene_card_primitives_max,
                        )
                    )
                    scene_card_family_warnings = _build_scene_card_family_warnings(
                        strategy_episode=strategy_episode,
                        primitive_pool_ids=set(primitive_ids),
                        primitive_by_id=primitive_lookup,
                    )
                    section_realization_reports, section_planning_warnings = (
                        _build_section_plan_realization(
                            episode=episode,
                            scene_cards=plan_draft.scene_cards,
                            words_per_minute=float(
                                self.settings.pipeline.spoken_words_per_minute
                            ),
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
                    scene_function_counts = _build_scene_function_counts(
                        plan_draft.scene_cards
                    )
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
                        + actor_explanation_warnings
                        + section_planning_warnings
                        + host_move_warnings
                    )
                    for warning in planning_warnings:
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
                        "scene_card_count_warnings": scene_card_count_warnings,
                        "scene_card_primitive_warnings": scene_card_primitive_warnings,
                        "scene_card_family_warnings": scene_card_family_warnings,
                        "structural_card_concreteness_warnings": structural_card_concreteness_warnings,
                        "human_grounding_warnings": human_grounding_warnings,
                        "section_load_warnings": section_load_warnings,
                        "section_realization": section_realization_reports,
                        "scene_card_warning_count": len(planning_warnings),
                        "scene_role_counts": scene_role_counts,
                        "scene_function_counts": scene_function_counts,
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
                            scene.estimated_duration_seconds
                            for scene in plan.scene_cards
                        ),
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

            _save_json(
                project_dir / "series_plan.json",
                {
                    "episodes": [
                        episode.model_dump(mode="json") for episode in planned_episodes
                    ]
                },
            )
            _save_json(
                project_dir / "episode_plan_realization.json",
                {"episodes": planning_reports},
            )

            ctx["output_summary"] = {
                "episode_count": len(planned_episodes),
                "titles": [
                    strategy_episode.title for _, strategy_episode in ordered_episodes
                ],
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
                utilized_passage_ids.update(
                    citation.passage_id for citation in section.citations
                )

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
    ) -> tuple[int, SpokenScript]:
        async with semaphore:
            ep_dir = project_dir / "episodes" / str(plan.episode_number)
            ep_dir.mkdir(parents=True, exist_ok=True)

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
                primitive_lookup,
            )

            if not project.config.skip_grounding:
                report = await self._validate_grounding(
                    plan.episode_number,
                    script,
                    corpus,
                    ep_dir,
                    project_dir,
                    architecture=architecture,
                )
                if report.overall_status != "PASSED":
                    script, report = await self._repair_loop(
                        plan.episode_number,
                        script,
                        report,
                        corpus,
                        ep_dir,
                        architecture,
                        project_dir,
                        max_attempts=project.config.max_repair_attempts,
                    )
            else:
                self.run_logger.log("grounding_skipped", episode=plan.episode_number)

            script = await self._style_audit_episode(
                plan.episode_number,
                script,
                architecture,
                ep_dir,
                project_dir,
                plan=plan,
                host_policy=host_policy,
                strategy_episode=strategy_episode,
                series_explanation_registry=series_explanation_registry,
            )

        if not project.config.skip_spoken_delivery:
            spoken_gate = spoken_semaphore or semaphore
            async with spoken_gate:
                spoken = await self._rewrite_for_speech(
                    plan.episode_number,
                    script,
                    project,
                    ep_dir,
                    project_dir,
                    architecture=architecture,
                    plan=plan,
                    host_policy=host_policy,
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
        primitive_lookup: dict[str, SynthesisPrimitiveBase] | None = None,
    ) -> EpisodeScript:
        actor_metadata = actor_metadata or ActorMetadata(project_id=project.project_id)
        primitive_lookup = primitive_lookup or {}
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
            book_metadata = [
                {"book_id": b.book_id, "title": b.title, "author": b.author}
                for b in project.books
            ]
            scene_word_count_targets_lower = _compute_scene_word_count_targets(
                plan.scene_cards,
                plan.target_word_count,
                145.0,
            )
            scene_word_count_targets_higher = _compute_scene_word_count_targets(
                plan.scene_cards,
                plan.target_word_count,
                160.0,
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
                scene_payload_by_id[scene_id] = _build_writer_scene_brief(scene_payload)
            writing_agent = (
                self.writing_agent_no_citations
                if project.config.skip_grounding
                else self.writing_agent
            )
            episode_target_word_count_lower = sum(
                scene_word_count_targets_lower.values()
            )
            episode_target_word_count_higher = sum(
                scene_word_count_targets_higher.values()
            )
            writing_windows = _split_episode_writing_windows(
                plan=plan,
                architecture=architecture,
                scene_word_count_targets_lower=scene_word_count_targets_lower,
                scene_word_count_targets_higher=scene_word_count_targets_higher,
                max_windows=project.config.episode_writing_batch_count,
            )
            warning_threshold = int(
                math.ceil(
                    float(episode_target_word_count_higher)
                    * _WRITING_WORD_OVERRUN_WARNING_RATIO
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
                    for part_number, window_scene_cards in enumerate(
                        writing_windows, start=1
                    ):
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
                                    (
                                        part_target_word_count_lower
                                        + part_target_word_count_higher
                                    )
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
                        window_architecture_payload = (
                            _build_window_architecture_payload(
                                architecture=architecture,
                                window_scene_cards=window_scene_cards,
                            )
                        )
                        window_passages = _build_window_passages(
                            window_scene_cards=window_scene_cards,
                            passage_lookup=passage_lookup,
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
                                strategy_episode=strategy_episode.model_dump(
                                    mode="json"
                                ),
                                architecture=window_architecture_payload,
                                episode_plan=window_plan_payload,
                                passages=window_passages,
                                book_metadata=book_metadata,
                                episode_target_word_count_lower=part_target_word_count_lower,
                                episode_target_word_count_higher=part_target_word_count_higher,
                                skip_grounding=project.config.skip_grounding,
                                host_policy=host_policy,
                                scene_primitive_briefs=window_scene_primitive_briefs,
                                actor_metadata=window_actor_metadata,
                                writing_feedback=writing_feedback_by_part.get(
                                    part_number
                                ),
                                prior_window_continuity=prior_window_continuity,
                            )
                            result = await asyncio.to_thread(writing_agent.run, payload)
                            part_word_count = _writing_result_word_count(result)
                            actual_word_count += part_word_count
                            try:
                                normalized_window_outputs = (
                                    _normalize_writing_section_outputs(
                                        result=result,
                                        architecture=architecture,
                                        scene_cards=window_scene_cards,
                                        episode_number=plan.episode_number,
                                        skip_grounding=project.config.skip_grounding,
                                    )
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
                        (contract_exc.data or {}).get("failed_part_number")
                        or len(writing_windows)
                    )
                    writing_feedback_by_part[failed_part_number] = (
                        _build_writing_retry_feedback(contract_exc)
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
                        (exc.data or {}).get("failed_part_number")
                        or len(writing_windows)
                    )
                    writing_feedback_by_part[failed_part_number] = (
                        _build_writing_retry_feedback(exc)
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
                    over_high_word_count=(
                        actual_word_count - episode_target_word_count_higher
                    ),
                    over_high_ratio=(
                        actual_word_count / max(1, episode_target_word_count_higher)
                    ),
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

    async def _validate_grounding(
        self,
        episode_number: int,
        script: EpisodeScript,
        corpus: ThematicCorpus,
        ep_dir: Path,
        project_dir: Path,
        architecture: EpisodeArchitecture | None = None,
    ) -> GroundingReport:
        async with _stage_log(
            self.run_logger,
            f"grounding_{episode_number}",
            project_dir,
            episode=episode_number,
            text_unit_count=len(script.prose_sections),
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
                script={
                    **script.model_dump(mode="json"),
                    "prose_sections": _build_script_sections_payload(
                        script=script,
                        architecture=architecture,
                    ),
                },
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
        self,
        episode_number: int,
        script: EpisodeScript,
        report: GroundingReport,
        corpus: ThematicCorpus,
        ep_dir: Path,
        architecture: EpisodeArchitecture,
        project_dir: Path,
        max_attempts: int = 3,
    ) -> tuple[EpisodeScript, GroundingReport]:
        current_script = script
        current_report = report

        for attempt in range(1, max_attempts + 1):
            if current_report.overall_status == "PASSED":
                break

            failing_claims = [
                ca
                for ca in current_report.claim_assessments
                if ca.status in ("UNSUPPORTED", "FABRICATED")
            ]
            if not failing_claims and not current_report.fairness_flags:
                break

            async with _stage_log(
                self.run_logger,
                f"repair_{episode_number}_attempt_{attempt}",
                project_dir,
                episode=episode_number,
                attempt=attempt,
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
                failing_unit_ids.update(
                    flag.text_unit_id for flag in current_report.fairness_flags
                )
                failing_sections = [
                    section
                    for section in _build_script_sections_payload(
                        script=current_script,
                        architecture=architecture,
                    )
                    if section["section_id"] in failing_unit_ids
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
                    failure_reasons=failure_reasons,
                    passages=passage_lookup,
                )
                result = await asyncio.to_thread(self.repair_agent.run, payload)

                repaired_sections = {
                    section.section_id: section for section in result.repaired_sections
                }
                new_sections = []
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

                new_script = current_script.model_copy(
                    update={
                        "prose_sections": new_sections,
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
                    episode_number,
                    new_script,
                    corpus,
                    ep_dir,
                    project_dir,
                    architecture=architecture,
                )

                remaining = len(
                    [
                        ca
                        for ca in new_report.claim_assessments
                        if ca.status in ("UNSUPPORTED", "FABRICATED")
                    ]
                )
                status = (
                    "RESOLVED"
                    if new_report.overall_status == "PASSED"
                    else "IMPROVED"
                    if new_report.grounding_score > current_report.grounding_score
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

    async def _style_audit_episode(
        self,
        episode_number: int,
        script: EpisodeScript,
        architecture: EpisodeArchitecture,
        ep_dir: Path,
        project_dir: Path,
        plan: EpisodePlan | None = None,
        host_policy: dict[str, Any] | None = None,
        strategy_episode: StrategyEpisode | None = None,
        series_explanation_registry: list[Any] | None = None,
    ) -> EpisodeScript:
        async with _stage_log(
            self.run_logger,
            f"style_audit_{episode_number}",
            project_dir,
            episode=episode_number,
            section_count=len(script.prose_sections),
        ) as ctx:
            payload = self.style_audit_agent.build_payload(
                episode_number=episode_number,
                title=script.title,
                sections=_build_style_audit_sections_payload(
                    script=script,
                    architecture=architecture,
                    plan=plan,
                ),
                host_policy=host_policy,
                series_explanation_registry=_build_episode_explanation_registry_payload(
                    strategy_episode=strategy_episode,
                    architecture=architecture,
                    series_explanation_registry=series_explanation_registry,
                ),
            )
            audit = await asyncio.to_thread(self.style_audit_agent.run, payload)
            _save_json(ep_dir / "style_audit_result.json", audit)
            audited_script = _apply_style_audit_to_script(
                script=script,
                audit=audit,
                spoken_words_per_minute=float(
                    self.settings.pipeline.spoken_words_per_minute
                ),
            )
            _save_json(ep_dir / "style_audited_script.json", audited_script)
            host_moves_text_diagnostics = _build_host_move_text_diagnostics(
                text_by_section_id={
                    section.section_id: section.text
                    for section in audited_script.prose_sections
                },
                plan=plan,
            )
            _save_json(
                ep_dir / "host_moves_script_diagnostics.json",
                host_moves_text_diagnostics,
            )
            ctx["output_summary"] = {
                "sections": len(audit.sections),
                "warnings": len(audit.episode_warnings),
                "word_delta": audited_script.total_word_count - script.total_word_count,
                "planned_host_phases": host_moves_text_diagnostics[
                    "planned_host_phase_count"
                ],
                "approx_realized_host_phases": host_moves_text_diagnostics[
                    "approx_realized_host_phase_count"
                ],
            }
            return audited_script

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
                payload = self.spoken_delivery_agent.build_payload(
                    episode_number=episode_number,
                    script=script_payload,
                    max_words_per_segment=project.config.spoken_chunk_max_words,
                    tts_provider=project.config.tts_provider,
                    host_policy=host_policy,
                    previous_spoken_tail=previous_spoken_tail,
                )
                result = await asyncio.to_thread(
                    self.spoken_delivery_agent.run, payload
                )
                result_sections = list(result.sections)
                if (
                    not result_sections
                    and len(prose_batch) == 1
                    and result.text is not None
                    and result.speech_hints is not None
                ):
                    result_sections = [
                        SpokenDeliveryBatchSection(
                            section_id=prose_batch[0].section_id,
                            text=result.text,
                            speech_hints=result.speech_hints,
                        )
                    ]
                for rewritten in result_sections:
                    rewritten_sections.append(
                        SpokenSection(
                            section_id=rewritten.section_id,
                            text=rewritten.text,
                            speech_hints=rewritten.speech_hints,
                        )
                    )
                if result_sections:
                    previous_spoken_tail = _extract_previous_spoken_tail(
                        result_sections[-1].text
                    )

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

            ctx["output_summary"] = {
                "sections": len(spoken.sections),
                "batch_count": len(spoken.sections),
                "planned_host_phases": spoken_host_moves_diagnostics[
                    "planned_host_phase_count"
                ],
                "approx_realized_host_phases": spoken_host_moves_diagnostics[
                    "approx_realized_host_phase_count"
                ],
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

    def _merge_audio_segments(
        self, segment_paths: list[Path], output_path: Path
    ) -> None:
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
                    audio_path = (
                        audio_dir / f"{seg.segment_id}.{self.settings.tts.audio_format}"
                    )
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
                        logger.error(
                            "TTS failed for segment %s: %s", seg.segment_id, exc
                        )
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
