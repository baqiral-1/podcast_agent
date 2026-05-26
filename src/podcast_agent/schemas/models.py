"""Strict schema contracts for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import math
import re
from hashlib import sha1
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_WORD_RE = re.compile(r"\S+")


def _word_count(value: str) -> int:
    return len(_WORD_RE.findall(str(value or "").strip()))


def _validate_max_words(value: str, *, max_words: int, field_name: str) -> str:
    text = str(value or "").strip()
    if text and _word_count(text) > max_words:
        raise ValueError(f"{field_name} must be at most {max_words} words")
    return text


def _validate_list_item_word_limit(
    values: list[str],
    *,
    max_words: int,
    field_name: str,
) -> list[str]:
    normalized = [str(value or "").strip() for value in values]
    for item in normalized:
        if item and _word_count(item) > max_words:
            raise ValueError(
                f"each {field_name} entry must be at most {max_words} words"
            )
    return normalized


def _nonempty_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _nonempty_text(value).lower()).strip("_")
    return text or "item"


def _legacy_continuity_item_id(label: str) -> str:
    normalized = _nonempty_text(label)
    digest = sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"legacy_{_slugify_text(normalized)}_{digest}"


def _has_evidence_backed_actor_intro_payload(
    *,
    role_label: str,
    source_primitive_ids: list[str],
    source_passage_ids: list[str],
    intro_facts: list[str],
    why_now: str,
) -> bool:
    return bool(
        _nonempty_text(role_label)
        and source_primitive_ids
        and source_passage_ids
        and intro_facts
        and _nonempty_text(why_now)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProjectStatus(str, Enum):
    INGESTING = "ingesting"
    INDEXING = "indexing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    PRODUCING = "producing"
    COMPLETE = "complete"
    FAILED = "failed"


class PodcastMode(str, Enum):
    FULL = "full"
    MINIFIED = "minified"


class SynthesisTag(str, Enum):
    AGREES_WITH = "agrees_with"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    EXEMPLIFIES = "exemplifies"
    CONTEXTUALIZES = "contextualizes"
    INDEPENDENT = "independent"

class SupportPrimitiveRole(str, Enum):
    STAKES = "stakes"
    MECHANISM = "mechanism"
    COUNTERPRESSURE = "counterpressure"
    CONSEQUENCE = "consequence"
    TEXTURE = "texture"


class SpineRelation(str, Enum):
    SPINE_ADVANCE = "spine_advance"
    SET_STAKES = "set_stakes"
    SUPPLY_MECHANISM = "supply_mechanism"
    APPLY_COUNTERPRESSURE = "apply_counterpressure"
    SHOW_CONSEQUENCE = "show_consequence"
    TURN = "turn"
    TEXTURE_SUPPORT = "texture_support"


class VerdictMode(str, Enum):
    ANSWER = "answer"
    CONSTRAIN = "constrain"
    REFRAME = "reframe"
    PRESERVE_AMBIGUITY = "preserve_ambiguity"


class SectionPurpose(str, Enum):
    OPENING = "opening"
    SETUP = "setup"
    TURN = "turn"
    MECHANISM = "mechanism"
    COUNTERPRESSURE = "counterpressure"
    COLLAPSE = "collapse"
    CLOSING = "closing"


class ArgumentRole(str, Enum):
    FRAME = "frame"
    ESTABLISH_MECHANISM = "establish_mechanism"
    TEST_VIABILITY = "test_viability"
    BREAK_ROMANTIC_READING = "break_romantic_reading"
    CONVERT_EVENT_INTO_STRUCTURE = "convert_event_into_structure"
    CLOSE = "close"


class InferenceMode(str, Enum):
    SCENE_FIRST = "scene_first"
    MECHANISM_FIRST = "mechanism_first"
    CONTRAST_FIRST = "contrast_first"
    AFTERMATH_FIRST = "aftermath_first"


class RecurrenceRole(str, Enum):
    NONE = "none"
    PLANT = "plant"
    DEEPEN = "deepen"
    PAYOFF = "payoff"


class PressureType(str, Enum):
    CONSTITUTIONAL = "constitutional"
    MASS_POLITICAL = "mass_political"
    COMMUNAL = "communal"
    MORAL = "moral"


class AlignmentType(str, Enum):
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    INSTITUTIONAL = "institutional"
    SITUATIONAL = "situational"


class CoalitionPhase(str, Enum):
    FORMING = "forming"
    HOLDING = "holding"
    FRACTURING = "fracturing"
    BREAKING = "breaking"
    IMPERIAL = "imperial"
    PERSONAL = "personal"


class ResolutionType(str, Enum):
    REDEFINITION = "redefinition"
    COLLAPSE = "collapse"
    REVERSAL = "reversal"
    DISILLUSION = "disillusion"
    ESCALATION = "escalation"
    CONTAINMENT = "containment"


class ClosureLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProgressionStage(str, Enum):
    SETUP = "setup"
    ADVANCE = "advance"
    ANSWER = "answer"
    AFTERPRESSURE = "afterpressure"
    CLOSE = "close"


class WordCountPriority(str, Enum):
    DEFAULT = "default"
    TIGHT = "tight"


class SceneRole(str, Enum):
    CONTEXT_SETUP = "context_setup"
    ACTOR_SETUP = "actor_setup"
    ACTION = "action"
    SHOCK = "shock"
    CONTESTATION = "contestation"
    REACTION = "reaction"
    FALLOUT = "fallout"
    IMPLICATION = "implication"


class SceneJob(str, Enum):
    OPENING = "opening"
    BUILD = "build"
    TURN = "turn"
    ANSWER = "answer"
    CLOSE = "close"


class SceneFunction(str, Enum):
    SCENE = "scene"
    HINGE = "hinge"
    MECHANISM = "mechanism"
    TURN = "turn"
    LANDING = "landing"
    CALLBACK = "callback"
    AFTERLIFE = "afterlife"


class PromisedBeatDecision(str, Enum):
    STAGE = "stage"
    DEFER = "defer"
    DROP = "drop"


_SCENE_ROLE_VALUES = {member.value for member in SceneRole}
_SCENE_JOB_VALUES = {member.value for member in SceneJob}
_SCENE_FUNCTION_VALUES = {member.value for member in SceneFunction}


# ---------------------------------------------------------------------------
# 3.1 Project-Level Models
# ---------------------------------------------------------------------------


class ChapterInfo(StrictModel):
    chapter_id: str = Field(default_factory=new_id)
    title: str
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    word_count: int = Field(ge=0)
    analysis: "ChapterAnalysis | None" = None


class ChapterAnalysis(StrictModel):
    themes_touched: list[str] = Field(default_factory=list, max_length=8)
    major_actors: list[str] = Field(default_factory=list, max_length=8)
    key_events_or_arguments: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_fields = {
            "key_places",
            "key_institutions",
            "timeframe",
            "major_tensions",
        }
        if not legacy_fields.intersection(data):
            return data
        return {key: value for key, value in data.items() if key not in legacy_fields}


class BookRecord(StrictModel):
    book_id: str = Field(default_factory=new_id)
    title: str
    author: str
    source_path: str
    source_type: str
    chapters: list[ChapterInfo] = Field(default_factory=list)
    chunk_count: int = Field(default=0, ge=0)
    total_words: int = Field(default=0, ge=0)
    ingestion_diagnostics: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(StrictModel):
    podcast_mode: PodcastMode = PodcastMode.FULL
    max_axes: int = Field(default=20, ge=1)
    min_axes: int = Field(default=12, ge=1)
    passage_retrieval_percentage: float = Field(default=0.25, gt=0.0, le=1.0)
    passage_retrieval_min_per_book: int = Field(default=10, ge=1)
    passage_retrieval_max_per_book: int = Field(default=25, ge=1)
    axis_candidate_target_total: int = Field(default=60, ge=1)
    pre_axis_total_budget: int = Field(default=1500, ge=1)
    pre_axis_floor: int = Field(default=30, ge=0)
    pre_axis_relevance_power: float = Field(default=1.3, ge=0.0)
    pre_axis_cross_axis_reuse_penalty: float = Field(default=0.25, ge=0.0, le=1.0)
    admission_floor_per_book: int = Field(default=0, ge=0)
    retrieval_relevance_power: float = Field(default=2.5, ge=0.0)
    retrieval_soft_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    chapter_penalty_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    mmr_enabled: bool = True
    mmr_synthesis_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    mmr_planning_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    synthesis_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    synthesis_axis_min: int = Field(default=12, ge=0)
    synthesis_axis_max: int = Field(default=20, ge=1)
    synthesis_total_passage_cap: int = Field(default=750, ge=1)
    synthesis_floor_budget_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    synthesis_axis_floor_min: int = Field(default=0, ge=0)
    synthesis_axis_floor_max: int = Field(default=0, ge=0)
    synthesis_axis_ceiling_multiplier: float = Field(default=1.68, ge=1.0)
    synthesis_trim_top_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    synthesis_trim_mid_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    synthesis_trim_top_keep_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    synthesis_trim_mid_keep_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    synthesis_trim_tail_keep_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    planning_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    planning_axis_min: int = Field(default=10, ge=0)
    planning_axis_max: int = Field(default=15, ge=1)
    planning_total_passage_cap: int = Field(default=300, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=3, ge=0)
    tts_provider: str = "openai"
    tts_concurrency: int = Field(default=12, ge=1)
    episode_architecture_concurrency: int = Field(default=12, ge=1)
    episode_planning_concurrency: int = Field(default=12, ge=1)
    episode_write_concurrency: int = Field(default=8, ge=1)
    episode_writing_batch_count: int = Field(default=5, ge=1)
    spoken_delivery_concurrency: int | None = Field(default=8, ge=1)
    architecture_section_target_min: int = Field(default=11, ge=1)
    architecture_section_target_max: int = Field(default=14, ge=1)
    episode_spine_core_primitive_target_min: int = Field(default=6, ge=1)
    episode_spine_core_primitive_target_max: int = Field(default=8, ge=1)
    episode_spine_support_primitive_target_min: int = Field(default=6, ge=1)
    episode_spine_support_primitive_target_max: int = Field(default=9, ge=1)
    episode_spine_recall_primitive_target_max: int = Field(default=2, ge=0)
    narrative_strategy_episode_count_min: int = Field(default=10, ge=1)
    narrative_strategy_episode_count_max: int = Field(default=16, ge=1)
    min_episode_minutes: float = Field(default=90.0, gt=0.0)
    max_episode_minutes: float = Field(default=120.0, gt=0.0)
    duration_shortfall_policy: Literal["warn"] = "warn"
    scene_card_target_min: int = Field(default=30, ge=1)
    scene_card_target_max: int = Field(default=38, ge=1)
    scene_card_target_policy: Literal["warn"] = "warn"
    scene_card_primitives_min: int = Field(default=1, ge=0)
    scene_card_primitives_max: int = Field(default=2, ge=1)
    scene_card_primitive_policy: Literal["warn"] = "warn"
    passage_extraction_concurrency: int = Field(default=16, ge=1)
    chunk_max_words: int = Field(default=750, ge=50)
    chunk_overlap_words: int = Field(default=30, ge=0)
    spoken_chunk_max_words: int = Field(default=250, ge=50)
    max_author_names_per_episode: int = Field(default=3, ge=0)
    attribution_budget: float = Field(default=0.2, ge=0.0, le=1.0)
    prefer_indirect_attribution: bool = True
    skip_grounding: bool = False
    skip_spoken_delivery: bool = False
    skip_audio: bool = False

    @model_validator(mode="after")
    def validate_retrieval_budget_bounds(self) -> "PipelineConfig":
        if self.passage_retrieval_max_per_book < self.passage_retrieval_min_per_book:
            raise ValueError(
                "passage_retrieval_max_per_book must be >= passage_retrieval_min_per_book"
            )
        if self.synthesis_axis_max < self.synthesis_axis_min:
            raise ValueError("synthesis_axis_max must be >= synthesis_axis_min")
        if self.synthesis_axis_floor_max < self.synthesis_axis_floor_min:
            raise ValueError(
                "synthesis_axis_floor_max must be >= synthesis_axis_floor_min"
            )
        if self.planning_axis_max < self.planning_axis_min:
            raise ValueError("planning_axis_max must be >= planning_axis_min")
        if self.synthesis_trim_top_fraction + self.synthesis_trim_mid_fraction > 1.0:
            raise ValueError(
                "synthesis trim top and mid fractions must sum to <= 1.0"
            )
        if self.architecture_section_target_max < self.architecture_section_target_min:
            raise ValueError(
                "architecture_section_target_max must be >= architecture_section_target_min"
            )
        if (
            self.episode_spine_core_primitive_target_max
            < self.episode_spine_core_primitive_target_min
        ):
            raise ValueError(
                "episode_spine_core_primitive_target_max must be >= "
                "episode_spine_core_primitive_target_min"
            )
        if (
            self.episode_spine_support_primitive_target_max
            < self.episode_spine_support_primitive_target_min
        ):
            raise ValueError(
                "episode_spine_support_primitive_target_max must be >= "
                "episode_spine_support_primitive_target_min"
            )
        if self.episode_spine_recall_primitive_target_max > 3:
            raise ValueError(
                "episode_spine_recall_primitive_target_max must be <= 3"
            )
        if (
            self.narrative_strategy_episode_count_max
            < self.narrative_strategy_episode_count_min
        ):
            raise ValueError(
                "narrative_strategy_episode_count_max must be >= "
                "narrative_strategy_episode_count_min"
            )
        if self.scene_card_target_max < self.scene_card_target_min:
            raise ValueError("scene_card_target_max must be >= scene_card_target_min")
        if self.scene_card_primitives_max < self.scene_card_primitives_min:
            raise ValueError(
                "scene_card_primitives_max must be >= scene_card_primitives_min"
            )
        if self.max_episode_minutes < self.min_episode_minutes:
            raise ValueError("max_episode_minutes must be >= min_episode_minutes")
        return self


def _divide_range_floor(
    lower_bound: int,
    upper_bound: int,
    *,
    minimum_floor: int = 0,
) -> tuple[int, int]:
    reduced_lower = max(minimum_floor, lower_bound // 3)
    reduced_upper = max(minimum_floor, upper_bound // 3)
    return reduced_lower, max(reduced_lower, reduced_upper)


def _scale_range_floor_by_ratio(
    range_values: tuple[int, int], *, ratio: float
) -> tuple[int, int]:
    lower_bound, upper_bound = range_values
    reduced_lower = max(0, math.floor(lower_bound * ratio))
    reduced_upper = max(0, math.floor(upper_bound * ratio))
    return reduced_lower, max(reduced_lower, reduced_upper)


MINIFIED_AUTHORIAL_PASSAGE_TARGET_RANGE: tuple[int, int] = (12, 16)
MINIFIED_DENSE_SECTION_AUTHORIAL_PASSAGE_RANGE: tuple[int, int] = (2, 5)


def _scale_range_containing_span(
    base_range: tuple[int, int], multiplier: float
) -> tuple[int, int]:
    lower_bound, upper_bound = base_range
    return math.floor(lower_bound * multiplier), math.ceil(upper_bound * multiplier)


FULL_AUTHORIAL_PASSAGE_TARGET_RANGE: tuple[int, int] = (18, 28)
FULL_DENSE_SECTION_AUTHORIAL_PASSAGE_RANGE: tuple[int, int] = (3, 8)


def authorial_passage_target_range_for_mode(
    mode: PodcastMode | str,
) -> tuple[int, int]:
    if PodcastMode(mode) == PodcastMode.MINIFIED:
        return MINIFIED_AUTHORIAL_PASSAGE_TARGET_RANGE
    return FULL_AUTHORIAL_PASSAGE_TARGET_RANGE


def dense_section_authorial_passage_range_for_mode(
    mode: PodcastMode | str,
) -> tuple[int, int]:
    if PodcastMode(mode) == PodcastMode.MINIFIED:
        return MINIFIED_DENSE_SECTION_AUTHORIAL_PASSAGE_RANGE
    return FULL_DENSE_SECTION_AUTHORIAL_PASSAGE_RANGE


def authorial_passage_target_for_mode(mode: PodcastMode | str) -> int:
    lower_bound, upper_bound = authorial_passage_target_range_for_mode(mode)
    return math.floor((lower_bound + upper_bound) / 2)


def scene_job_budget_for_mode(mode: PodcastMode | str) -> dict[str, int]:
    if PodcastMode(mode) == PodcastMode.MINIFIED:
        return {
            "total_min": 18,
            "total_max": 21,
            "opening_min": 2,
            "opening_max": 2,
            "build_min": 12,
            "build_max": 15,
            "turn_min": 2,
            "turn_max": 2,
            "answer_min": 1,
            "answer_max": 1,
            "close_min": 1,
            "close_max": 1,
            "max_recap_build_scenes": 1,
        }
    return {
        "total_min": 30,
        "total_max": 38,
        "opening_min": 2,
        "opening_max": 3,
        "build_min": 22,
        "build_max": 28,
        "turn_min": 4,
        "turn_max": 5,
        "answer_min": 1,
        "answer_max": 1,
        "close_min": 1,
        "close_max": 1,
        "max_recap_build_scenes": 1,
    }


def scene_discovery_candidate_range_for_mode(
    mode: PodcastMode | str,
) -> tuple[int, int]:
    if PodcastMode(mode) == PodcastMode.MINIFIED:
        return 18, 26
    return 53, 79


def resolve_pipeline_config_for_mode(config: "PipelineConfig") -> "PipelineConfig":
    mode = PodcastMode(config.podcast_mode)
    if mode == PodcastMode.MINIFIED:
        updates = {
            "podcast_mode": mode,
            "min_axes": 4,
            "max_axes": 6,
            "pre_axis_total_budget": 500,
            "synthesis_axis_min": 4,
            "synthesis_axis_max": 6,
            "episode_spine_core_primitive_target_min": 3,
            "episode_spine_core_primitive_target_max": 5,
            "episode_spine_support_primitive_target_min": 4,
            "episode_spine_support_primitive_target_max": 6,
            "episode_spine_recall_primitive_target_max": 1,
            "narrative_strategy_episode_count_min": 2,
            "narrative_strategy_episode_count_max": 4,
            "episode_writing_batch_count": 2,
            "architecture_section_target_min": 7,
            "architecture_section_target_max": 9,
            "min_episode_minutes": 54.0,
            "max_episode_minutes": 63.0,
            "scene_card_target_min": 18,
            "scene_card_target_max": 21,
            "synthesis_total_passage_cap": 200,
        }
    else:
        updates = {
            "podcast_mode": mode,
            "min_axes": 12,
            "max_axes": 20,
            "synthesis_axis_min": 12,
            "synthesis_axis_max": 20,
            "episode_spine_core_primitive_target_min": 6,
            "episode_spine_core_primitive_target_max": 8,
            "episode_spine_support_primitive_target_min": 6,
            "episode_spine_support_primitive_target_max": 9,
            "episode_spine_recall_primitive_target_max": 2,
            "narrative_strategy_episode_count_min": 10,
            "narrative_strategy_episode_count_max": 16,
            "episode_writing_batch_count": 5,
            "architecture_section_target_min": 11,
            "architecture_section_target_max": 14,
            "min_episode_minutes": 90.0,
            "max_episode_minutes": 120.0,
            "scene_card_target_min": 30,
            "scene_card_target_max": 38,
            "synthesis_total_passage_cap": 750,
        }
    return config.model_copy(update=updates)


class ThematicProject(StrictModel):
    project_id: str = Field(default_factory=new_id)
    theme: str
    theme_elaboration: str | None = None
    sub_themes: list[str] = Field(default_factory=list, max_length=40)
    books: list[BookRecord] = Field(default_factory=list)
    requested_episode_count: int | None = Field(default=None, ge=1)
    recommended_episode_count: int | None = Field(default=None, ge=1)
    episode_count: int = Field(default=3, ge=1)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    created_at: datetime = Field(default_factory=utc_now)
    status: ProjectStatus = ProjectStatus.INGESTING

    @field_validator("sub_themes", mode="before")
    @classmethod
    def normalize_sub_themes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("sub_themes must be a list of strings.")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_item in value:
            if not isinstance(raw_item, str):
                raise ValueError("sub_themes must contain only strings.")
            item = raw_item.strip()
            if not item:
                raise ValueError("sub_themes entries must be non-empty after trimming.")
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        if len(normalized) > 40:
            raise ValueError("sub_themes supports at most 40 entries.")
        return normalized


# ---------------------------------------------------------------------------
# 3.2 Chunk & Retrieval Models
# ---------------------------------------------------------------------------


class ChunkingConfig(StrictModel):
    max_chunk_words: int = Field(default=750, ge=50)
    overlap_words: int = Field(default=30, ge=0)
    min_chunk_words: int = Field(default=80, ge=10)
    split_on: list[str] = Field(default_factory=lambda: ["\n\n", ". "])


class TextChunk(StrictModel):
    chunk_id: str = Field(default_factory=new_id)
    book_id: str
    chapter_id: str
    text: str
    word_count: int = Field(ge=0)
    position: int = Field(ge=0)
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3.3 Thematic Analysis Models
# ---------------------------------------------------------------------------


class ThematicAxis(StrictModel):
    axis_id: str = Field(default_factory=new_id)
    name: str
    description: str
    theme_importance_score: float = Field(ge=0.0, le=1.0)
    guiding_questions: list[str] = Field(default_factory=list)
    relevance_by_book: dict[str, float] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    parent_axis_id: str | None = None
    actor_ids: list[str] = Field(default_factory=list)


class ActorChapterRef(StrictModel):
    book_id: str
    chapter_id: str
    chapter_title: str = ""


class ActorProfile(StrictModel):
    actor_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    actor_type: Literal[
        "person",
        "institution",
        "faction",
        "military",
        "party",
        "movement",
        "other",
    ]
    description: str = ""
    book_ids: list[str] = Field(default_factory=list)
    chapter_refs: list[ActorChapterRef] = Field(default_factory=list)
    narrative_functions: list[
        Literal[
            "decision_maker",
            "broker",
            "victim",
            "witness",
            "ideologue",
            "commander",
            "administrator",
            "opposition",
            "beneficiary",
            "constraint",
            "catalyst",
            "symbol",
            "other",
        ]
    ] = Field(default_factory=list)
    goals_or_motivational_pressures: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    stakes: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)
    uncertainty_notes: str = ""
    evidence_confidence: Literal["high", "medium", "low"] = "medium"
    narrative_importance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("actor_id")
    @classmethod
    def validate_actor_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError("actor_id must be snake_case and start with a letter")
        return value


class ActorRelationship(StrictModel):
    source_actor_id: str
    target_actor_id: str
    relationship_type: Literal[
        "enables",
        "blocks",
        "pressures",
        "protects",
        "legitimizes",
        "delegitimizes",
        "replaces",
        "absorbs",
        "betrays",
        "other",
    ]
    description: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"


class ActorMention(StrictModel):
    raw_name: str
    source: str = ""
    book_id: str | None = None
    chapter_id: str | None = None
    matched_actor_id: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"


class ActorMetadata(StrictModel):
    project_id: str = ""
    actors: list[ActorProfile] = Field(default_factory=list)
    relationships: list[ActorRelationship] = Field(default_factory=list)
    unresolved_mentions: list[ActorMention] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_actor_links(self) -> "ActorMetadata":
        actor_ids = [actor.actor_id for actor in self.actors]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor_id values must be unique")
        actor_id_set = set(actor_ids)
        for relationship in self.relationships:
            if relationship.source_actor_id not in actor_id_set:
                raise ValueError(
                    f"relationship source_actor_id is unknown: {relationship.source_actor_id}"
                )
            if relationship.target_actor_id not in actor_id_set:
                raise ValueError(
                    f"relationship target_actor_id is unknown: {relationship.target_actor_id}"
                )
        return self


class ExtractedPassage(StrictModel):
    passage_id: str = Field(default_factory=new_id)
    book_id: str
    chunk_ids: list[str] = Field(min_length=1)
    text: str
    trimmed_text: str = ""
    full_text: str = ""
    chapter_ref: str = ""
    axis_id: str
    secondary_axes: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quotability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    synthesis_tags: list[SynthesisTag] = Field(default_factory=list)


class PassagePair(StrictModel):
    passage_a_id: str
    passage_b_id: str
    relationship: SynthesisTag
    strength: float = Field(ge=0.0, le=1.0)
    axis_id: str


class CoverageStats(StrictModel):
    total_passages: int = Field(default=0, ge=0)
    axes_covered: int = Field(default=0, ge=0)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ThematicCorpus(StrictModel):
    project_id: str
    axes: list[ThematicAxis] = Field(default_factory=list)
    passages_by_axis: dict[str, list[ExtractedPassage]] = Field(default_factory=dict)
    cross_book_pairs: list[PassagePair] = Field(default_factory=list)
    book_coverage: dict[str, CoverageStats] = Field(default_factory=dict)
    total_passages: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# 3.4 Synthesis Models
# ---------------------------------------------------------------------------


class PrimitiveSubstrate(str, Enum):
    EVENTS = "events"
    ACTS = "acts"
    UTTERANCES = "utterances"
    ACTOR_PORTRAITS = "actor_portraits"
    MECHANISMS = "mechanisms"
    CONDITIONS = "conditions"
    ARTIFACTS = "artifacts"
    READINGS = "readings"


class PrimitiveFunction(str, Enum):
    PIVOT = "pivot"
    STAKE = "stake"
    TEXTURE = "texture"
    COST = "cost"
    COMPLICATION = "complication"
    RECURRENCE = "recurrence"
    CONTEST = "contest"


class PrimitiveIrreversibility(str, Enum):
    LOW = "low"
    MEDIUM = "med"
    HIGH = "high"


class PrimitiveSalience(StrictModel):
    score: float = Field(ge=0.0, le=1.0)
    justification: str = Field(min_length=1)


class PivotJustification(StrictModel):
    what_changed: str = Field(min_length=1)
    irreversibility: PrimitiveIrreversibility = PrimitiveIrreversibility.MEDIUM


class StakeJustification(StrictModel):
    whose: str = Field(min_length=1)
    what_at_stake: str = Field(min_length=1)


class TextureJustification(StrictModel):
    what_it_anchors: str = Field(min_length=1)


class CostJustification(StrictModel):
    who_paid: str = Field(min_length=1)
    what_was_paid: str = Field(min_length=1)


class ComplicationJustification(StrictModel):
    what_is_compromised: str = Field(min_length=1)
    why_no_clean_option: str = Field(min_length=1)


class RecurrenceJustification(StrictModel):
    connects_to: list[str] = Field(default_factory=list)
    meaning_accrued: str = Field(min_length=1)


class ContestCandidateReading(StrictModel):
    reading: str = Field(min_length=1)
    support: str = ""
    weakness: str = ""


class ContestJustification(StrictModel):
    candidate_readings: list[ContestCandidateReading] = Field(
        default_factory=list, min_length=2
    )


CandidateReading = ContestCandidateReading


class NarrationHooks(StrictModel):
    concrete_detail: str = Field(min_length=1)
    host_lens: str = Field(min_length=1)
    carry_forward: str = Field(min_length=1)
    quote_anchor: str = ""
    plain_gloss: str = ""
    why_it_matters: str = ""
    best_use: str = ""
    natural_host_move: str = ""
    listener_confusion: str = ""
    authorial_move: Literal[
        "none",
        "quote_then_gloss",
        "doctrinal_unpack",
        "institutional_clarifier",
        "causal_compression",
        "comparative_aside",
        "verdict_landing",
    ] = "none"


class PrimitiveEnrichmentOverlay(StrictModel):
    functions: list[PrimitiveFunction] = Field(default_factory=list, max_length=3)
    salience: PrimitiveSalience | None = None
    pivot: PivotJustification | None = None
    stake: StakeJustification | None = None
    texture: TextureJustification | None = None
    cost: CostJustification | None = None
    complication: ComplicationJustification | None = None
    recurrence: RecurrenceJustification | None = None
    contest: ContestJustification | None = None
    narration_hooks: NarrationHooks | None = None
    event_result: str = ""
    immediate_result: str = ""
    operating_pressure: str = ""
    operating_chain: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    active_tension: str = ""
    artifact_detail: str = ""

    @model_validator(mode="after")
    def validate_functions(self) -> "PrimitiveEnrichmentOverlay":
        deduped_functions: list[PrimitiveFunction] = []
        seen: set[PrimitiveFunction] = set()
        for function in self.functions:
            if function in seen:
                continue
            seen.add(function)
            deduped_functions.append(function)
        self.functions = deduped_functions
        if len(self.functions) > 3:
            raise ValueError("functions must contain at most 3 entries")
        required_payloads = {
            PrimitiveFunction.PIVOT: self.pivot,
            PrimitiveFunction.STAKE: self.stake,
            PrimitiveFunction.TEXTURE: self.texture,
            PrimitiveFunction.COST: self.cost,
            PrimitiveFunction.COMPLICATION: self.complication,
            PrimitiveFunction.RECURRENCE: self.recurrence,
            PrimitiveFunction.CONTEST: self.contest,
        }
        for function, payload in required_payloads.items():
            if function in self.functions and payload is None:
                raise ValueError(
                    f"{function.value} requires its paired justification payload"
                )
            if function not in self.functions and payload is not None:
                raise ValueError(
                    f"{function.value} justification is not allowed unless the function tag is present"
                )
        return self


class BaseSynthesisPrimitive(StrictModel):
    id: str = Field(default_factory=new_id)
    substrate: PrimitiveSubstrate
    title: str = Field(min_length=1)
    core_passage_ids: list[str] = Field(default_factory=list)
    support_passage_ids: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    geography: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    functions: list[PrimitiveFunction] = Field(default_factory=list, max_length=3)
    salience: PrimitiveSalience | None = None
    pivot: PivotJustification | None = None
    stake: StakeJustification | None = None
    texture: TextureJustification | None = None
    cost: CostJustification | None = None
    complication: ComplicationJustification | None = None
    recurrence: RecurrenceJustification | None = None
    contest: ContestJustification | None = None
    narration_hooks: NarrationHooks | None = None

    @property
    def passage_ids(self) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        for passage_id in [*self.core_passage_ids, *self.support_passage_ids]:
            if not passage_id or passage_id in seen:
                continue
            seen.add(passage_id)
            combined.append(passage_id)
        return combined

    @property
    def narrative_importance_score(self) -> float:
        return self.salience.score if self.salience is not None else 0.0

    @model_validator(mode="after")
    def validate_functions(self) -> "BaseSynthesisPrimitive":
        deduped_functions: list[PrimitiveFunction] = []
        seen: set[PrimitiveFunction] = set()
        for function in self.functions:
            if function in seen:
                continue
            seen.add(function)
            deduped_functions.append(function)
        self.functions = deduped_functions
        if len(self.functions) > 3:
            raise ValueError("functions must contain at most 3 entries")
        if not self.passage_ids:
            raise ValueError(
                "primitives must include at least one core or support passage id"
            )
        required_payloads = {
            PrimitiveFunction.PIVOT: self.pivot,
            PrimitiveFunction.STAKE: self.stake,
            PrimitiveFunction.TEXTURE: self.texture,
            PrimitiveFunction.COST: self.cost,
            PrimitiveFunction.COMPLICATION: self.complication,
            PrimitiveFunction.RECURRENCE: self.recurrence,
            PrimitiveFunction.CONTEST: self.contest,
        }
        for function, payload in required_payloads.items():
            if function in self.functions and payload is None:
                raise ValueError(
                    f"{function.value} requires its paired justification payload"
                )
            if function not in self.functions and payload is not None:
                raise ValueError(
                    f"{function.value} justification is not allowed unless the function tag is present"
                )
        return self


SynthesisPrimitiveBase = BaseSynthesisPrimitive


class EventPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.EVENTS]
    event_type: str = Field(min_length=1)
    what_happened: str = Field(min_length=1)
    event_result: str = ""


class ActPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.ACTS]
    act_type: Literal[
        "decision",
        "refusal",
        "delay",
        "deferral",
        "order",
        "defection",
        "other",
    ] = "other"
    acting_subject: str = ""
    act_summary: str = Field(min_length=1)
    immediate_result: str = ""


class UtterancePrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.UTTERANCES]
    utterance_type: Literal[
        "speech",
        "writing",
        "broadcast",
        "decree",
        "testimony",
        "manifesto",
        "letter",
        "other",
    ] = "other"
    speaker: str = ""
    audience: str = ""
    utterance_summary: str = Field(min_length=1)
    key_quote: str = ""


class ActorPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.ACTOR_PORTRAITS]
    focus_actor_id: str | None = None
    actor_label: str = ""
    goal_or_project: str = Field(min_length=1)
    stakes_or_fears: str = Field(min_length=1)
    operating_pressure: str = ""


class MechanismPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.MECHANISMS]
    mechanism_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    failure_mode: str = ""


class ConditionPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.CONDITIONS]
    condition_type: str = Field(min_length=1)
    condition_summary: str = Field(min_length=1)
    active_tension: str = ""


class ArtifactPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.ARTIFACTS]
    artifact_type: Literal[
        "object",
        "place",
        "document",
        "image",
        "slogan",
        "ritual",
        "gesture",
        "detail",
        "other",
    ] = "other"
    artifact_label: str = Field(min_length=1)
    artifact_detail: str = Field(min_length=1)


class ReadingPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.READINGS]
    reading_type: Literal[
        "actor_belief",
        "historiographical_dispute",
        "interpretive_claim",
        "counterfactual",
        "other",
    ] = "other"
    subject_of_reading: str = ""
    attributed_to: str = ""
    reading_summary: str = Field(min_length=1)


AnySynthesisPrimitive = Annotated[
    EventPrimitive
    | ActPrimitive
    | UtterancePrimitive
    | ActorPrimitive
    | MechanismPrimitive
    | ConditionPrimitive
    | ArtifactPrimitive
    | ReadingPrimitive,
    Field(discriminator="substrate"),
]


class BaseExtractionPrimitive(StrictModel):
    substrate: PrimitiveSubstrate
    title: str = Field(min_length=1)
    core_passage_ids: list[str] = Field(default_factory=list)
    support_passage_ids: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    geography: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    functions: list[PrimitiveFunction] = Field(default_factory=list, max_length=3)
    salience: PrimitiveSalience | None = None
    pivot: PivotJustification | None = None
    stake: StakeJustification | None = None
    texture: TextureJustification | None = None
    cost: CostJustification | None = None
    complication: ComplicationJustification | None = None
    recurrence: RecurrenceJustification | None = None
    contest: ContestJustification | None = None
    narration_hooks: NarrationHooks | None = None

    @property
    def passage_ids(self) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        for passage_id in [*self.core_passage_ids, *self.support_passage_ids]:
            if not passage_id or passage_id in seen:
                continue
            seen.add(passage_id)
            combined.append(passage_id)
        return combined

    @model_validator(mode="after")
    def validate_functions(self) -> "BaseExtractionPrimitive":
        deduped_functions: list[PrimitiveFunction] = []
        seen: set[PrimitiveFunction] = set()
        for function in self.functions:
            if function in seen:
                continue
            seen.add(function)
            deduped_functions.append(function)
        self.functions = deduped_functions
        if len(self.functions) > 3:
            raise ValueError("functions must contain at most 3 entries")
        if not self.passage_ids:
            raise ValueError(
                "primitives must include at least one core or support passage id"
            )
        required_payloads = {
            PrimitiveFunction.PIVOT: self.pivot,
            PrimitiveFunction.STAKE: self.stake,
            PrimitiveFunction.TEXTURE: self.texture,
            PrimitiveFunction.COST: self.cost,
            PrimitiveFunction.COMPLICATION: self.complication,
            PrimitiveFunction.RECURRENCE: self.recurrence,
            PrimitiveFunction.CONTEST: self.contest,
        }
        for function, payload in required_payloads.items():
            if function in self.functions and payload is None:
                raise ValueError(
                    f"{function.value} requires its paired justification payload"
                )
            if function not in self.functions and payload is not None:
                raise ValueError(
                    f"{function.value} justification is not allowed unless the function tag is present"
                )
        return self


class ExtractionEventPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.EVENTS]
    event_type: str = Field(min_length=1)
    what_happened: str = Field(min_length=1)
    event_result: str = ""


class ExtractionActPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.ACTS]
    act_type: Literal[
        "decision",
        "refusal",
        "delay",
        "deferral",
        "order",
        "defection",
        "other",
    ] = "other"
    acting_subject: str = ""
    act_summary: str = Field(min_length=1)
    immediate_result: str = ""


class ExtractionUtterancePrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.UTTERANCES]
    utterance_type: Literal[
        "speech",
        "writing",
        "broadcast",
        "decree",
        "testimony",
        "manifesto",
        "letter",
        "other",
    ] = "other"
    speaker: str = ""
    audience: str = ""
    utterance_summary: str = Field(min_length=1)
    key_quote: str = ""


class ExtractionActorPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.ACTOR_PORTRAITS]
    focus_actor_id: str | None = None
    actor_label: str = ""
    goal_or_project: str = Field(min_length=1)
    stakes_or_fears: str = Field(min_length=1)
    operating_pressure: str = ""


class ExtractedMechanismPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.MECHANISMS]
    mechanism_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    failure_mode: str = ""


class ExtractedArtifactPrimitive(BaseSynthesisPrimitive):
    substrate: Literal[PrimitiveSubstrate.ARTIFACTS]
    artifact_type: Literal[
        "object",
        "place",
        "document",
        "image",
        "slogan",
        "ritual",
        "gesture",
        "detail",
        "other",
    ] = "other"
    artifact_label: str = Field(min_length=1)
    artifact_detail: str = ""


class ExtractionMechanismPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.MECHANISMS]
    mechanism_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    failure_mode: str = ""


class ExtractionConditionPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.CONDITIONS]
    condition_type: str = Field(min_length=1)
    condition_summary: str = Field(min_length=1)
    active_tension: str = ""


class ExtractionArtifactPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.ARTIFACTS]
    artifact_type: Literal[
        "object",
        "place",
        "document",
        "image",
        "slogan",
        "ritual",
        "gesture",
        "detail",
        "other",
    ] = "other"
    artifact_label: str = Field(min_length=1)
    artifact_detail: str = ""


class ExtractionReadingPrimitive(BaseExtractionPrimitive):
    substrate: Literal[PrimitiveSubstrate.READINGS]
    reading_type: Literal[
        "actor_belief",
        "historiographical_dispute",
        "interpretive_claim",
        "counterfactual",
        "other",
    ] = "other"
    subject_of_reading: str = ""
    attributed_to: str = ""
    reading_summary: str = Field(min_length=1)


AnyExtractionSynthesisPrimitive = Annotated[
    ExtractionEventPrimitive
    | ExtractionActPrimitive
    | ExtractionUtterancePrimitive
    | ExtractionActorPrimitive
    | ExtractionMechanismPrimitive
    | ExtractionConditionPrimitive
    | ExtractionArtifactPrimitive
    | ExtractionReadingPrimitive,
    Field(discriminator="substrate"),
]


AnyExtractedSynthesisPrimitive = Annotated[
    EventPrimitive
    | ActPrimitive
    | UtterancePrimitive
    | ActorPrimitive
    | ExtractedMechanismPrimitive
    | ConditionPrimitive
    | ExtractedArtifactPrimitive
    | ReadingPrimitive,
    Field(discriminator="substrate"),
]


PRIMITIVE_SUBSTRATES: tuple[str, ...] = tuple(
    member.value for member in PrimitiveSubstrate
)
PRIMITIVE_FUNCTIONS: tuple[str, ...] = tuple(member.value for member in PrimitiveFunction)
PRIMITIVE_SUBSTRATE_SET = set(PRIMITIVE_SUBSTRATES)
PRIMITIVE_FUNCTION_SET = set(PRIMITIVE_FUNCTIONS)

FULL_PRIMITIVE_SUBSTRATE_TARGET_RANGES: dict[str, tuple[int, int]] = {
    "events": (87, 120),
    "acts": (45, 59),
    "utterances": (12, 17),
    "actor_portraits": (12, 16),
    "mechanisms": (37, 51),
    "conditions": (21, 27),
    "artifacts": (18, 24),
    "readings": (11, 13),
}
MINIFIED_PRIMITIVE_SUBSTRATE_TARGET_RANGES: dict[str, tuple[int, int]] = {
    "events": (19, 27),
    "acts": (10, 13),
    "utterances": (2, 3),
    "actor_portraits": (2, 3),
    "mechanisms": (7, 12),
    "conditions": (4, 5),
    "artifacts": (3, 4),
    "readings": (2, 2),
}


def primitive_substrate_target_ranges_for_mode(
    mode: PodcastMode | str,
) -> dict[str, tuple[int, int]]:
    if PodcastMode(mode) == PodcastMode.MINIFIED:
        return dict(MINIFIED_PRIMITIVE_SUBSTRATE_TARGET_RANGES)
    return dict(FULL_PRIMITIVE_SUBSTRATE_TARGET_RANGES)


PRIMITIVE_SUBSTRATE_TARGET_RANGES = FULL_PRIMITIVE_SUBSTRATE_TARGET_RANGES


def _flatten_grouped_primitive_artifact_input(data: Any) -> Any:
    if not isinstance(data, dict) or "primitives" in data:
        return data
    if not any(substrate in data for substrate in PRIMITIVE_SUBSTRATES):
        return data

    normalized = {key: value for key, value in data.items() if key not in PRIMITIVE_SUBSTRATES}
    primitives: list[dict[str, Any]] = []
    for substrate in PRIMITIVE_SUBSTRATES:
        bucket = data.get(substrate, [])
        if bucket is None:
            continue
        if not isinstance(bucket, list):
            raise TypeError(f"{substrate} must be a list")
        for item in bucket:
            if not isinstance(item, dict):
                raise TypeError(f"{substrate} items must be objects")
            primitive_payload = dict(item)
            primitive_payload.setdefault("substrate", substrate)
            primitives.append(primitive_payload)
    normalized["primitives"] = primitives
    return normalized


def _serialize_grouped_primitive_artifact(
    *,
    project_id: str,
    primitives: list[Any],
    quality_score: float,
    quality_notes: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        substrate: [] for substrate in PRIMITIVE_SUBSTRATES
    }
    for primitive in primitives:
        payload = (
            primitive.model_dump(mode="json")
            if hasattr(primitive, "model_dump")
            else dict(primitive)
        )
        substrate_value = str(payload.pop("substrate", "") or "").strip()
        if substrate_value not in grouped:
            raise ValueError(f"unknown primitive substrate during serialization: {substrate_value}")
        grouped[substrate_value].append(payload)
    return {
        "project_id": project_id,
        **grouped,
        "quality_score": quality_score,
        "quality_notes": list(quality_notes),
    }


class GroupedExtractionPrimitiveBase(StrictModel):
    title: str = Field(min_length=1)
    core_passage_ids: list[str] = Field(default_factory=list)
    support_passage_ids: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    geography: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    functions: list[PrimitiveFunction] = Field(default_factory=list, max_length=3)
    salience: PrimitiveSalience | None = None
    pivot: PivotJustification | None = None
    stake: StakeJustification | None = None
    texture: TextureJustification | None = None
    cost: CostJustification | None = None
    complication: ComplicationJustification | None = None
    recurrence: RecurrenceJustification | None = None
    contest: ContestJustification | None = None
    narration_hooks: NarrationHooks | None = None

    @property
    def passage_ids(self) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        for passage_id in [*self.core_passage_ids, *self.support_passage_ids]:
            if not passage_id or passage_id in seen:
                continue
            seen.add(passage_id)
            combined.append(passage_id)
        return combined

    @model_validator(mode="after")
    def validate_functions(self) -> "GroupedExtractionPrimitiveBase":
        deduped_functions: list[PrimitiveFunction] = []
        seen: set[PrimitiveFunction] = set()
        for function in self.functions:
            if function in seen:
                continue
            seen.add(function)
            deduped_functions.append(function)
        self.functions = deduped_functions
        if len(self.functions) > 3:
            raise ValueError("functions must contain at most 3 entries")
        if not self.passage_ids:
            raise ValueError(
                "primitives must include at least one core or support passage id"
            )
        required_payloads = {
            PrimitiveFunction.PIVOT: self.pivot,
            PrimitiveFunction.STAKE: self.stake,
            PrimitiveFunction.TEXTURE: self.texture,
            PrimitiveFunction.COST: self.cost,
            PrimitiveFunction.COMPLICATION: self.complication,
            PrimitiveFunction.RECURRENCE: self.recurrence,
            PrimitiveFunction.CONTEST: self.contest,
        }
        for function, payload in required_payloads.items():
            if function in self.functions and payload is None:
                raise ValueError(
                    f"{function.value} requires its paired justification payload"
                )
            if function not in self.functions and payload is not None:
                raise ValueError(
                    f"{function.value} justification is not allowed unless the function tag is present"
                )
        return self


class GroupedExtractionEventPrimitive(GroupedExtractionPrimitiveBase):
    event_type: str = Field(min_length=1)
    what_happened: str = Field(min_length=1)
    event_result: str = ""


class GroupedExtractionActPrimitive(GroupedExtractionPrimitiveBase):
    act_type: Literal[
        "decision",
        "refusal",
        "delay",
        "deferral",
        "order",
        "defection",
        "other",
    ] = "other"
    acting_subject: str = ""
    act_summary: str = Field(min_length=1)
    immediate_result: str = ""


class GroupedExtractionUtterancePrimitive(GroupedExtractionPrimitiveBase):
    utterance_type: Literal[
        "speech",
        "writing",
        "broadcast",
        "decree",
        "testimony",
        "manifesto",
        "letter",
        "other",
    ] = "other"
    speaker: str = ""
    audience: str = ""
    utterance_summary: str = Field(min_length=1)
    key_quote: str = ""


class GroupedExtractionActorPrimitive(GroupedExtractionPrimitiveBase):
    focus_actor_id: str | None = None
    actor_label: str = ""
    goal_or_project: str = Field(min_length=1)
    stakes_or_fears: str = Field(min_length=1)
    operating_pressure: str = ""


class GroupedExtractionMechanismPrimitive(GroupedExtractionPrimitiveBase):
    mechanism_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    failure_mode: str = ""


class GroupedExtractionConditionPrimitive(GroupedExtractionPrimitiveBase):
    condition_type: str = Field(min_length=1)
    condition_summary: str = Field(min_length=1)
    active_tension: str = ""


class GroupedExtractionArtifactPrimitive(GroupedExtractionPrimitiveBase):
    artifact_type: Literal[
        "object",
        "place",
        "document",
        "image",
        "slogan",
        "ritual",
        "gesture",
        "detail",
        "other",
    ] = "other"
    artifact_label: str = Field(min_length=1)
    artifact_detail: str = ""


class GroupedExtractionReadingPrimitive(GroupedExtractionPrimitiveBase):
    reading_type: Literal[
        "actor_belief",
        "historiographical_dispute",
        "interpretive_claim",
        "counterfactual",
        "other",
    ] = "other"
    subject_of_reading: str = ""
    attributed_to: str = ""
    reading_summary: str = Field(min_length=1)


class ExtractedPrimitiveArtifactBase(StrictModel):
    project_id: str
    primitives: list[AnyExtractedSynthesisPrimitive] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flatten_grouped_input(cls, data: Any) -> Any:
        return _flatten_grouped_primitive_artifact_input(data)

    def primitive_by_id(self) -> dict[str, BaseSynthesisPrimitive]:
        return {primitive.id: primitive for primitive in self.primitives}

    def primitives_by_substrate(self) -> dict[str, list[BaseSynthesisPrimitive]]:
        grouped: dict[str, list[BaseSynthesisPrimitive]] = {
            substrate: [] for substrate in PRIMITIVE_SUBSTRATES
        }
        for primitive in self.primitives:
            grouped[primitive.substrate.value].append(primitive)
        return grouped

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ExtractedPrimitiveArtifactBase":
        primitive_ids = [primitive.id for primitive in self.primitives]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("primitives must use unique ids")
        return self

    @model_serializer(mode="plain")
    def serialize_grouped(self) -> dict[str, Any]:
        return _serialize_grouped_primitive_artifact(
            project_id=self.project_id,
            primitives=self.primitives,
            quality_score=self.quality_score,
            quality_notes=self.quality_notes,
        )


class ExtractionPrimitiveArtifactBase(StrictModel):
    project_id: str
    primitives: list[AnyExtractionSynthesisPrimitive] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flatten_grouped_input(cls, data: Any) -> Any:
        return _flatten_grouped_primitive_artifact_input(data)

    @model_serializer(mode="plain")
    def serialize_grouped(self) -> dict[str, Any]:
        return _serialize_grouped_primitive_artifact(
            project_id=self.project_id,
            primitives=self.primitives,
            quality_score=self.quality_score,
            quality_notes=self.quality_notes,
        )


class PrimitiveArtifactBase(StrictModel):
    project_id: str
    primitives: list[AnySynthesisPrimitive] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flatten_grouped_input(cls, data: Any) -> Any:
        return _flatten_grouped_primitive_artifact_input(data)

    def primitive_by_id(self) -> dict[str, BaseSynthesisPrimitive]:
        return {primitive.id: primitive for primitive in self.primitives}

    def primitives_by_substrate(self) -> dict[str, list[AnySynthesisPrimitive]]:
        grouped: dict[str, list[AnySynthesisPrimitive]] = {
            substrate: [] for substrate in PRIMITIVE_SUBSTRATES
        }
        for primitive in self.primitives:
            grouped[primitive.substrate.value].append(primitive)
        return grouped

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "PrimitiveArtifactBase":
        primitive_ids = [primitive.id for primitive in self.primitives]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("primitives must use unique ids")
        return self

    @model_serializer(mode="plain")
    def serialize_grouped(self) -> dict[str, Any]:
        return _serialize_grouped_primitive_artifact(
            project_id=self.project_id,
            primitives=self.primitives,
            quality_score=self.quality_score,
            quality_notes=self.quality_notes,
        )


class SynthesisPrimitivesArtifact(ExtractedPrimitiveArtifactBase):
    """Step-1 substrate extraction artifact."""


class RawSynthesisPrimitivesArtifact(StrictModel):
    """Model-facing grouped extraction artifact before local primitive ids are materialized."""

    project_id: str
    events: list[GroupedExtractionEventPrimitive] = Field(default_factory=list)
    acts: list[GroupedExtractionActPrimitive] = Field(default_factory=list)
    utterances: list[GroupedExtractionUtterancePrimitive] = Field(default_factory=list)
    actor_portraits: list[GroupedExtractionActorPrimitive] = Field(default_factory=list)
    mechanisms: list[GroupedExtractionMechanismPrimitive] = Field(default_factory=list)
    conditions: list[GroupedExtractionConditionPrimitive] = Field(default_factory=list)
    artifacts: list[GroupedExtractionArtifactPrimitive] = Field(default_factory=list)
    readings: list[GroupedExtractionReadingPrimitive] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitives_by_substrate(self) -> dict[str, list[GroupedExtractionPrimitiveBase]]:
        return {
            PrimitiveSubstrate.EVENTS.value: list(self.events),
            PrimitiveSubstrate.ACTS.value: list(self.acts),
            PrimitiveSubstrate.UTTERANCES.value: list(self.utterances),
            PrimitiveSubstrate.ACTOR_PORTRAITS.value: list(self.actor_portraits),
            PrimitiveSubstrate.MECHANISMS.value: list(self.mechanisms),
            PrimitiveSubstrate.CONDITIONS.value: list(self.conditions),
            PrimitiveSubstrate.ARTIFACTS.value: list(self.artifacts),
            PrimitiveSubstrate.READINGS.value: list(self.readings),
        }


class PrimitiveFunctionTaggingArtifact(PrimitiveArtifactBase):
    """Step-2 post-tagging artifact."""


class PrimitiveFunctionTaggingOverlayArtifact(StrictModel):
    project_id: str
    overlays_by_id: dict[str, PrimitiveEnrichmentOverlay] = Field(default_factory=dict)


def apply_primitive_enrichment_overlay(
    primitive: BaseSynthesisPrimitive | dict[str, Any],
    overlay: PrimitiveEnrichmentOverlay,
) -> dict[str, Any]:
    base_payload = (
        primitive.model_dump(mode="json")
        if isinstance(primitive, BaseSynthesisPrimitive)
        else dict(primitive)
    )
    updates = overlay.model_dump(
        mode="json",
        include=overlay.model_fields_set,
    )
    base_payload.update(updates)
    return base_payload


class SynthesisMap(PrimitiveArtifactBase):
    """Retained primitive artifact consumed by downstream stages."""


class EpisodeSpine(StrictModel):
    listener_problem: str = Field(
        min_length=1,
        validation_alias=AliasChoices("listener_problem", "listener_question"),
        serialization_alias="listener_problem",
    )
    episode_answer: str = Field(
        min_length=1,
        validation_alias=AliasChoices("episode_answer", "argument"),
        serialization_alias="episode_answer",
    )
    pressure_line: str = ""
    core_primitive_ids: list[str] = Field(min_length=1)
    support_primitive_roles: dict[str, SupportPrimitiveRole] = Field(
        default_factory=dict
    )
    recall_primitive_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spine(self) -> "EpisodeSpine":
        self.listener_problem = str(self.listener_problem or "").strip()
        self.episode_answer = str(self.episode_answer or "").strip()
        self.pressure_line = str(self.pressure_line or "").strip()
        if not self.pressure_line:
            self.pressure_line = self.episode_answer
        if not self.listener_problem:
            raise ValueError("listener_problem must not be blank")
        if not self.episode_answer:
            raise ValueError("episode_answer must not be blank")
        if not self.pressure_line:
            raise ValueError("pressure_line must not be blank")
        seen_primitive_ids: set[str] = set()
        deduped_core_primitive_ids: list[str] = []
        for primitive_id in self.core_primitive_ids:
            if not primitive_id or primitive_id in seen_primitive_ids:
                continue
            seen_primitive_ids.add(primitive_id)
            deduped_core_primitive_ids.append(primitive_id)
        self.core_primitive_ids = deduped_core_primitive_ids
        if not self.core_primitive_ids:
            raise ValueError(
                "core_primitive_ids must contain at least one primitive id"
            )
        if len(self.core_primitive_ids) < 2 or len(self.core_primitive_ids) > 11:
            raise ValueError("core_primitive_ids must contain 2-11 primitive ids")
        if (
            len(self.support_primitive_roles) < 2
            or len(self.support_primitive_roles) > 13
        ):
            raise ValueError("support_primitive_roles must contain 2-13 primitive ids")

        overlap = sorted(
            set(self.core_primitive_ids).intersection(self.support_primitive_roles)
        )
        if overlap:
            raise ValueError(
                f"support primitives cannot also appear in core_primitive_ids: {overlap}"
            )

        recall_primitive_ids: list[str] = []
        seen_recall_ids: set[str] = set()
        reserved_primitive_ids = set(self.core_primitive_ids).union(
            self.support_primitive_roles
        )
        for primitive_id in self.recall_primitive_ids:
            if (
                not primitive_id
                or primitive_id in seen_recall_ids
                or primitive_id in reserved_primitive_ids
            ):
                continue
            seen_recall_ids.add(primitive_id)
            recall_primitive_ids.append(primitive_id)
        self.recall_primitive_ids = recall_primitive_ids
        if len(self.recall_primitive_ids) > 3:
            raise ValueError(
                "recall_primitive_ids must contain at most 3 primitive ids"
            )
        return self

    @property
    def assigned_primitive_ids(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for primitive_id in [
            *self.core_primitive_ids,
            *self.support_primitive_roles.keys(),
            *self.recall_primitive_ids,
        ]:
            if not primitive_id or primitive_id in seen:
                continue
            seen.add(primitive_id)
            ordered.append(primitive_id)
        return ordered


def validate_episode_spine_targets(
    spine: EpisodeSpine,
    *,
    core_target_min: int,
    core_target_max: int,
    support_target_min: int,
    support_target_max: int,
    recall_target_max: int,
) -> None:
    core_count = len(spine.core_primitive_ids)
    if core_count < core_target_min or core_count > core_target_max:
        raise ValueError(
            "core_primitive_ids must contain "
            f"{core_target_min}-{core_target_max} primitive ids"
        )
    support_count = len(spine.support_primitive_roles)
    if support_count < support_target_min or support_count > support_target_max:
        raise ValueError(
            "support_primitive_roles must contain "
            f"{support_target_min}-{support_target_max} primitive ids"
        )
    recall_count = len(spine.recall_primitive_ids)
    if recall_count > recall_target_max:
        primitive_label = "primitive id" if recall_target_max == 1 else "primitive ids"
        raise ValueError(
            "recall_primitive_ids must contain at most "
            f"{recall_target_max} {primitive_label}"
        )


# ---------------------------------------------------------------------------
# 3.5 Episode Planning Models
# ---------------------------------------------------------------------------


class ChronologyBreak(StrictModel):
    break_type: Literal["setup_jump", "flashback", "payoff_return"]
    note: str = Field(min_length=1)


class ActorArcThread(StrictModel):
    model_config = ConfigDict(extra="ignore")

    thread_id: str = Field(min_length=1)
    arc_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    premise: str = Field(min_length=1)
    pressure: str = ""
    movement: str = ""
    resolution: str = Field(
        default="",
        validation_alias=AliasChoices("resolution", "payoff"),
        serialization_alias="resolution",
    )


class ActorArcDirective(StrictModel):
    actor_id: str = Field(min_length=1)
    arc_threads: list[ActorArcThread] = Field(
        default_factory=list, min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def validate_unique_thread_ids(self) -> "ActorArcDirective":
        thread_ids = [thread.thread_id for thread in self.arc_threads]
        if len(thread_ids) != len(set(thread_ids)):
            raise ValueError("actor arc directive thread ids must be unique")
        return self


class EpisodeAuthorialContract(StrictModel):
    analysis_weight: Literal["light", "medium", "heavy"] = "medium"
    priority_moves: list[
        Literal[
            "quote_then_gloss",
            "doctrinal_unpack",
            "institutional_clarifier",
            "causal_compression",
            "comparative_aside",
            "verdict_landing",
        ]
    ] = Field(default_factory=list, max_length=3)
    governing_lenses: list[str] = Field(default_factory=list, max_length=3)
    must_clarify_terms: list[str] = Field(default_factory=list, max_length=4)
    must_clarify_institutions: list[str] = Field(default_factory=list, max_length=4)
    introduce_explanation_item_ids: list[str] = Field(
        default_factory=list, max_length=4
    )
    remind_explanation_item_ids: list[str] = Field(default_factory=list, max_length=4)
    introduce_actor_ids: list[str] = Field(default_factory=list, max_length=4)
    remind_actor_ids: list[str] = Field(default_factory=list, max_length=4)
    callback_obligations: list[str] = Field(default_factory=list, max_length=3)


class EpisodeNarratorContract(StrictModel):
    analysis_weight: Literal["light", "medium", "heavy"] = "medium"
    priority_moves: list[
        Literal[
            "quote_then_gloss",
            "doctrinal_unpack",
            "institutional_clarifier",
            "causal_compression",
            "comparative_aside",
            "verdict_landing",
        ]
    ] = Field(default_factory=list, max_length=3)
    governing_lenses: list[str] = Field(default_factory=list, max_length=3)


class ListenerQuestionMove(StrictModel):
    question_id: str = Field(min_length=1)
    action: Literal["open", "advance", "resolve", "reframe"]
    text: str = ""

    @model_validator(mode="after")
    def validate_move(self) -> "ListenerQuestionMove":
        if self.action == "open" and not _nonempty_text(self.text):
            raise ValueError("listener question open moves must include text")
        return self


class ListenerMemoryThreadMove(StrictModel):
    thread_id: str = Field(min_length=1)
    action: Literal["open", "refresh", "payoff", "retire"]
    label: str = ""
    source_promised_beat_id: str | None = None

    @model_validator(mode="after")
    def validate_move(self) -> "ListenerMemoryThreadMove":
        if self.action == "open" and not _nonempty_text(self.label):
            raise ValueError("listener memory-thread open moves must include label")
        return self


class ContinuityCarryItem(StrictModel):
    item_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: Literal[
        "takeaway",
        "question",
        "memory_thread",
        "actor",
        "institution",
        "contrast",
        "residue",
    ] = "takeaway"
    source_episode_number: int = Field(default=0, ge=0)
    source_section_id: str | None = None
    priority: Literal["high", "normal"] = "normal"
    desired_surface: Literal["recap", "opening", "body", "closing"] = "body"
    recommended_action: Literal["remind", "payoff", "keep_live"] = "keep_live"
    source_promised_beat_id: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "ContinuityCarryItem":
        self.item_id = _nonempty_text(self.item_id)
        self.label = _nonempty_text(self.label)
        self.source_section_id = _nonempty_text(self.source_section_id) or None
        self.source_promised_beat_id = _nonempty_text(self.source_promised_beat_id) or None
        if not self.item_id:
            raise ValueError("continuity carry items must include item_id")
        if not self.label:
            raise ValueError("continuity carry items must include label")
        return self

    @classmethod
    def from_legacy_string(
        cls,
        value: str,
        *,
        source_episode_number: int = 0,
    ) -> "ContinuityCarryItem":
        label = _nonempty_text(value)
        return cls(
            item_id=_legacy_continuity_item_id(label),
            label=label,
            kind="takeaway",
            source_episode_number=source_episode_number,
            priority="normal",
            desired_surface="body",
            recommended_action="keep_live",
        )


def _coerce_continuity_carry_items(
    values: Any,
    *,
    source_episode_number: int = 0,
) -> list[ContinuityCarryItem]:
    coerced: list[ContinuityCarryItem] = []
    for value in list(values or []):
        if isinstance(value, ContinuityCarryItem):
            coerced.append(value)
            continue
        if isinstance(value, str):
            text = _nonempty_text(value)
            if text:
                coerced.append(
                    ContinuityCarryItem.from_legacy_string(
                        text, source_episode_number=source_episode_number
                    )
                )
            continue
        if isinstance(value, dict):
            coerced.append(ContinuityCarryItem.model_validate(value))
            continue
        raise TypeError("continuity carry items must be strings, dicts, or ContinuityCarryItem values")
    return coerced


class ListenerEpisodeAgenda(StrictModel):
    introduce_explanation_item_ids: list[str] = Field(default_factory=list, max_length=4)
    remind_explanation_item_ids: list[str] = Field(default_factory=list, max_length=4)
    introduce_actor_ids: list[str] = Field(default_factory=list, max_length=4)
    remind_actor_ids: list[str] = Field(default_factory=list, max_length=4)
    question_moves: list[ListenerQuestionMove] = Field(default_factory=list, max_length=6)
    memory_thread_moves: list[ListenerMemoryThreadMove] = Field(
        default_factory=list, max_length=6
    )
    carry_forward_memory: list[ContinuityCarryItem] = Field(default_factory=list, max_length=4)
    episode_takeaway: str = ""

    @field_validator("carry_forward_memory", mode="before")
    @classmethod
    def validate_carry_forward_memory(cls, value: Any) -> list[ContinuityCarryItem]:
        return _coerce_continuity_carry_items(value)


class HostMysteryMove(StrictModel):
    mystery_id: str = Field(min_length=1)
    action: Literal["open", "advance", "resolve", "reframe"]
    text: str = ""

    @model_validator(mode="after")
    def validate_move(self) -> "HostMysteryMove":
        if self.action == "open" and not _nonempty_text(self.text):
            raise ValueError("host mystery open moves must include text")
        return self


class HostAssumptionMove(StrictModel):
    assumption_id: str = Field(min_length=1)
    action: Literal["introduce", "weaken", "revise", "drop"]
    statement: str = ""
    revised_statement: str = ""

    @model_validator(mode="after")
    def validate_move(self) -> "HostAssumptionMove":
        if self.action in {"introduce", "revise"} and not _nonempty_text(self.statement):
            raise ValueError("host assumption introduce/revise moves must include statement")
        return self


class HostTheoryMove(StrictModel):
    theory_id: str = Field(min_length=1)
    action: Literal["propose", "strengthen", "replace", "retire"]
    statement: str = ""

    @model_validator(mode="after")
    def validate_move(self) -> "HostTheoryMove":
        if self.action in {"propose", "replace"} and not _nonempty_text(self.statement):
            raise ValueError("host theory propose/replace moves must include statement")
        return self


class HostEpisodeAgenda(StrictModel):
    mystery_moves: list[HostMysteryMove] = Field(default_factory=list, max_length=6)
    assumption_moves: list[HostAssumptionMove] = Field(default_factory=list, max_length=6)
    theory_moves: list[HostTheoryMove] = Field(default_factory=list, max_length=6)
    intended_revision_beats: list[str] = Field(default_factory=list, max_length=4)
    episode_takeaway: str = ""


class EpisodeNarrativeAgenda(StrictModel):
    listener: ListenerEpisodeAgenda = Field(default_factory=ListenerEpisodeAgenda)
    host: HostEpisodeAgenda = Field(default_factory=HostEpisodeAgenda)


class ListenerQuestionState(StrictModel):
    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["open", "advanced", "resolved", "reframed"] = "open"


class ListenerMemoryThreadState(StrictModel):
    thread_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: Literal["open", "refreshed", "paid_off", "retired"] = "open"
    source_promised_beat_id: str | None = None


class ListenerNarrativeState(StrictModel):
    known_explanation_item_ids: list[str] = Field(default_factory=list)
    known_actor_ids: list[str] = Field(default_factory=list)
    questions: list[ListenerQuestionState] = Field(default_factory=list)
    memory_threads: list[ListenerMemoryThreadState] = Field(default_factory=list)
    carry_forward_memory: list[ContinuityCarryItem] = Field(default_factory=list, max_length=4)
    last_episode_takeaway: str = ""

    @field_validator("carry_forward_memory", mode="before")
    @classmethod
    def validate_carry_forward_memory(cls, value: Any) -> list[ContinuityCarryItem]:
        return _coerce_continuity_carry_items(value)


class HostMysteryState(StrictModel):
    mystery_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["open", "advanced", "resolved", "reframed"] = "open"


class HostAssumptionState(StrictModel):
    assumption_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: Literal["active", "weakened", "revised", "dropped"] = "active"


class HostTheoryState(StrictModel):
    theory_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: Literal["proposed", "strengthened", "replaced", "retired"] = "proposed"


class HostNarrativeState(StrictModel):
    mysteries: list[HostMysteryState] = Field(default_factory=list)
    assumptions: list[HostAssumptionState] = Field(default_factory=list)
    working_theories: list[HostTheoryState] = Field(default_factory=list)
    recent_revisions: list[str] = Field(default_factory=list, max_length=4)
    confidence_posture: Literal["tentative", "mixed", "grounded"] = "mixed"
    last_episode_takeaway: str = ""


class NarrativeState(StrictModel):
    project_id: str = Field(min_length=1)
    next_episode_number: int = Field(default=1, ge=1)
    listener: ListenerNarrativeState = Field(default_factory=ListenerNarrativeState)
    host: HostNarrativeState = Field(default_factory=HostNarrativeState)


class NarrativeStateDelta(StrictModel):
    introduced_explanation_item_ids: list[str] = Field(default_factory=list)
    introduced_actor_ids: list[str] = Field(default_factory=list)
    listener_question_updates: list[ListenerQuestionState] = Field(default_factory=list)
    listener_memory_thread_updates: list[ListenerMemoryThreadState] = Field(
        default_factory=list
    )
    host_mystery_updates: list[HostMysteryState] = Field(default_factory=list)
    host_assumption_updates: list[HostAssumptionState] = Field(default_factory=list)
    host_theory_updates: list[HostTheoryState] = Field(default_factory=list)
    listener_carry_forward_memory: list[ContinuityCarryItem] = Field(
        default_factory=list, max_length=4
    )
    listener_episode_takeaway: str = ""
    host_recent_revisions: list[str] = Field(default_factory=list, max_length=4)
    host_episode_takeaway: str = ""
    host_confidence_posture: Literal["tentative", "mixed", "grounded"] = "mixed"

    @field_validator("listener_carry_forward_memory", mode="before")
    @classmethod
    def validate_listener_carry_forward_memory(
        cls, value: Any
    ) -> list[ContinuityCarryItem]:
        return _coerce_continuity_carry_items(value)


class NarrativeStateReconciliation(StrictModel):
    episode_number: int = Field(ge=1)
    state_post: NarrativeState
    delta: NarrativeStateDelta = Field(default_factory=NarrativeStateDelta)
    warnings: list[str] = Field(default_factory=list)
    rationale: str = ""


class SeriesExplanationItem(StrictModel):
    item_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list, max_length=4)
    kind: Literal["term", "institution"]
    importance: Literal["foundational", "episode_core"]
    introduction_episode_number: int = Field(ge=1)
    first_definition_depth: Literal["concise", "full"] = "full"
    preferred_plain_gloss: str = Field(min_length=1)
    later_episode_policy: Literal["mention_only", "brief_reminder"] = "brief_reminder"
    max_reminder_sentences: Literal[1, 2] = 1


class SeriesActorExplanationItem(StrictModel):
    actor_id: str = Field(min_length=1)
    introduction_episode_number: int = Field(ge=1)
    first_background_depth: Literal["appositive", "full"] = "appositive"
    later_episode_policy: Literal["name_only", "brief_reminder"] = "brief_reminder"
    preferred_plain_gloss: str = ""


class SceneDiscoveryCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    primitive_ids: list[str] = Field(default_factory=list, min_length=1)
    passage_ids: list[str] = Field(default_factory=list, min_length=1)
    scene_sketch: str = Field(min_length=1)
    scene_jobs: list[SceneJob] = Field(default_factory=list, min_length=1, max_length=2)
    anchor_image: str = Field(min_length=1)
    why_sceneable: str = Field(min_length=1)
    quote_anchor: str = ""
    actor_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_references(self) -> "SceneDiscoveryCandidate":
        self.primitive_ids = list(dict.fromkeys(_nonempty_text(item) for item in self.primitive_ids if _nonempty_text(item)))
        self.passage_ids = list(dict.fromkeys(_nonempty_text(item) for item in self.passage_ids if _nonempty_text(item)))
        self.actor_ids = list(dict.fromkeys(_nonempty_text(item) for item in self.actor_ids if _nonempty_text(item)))
        self.scene_jobs = list(dict.fromkeys(self.scene_jobs))
        if not self.primitive_ids:
            raise ValueError("scene discovery candidates must reference primitive_ids")
        if not self.passage_ids:
            raise ValueError("scene discovery candidates must reference passage_ids")
        if any(scene_job == SceneJob.CLOSE for scene_job in self.scene_jobs):
            raise ValueError("scene discovery candidates may not use close scene_jobs")
        return self


class SceneDiscoveryArtifact(StrictModel):
    candidates: list[SceneDiscoveryCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "SceneDiscoveryArtifact":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("scene discovery candidates must use unique candidate_id values")
        return self


class PromisedBeat(StrictModel):
    beat_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    intended_job: SceneJob
    source_candidate_ids: list[str] = Field(default_factory=list, max_length=4)
    source_primitive_ids: list[str] = Field(default_factory=list, max_length=6)
    why_load_bearing: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "PromisedBeat":
        self.source_candidate_ids = list(
            dict.fromkeys(
                _nonempty_text(item)
                for item in self.source_candidate_ids
                if _nonempty_text(item)
            )
        )
        self.source_primitive_ids = list(
            dict.fromkeys(
                _nonempty_text(item)
                for item in self.source_primitive_ids
                if _nonempty_text(item)
            )
        )
        if not self.source_candidate_ids and not self.source_primitive_ids:
            raise ValueError(
                "promised beats must reference at least one source_candidate_id or source_primitive_id"
            )
        return self


class NegativeScope(StrictModel):
    boundary: str = ""
    excluded_topics: list[str] = Field(default_factory=list, max_length=5)
    tempting_but_out: list[str] = Field(default_factory=list, max_length=3)
    omission_logic: str = ""


class PromisedBeatDecisionRecord(StrictModel):
    beat_id: str = Field(min_length=1)
    decision: PromisedBeatDecision
    section_id: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "PromisedBeatDecisionRecord":
        section_id = _nonempty_text(self.section_id)
        reason = _nonempty_text(self.reason)
        self.section_id = section_id or None
        self.reason = reason
        if self.decision == PromisedBeatDecision.STAGE:
            if not self.section_id:
                raise ValueError("staged promised beats must include section_id")
        else:
            if not self.reason:
                raise ValueError("deferred or dropped promised beats must include reason")
        return self


class StrategyEpisode(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    thematic_focus: str = Field(default="")
    arc_summary: str = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    episode_spine: EpisodeSpine
    actor_arc_directives: list[ActorArcDirective] = Field(
        default_factory=list, max_length=4
    )
    narrator_contract: EpisodeNarratorContract = Field(
        default_factory=EpisodeNarratorContract
    )
    authorial_contract: EpisodeAuthorialContract = Field(
        default_factory=EpisodeAuthorialContract
    )
    narrative_agenda: EpisodeNarrativeAgenda = Field(
        default_factory=EpisodeNarrativeAgenda
    )
    promised_beats: list[PromisedBeat] = Field(default_factory=list, max_length=12)
    negative_scope: NegativeScope = Field(default_factory=NegativeScope)

    @model_validator(mode="after")
    def validate_commitments(self) -> "StrategyEpisode":
        if not self.narrator_contract.model_fields_set:
            self.narrator_contract = EpisodeNarratorContract(
                analysis_weight=self.authorial_contract.analysis_weight,
                priority_moves=list(self.authorial_contract.priority_moves),
                governing_lenses=list(self.authorial_contract.governing_lenses),
            )
        listener_agenda = self.narrative_agenda.listener
        if not listener_agenda.model_fields_set:
            question_moves = [
                ListenerQuestionMove(
                    question_id=f"episode_{self.episode_number}_question_{index}",
                    action="open",
                    text=question,
                )
                for index, question in enumerate(self.unresolved_questions, start=1)
                if _nonempty_text(question)
            ]
            memory_thread_moves = [
                ListenerMemoryThreadMove(
                    thread_id=f"episode_{self.episode_number}_callback_{index}",
                    action="open",
                    label=callback,
                )
                for index, callback in enumerate(
                    self.authorial_contract.callback_obligations, start=1
                )
                if _nonempty_text(callback)
            ]
            self.narrative_agenda = EpisodeNarrativeAgenda(
                listener=ListenerEpisodeAgenda(
                    introduce_explanation_item_ids=list(
                        self.authorial_contract.introduce_explanation_item_ids
                    ),
                    remind_explanation_item_ids=list(
                        self.authorial_contract.remind_explanation_item_ids
                    ),
                    introduce_actor_ids=list(self.authorial_contract.introduce_actor_ids),
                    remind_actor_ids=list(self.authorial_contract.remind_actor_ids),
                    question_moves=question_moves,
                    memory_thread_moves=memory_thread_moves,
                    carry_forward_memory=[
                        ContinuityCarryItem(
                            item_id=f"episode_{self.episode_number}_carry_term_{index}",
                            label=item,
                            kind="institution" if item in self.authorial_contract.must_clarify_institutions else "takeaway",
                            source_episode_number=self.episode_number,
                            priority="normal",
                            desired_surface="body",
                            recommended_action="keep_live",
                        )
                        for index, item in enumerate(
                            [
                                *list(self.authorial_contract.must_clarify_terms),
                                *list(self.authorial_contract.must_clarify_institutions),
                            ],
                            start=1,
                        )
                        if _nonempty_text(item)
                    ][:4],
                ),
                host=HostEpisodeAgenda(),
            )
        beat_ids = [beat.beat_id for beat in self.promised_beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("promised_beats must use unique beat_id values")
        answer_count = sum(
            1 for beat in self.promised_beats if beat.intended_job == SceneJob.ANSWER
        )
        if answer_count > 1:
            raise ValueError("promised_beats may contain at most one answer beat")
        return self


class StrategyEpisodeSkeleton(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    thematic_focus: str = Field(default="")
    arc_summary: str = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    episode_spine: EpisodeSpine
    actor_arc_directives: list[ActorArcDirective] = Field(
        default_factory=list, max_length=4
    )
    negative_scope: NegativeScope = Field(default_factory=NegativeScope)


class NarrativeStrategySkeleton(StrictModel):
    strategy_type: Literal[
        "thesis_driven", "debate", "chronological", "convergence", "mosaic"
    ]
    justification: str = Field(min_length=1)
    series_arc: str = Field(min_length=1)
    episode_arc_outline: list[str] = Field(default_factory=list)
    recommended_episode_count: int | None = Field(default=None, ge=1)
    episodes: list[StrategyEpisodeSkeleton] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self) -> "NarrativeStrategySkeleton":
        episode_numbers = [episode.episode_number for episode in self.episodes]
        if len(episode_numbers) != len(set(episode_numbers)):
            raise ValueError(
                "episodes must not contain duplicate episode_number values"
            )
        if self.episode_arc_outline and len(self.episode_arc_outline) != len(
            self.episodes
        ):
            raise ValueError(
                "episode_arc_outline must align with episodes when provided"
            )
        if self.recommended_episode_count is not None and self.episodes:
            if self.recommended_episode_count != len(self.episodes):
                raise ValueError(
                    "recommended_episode_count must match the number of episodes"
                )
        return self


_DEFAULT_NARRATOR_ALLOWED_MOVES: tuple[str, ...] = (
    "orient",
    "clarify",
    "evaluate",
    "contrast",
    "callback",
    "light_aside",
    "naming_note",
    "uncertainty",
    "revision",
    "surprise",
)

COMPARATIVE_ASIDE_TOLERANCE: Literal["high"] = "high"


def effective_narrator_allowed_moves(
    allowed_moves: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    normalized = list(
        dict.fromkeys(
            str(move).strip() for move in (allowed_moves or []) if str(move).strip()
        )
    )
    if not normalized:
        return list(_DEFAULT_NARRATOR_ALLOWED_MOVES)
    for move in ("uncertainty", "revision", "surprise"):
        if move not in normalized:
            normalized.append(move)
    return [move for move in _DEFAULT_NARRATOR_ALLOWED_MOVES if move in normalized]


class SeriesNarratorProfile(StrictModel):
    presence_mode: Literal["visible_host"] = "visible_host"
    baseline_tone: Literal["dry", "plainspoken", "wry", "grave"] = "plainspoken"
    spoken_style_contract: Literal["classic", "anti_academic_oral"] = (
        "anti_academic_oral"
    )
    allowed_moves: list[
        Literal[
            "orient",
            "clarify",
            "evaluate",
            "contrast",
            "callback",
            "light_aside",
            "naming_note",
            "uncertainty",
            "revision",
            "surprise",
        ]
    ] = Field(
        default_factory=lambda: list(_DEFAULT_NARRATOR_ALLOWED_MOVES)
    )
    forbidden_moves: list[
        Literal["unsupported_psychology", "cheap_joke", "fake_banter", "teaser_hype"]
    ] = Field(
        default_factory=lambda: [
            "unsupported_psychology",
            "cheap_joke",
            "fake_banter",
            "teaser_hype",
        ]
    )
    target_full_phase_scene_coverage_min: float = Field(default=0.60, ge=0.0, le=1.0)
    target_full_phase_scene_coverage_target: float = Field(
        default=0.75, ge=0.0, le=1.0
    )
    analysis_mode: Literal["scene_led", "hybrid", "analysis_forward"] = "hybrid"
    analysis_density: Literal["light", "medium", "high"] = "medium"
    quote_gloss_preference: Literal["avoid", "allow", "prefer"] = "allow"
    clarifier_tolerance: Literal["low", "medium", "high"] = "medium"
    wit_ceiling: Literal["none", "dry", "wry"] = "dry"
    target_authorial_passages_per_episode: int = Field(
        default=authorial_passage_target_for_mode(PodcastMode.FULL), ge=0, le=40
    )

    @model_validator(mode="after")
    def validate_coverage_targets(self) -> "SeriesNarratorProfile":
        if (
            self.target_full_phase_scene_coverage_target
            < self.target_full_phase_scene_coverage_min
        ):
            raise ValueError(
                "target_full_phase_scene_coverage_target must be greater than or "
                "equal to target_full_phase_scene_coverage_min"
            )
        self.allowed_moves = effective_narrator_allowed_moves(self.allowed_moves)
        return self


class NarrativeStrategy(StrictModel):
    strategy_type: Literal[
        "thesis_driven", "debate", "chronological", "convergence", "mosaic"
    ]
    justification: str = Field(min_length=1)
    series_arc: str = Field(min_length=1)
    episode_arc_outline: list[str] = Field(default_factory=list)
    recommended_episode_count: int | None = Field(default=None, ge=1)
    narrator_profile: SeriesNarratorProfile = Field(
        default_factory=SeriesNarratorProfile
    )
    series_explanation_registry: list[SeriesExplanationItem] = Field(
        default_factory=list, max_length=12
    )
    series_actor_explanation_registry: list[SeriesActorExplanationItem] = Field(
        default_factory=list, max_length=12
    )
    episodes: list[StrategyEpisode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self) -> "NarrativeStrategy":
        episode_numbers = [episode.episode_number for episode in self.episodes]
        if len(episode_numbers) != len(set(episode_numbers)):
            raise ValueError(
                "episodes must not contain duplicate episode_number values"
            )
        if self.episode_arc_outline and len(self.episode_arc_outline) != len(
            self.episodes
        ):
            raise ValueError(
                "episode_arc_outline must align with episodes when provided"
            )
        if self.recommended_episode_count is not None and self.episodes:
            if self.recommended_episode_count != len(self.episodes):
                raise ValueError(
                    "recommended_episode_count must match the number of episodes"
                )
        registry_ids = [item.item_id for item in self.series_explanation_registry]
        if len(registry_ids) != len(set(registry_ids)):
            raise ValueError(
                "series_explanation_registry must not contain duplicate item_id values"
            )
        actor_registry_ids = [
            item.actor_id for item in self.series_actor_explanation_registry
        ]
        if len(actor_registry_ids) != len(set(actor_registry_ids)):
            raise ValueError(
                "series_actor_explanation_registry must not contain duplicate actor_id values"
            )
        if self.episodes and self.series_explanation_registry:
            valid_episode_numbers = {
                episode.episode_number for episode in self.episodes
            }
            for item in self.series_explanation_registry:
                if item.introduction_episode_number not in valid_episode_numbers:
                    raise ValueError(
                        "series_explanation_registry introduction_episode_number "
                        "must reference an existing episode"
                    )
        if self.episodes and self.series_actor_explanation_registry:
            valid_episode_numbers = {
                episode.episode_number for episode in self.episodes
            }
            for item in self.series_actor_explanation_registry:
                if item.introduction_episode_number not in valid_episode_numbers:
                    raise ValueError(
                        "series_actor_explanation_registry introduction_episode_number "
                        "must reference an existing episode"
                    )
        if self.episodes:
            valid_actor_registry_ids = {
                item.actor_id for item in self.series_actor_explanation_registry
            }
            actor_registry_by_id = {
                item.actor_id: item for item in self.series_actor_explanation_registry
            }
            valid_explanation_registry_ids = {
                item.item_id for item in self.series_explanation_registry
            }
            for episode in self.episodes:
                introduce_item_ids = (
                    episode.narrative_agenda.listener.introduce_explanation_item_ids
                    or episode.authorial_contract.introduce_explanation_item_ids
                )
                remind_item_ids = (
                    episode.narrative_agenda.listener.remind_explanation_item_ids
                    or episode.authorial_contract.remind_explanation_item_ids
                )
                unknown_explanation_item_ids = sorted(
                    {
                        item_id
                        for item_id in [*introduce_item_ids, *remind_item_ids]
                        if item_id not in valid_explanation_registry_ids
                    }
                )
                if unknown_explanation_item_ids:
                    raise ValueError(
                        "episode authorial_contract explanation ids must reference "
                        "series_explanation_registry items"
                    )
                if set(introduce_item_ids) & set(remind_item_ids):
                    raise ValueError(
                        "episode authorial_contract explanation ids must not be both "
                        "introduced and reminded in the same episode"
                    )
                introduce_actor_ids = (
                    episode.narrative_agenda.listener.introduce_actor_ids
                    or episode.authorial_contract.introduce_actor_ids
                )
                remind_actor_ids = (
                    episode.narrative_agenda.listener.remind_actor_ids
                    or episode.authorial_contract.remind_actor_ids
                )
                unknown_actor_ids = sorted(
                    {
                        actor_id
                        for actor_id in [*introduce_actor_ids, *remind_actor_ids]
                        if actor_id not in valid_actor_registry_ids
                    }
                )
                if unknown_actor_ids:
                    raise ValueError(
                        "episode authorial_contract actor ids must reference "
                        "series_actor_explanation_registry items"
                    )
                if set(introduce_actor_ids) & set(remind_actor_ids):
                    raise ValueError(
                        "episode authorial_contract actor ids must not be both "
                        "introduced and reminded in the same episode"
                    )
                for actor_id in introduce_actor_ids:
                    registry_item = actor_registry_by_id[actor_id]
                    if (
                        registry_item.introduction_episode_number
                        != episode.episode_number
                    ):
                        raise ValueError(
                            "introduced actor ids must match their canonical "
                            "introduction_episode_number"
                        )
                for actor_id in remind_actor_ids:
                    registry_item = actor_registry_by_id[actor_id]
                    if (
                        registry_item.introduction_episode_number
                        > episode.episode_number
                    ):
                        raise ValueError(
                            "remind actor ids must not appear before the actor's "
                            "introduction episode"
                        )
        return self


class StrategyEpisodeEnrichment(StrictModel):
    episode_number: int = Field(ge=1)
    narrator_contract: EpisodeNarratorContract = Field(
        default_factory=EpisodeNarratorContract
    )
    authorial_contract: EpisodeAuthorialContract = Field(
        default_factory=EpisodeAuthorialContract
    )
    narrative_agenda: EpisodeNarrativeAgenda = Field(
        default_factory=EpisodeNarrativeAgenda
    )
    promised_beats: list[PromisedBeat] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_commitments(self) -> "StrategyEpisodeEnrichment":
        beat_ids = [beat.beat_id for beat in self.promised_beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("promised_beats must use unique beat_id values")
        answer_count = sum(
            1 for beat in self.promised_beats if beat.intended_job == SceneJob.ANSWER
        )
        if answer_count > 1:
            raise ValueError("promised_beats may contain at most one answer beat")
        return self


class NarrativeStrategyEnrichment(StrictModel):
    narrator_profile: SeriesNarratorProfile = Field(
        default_factory=SeriesNarratorProfile
    )
    series_explanation_registry: list[SeriesExplanationItem] = Field(
        default_factory=list, max_length=12
    )
    series_actor_explanation_registry: list[SeriesActorExplanationItem] = Field(
        default_factory=list, max_length=12
    )
    episodes: list[StrategyEpisodeEnrichment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self) -> "NarrativeStrategyEnrichment":
        episode_numbers = [episode.episode_number for episode in self.episodes]
        if len(episode_numbers) != len(set(episode_numbers)):
            raise ValueError(
                "episodes must not contain duplicate episode_number values"
            )
        registry_ids = [item.item_id for item in self.series_explanation_registry]
        if len(registry_ids) != len(set(registry_ids)):
            raise ValueError(
                "series_explanation_registry must not contain duplicate item_id values"
            )
        actor_registry_ids = [
            item.actor_id for item in self.series_actor_explanation_registry
        ]
        if len(actor_registry_ids) != len(set(actor_registry_ids)):
            raise ValueError(
                "series_actor_explanation_registry must not contain duplicate actor_id values"
            )
        if self.episodes and self.series_explanation_registry:
            valid_episode_numbers = {
                episode.episode_number for episode in self.episodes
            }
            for item in self.series_explanation_registry:
                if item.introduction_episode_number not in valid_episode_numbers:
                    raise ValueError(
                        "series_explanation_registry introduction_episode_number "
                        "must reference an existing episode"
                    )
        if self.episodes and self.series_actor_explanation_registry:
            valid_episode_numbers = {
                episode.episode_number for episode in self.episodes
            }
            for item in self.series_actor_explanation_registry:
                if item.introduction_episode_number not in valid_episode_numbers:
                    raise ValueError(
                        "series_actor_explanation_registry introduction_episode_number "
                        "must reference an existing episode"
                    )
        if self.episodes:
            valid_actor_registry_ids = {
                item.actor_id for item in self.series_actor_explanation_registry
            }
            actor_registry_by_id = {
                item.actor_id: item for item in self.series_actor_explanation_registry
            }
            valid_explanation_registry_ids = {
                item.item_id for item in self.series_explanation_registry
            }
            for episode in self.episodes:
                introduce_item_ids = (
                    episode.narrative_agenda.listener.introduce_explanation_item_ids
                    or episode.authorial_contract.introduce_explanation_item_ids
                )
                remind_item_ids = (
                    episode.narrative_agenda.listener.remind_explanation_item_ids
                    or episode.authorial_contract.remind_explanation_item_ids
                )
                unknown_explanation_item_ids = sorted(
                    {
                        item_id
                        for item_id in [*introduce_item_ids, *remind_item_ids]
                        if item_id not in valid_explanation_registry_ids
                    }
                )
                if unknown_explanation_item_ids:
                    raise ValueError(
                        "episode authorial_contract explanation ids must reference "
                        "series_explanation_registry items"
                    )
                if set(introduce_item_ids) & set(remind_item_ids):
                    raise ValueError(
                        "episode authorial_contract explanation ids must not be both "
                        "introduced and reminded in the same episode"
                    )
                introduce_actor_ids = (
                    episode.narrative_agenda.listener.introduce_actor_ids
                    or episode.authorial_contract.introduce_actor_ids
                )
                remind_actor_ids = (
                    episode.narrative_agenda.listener.remind_actor_ids
                    or episode.authorial_contract.remind_actor_ids
                )
                unknown_actor_ids = sorted(
                    {
                        actor_id
                        for actor_id in [*introduce_actor_ids, *remind_actor_ids]
                        if actor_id not in valid_actor_registry_ids
                    }
                )
                if unknown_actor_ids:
                    raise ValueError(
                        "episode authorial_contract actor ids must reference "
                        "series_actor_explanation_registry items"
                    )
                if set(introduce_actor_ids) & set(remind_actor_ids):
                    raise ValueError(
                        "episode authorial_contract actor ids must not be both "
                        "introduced and reminded in the same episode"
                    )
                for actor_id in introduce_actor_ids:
                    registry_item = actor_registry_by_id[actor_id]
                    if (
                        registry_item.introduction_episode_number
                        != episode.episode_number
                    ):
                        raise ValueError(
                            "introduced actor ids must match their canonical "
                            "introduction_episode_number"
                        )
                for actor_id in remind_actor_ids:
                    registry_item = actor_registry_by_id[actor_id]
                    if (
                        registry_item.introduction_episode_number
                        > episode.episode_number
                    ):
                        raise ValueError(
                            "remind actor ids must not appear before the actor's "
                            "introduction episode"
                        )
        return self


class FramingBlock(StrictModel):
    opening_image: str = Field(min_length=1)
    threat_or_unresolved_action: str = Field(min_length=1)
    opening_question: str = Field(min_length=1)
    handoff_scene_card_id: str = Field(min_length=1)
    recap: str | None = None
    preview: str | None = None


class MustLandFacts(StrictModel):
    required: list[str] = Field(default_factory=list, min_length=1)
    strongly_preferred: list[str] = Field(default_factory=list)
    if_room: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fact_tiers(self) -> "MustLandFacts":
        seen: set[str] = set()

        def _normalize(values: list[str]) -> list[str]:
            normalized_values: list[str] = []
            for value in values:
                normalized = str(value or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                normalized_values.append(normalized)
            return normalized_values

        self.required = _normalize(self.required)
        self.strongly_preferred = _normalize(self.strongly_preferred)
        self.if_room = _normalize(self.if_room)
        if not self.required:
            raise ValueError("must_land_facts.required must contain at least one fact")
        return self

    def ordered_facts(self) -> list[str]:
        return [
            *self.required,
            *self.strongly_preferred,
            *self.if_room,
        ]

    def first_required_fact(self) -> str:
        return self.required[0]

    def total_count(self) -> int:
        return len(self.ordered_facts())


class WithholdUntil(StrictModel):
    subject: str = Field(min_length=1)
    reveal_phase: Literal["open", "pivot", "close"]
    reveal_scene_id: str | None = None
    surrogate_label: str = ""

    @model_validator(mode="after")
    def normalize_values(self) -> "WithholdUntil":
        reveal_scene_id = str(self.reveal_scene_id or "").strip()
        self.reveal_scene_id = reveal_scene_id or None
        self.surrogate_label = str(self.surrogate_label or "").strip()
        return self


class AuthorialPassage(StrictModel):
    authorial_passage_id: str = Field(
        default_factory=lambda: f"authorial_{new_id()[:8]}"
    )
    passage_id: str = Field(min_length=1)
    mode: Literal[
        "quote_then_gloss",
        "doctrinal_unpack",
        "institutional_clarifier",
        "causal_compression",
        "comparative_aside",
        "verdict_landing",
    ]
    placement: Literal["open", "mid", "close"] = "mid"
    claim: str = Field(min_length=1)
    claim_certainty: Literal["settled", "probable", "contested_memory"] = "settled"
    source_primitive_ids: list[str] = Field(
        default_factory=list, min_length=1, max_length=3
    )
    source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    counter_source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    quote_anchor: str = ""
    gloss_seed: str = ""
    must_name_terms: list[str] = Field(default_factory=list, max_length=4)
    budget_sentences: Literal[2, 3, 4, 5, 6] = 3


class TermExplanationPlan(StrictModel):
    item_id: str = Field(min_length=1)
    stage: Literal["define", "payoff", "reminder"]
    delivery_zone: Literal["open", "mid", "close"] = "mid"
    plain_gloss_seed: str = ""
    must_survive_style_audit: bool = True


class ActorExplanationPlan(StrictModel):
    actor_id: str = Field(min_length=1)
    stage: Literal["introduce", "reminder"]
    background_depth: Literal["appositive", "full"] = "appositive"
    role_label: str = ""
    source_primitive_ids: list[str] = Field(default_factory=list, max_length=3)
    source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    intro_facts: list[str] = Field(default_factory=list, max_length=4)
    why_now: str = ""
    preferred_plain_gloss: str = ""

    @model_validator(mode="after")
    def validate_actor_intro_payload(self) -> "ActorExplanationPlan":
        if self.stage != "introduce":
            return self
        has_evidence_payload = _has_evidence_backed_actor_intro_payload(
            role_label=self.role_label,
            source_primitive_ids=self.source_primitive_ids,
            source_passage_ids=self.source_passage_ids,
            intro_facts=self.intro_facts,
            why_now=self.why_now,
        )
        has_legacy_gloss = bool(_nonempty_text(self.preferred_plain_gloss))
        if not has_evidence_payload and not has_legacy_gloss:
            raise ValueError(
                "introduce actor_explanations must include an evidence-backed intro "
                "payload or a legacy preferred_plain_gloss"
            )
        if has_evidence_payload and self.background_depth == "full" and len(self.intro_facts) < 2:
            raise ValueError(
                "introduce actor_explanations with background_depth='full' must include at least 2 intro_facts"
            )
        return self


class HostPresenceBeat(StrictModel):
    kind: Literal[
        "orientation",
        "clarify",
        "contrast",
        "evaluate",
        "callback",
        "term_reminder",
        "light_aside",
        "naming_note",
    ]
    placement: Literal["open", "pivot", "close"]
    seed: str = Field(min_length=1)
    scope: Literal["beat", "scene"] = "scene"
    address_mode: Literal["implicit", "we", "you", "i"] = "implicit"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_kind(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        kind = str(cleaned.get("kind", "") or "").strip()
        if kind == "term_reminder":
            cleaned["kind"] = "clarify"
        return cleaned


class SectionSonicObligation(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class SectionSonicBeat(StrictModel):
    moment: str = Field(min_length=1)
    cue: str = Field(min_length=1)
    why_here: str = ""

    @field_validator("moment", "cue", "why_here", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: Any) -> str:
        return str(value or "").strip()


class SectionSonicPlan(StrictModel):
    obligation: SectionSonicObligation = SectionSonicObligation.PREFERRED
    opening_anchor: str = Field(min_length=1)
    opening_pressure: str = Field(min_length=1)
    opening_realization_note: str = ""
    later_beats: list[SectionSonicBeat] = Field(default_factory=list, max_length=2)

    @field_validator(
        "opening_anchor",
        "opening_pressure",
        "opening_realization_note",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: Any) -> str:
        return str(value or "").strip()


class SectionStateEffects(StrictModel):
    """Continuity/state mutations a section causes, nested under section_progression."""

    question_moves: list[ListenerQuestionMove] = Field(default_factory=list, max_length=4)
    memory_thread_moves: list[ListenerMemoryThreadMove] = Field(
        default_factory=list, max_length=4
    )
    host_mystery_moves: list[HostMysteryMove] = Field(
        default_factory=list, max_length=4
    )
    host_assumption_moves: list[HostAssumptionMove] = Field(
        default_factory=list, max_length=4
    )
    host_theory_moves: list[HostTheoryMove] = Field(
        default_factory=list, max_length=4
    )


class SectionProgression(StrictModel):
    """How one section advances the episode's answer and hands pressure forward."""

    stage: ProgressionStage
    becomes_obvious: str = Field(min_length=1)
    answer_contribution: str = Field(min_length=1)
    theme_link: str = Field(min_length=1)
    what_remains_live: str = Field(min_length=1)
    state_effects: SectionStateEffects = Field(default_factory=SectionStateEffects)

    @field_validator(
        "becomes_obvious", "answer_contribution", "theme_link", "what_remains_live"
    )
    @classmethod
    def _strip_required_prose(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "section_progression prose fields must not be blank"
            )
        return cleaned


class ArchitectureSection(StrictModel):
    section_id: str = Field(min_length=1)
    purpose: SectionPurpose
    approx_runtime_minutes: float = Field(gt=0.0)
    primitive_ids: list[str] = Field(default_factory=list, min_length=1)
    open_mode: Literal[
        "scene_anchor", "voice_first", "question_first", "condition_first"
    ] = "scene_anchor"
    section_anchor: str = Field(
        default="",
        validation_alias=AliasChoices("section_anchor", "anchor"),
        serialization_alias="section_anchor",
    )
    must_stage_beats: list[str] = Field(default_factory=list, max_length=4)
    priority_core_passage_ids: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list, max_length=6)
    authorial_passages: list[AuthorialPassage] = Field(
        default_factory=list, max_length=4
    )
    term_explanations: list[TermExplanationPlan] = Field(
        default_factory=list, max_length=4
    )
    actor_explanations: list[ActorExplanationPlan] = Field(
        default_factory=list, max_length=4
    )
    section_progression: SectionProgression
    section_sonic_plan: SectionSonicPlan | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        for legacy_field in (
            "entry_state",
            "exit_state",
            "argument_role",
            "inference_mode",
            "listener_tension",
            "section_question",
            "section_turn",
            "section_resolution",
            "transition_logic",
            "depends_on_section_ids",
            "sets_up_section_ids",
            "recurrence_role",
            "pressure_type",
            "resolution_type",
            "closure_level",
            "analysis_goal",
            "host_presence_beats",
            "host_beats",
        ):
            cleaned.pop(legacy_field, None)
        return cleaned

    @field_validator("section_anchor")
    @classmethod
    def normalize_section_anchor(cls, value: str) -> str:
        return value.strip()

    @field_validator("must_stage_beats")
    @classmethod
    def validate_must_stage_beats(cls, value: list[str]) -> list[str]:
        if not value:
            return value
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("must_stage_beats must not contain blank items")
        if len(cleaned) < 2:
            raise ValueError(
                "must_stage_beats must contain at least 2 items when provided"
            )
        return cleaned

    @model_validator(mode="after")
    def validate_authorial_passage_ids(self) -> "ArchitectureSection":
        seen_ids: set[str] = set()
        for passage in self.authorial_passages:
            if passage.authorial_passage_id in seen_ids:
                raise ValueError(
                    "authorial_passages must use unique authorial_passage_id values within a section"
                )
            seen_ids.add(passage.authorial_passage_id)
        return self


class EpisodeArchitecture(StrictModel):
    episode_number: int = Field(ge=1)
    major_turn_section_id: str = Field(min_length=1)
    allowed_recurring_primitive_ids: list[str] = Field(default_factory=list)
    forbidden_redundancies: list[str] = Field(default_factory=list)
    promised_beat_decisions: list[PromisedBeatDecisionRecord] = Field(
        default_factory=list
    )
    sections: list[ArchitectureSection] = Field(
        default_factory=list, min_length=6, max_length=13
    )
    architecture_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_target_duration_minutes(cls, data: Any) -> Any:
        if isinstance(data, dict) and "target_duration_minutes" in data:
            return {k: v for k, v in data.items() if k != "target_duration_minutes"}
        return data

    @property
    def runtime_minutes(self) -> float:
        return sum(section.approx_runtime_minutes for section in self.sections)

    @model_validator(mode="after")
    def validate_architecture(self) -> "EpisodeArchitecture":
        seen_section_ids: set[str] = set()
        section_by_id: dict[str, ArchitectureSection] = {}
        for section in self.sections:
            if section.section_id in seen_section_ids:
                raise ValueError("sections must use unique section_id values")
            seen_section_ids.add(section.section_id)
            section_by_id[section.section_id] = section

        if self.major_turn_section_id not in section_by_id:
            raise ValueError("major_turn_section_id must reference an existing section")

        # Section progression stage contract (replaces answer/residue pointers).
        answer_indices = [
            idx
            for idx, section in enumerate(self.sections)
            if section.section_progression.stage == ProgressionStage.ANSWER
        ]
        close_indices = [
            idx
            for idx, section in enumerate(self.sections)
            if section.section_progression.stage == ProgressionStage.CLOSE
        ]
        if len(answer_indices) != 1:
            raise ValueError(
                "exactly one section must have section_progression.stage='answer'"
            )
        if len(close_indices) != 1:
            raise ValueError(
                "exactly one section must have section_progression.stage='close'"
            )
        answer_idx = answer_indices[0]
        close_idx = close_indices[0]
        if close_idx != len(self.sections) - 1:
            raise ValueError(
                "the close stage must be on the final section"
            )
        if answer_idx >= close_idx:
            raise ValueError("the answer stage must occur before the close stage")
        for idx, section in enumerate(self.sections):
            stage = section.section_progression.stage
            if stage == ProgressionStage.AFTERPRESSURE and not (
                answer_idx < idx < close_idx
            ):
                raise ValueError(
                    "afterpressure sections must occur after the answer stage "
                    "and before the close stage"
                )
        answer_section = self.sections[answer_idx]
        if answer_section.purpose == SectionPurpose.OPENING:
            raise ValueError(
                "the answer stage must not be on an opening-purpose section"
            )
        for section in self.sections:
            is_close_stage = (
                section.section_progression.stage == ProgressionStage.CLOSE
            )
            is_closing_purpose = section.purpose == SectionPurpose.CLOSING
            if is_close_stage != is_closing_purpose:
                raise ValueError(
                    "purpose='closing' must align with section_progression.stage='close'"
                )

        beat_ids = [record.beat_id for record in self.promised_beat_decisions]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError(
                "promised_beat_decisions must use unique beat_id values"
            )
        return self


def validate_episode_architecture_targets(
    architecture: EpisodeArchitecture,
    *,
    section_target_min: int,
    section_target_max: int,
) -> None:
    section_count = len(architecture.sections)
    if section_count < section_target_min or section_count > section_target_max:
        raise ValueError(
            "sections must contain "
            f"{section_target_min}-{section_target_max} items"
        )


class SceneActor(StrictModel):
    name: str = Field(min_length=1)
    actor_id: str | None = None
    affiliation: str | None = None
    presence: Literal["primary", "secondary", "background"] = "secondary"
    explanation_stage: Literal["introduce", "reminder"] | None = None
    background_depth: Literal["appositive", "full"] | None = None
    role_label: str = ""
    source_primitive_ids: list[str] = Field(default_factory=list, max_length=3)
    source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    intro_facts: list[str] = Field(default_factory=list, max_length=4)
    why_now: str = ""
    preferred_plain_gloss: str = ""

    @model_validator(mode="after")
    def validate_actor_explanation_payload(self) -> "SceneActor":
        has_evidence_payload = _has_evidence_backed_actor_intro_payload(
            role_label=self.role_label,
            source_primitive_ids=self.source_primitive_ids,
            source_passage_ids=self.source_passage_ids,
            intro_facts=self.intro_facts,
            why_now=self.why_now,
        )
        has_legacy_gloss = bool(_nonempty_text(self.preferred_plain_gloss))
        if self.explanation_stage == "introduce":
            if not has_evidence_payload and not has_legacy_gloss:
                raise ValueError(
                    "scene actors marked for introduce must include an evidence-backed intro payload or a legacy preferred_plain_gloss"
                )
            if self.background_depth == "full" and has_evidence_payload and len(self.intro_facts) < 2:
                raise ValueError(
                    "scene actors marked for introduce with background_depth='full' must include at least 2 intro_facts"
                )
        return self


class SceneActorArcBinding(StrictModel):
    """Deprecated compatibility model retained for legacy imports and fixtures."""

    thread_id: str = Field(min_length=1)
    scene_role: Literal["driver", "blocked", "counterforce", "subject"]
    scene_use: Literal[
        "introduce",
        "develop",
        "complicate",
        "stage_choice",
        "show_consequence",
        "pay_off",
        "avoid",
    ]
    weight: Literal["light", "standard", "strong"] = "standard"


class HostMoveCue(StrictModel):
    move_type: Literal[
        "orient",
        "clarify",
        "evaluate",
        "contrast",
        "callback",
        "light_aside",
        "naming_note",
        "uncertainty",
        "revision",
        "surprise",
    ]
    target: str = ""
    surface_mode: Literal["woven", "distinct", "mixed"] = "mixed"
    address_mode: Literal["implicit", "we", "you", "i"] = "implicit"

    @model_validator(mode="before")
    @classmethod
    def _normalize_host_move(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        move_type = str(cleaned.get("move_type", "") or "").strip()
        if not move_type:
            raise ValueError("host move cues must include move_type")
        cleaned["move_type"] = move_type
        target = cleaned.get("target", cleaned.get("note", "")) or ""
        cleaned["target"] = str(target).strip()
        cleaned.pop("note", None)
        cleaned["surface_mode"] = (
            str(cleaned.get("surface_mode", "mixed") or "mixed").strip() or "mixed"
        )
        cleaned["address_mode"] = (
            str(cleaned.get("address_mode", "implicit") or "implicit").strip()
            or "implicit"
        )
        return cleaned


class HostMovesByPhase(StrictModel):
    open: list[HostMoveCue] = Field(default_factory=list, max_length=2)
    pivot: list[HostMoveCue] = Field(default_factory=list, max_length=2)
    close: list[HostMoveCue] = Field(default_factory=list, max_length=2)


class _SceneCardBase(StrictModel):
    scene_id: str = Field(default_factory=lambda: f"scene_{new_id()[:8]}")
    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scene_role: SceneRole
    scene_job: SceneJob
    beat_change: str = Field(
        min_length=1,
        validation_alias=AliasChoices("beat_change", "state_effect"),
        serialization_alias="beat_change",
    )
    must_land_facts: MustLandFacts
    entry_image: str = ""
    observable_detail: str = ""
    audible_detail: str = Field(
        default="",
        validation_alias=AliasChoices("audible_detail", "what_we_hear", "audio"),
        serialization_alias="audible_detail",
    )
    withhold_until: WithholdUntil | None = None
    timeframe: str | None = None
    location: str | None = None
    actors: list[SceneActor] = Field(default_factory=list, max_length=4)
    primitive_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)
    authorial_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    word_count_priority: WordCountPriority = WordCountPriority.DEFAULT
    host_moves: HostMovesByPhase

    @model_validator(mode="after")
    def validate_card_shape(self) -> "_SceneCardBase":
        deduped_primitive_ids: list[str] = []
        seen_primitive_ids: set[str] = set()
        for primitive_id in self.primitive_ids:
            normalized = str(primitive_id or "").strip()
            if not normalized or normalized in seen_primitive_ids:
                continue
            seen_primitive_ids.add(normalized)
            deduped_primitive_ids.append(normalized)
        self.primitive_ids = deduped_primitive_ids
        deduped_authorial_ids: list[str] = []
        seen_authorial_ids: set[str] = set()
        for authorial_passage_id in self.authorial_passage_ids:
            normalized = str(authorial_passage_id or "").strip()
            if not normalized or normalized in seen_authorial_ids:
                continue
            seen_authorial_ids.add(normalized)
            deduped_authorial_ids.append(normalized)
        self.authorial_passage_ids = deduped_authorial_ids
        return self


_LEGACY_SCENE_ROLE_TO_ROLE_JOB: dict[str, tuple[str, str | None]] = {
    "setup": (SceneRole.CONTEXT_SETUP.value, SceneJob.BUILD.value),
    "shock": (SceneRole.SHOCK.value, SceneJob.BUILD.value),
    "action": (SceneRole.ACTION.value, SceneJob.BUILD.value),
    "consequence": (SceneRole.FALLOUT.value, SceneJob.BUILD.value),
    "reaction": (SceneRole.REACTION.value, SceneJob.BUILD.value),
    "contestation": (SceneRole.CONTESTATION.value, SceneJob.BUILD.value),
    "synthesis": (SceneRole.IMPLICATION.value, SceneJob.BUILD.value),
    "process": (SceneRole.ACTION.value, SceneJob.BUILD.value),
    "perspective shift": (SceneRole.IMPLICATION.value, SceneJob.BUILD.value),
    "perspective_shift": (SceneRole.IMPLICATION.value, SceneJob.BUILD.value),
    "reveal": (SceneRole.IMPLICATION.value, SceneJob.TURN.value),
    "reversal": (SceneRole.SHOCK.value, SceneJob.TURN.value),
    "stage_choice": (SceneRole.ACTION.value, SceneJob.TURN.value),
    "turn": (SceneRole.SHOCK.value, SceneJob.TURN.value),
    "closing": (SceneRole.IMPLICATION.value, SceneJob.CLOSE.value),
}

_LEGACY_SCENE_FUNCTION_TO_JOB: dict[str, str] = {
    SceneFunction.SCENE.value: SceneJob.BUILD.value,
    SceneFunction.HINGE.value: SceneJob.TURN.value,
    SceneFunction.MECHANISM.value: SceneJob.BUILD.value,
    SceneFunction.TURN.value: SceneJob.TURN.value,
    SceneFunction.LANDING.value: SceneJob.BUILD.value,
    SceneFunction.CALLBACK.value: SceneJob.BUILD.value,
    SceneFunction.AFTERLIFE.value: SceneJob.BUILD.value,
}

_SCENE_JOB_DEFAULT_BY_ROLE: dict[str, str] = {
    SceneRole.CONTEXT_SETUP.value: SceneJob.BUILD.value,
    SceneRole.ACTOR_SETUP.value: SceneJob.BUILD.value,
    SceneRole.ACTION.value: SceneJob.BUILD.value,
    SceneRole.SHOCK.value: SceneJob.BUILD.value,
    SceneRole.CONTESTATION.value: SceneJob.BUILD.value,
    SceneRole.REACTION.value: SceneJob.BUILD.value,
    SceneRole.FALLOUT.value: SceneJob.BUILD.value,
    SceneRole.IMPLICATION.value: SceneJob.BUILD.value,
}


def _migrate_legacy_scene_card(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    cleaned.pop("coverage_depth", None)
    cleaned.pop("dominant_pack_id", None)
    cleaned.pop("dominant_primitive_id", None)
    cleaned.pop("spine_relation", None)
    role = str(cleaned.get("scene_role", "") or "").strip()
    if role:
        if role in _SCENE_FUNCTION_VALUES and role not in _LEGACY_SCENE_ROLE_TO_ROLE_JOB:
            raise ValueError(
                f"scene_role={role!r} looks like a scene_function value; move it to scene_job and use one of: "
                f"{', '.join(member.value for member in SceneRole)}"
            )
        normalized_role, default_job = _LEGACY_SCENE_ROLE_TO_ROLE_JOB.get(
            role,
            (role, None),
        )
        cleaned["scene_role"] = normalized_role
        if not str(cleaned.get("scene_job", "") or "").strip() and default_job:
            cleaned["scene_job"] = default_job
    legacy_scene_function = str(cleaned.get("scene_function", "") or "").strip()
    scene_job = str(cleaned.get("scene_job", "") or "").strip()
    if legacy_scene_function:
        mapped_job = _LEGACY_SCENE_FUNCTION_TO_JOB.get(legacy_scene_function)
        if mapped_job:
            scene_job = scene_job or mapped_job
            cleaned["scene_job"] = scene_job
        elif legacy_scene_function in _SCENE_ROLE_VALUES:
            raise ValueError(
                f"scene_function={legacy_scene_function!r} looks like a scene_role value; move it to scene_role and use one of: "
                f"{', '.join(member.value for member in SceneRole)}"
            )
    if scene_job and scene_job in _SCENE_ROLE_VALUES:
        raise ValueError(
            f"scene_job={scene_job!r} looks like a scene_role value; move it to scene_role and use one of: "
            f"{', '.join(member.value for member in SceneJob)}"
        )
    if not str(cleaned.get("scene_job", "") or "").strip():
        normalized_role = str(cleaned.get("scene_role", "") or "").strip()
        default_job = _SCENE_JOB_DEFAULT_BY_ROLE.get(normalized_role)
        if default_job:
            cleaned["scene_job"] = default_job
    if "beat_change" not in cleaned and cleaned.get("state_effect"):
        cleaned["beat_change"] = cleaned["state_effect"]
    raw_facts = cleaned.get("must_land_facts")
    if isinstance(raw_facts, list):
        cleaned["must_land_facts"] = {
            "required": raw_facts,
            "strongly_preferred": [],
            "if_room": [],
        }
    elif not isinstance(raw_facts, dict):
        beat_change = str(cleaned.get("beat_change", "") or "").strip()
        cleaned["must_land_facts"] = {
            "required": [beat_change] if beat_change else [],
            "strongly_preferred": [],
            "if_room": [],
        }
    raw_withhold_until = cleaned.get("withhold_until")
    if isinstance(raw_withhold_until, str):
        normalized_withhold = raw_withhold_until.strip()
        if normalized_withhold:
            if normalized_withhold.startswith("scene_"):
                cleaned["withhold_until"] = {
                    "subject": "withheld material",
                    "reveal_scene_id": normalized_withhold,
                    "reveal_phase": "open",
                    "surrogate_label": "",
                }
            else:
                cleaned["withhold_until"] = {
                    "subject": normalized_withhold,
                    "reveal_scene_id": None,
                    "reveal_phase": "close",
                    "surrogate_label": "",
                }
    elif isinstance(raw_withhold_until, dict):
        normalized_withhold = dict(raw_withhold_until)
        subject = str(normalized_withhold.get("subject", "") or "").strip()
        acting_subject = str(normalized_withhold.get("acting_subject", "") or "").strip()
        if not subject and acting_subject:
            normalized_withhold["subject"] = acting_subject
        normalized_withhold.pop("acting_subject", None)
        cleaned["withhold_until"] = normalized_withhold
    cleaned.pop("state_effect", None)
    cleaned.pop("scene_function", None)
    cleaned.pop("local_question", None)
    cleaned.pop("what_becomes_legible_later", None)
    cleaned.pop("intended_move", None)
    for actor in cleaned.get("actors", []) or []:
        if isinstance(actor, dict):
            actor.pop("arc_bindings", None)
    if "host_move" in cleaned:
        raise ValueError("scene cards must use host_moves; legacy host_move is not supported")
    host_moves = cleaned.get("host_moves")
    if not isinstance(host_moves, dict):
        raise ValueError("scene cards must include host_moves")
    return cleaned


class SceneCardDraft(_SceneCardBase):
    estimated_duration_seconds: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        return _migrate_legacy_scene_card(data)


class SceneCard(_SceneCardBase):
    estimated_duration_seconds: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        return _migrate_legacy_scene_card(data)


class EpisodePlanDraft(StrictModel):
    episode_number: int = Field(ge=1)
    framing: FramingBlock
    scene_cards: list[SceneCardDraft] = Field(default_factory=list, min_length=1)
    answer_scene_card_id: str | None = None
    dropped_support_primitive_reasons: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scene_cards(self) -> "EpisodePlanDraft":
        scene_ids = [scene.scene_id for scene in self.scene_cards]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene_cards must use unique scene_id values")
        if self.framing.handoff_scene_card_id not in set(scene_ids):
            raise ValueError(
                "framing.handoff_scene_card_id must reference an existing scene card"
            )
        scene_by_id = {scene.scene_id: scene for scene in self.scene_cards}
        if self.answer_scene_card_id is not None:
            answer_scene = scene_by_id.get(self.answer_scene_card_id)
            if answer_scene is None:
                raise ValueError(
                    "answer_scene_card_id must reference an existing scene card"
                )
            if answer_scene.scene_job != SceneJob.ANSWER:
                raise ValueError(
                    "answer_scene_card_id must reference a scene card with scene_job='answer'"
                )
        scene_index_by_id = {
            scene.scene_id: idx for idx, scene in enumerate(self.scene_cards)
        }
        for scene in self.scene_cards:
            withhold = scene.withhold_until
            if withhold is None or withhold.reveal_scene_id is None:
                continue
            reveal_index = scene_index_by_id.get(withhold.reveal_scene_id)
            if reveal_index is None:
                raise ValueError(
                    "withhold_until.reveal_scene_id must reference an existing scene card"
                )
            if reveal_index < scene_index_by_id[scene.scene_id]:
                raise ValueError(
                    "withhold_until.reveal_scene_id must not point to an earlier scene card"
                )
        return self


class EpisodePlan(EpisodePlanDraft):
    scene_cards: list[SceneCard] = Field(default_factory=list, min_length=1)
    target_word_count: int = Field(ge=1)


# ---------------------------------------------------------------------------
# 3.6 Script & Validation Models
# ---------------------------------------------------------------------------


class Citation(StrictModel):
    citation_id: str = Field(default_factory=new_id)
    text_span: str = ""
    passage_id: str
    book_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ActorExplanationRealization(StrictModel):
    actor_id: str = Field(min_length=1)
    scene_card_id: str = Field(min_length=1)
    stage: Literal["introduce", "reminder"]
    text_span: str = Field(min_length=1)
    source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    realized_facts: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_realization(self) -> "ActorExplanationRealization":
        if self.stage == "introduce":
            if not self.source_passage_ids:
                raise ValueError(
                    "introduce actor_explanation_realizations must include source_passage_ids"
                )
            if not self.realized_facts:
                raise ValueError(
                    "introduce actor_explanation_realizations must include realized_facts"
                )
        return self


class ProseSection(StrictModel):
    section_id: str = Field(default_factory=lambda: f"section_{new_id()[:8]}")
    scene_card_ids: list[str] = Field(default_factory=list, min_length=1)
    movement_goal: str = Field(min_length=1)
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_book_ids: list[str] = Field(default_factory=list)
    section_sonic_plan: SectionSonicPlan | None = None
    actor_explanation_realizations: list[ActorExplanationRealization] = Field(
        default_factory=list, max_length=4
    )

    @model_validator(mode="after")
    def validate_actor_explanation_realizations(self) -> "ProseSection":
        valid_scene_ids = set(self.scene_card_ids)
        for realization in self.actor_explanation_realizations:
            if realization.scene_card_id not in valid_scene_ids:
                raise ValueError(
                    "actor_explanation_realizations scene_card_id must reference a scene_card_id in the prose section"
                )
        return self


class EpisodeScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    framing: FramingBlock
    prose_sections: list[ProseSection] = Field(default_factory=list)
    total_word_count: int = Field(default=0, ge=0)
    estimated_duration_seconds: int = Field(default=0, ge=0)


class ClaimAssessment(StrictModel):
    text_unit_id: str = Field(min_length=1)
    claim_text: str
    cited_passage_id: str
    status: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "FABRICATED"]
    explanation: str = ""


class CrossBookClaimAssessment(StrictModel):
    text_unit_id: str = Field(min_length=1)
    claim_text: str
    book_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)
    comparison_valid: bool = True
    failure_reason: str | None = None


class FairnessFlag(StrictModel):
    text_unit_id: str = Field(min_length=1)
    book_id: str
    claim_text: str
    issue: Literal["straw_man", "oversimplified", "out_of_context", "false_equivalence"]
    suggestion: str = ""


class GroundingReport(StrictModel):
    episode_number: int = Field(ge=1)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    cross_book_claims: list[CrossBookClaimAssessment] = Field(default_factory=list)
    overall_status: Literal["PASSED", "NEEDS_REPAIR", "FAILED"] = "PASSED"
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    attribution_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    fairness_flags: list[FairnessFlag] = Field(default_factory=list)


class SegmentDiff(StrictModel):
    text_unit_id: str
    before: str
    after: str


class RepairResult(StrictModel):
    attempt_number: int = Field(ge=1)
    original_script: EpisodeScript
    repaired_script: EpisodeScript
    claims_repaired: int = Field(default=0, ge=0)
    remaining_failures: int = Field(default=0, ge=0)
    diffs: list[SegmentDiff] = Field(default_factory=list)
    status: Literal["RESOLVED", "IMPROVED", "NO_PROGRESS"] = "NO_PROGRESS"


# ---------------------------------------------------------------------------
# 3.7 Speech, Style, & Audio Models
# ---------------------------------------------------------------------------


class PronunciationHint(StrictModel):
    text: str = Field(min_length=1)
    spoken_as: str = Field(min_length=1)

    @field_validator("text", "spoken_as", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()


class SpeechHints(StrictModel):
    style: Literal["neutral", "measured", "urgent", "dramatic"] = "neutral"
    intensity: Literal["none", "light", "medium", "strong"] = "none"
    pace: Literal["slower", "normal", "faster"] = "normal"
    pause_before_ms: int = Field(default=300, ge=0, le=2000)
    pause_after_ms: int = Field(default=300, ge=0, le=2000)
    pronunciation_hints: list[PronunciationHint] = Field(default_factory=list)
    emphasis_targets: list[str] = Field(default_factory=list)
    render_strategy: Literal[
        "plain", "isolate_phrase", "split_sentences", "slow_clause"
    ] = "plain"


class SonicCue(StrictModel):
    scene_id: str = Field(min_length=1)
    scene_job: str = Field(min_length=1)
    entry_image: str = ""
    observable_detail: str = ""
    audible_detail: str = ""


class SpokenSection(StrictModel):
    section_id: str
    text: str
    speech_hints: SpeechHints = Field(default_factory=SpeechHints)
    section_sonic_plan: SectionSonicPlan | None = None
    sonic_cues: list[SonicCue] = Field(default_factory=list)


class SpokenScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    framing: FramingBlock
    sections: list[SpokenSection] = Field(default_factory=list)
    tts_provider: str = "openai"


class StyleAuditSection(StrictModel):
    section_id: str = Field(min_length=1)
    edited_text: str
    edit_notes: list[str] = Field(default_factory=list)


class StyleAuditResponse(StrictModel):
    episode_number: int = Field(ge=1)
    sections: list[StyleAuditSection] = Field(default_factory=list)
    episode_warnings: list[str] = Field(default_factory=list)


class RenderSegment(StrictModel):
    segment_id: str = Field(default_factory=new_id)
    text: str
    voice_id: str = "fable"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    pause_before_ms: int = Field(default=0, ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    instructions: str | None = None
    hint_degradations: list[str] = Field(default_factory=list)


class RenderManifest(StrictModel):
    episode_number: int = Field(ge=1)
    segments: list[RenderSegment] = Field(default_factory=list)
    total_segments: int = Field(default=0, ge=0)
    estimated_duration_seconds: int = Field(default=0, ge=0)


class AudioSegmentResult(StrictModel):
    segment_id: str
    audio_path: str
    duration_seconds: float = Field(default=0.0, ge=0.0)
    success: bool = True
    error: str | None = None


class AudioManifest(StrictModel):
    episode_number: int = Field(ge=1)
    audio_segments: list[AudioSegmentResult] = Field(default_factory=list)
    merged_audio_path: str | None = None
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
