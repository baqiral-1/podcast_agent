"""Strict schema contracts for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


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


def _default_host_move_placement(move_type: str) -> str:
    normalized = str(move_type or "").strip() or "none"
    if normalized in {"orient", "naming_note"}:
        return "open"
    if normalized in {"clarify", "contrast", "light_aside"}:
        return "pivot"
    return "close"


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


class SynthesisTag(str, Enum):
    AGREES_WITH = "agrees_with"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    EXEMPLIFIES = "exemplifies"
    CONTEXTUALIZES = "contextualizes"
    INDEPENDENT = "independent"


class InsightType(str, Enum):
    SYNCHRONICITY = "synchronicity"
    PRODUCTIVE_FRICTION = "productive_friction"
    INTELLECTUAL_SCAFFOLDING = "intellectual_scaffolding"
    LATENT_PATTERN = "latent_pattern"
    EPISTEMIC_DRIFT = "epistemic_drift"


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


class ClosureMode(str, Enum):
    RESIDUE = "residue"
    TURN = "turn"
    PARTIAL_ANSWER = "partial_answer"
    FINAL_ANSWER = "final_answer"


class SceneRole(str, Enum):
    CONTEXT_SETUP = "context_setup"
    ACTOR_SETUP = "actor_setup"
    ACTION = "action"
    SHOCK = "shock"
    CONTESTATION = "contestation"
    REACTION = "reaction"
    FALLOUT = "fallout"
    IMPLICATION = "implication"


class SceneFunction(str, Enum):
    SCENE = "scene"
    HINGE = "hinge"
    MECHANISM = "mechanism"
    TURN = "turn"
    LANDING = "landing"
    CALLBACK = "callback"
    AFTERLIFE = "afterlife"


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
        return {
            key: value
            for key, value in data.items()
            if key not in legacy_fields
        }


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
    max_axes: int = Field(default=15, ge=1)
    min_axes: int = Field(default=10, ge=1)
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
    synthesis_axis_min: int = Field(default=10, ge=0)
    synthesis_axis_max: int = Field(default=15, ge=1)
    synthesis_total_passage_cap: int = Field(default=450, ge=1)
    synthesis_floor_budget_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    synthesis_axis_floor_min: int = Field(default=0, ge=0)
    synthesis_axis_floor_max: int = Field(default=0, ge=0)
    synthesis_axis_ceiling_multiplier: float = Field(default=1.68, ge=1.0)
    synthesis_trim_top_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    synthesis_trim_mid_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    synthesis_trim_next_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    synthesis_trim_top_keep_fraction: float = Field(default=0.375, gt=0.0, le=1.0)
    synthesis_trim_mid_keep_fraction: float = Field(default=0.275, gt=0.0, le=1.0)
    synthesis_trim_next_keep_fraction: float = Field(default=0.325, gt=0.0, le=1.0)
    synthesis_trim_tail_keep_fraction: float = Field(default=0.175, gt=0.0, le=1.0)
    planning_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    planning_axis_min: int = Field(default=10, ge=0)
    planning_axis_max: int = Field(default=15, ge=1)
    planning_total_passage_cap: int = Field(default=300, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=3, ge=0)
    tts_provider: str = "openai"
    tts_concurrency: int = Field(default=12, ge=1)
    episode_architecture_concurrency: int = Field(default=8, ge=1)
    episode_planning_concurrency: int = Field(default=8, ge=1)
    episode_write_concurrency: int = Field(default=8, ge=1)
    spoken_delivery_concurrency: int | None = Field(default=8, ge=1)
    architecture_section_target_min: int = Field(default=9, ge=1)
    architecture_section_target_max: int = Field(default=12, ge=1)
    narrative_strategy_episode_count_min: int = Field(default=8, ge=1)
    narrative_strategy_episode_count_max: int = Field(default=12, ge=1)
    min_episode_minutes: float = Field(default=90.0, gt=0.0)
    max_episode_minutes: float = Field(default=105.0, gt=0.0)
    duration_shortfall_policy: Literal["warn"] = "warn"
    scene_card_target_min: int = Field(default=27, ge=1)
    scene_card_target_max: int = Field(default=36, ge=1)
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
            raise ValueError("synthesis_axis_floor_max must be >= synthesis_axis_floor_min")
        if self.planning_axis_max < self.planning_axis_min:
            raise ValueError("planning_axis_max must be >= planning_axis_min")
        if (
            self.synthesis_trim_top_fraction
            + self.synthesis_trim_mid_fraction
            + self.synthesis_trim_next_fraction
        ) > 1.0:
            raise ValueError("synthesis trim top, mid, and next fractions must sum to <= 1.0")
        if self.architecture_section_target_max < self.architecture_section_target_min:
            raise ValueError(
                "architecture_section_target_max must be >= architecture_section_target_min"
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
            raise ValueError("scene_card_primitives_max must be >= scene_card_primitives_min")
        if self.max_episode_minutes < self.min_episode_minutes:
            raise ValueError("max_episode_minutes must be >= min_episode_minutes")
        return self


class ThematicProject(StrictModel):
    project_id: str = Field(default_factory=new_id)
    theme: str
    theme_elaboration: str | None = None
    sub_themes: list[str] = Field(default_factory=list, max_length=30)
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
        if len(normalized) > 30:
            raise ValueError("sub_themes supports at most 30 entries.")
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


class SynthesisPrimitiveBase(StrictModel):
    id: str = Field(default_factory=new_id)
    family: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    axis_ids: list[str] = Field(default_factory=list)
    core_passage_ids: list[str] = Field(default_factory=list)
    support_passage_ids: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    geography: str | None = None
    primary_actor_ids: list[str] = Field(default_factory=list)
    affected_actor_ids: list[str] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    actor_tags: list[str] = Field(default_factory=list)
    institution_tags: list[str] = Field(default_factory=list)
    unresolved_actor_tags: list[str] = Field(default_factory=list)
    narrative_importance_score: float = Field(default=0.5, ge=0.0, le=1.0)

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


class BaseSynthesisPrimitive(StrictModel):
    id: str = Field(default_factory=new_id)
    family: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    axis_ids: list[str] = Field(default_factory=list)
    core_passage_ids: list[str] = Field(default_factory=list)
    support_passage_ids: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    geography: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    narrative_importance_score: float = Field(default=0.5, ge=0.0, le=1.0)

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


class NarrationHooks(StrictModel):
    concrete_detail: str = Field(min_length=1)
    host_lens: str = Field(min_length=1)
    carry_forward: str = Field(min_length=1)
    quote_anchor: str = ""
    plain_gloss: str = ""
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


class EnrichmentNarrationHooks(StrictModel):
    concrete_detail: str = Field(min_length=1)
    host_lens: str = Field(min_length=1)
    carry_forward: str = Field(min_length=1)
    quote_anchor: str = ""
    plain_gloss: str = ""
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


class CandidateReading(StrictModel):
    label: str = Field(min_length=1)
    claim: str = Field(
        min_length=1,
        validation_alias=AliasChoices("claim", "summary"),
        serialization_alias="claim",
    )
    emphasizes: str = ""
    downplays: str = ""
    support_passage_ids: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.claim


class EnrichmentCandidateReading(StrictModel):
    label: str = Field(min_length=1)
    claim: str = Field(
        min_length=1,
        validation_alias=AliasChoices("claim", "summary"),
        serialization_alias="claim",
    )
    emphasizes: str = ""
    downplays: str = ""
    support_passage_ids: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.claim


class SynthesisPrimitive(SynthesisPrimitiveBase):
    """Generic synthesis primitive for families without richer typed fields."""

    family: Literal[
        "telling_details",
        "misreadings_and_fantasies",
        "perspective_windows",
        "afterlives",
        "recurring_images_and_symbols",
    ]
    narration_hooks: NarrationHooks | None = None


class EpochalTurnPrimitive(SynthesisPrimitiveBase):
    family: Literal["epochal_turns"]
    before_state: str = Field(min_length=1)
    after_state: str = Field(min_length=1)
    change_driver: str = Field(min_length=1)
    why_no_return: str = Field(
        min_length=1,
        validation_alias=AliasChoices("why_no_return", "irreversibility_reason"),
        serialization_alias="why_no_return",
    )
    proof_of_change: str = ""
    narration_hooks: NarrationHooks | None = None

    @property
    def irreversibility_reason(self) -> str:
        return self.why_no_return


class DecisionPrimitive(SynthesisPrimitiveBase):
    family: Literal["decisions_and_nondecisions"]
    actor_ids: list[str] = Field(default_factory=list)
    decision_trigger: str = ""
    decision_question: str = Field(min_length=1)
    decision_mode: Literal["decision", "refusal", "delay", "nondecision"]
    options_considered: list[str] = Field(default_factory=list, min_length=1)
    next_result: str = Field(
        min_length=1,
        validation_alias=AliasChoices("next_result", "immediate_consequence"),
        serialization_alias="next_result",
    )
    narration_hooks: NarrationHooks | None = None

    @property
    def immediate_consequence(self) -> str:
        return self.next_result


class SetPieceScenePrimitive(SynthesisPrimitiveBase):
    family: Literal["set_piece_scenes"]
    actor_ids: list[str] = Field(default_factory=list)
    scene_anchor: str = Field(min_length=1)
    hinge_action: str = ""
    scene_outcome: str = Field(min_length=1)
    location: str = Field(min_length=1)
    narration_hooks: NarrationHooks | None = None


class HumanCostPrimitive(SynthesisPrimitiveBase):
    family: Literal["human_costs"]
    actor_ids: list[str] = Field(default_factory=list)
    affected_group: str = Field(min_length=1)
    cost_type: str = Field(min_length=1)
    concrete_marker: str = ""
    lived_consequence: str = Field(min_length=1)
    who_saw_it: str = Field(
        min_length=1,
        validation_alias=AliasChoices("who_saw_it", "visibility"),
        serialization_alias="who_saw_it",
    )
    narration_hooks: NarrationHooks | None = None

    @property
    def visibility(self) -> str:
        return self.who_saw_it


class CharacterEnginePrimitive(SynthesisPrimitiveBase):
    family: Literal["character_engines"]
    actor_id: str | None = None
    goal: str = Field(min_length=1)
    pressure_box: str = Field(min_length=1)
    risk_if_it_breaks: str = Field(min_length=1)
    tell: str = ""
    narration_hooks: NarrationHooks | None = None

    @model_validator(mode="before")
    @classmethod
    def map_legacy_character_engine_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("pressure_box"):
            legacy_parts = [
                str(payload.get("constraint", "")).strip(),
                str(payload.get("fear", "")).strip(),
            ]
            payload["pressure_box"] = "; ".join(part for part in legacy_parts if part)
        if not payload.get("risk_if_it_breaks"):
            payload["risk_if_it_breaks"] = str(payload.get("stakes", "")).strip()
        payload.pop("fear", None)
        payload.pop("constraint", None)
        payload.pop("stakes", None)
        return payload


class CoalitionFaultLinePrimitive(SynthesisPrimitiveBase):
    family: Literal["coalitions_and_fault_lines"]
    actor_ids: list[str] = Field(default_factory=list)
    alignment_type: AlignmentType
    coalition_phase: CoalitionPhase
    alignment_shape: str = Field(min_length=1)
    alignment_basis: str = Field(min_length=1)
    fracture_trigger: str = Field(min_length=1)
    narration_hooks: NarrationHooks | None = None

    @model_validator(mode="before")
    @classmethod
    def map_legacy_coalition_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload.setdefault("alignment_shape", str(payload.get("coalition", "")).strip())
        payload.setdefault("alignment_basis", str(payload.get("shared_interest", "")).strip())
        trigger = str(payload.get("stress_point", "")).strip() or str(payload.get("fault_line", "")).strip()
        payload.setdefault("fracture_trigger", trigger)
        payload.pop("coalition", None)
        payload.pop("shared_interest", None)
        payload.pop("fault_line", None)
        payload.pop("stress_point", None)
        return payload


class SystemsOperatingLogicPrimitive(SynthesisPrimitiveBase):
    family: Literal["systems_and_operating_logics"]
    system_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(
        default_factory=list,
        min_length=2,
        validation_alias=AliasChoices("operating_chain", "mechanism_steps"),
        serialization_alias="operating_chain",
    )
    inputs: list[str] = Field(default_factory=list, min_length=1)
    outputs: list[str] = Field(default_factory=list, min_length=1)
    where_it_shows_up: str = ""
    failure_mode: str = Field(min_length=1)
    narration_hooks: NarrationHooks | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_system_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload.pop("mechanism", None)
        return payload

    @property
    def mechanism_steps(self) -> list[str]:
        return list(self.operating_chain)


class ContestedExplanationPrimitive(SynthesisPrimitiveBase):
    family: Literal["contested_explanations"]
    candidate_readings: list[CandidateReading]
    narration_hooks: NarrationHooks | None = None


class MoralTrapPrimitive(SynthesisPrimitiveBase):
    family: Literal["moral_traps"]
    actor_ids: list[str] = Field(default_factory=list)
    competing_obligations: list[str] = Field(default_factory=list, min_length=1)
    compromised_options: list[str] = Field(default_factory=list, min_length=1)
    trap_structure: str = Field(
        min_length=1,
        validation_alias=AliasChoices("trap_structure", "no_clean_exit_reason"),
        serialization_alias="trap_structure",
    )
    narration_hooks: NarrationHooks | None = None

    @property
    def no_clean_exit_reason(self) -> str:
        return self.trap_structure


class IronyReversalPrimitive(SynthesisPrimitiveBase):
    family: Literal["ironies_and_reversals"]
    actor_ids: list[str] = Field(default_factory=list)
    expected_outcome: str = Field(min_length=1)
    actual_outcome: str = Field(min_length=1)
    flip_cause: str = Field(
        min_length=1,
        validation_alias=AliasChoices("flip_cause", "reversal_driver"),
        serialization_alias="flip_cause",
    )
    narration_hooks: NarrationHooks | None = None

    @property
    def reversal_driver(self) -> str:
        return self.flip_cause


AnySynthesisPrimitive = Annotated[
    EpochalTurnPrimitive
    | DecisionPrimitive
    | SetPieceScenePrimitive
    | HumanCostPrimitive
    | CharacterEnginePrimitive
    | CoalitionFaultLinePrimitive
    | SystemsOperatingLogicPrimitive
    | ContestedExplanationPrimitive
    | MoralTrapPrimitive
    | IronyReversalPrimitive
    | SynthesisPrimitive,
    Field(discriminator="family"),
]


SYNTHESIS_PRIMITIVE_FAMILIES: tuple[str, ...] = (
    "epochal_turns",
    "decisions_and_nondecisions",
    "set_piece_scenes",
    "telling_details",
    "human_costs",
    "character_engines",
    "coalitions_and_fault_lines",
    "systems_and_operating_logics",
    "misreadings_and_fantasies",
    "contested_explanations",
    "perspective_windows",
    "moral_traps",
    "afterlives",
    "recurring_images_and_symbols",
    "ironies_and_reversals",
)
#
# These prompt-time family ranges are retention-informed soft targets.
# They preserve family-level flexibility while shifting more budget into
# epochal turns from the wider retained-pool baseline.
#
SYNTHESIS_PRIMITIVE_TARGET_RANGES: dict[str, tuple[int, int]] = {
    "epochal_turns": (30, 38),
    "decisions_and_nondecisions": (25, 32),
    "set_piece_scenes": (18, 31),
    "telling_details": (4, 8),
    "human_costs": (17, 22),
    "character_engines": (14, 22),
    "coalitions_and_fault_lines": (11, 16),
    "systems_and_operating_logics": (12, 18),
    "misreadings_and_fantasies": (5, 8),
    "contested_explanations": (4, 7),
    "perspective_windows": (2, 5),
    "moral_traps": (4, 7),
    "afterlives": (5, 10),
    "recurring_images_and_symbols": (2, 5),
    "ironies_and_reversals": (10, 13),
}
SYNTHESIS_PRIMITIVE_TARGET_MAX_COUNTS: dict[str, int] = {
    family: upper_bound
    for family, (_lower_bound, upper_bound) in SYNTHESIS_PRIMITIVE_TARGET_RANGES.items()
}
SYNTHESIS_PRIMITIVE_FAMILY_SET = set(SYNTHESIS_PRIMITIVE_FAMILIES)
RICH_SYNTHESIS_PRIMITIVE_FAMILIES: tuple[str, ...] = (
    "epochal_turns",
    "decisions_and_nondecisions",
    "set_piece_scenes",
    "human_costs",
    "character_engines",
    "coalitions_and_fault_lines",
    "systems_and_operating_logics",
    "contested_explanations",
    "moral_traps",
    "ironies_and_reversals",
)
RICH_SYNTHESIS_PRIMITIVE_FAMILY_SET = set(RICH_SYNTHESIS_PRIMITIVE_FAMILIES)
GENERIC_SYNTHESIS_PRIMITIVE_FAMILIES: tuple[str, ...] = (
    "telling_details",
    "misreadings_and_fantasies",
    "perspective_windows",
    "afterlives",
    "recurring_images_and_symbols",
)
GENERIC_SYNTHESIS_PRIMITIVE_FAMILY_SET = set(GENERIC_SYNTHESIS_PRIMITIVE_FAMILIES)


def _normalize_family_mapping(
    mapping: dict[str, list[Any]],
    *,
    mapping_name: str,
) -> dict[str, list[Any]]:
    unknown_families = sorted(set(mapping) - SYNTHESIS_PRIMITIVE_FAMILY_SET)
    if unknown_families:
        raise ValueError(
            f"{mapping_name} contains unknown families: {unknown_families}"
        )
    normalized: dict[str, list[Any]] = {}
    for family in SYNTHESIS_PRIMITIVE_FAMILIES:
        normalized[family] = list(mapping.get(family, []))
    return normalized


def _validate_family_bucket_alignment(
    mapping: dict[str, list[AnySynthesisPrimitive]],
    *,
    mapping_name: str,
) -> None:
    mismatches: list[str] = []
    for family, items in mapping.items():
        for item in items:
            if item.family == family:
                continue
            mismatches.append(f"{item.id}: expected {family}, got {item.family}")
    if mismatches:
        preview = ", ".join(mismatches[:10])
        raise ValueError(f"{mapping_name} contains family mismatches: {preview}")


def _drop_invalid_contested_explanations(
    mapping: dict[str, list[AnySynthesisPrimitive]],
) -> dict[str, list[AnySynthesisPrimitive]]:
    mapping["contested_explanations"] = [
        item
        for item in mapping.get("contested_explanations", [])
        if not isinstance(item, ContestedExplanationPrimitive)
        or len(item.candidate_readings) >= 2
    ]
    return mapping


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
    support_primitive_roles: dict[str, SupportPrimitiveRole] = Field(default_factory=dict)
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
            raise ValueError("core_primitive_ids must contain at least one primitive id")
        if len(self.core_primitive_ids) < 5 or len(self.core_primitive_ids) > 7:
            raise ValueError("core_primitive_ids must contain 5-7 primitive ids")
        if len(self.support_primitive_roles) < 5 or len(self.support_primitive_roles) > 7:
            raise ValueError("support_primitive_roles must contain 5-7 primitive ids")

        overlap = sorted(set(self.core_primitive_ids).intersection(self.support_primitive_roles))
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
        if len(self.recall_primitive_ids) > 2:
            raise ValueError("recall_primitive_ids must contain at most 2 primitive ids")
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


class SynthesisPrimitivesArtifact(StrictModel):
    project_id: str
    primitives_by_family: dict[str, list[BaseSynthesisPrimitive]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_primitives(self) -> "SynthesisPrimitivesArtifact":
        normalized = _normalize_family_mapping(
            self.primitives_by_family,
            mapping_name="primitives_by_family",
        )
        _validate_family_bucket_alignment(normalized, mapping_name="primitives_by_family")
        self.primitives_by_family = normalized
        return self


class PrimitiveEnrichmentDeltaBase(StrictModel):
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)


class EpochalTurnPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["epochal_turns"]
    before_state: str = Field(min_length=1)
    after_state: str = Field(min_length=1)
    change_driver: str = Field(min_length=1)
    why_no_return: str = Field(
        min_length=1,
        validation_alias=AliasChoices("why_no_return", "irreversibility_reason"),
        serialization_alias="why_no_return",
    )
    proof_of_change: str = Field(min_length=1)
    narration_hooks: EnrichmentNarrationHooks


class DecisionPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["decisions_and_nondecisions"]
    actor_ids: list[str] = Field(default_factory=list)
    decision_trigger: str = Field(min_length=1)
    decision_question: str = Field(min_length=1)
    decision_mode: Literal["decision", "refusal", "delay", "nondecision"]
    options_considered: list[str] = Field(default_factory=list, min_length=1)
    next_result: str = Field(
        min_length=1,
        validation_alias=AliasChoices("next_result", "immediate_consequence"),
        serialization_alias="next_result",
    )
    narration_hooks: EnrichmentNarrationHooks


class SetPieceScenePrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["set_piece_scenes"]
    actor_ids: list[str] = Field(default_factory=list)
    scene_anchor: str = Field(min_length=1)
    hinge_action: str = Field(min_length=1)
    scene_outcome: str = Field(min_length=1)
    location: str = Field(min_length=1)
    narration_hooks: EnrichmentNarrationHooks


class HumanCostPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["human_costs"]
    actor_ids: list[str] = Field(default_factory=list)
    affected_group: str = Field(min_length=1)
    cost_type: str = Field(min_length=1)
    concrete_marker: str = Field(min_length=1)
    lived_consequence: str = Field(min_length=1)
    who_saw_it: str = Field(
        min_length=1,
        validation_alias=AliasChoices("who_saw_it", "visibility"),
        serialization_alias="who_saw_it",
    )
    narration_hooks: EnrichmentNarrationHooks


class CharacterEnginePrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["character_engines"]
    actor_id: str | None = None
    goal: str = Field(min_length=1)
    pressure_box: str = Field(min_length=1)
    risk_if_it_breaks: str = Field(min_length=1)
    tell: str = Field(min_length=1)
    narration_hooks: EnrichmentNarrationHooks

    @model_validator(mode="before")
    @classmethod
    def map_legacy_character_engine_delta_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("pressure_box"):
            legacy_parts = [
                str(payload.get("constraint", "")).strip(),
                str(payload.get("fear", "")).strip(),
            ]
            payload["pressure_box"] = "; ".join(part for part in legacy_parts if part)
        if not payload.get("risk_if_it_breaks"):
            payload["risk_if_it_breaks"] = str(payload.get("stakes", "")).strip()
        payload.pop("fear", None)
        payload.pop("constraint", None)
        payload.pop("stakes", None)
        return payload

class CoalitionFaultLinePrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["coalitions_and_fault_lines"]
    actor_ids: list[str] = Field(default_factory=list)
    alignment_type: AlignmentType
    coalition_phase: CoalitionPhase
    alignment_shape: str = Field(min_length=1)
    alignment_basis: str = Field(min_length=1)
    fracture_trigger: str = Field(min_length=1)
    narration_hooks: EnrichmentNarrationHooks

    @model_validator(mode="before")
    @classmethod
    def map_legacy_coalition_delta_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload.setdefault("alignment_shape", str(payload.get("coalition", "")).strip())
        payload.setdefault("alignment_basis", str(payload.get("shared_interest", "")).strip())
        trigger = str(payload.get("stress_point", "")).strip() or str(payload.get("fault_line", "")).strip()
        payload.setdefault("fracture_trigger", trigger)
        payload.pop("coalition", None)
        payload.pop("shared_interest", None)
        payload.pop("fault_line", None)
        payload.pop("stress_point", None)
        return payload

class SystemsOperatingLogicPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["systems_and_operating_logics"]
    system_name: str = Field(min_length=1)
    operating_chain: list[str] = Field(
        default_factory=list,
        min_length=2,
        validation_alias=AliasChoices("operating_chain", "mechanism_steps"),
        serialization_alias="operating_chain",
    )
    inputs: list[str] = Field(default_factory=list, min_length=1)
    outputs: list[str] = Field(default_factory=list, min_length=1)
    where_it_shows_up: str = Field(min_length=1)
    failure_mode: str = Field(min_length=1)
    narration_hooks: EnrichmentNarrationHooks

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_system_delta_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload.pop("mechanism", None)
        return payload

class ContestedExplanationPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["contested_explanations"]
    candidate_readings: list[EnrichmentCandidateReading]
    narration_hooks: EnrichmentNarrationHooks


class MoralTrapPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["moral_traps"]
    actor_ids: list[str] = Field(default_factory=list)
    competing_obligations: list[str] = Field(default_factory=list, min_length=1)
    compromised_options: list[str] = Field(default_factory=list, min_length=1)
    trap_structure: str = Field(
        min_length=1,
        validation_alias=AliasChoices("trap_structure", "no_clean_exit_reason"),
        serialization_alias="trap_structure",
    )
    narration_hooks: EnrichmentNarrationHooks


class IronyReversalPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal["ironies_and_reversals"]
    actor_ids: list[str] = Field(default_factory=list)
    expected_outcome: str = Field(min_length=1)
    actual_outcome: str = Field(min_length=1)
    flip_cause: str = Field(
        min_length=1,
        validation_alias=AliasChoices("flip_cause", "reversal_driver"),
        serialization_alias="flip_cause",
    )
    narration_hooks: EnrichmentNarrationHooks


class GenericNarrationOnlyPrimitiveDelta(PrimitiveEnrichmentDeltaBase):
    family: Literal[
        "telling_details",
        "misreadings_and_fantasies",
        "perspective_windows",
        "afterlives",
        "recurring_images_and_symbols",
    ]
    narration_hooks: EnrichmentNarrationHooks


AnyPrimitiveEnrichmentDelta = Annotated[
    EpochalTurnPrimitiveDelta
    | DecisionPrimitiveDelta
    | SetPieceScenePrimitiveDelta
    | HumanCostPrimitiveDelta
    | CharacterEnginePrimitiveDelta
    | CoalitionFaultLinePrimitiveDelta
    | SystemsOperatingLogicPrimitiveDelta
    | ContestedExplanationPrimitiveDelta
    | MoralTrapPrimitiveDelta
    | IronyReversalPrimitiveDelta
    | GenericNarrationOnlyPrimitiveDelta,
    Field(discriminator="family"),
]


class PrimitiveEnrichmentArtifact(StrictModel):
    project_id: str
    family: str = Field(min_length=1)
    enriched_primitives: list[AnyPrimitiveEnrichmentDelta] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_family_alignment(self) -> "PrimitiveEnrichmentArtifact":
        if self.family not in SYNTHESIS_PRIMITIVE_FAMILY_SET:
            raise ValueError(
                "primitive enrichment family must be one of "
                f"{sorted(SYNTHESIS_PRIMITIVE_FAMILY_SET)}"
            )
        for item in self.enriched_primitives:
            if item.family != self.family:
                raise ValueError(
                    f"primitive enrichment item {item.id} expected family {self.family}, got {item.family}"
                )
        return self


class PrimitiveEnrichmentArtifactBase(StrictModel):
    project_id: str
    family: str = Field(min_length=1)
    enriched_primitives: list[PrimitiveEnrichmentDeltaBase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_family_alignment(self) -> "PrimitiveEnrichmentArtifactBase":
        if self.family not in SYNTHESIS_PRIMITIVE_FAMILY_SET:
            raise ValueError(
                "primitive enrichment family must be one of "
                f"{sorted(SYNTHESIS_PRIMITIVE_FAMILY_SET)}"
            )
        for item in self.enriched_primitives:
            if item.family != self.family:
                raise ValueError(
                    f"primitive enrichment item {item.id} expected family {self.family}, got {item.family}"
                )
        return self


class EpochalTurnEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["epochal_turns"]
    enriched_primitives: list[EpochalTurnPrimitiveDelta] = Field(default_factory=list)


class DecisionEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["decisions_and_nondecisions"]
    enriched_primitives: list[DecisionPrimitiveDelta] = Field(default_factory=list)


class SetPieceSceneEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["set_piece_scenes"]
    enriched_primitives: list[SetPieceScenePrimitiveDelta] = Field(default_factory=list)


class HumanCostEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["human_costs"]
    enriched_primitives: list[HumanCostPrimitiveDelta] = Field(default_factory=list)


class CharacterEngineEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["character_engines"]
    enriched_primitives: list[CharacterEnginePrimitiveDelta] = Field(default_factory=list)


class CoalitionFaultLineEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["coalitions_and_fault_lines"]
    enriched_primitives: list[CoalitionFaultLinePrimitiveDelta] = Field(default_factory=list)


class SystemsOperatingLogicEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["systems_and_operating_logics"]
    enriched_primitives: list[SystemsOperatingLogicPrimitiveDelta] = Field(default_factory=list)


class ContestedExplanationEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["contested_explanations"]
    enriched_primitives: list[ContestedExplanationPrimitiveDelta] = Field(default_factory=list)


class MoralTrapEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["moral_traps"]
    enriched_primitives: list[MoralTrapPrimitiveDelta] = Field(default_factory=list)


class IronyReversalEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal["ironies_and_reversals"]
    enriched_primitives: list[IronyReversalPrimitiveDelta] = Field(default_factory=list)


class GenericNarrationOnlyEnrichmentArtifact(PrimitiveEnrichmentArtifactBase):
    family: Literal[
        "telling_details",
        "misreadings_and_fantasies",
        "perspective_windows",
        "afterlives",
        "recurring_images_and_symbols",
    ]
    enriched_primitives: list[GenericNarrationOnlyPrimitiveDelta] = Field(
        default_factory=list
    )


PRIMITIVE_ENRICHMENT_ARTIFACT_MODEL_BY_FAMILY: dict[
    str, type[PrimitiveEnrichmentArtifactBase]
] = {
    "epochal_turns": EpochalTurnEnrichmentArtifact,
    "decisions_and_nondecisions": DecisionEnrichmentArtifact,
    "set_piece_scenes": SetPieceSceneEnrichmentArtifact,
    "human_costs": HumanCostEnrichmentArtifact,
    "character_engines": CharacterEngineEnrichmentArtifact,
    "coalitions_and_fault_lines": CoalitionFaultLineEnrichmentArtifact,
    "systems_and_operating_logics": SystemsOperatingLogicEnrichmentArtifact,
    "contested_explanations": ContestedExplanationEnrichmentArtifact,
    "moral_traps": MoralTrapEnrichmentArtifact,
    "ironies_and_reversals": IronyReversalEnrichmentArtifact,
    "telling_details": GenericNarrationOnlyEnrichmentArtifact,
    "misreadings_and_fantasies": GenericNarrationOnlyEnrichmentArtifact,
    "perspective_windows": GenericNarrationOnlyEnrichmentArtifact,
    "afterlives": GenericNarrationOnlyEnrichmentArtifact,
    "recurring_images_and_symbols": GenericNarrationOnlyEnrichmentArtifact,
}


class SynthesisMap(StrictModel):
    project_id: str
    primitives_by_family: dict[str, list[AnySynthesisPrimitive]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitive_by_id(self) -> dict[str, SynthesisPrimitiveBase]:
        mapping: dict[str, SynthesisPrimitiveBase] = {}
        for family in SYNTHESIS_PRIMITIVE_FAMILIES:
            for item in self.primitives_by_family.get(family, []):
                mapping[item.id] = item
        return mapping

    @model_validator(mode="after")
    def validate_cluster_members(self) -> "SynthesisMap":
        normalized = _normalize_family_mapping(
            self.primitives_by_family,
            mapping_name="primitives_by_family",
        )
        _validate_family_bucket_alignment(normalized, mapping_name="primitives_by_family")
        self.primitives_by_family = _drop_invalid_contested_explanations(normalized)
        return self

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
    arc_threads: list[ActorArcThread] = Field(default_factory=list, min_length=1, max_length=8)

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


class StrategyEpisode(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    thematic_focus: str = Field(default="")
    arc_summary: str = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    episode_spine: EpisodeSpine
    actor_arc_directives: list[ActorArcDirective] = Field(default_factory=list, max_length=4)
    authorial_contract: EpisodeAuthorialContract = Field(
        default_factory=EpisodeAuthorialContract
    )


class SeriesNarratorProfile(StrictModel):
    presence_mode: Literal["visible_host"] = "visible_host"
    baseline_tone: Literal["dry", "plainspoken", "wry", "grave"] = "plainspoken"
    allowed_moves: list[
        Literal[
            "orient",
            "clarify",
            "evaluate",
            "contrast",
            "callback",
            "light_aside",
            "naming_note",
        ]
    ] = Field(
        default_factory=lambda: [
            "orient",
            "clarify",
            "evaluate",
            "contrast",
            "callback",
            "light_aside",
            "naming_note",
        ]
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
    target_host_moves_per_episode: int = Field(default=5, ge=0, le=12)
    analysis_mode: Literal["scene_led", "hybrid", "analysis_forward"] = "hybrid"
    analysis_density: Literal["light", "medium", "high"] = "medium"
    quote_gloss_preference: Literal["avoid", "allow", "prefer"] = "allow"
    clarifier_tolerance: Literal["low", "medium", "high"] = "medium"
    comparative_aside_tolerance: Literal["none", "light", "medium"] = "light"
    wit_ceiling: Literal["none", "dry", "wry"] = "dry"
    target_authorial_passages_per_episode: int = Field(default=3, ge=0, le=6)


class NarrativeStrategy(StrictModel):
    strategy_type: Literal[
        "thesis_driven", "debate", "chronological", "convergence", "mosaic"
    ]
    justification: str = Field(min_length=1)
    series_arc: str = Field(min_length=1)
    episode_arc_outline: list[str] = Field(default_factory=list)
    recommended_episode_count: int | None = Field(default=None, ge=1)
    narrator_profile: SeriesNarratorProfile = Field(default_factory=SeriesNarratorProfile)
    episodes: list[StrategyEpisode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self) -> "NarrativeStrategy":
        episode_numbers = [episode.episode_number for episode in self.episodes]
        if len(episode_numbers) != len(set(episode_numbers)):
            raise ValueError("episodes must not contain duplicate episode_number values")
        if self.episode_arc_outline and len(self.episode_arc_outline) != len(self.episodes):
            raise ValueError("episode_arc_outline must align with episodes when provided")
        if self.recommended_episode_count is not None and self.episodes:
            if self.recommended_episode_count != len(self.episodes):
                raise ValueError("recommended_episode_count must match the number of episodes")
        return self


class FramingBlock(StrictModel):
    opening_image: str = Field(min_length=1)
    threat_or_unresolved_action: str = Field(min_length=1)
    opening_question: str = Field(min_length=1)
    handoff_scene_card_id: str = Field(min_length=1)
    recap: str | None = None
    preview: str | None = None


class AuthorialPassage(StrictModel):
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
    source_primitive_ids: list[str] = Field(default_factory=list, min_length=1, max_length=3)
    source_passage_ids: list[str] = Field(default_factory=list, max_length=4)
    quote_anchor: str = ""
    gloss_seed: str = ""
    must_name_terms: list[str] = Field(default_factory=list, max_length=4)
    budget_sentences: Literal[2, 3, 4, 5] = 3


class ArchitectureSection(StrictModel):
    section_id: str = Field(min_length=1)
    purpose: SectionPurpose
    approx_runtime_minutes: float = Field(gt=0.0)
    primitive_ids: list[str] = Field(default_factory=list, min_length=1)
    section_anchor: str = Field(
        default="",
        validation_alias=AliasChoices("section_anchor", "anchor"),
        serialization_alias="section_anchor",
    )
    must_stage_beats: list[str] = Field(default_factory=list, max_length=4)
    listener_tension: str = Field(
        min_length=1,
        validation_alias=AliasChoices("listener_tension", "section_question"),
        serialization_alias="listener_tension",
    )
    section_turn: str = Field(
        min_length=1,
        validation_alias=AliasChoices("section_turn", "section_resolution"),
        serialization_alias="section_turn",
    )
    transition_logic: str = Field(min_length=1)
    depends_on_section_ids: list[str] = Field(default_factory=list)
    sets_up_section_ids: list[str] = Field(default_factory=list)
    recurrence_role: RecurrenceRole = RecurrenceRole.NONE
    closure_mode: ClosureMode | None = None
    priority_core_passage_ids: list[str] = Field(default_factory=list)
    analysis_goal: str = ""
    key_terms: list[str] = Field(default_factory=list, max_length=6)
    authorial_passages: list[AuthorialPassage] = Field(default_factory=list, max_length=2)

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
            "pressure_type",
            "resolution_type",
            "closure_level",
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
    def populate_closure_mode(self) -> "ArchitectureSection":
        if self.closure_mode is not None:
            return self
        if self.purpose == SectionPurpose.CLOSING:
            self.closure_mode = ClosureMode.FINAL_ANSWER
        elif self.purpose == SectionPurpose.TURN:
            self.closure_mode = ClosureMode.TURN
        else:
            self.closure_mode = ClosureMode.RESIDUE
        return self


class EpisodeArchitecture(StrictModel):
    episode_number: int = Field(ge=1)
    major_turn_section_id: str = Field(min_length=1)
    allowed_recurring_primitive_ids: list[str] = Field(default_factory=list)
    forbidden_redundancies: list[str] = Field(default_factory=list)
    sections: list[ArchitectureSection] = Field(default_factory=list, min_length=9, max_length=12)
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
        ordered_section_ids: list[str] = []
        section_by_id: dict[str, ArchitectureSection] = {}
        for section in self.sections:
            if section.section_id in seen_section_ids:
                raise ValueError("sections must use unique section_id values")
            seen_section_ids.add(section.section_id)
            ordered_section_ids.append(section.section_id)
            section_by_id[section.section_id] = section

        if self.major_turn_section_id not in section_by_id:
            raise ValueError("major_turn_section_id must reference an existing section")

        section_indices = {section_id: idx for idx, section_id in enumerate(ordered_section_ids)}
        for section in self.sections:
            for dependency_id in [*section.depends_on_section_ids, *section.sets_up_section_ids]:
                if dependency_id not in section_by_id:
                    raise ValueError(
                        f"section dependency references unknown section_id: {dependency_id}"
                    )
            if any(
                section_indices[dependency_id] >= section_indices[section.section_id]
                for dependency_id in section.depends_on_section_ids
            ):
                raise ValueError("depends_on_section_ids must reference earlier sections")
            if any(
                section_indices[next_id] <= section_indices[section.section_id]
                for next_id in section.sets_up_section_ids
            ):
                raise ValueError("sets_up_section_ids must reference later sections")
        return self


class SceneActor(StrictModel):
    name: str = Field(min_length=1)
    actor_id: str | None = None
    affiliation: str | None = None
    presence: Literal["primary", "secondary", "background"] = "secondary"


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


class HostMove(StrictModel):
    move_type: Literal[
        "none",
        "orient",
        "clarify",
        "evaluate",
        "contrast",
        "callback",
        "light_aside",
        "naming_note",
    ] = "none"
    note: str = ""
    max_sentences: Literal[1, 2] = 1
    placement: Literal["open", "pivot", "close"] = "close"

    @model_validator(mode="before")
    @classmethod
    def _normalize_host_move(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        move_type = str(cleaned.get("move_type", "none") or "none").strip() or "none"
        cleaned["move_type"] = move_type
        cleaned["note"] = str(cleaned.get("note", "") or "").strip()
        placement = str(cleaned.get("placement", "") or "").strip()
        if not placement:
            cleaned["placement"] = _default_host_move_placement(move_type)
        return cleaned


class _SceneCardBase(StrictModel):
    scene_id: str = Field(default_factory=lambda: f"scene_{new_id()[:8]}")
    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scene_role: SceneRole
    scene_function: SceneFunction
    beat_change: str = Field(
        min_length=1,
        validation_alias=AliasChoices("beat_change", "state_effect"),
        serialization_alias="beat_change",
    )
    must_land_facts: list[str] = Field(default_factory=list)
    entry_image: str = ""
    observable_detail: str = ""
    withhold_until: str | None = None
    timeframe: str | None = None
    location: str | None = None
    actors: list[SceneActor] = Field(default_factory=list, max_length=4)
    passage_ids: list[str] = Field(default_factory=list)
    host_move: HostMove = Field(default_factory=HostMove)

    @model_validator(mode="after")
    def validate_card_shape(self) -> "_SceneCardBase":
        deduped_facts: list[str] = []
        seen_facts: set[str] = set()
        for fact in self.must_land_facts:
            normalized = str(fact or "").strip()
            if not normalized or normalized in seen_facts:
                continue
            seen_facts.add(normalized)
            deduped_facts.append(normalized)
        self.must_land_facts = deduped_facts
        return self


_LEGACY_SCENE_ROLE_TO_ROLE_FUNCTION: dict[str, tuple[str, str | None]] = {
    "setup": (SceneRole.CONTEXT_SETUP.value, SceneFunction.SCENE.value),
    "shock": (SceneRole.SHOCK.value, SceneFunction.SCENE.value),
    "action": (SceneRole.ACTION.value, SceneFunction.SCENE.value),
    "consequence": (SceneRole.FALLOUT.value, SceneFunction.SCENE.value),
    "reaction": (SceneRole.REACTION.value, SceneFunction.SCENE.value),
    "contestation": (SceneRole.CONTESTATION.value, SceneFunction.SCENE.value),
    "synthesis": (SceneRole.IMPLICATION.value, SceneFunction.LANDING.value),
    "process": (SceneRole.ACTION.value, SceneFunction.MECHANISM.value),
    "perspective shift": (SceneRole.IMPLICATION.value, SceneFunction.CALLBACK.value),
    "perspective_shift": (SceneRole.IMPLICATION.value, SceneFunction.CALLBACK.value),
    "reveal": (SceneRole.IMPLICATION.value, SceneFunction.HINGE.value),
    "reversal": (SceneRole.SHOCK.value, SceneFunction.TURN.value),
    "stage_choice": (SceneRole.ACTION.value, SceneFunction.HINGE.value),
    "turn": (SceneRole.SHOCK.value, SceneFunction.TURN.value),
    "closing": (SceneRole.IMPLICATION.value, SceneFunction.LANDING.value),
}

_SCENE_FUNCTION_DEFAULT_BY_ROLE: dict[str, str] = {
    SceneRole.CONTEXT_SETUP.value: SceneFunction.SCENE.value,
    SceneRole.ACTOR_SETUP.value: SceneFunction.SCENE.value,
    SceneRole.ACTION.value: SceneFunction.SCENE.value,
    SceneRole.SHOCK.value: SceneFunction.SCENE.value,
    SceneRole.CONTESTATION.value: SceneFunction.SCENE.value,
    SceneRole.REACTION.value: SceneFunction.SCENE.value,
    SceneRole.FALLOUT.value: SceneFunction.SCENE.value,
    SceneRole.IMPLICATION.value: SceneFunction.LANDING.value,
}


def _migrate_legacy_scene_card(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    cleaned.pop("coverage_depth", None)
    cleaned.pop("dominant_pack_id", None)
    cleaned.pop("dominant_primitive_id", None)
    cleaned.pop("primitive_ids", None)
    cleaned.pop("spine_relation", None)
    role = str(cleaned.get("scene_role", "") or "").strip()
    if role:
        normalized_role, default_function = _LEGACY_SCENE_ROLE_TO_ROLE_FUNCTION.get(
            role,
            (role, None),
        )
        cleaned["scene_role"] = normalized_role
        if not str(cleaned.get("scene_function", "") or "").strip() and default_function:
            cleaned["scene_function"] = default_function
    if not str(cleaned.get("scene_function", "") or "").strip():
        normalized_role = str(cleaned.get("scene_role", "") or "").strip()
        default_function = _SCENE_FUNCTION_DEFAULT_BY_ROLE.get(normalized_role)
        if default_function:
            cleaned["scene_function"] = default_function
    if "beat_change" not in cleaned and cleaned.get("state_effect"):
        cleaned["beat_change"] = cleaned["state_effect"]
    cleaned.pop("state_effect", None)
    cleaned.pop("local_question", None)
    cleaned.pop("what_becomes_legible_later", None)
    cleaned.pop("intended_move", None)
    for actor in cleaned.get("actors", []) or []:
        if isinstance(actor, dict):
            actor.pop("arc_bindings", None)
    host_move = cleaned.get("host_move")
    if not isinstance(host_move, dict):
        cleaned["host_move"] = {
            "move_type": "none",
            "note": "",
            "max_sentences": 1,
            "placement": "close",
        }
    else:
        normalized_host_move = dict(host_move)
        move_type = str(normalized_host_move.get("move_type", "none") or "none").strip() or "none"
        if not str(normalized_host_move.get("placement", "") or "").strip():
            normalized_host_move["placement"] = _default_host_move_placement(move_type)
        cleaned["host_move"] = normalized_host_move
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
    dropped_support_primitive_reasons: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scene_cards(self) -> "EpisodePlanDraft":
        scene_ids = [scene.scene_id for scene in self.scene_cards]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene_cards must use unique scene_id values")
        if self.framing.handoff_scene_card_id not in set(scene_ids):
            raise ValueError("framing.handoff_scene_card_id must reference an existing scene card")
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


class ProseSection(StrictModel):
    section_id: str = Field(default_factory=lambda: f"section_{new_id()[:8]}")
    scene_card_ids: list[str] = Field(default_factory=list, min_length=1)
    movement_goal: str = Field(min_length=1)
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_book_ids: list[str] = Field(default_factory=list)


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
    render_strategy: Literal["plain", "isolate_phrase", "split_sentences", "slow_clause"] = "plain"


class SpokenSection(StrictModel):
    section_id: str
    text: str
    speech_hints: SpeechHints = Field(default_factory=SpeechHints)


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
