"""Strict schema contracts for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class SupportPackRole(str, Enum):
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
    synthesis_total_passage_cap: int = Field(default=500, ge=1)
    synthesis_floor_budget_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    synthesis_axis_floor_min: int = Field(default=10, ge=0)
    synthesis_axis_floor_max: int = Field(default=15, ge=1)
    synthesis_axis_ceiling_multiplier: float = Field(default=1.4, ge=1.0)
    synthesis_trim_top_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    synthesis_trim_mid_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    synthesis_trim_top_keep_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    synthesis_trim_mid_keep_fraction: float = Field(default=0.40, gt=0.0, le=1.0)
    synthesis_trim_tail_keep_fraction: float = Field(default=0.30, gt=0.0, le=1.0)
    planning_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    planning_axis_min: int = Field(default=10, ge=0)
    planning_axis_max: int = Field(default=15, ge=1)
    planning_total_passage_cap: int = Field(default=300, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=3, ge=0)
    tts_provider: str = "openai"
    tts_concurrency: int = Field(default=12, ge=1)
    episode_planning_concurrency: int = Field(default=6, ge=1)
    episode_write_concurrency: int = Field(default=6, ge=1)
    spoken_delivery_concurrency: int | None = Field(default=None, ge=1)
    target_episode_minutes: float = Field(default=90.0, gt=0.0)
    min_episode_minutes: float = Field(default=85.0, gt=0.0)
    duration_shortfall_policy: Literal["warn"] = "warn"
    scene_card_target_min: int = Field(default=35, ge=1)
    scene_card_target_max: int = Field(default=45, ge=1)
    scene_card_target_policy: Literal["warn"] = "warn"
    scene_card_primitives_min: int = Field(default=1, ge=0)
    scene_card_primitives_max: int = Field(default=2, ge=1)
    scene_card_primitive_policy: Literal["warn"] = "warn"
    passage_extraction_concurrency: int = Field(default=16, ge=1)
    chunk_max_words: int = Field(default=1000, ge=50)
    chunk_overlap_words: int = Field(default=75, ge=0)
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
        if (self.synthesis_trim_top_fraction + self.synthesis_trim_mid_fraction) > 1.0:
            raise ValueError("synthesis trim top and mid fractions must sum to <= 1.0")
        if self.scene_card_target_max < self.scene_card_target_min:
            raise ValueError("scene_card_target_max must be >= scene_card_target_min")
        if self.scene_card_primitives_max < self.scene_card_primitives_min:
            raise ValueError("scene_card_primitives_max must be >= scene_card_primitives_min")
        return self


class ThematicProject(StrictModel):
    project_id: str = Field(default_factory=new_id)
    theme: str
    theme_elaboration: str | None = None
    sub_themes: list[str] = Field(default_factory=list, max_length=15)
    books: list[BookRecord] = Field(default_factory=list)
    requested_episode_count: int | None = Field(default=None, ge=1)
    recommended_episode_count: int | None = Field(default=None, ge=6, le=10)
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
        if len(normalized) > 15:
            raise ValueError("sub_themes supports at most 15 entries.")
        return normalized


# ---------------------------------------------------------------------------
# 3.2 Chunk & Retrieval Models
# ---------------------------------------------------------------------------


class ChunkingConfig(StrictModel):
    max_chunk_words: int = Field(default=1000, ge=50)
    overlap_words: int = Field(default=75, ge=0)
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


class CandidateReading(StrictModel):
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    support_passage_ids: list[str] = Field(default_factory=list)


class SynthesisPrimitive(SynthesisPrimitiveBase):
    """Family-agnostic synthesis primitive."""

    candidate_readings: list[CandidateReading] = Field(default_factory=list)


class TurningPoint(SynthesisPrimitive):
    pass


class SceneWorthyConsequence(SynthesisPrimitive):
    pass


class CausalMechanism(SynthesisPrimitive):
    pass


class LiveQuestion(SynthesisPrimitive):
    pass


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
    "worlds_in_collision",
    "ironies_and_reversals",
)
SYNTHESIS_PRIMITIVE_FAMILY_SET = set(SYNTHESIS_PRIMITIVE_FAMILIES)


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


def _drop_invalid_contested_explanations(
    mapping: dict[str, list[SynthesisPrimitive]],
) -> dict[str, list[SynthesisPrimitive]]:
    mapping["contested_explanations"] = [
        item
        for item in mapping.get("contested_explanations", [])
        if len(item.candidate_readings) >= 2
    ]
    return mapping


class EvidencePack(StrictModel):
    pack_id: str = Field(default_factory=lambda: f"pack_{new_id()[:8]}")
    title: str = Field(min_length=1)
    local_summary: str = Field(min_length=1)
    primitive_ids: list[str] = Field(min_length=1)
    actor_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pack(self) -> "EvidencePack":
        seen_primitive_ids: set[str] = set()
        deduped_primitive_ids: list[str] = []
        for primitive_id in self.primitive_ids:
            if primitive_id in seen_primitive_ids:
                continue
            seen_primitive_ids.add(primitive_id)
            deduped_primitive_ids.append(primitive_id)
        self.primitive_ids = deduped_primitive_ids

        seen_actor_ids: set[str] = set()
        deduped_actor_ids: list[str] = []
        for actor_id in self.actor_ids:
            if not actor_id or actor_id in seen_actor_ids:
                continue
            seen_actor_ids.add(actor_id)
            deduped_actor_ids.append(actor_id)
        self.actor_ids = deduped_actor_ids
        return self

class EpisodeSpine(StrictModel):
    listener_question: str = Field(min_length=1)
    working_claim: str = Field(min_length=1)
    target_end_state: str = Field(min_length=1)
    verdict_mode: VerdictMode
    primary_counterposition: str = Field(min_length=1)
    spine_pack_ids: list[str] = Field(min_length=1, max_length=3)
    support_pack_roles: dict[str, SupportPackRole] = Field(default_factory=dict)
    allowed_recalls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spine(self) -> "EpisodeSpine":
        seen_pack_ids: set[str] = set()
        deduped_spine_pack_ids: list[str] = []
        for pack_id in self.spine_pack_ids:
            if not pack_id or pack_id in seen_pack_ids:
                continue
            seen_pack_ids.add(pack_id)
            deduped_spine_pack_ids.append(pack_id)
        self.spine_pack_ids = deduped_spine_pack_ids
        if not self.spine_pack_ids:
            raise ValueError("spine_pack_ids must contain at least one pack id")
        if len(self.spine_pack_ids) > 3:
            raise ValueError("spine_pack_ids supports at most 3 pack ids")

        overlap = sorted(set(self.spine_pack_ids).intersection(self.support_pack_roles))
        if overlap:
            raise ValueError(f"support packs cannot also appear in spine_pack_ids: {overlap}")

        allowed_recalls: list[str] = []
        seen_recall_ids: set[str] = set()
        reserved_pack_ids = set(self.spine_pack_ids).union(self.support_pack_roles)
        for pack_id in self.allowed_recalls:
            if not pack_id or pack_id in seen_recall_ids or pack_id in reserved_pack_ids:
                continue
            seen_recall_ids.add(pack_id)
            allowed_recalls.append(pack_id)
        self.allowed_recalls = allowed_recalls
        return self

    @property
    def assigned_pack_ids(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for pack_id in [
            *self.spine_pack_ids,
            *self.support_pack_roles.keys(),
            *self.allowed_recalls,
        ]:
            if not pack_id or pack_id in seen:
                continue
            seen.add(pack_id)
            ordered.append(pack_id)
        return ordered


class SynthesisPrimitivesArtifact(StrictModel):
    project_id: str
    primitives_by_family: dict[str, list[SynthesisPrimitive]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_primitives(self) -> "SynthesisPrimitivesArtifact":
        normalized = _normalize_family_mapping(
            self.primitives_by_family,
            mapping_name="primitives_by_family",
        )
        self.primitives_by_family = _drop_invalid_contested_explanations(normalized)
        return self


class SynthesisConsolidationResult(StrictModel):
    project_id: str
    evidence_packs: list[EvidencePack] = Field(default_factory=list)
    primitive_ids_by_family: dict[str, list[str]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitive_ids(self) -> set[str]:
        ids: set[str] = set()
        for family_ids in self.primitive_ids_by_family.values():
            ids.update(family_ids)
        return ids

    @model_validator(mode="after")
    def validate_cluster_members(self) -> "SynthesisConsolidationResult":
        self.primitive_ids_by_family = _normalize_family_mapping(
            self.primitive_ids_by_family,
            mapping_name="primitive_ids_by_family",
        )
        primitive_ids = self.primitive_ids()
        for pack in self.evidence_packs:
            missing = [member_id for member_id in pack.primitive_ids if member_id not in primitive_ids]
            if missing:
                raise ValueError(
                    "evidence_packs contains unknown primitive_ids "
                    f"for {pack.pack_id}: {missing}"
                )
        return self

class SynthesisMap(StrictModel):
    project_id: str
    evidence_packs: list[EvidencePack] = Field(default_factory=list)
    primitives_by_family: dict[str, list[SynthesisPrimitive]] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitive_by_id(self) -> dict[str, SynthesisPrimitive]:
        mapping: dict[str, SynthesisPrimitive] = {}
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
        self.primitives_by_family = _drop_invalid_contested_explanations(normalized)
        primitive_ids = set(self.primitive_by_id())
        for pack in self.evidence_packs:
            missing = [member_id for member_id in pack.primitive_ids if member_id not in primitive_ids]
            if missing:
                raise ValueError(
                    f"evidence_packs contains unknown primitive_ids for {pack.pack_id}: {missing}"
                )
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
    payoff: str = ""


class ActorArcDirective(StrictModel):
    actor_id: str = Field(min_length=1)
    arc_threads: list[ActorArcThread] = Field(default_factory=list, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_unique_thread_ids(self) -> "ActorArcDirective":
        thread_ids = [thread.thread_id for thread in self.arc_threads]
        if len(thread_ids) != len(set(thread_ids)):
            raise ValueError("actor arc directive thread ids must be unique")
        return self


class StrategyEpisode(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    driving_question: str = Field(min_length=1)
    thematic_focus: str = Field(default="")
    arc_summary: str = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    episode_spine: EpisodeSpine
    actor_arc_directives: list[ActorArcDirective] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_spine(self) -> "StrategyEpisode":
        if self.driving_question != self.episode_spine.listener_question:
            self.driving_question = self.episode_spine.listener_question
        return self


class NarrativeStrategy(StrictModel):
    strategy_type: Literal[
        "thesis_driven", "debate", "chronological", "convergence", "mosaic"
    ]
    justification: str = Field(min_length=1)
    series_arc: str = Field(min_length=1)
    episode_arc_outline: list[str] = Field(default_factory=list)
    recommended_episode_count: int | None = Field(default=None, ge=6, le=10)
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
        pack_primary_homes: dict[str, int] = {}
        for episode in self.episodes:
            if not episode.episode_spine.spine_pack_ids:
                raise ValueError(
                    f"episode {episode.episode_number} must contain at least one spine pack"
                )
            for pack_id in episode.episode_spine.assigned_pack_ids:
                if pack_id in episode.episode_spine.allowed_recalls:
                    continue
                if pack_id not in pack_primary_homes:
                    pack_primary_homes[pack_id] = episode.episode_number
                elif pack_primary_homes[pack_id] != episode.episode_number:
                    raise ValueError(
                        f"pack {pack_id} has multiple primary home episodes"
                    )
        return self


class FramingBlock(StrictModel):
    opening_image: str = Field(min_length=1)
    threat_or_unresolved_action: str = Field(min_length=1)
    opening_question: str = Field(min_length=1)
    handoff_scene_card_id: str = Field(min_length=1)
    recap: str | None = None
    preview: str | None = None


class SceneActorArcBinding(StrictModel):
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


class SceneActor(StrictModel):
    name: str = Field(min_length=1)
    actor_id: str | None = None
    affiliation: str | None = None
    presence: Literal["primary", "secondary", "background"] = "secondary"
    arc_bindings: list[SceneActorArcBinding] = Field(default_factory=list, max_length=2)


class _SceneCardBase(StrictModel):
    scene_id: str = Field(default_factory=lambda: f"scene_{new_id()[:8]}")
    batch_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scene_role: str = Field(min_length=1)
    dominant_pack_id: str | None = None
    spine_relation: SpineRelation
    state_effect: str = Field(min_length=1)
    entry_image: str = ""
    local_question: str = ""
    observable_detail: str = ""
    withhold_until: str | None = None
    what_becomes_legible_later: str | None = None
    intended_move: str = ""
    timeframe: str | None = None
    location: str | None = None
    actors: list[SceneActor] = Field(default_factory=list, max_length=4)
    primitive_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_card_shape(self) -> "_SceneCardBase":
        if not self.scene_role.strip():
            raise ValueError("scene_role must not be blank")
        if not self.dominant_pack_id:
            raise ValueError("scene cards require dominant_pack_id")
        return self


def _migrate_legacy_scene_card(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    cleaned.setdefault("batch_id", "b01")
    cleaned.pop("coverage_depth", None)
    role = cleaned.get("scene_role")
    if role == "process":
        cleaned["scene_role"] = "action"
    elif role == "perspective shift":
        cleaned["scene_role"] = "perspective_shift"
    return cleaned


class SceneCardDraft(_SceneCardBase):
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = _migrate_legacy_scene_card(data)
        cleaned = dict(cleaned)
        cleaned.pop("estimated_duration_seconds", None)
        return cleaned


class SceneCard(_SceneCardBase):
    estimated_duration_seconds: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        return _migrate_legacy_scene_card(data)


class EpisodePlanDraft(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    driving_question: str = Field(min_length=1)
    thematic_focus: str = ""
    arc_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    episode_spine: EpisodeSpine
    actor_arc_directives: list[ActorArcDirective] = Field(default_factory=list, max_length=4)
    framing: FramingBlock
    scene_cards: list[SceneCardDraft] = Field(default_factory=list, min_length=1)
    dropped_support_pack_reasons: dict[str, str] = Field(default_factory=dict)
    target_duration_minutes: float = Field(default=90.0, gt=0.0)

    @model_validator(mode="after")
    def validate_scene_cards(self) -> "EpisodePlanDraft":
        scene_ids = [scene.scene_id for scene in self.scene_cards]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene_cards must use unique scene_id values")
        seen_batch_ids: set[str] = set()
        current_batch_id: str | None = None
        for scene in self.scene_cards:
            batch_id = scene.batch_id.strip()
            if not batch_id:
                raise ValueError("scene_cards require non-blank batch_id values")
            if current_batch_id is None:
                current_batch_id = batch_id
                seen_batch_ids.add(batch_id)
                continue
            if batch_id == current_batch_id:
                continue
            if batch_id in seen_batch_ids:
                raise ValueError("scene_cards must use contiguous batch_id runs")
            current_batch_id = batch_id
            seen_batch_ids.add(batch_id)
        if self.framing.handoff_scene_card_id not in set(scene_ids):
            raise ValueError("framing.handoff_scene_card_id must reference an existing scene card")
        if self.driving_question != self.episode_spine.listener_question:
            self.driving_question = self.episode_spine.listener_question
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
