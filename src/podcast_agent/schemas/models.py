"""Strict schema contracts for the multi-book thematic podcast pipeline."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 3.1 Project-Level Models
# ---------------------------------------------------------------------------


class ChapterInfo(StrictModel):
    chapter_id: str = Field(default_factory=new_id)
    title: str
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    word_count: int = Field(ge=0)
    summary: str = ""
    analysis: "ChapterAnalysis | None" = None


class ChapterAnalysis(StrictModel):
    themes_touched: list[str] = Field(default_factory=list, max_length=8)
    major_actors: list[str] = Field(default_factory=list, max_length=8)
    key_places: list[str] = Field(default_factory=list, max_length=8)
    key_institutions: list[str] = Field(default_factory=list, max_length=8)
    timeframe: str = ""
    key_events_or_arguments: list[str] = Field(default_factory=list)
    major_tensions: list[str] = Field(default_factory=list, max_length=6)


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
    pre_axis_total_budget: int = Field(default=1200, ge=1)
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
    synthesis_total_passage_cap: int = Field(default=720, ge=1)
    planning_axis_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    planning_axis_min: int = Field(default=10, ge=0)
    planning_axis_max: int = Field(default=15, ge=1)
    planning_total_passage_cap: int = Field(default=300, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=3, ge=0)
    tts_provider: str = "openai"
    tts_concurrency: int = Field(default=12, ge=1)
    episode_planning_concurrency: int = Field(default=9, ge=1)
    episode_write_concurrency: int = Field(default=9, ge=1)
    target_episode_minutes: float = Field(default=140.0, gt=0.0)
    min_episode_minutes: float = Field(default=125.0, gt=0.0)
    duration_shortfall_policy: Literal["warn"] = "warn"
    scene_card_target_min: int = Field(default=25, ge=1)
    scene_card_target_max: int = Field(default=40, ge=1)
    scene_card_target_policy: Literal["warn"] = "warn"
    scene_card_primitives_min: int = Field(default=1, ge=0)
    scene_card_primitives_max: int = Field(default=2, ge=1)
    scene_card_primitive_policy: Literal["warn"] = "warn"
    passage_extraction_concurrency: int = Field(default=8, ge=1)
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
        if self.planning_axis_max < self.planning_axis_min:
            raise ValueError("planning_axis_max must be >= planning_axis_min")
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
    actor_tags: list[str] = Field(default_factory=list)
    institution_tags: list[str] = Field(default_factory=list)

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


class TurningPoint(SynthesisPrimitiveBase):
    pass


class SceneWorthyConsequence(SynthesisPrimitiveBase):
    pass


class CausalMechanism(SynthesisPrimitiveBase):
    pass


class LiveQuestion(SynthesisPrimitiveBase):
    candidate_readings: list[CandidateReading] = Field(default_factory=list, min_length=2)


class EpisodeCandidateCluster(StrictModel):
    cluster_id: str = Field(default_factory=lambda: f"cluster_{new_id()[:8]}")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    primary_member_id: str = Field(min_length=1)
    member_ids: list[str] = Field(min_length=1)
    local_question: str = Field(min_length=1)
    local_payoff_shape: Literal[
        "reveal", "reversal", "escalation", "fallout", "unresolved"
    ]

    @model_validator(mode="after")
    def validate_membership(self) -> "EpisodeCandidateCluster":
        if self.primary_member_id not in self.member_ids:
            raise ValueError("primary_member_id must also appear in member_ids")
        return self


class SynthesisPrimitivesArtifact(StrictModel):
    project_id: str
    turning_points: list[TurningPoint] = Field(default_factory=list)
    scene_worthy_consequences: list[SceneWorthyConsequence] = Field(default_factory=list)
    causal_mechanisms: list[CausalMechanism] = Field(default_factory=list)
    live_questions: list[LiveQuestion] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)


class SynthesisConsolidationResult(StrictModel):
    project_id: str
    episode_candidate_clusters: list[EpisodeCandidateCluster] = Field(default_factory=list)
    turning_point_ids: list[str] = Field(default_factory=list)
    scene_worthy_consequence_ids: list[str] = Field(default_factory=list)
    causal_mechanism_ids: list[str] = Field(default_factory=list)
    live_question_ids: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitive_ids(self) -> set[str]:
        return {
            *self.turning_point_ids,
            *self.scene_worthy_consequence_ids,
            *self.causal_mechanism_ids,
            *self.live_question_ids,
        }

    @model_validator(mode="after")
    def validate_cluster_members(self) -> "SynthesisConsolidationResult":
        primitive_ids = self.primitive_ids()
        for cluster in self.episode_candidate_clusters:
            missing = [member_id for member_id in cluster.member_ids if member_id not in primitive_ids]
            if missing:
                raise ValueError(
                    "episode_candidate_clusters contains unknown member_ids "
                    f"for {cluster.cluster_id}: {missing}"
                )
        return self


class SynthesisMap(StrictModel):
    project_id: str
    episode_candidate_clusters: list[EpisodeCandidateCluster] = Field(default_factory=list)
    turning_points: list[TurningPoint] = Field(default_factory=list)
    scene_worthy_consequences: list[SceneWorthyConsequence] = Field(default_factory=list)
    causal_mechanisms: list[CausalMechanism] = Field(default_factory=list)
    live_questions: list[LiveQuestion] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_notes: list[str] = Field(default_factory=list)

    def primitive_by_id(self) -> dict[str, SynthesisPrimitiveBase]:
        mapping: dict[str, SynthesisPrimitiveBase] = {}
        for item in [
            *self.turning_points,
            *self.scene_worthy_consequences,
            *self.causal_mechanisms,
            *self.live_questions,
        ]:
            mapping[item.id] = item
        return mapping

    @model_validator(mode="after")
    def validate_cluster_members(self) -> "SynthesisMap":
        primitive_ids = set(self.primitive_by_id())
        for cluster in self.episode_candidate_clusters:
            missing = [member_id for member_id in cluster.member_ids if member_id not in primitive_ids]
            if missing:
                raise ValueError(
                    f"episode_candidate_clusters contains unknown member_ids for {cluster.cluster_id}: {missing}"
                )
        return self


# ---------------------------------------------------------------------------
# 3.5 Episode Planning Models
# ---------------------------------------------------------------------------


class ChronologyBreak(StrictModel):
    break_type: Literal["setup_jump", "flashback", "payoff_return"]
    note: str = Field(min_length=1)


class ClusterPathOccurrence(StrictModel):
    occurrence_id: str = Field(default_factory=lambda: f"occ_{new_id()[:8]}")
    cluster_id: str = Field(min_length=1)
    usage: Literal["primary", "echo"]
    transition_note: str = ""
    chronology_break: ChronologyBreak | None = None


class StrategyEpisode(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    driving_question: str = Field(min_length=1)
    thematic_focus: str = Field(default="")
    arc_summary: str = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    cluster_path: list[ClusterPathOccurrence] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def validate_cluster_path(self) -> "StrategyEpisode":
        for idx, occurrence in enumerate(self.cluster_path[1:], start=1):
            if not occurrence.transition_note.strip():
                raise ValueError(
                    f"cluster_path occurrence {idx + 1} must include a transition_note"
                )
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
        cluster_primary_homes: dict[str, int] = {}
        for episode in self.episodes:
            has_primary = False
            for occurrence in episode.cluster_path:
                if occurrence.usage == "primary":
                    has_primary = True
                    if occurrence.cluster_id not in cluster_primary_homes:
                        cluster_primary_homes[occurrence.cluster_id] = episode.episode_number
                    elif cluster_primary_homes[occurrence.cluster_id] != episode.episode_number:
                        raise ValueError(
                            f"cluster {occurrence.cluster_id} has multiple primary home episodes"
                        )
            if not has_primary:
                raise ValueError(
                    f"episode {episode.episode_number} must contain at least one primary cluster"
                )
        return self


class FramingBlock(StrictModel):
    opening_image: str = Field(min_length=1)
    threat_or_unresolved_action: str = Field(min_length=1)
    opening_question: str = Field(min_length=1)
    handoff_scene_card_id: str = Field(min_length=1)
    recap: str | None = None
    preview: str | None = None


class SceneActor(StrictModel):
    name: str = Field(min_length=1)
    role_in_scene: str = Field(min_length=1)
    affiliation: str | None = None


class SceneCard(StrictModel):
    scene_id: str = Field(default_factory=lambda: f"scene_{new_id()[:8]}")
    title: str = Field(min_length=1)
    card_kind: Literal["normal", "bridge"] = "normal"
    scene_role: str = Field(min_length=1)
    dominant_cluster_occurrence_id: str | None = None
    bridge_from_occurrence_id: str | None = None
    bridge_to_occurrence_id: str | None = None
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
    estimated_duration_seconds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_card_shape(self) -> "SceneCard":
        if not self.scene_role.strip():
            raise ValueError("scene_role must not be blank")
        if self.card_kind == "normal":
            if not self.dominant_cluster_occurrence_id:
                raise ValueError("normal scene cards require dominant_cluster_occurrence_id")
            if self.bridge_from_occurrence_id or self.bridge_to_occurrence_id:
                raise ValueError("normal scene cards must not define bridge occurrence ids")
        else:
            if not self.bridge_from_occurrence_id or not self.bridge_to_occurrence_id:
                raise ValueError("bridge scene cards require bridge_from_occurrence_id and bridge_to_occurrence_id")
            if self.dominant_cluster_occurrence_id is not None:
                raise ValueError("bridge scene cards must not define dominant_cluster_occurrence_id")
        return self


class EpisodePlanDraft(StrictModel):
    episode_number: int = Field(ge=1)
    title: str = Field(min_length=1)
    driving_question: str = Field(min_length=1)
    thematic_focus: str = ""
    arc_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    framing: FramingBlock
    scene_cards: list[SceneCard] = Field(default_factory=list, min_length=1)
    target_duration_minutes: float = Field(default=140.0, gt=0.0)

    @model_validator(mode="after")
    def validate_scene_cards(self) -> "EpisodePlanDraft":
        scene_ids = [scene.scene_id for scene in self.scene_cards]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene_cards must use unique scene_id values")
        if self.framing.handoff_scene_card_id not in set(scene_ids):
            raise ValueError("framing.handoff_scene_card_id must reference an existing scene card")
        bridge_count = sum(1 for scene in self.scene_cards if scene.card_kind == "bridge")
        if bridge_count > 1:
            raise ValueError("at most one bridge scene card is allowed per episode")
        return self


class EpisodePlan(EpisodePlanDraft):
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


class ScriptTransition(StrictModel):
    transition_id: str = Field(default_factory=lambda: f"transition_{new_id()[:8]}")
    after_section_id: str = Field(min_length=1)
    before_section_id: str | None = None
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_book_ids: list[str] = Field(default_factory=list)


class WindowMapEntry(StrictModel):
    batch_id: str = Field(min_length=1)
    section_ids: list[str] = Field(default_factory=list)
    transition_ids: list[str] = Field(default_factory=list)


class EpisodeScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    framing: FramingBlock
    prose_sections: list[ProseSection] = Field(default_factory=list)
    transitions: list[ScriptTransition] = Field(default_factory=list)
    window_map: list[WindowMapEntry] = Field(default_factory=list)
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


class SpokenTransition(StrictModel):
    transition_id: str
    text: str
    speech_hints: SpeechHints = Field(default_factory=SpeechHints)


class SpokenScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    framing: FramingBlock
    sections: list[SpokenSection] = Field(default_factory=list)
    transitions: list[SpokenTransition] = Field(default_factory=list)
    tts_provider: str = "openai"


class RenderSegment(StrictModel):
    segment_id: str = Field(default_factory=new_id)
    text: str
    voice_id: str = "ballad"
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
