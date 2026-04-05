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
    key_events_or_arguments: list[str] = Field(default_factory=list, max_length=6)
    major_tensions: list[str] = Field(default_factory=list, max_length=6)
    causal_shifts: list[str] = Field(default_factory=list, max_length=6)
    narrative_hooks: list[str] = Field(default_factory=list, max_length=5)
    retrieval_keywords: list[str] = Field(default_factory=list, max_length=12)


class BookRecord(StrictModel):
    book_id: str = Field(default_factory=new_id)
    title: str
    author: str
    source_path: str
    source_type: str  # "pdf", "txt", "md"
    chapters: list[ChapterInfo] = Field(default_factory=list)
    chunk_count: int = Field(default=0, ge=0)
    total_words: int = Field(default=0, ge=0)
    ingestion_diagnostics: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(StrictModel):
    max_axes: int = Field(default=15, ge=1)
    min_axes: int = Field(default=5, ge=1)
    passage_retrieval_percentage: float = Field(default=0.25, gt=0.0, le=1.0)
    passage_retrieval_min_per_book: int = Field(default=20, ge=1)
    passage_retrieval_max_per_book: int = Field(default=50, ge=1)
    axis_candidate_target_total: int = Field(default=180, ge=1)
    admission_floor_per_book: int = Field(default=2, ge=0)
    retrieval_conf_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    retrieval_size_basis: Literal["total_words"] = "total_words"
    retrieval_size_exponent: float = Field(default=0.68, ge=0.0)
    retrieval_relevance_power: float = Field(default=1.2, ge=0.0)
    retrieval_soft_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    chapter_penalty_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    rerank_top_k: int = Field(default=30, ge=1)
    synthesis_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=3, ge=0)
    tts_provider: str = "openai"
    tts_concurrency: int = Field(default=4, ge=1)
    episode_write_concurrency: int = Field(default=5, ge=1)
    target_episode_minutes: float = Field(default=100.0, gt=0.0)
    min_episode_minutes: float = Field(default=90.0, gt=0.0)
    duration_shortfall_policy: Literal["warn"] = "warn"
    passage_extraction_concurrency: int = Field(default=8, ge=1)
    chunk_max_words: int = Field(default=400, ge=50)
    chunk_overlap_words: int = Field(default=50, ge=0)
    spoken_chunk_max_words: int = Field(default=250, ge=50)
    max_author_names_per_episode: int = Field(default=3, ge=0)
    attribution_budget: float = Field(default=0.2, ge=0.0, le=1.0)
    prefer_indirect_attribution: bool = True
    skip_grounding: bool = False
    skip_spoken_delivery: bool = False
    skip_audio: bool = False

    @model_validator(mode="after")
    def validate_retrieval_budget_bounds(self) -> PipelineConfig:
        if self.passage_retrieval_max_per_book < self.passage_retrieval_min_per_book:
            raise ValueError(
                "passage_retrieval_max_per_book must be >= passage_retrieval_min_per_book"
            )
        return self


class ThematicProject(StrictModel):
    project_id: str = Field(default_factory=new_id)
    theme: str
    theme_elaboration: str | None = None
    sub_themes: list[str] = Field(default_factory=list, max_length=8)
    books: list[BookRecord] = Field(default_factory=list)
    requested_episode_count: int | None = Field(default=None, ge=1)
    recommended_episode_count: int | None = Field(default=None, ge=2, le=8)
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

        if len(normalized) > 8:
            raise ValueError("sub_themes supports at most 8 entries.")
        return normalized


# ---------------------------------------------------------------------------
# 3.2 Chunk & Retrieval Models
# ---------------------------------------------------------------------------


class ChunkingConfig(StrictModel):
    max_chunk_words: int = Field(default=400, ge=50)
    overlap_words: int = Field(default=50, ge=0)
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


class SynthesisInsight(StrictModel):
    insight_id: str = Field(default_factory=new_id)
    insight_type: InsightType
    title: str
    description: str
    passage_ids: list[str] = Field(min_length=2)
    axis_ids: list[str] = Field(default_factory=list)
    podcast_potential: float = Field(default=0.5, ge=0.0, le=1.0)
    treatment: Literal["debate", "build", "contrast", "resolve", "leave_open"] = "contrast"


class NarrativeThread(StrictModel):
    thread_id: str = Field(default_factory=new_id)
    title: str
    description: str
    insight_ids: list[str] = Field(default_factory=list)
    arc_type: Literal["convergence", "divergence", "evolution", "dialectic", "deepening"] = "convergence"


class MergedNarrative(StrictModel):
    topic: str
    narrative: str
    source_passage_ids: list[str] = Field(default_factory=list)
    points_of_consensus: list[str] = Field(default_factory=list)
    points_of_disagreement: list[str] = Field(default_factory=list)


class EpisodeSynthesisTension(StrictModel):
    tension_id: str
    question: str


class EpisodeMergedNarrativeRef(StrictModel):
    merged_narrative_id: str
    topic: str
    narrative: str
    source_passage_ids: list[str] = Field(default_factory=list)


class SynthesisMap(StrictModel):
    project_id: str
    insights: list[SynthesisInsight] = Field(default_factory=list)
    narrative_threads: list[NarrativeThread] = Field(default_factory=list)
    book_relationship_matrix: dict[str, dict[str, str]] = Field(default_factory=dict)
    unresolved_tensions: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    merged_narratives: list[MergedNarrative] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3.5 Episode Planning Models
# ---------------------------------------------------------------------------


class NarrativeStrategy(StrictModel):
    @model_validator(mode="after")
    def validate_episode_arc_details(self) -> NarrativeStrategy:
        detail_numbers = [detail.episode_number for detail in self.episode_arc_details]
        if len(detail_numbers) != len(set(detail_numbers)):
            raise ValueError("episode_arc_details must not contain duplicate episode_number values.")

        if self.episode_arc_outline and len(self.episode_arc_outline) != len(self.episode_arc_details):
            raise ValueError(
                "episode_arc_outline and episode_arc_details must have the same length when outlines are provided."
            )

        assignment_numbers = {assignment.episode_number for assignment in self.episode_assignments}
        detail_number_set = set(detail_numbers)
        if assignment_numbers and assignment_numbers != detail_number_set:
            missing = sorted(assignment_numbers - detail_number_set)
            extra = sorted(detail_number_set - assignment_numbers)
            parts: list[str] = []
            if missing:
                parts.append(f"missing episode_arc_details for episode_numbers={missing}")
            if extra:
                parts.append(f"unexpected episode_arc_details for episode_numbers={extra}")
            raise ValueError("episode_arc_details must align with episode_assignments: " + "; ".join(parts))
        return self

    strategy_type: Literal[
        "thesis_driven", "debate", "chronological", "convergence", "mosaic"
    ]
    justification: str
    series_arc: str
    episode_arc_outline: list[str] = Field(default_factory=list)
    episode_arc_details: list["EpisodeArcDetail"]
    recommended_episode_count: int | None = Field(default=None, ge=2, le=8)
    episode_assignments: list["EpisodeAssignment"] = Field(default_factory=list)


class EpisodeAssignment(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    thematic_focus: str = ""
    axis_ids: list[str] = Field(default_factory=list)
    insight_ids: list[str] = Field(default_factory=list)
    merged_narrative_ids: list[str] = Field(default_factory=list)
    tension_ids: list[str] = Field(default_factory=list)
    episode_strategy: str = ""


class EpisodeArcDetail(StrictModel):
    episode_number: int = Field(ge=1)
    arc_summary: str
    narrative_stakes: str
    progression_beats: list[str] = Field(default_factory=list, min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list, min_length=1)
    payoff_shape: str


class CrossReference(StrictModel):
    from_book_id: str
    to_book_id: str
    connection_type: Literal["agrees", "disagrees", "extends", "provides_example"]
    bridge_note: str = ""


class SpineSegment(StrictModel):
    segment_id: str = Field(default_factory=new_id)
    narrative_text: str
    source_passages: list[str] = Field(default_factory=list)
    segment_function: Literal[
        "scene_setting",
        "event",
        "context",
        "consequence",
        "turning_point",
        "tension",
        "resolution",
    ]
    era_or_moment: str = ""


class AttributionMoment(StrictModel):
    moment_id: str = Field(default_factory=new_id)
    insert_after_segment_id: str
    disagreement_type: Literal["factual", "interpretive", "causal"]
    description: str
    books_involved: list[str] = Field(default_factory=list)
    narrative_function: Literal[
        "complicates_simple_reading",
        "reveals_hidden_motive",
        "shows_stakes_of_interpretation",
        "listener_must_choose",
    ] = "complicates_simple_reading"
    suggested_treatment: Literal[
        "brief_aside",
        "extended_exploration",
        "rhetorical_question",
        "cliffhanger",
    ] = "brief_aside"


class NarrativeSpine(StrictModel):
    episode_number: int = Field(ge=1)
    spine_segments: list[SpineSegment] = Field(default_factory=list)
    attribution_moments: list[AttributionMoment] = Field(default_factory=list)
    narrative_voice: str = "omniscient narrator telling a story"


class EpisodeBeat(StrictModel):
    beat_id: str = Field(default_factory=new_id)
    description: str
    passage_ids: list[str] = Field(default_factory=list)
    primary_book_id: str = ""
    supporting_book_ids: list[str] = Field(default_factory=list)
    synthesis_instruction: str | None = None
    narrative_instruction: Literal[
        "set_the_scene",
        "advance_events",
        "explain_context",
        "build_tension",
        "reveal_consequence",
        "pivot_to_new_thread",
    ] = "advance_events"
    attribution_level: Literal["none", "light", "full"] = "none"
    transition_hint: str | None = None
    estimated_duration_seconds: int = Field(default=120, ge=0)


class EpisodeSynthesisContext(StrictModel):
    insights: list[SynthesisInsight] = Field(default_factory=list)
    narrative_threads: list[NarrativeThread] = Field(default_factory=list)
    merged_narratives: list[EpisodeMergedNarrativeRef] = Field(default_factory=list)
    unresolved_tensions: list[EpisodeSynthesisTension] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EpisodePlan(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    thematic_focus: str = ""
    axis_ids: list[str] = Field(default_factory=list)
    insight_ids: list[str] = Field(default_factory=list)
    beats: list[EpisodeBeat] = Field(default_factory=list)
    attribution_budget: float = Field(default=0.2, ge=0.0, le=1.0)
    narrative_spine: NarrativeSpine | None = None
    synthesis_context: EpisodeSynthesisContext | None = None
    book_balance: dict[str, float] = Field(default_factory=dict)
    cross_references: list[CrossReference] = Field(default_factory=list)
    target_duration_minutes: float = Field(default=100.0, gt=0.0)
    episode_strategy: str = ""


# ---------------------------------------------------------------------------
# 3.6 Script & Validation Models
# ---------------------------------------------------------------------------


class Citation(StrictModel):
    citation_id: str = Field(default_factory=new_id)
    text_span: str
    passage_id: str
    book_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ScriptSegment(StrictModel):
    segment_id: str = Field(default_factory=new_id)
    text: str
    segment_type: Literal["intro", "body", "transition", "outro", "recap", "bridge"] = "body"
    beat_id: str | None = None
    source_book_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    attribution_level: Literal["none", "light", "full"] = "none"


class EpisodeScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    segments: list[ScriptSegment] = Field(default_factory=list)
    total_word_count: int = Field(default=0, ge=0)
    estimated_duration_seconds: int = Field(default=0, ge=0)
    citations: list[Citation] = Field(default_factory=list)


class ClaimAssessment(StrictModel):
    claim_text: str
    cited_passage_id: str
    status: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "FABRICATED"]
    explanation: str = ""


class CrossBookClaimAssessment(StrictModel):
    claim_text: str
    book_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)
    comparison_valid: bool = True
    failure_reason: str | None = None


class FairnessFlag(StrictModel):
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
    segment_id: str
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
# 3.7 Speech & Audio Models
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        alias_map = {
            "delivery_style": "style",
            "emphasis_level": "intensity",
            "speech_rate": "pace",
        }
        for legacy_key, canonical_key in alias_map.items():
            if canonical_key not in normalized and legacy_key in normalized:
                normalized[canonical_key] = normalized.pop(legacy_key)
        return normalized

    @field_validator("style", mode="before")
    @classmethod
    def _normalize_style(cls, value: Any) -> str:
        allowed = {"neutral", "measured", "urgent", "dramatic"}
        if value is None:
            return "neutral"
        normalized = str(value).strip().lower()
        return normalized if normalized in allowed else "neutral"

    @field_validator("intensity", mode="before")
    @classmethod
    def _normalize_intensity(cls, value: Any) -> str:
        aliases = {
            "high": "strong",
            "maximum": "strong",
            "max": "strong",
            "low": "light",
        }
        allowed = {"none", "light", "medium", "strong"}
        if value is None:
            return "none"
        normalized = str(value).strip().lower()
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in allowed else "none"

    @field_validator("pace", mode="before")
    @classmethod
    def _normalize_pace(cls, value: Any) -> str:
        aliases = {
            "slow": "slower",
            "deliberate": "slower",
            "conversational": "normal",
            "fast": "faster",
        }
        allowed = {"slower", "normal", "faster"}
        if value is None:
            return "normal"
        normalized = str(value).strip().lower()
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in allowed else "normal"

    @field_validator("pause_before_ms", "pause_after_ms", mode="before")
    @classmethod
    def _clamp_pause_ms(cls, value: Any) -> int:
        if value is None:
            return 300
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 300
        return max(0, min(2000, numeric))

    @field_validator("pronunciation_hints", mode="before")
    @classmethod
    def _normalize_pronunciation_hints(
        cls,
        value: Any,
    ) -> list[dict[str, Any]] | list[PronunciationHint]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator("emphasis_targets", mode="before")
    @classmethod
    def _normalize_emphasis_targets(cls, value: Any) -> list[str]:
        if value is None:
            return []
        raw_values = [value] if isinstance(value, str) else list(value)
        normalized: list[str] = []
        for item in raw_values:
            phrase = str(item or "").strip()
            if phrase and phrase not in normalized:
                normalized.append(phrase)
        return normalized

    @property
    def delivery_style(self) -> str:
        return self.style

    @property
    def emphasis_level(self) -> str:
        return self.intensity

    @property
    def speech_rate(self) -> str:
        return self.pace


SpokenDeliveryHints = SpeechHints


class SpokenSegment(StrictModel):
    segment_id: str
    text: str
    max_words: int = Field(default=250, ge=1)
    speech_hints: SpeechHints = Field(default_factory=SpeechHints)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_hint_field(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "speech_hints" not in normalized and "ssml_hints" in normalized:
            normalized["speech_hints"] = normalized.pop("ssml_hints")
        return normalized

    @field_validator("speech_hints", mode="before")
    @classmethod
    def _default_speech_hints(cls, value: Any) -> dict[str, Any] | SpeechHints:
        return {} if value is None else value

    @property
    def ssml_hints(self) -> SpeechHints:
        return self.speech_hints


class SpokenScript(StrictModel):
    episode_number: int = Field(ge=1)
    title: str
    segments: list[SpokenSegment] = Field(default_factory=list)
    arc_plan: str | None = None
    tts_provider: str = "openai"


class EpisodeFraming(StrictModel):
    episode_number: int = Field(ge=1)
    recap: str | None = None
    preview: str | None = None
    cold_open: str | None = None


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
