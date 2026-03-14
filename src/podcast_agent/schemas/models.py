"""Strict schema contracts for the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model with strict validation defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)


class SourceType(str, Enum):
    """Supported source formats."""

    TEXT = "text"
    MARKDOWN = "markdown"


class BookIngestionResult(StrictModel):
    """Metadata captured from source ingestion."""

    book_id: str
    title: str
    author: str
    source_path: str
    source_type: SourceType
    raw_text: str
    ingested_at: datetime = Field(default_factory=utc_now)


class BookChunk(StrictModel):
    """Canonical chunk record derived from a chapter or section."""

    chunk_id: str
    chapter_id: str
    chapter_title: str
    chapter_number: int = Field(ge=1)
    sequence: int = Field(ge=1)
    text: str
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    source_offsets: list[int] = Field(min_length=2, max_length=2)
    themes: list[str] = Field(default_factory=list)


class BookChapter(StrictModel):
    """Canonical chapter record with chunk references."""

    chapter_id: str
    chapter_number: int = Field(ge=1)
    title: str
    summary: str
    chunk_ids: list[str]


class StructuredChunkDraft(StrictModel):
    """Intermediate chunk emitted during chapter-level structuring."""

    text: str
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    source_offsets: list[int] = Field(min_length=2, max_length=2)
    themes: list[str] = Field(default_factory=list)


class StructuredChapter(StrictModel):
    """Intermediate chapter structure used before merging a full book."""

    chapter_number: int = Field(ge=1)
    title: str
    summary: str
    chunks: list[StructuredChunkDraft]


class BookStructure(StrictModel):
    """Normalized structure emitted by the structuring agent."""

    book_id: str
    title: str
    chapters: list[BookChapter]
    chunks: list[BookChunk]
    created_at: datetime = Field(default_factory=utc_now)


class ContinuityArc(StrictModel):
    """Cross-chapter continuity signal extracted from the book."""

    arc_id: str
    label: str
    description: str
    chapter_ids: list[str]


class EpisodeCluster(StrictModel):
    """Candidate episode group identified during analysis."""

    cluster_id: str
    label: str
    rationale: str
    chapter_ids: list[str]
    chunk_ids: list[str]
    themes: list[str]


class BookAnalysis(StrictModel):
    """Analytical context used to create the series plan."""

    book_id: str
    themes: list[str]
    continuity_arcs: list[ContinuityArc]
    notable_claims: list[str]
    episode_clusters: list[EpisodeCluster]
    created_at: datetime = Field(default_factory=utc_now)


class EpisodeBeat(StrictModel):
    """Beat inside an episode plan."""

    beat_id: str
    title: str
    objective: str
    chunk_ids: list[str]
    claim_requirements: list[str]


class EpisodePlan(StrictModel):
    """One podcast episode plan."""

    episode_id: str
    sequence: int = Field(ge=1)
    title: str
    synopsis: str
    chapter_ids: list[str]
    chunk_ids: list[str]
    themes: list[str]
    beats: list[EpisodeBeat]


class SeriesPlan(StrictModel):
    """Series-level plan for the full book adaptation."""

    book_id: str
    format: Literal["single_narrator"]
    strategy_summary: str
    episodes: list[EpisodePlan]
    created_at: datetime = Field(default_factory=utc_now)


class RetrievalHit(StrictModel):
    """Retrieved chunk used for writing or validation."""

    chunk_id: str
    chapter_id: str
    chapter_title: str
    score: float
    text: str


class ScriptClaim(StrictModel):
    """Claim included in a script segment."""

    claim_id: str
    text: str
    evidence_chunk_ids: list[str]


class EpisodeSegment(StrictModel):
    """One narratable segment in an episode script."""

    segment_id: str
    beat_id: str
    heading: str
    narration: str
    claims: list[ScriptClaim]
    citations: list[str]


class EpisodeScript(StrictModel):
    """Single-episode script emitted by the writing stage."""

    episode_id: str
    title: str
    narrator: str
    segments: list[EpisodeSegment]
    created_at: datetime = Field(default_factory=utc_now)


class GroundingStatus(str, Enum):
    """Grounding status for a script claim."""

    GROUNDED = "grounded"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


class ClaimAssessment(StrictModel):
    """Grounding assessment for one claim."""

    claim_id: str
    status: GroundingStatus
    reason: str
    evidence_chunk_ids: list[str]


class GroundingReport(StrictModel):
    """Claim-level validation report for an episode script."""

    episode_id: str
    overall_status: Literal["pass", "fail"]
    claim_assessments: list[ClaimAssessment]
    validated_at: datetime = Field(default_factory=utc_now)


class RepairResult(StrictModel):
    """Outcome of a repair attempt."""

    episode_id: str
    attempt: int = Field(ge=1)
    repaired_segment_ids: list[str]
    script: EpisodeScript
    report: GroundingReport


class RenderSegment(StrictModel):
    """Final renderable segment."""

    segment_id: str
    speaker: str
    text: str
    ssml: str
    grounded_claim_ids: list[str]


class RenderManifest(StrictModel):
    """Validated TTS-ready output."""

    episode_id: str
    title: str
    narrator: str
    segments: list[RenderSegment]
    generated_at: datetime = Field(default_factory=utc_now)


class AudioSegmentFile(StrictModel):
    """Metadata for one rendered segment inside an episode audio output."""

    segment_id: str
    speaker: str
    text: str
    grounded_claim_ids: list[str]


class AudioManifest(StrictModel):
    """Audio synthesis output for one episode."""

    episode_id: str
    title: str
    narrator: str
    voice: str
    audio_path: str
    audio_format: str
    segments: list[AudioSegmentFile]
    generated_at: datetime = Field(default_factory=utc_now)


class EpisodeOutput(StrictModel):
    """Canonical per-episode artifact persisted by the pipeline."""

    plan: EpisodePlan
    script: EpisodeScript
    report: GroundingReport
    manifest: RenderManifest | None = None
    audio_manifest: AudioManifest | None = None
    repair_attempts: list[RepairResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
