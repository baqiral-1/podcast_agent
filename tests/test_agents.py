"""Unit tests for individual agent classes."""

from __future__ import annotations

from threading import Event, Lock
import json
import time

import pytest
from pydantic import ValidationError

from podcast_agent.llm.base import LLMClient
from podcast_agent.agents import (
    AnalysisAgent,
    EpisodePlanningAgent,
    GroundingValidationAgent,
    RepairAgent,
    StructuringAgent,
    WritingAgent,
)
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    BeatScript,
    BookAnalysis,
    BookChapter,
    BookChunk,
    BookIngestionResult,
    BookStructure,
    EpisodeBeat,
    EpisodeCluster,
    EpisodePlan,
    EpisodeSegment,
    EpisodeScript,
    GroundingReport,
    GroundingStatus,
    RetrievalHit,
    SeriesPlan,
    ScriptClaim,
    SourceType,
)
from podcast_agent.utils import split_into_chapters


BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory. The historian argues that the harbor served as a relay for scholars, not soldiers.

Chapter 2: Signals
Inside the city archive, the team finds coded letters describing the observatory's mirrors and the way they coordinated messages across valleys. The engineer explains that the mirror network only worked because local guilds kept the routes maintained.

Chapter 3: Fracture
The discovery creates tension among the sponsors, who want the observatory framed as a military outpost. The historian pushes back and points to logs showing that teachers, navigators, and astronomers shared the site.

Chapter 4: Legacy
By the end of the journey, the team understands that the observatory connected communities through knowledge exchange. The final journal entries argue that its legacy was cooperation, memory, and scientific practice.
"""


def _build_ingestion() -> BookIngestionResult:
    return BookIngestionResult(
        book_id="observatory-book",
        title="Observatory Book",
        author="A. Writer",
        source_path="/tmp/observatory-book.txt",
        source_type=SourceType.TEXT,
        raw_text=BOOK_TEXT,
    )


def _build_structure():
    agent = StructuringAgent(HeuristicLLMClient())
    return agent.structure(_build_ingestion())


def _build_analysis_and_plan():
    llm = HeuristicLLMClient()
    structure = _build_structure()
    analysis = AnalysisAgent(llm).analyze(structure, episode_count=2)
    plan = EpisodePlanningAgent(llm).plan(structure, analysis, episode_count=2)
    return structure, analysis, plan


def _build_retrieval_hits(structure, chunk_ids: list[str]) -> list[RetrievalHit]:
    chapter_titles = {chapter.chapter_id: chapter.title for chapter in structure.chapters}
    hits = []
    for chunk in structure.chunks:
        if chunk.chunk_id not in chunk_ids:
            continue
        hits.append(
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                chapter_id=chunk.chapter_id,
                chapter_title=chapter_titles[chunk.chapter_id],
                score=1.0,
                text=chunk.text,
            )
        )
    return hits


def _build_script_and_report() -> tuple[EpisodeScript, GroundingReport, list[RetrievalHit]]:
    llm = HeuristicLLMClient()
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    retrieval_hits = _build_retrieval_hits(structure, episode.chunk_ids)
    script = WritingAgent(llm).write(episode, retrieval_hits)
    report = GroundingValidationAgent(llm).validate(script, retrieval_hits)
    return script, report, retrieval_hits


class BoundaryOnlyStructuringLLM(LLMClient):
    """LLM stub that returns metadata-only chunk plans."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            draft = payload["draft"]
            return response_model.model_validate(
                {
                    "chapter_number": draft["chapter_number"],
                    "title": draft["title"],
                    "summary": "Boundary-only summary.",
                    "chunks": [
                        {
                            "start_word": draft["chunks"][0]["start_word"],
                            "end_word": draft["chunks"][1]["end_word"],
                            "themes": ["observatory"],
                        },
                        {
                            "start_word": draft["chunks"][2]["start_word"],
                            "end_word": draft["chunks"][-1]["end_word"],
                            "themes": [],
                        },
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class MissingMetadataStructuringLLM(LLMClient):
    """LLM stub that omits redundant chapter metadata."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            draft = payload["draft"]
            return response_model.model_validate(
                {
                    "summary": "",
                    "chunks": [
                        {
                            "start_word": draft["chunks"][0]["start_word"],
                            "end_word": draft["chunks"][-1]["end_word"],
                            "themes": ["observatory"],
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class FailingStructuringBoundaryLLM(LLMClient):
    """LLM stub that should never be reached for oversized chapters."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            raise AssertionError("oversized chapters should skip LLM structuring")
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class ParallelWindowStructuringLLM(LLMClient):
    """LLM stub that records whether windowed structuring overlaps in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = Lock()
        self._overlap_ready = Event()

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            with self._lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
                if self.active_calls >= 2:
                    self._overlap_ready.set()
            try:
                if payload.get("window") is not None and self.max_active_calls == 1:
                    self._overlap_ready.wait(timeout=0.5)
                time.sleep(0.05)
                draft = payload["draft"]
                return response_model.model_validate(
                    {
                        "chapter_number": draft["chapter_number"],
                        "title": draft["title"],
                        "summary": draft["summary"],
                        "chunks": [
                            {
                                "start_word": chunk["start_word"],
                                "end_word": chunk["end_word"],
                                "themes": chunk.get("themes", []),
                            }
                            for chunk in draft["chunks"]
                        ],
                    }
                )
            finally:
                with self._lock:
                    self.active_calls -= 1
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class RetryableStructuring400LLM(LLMClient):
    """LLM stub that raises one transport-style 400 before succeeding."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("LLM request failed with status 400: parse error")
            draft = payload["draft"]
            return response_model.model_validate(
                {
                    "chapter_number": draft["chapter_number"],
                    "title": draft["title"],
                    "summary": draft["summary"],
                    "chunks": [
                        {
                            "start_word": chunk["start_word"],
                            "end_word": chunk["end_word"],
                            "themes": chunk.get("themes", []),
                        }
                        for chunk in draft["chunks"]
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class Always400StructuringLLM(LLMClient):
    """LLM stub that always raises the transport-style 400 error."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            self.calls += 1
            raise RuntimeError("LLM request failed with status 400: parse error")
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class NonRetryableStructuringRuntimeLLM(LLMClient):
    """LLM stub that raises a non-400 runtime error without the extra retry."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            self.calls += 1
            raise RuntimeError("LLM request timed out after 300 seconds")
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class CitationLoggingLLM(LLMClient):
    """LLM stub that leaves claim evidence empty so derived citations stay empty."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            beat = payload["beat"]
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": "A grounded narration.",
                            "claims": [
                                {
                                    "text": "A claim with no evidence ids.",
                                    "evidence_chunk_ids": [],
                                }
                            ],
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class CapturingRepairLLM(LLMClient):
    """LLM stub that captures the narrowed repair payload."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "episode_repair":
            failed_segments = []
            for segment in payload["failed_segments"]:
                repaired = dict(segment)
                repaired.pop("segment_id", None)
                repaired.pop("beat_id", None)
                repaired.pop("citations", None)
                repaired["claims"] = [
                    {
                        "text": claim["text"],
                        "evidence_chunk_ids": claim["evidence_chunk_ids"],
                    }
                    for claim in repaired["claims"]
                ]
                failed_segments.append(repaired)
            self.payloads.append(payload)
            return response_model.model_validate(
                {
                    "episode_id": payload["episode_id"],
                    "attempt": payload["attempt"],
                    "repaired_segment_ids": [segment["segment_id"] for segment in payload["failed_segments"]],
                    "repaired_segments": failed_segments,
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class RetryingClaimWritingLLM(LLMClient):
    """LLM stub that fails the first beat attempt, then returns valid claim evidence."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.calls += 1
            beat = payload["beat"]
            chunk_ids = [hit["chunk_id"] for hit in payload["retrieval_hits"][:2]]
            if self.calls == 1:
                raise ValidationError.from_exception_data(
                    "BeatScript",
                    [
                        {
                            "type": "too_short",
                            "loc": ("segments", 0, "claims"),
                            "msg": "List should have at least 1 item after validation, not 0",
                            "input": [],
                            "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0},
                        }
                    ],
                )
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": "A longer grounded narration for the retry path." * 8,
                            "claims": [
                                {
                                    "text": "Grounded claim one.",
                                    "evidence_chunk_ids": [chunk_ids[0]],
                                },
                            ]
                            + (
                                [
                                    {
                                        "text": "Grounded claim two.",
                                        "evidence_chunk_ids": [chunk_ids[1]],
                                    }
                                ]
                                if len(chunk_ids) > 1
                                else []
                            ),
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class SegmentRecordingValidationLLM(LLMClient):
    """LLM stub that records segment-scoped validation payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "grounding_report":
            self.payloads.append(payload)
            script = payload["script"]
            assessments = []
            for segment in script["segments"]:
                for claim in segment["claims"]:
                    assessments.append(
                        {
                            "claim_id": claim["claim_id"],
                            "status": GroundingStatus.GROUNDED,
                            "reason": "Segment-scoped validation.",
                            "evidence_chunk_ids": claim["evidence_chunk_ids"],
                        }
                    )
            return response_model.model_validate(
                {
                    "episode_id": script["episode_id"],
                    "overall_status": "pass",
                    "claim_assessments": assessments,
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class ParallelValidationLLM(LLMClient):
    """LLM stub that blocks until multiple segment validations are in flight."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active = 0
        self.max_active = 0
        self._gate = Event()

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "grounding_report":
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                if self.max_active >= 3:
                    self._gate.set()
            self._gate.wait(timeout=1.0)
            script = payload["script"]
            assessments = []
            for segment in script["segments"]:
                for claim in segment["claims"]:
                    assessments.append(
                        {
                            "claim_id": claim["claim_id"],
                            "status": GroundingStatus.GROUNDED,
                            "reason": "Parallel validation.",
                            "evidence_chunk_ids": claim["evidence_chunk_ids"],
                        }
                    )
            with self._lock:
                self._active -= 1
            return response_model.model_validate(
                {
                    "episode_id": script["episode_id"],
                    "overall_status": "pass",
                    "claim_assessments": assessments,
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class AlwaysInvalidClaimWritingLLM(LLMClient):
    """LLM stub that never returns valid claim evidence for a beat."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            raise ValidationError.from_exception_data(
                "BeatScript",
                [
                    {
                        "type": "too_short",
                        "loc": ("segments", 0, "claims"),
                        "msg": "List should have at least 1 item after validation, not 0",
                        "input": [],
                        "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0},
                    }
                ],
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class CoverageAwareRetryWritingLLM(LLMClient):
    """LLM stub that under-covers first, then succeeds after receiving retry guidance."""

    def __init__(self) -> None:
        self.instructions_seen: list[str] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.instructions_seen.append(instructions)
            beat = payload["beat"]
            chunk_ids = [hit["chunk_id"] for hit in payload["retrieval_hits"]]
            if len(self.instructions_seen) == 1:
                return response_model.model_validate(
                    {
                        "segments": [
                            {
                                "heading": beat["title"],
                                "narration": "A grounded but too narrow narration." * 6,
                                "claims": [
                                    {
                                        "text": "Only covers the first chunk.",
                                        "evidence_chunk_ids": [chunk_ids[0]],
                                    }
                                ],
                            }
                        ],
                    }
                )
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": ("A grounded narration covering one assigned chunk in detail. " * 8).strip(),
                            "claims": [
                                {
                                    "text": f"Covers assigned chunk {index}.",
                                    "evidence_chunk_ids": [chunk_id],
                                }
                            ],
                        }
                        for index, chunk_id in enumerate(chunk_ids, start=1)
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class CitationMismatchRetryWritingLLM(LLMClient):
    """LLM stub that cites a chunk without attaching it to claim evidence, then repairs on retry."""

    def __init__(self) -> None:
        self.instructions_seen: list[str] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.instructions_seen.append(instructions)
            beat = payload["beat"]
            chunk_ids = [hit["chunk_id"] for hit in payload["retrieval_hits"]]
            if len(self.instructions_seen) == 1:
                return response_model.model_validate(
                    {
                        "segments": [
                            {
                                "heading": beat["title"],
                                "narration": ("A grounded narration with mismatched citations. " * 8).strip(),
                                "claims": [
                                    {
                                        "text": "Grounded claim one.",
                                        "evidence_chunk_ids": [chunk_ids[1]],
                                    }
                                ],
                            }
                        ],
                    }
                )
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": ("A repaired grounded narration that fully covers the assigned material in compact spoken form. " * 8).strip(),
                            "claims": [
                                {
                                    "text": f"Grounded claim {index}.",
                                    "evidence_chunk_ids": [chunk_id],
                                }
                                for index, chunk_id in enumerate(chunk_ids, start=1)
                            ],
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class TruncatingThenValidWritingLLM(LLMClient):
    """LLM stub that truncates once before succeeding."""

    def __init__(self) -> None:
        self.calls = 0
        self.instructions_seen: list[str] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.calls += 1
            self.instructions_seen.append(instructions)
            if self.calls == 1:
                raise RuntimeError("LLM response was truncated because it hit the completion token limit.")
            beat = payload["beat"]
            chunk_ids = [hit["chunk_id"] for hit in payload["retrieval_hits"]]
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": ("Compact grounded narration. " * 16).strip(),
                            "claims": [
                                {
                                    "text": f"Grounded claim {index}.",
                                    "evidence_chunk_ids": [chunk_id],
                                }
                                for index, chunk_id in enumerate(chunk_ids, start=1)
                            ],
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class OutOfOrderWritingLLM(LLMClient):
    """LLM stub that finishes beats out of order while returning valid output."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            beat = payload["beat"]
            if beat["beat_id"].endswith("1"):
                time.sleep(0.05)
            chunk_ids = [hit["chunk_id"] for hit in payload["retrieval_hits"]]
            return response_model.model_validate(
                {
                    "segments": [
                        {
                            "heading": beat["title"],
                            "narration": ("Ordered grounded narration. " * 12).strip(),
                            "claims": [
                                {
                                    "text": "Grounded claim.",
                                    "evidence_chunk_ids": chunk_ids[:1],
                                }
                            ],
                        }
                    ],
                }
            )
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


def test_structuring_agent_creates_chapters_and_chunks() -> None:
    structure = _build_structure()

    assert structure.chapters
    assert structure.chunks
    assert all(chapter.title for chapter in structure.chapters)
    assert all(chunk.chunk_id.startswith("observatory-book-chapter-") for chunk in structure.chunks)


def test_structuring_rebuilds_chunk_text_from_metadata_only_response() -> None:
    long_ingestion = _build_ingestion().model_copy(
        update={
            "raw_text": """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory. The historian argues that the harbor served as a relay for scholars, not soldiers. The expedition catalogues every signal tower, cross-checks old manifests, and compares copies of damaged letters. When the group returns to the archive, it reconstructs a timeline of arrivals, repairs, and vanished couriers.
"""
        }
    )
    structure = StructuringAgent(
        BoundaryOnlyStructuringLLM(),
        max_chunk_words=20,
        chunk_overlap_words=0,
    ).structure(long_ingestion)

    chapter_one_chunks = [chunk for chunk in structure.chunks if chunk.chapter_number == 1]

    assert len(chapter_one_chunks) == 2
    assert "The expedition arrives at the northern harbor" in chapter_one_chunks[0].text
    assert chapter_one_chunks[0].themes == ["observatory"]
    assert chapter_one_chunks[1].text


def test_structuring_fills_missing_plan_metadata_from_draft() -> None:
    structure = StructuringAgent(
        MissingMetadataStructuringLLM(),
        max_chunk_words=20,
        chunk_overlap_words=0,
    ).structure(_build_ingestion())

    chapter_one = next(chapter for chapter in structure.chapters if chapter.chapter_number == 1)
    chapter_one_chunks = [chunk for chunk in structure.chunks if chunk.chapter_number == 1]

    assert chapter_one.title == "Chapter 1: Arrival"
    assert chapter_one.summary
    assert chapter_one_chunks[0].themes == ["observatory"]


def test_structuring_skips_oversized_chapters() -> None:
    agent = StructuringAgent(
        FailingStructuringBoundaryLLM(),
        max_chunk_words=200,
        chunk_overlap_words=0,
        max_structuring_llm_chapter_words=75000,
    )

    chapter = agent._structure_chapter(1, "Chapter 1: Giant", ("observatory " * 76000).strip())

    assert chapter is None


def test_structuring_parallelizes_window_fallback_calls() -> None:
    llm = ParallelWindowStructuringLLM()
    long_ingestion = _build_ingestion().model_copy(
        update={
            "raw_text": "Chapter 1: Arrival\n" + ("observatory signals legacy " * 1100).strip(),
        }
    )
    StructuringAgent(
        llm,
        max_structuring_chapter_words=500,
        structuring_parallelism=3,
        structuring_window_words=600,
        structuring_window_overlap_words=0,
    ).structure(long_ingestion)

    assert llm.max_active_calls >= 2


def test_structuring_retries_once_after_retryable_400() -> None:
    llm = RetryableStructuring400LLM()
    agent = StructuringAgent(llm, max_structuring_chapter_words=5000)

    chapter = agent._structure_chapter(1, "Chapter 1: Signals", ("observatory " * 300).strip())

    assert chapter is not None
    assert llm.calls == 2


def test_structuring_falls_back_after_second_retryable_400() -> None:
    llm = Always400StructuringLLM()
    agent = StructuringAgent(llm, max_structuring_chapter_words=5000)

    chapter = agent._structure_chapter(1, "Chapter 1: Signals", ("observatory " * 300).strip())

    assert chapter is not None
    assert chapter.title == "Chapter 1: Signals"
    assert llm.calls == 2


def test_structuring_does_not_extra_retry_non_400_runtime_errors() -> None:
    llm = NonRetryableStructuringRuntimeLLM()
    agent = StructuringAgent(llm, max_structuring_chapter_words=5000)

    chapter = agent._structure_chapter(1, "Chapter 1: Signals", ("observatory " * 300).strip())

    assert chapter is not None
    assert chapter.title == "Chapter 1: Signals"
    assert llm.calls == 1


def test_writing_diagnostics_distinguish_claim_and_segment_citations(tmp_path) -> None:
    llm = CitationLoggingLLM()
    run_logger = RunLogger(tmp_path / "runs")
    run_logger.bind_run("citation-run")
    llm.set_run_logger(run_logger)
    writer = WritingAgent(llm, beat_parallelism=1)
    retrieval_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{index}",
            chapter_id="chapter-1",
            chapter_title="Chapter 1",
            score=1.0,
            text=("word " * 120).strip(),
        )
        for index in range(1, 4)
    ]
    episode = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        synopsis="Synopsis",
        chapter_ids=["chapter-1"],
        chunk_ids=[hit.chunk_id for hit in retrieval_hits],
        themes=["alpha"],
        beats=[
            EpisodeBeat(
                beat_id="beat-1",
                title="Beat 1",
                objective="Objective",
                chunk_ids=[hit.chunk_id for hit in retrieval_hits],
                claim_requirements=[],
            )
            ],
        )

    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]
    beat_script = BeatScript.model_construct(
        beat_id="beat-1",
        segments=[
            EpisodeSegment.model_construct(
                segment_id="segment-1",
                beat_id="beat-1",
                heading="Beat 1",
                narration="A grounded narration.",
                claims=[
                    ScriptClaim.model_construct(
                        claim_id="beat-1-claim-1",
                        text="A claim with no evidence ids.",
                        evidence_chunk_ids=[],
                    )
                ],
                citations=[],
            )
        ],
    )
    violations = writer._compliance_violations(beat_script, beat_payload)
    writer._log_writing_metrics(beat_script, beat_payload, violations, retried=False)

    lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "citation-run" / "run.log").read_text(encoding="utf-8").splitlines()
    ]
    diagnostics = [line for line in lines if line["event_type"] == "writing_diagnostics"]
    assert diagnostics
    payload = diagnostics[-1]["payload"]
    assert payload["claim_cited_chunk_count"] == 0
    assert payload["segment_citation_chunk_count"] == 0
    assert payload["cited_chunk_count"] == 0
    assert payload["segments_with_zero_citations"] == ["segment-1"]
    assert payload["claims_with_zero_evidence"] == ["beat-1-claim-1"]


def test_writing_retries_when_initial_script_has_no_claims() -> None:
    llm = RetryingClaimWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    beat = episode.beats[0]
    retrieval_hits = _build_retrieval_hits(structure, beat.chunk_ids)
    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]

    beat_script = writer._write_beat(beat_payload)

    assert llm.calls == 2
    assert beat_script.segments
    assert beat_script.beat_id == beat.beat_id
    assert beat_script.segments[0].segment_id == f"{beat.beat_id}-segment-1"
    assert all(segment.beat_id == beat.beat_id for segment in beat_script.segments)
    assert all(segment.claims for segment in beat_script.segments)
    assert all(claim.evidence_chunk_ids for segment in beat_script.segments for claim in segment.claims)
    assert all(claim.claim_id.startswith(f"{beat.beat_id}-claim-") for segment in beat_script.segments for claim in segment.claims)


def test_writing_fails_after_retry_when_claims_remain_invalid() -> None:
    llm = AlwaysInvalidClaimWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    beat = episode.beats[0]
    retrieval_hits = _build_retrieval_hits(structure, beat.chunk_ids)
    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]

    try:
        writer._write_beat(beat_payload)
    except RuntimeError as exc:
        assert "Beat script generation failed after retry" in str(exc)
    else:
        raise AssertionError("expected invalid claim-only beat script generation to fail")


def test_writing_retry_instructions_include_missing_chunk_ids() -> None:
    llm = CoverageAwareRetryWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)
    retrieval_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{index}",
            chapter_id="chapter-1",
            chapter_title="Chapter 1",
            score=1.0,
            text=("word " * 120).strip(),
        )
        for index in range(1, 4)
    ]
    episode = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        synopsis="Synopsis",
        chapter_ids=["chapter-1"],
        chunk_ids=[hit.chunk_id for hit in retrieval_hits],
        themes=["alpha"],
        beats=[
            EpisodeBeat(
                beat_id="beat-1",
                title="Beat 1",
                objective="Objective",
                chunk_ids=[hit.chunk_id for hit in retrieval_hits],
                claim_requirements=[],
            )
        ],
    )
    beat = episode.beats[0]
    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]

    beat_script = writer._write_beat(beat_payload)

    assert beat_script.segments
    assert len(llm.instructions_seen) == 2
    retry_instructions = llm.instructions_seen[1]
    assert "every assigned chunk_id must appear in claim evidence_chunk_ids" in retry_instructions
    assert "Revise and expand the previous draft instead of restarting from a blank outline." in retry_instructions
    assert "Preserve valid narration from the previous draft while fixing missing claim evidence." in retry_instructions
    assert "A missing chunk is not covered if it appears only in narration without being attached to a claim." in retry_instructions
    for chunk_id in beat.chunk_ids[1:]:
        assert chunk_id in retry_instructions


def test_writing_retry_instructions_include_cited_only_chunk_ids() -> None:
    llm = CitationMismatchRetryWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)
    retrieval_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{index}",
            chapter_id="chapter-1",
            chapter_title="Chapter 1",
            score=1.0,
            text=("word " * 120).strip(),
        )
        for index in range(1, 4)
    ]
    episode = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        synopsis="Synopsis",
        chapter_ids=["chapter-1"],
        chunk_ids=[hit.chunk_id for hit in retrieval_hits],
        themes=["alpha"],
        beats=[
            EpisodeBeat(
                beat_id="beat-1",
                title="Beat 1",
                objective="Objective",
                chunk_ids=[hit.chunk_id for hit in retrieval_hits],
                claim_requirements=[],
            )
        ],
    )
    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]

    beat_script = writer._write_beat(beat_payload)

    assert beat_script.segments
    assert len(llm.instructions_seen) == 2
    retry_instructions = llm.instructions_seen[1]
    assert "chunk-1" in retry_instructions
    assert "Revise and expand the previous draft instead of restarting from a blank outline." in retry_instructions


def test_writing_instructions_use_simplified_hard_rules_prompt() -> None:
    instructions = WritingAgent.instructions

    assert "Write spoken narration, not notes or analysis." in instructions
    assert "Coverage requirements:" in instructions
    assert "Hard rules:" in instructions
    assert "1. Every segment must contain at least one claim." in instructions
    assert "If the beat has 7 or fewer assigned chunk_ids" in instructions
    assert "Do not invent, rename, or omit assigned chunk ids in claim evidence." in instructions


def test_writing_retries_after_truncation_with_compact_retry_instructions() -> None:
    llm = TruncatingThenValidWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    beat = episode.beats[0]
    retrieval_hits = _build_retrieval_hits(structure, beat.chunk_ids)
    payload = writer.build_payload(episode, retrieval_hits)
    beat_payload = writer._build_beat_payloads(payload)[0]

    beat_script = writer._write_beat(beat_payload)

    assert llm.calls == 2
    assert beat_script.segments
    assert "The previous response was truncated." in llm.instructions_seen[-1]
    assert "Use at most two segments" in llm.instructions_seen[-1]
    assert "Keep the narration at or above" in llm.instructions_seen[-1]


def test_writing_preserves_plan_order_when_beats_finish_out_of_order() -> None:
    llm = OutOfOrderWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=2)
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    retrieval_hits = _build_retrieval_hits(structure, episode.chunk_ids)

    script = writer.write(episode, retrieval_hits)

    observed = [segment.beat_id for segment in script.segments]
    expected = [beat.beat_id for beat in episode.beats[: len(script.segments)]]
    assert observed == expected


def test_split_into_chapters_handles_front_matter_roman_and_appendix() -> None:
    text = """
    Introduction
    The editor frames the expedition.

    Chapter I
    Arrival
    The ship reaches the harbor.

    Chapter II: Signals
    The archive reveals the mirror network.

    Appendix A
    Notes and source references.
    """

    sections = split_into_chapters(text)

    assert [section.title for section in sections] == [
        "Introduction: The editor frames the expedition.",
        "Chapter I: Arrival",
        "Chapter II: Signals",
        "Appendix A: Notes and source references.",
    ]


def test_split_into_chapters_falls_back_when_no_headings_exist() -> None:
    text = "\n\n".join(
        [
            ("astronomy " * 700).strip(),
            ("observatory " * 700).strip(),
            ("navigation " * 700).strip(),
        ]
    )

    sections = split_into_chapters(text)

    assert len(sections) == 2
    assert sections[0].title == "Section 1"
    assert sections[1].title == "Section 2"


def test_split_into_chapters_rejects_toc_and_note_like_chapter_lines() -> None:
    text = """
    26. Rights 27. Riots 28. Rulers 29. Riches 30. A People's Entertainments Epilogue: Why India Survives Acknowledgements Notes Index

    Prologue
    Opening body text.

    Chapter 2: The granting of maintenance to the wife who chose to live separately from the husband if he had a loathsome disease.
    This should stay inside the prior body rather than start a chapter.

    Chapter 1989: In November of that year Rajiv Gandhi was replaced as prime minister.
    This also should not become a chapter.

    Epilogue
    Closing body text.
    """

    sections = split_into_chapters(text)

    assert sections[0].title == "Section 1"
    assert all("Chapter 1989" not in section.title for section in sections)
    assert all("The granting of maintenance" not in section.title for section in sections)


def test_split_into_chapters_falls_back_when_headings_are_too_sparse() -> None:
    text = "\n\n".join(
        [
            "Prologue",
            ("opening " * 2000).strip(),
            "Epilogue",
            ("closing " * 100).strip(),
            "Notes",
            ("citation " * 500).strip(),
        ]
    )

    sections = split_into_chapters(text)

    assert len(sections) >= 2
    assert sections[0].title == "Section 1"
    assert all("citation" not in section.body for section in sections)


def test_structuring_can_start_from_matching_chapter_title() -> None:
    structure = StructuringAgent(HeuristicLLMClient()).structure(
        _build_ingestion(),
        start_chapter="chapter 3: fracture",
    )

    assert [chapter.title for chapter in structure.chapters] == [
        "Chapter 3: Fracture",
        "Chapter 4: Legacy",
    ]
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2]


def test_structuring_can_end_at_matching_chapter_title() -> None:
    structure = StructuringAgent(HeuristicLLMClient()).structure(
        _build_ingestion(),
        end_chapter="chapter 2: signals",
    )

    assert [chapter.title for chapter in structure.chapters] == [
        "Chapter 1: Arrival",
        "Chapter 2: Signals",
    ]
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2]


def test_structuring_can_apply_inclusive_chapter_range() -> None:
    structure = StructuringAgent(HeuristicLLMClient()).structure(
        _build_ingestion(),
        start_chapter="chapter 2: signals",
        end_chapter="chapter 3: fracture",
    )

    assert [chapter.title for chapter in structure.chapters] == [
        "Chapter 2: Signals",
        "Chapter 3: Fracture",
    ]
    assert [chapter.chapter_number for chapter in structure.chapters] == [1, 2]


def test_structuring_raises_when_start_chapter_is_missing() -> None:
    agent = StructuringAgent(HeuristicLLMClient())

    with pytest.raises(ValueError, match="Unable to find start chapter"):
        agent.structure(_build_ingestion(), start_chapter="Chapter 9: Missing")


def test_structuring_raises_when_end_chapter_is_missing() -> None:
    agent = StructuringAgent(HeuristicLLMClient())

    with pytest.raises(ValueError, match="Unable to find end chapter"):
        agent.structure(_build_ingestion(), end_chapter="Chapter 9: Missing")


def test_structuring_raises_when_end_precedes_start() -> None:
    agent = StructuringAgent(HeuristicLLMClient())

    with pytest.raises(ValueError, match="appears before start chapter"):
        agent.structure(
            _build_ingestion(),
            start_chapter="Chapter 3: Fracture",
            end_chapter="Chapter 2: Signals",
        )


def test_structuring_skips_ocr_normalization_for_text_sources(monkeypatch) -> None:
    called = False

    def fail_if_called(text: str) -> str:
        nonlocal called
        called = True
        raise AssertionError("text sources should not be OCR-normalized")

    monkeypatch.setattr("podcast_agent.agents.structuring.normalize_source_text", fail_if_called)

    structure = StructuringAgent(HeuristicLLMClient()).structure(_build_ingestion())

    assert structure.chapters
    assert called is False


def test_structuring_applies_ocr_normalization_for_pdf_sources(monkeypatch) -> None:
    called = False

    def fake_normalize(text: str) -> str:
        nonlocal called
        called = True
        return text

    monkeypatch.setattr("podcast_agent.agents.structuring.normalize_source_text", fake_normalize)
    ingestion = _build_ingestion().model_copy(update={"source_type": SourceType.PDF})

    structure = StructuringAgent(HeuristicLLMClient()).structure(ingestion)

    assert structure.chapters
    assert called is True


def test_analysis_agent_creates_multi_chapter_clusters() -> None:
    structure = _build_structure()
    analysis = AnalysisAgent(HeuristicLLMClient()).analyze(structure, episode_count=2)

    assert analysis.themes
    assert analysis.episode_clusters
    assert any(len(cluster.chapter_ids) > 1 for cluster in analysis.episode_clusters)


def test_analysis_payload_uses_compact_chunk_summaries() -> None:
    structure = _build_structure()

    payload = AnalysisAgent(HeuristicLLMClient()).build_payload(structure, episode_count=2)

    assert "chunks" not in payload["structure"]
    assert payload["episode_count"] == 2
    assert payload["structure"]["chapters"][0]["sections"]
    assert "excerpt" not in payload["structure"]["chapters"][0]["sections"][0]
    assert "chunk_ids" not in payload["structure"]["chapters"][0]


def test_analysis_normalization_derives_chunk_ids_from_chapter_ids() -> None:
    structure = _build_structure()
    chapter_ids = [chapter.chapter_id for chapter in structure.chapters[:2]]
    analysis = BookAnalysis.model_validate(
        {
            "book_id": structure.book_id,
            "themes": ["observatory", "memory"],
            "continuity_arcs": [],
            "notable_claims": ["The archive connects the chapters."],
            "episode_clusters": [
                {
                    "cluster_id": "cluster-1",
                    "label": "Joined chapters",
                    "rationale": "Keep the first two chapters together.",
                    "chapter_ids": chapter_ids,
                    "chunk_ids": [],
                    "themes": [],
                }
            ],
        }
    )

    normalized = AnalysisAgent(HeuristicLLMClient())._normalize_analysis(analysis, structure)

    assert normalized.episode_clusters[0].chapter_ids == chapter_ids
    assert normalized.episode_clusters[0].chunk_ids == [
        chunk_id for chapter in structure.chapters[:2] for chunk_id in chapter.chunk_ids
    ]
    assert normalized.episode_clusters[0].themes


def test_episode_planning_agent_creates_hierarchical_plan() -> None:
    structure, analysis, plan = _build_analysis_and_plan()

    assert plan.book_id == structure.book_id
    assert len(plan.episodes) == 2
    assert all(episode.beats for episode in plan.episodes)


def test_episode_planning_uses_configured_min_episode_source_ratio() -> None:
    def words(label: str, count: int) -> str:
        return ((label + " ") * count).strip()

    chapter_sizes = [47500, 47500, 25000]
    chapters = []
    chunks = []
    clusters = []
    for index, chapter_word_count in enumerate(chapter_sizes, start=1):
        chapter_id = f"ratio-book-chapter-{index}"
        chunk_id = f"{chapter_id}-chunk-1"
        title = f"Chapter {index}"
        chapters.append(
            BookChapter(
                chapter_id=chapter_id,
                chapter_number=index,
                title=title,
                summary=f"Summary {index}",
                chunk_ids=[chunk_id],
            )
        )
        chunks.append(
            BookChunk(
                chunk_id=chunk_id,
                chapter_id=chapter_id,
                chapter_title=title,
                chapter_number=index,
                sequence=1,
                text=words(f"chapter{index}", chapter_word_count),
                start_word=0,
                end_word=chapter_word_count,
                source_offsets=[0, chapter_word_count],
                themes=[f"theme-{index}"],
            )
        )
        clusters.append(
            EpisodeCluster(
                cluster_id=f"cluster-{index}",
                label=f"Cluster {index}",
                rationale=f"Part {index}",
                chapter_ids=[chapter_id],
                chunk_ids=[chunk_id],
                themes=[f"theme-{index}"],
            )
        )

    structure = BookStructure(book_id="ratio-book", title="Ratio Book", chapters=chapters, chunks=chunks)
    analysis = BookAnalysis(
        book_id="ratio-book",
        themes=["theme-1"],
        continuity_arcs=[],
        notable_claims=[],
        episode_clusters=clusters,
    )
    plan = SeriesPlan.model_validate(
        {
            "book_id": "ratio-book",
            "format": "single_narrator",
            "strategy_summary": "Summary",
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "sequence": 1,
                    "title": "Episode 1",
                    "synopsis": "Part 1",
                    "chapter_ids": ["ratio-book-chapter-1", "ratio-book-chapter-2"],
                    "chunk_ids": ["ratio-book-chapter-1-chunk-1", "ratio-book-chapter-2-chunk-1"],
                    "themes": ["theme-1", "theme-2"],
                    "beats": [
                        {
                            "beat_id": "episode-1-beat-1",
                            "title": "Beat 1",
                            "objective": "Objective",
                            "chunk_ids": ["ratio-book-chapter-1-chunk-1", "ratio-book-chapter-2-chunk-1"],
                            "claim_requirements": [],
                        }
                    ],
                },
                {
                    "episode_id": "episode-2",
                    "sequence": 2,
                    "title": "Episode 2",
                    "synopsis": "Part 2",
                    "chapter_ids": ["ratio-book-chapter-3"],
                    "chunk_ids": ["ratio-book-chapter-3-chunk-1"],
                    "themes": ["theme-3"],
                    "beats": [
                        {
                            "beat_id": "episode-2-beat-1",
                            "title": "Beat 2",
                            "objective": "Objective",
                            "chunk_ids": ["ratio-book-chapter-3-chunk-1"],
                            "claim_requirements": [],
                        }
                    ],
                },
            ],
        }
    )

    planner = EpisodePlanningAgent(
        HeuristicLLMClient(),
        min_episode_source_ratio=0.3,
        max_episode_minutes=1000,
    )
    strict_planner = EpisodePlanningAgent(
        HeuristicLLMClient(),
        min_episode_source_ratio=0.5,
        max_episode_minutes=1000,
    )

    assert planner._target_source_words_per_episode(structure, episode_count=2) == 60000
    assert planner._compliance_violations(plan, structure, analysis, episode_count=2) == []
    assert strict_planner._compliance_violations(plan, structure, analysis, episode_count=2) == [
        "episode-2 estimated at 25000 source words, too small for target 60000"
    ]


def test_analysis_and_planning_reject_episode_count_above_chapter_count() -> None:
    structure = _build_structure()

    try:
        AnalysisAgent(HeuristicLLMClient()).analyze(structure, episode_count=10)
    except RuntimeError as exc:
        assert "Requested 10 episodes" in str(exc)
    else:
        raise AssertionError("Expected analysis to reject impossible episode counts")


def test_writing_agent_creates_cited_episode_script() -> None:
    llm = HeuristicLLMClient()
    structure, _, plan = _build_analysis_and_plan()
    episode = plan.episodes[0]
    retrieval_hits = _build_retrieval_hits(structure, episode.chunk_ids)

    script = WritingAgent(llm).write(episode, retrieval_hits)

    assert script.segments
    assert all(segment.citations for segment in script.segments)
    assert any(segment.claims for segment in script.segments)


def test_grounding_validation_agent_returns_claim_assessments() -> None:
    script, _, retrieval_hits = _build_script_and_report()
    report = GroundingValidationAgent(HeuristicLLMClient()).validate(script, retrieval_hits)

    assert report.overall_status in {"pass", "fail"}
    assert report.claim_assessments
    assert all(assessment.claim_id for assessment in report.claim_assessments)


def test_grounding_validation_agent_scopes_payloads_to_segment_citations() -> None:
    script, _, retrieval_hits = _build_script_and_report()
    llm = SegmentRecordingValidationLLM()

    report = GroundingValidationAgent(llm, grounding_parallelism=1).validate(script, retrieval_hits)

    assert len(llm.payloads) == len(script.segments)
    assert len(report.claim_assessments) == sum(len(segment.claims) for segment in script.segments)
    for payload, segment in zip(llm.payloads, script.segments, strict=True):
        assert len(payload["script"]["segments"]) == 1
        assert payload["script"]["segments"][0]["segment_id"] == segment.segment_id
        assert sorted(hit["chunk_id"] for hit in payload["retrieval_hits"]) == sorted(set(segment.citations))


def test_grounding_validation_agent_preserves_claim_order_across_segments() -> None:
    script, _, retrieval_hits = _build_script_and_report()
    llm = SegmentRecordingValidationLLM()

    report = GroundingValidationAgent(llm, grounding_parallelism=3).validate(script, retrieval_hits)

    expected_claim_ids = [claim.claim_id for segment in script.segments for claim in segment.claims]
    assert [assessment.claim_id for assessment in report.claim_assessments] == expected_claim_ids


def test_grounding_validation_agent_uses_configured_parallelism() -> None:
    script, _, retrieval_hits = _build_script_and_report()
    llm = ParallelValidationLLM()

    report = GroundingValidationAgent(llm, grounding_parallelism=3).validate(script, retrieval_hits)

    assert report.overall_status == "pass"
    assert llm.max_active == min(3, len(script.segments))


def test_repair_agent_targets_only_failed_segments() -> None:
    llm = CapturingRepairLLM()
    script, _, retrieval_hits = _build_script_and_report()

    weakened_first_segment = script.segments[0].model_copy(
        update={
            "claims": [
                script.segments[0].claims[0].model_copy(
                    update={"text": "This claim now describes an unrelated volcanic eruption in a distant empire."}
                )
            ]
        }
    )
    weakened_script = EpisodeScript.model_construct(
        **{
            **script.model_dump(mode="python"),
            "segments": [
                weakened_first_segment,
                *script.segments[1:],
            ],
        }
    )
    weakened_report = GroundingValidationAgent(llm).validate(weakened_script, retrieval_hits)

    repair = RepairAgent(llm).repair(weakened_script, weakened_report, attempt=1)

    assert repair.attempt == 1
    assert repair.repaired_segment_ids
    assert repair.repaired_segment_ids[0] == weakened_script.segments[0].segment_id
    payload = llm.payloads[0]
    assert payload["episode_id"] == weakened_script.episode_id
    assert len(payload["failed_segments"]) == 1
    assert payload["failed_segments"][0]["segment_id"] == weakened_script.segments[0].segment_id
    assert payload["failed_segments"][0]["beat_id"] == weakened_script.segments[0].beat_id
    assert {assessment["claim_id"] for assessment in payload["report"]["claim_assessments"]} == {
        weakened_script.segments[0].claims[0].claim_id
    }
