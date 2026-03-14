"""Regression tests for reviewed runtime issues."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from podcast_agent.cli.app import _build_orchestrator
from podcast_agent.agents import EpisodePlanningAgent, WritingAgent
from podcast_agent.config import PipelineConfig, Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.db.repository import _format_pgvector
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.openai_compatible import HTTPTransport, OpenAICompatibleLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    BookAnalysis,
    BookChapter,
    BookChunk,
    BookStructure,
    ContinuityArc,
    EpisodeBeat,
    EpisodeCluster,
    EpisodePlan,
    GroundingStatus,
    RetrievalHit,
    SeriesPlan,
)


BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory.

Chapter 2: Signals
Inside the city archive, the team finds coded letters describing the observatory's mirrors and the way they coordinated messages across valleys.
"""

ANALYSIS_BOOK_TEXT = """
Chapter 1: Arrival
The expedition arrives at the northern harbor after months at sea. The crew studies old maps and notices repeated references to a hidden observatory. The historian argues that the harbor served as a relay for scholars, not soldiers.

Chapter 2: Signals
Inside the city archive, the team finds coded letters describing the observatory's mirrors and the way they coordinated messages across valleys. The engineer explains that the mirror network only worked because local guilds kept the routes maintained.

Chapter 3: Fracture
The discovery creates tension among the sponsors, who want the observatory framed as a military outpost. The historian pushes back and points to logs showing that teachers, navigators, and astronomers shared the site.

Chapter 4: Legacy
By the end of the journey, the team understands that the observatory connected communities through knowledge exchange. The final journal entries argue that its legacy was cooperation, memory, and scientific practice.
"""


class FailingValidationLLM(LLMClient):
    """LLM stub that never produces a passing grounding report."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "grounding_report":
            script = payload["script"]
            assessments = []
            for segment in script["segments"]:
                for claim in segment["claims"]:
                    assessments.append(
                        {
                            "claim_id": claim["claim_id"],
                            "status": GroundingStatus.UNSUPPORTED,
                            "reason": "Synthetic regression failure.",
                            "evidence_chunk_ids": [],
                        }
                    )
            return response_model.model_validate(
                {
                    "episode_id": script["episode_id"],
                    "overall_status": "fail",
                    "claim_assessments": assessments,
                }
            )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


class FlakyPlanningLLM(LLMClient):
    """LLM stub that returns one invalid series plan before recovering."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()
        self.series_plan_calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "series_plan":
            self.series_plan_calls += 1
            if self.series_plan_calls == 1:
                structure = payload["structure"]
                chapter = structure["chapters"][0]
                return response_model.model_validate(
                    {
                        "book_id": structure["book_id"],
                        "format": "single_narrator",
                        "strategy_summary": "Bad initial plan.",
                        "episodes": [
                            {
                                "episode_id": "episode-1",
                                "sequence": 1,
                                "title": chapter["title"],
                                "synopsis": "Too short and chapter-scoped.",
                                "chapter_ids": [chapter["chapter_id"]],
                                "chunk_ids": chapter["chunk_ids"][:1],
                                "themes": ["test"],
                                "beats": [
                                    {
                                        "beat_id": "beat-1",
                                        "title": "Beat 1",
                                        "objective": "Test objective.",
                                        "chunk_ids": chapter["chunk_ids"][:1],
                                        "claim_requirements": ["Test claim."],
                                    }
                                ],
                            }
                        ],
                    }
                )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


class FlakyAnalysisLLM(LLMClient):
    """LLM stub that returns one invalid book analysis before recovering."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()
        self.analysis_calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "book_analysis":
            self.analysis_calls += 1
            if self.analysis_calls == 1:
                structure = payload["structure"]
                return response_model.model_validate(
                    {
                        "book_id": structure["book_id"],
                        "themes": ["identity", "memory"],
                        "continuity_arcs": [
                            {
                                "arc_id": "arc-1",
                                "label": "Sparse arc",
                                "description": "Bad sparse arc.",
                                "chapter_ids": [
                                    structure["chapters"][1]["chapter_id"],
                                    structure["chapters"][3]["chapter_id"],
                                ],
                            }
                        ],
                        "notable_claims": ["Sparse claim."],
                        "episode_clusters": [
                            {
                                "cluster_id": "cluster-1",
                                "label": "Sparse cluster",
                                "rationale": "Bad sparse cluster.",
                                "chapter_ids": [
                                    structure["chapters"][1]["chapter_id"],
                                    structure["chapters"][3]["chapter_id"],
                                ],
                                "chunk_ids": [structure["chapters"][1]["chunk_ids"][0]],
                                "themes": ["identity"],
                            }
                        ],
                    }
                )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


class FlakyWritingLLM(LLMClient):
    """LLM stub that returns one summary-length script before recovering."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()
        self.write_calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.write_calls += 1
            if self.write_calls == 1:
                beat = payload["beat"]
                first_chunk = payload["retrieval_hits"][0]["chunk_id"]
                return response_model.model_validate(
                    {
                        "beat_id": beat["beat_id"],
                        "segments": [
                            {
                                "segment_id": f"{beat['beat_id']}-segment-1",
                                "beat_id": beat["beat_id"],
                                "heading": beat["title"],
                                "narration": "Short summary.",
                                "claims": [
                                    {
                                        "claim_id": "claim-1",
                                        "text": "Short claim.",
                                        "evidence_chunk_ids": [first_chunk],
                                    }
                                ],
                                "citations": [first_chunk],
                            }
                        ],
                    }
                )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


class InspectingWritingLLM(LLMClient):
    """LLM stub that records beat payload details."""

    def __init__(self) -> None:
        from podcast_agent.llm import HeuristicLLMClient

        self.delegate = HeuristicLLMClient()
        self.beat_calls: list[dict] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "beat_script":
            self.beat_calls.append(
                {
                    "beat_id": payload["beat"]["beat_id"],
                    "chunk_count": len(payload["retrieval_hits"]),
                    "chunk_ids": [hit["chunk_id"] for hit in payload["retrieval_hits"]],
                }
            )
        return self.delegate.generate_json(schema_name, instructions, payload, response_model)


class FailingStructuringLLM(LLMClient):
    """LLM stub that fails during chapter structuring."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        if schema_name == "structured_chapter":
            chapter_number = payload["draft"]["chapter_number"]
            if chapter_number == 2:
                raise TimeoutError("simulated chapter timeout")
        return HeuristicLLMClient().generate_json(schema_name, instructions, payload, response_model)


class TimeoutTransport(HTTPTransport):
    """Transport stub that raises a raw timeout-like exception."""

    def post_json(self, url, headers, payload, timeout_seconds):
        raise TimeoutError("simulated transport timeout")


def test_build_orchestrator_uses_in_memory_when_database_url_missing(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    orchestrator = _build_orchestrator(None)

    assert isinstance(orchestrator.repository, InMemoryRepository)
    assert Settings().database.dsn is None


def test_pgvector_serialization_uses_vector_literal() -> None:
    assert _format_pgvector([0.1, -0.2, 0.0]) == "[0.1,-0.2,0.0]"


def test_failed_grounding_skips_render_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
    )

    result = orchestrator.run_pipeline(book_path, title="Broken Grounding", author="A. Writer")

    assert result["episodes"]
    assert result["episodes"][0]["report"]["overall_status"] == "fail"
    assert result["episodes"][0]["manifest"] is None


def test_each_run_uses_book_title_and_minute_timestamp_in_artifact_subdirectory(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(
        pipeline=PipelineConfig(artifact_root=tmp_path / "runs"),
    )

    first = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
        settings=settings,
    )
    second = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingValidationLLM(),
        settings=settings,
    )

    first.run_pipeline(book_path, title="Run One", author="A. Writer")
    second.run_pipeline(book_path, title="Run Two", author="A. Writer")

    run_dirs = sorted(path.name for path in (tmp_path / "runs").iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert run_dirs[0].startswith("run-one-")
    assert run_dirs[1].startswith("run-two-")


def test_run_log_captures_llm_prompts_and_artifact_writes(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    orchestrator.run_pipeline(book_path, title="Logged Book", author="A. Writer")

    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]

    assert any(line["event_type"] == "llm_request" for line in lines)
    assert any(line["event_type"] == "artifact_written" for line in lines)
    assert any(line["event_type"] == "stage_start" for line in lines)


def test_incremental_structuring_logs_one_request_per_chapter(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Chapter Logged Book", author="A. Writer")
    orchestrator.index_book(ingestion)

    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    chapter_requests = [
        line for line in lines if line["event_type"] == "llm_request" and line["payload"]["schema_name"] == "structured_chapter"
    ]

    assert len(chapter_requests) == 2


def test_structuring_logs_failed_chapter_before_aborting(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=FailingStructuringLLM(),
        settings=settings,
    )

    ingestion = orchestrator.ingest_book(book_path, title="Broken Structuring", author="A. Writer")

    try:
        orchestrator.index_book(ingestion)
    except RuntimeError as exc:
        assert "Structuring failed for chapter 2" in str(exc)
    else:
        raise AssertionError("Expected chapter structuring to fail")

    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    failures = [line for line in lines if line["event_type"] == "structuring_chapter_failed"]

    assert failures
    assert failures[0]["payload"]["chapter_number"] == 2
    assert failures[0]["payload"]["error_type"] == "TimeoutError"


def test_openai_client_logs_llm_error_for_timeout(tmp_path: Path) -> None:
    settings = Settings(
        llm=Settings().llm.model_copy(update={"api_key": "test-key"}),
        pipeline=PipelineConfig(artifact_root=tmp_path / "runs"),
    )
    run_logger = RunLogger(tmp_path / "runs")
    run_logger.bind_run("test-run")
    client = OpenAICompatibleLLMClient(settings.llm, transport=TimeoutTransport())
    client.set_run_logger(run_logger)

    try:
        client.generate_json(
            schema_name="structured_chapter",
            instructions="Normalize this chapter into canonical chunks.",
            payload={"draft": {"chapter_number": 1, "title": "Chapter 1", "summary": "S", "chunks": []}},
            response_model=BookAnalysis,
        )
    except TimeoutError as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("Expected timeout to be raised")

    run_log = tmp_path / "runs" / "test-run" / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    errors = [line for line in lines if line["event_type"] == "llm_error"]

    assert errors
    assert errors[0]["payload"]["schema_name"] == "structured_chapter"
    assert errors[0]["payload"]["error_type"] == "TimeoutError"


def test_run_logger_remains_parseable_under_concurrent_writes(tmp_path: Path) -> None:
    run_logger = RunLogger(tmp_path / "runs")
    run_logger.bind_run("threaded-run")

    def emit(thread_id: int) -> None:
        for index in range(100):
            run_logger.log("thread_event", thread_id=thread_id, sequence=index)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for thread_id in range(8):
            executor.submit(emit, thread_id)

    run_log = tmp_path / "runs" / "threaded-run" / "run.log"
    lines = run_log.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]

    assert len(parsed) == 800
    assert all(line["event_type"] == "thread_event" for line in parsed)


def test_run_logger_preserves_buffered_and_concurrent_events(tmp_path: Path) -> None:
    run_logger = RunLogger(tmp_path / "runs")
    for index in range(10):
        run_logger.log("buffered_event", sequence=index)

    run_logger.bind_run("buffered-threaded-run")

    def emit(thread_id: int) -> None:
        for index in range(50):
            run_logger.log("thread_event", thread_id=thread_id, sequence=index)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for thread_id in range(4):
            executor.submit(emit, thread_id)

    run_log = tmp_path / "runs" / "buffered-threaded-run" / "run.log"
    lines = run_log.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]

    assert len(parsed) == 210
    assert sum(1 for line in parsed if line["event_type"] == "buffered_event") == 10
    assert sum(1 for line in parsed if line["event_type"] == "thread_event") == 200


def test_planning_retries_after_non_compliant_series_plan(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(
        (
            BOOK_TEXT
            + "\n\nChapter 3: Fracture\n"
            + "The archive reveals more evidence. " * 120
            + "\n\nChapter 4: Legacy\n"
            + "The community preserves shared memory. " * 120
        ),
        encoding="utf-8",
    )
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    llm = FlakyPlanningLLM()
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=llm,
        settings=settings,
    )

    result = orchestrator.run_pipeline(book_path, title="Retry Planned", author="A. Writer")

    assert llm.series_plan_calls == 2
    assert result["series_plan"]["episodes"]
    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    diagnostics = [line for line in lines if line["event_type"] == "planning_diagnostics"]
    assert diagnostics
    assert diagnostics[0]["payload"]["violations"]


def test_analysis_retries_after_sparse_cluster_output(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(ANALYSIS_BOOK_TEXT, encoding="utf-8")
    settings = Settings(pipeline=PipelineConfig(artifact_root=tmp_path / "runs"))
    llm = FlakyAnalysisLLM()
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=llm,
        settings=settings,
    )

    result = orchestrator.run_pipeline(book_path, title="Retry Analysis", author="A. Writer")

    assert llm.analysis_calls == 2
    assert result["analysis"]["episode_clusters"]
    run_log = tmp_path / "runs" / orchestrator.run_id / "run.log"
    lines = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
    diagnostics = [line for line in lines if line["event_type"] == "analysis_diagnostics"]
    assert diagnostics
    assert diagnostics[0]["payload"]["violations"]
    assert diagnostics[-1]["payload"]["violations"] == []


def test_planning_returns_single_episode_when_book_is_below_source_word_floor() -> None:
    chunk_a = "alpha " * 3000
    chunk_b = "beta " * 3000
    structure = BookStructure(
        book_id="medium-book",
        title="Medium Book",
        chapters=[
            BookChapter(chapter_id="medium-book-chapter-1", chapter_number=1, title="Chapter 1", summary="A", chunk_ids=["c1"]),
            BookChapter(chapter_id="medium-book-chapter-2", chapter_number=2, title="Chapter 2", summary="B", chunk_ids=["c2"]),
        ],
        chunks=[
            BookChunk(
                chunk_id="c1",
                chapter_id="medium-book-chapter-1",
                chapter_title="Chapter 1",
                chapter_number=1,
                sequence=1,
                text=chunk_a.strip(),
                start_word=0,
                end_word=3000,
                source_offsets=[0, len(chunk_a)],
                themes=["alpha"],
            ),
            BookChunk(
                chunk_id="c2",
                chapter_id="medium-book-chapter-2",
                chapter_title="Chapter 2",
                chapter_number=2,
                sequence=1,
                text=chunk_b.strip(),
                start_word=0,
                end_word=3000,
                source_offsets=[0, len(chunk_b)],
                themes=["beta"],
            ),
        ],
    )
    analysis = BookAnalysis(
        book_id="medium-book",
        themes=["alpha", "beta"],
        continuity_arcs=[
            ContinuityArc(
                arc_id="arc-1",
                label="Arc 1",
                description="Links the book.",
                chapter_ids=["medium-book-chapter-1", "medium-book-chapter-2"],
            )
        ],
        notable_claims=["Claim"],
        episode_clusters=[
            EpisodeCluster(
                cluster_id="cluster-1",
                label="Cluster 1",
                rationale="Part 1",
                chapter_ids=["medium-book-chapter-1"],
                chunk_ids=["c1"],
                themes=["alpha"],
            ),
            EpisodeCluster(
                cluster_id="cluster-2",
                label="Cluster 2",
                rationale="Part 2",
                chapter_ids=["medium-book-chapter-2"],
                chunk_ids=["c2"],
                themes=["beta"],
            ),
        ],
    )
    plan = SeriesPlan(
        book_id="medium-book",
        format="single_narrator",
        strategy_summary="Two-part plan.",
        episodes=[
            EpisodePlan(
                episode_id="episode-1",
                sequence=1,
                title="Episode 1",
                synopsis="Part 1",
                chapter_ids=["medium-book-chapter-1"],
                chunk_ids=["c1"],
                themes=["alpha"],
                beats=[
                    EpisodeBeat(
                        beat_id="beat-1",
                        title="Beat 1",
                        objective="Objective",
                        chunk_ids=["c1"],
                        claim_requirements=["Claim"],
                    )
                ],
            ),
            EpisodePlan(
                episode_id="episode-2",
                sequence=2,
                title="Episode 2",
                synopsis="Part 2",
                chapter_ids=["medium-book-chapter-2"],
                chunk_ids=["c2"],
                themes=["beta"],
                beats=[
                    EpisodeBeat(
                        beat_id="beat-2",
                        title="Beat 2",
                        objective="Objective",
                        chunk_ids=["c2"],
                        claim_requirements=["Claim"],
                    )
                ],
            ),
        ],
    )
    planner = EpisodePlanningAgent(HeuristicLLMClient())

    normalized = planner._normalize_plan(plan, structure)

    assert len(normalized.episodes) == 1
    assert normalized.episodes[0].chapter_ids == ["medium-book-chapter-1", "medium-book-chapter-2"]
    assert planner._compliance_violations(normalized, structure, analysis) == []


def test_planning_splits_large_books_only_when_each_episode_meets_source_floor() -> None:
    def words(label: str, count: int) -> str:
        return ((label + " ") * count).strip()

    chapter_word_count = 26000
    chapters = []
    chunks = []
    clusters = []
    for index in range(1, 5):
        chapter_id = f"large-book-chapter-{index}"
        chunk_id = f"{chapter_id}-chunk-1"
        chapters.append(
            BookChapter(
                chapter_id=chapter_id,
                chapter_number=index,
                title=f"Chapter {index}",
                summary=f"Summary {index}",
                chunk_ids=[chunk_id],
            )
        )
        chunks.append(
            BookChunk(
                chunk_id=chunk_id,
                chapter_id=chapter_id,
                chapter_title=f"Chapter {index}",
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
    structure = BookStructure(book_id="large-book", title="Large Book", chapters=chapters, chunks=chunks)
    analysis = BookAnalysis(
        book_id="large-book",
        themes=["theme-1"],
        continuity_arcs=[],
        notable_claims=[],
        episode_clusters=clusters,
    )
    plan = SeriesPlan(
        book_id="large-book",
        format="single_narrator",
        strategy_summary="Four-part draft.",
        episodes=[
            EpisodePlan(
                episode_id=f"episode-{index}",
                sequence=index,
                title=f"Episode {index}",
                synopsis=f"Part {index}",
                chapter_ids=[chapter.chapter_id],
                chunk_ids=[chapter.chunk_ids[0]],
                themes=[f"theme-{index}"],
                beats=[
                    EpisodeBeat(
                        beat_id=f"beat-{index}",
                        title="Beat",
                        objective="Objective",
                        chunk_ids=[chapter.chunk_ids[0]],
                        claim_requirements=[],
                    )
                ],
            )
            for index, chapter in enumerate(chapters, start=1)
        ],
    )
    planner = EpisodePlanningAgent(HeuristicLLMClient())

    normalized = planner._normalize_plan(plan, structure)

    assert len(normalized.episodes) == 2
    assert [episode.chapter_ids for episode in normalized.episodes] == [
        ["large-book-chapter-1", "large-book-chapter-2"],
        ["large-book-chapter-3", "large-book-chapter-4"],
    ]
    assert planner._compliance_violations(normalized, structure, analysis) == []


def test_writing_retries_after_summary_length_script() -> None:
    structure = BookStructure(
        book_id="writer-book",
        title="Writer Book",
        chapters=[
            BookChapter(chapter_id="writer-book-chapter-1", chapter_number=1, title="Chapter 1", summary="A", chunk_ids=["c1", "c2"]),
            BookChapter(chapter_id="writer-book-chapter-2", chapter_number=2, title="Chapter 2", summary="B", chunk_ids=["c3", "c4"]),
        ],
        chunks=[
            BookChunk(chunk_id="c1", chapter_id="writer-book-chapter-1", chapter_title="Chapter 1", chapter_number=1, sequence=1, text=("alpha " * 500).strip(), start_word=0, end_word=500, source_offsets=[0, 500], themes=["alpha"]),
            BookChunk(chunk_id="c2", chapter_id="writer-book-chapter-1", chapter_title="Chapter 1", chapter_number=1, sequence=2, text=("beta " * 500).strip(), start_word=500, end_word=1000, source_offsets=[500, 1000], themes=["beta"]),
            BookChunk(chunk_id="c3", chapter_id="writer-book-chapter-2", chapter_title="Chapter 2", chapter_number=2, sequence=1, text=("gamma " * 500).strip(), start_word=0, end_word=500, source_offsets=[0, 500], themes=["gamma"]),
            BookChunk(chunk_id="c4", chapter_id="writer-book-chapter-2", chapter_title="Chapter 2", chapter_number=2, sequence=2, text=("delta " * 500).strip(), start_word=500, end_word=1000, source_offsets=[500, 1000], themes=["delta"]),
        ],
    )
    episode = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        synopsis="Full episode",
        chapter_ids=["writer-book-chapter-1", "writer-book-chapter-2"],
        chunk_ids=["c1", "c2", "c3", "c4"],
        themes=["alpha"],
        beats=[
            EpisodeBeat(beat_id="beat-1", title="Beat 1", objective="Objective", chunk_ids=["c1", "c2"], claim_requirements=[]),
            EpisodeBeat(beat_id="beat-2", title="Beat 2", objective="Objective", chunk_ids=["c3", "c4"], claim_requirements=[]),
        ],
    )
    retrieval_hits = [
        RetrievalHit(
            chunk_id=chunk.chunk_id,
            chapter_id=chunk.chapter_id,
            chapter_title=chunk.chapter_title,
            score=1.0,
            text=chunk.text,
        )
        for chunk in structure.chunks
    ]
    llm = FlakyWritingLLM()
    writer = WritingAgent(llm, beat_parallelism=1)

    script = writer.write(episode, retrieval_hits)

    assert llm.write_calls == 3
    assert sum(len(segment.narration.split()) for segment in script.segments) >= writer._target_script_words(2000)


def test_writing_runs_per_beat_with_top_six_evidence() -> None:
    retrieval_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{index}",
            chapter_id="chapter-1",
            chapter_title="Chapter 1",
            score=1.0,
            text=("word " * 120).strip(),
        )
        for index in range(1, 9)
    ]
    episode = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        synopsis="Full episode",
        chapter_ids=["chapter-1"],
        chunk_ids=[hit.chunk_id for hit in retrieval_hits],
        themes=["alpha"],
        beats=[
            EpisodeBeat(
                beat_id="beat-1",
                title="Beat 1",
                objective="Objective 1",
                chunk_ids=[hit.chunk_id for hit in retrieval_hits],
                claim_requirements=[],
            ),
            EpisodeBeat(
                beat_id="beat-2",
                title="Beat 2",
                objective="Objective 2",
                chunk_ids=["chunk-3", "chunk-4", "chunk-5"],
                claim_requirements=[],
            ),
        ],
    )
    llm = InspectingWritingLLM()
    writer = WritingAgent(llm)

    script = writer.write(episode, retrieval_hits)

    assert len(llm.beat_calls) == 2
    assert llm.beat_calls[0]["chunk_count"] == 6
    assert llm.beat_calls[0]["chunk_ids"] == [f"chunk-{index}" for index in range(1, 7)]
    assert llm.beat_calls[1]["chunk_count"] == 3
    assert [segment.beat_id for segment in script.segments] == ["beat-1", "beat-2"]
