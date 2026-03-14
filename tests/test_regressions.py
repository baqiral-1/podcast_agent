"""Regression tests for reviewed runtime issues."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_agent.cli.app import _build_orchestrator
from podcast_agent.agents import EpisodePlanningAgent
from podcast_agent.config import PipelineConfig, Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.db.repository import _format_pgvector
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.llm.base import LLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
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


def test_planning_allows_multi_episode_books_when_total_words_cannot_fill_all_targets() -> None:
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

    assert planner._compliance_violations(plan, structure, analysis) == []


def test_planning_rebalances_short_episode_above_standalone_minimum() -> None:
    def words(label: str, count: int) -> str:
        return ((label + " ") * count).strip()

    chapter_counts = [1325, 1313, 1328, 1365, 1458, 1426, 1321, 1335, 1337, 1355, 1415, 1435]
    chapters = []
    chunks = []
    clusters = []
    for index, word_count in enumerate(chapter_counts, start=1):
        chapter_id = f"river-of-hours-chapter-{index}"
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
                text=words(f"chapter{index}", word_count),
                start_word=0,
                end_word=word_count,
                source_offsets=[0, word_count],
                themes=[f"theme-{index}"],
            )
        )
    structure = BookStructure(book_id="river-of-hours", title="River of Hours", chapters=chapters, chunks=chunks)
    clusters = [
        EpisodeCluster(
            cluster_id="cluster-1",
            label="Cluster 1",
            rationale="Chapters 1-3",
            chapter_ids=[chapters[0].chapter_id, chapters[1].chapter_id, chapters[2].chapter_id],
            chunk_ids=[chapters[0].chunk_ids[0], chapters[1].chunk_ids[0], chapters[2].chunk_ids[0]],
            themes=["theme-1"],
        ),
        EpisodeCluster(
            cluster_id="cluster-2",
            label="Cluster 2",
            rationale="Chapters 4-6",
            chapter_ids=[chapters[3].chapter_id, chapters[4].chapter_id, chapters[5].chapter_id],
            chunk_ids=[chapters[3].chunk_ids[0], chapters[4].chunk_ids[0], chapters[5].chunk_ids[0]],
            themes=["theme-4"],
        ),
        EpisodeCluster(
            cluster_id="cluster-3",
            label="Cluster 3",
            rationale="Chapters 7-8",
            chapter_ids=[chapters[6].chapter_id, chapters[7].chapter_id],
            chunk_ids=[chapters[6].chunk_ids[0], chapters[7].chunk_ids[0]],
            themes=["theme-7"],
        ),
        EpisodeCluster(
            cluster_id="cluster-4",
            label="Cluster 4",
            rationale="Chapters 9-12",
            chapter_ids=[chapters[8].chapter_id, chapters[9].chapter_id, chapters[10].chapter_id, chapters[11].chapter_id],
            chunk_ids=[chapters[8].chunk_ids[0], chapters[9].chunk_ids[0], chapters[10].chunk_ids[0], chapters[11].chunk_ids[0]],
            themes=["theme-9"],
        ),
    ]
    analysis = BookAnalysis(
        book_id="river-of-hours",
        themes=["identity"],
        continuity_arcs=[],
        notable_claims=[],
        episode_clusters=clusters,
    )
    plan = SeriesPlan(
        book_id="river-of-hours",
        format="single_narrator",
        strategy_summary="Initial plan.",
        episodes=[
            EpisodePlan(
                episode_id="episode-1",
                sequence=1,
                title="Episode 1",
                synopsis="Chapters 1-3",
                chapter_ids=[chapters[0].chapter_id, chapters[1].chapter_id, chapters[2].chapter_id],
                chunk_ids=[chapters[0].chunk_ids[0], chapters[1].chunk_ids[0], chapters[2].chunk_ids[0]],
                themes=["theme-1"],
                beats=[EpisodeBeat(beat_id="beat-1", title="Beat 1", objective="Objective", chunk_ids=[chapters[0].chunk_ids[0]], claim_requirements=[])],
            ),
            EpisodePlan(
                episode_id="episode-2",
                sequence=2,
                title="Episode 2",
                synopsis="Chapters 4-6",
                chapter_ids=[chapters[3].chapter_id, chapters[4].chapter_id, chapters[5].chapter_id],
                chunk_ids=[chapters[3].chunk_ids[0], chapters[4].chunk_ids[0], chapters[5].chunk_ids[0]],
                themes=["theme-4"],
                beats=[EpisodeBeat(beat_id="beat-2", title="Beat 2", objective="Objective", chunk_ids=[chapters[3].chunk_ids[0]], claim_requirements=[])],
            ),
            EpisodePlan(
                episode_id="episode-3",
                sequence=3,
                title="Episode 3",
                synopsis="Chapters 7-8",
                chapter_ids=[chapters[6].chapter_id, chapters[7].chapter_id],
                chunk_ids=[chapters[6].chunk_ids[0], chapters[7].chunk_ids[0]],
                themes=["theme-7"],
                beats=[EpisodeBeat(beat_id="beat-3", title="Beat 3", objective="Objective", chunk_ids=[chapters[6].chunk_ids[0]], claim_requirements=[])],
            ),
            EpisodePlan(
                episode_id="episode-4",
                sequence=4,
                title="Episode 4",
                synopsis="Chapters 9-12",
                chapter_ids=[chapters[8].chapter_id, chapters[9].chapter_id, chapters[10].chapter_id, chapters[11].chapter_id],
                chunk_ids=[chapters[8].chunk_ids[0], chapters[9].chunk_ids[0], chapters[10].chunk_ids[0], chapters[11].chunk_ids[0]],
                themes=["theme-9"],
                beats=[EpisodeBeat(beat_id="beat-4", title="Beat 4", objective="Objective", chunk_ids=[chapters[8].chunk_ids[0]], claim_requirements=[])],
            ),
        ],
    )
    planner = EpisodePlanningAgent(HeuristicLLMClient())

    normalized = planner._normalize_plan(plan, structure)

    assert [episode.chapter_ids for episode in normalized.episodes] == [
        [chapters[0].chapter_id, chapters[1].chapter_id, chapters[2].chapter_id],
        [chapters[3].chapter_id, chapters[4].chapter_id, chapters[5].chapter_id],
        [chapters[6].chapter_id, chapters[7].chapter_id, chapters[8].chapter_id],
        [chapters[9].chapter_id, chapters[10].chapter_id, chapters[11].chapter_id],
    ]
    assert planner._compliance_violations(normalized, structure, analysis) == []


def test_planning_merges_episode_only_below_ten_minutes() -> None:
    def words(label: str, count: int) -> str:
        return ((label + " ") * count).strip()

    chapters = [
        BookChapter(chapter_id="book-chapter-1", chapter_number=1, title="Chapter 1", summary="A", chunk_ids=["c1"]),
        BookChapter(chapter_id="book-chapter-2", chapter_number=2, title="Chapter 2", summary="B", chunk_ids=["c2"]),
        BookChapter(chapter_id="book-chapter-3", chapter_number=3, title="Chapter 3", summary="C", chunk_ids=["c3"]),
    ]
    chunks = [
        BookChunk(chunk_id="c1", chapter_id="book-chapter-1", chapter_title="Chapter 1", chapter_number=1, sequence=1, text=words("alpha", 2200), start_word=0, end_word=2200, source_offsets=[0, 2200], themes=["alpha"]),
        BookChunk(chunk_id="c2", chapter_id="book-chapter-2", chapter_title="Chapter 2", chapter_number=2, sequence=1, text=words("beta", 2100), start_word=0, end_word=2100, source_offsets=[0, 2100], themes=["beta"]),
        BookChunk(chunk_id="c3", chapter_id="book-chapter-3", chapter_title="Chapter 3", chapter_number=3, sequence=1, text=words("gamma", 900), start_word=0, end_word=900, source_offsets=[0, 900], themes=["gamma"]),
    ]
    structure = BookStructure(book_id="short-book", title="Short Book", chapters=chapters, chunks=chunks)
    analysis = BookAnalysis(
        book_id="short-book",
        themes=["alpha"],
        continuity_arcs=[],
        notable_claims=[],
        episode_clusters=[
            EpisodeCluster(cluster_id="cluster-1", label="Cluster 1", rationale="Part 1", chapter_ids=["book-chapter-1"], chunk_ids=["c1"], themes=["alpha"]),
            EpisodeCluster(cluster_id="cluster-2", label="Cluster 2", rationale="Part 2", chapter_ids=["book-chapter-2"], chunk_ids=["c2"], themes=["beta"]),
            EpisodeCluster(cluster_id="cluster-3", label="Cluster 3", rationale="Part 3", chapter_ids=["book-chapter-3"], chunk_ids=["c3"], themes=["gamma"]),
        ],
    )
    plan = SeriesPlan(
        book_id="short-book",
        format="single_narrator",
        strategy_summary="Initial plan.",
        episodes=[
            EpisodePlan(episode_id="episode-1", sequence=1, title="Episode 1", synopsis="Part 1", chapter_ids=["book-chapter-1"], chunk_ids=["c1"], themes=["alpha"], beats=[EpisodeBeat(beat_id="beat-1", title="Beat 1", objective="Objective", chunk_ids=["c1"], claim_requirements=[])]),
            EpisodePlan(episode_id="episode-2", sequence=2, title="Episode 2", synopsis="Part 2", chapter_ids=["book-chapter-2"], chunk_ids=["c2"], themes=["beta"], beats=[EpisodeBeat(beat_id="beat-2", title="Beat 2", objective="Objective", chunk_ids=["c2"], claim_requirements=[])]),
            EpisodePlan(episode_id="episode-3", sequence=3, title="Episode 3", synopsis="Part 3", chapter_ids=["book-chapter-3"], chunk_ids=["c3"], themes=["gamma"], beats=[EpisodeBeat(beat_id="beat-3", title="Beat 3", objective="Objective", chunk_ids=["c3"], claim_requirements=[])]),
        ],
    )
    planner = EpisodePlanningAgent(HeuristicLLMClient())

    normalized = planner._normalize_plan(plan, structure)

    assert len(normalized.episodes) == 2
    assert [episode.chapter_ids for episode in normalized.episodes] == [
        ["book-chapter-1"],
        ["book-chapter-2", "book-chapter-3"],
    ]
    assert planner._compliance_violations(normalized, structure, analysis) == []
