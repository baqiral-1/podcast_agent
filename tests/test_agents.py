"""Unit tests for individual agent classes."""

from __future__ import annotations

from podcast_agent.agents import (
    AnalysisAgent,
    EpisodePlanningAgent,
    GroundingValidationAgent,
    RepairAgent,
    StructuringAgent,
    WritingAgent,
)
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.schemas.models import (
    BookIngestionResult,
    EpisodeScript,
    GroundingReport,
    RetrievalHit,
    SourceType,
)


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
    analysis = AnalysisAgent(llm).analyze(structure)
    plan = EpisodePlanningAgent(llm).plan(structure, analysis)
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


def test_structuring_agent_creates_chapters_and_chunks() -> None:
    structure = _build_structure()

    assert structure.chapters
    assert structure.chunks
    assert all(chapter.title for chapter in structure.chapters)
    assert all(chunk.chunk_id.startswith("observatory-book-chapter-") for chunk in structure.chunks)


def test_analysis_agent_creates_multi_chapter_clusters() -> None:
    structure = _build_structure()
    analysis = AnalysisAgent(HeuristicLLMClient()).analyze(structure)

    assert analysis.themes
    assert analysis.episode_clusters
    assert any(len(cluster.chapter_ids) > 1 for cluster in analysis.episode_clusters)


def test_episode_planning_agent_creates_hierarchical_plan() -> None:
    structure, analysis, plan = _build_analysis_and_plan()

    assert plan.book_id == structure.book_id
    assert plan.episodes
    assert all(episode.beats for episode in plan.episodes)


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


def test_repair_agent_targets_only_failed_segments() -> None:
    llm = HeuristicLLMClient()
    script, report, _ = _build_script_and_report()

    weakened_script = EpisodeScript.model_validate(
        {
            **script.model_dump(mode="python"),
            "segments": [
                {
                    **script.segments[0].model_dump(mode="python"),
                    "claims": [
                        {
                            **script.segments[0].claims[0].model_dump(mode="python"),
                            "evidence_chunk_ids": [],
                        }
                    ],
                },
                *[segment.model_dump(mode="python") for segment in script.segments[1:]],
            ],
        }
    )
    weakened_report = GroundingValidationAgent(llm).validate(weakened_script, [])

    repair = RepairAgent(llm).repair(weakened_script, weakened_report, attempt=1)

    assert repair.attempt == 1
    assert repair.repaired_segment_ids
    assert repair.repaired_segment_ids[0] == weakened_script.segments[0].segment_id
