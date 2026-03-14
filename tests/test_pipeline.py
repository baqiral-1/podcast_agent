"""Integration and unit tests for the podcast pipeline."""

from __future__ import annotations

from pathlib import Path

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository
from podcast_agent.llm import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.tts import TTSClient


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


class FakeTTSClient(TTSClient):
    """Simple deterministic TTS stub for pipeline tests."""

    def synthesize(self, text: str, voice: str | None = None, audio_format: str | None = None) -> bytes:
        return f"{voice or 'default'}::{audio_format or 'mp3'}::{text}".encode("utf-8")


def test_pipeline_creates_multi_chapter_episode_and_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(book_path, title="Observatory Book", author="A. Writer")

    assert result["series_plan"]["episodes"]
    assert len(result["series_plan"]["episodes"]) == 1
    assert len(result["series_plan"]["episodes"][0]["chapter_ids"]) == 4
    assert result["episodes"][0]["manifest"]["segments"]
    episode_output_path = (
        tmp_path
        / "runs"
        / orchestrator.run_id
        / "observatory-book"
        / result["episodes"][0]["plan"]["episode_id"]
        / "episode_output.json"
    )
    assert episode_output_path.exists()
    assert not episode_output_path.with_name("script.json").exists()
    assert not episode_output_path.with_name("grounding_report.json").exists()
    assert not episode_output_path.with_name("render_manifest.json").exists()


def test_index_book_persists_chunks_and_embeddings(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    repository = InMemoryRepository()
    orchestrator = PipelineOrchestrator(repository=repository, llm=HeuristicLLMClient())

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion)

    assert structure.book_id in repository.structures
    assert structure.book_id in repository.embeddings
    assert len(repository.structures[structure.book_id].chunks) == len(repository.embeddings[structure.book_id])


def test_ingest_book_reads_pdf_source_and_persists_artifact(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.read_source_text",
        lambda path: "[Page 1]\nChapter 1: Arrival\nThe observatory appears.",
    )

    ingestion = orchestrator.ingest_book(pdf_path, title="Observatory Book", author="A. Writer")

    assert ingestion.source_type.value == "pdf"
    assert ingestion.raw_text.startswith("[Page 1]")
    artifact_path = tmp_path / "runs" / orchestrator.run_id / "observatory-book" / "ingestion.json"
    assert artifact_path.exists()
    artifact_payload = artifact_path.read_text(encoding="utf-8")
    assert '"source_type": "pdf"' in artifact_payload
    assert '"raw_text": "[Page 1]\\nChapter 1: Arrival\\nThe observatory appears."' in artifact_payload


def test_validation_schema_is_claim_level(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
    )

    ingestion = orchestrator.ingest_book(book_path, title="Observatory Book", author="A. Writer")
    structure = orchestrator.index_book(ingestion)
    _, plan = orchestrator.plan_episodes(structure)
    script = orchestrator.write_episode(structure.book_id, plan.episodes[0])
    report = orchestrator.validate_episode(structure.book_id, script)

    assert report.claim_assessments
    assert all(assessment.claim_id for assessment in report.claim_assessments)


def test_books_under_source_word_floor_collapse_to_single_episode() -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    result = orchestrator.run_pipeline(
        Path("examples/river_of_hours.txt"),
        title="River of Hours",
        author="Sample Author",
    )

    assert len(result["series_plan"]["episodes"]) == 1
    assert len(result["series_plan"]["episodes"][0]["chapter_ids"]) == 12


def test_river_of_hours_text_indexes_same_chapter_count(tmp_path: Path) -> None:
    settings = Settings()
    settings = settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": tmp_path / "runs"})})
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        settings=settings,
    )

    text_path = Path("tests/fixtures/river_of_hours.txt")
    ingestion = orchestrator.ingest_book(text_path, title="River of Hours", author="Sample Author")
    structure = orchestrator.index_book(ingestion)

    assert ingestion.source_type.value == "text"
    assert len(structure.chapters) == 12
    assert structure.chapters[0].title == "Chapter 1: On Beginnings"
    assert structure.chapters[-1].title == "Chapter 12: A Practical Metaphysics"


def test_pipeline_can_synthesize_audio_manifest(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text(BOOK_TEXT, encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        repository=InMemoryRepository(),
        llm=HeuristicLLMClient(),
        tts=FakeTTSClient(),
    )

    result = orchestrator.run_pipeline(
        book_path,
        title="Observatory Book",
        author="A. Writer",
        synthesize_audio=True,
    )

    audio_manifest = result["episodes"][0]["audio_manifest"]
    assert audio_manifest is not None
    assert audio_manifest["segments"]
    assert audio_manifest["audio_path"].endswith(".mp3")
    assert Path(audio_manifest["audio_path"]).exists()
