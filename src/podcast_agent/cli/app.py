"""Typer CLI for the podcast agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository, PostgresRepository
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator

app = typer.Typer(help="Book-to-podcast pipeline with strict stage artifacts.")


@app.command("ingest-book")
def ingest_book(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Ingest a book source file."""

    orchestrator = _build_orchestrator(database_url)
    orchestrator.log_command(
        "ingest-book",
        {"source_path": str(source_path), "title": title, "author": author, "database_url": database_url},
    )
    result = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    typer.echo(result.model_dump_json(indent=2))


@app.command("index-book")
def index_book(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Structure a book and persist chunks plus embeddings."""

    orchestrator = _build_orchestrator(database_url)
    orchestrator.log_command(
        "index-book",
        {"source_path": str(source_path), "title": title, "author": author, "database_url": database_url},
    )
    ingestion = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    structure = orchestrator.index_book(ingestion)
    typer.echo(structure.model_dump_json(indent=2))


@app.command("plan-episodes")
def plan_episodes(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Create a series plan from a source book."""

    orchestrator = _build_orchestrator(database_url)
    orchestrator.log_command(
        "plan-episodes",
        {"source_path": str(source_path), "title": title, "author": author, "database_url": database_url},
    )
    ingestion = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    structure = orchestrator.index_book(ingestion)
    _, plan = orchestrator.plan_episodes(structure)
    typer.echo(plan.model_dump_json(indent=2))


@app.command("run-pipeline")
def run_pipeline(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    with_audio: bool = typer.Option(
        default=False,
        help="Synthesize audio files after render-manifest generation.",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the full pipeline and print the resulting artifacts."""

    orchestrator = _build_orchestrator(database_url)
    orchestrator.log_command(
        "run-pipeline",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "database_url": database_url,
            "with_audio": with_audio,
        },
    )
    result = orchestrator.run_pipeline(
        source_path=source_path,
        title=title,
        author=author,
        synthesize_audio=with_audio,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("render-audio")
def render_audio(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the pipeline and synthesize per-segment audio output."""

    orchestrator = _build_orchestrator(database_url)
    orchestrator.log_command(
        "render-audio",
        {"source_path": str(source_path), "title": title, "author": author, "database_url": database_url},
    )
    result = orchestrator.run_pipeline(
        source_path=source_path,
        title=title,
        author=author,
        synthesize_audio=True,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


def main() -> None:
    """Entrypoint used by the console script."""

    app()


def _build_orchestrator(database_url: str | None) -> PipelineOrchestrator:
    settings = Settings()
    resolved_database_url = database_url or settings.database.dsn
    repository = (
        PostgresRepository(resolved_database_url)
        if resolved_database_url
        else InMemoryRepository()
    )
    settings = settings.model_copy(
        update={
            "database": settings.database.model_copy(update={"dsn": resolved_database_url}),
        }
    )
    return PipelineOrchestrator(settings=settings, repository=repository)


if __name__ == "__main__":
    main()
