"""Typer CLI for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="podcast-agent", help="Multi-book thematic podcast pipeline.")


@app.command()
def run(
    sources: list[str] = typer.Argument(..., help="Paths to book files (PDF, TXT, MD)."),
    theme: str = typer.Option(..., "--theme", "-t", help="Theme to explore across books."),
    episodes: int = typer.Option(3, "--episodes", "-n", help="Number of episodes to produce."),
    theme_elaboration: Optional[str] = typer.Option(None, "--elaboration", help="Optional longer theme description."),
    titles: Optional[str] = typer.Option(None, "--titles", help="Comma-separated book titles."),
    authors: Optional[str] = typer.Option(None, "--authors", help="Comma-separated author names."),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Force narrative strategy (thesis_driven, debate, chronological, convergence, mosaic)."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Custom output directory."),
    skip_grounding: bool = typer.Option(False, "--skip-grounding", help="Skip grounding validation and repair."),
    skip_spoken_delivery: bool = typer.Option(False, "--skip-spoken-delivery", help="Skip spoken delivery rewrite."),
    skip_audio: bool = typer.Option(False, "--skip-audio", help="Skip audio synthesis (still writes render manifest)."),
    project_id: Optional[str] = typer.Option(None, "--project-id", help="Custom project ID (default: auto-generated UUID)."),
) -> None:
    """Run the full multi-book thematic podcast pipeline."""
    from podcast_agent.config import Settings
    from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
    from podcast_agent.schemas.models import PipelineConfig

    settings = Settings()
    if output_dir:
        settings = settings.model_copy(
            update={"pipeline": settings.pipeline.model_copy(update={"artifact_root": Path(output_dir)})}
        )

    config_updates: dict = {}
    if strategy:
        config_updates["narrative_strategy_override"] = strategy
    if skip_grounding:
        config_updates["skip_grounding"] = True
    if skip_spoken_delivery:
        config_updates["skip_spoken_delivery"] = True
    if skip_audio:
        config_updates["skip_audio"] = True
    config = PipelineConfig(**config_updates) if config_updates else PipelineConfig()

    title_list = [t.strip() for t in titles.split(",")] if titles else None
    author_list = [a.strip() for a in authors.split(",")] if authors else None

    orchestrator = PipelineOrchestrator(settings)

    try:
        if not settings.database.dsn:
            raise RuntimeError(
                "DATABASE_URL must be set to enable vector retrieval for passage extraction."
            )
        project = asyncio.run(
            orchestrator.run_multi_book_podcast(
                source_paths=sources,
                theme=theme,
                episode_count=episodes,
                config=config,
                theme_elaboration=theme_elaboration,
                titles=title_list,
                authors=author_list,
                project_id=project_id,
            )
        )
        typer.echo(f"Pipeline complete. Project ID: {project.project_id}")
        typer.echo(f"Artifacts: {settings.pipeline.artifact_root / project.project_id}")
        typer.echo(f"Status: {project.status.value}")
        typer.echo(f"Books: {len(project.books)}")
    except RuntimeError as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def status(
    project_id: str = typer.Argument(..., help="Project ID to check."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory."),
) -> None:
    """Check the status of a pipeline run."""
    from podcast_agent.config import Settings

    settings = Settings()
    if output_dir:
        artifact_root = Path(output_dir)
    else:
        artifact_root = settings.pipeline.artifact_root

    project_file = artifact_root / project_id / "thematic_project.json"
    if not project_file.exists():
        typer.echo(f"Project not found: {project_id}", err=True)
        raise typer.Exit(code=1)

    data = json.loads(project_file.read_text())
    typer.echo(f"Project: {project_id}")
    typer.echo(f"Theme: {data.get('theme', 'N/A')}")
    typer.echo(f"Status: {data.get('status', 'unknown')}")
    typer.echo(f"Books: {len(data.get('books', []))}")
    typer.echo(f"Episodes: {data.get('episode_count', 0)}")


def main() -> None:
    app()
