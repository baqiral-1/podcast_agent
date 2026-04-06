"""Typer CLI for the multi-book thematic podcast pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="podcast-agent", help="Multi-book thematic podcast pipeline.")


def _parse_sub_themes(raw: Optional[str]) -> list[str]:
    if not raw:
        return []

    items = raw.split(",")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        trimmed = item.strip()
        if not trimmed:
            raise typer.BadParameter(
                "Sub-themes must be a comma-separated list of non-empty values."
            )
        if trimmed in seen:
            continue
        seen.add(trimmed)
        normalized.append(trimmed)

    if len(normalized) > 15:
        raise typer.BadParameter("Sub-themes supports at most 15 values.")
    return normalized


def _normalize_tts_provider(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "openai":
        return "openai-compatible"
    if normalized in {"openai-compatible", "kokoro"}:
        return normalized
    raise typer.BadParameter(
        f"Invalid --tts-provider value '{value}'. Expected openai-compatible or kokoro.",
        param_hint="tts-provider",
    )


@app.command()
def run(
    sources: list[str] = typer.Argument(..., help="Paths to book files (PDF, TXT, MD)."),
    theme: str = typer.Option(..., "--theme", "-t", help="Theme to explore across books."),
    episodes: Optional[int] = typer.Option(
        None,
        "--episodes",
        "-n",
        help=(
            "Override number of episodes "
            "(otherwise inferred from narrative strategy)."
        ),
    ),
    theme_elaboration: Optional[str] = typer.Option(None, "--elaboration", help="Optional longer theme description."),
    sub_themes: Optional[str] = typer.Option(
        None,
        "--sub-themes",
        help="Optional comma-separated sub-themes to augment decomposition.",
    ),
    titles: Optional[str] = typer.Option(None, "--titles", help="Comma-separated book titles."),
    authors: Optional[str] = typer.Option(None, "--authors", help="Comma-separated author names."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Custom output directory."),
    skip_grounding: bool = typer.Option(False, "--skip-grounding", help="Skip grounding validation and repair."),
    skip_spoken_delivery: bool = typer.Option(False, "--skip-spoken-delivery", help="Skip spoken delivery rewrite."),
    skip_audio: bool = typer.Option(False, "--skip-audio", help="Skip audio synthesis (still writes render manifest)."),
    tts_provider: Optional[str] = typer.Option(
        None,
        "--tts-provider",
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
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
    resolved_tts_provider = _normalize_tts_provider(tts_provider)
    if resolved_tts_provider is not None:
        settings = settings.model_copy(
            update={"tts": settings.tts.model_copy(update={"provider": resolved_tts_provider})}
        )

    config_updates: dict = {"tts_provider": settings.tts.provider}
    if skip_grounding:
        config_updates["skip_grounding"] = True
    if skip_spoken_delivery:
        config_updates["skip_spoken_delivery"] = True
    if skip_audio:
        config_updates["skip_audio"] = True
    config = PipelineConfig(**config_updates) if config_updates else PipelineConfig()

    title_list = [t.strip() for t in titles.split(",")] if titles else None
    author_list = [a.strip() for a in authors.split(",")] if authors else None
    sub_theme_list = _parse_sub_themes(sub_themes)

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
                sub_themes=sub_theme_list,
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
    sub_themes = data.get("sub_themes", [])
    typer.echo(
        f"Sub-themes: {', '.join(sub_themes) if sub_themes else 'None'}"
    )
    typer.echo(f"Status: {data.get('status', 'unknown')}")
    typer.echo(f"Books: {len(data.get('books', []))}")
    typer.echo(f"Episodes: {data.get('episode_count', 0)}")


@app.command("synthesize-audio")
def synthesize_audio(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Existing run directory containing episode render manifests.",
    ),
    tts_provider: str = typer.Option(
        ...,
        "--tts-provider",
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
) -> None:
    """Synthesize audio from existing render manifests in a run directory."""
    from podcast_agent.config import Settings
    from podcast_agent.pipeline.orchestrator import PipelineOrchestrator

    settings = Settings()
    resolved_tts_provider = _normalize_tts_provider(tts_provider)
    settings = settings.model_copy(
        update={"tts": settings.tts.model_copy(update={"provider": resolved_tts_provider})}
    )
    orchestrator = PipelineOrchestrator(settings)

    try:
        summary = asyncio.run(orchestrator.synthesize_audio_from_run(run_dir))
    except RuntimeError as exc:
        typer.echo(f"Audio synthesis failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Run: {summary['run_dir']}")
    typer.echo(f"Processed: {summary['processed']}")
    typer.echo(f"Succeeded: {summary['succeeded']}")
    typer.echo(f"Failed: {summary['failed']}")
    typer.echo(f"Skipped: {summary['skipped']}")
    if summary["skipped_episodes"]:
        typer.echo(
            "Skipped episodes: "
            + ", ".join(str(episode) for episode in summary["skipped_episodes"])
        )
    if summary["failures"]:
        for failure in summary["failures"]:
            typer.echo(failure, err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()
