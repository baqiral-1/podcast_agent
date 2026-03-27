"""Typer CLI for the podcast agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository, PostgresRepository
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas import BatchRunManifest

app = typer.Typer(help="Book-to-podcast pipeline with strict stage artifacts.")


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


@app.command("ingest-book")
def ingest_book(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Ingest a book source file."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
    )
    orchestrator.log_command(
        "ingest-book",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
        },
    )
    result = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    typer.echo(result.model_dump_json(indent=2))


@app.command("index-book")
def index_book(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    start_chapter: str | None = typer.Option(
        default=None,
        help="Start from this detected chapter title, matched case-insensitively.",
    ),
    end_chapter: str | None = typer.Option(
        default=None,
        help="Stop at this detected chapter title, matched case-insensitively and included in the run.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Structure a book and persist chunks plus embeddings."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
    )
    orchestrator.log_command(
        "index-book",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
        },
    )
    ingestion = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    structure = orchestrator.index_book(
        ingestion,
        start_chapter=start_chapter,
        end_chapter=end_chapter,
    )
    typer.echo(structure.model_dump_json(indent=2))


@app.command("plan-episodes")
def plan_episodes(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    episode_count: int = typer.Option(
        ...,
        min=1,
        help="Generate exactly this many episodes.",
    ),
    start_chapter: str | None = typer.Option(
        default=None,
        help="Start from this detected chapter title, matched case-insensitively.",
    ),
    end_chapter: str | None = typer.Option(
        default=None,
        help="Stop at this detected chapter title, matched case-insensitively and included in the run.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Create a series plan from a source book."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
    )
    orchestrator.log_command(
        "plan-episodes",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "episode_count": episode_count,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
        },
    )
    ingestion = orchestrator.ingest_book(source_path=source_path, title=title, author=author)
    structure = orchestrator.index_book(
        ingestion,
        start_chapter=start_chapter,
        end_chapter=end_chapter,
    )
    _, plan = orchestrator.plan_episodes(structure, episode_count=episode_count)
    typer.echo(plan.model_dump_json(indent=2))


@app.command("run-pipeline")
def run_pipeline(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    episode_count: int = typer.Option(
        ...,
        min=1,
        help="Generate exactly this many episodes.",
    ),
    start_chapter: str | None = typer.Option(
        default=None,
        help="Start from this detected chapter title, matched case-insensitively.",
    ),
    end_chapter: str | None = typer.Option(
        default=None,
        help="Stop at this detected chapter title, matched case-insensitively and included in the run.",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Synthesize audio files after render-manifest generation.",
    ),
    skip_grounding: bool = typer.Option(
        default=False,
        help="Skip grounding (validation and repair) and mark episodes as grounded-pass.",
    ),
    tts_provider: str | None = typer.Option(
        default=None,
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the full pipeline and print the resulting artifacts."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
        tts_provider=_normalize_tts_provider(tts_provider),
        skip_grounding=skip_grounding,
    )
    orchestrator.log_command(
        "run-pipeline",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "episode_count": episode_count,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "database_url": database_url,
            "with_audio": with_audio,
            "skip_grounding": skip_grounding,
            "tts_provider": tts_provider,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
        },
    )
    result = orchestrator.run_pipeline(
        source_path=source_path,
        title=title,
        author=author,
        start_chapter=start_chapter,
        end_chapter=end_chapter,
        episode_count=episode_count,
        synthesize_audio=with_audio,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("run-batch")
def run_batch(
    manifest_path: Path,
    with_audio: bool | None = typer.Option(
        None,
        "--with-audio/--no-with-audio",
        help="Override the manifest audio setting.",
    ),
    skip_grounding: bool = typer.Option(
        default=False,
        help="Skip grounding (validation and repair) and mark episodes as grounded-pass.",
    ),
    run_id: str | None = typer.Option(
        default=None,
        help="Override the manifest run ID.",
    ),
    batch_parallelism: int | None = typer.Option(
        default=None,
        help="Override the batch parallelism for per-stage book processing.",
    ),
    tts_provider: str | None = typer.Option(
        default=None,
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the pipeline for multiple books described in a manifest file."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
        tts_provider=_normalize_tts_provider(tts_provider),
        skip_grounding=skip_grounding,
    )
    try:
        manifest = BatchRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="manifest_path") from exc
    orchestrator.log_command(
        "run-batch",
        {
            "manifest_path": str(manifest_path),
            "database_url": database_url,
            "with_audio": with_audio,
            "skip_grounding": skip_grounding,
            "run_id": run_id,
            "batch_parallelism": batch_parallelism,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
            "tts_provider": tts_provider,
        },
    )
    result = orchestrator.run_batch(
        manifest,
        synthesize_audio=with_audio,
        run_id=run_id,
        batch_parallelism=batch_parallelism,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("render-audio")
def render_audio(
    source_path: Path,
    title: str | None = None,
    author: str = "Unknown",
    episode_count: int = typer.Option(
        ...,
        min=1,
        help="Generate exactly this many episodes.",
    ),
    start_chapter: str | None = typer.Option(
        default=None,
        help="Start from this detected chapter title, matched case-insensitively.",
    ),
    end_chapter: str | None = typer.Option(
        default=None,
        help="Stop at this detected chapter title, matched case-insensitively and included in the run.",
    ),
    tts_provider: str | None = typer.Option(
        default=None,
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the pipeline and synthesize one audio file per episode."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
        tts_provider=_normalize_tts_provider(tts_provider),
    )
    orchestrator.log_command(
        "render-audio",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "episode_count": episode_count,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
            "tts_provider": tts_provider,
        },
    )
    result = orchestrator.run_pipeline(
        source_path=source_path,
        title=title,
        author=author,
        start_chapter=start_chapter,
        end_chapter=end_chapter,
        episode_count=episode_count,
        synthesize_audio=True,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("render-audio-from-manifest")
def render_audio_from_manifest(
    artifact_path: Path,
    book_id: str | None = typer.Option(
        default=None,
        help="Book ID fallback when it cannot be inferred from the artifact path.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    tts_provider: str | None = typer.Option(
        default=None,
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Synthesize episode audio from a saved artifact directory using spoken_script.json."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
        tts_provider=_normalize_tts_provider(tts_provider),
    )
    orchestrator.log_command(
        "render-audio-from-manifest",
        {
            "artifact_path": str(artifact_path),
            "book_id": book_id,
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
            "tts_provider": tts_provider,
        },
    )
    try:
        result = orchestrator.regenerate_audio_from_artifact(artifact_path=artifact_path, book_id=book_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="artifact_path") from exc
    typer.echo(result.model_dump_json(indent=2))


@app.command("spoken-delivery")
def spoken_delivery(
    artifact_path: Path,
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    reasoning_effort: str | None = typer.Option(
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models (none, low, medium, high, xhigh).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Rewrite a saved factual script artifact into spoken-form delivery."""

    orchestrator = _build_orchestrator(
        database_url,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
    )
    orchestrator.log_command(
        "spoken-delivery",
        {
            "artifact_path": str(artifact_path),
            "database_url": database_url,
            "llm_provider": llm_provider,
            "reasoning_effort": reasoning_effort,
        },
    )
    try:
        result = orchestrator.spoken_delivery_from_artifact(artifact_path=artifact_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="artifact_path") from exc
    typer.echo(json.dumps(result, indent=2, default=str))


def main() -> None:
    """Entrypoint used by the console script."""

    app()


def _build_orchestrator(
    database_url: str | None,
    *,
    llm_provider: str | None = None,
    reasoning_effort: str | None = None,
    tts_provider: str | None = None,
    skip_grounding: bool | None = None,
) -> PipelineOrchestrator:
    settings = Settings()
    resolved_database_url = database_url or settings.database.dsn
    repository = (
        PostgresRepository(resolved_database_url)
        if resolved_database_url
        else InMemoryRepository()
    )
    llm_updates: dict[str, object] = {}
    if llm_provider is not None:
        llm_updates["llm_provider"] = llm_provider
    if reasoning_effort is not None:
        llm_updates["reasoning_effort"] = reasoning_effort

    settings_updates: dict[str, object] = {
        "database": settings.database.model_copy(update={"dsn": resolved_database_url}),
    }
    if llm_updates:
        settings_updates["llm"] = settings.llm.model_copy(update=llm_updates)
    if tts_provider is not None:
        settings_updates["tts"] = settings.tts.model_copy(update={"provider": tts_provider})
    if skip_grounding is not None:
        settings_updates["pipeline"] = settings.pipeline.model_copy(
            update={"skip_grounding": skip_grounding}
        )
    settings = settings.model_copy(update=settings_updates)
    return PipelineOrchestrator(settings=settings, repository=repository)


if __name__ == "__main__":
    main()
