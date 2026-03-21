"""Typer CLI for the podcast agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from podcast_agent.config import Settings
from podcast_agent.db import InMemoryRepository, PostgresRepository
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator

app = typer.Typer(help="Book-to-podcast pipeline with strict stage artifacts.")

_AGENT_SCHEMA_OVERRIDES: dict[str, str] = {
    "structuring": "structured_chapter",
    "analysis": "book_analysis",
    "planning": "series_plan",
    "writing": "beat_script",
    "validation": "grounding_report",
    "repair": "episode_repair",
    "spoken_delivery": "spoken_delivery_narration",
    "spoken_delivery_plan": "spoken_delivery_plan",
    "spoken_delivery_narration": "spoken_delivery_narration",
}


def _parse_agent_model_overrides(
    raw_overrides: list[str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if not raw_overrides:
        return {}, {}
    model_overrides: dict[str, str] = {}
    provider_overrides: dict[str, str] = {}
    supported_providers = {"openai", "openai-compatible", "anthropic", "heuristic"}
    for entry in raw_overrides:
        if "=" not in entry:
            raise typer.BadParameter(
                f"Invalid --agent-model value '{entry}'. Expected AGENT=MODEL or AGENT=PROVIDER:MODEL.",
                param_hint="agent-model",
            )
        raw_agent, raw_model = entry.split("=", 1)
        agent = raw_agent.strip().lower().replace("-", "_")
        model = raw_model.strip()
        if not agent:
            raise typer.BadParameter(
                f"Invalid --agent-model value '{entry}'. Agent name is empty.",
                param_hint="agent-model",
            )
        if not model:
            raise typer.BadParameter(
                f"Invalid --agent-model value '{entry}'. Model name is empty.",
                param_hint="agent-model",
            )
        provider = None
        if ":" in model:
            provider_candidate, model_candidate = (part.strip() for part in model.split(":", 1))
            if provider_candidate.lower() in supported_providers:
                if not model_candidate:
                    raise typer.BadParameter(
                        f"Invalid --agent-model value '{entry}'. Expected AGENT=PROVIDER:MODEL.",
                        param_hint="agent-model",
                    )
                provider = provider_candidate.lower()
                model = model_candidate
        schema_name = _AGENT_SCHEMA_OVERRIDES.get(agent)
        if schema_name is None:
            allowed = ", ".join(sorted(_AGENT_SCHEMA_OVERRIDES))
            raise typer.BadParameter(
                f"Unknown agent '{agent}' in --agent-model. Allowed agents: {allowed}.",
                param_hint="agent-model",
            )
        model_overrides[schema_name] = model
        if provider is not None:
            provider_overrides[schema_name] = provider
    return model_overrides, provider_overrides


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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Ingest a book source file."""

    orchestrator = _build_orchestrator(
        database_url, model=model, agent_model=agent_model, llm_provider=llm_provider
    )
    orchestrator.log_command(
        "ingest-book",
        {
            "source_path": str(source_path),
            "title": title,
            "author": author,
            "database_url": database_url,
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Structure a book and persist chunks plus embeddings."""

    orchestrator = _build_orchestrator(
        database_url, model=model, agent_model=agent_model, llm_provider=llm_provider
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
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Create a series plan from a source book."""

    orchestrator = _build_orchestrator(
        database_url, model=model, agent_model=agent_model, llm_provider=llm_provider
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
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    tts_provider: str | None = typer.Option(
        default=None,
        help="TTS provider for audio synthesis (openai-compatible, kokoro).",
    ),
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the full pipeline and print the resulting artifacts."""

    orchestrator = _build_orchestrator(
        database_url,
        model=model,
        agent_model=agent_model,
        llm_provider=llm_provider,
        tts_provider=_normalize_tts_provider(tts_provider),
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
            "tts_provider": tts_provider,
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Run the pipeline and synthesize one audio file per episode."""

    orchestrator = _build_orchestrator(
        database_url,
        model=model,
        agent_model=agent_model,
        llm_provider=llm_provider,
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
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
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
        model=model,
        agent_model=agent_model,
        llm_provider=llm_provider,
        tts_provider=_normalize_tts_provider(tts_provider),
    )
    orchestrator.log_command(
        "render-audio-from-manifest",
        {
            "artifact_path": str(artifact_path),
            "book_id": book_id,
            "database_url": database_url,
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = typer.Option(
        default=None,
        help="Override the default chat model name for all LLM calls.",
    ),
    llm_provider: str | None = typer.Option(
        default=None,
        help="Override the LLM provider (openai, anthropic, heuristic).",
    ),
    agent_model: list[str] | None = typer.Option(
        None,
        "--agent-model",
        help="Per-agent model override as AGENT=MODEL or AGENT=PROVIDER:MODEL (repeatable).",
    ),
    database_url: str | None = typer.Option(
        default=None,
        help="PostgreSQL database URL. If omitted, falls back to DATABASE_URL when set.",
    ),
) -> None:
    """Rewrite a saved factual script artifact into spoken-form delivery."""

    orchestrator = _build_orchestrator(
        database_url,
        model=model,
        agent_model=agent_model,
        llm_provider=llm_provider,
    )
    orchestrator.log_command(
        "spoken-delivery",
        {
            "artifact_path": str(artifact_path),
            "database_url": database_url,
            "model": model,
            "llm_provider": llm_provider,
            "agent_model": agent_model,
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
    model: str | None = None,
    llm_provider: str | None = None,
    agent_model: list[str] | None = None,
    tts_provider: str | None = None,
) -> PipelineOrchestrator:
    settings = Settings()
    resolved_database_url = database_url or settings.database.dsn
    repository = (
        PostgresRepository(resolved_database_url)
        if resolved_database_url
        else InMemoryRepository()
    )
    llm_updates: dict[str, object] = {}
    agent_model_overrides, provider_overrides = _parse_agent_model_overrides(agent_model)
    if model is not None:
        llm_updates["model_name"] = model
    if llm_provider is not None:
        llm_updates["llm_provider"] = llm_provider
    if agent_model_overrides:
        llm_updates["model_overrides"] = {**settings.llm.model_overrides, **agent_model_overrides}
    if provider_overrides:
        llm_updates["provider_overrides"] = {
            **settings.llm.provider_overrides,
            **provider_overrides,
        }

    settings_updates: dict[str, object] = {
        "database": settings.database.model_copy(update={"dsn": resolved_database_url}),
    }
    if llm_updates:
        settings_updates["llm"] = settings.llm.model_copy(update=llm_updates)
    if tts_provider is not None:
        settings_updates["tts"] = settings.tts.model_copy(update={"provider": tts_provider})
    settings = settings.model_copy(update=settings_updates)
    return PipelineOrchestrator(settings=settings, repository=repository)


if __name__ == "__main__":
    main()
