from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _save_json
from podcast_agent.schemas.models import (
    ActorMetadata,
    EpisodeArchitecture,
    ProjectStatus,
    SpokenScript,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


STRICT_TRACKED_FILES = (
    "thematic_axes.json",
    "thematic_corpus.json",
    "actor_metadata.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "replacement_char_count": decoded.count("\ufffd"),
        "size_bytes": len(raw),
    }


def _capture_snapshots(project_dir: Path) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for filename in STRICT_TRACKED_FILES:
        path = project_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing required artifact for strict resume: {path}")
        snapshots[filename] = _snapshot_file(path)
    return snapshots


def _verify_snapshots_unchanged(
    *,
    project_dir: Path,
    before: dict[str, dict[str, Any]],
) -> None:
    for filename, previous in before.items():
        current = _snapshot_file(project_dir / filename)
        if current["sha256"] != previous["sha256"]:
            raise RuntimeError(
                f"Strict integrity check failed: {filename} changed during resume "
                f"({previous['sha256']} -> {current['sha256']})."
            )
        if current["replacement_char_count"] != previous["replacement_char_count"]:
            raise RuntimeError(
                "Strict integrity check failed: replacement character count changed for "
                f"{filename} ({previous['replacement_char_count']} -> "
                f"{current['replacement_char_count']})."
            )


def _load_axes_from_file(project_dir: Path) -> list[ThematicAxis]:
    payload = _load_json(project_dir / "thematic_axes.json")
    axes_payload = payload.get("axes")
    if not isinstance(axes_payload, list):
        raise RuntimeError("Invalid thematic_axes.json: expected top-level 'axes' list.")
    return [ThematicAxis.model_validate(axis) for axis in axes_payload]


def _axis_digest(axes: list[ThematicAxis]) -> str:
    normalized = [axis.model_dump(mode="json") for axis in axes]
    serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_persisted_architectures(project_dir: Path) -> list[EpisodeArchitecture]:
    payload = _load_json(project_dir / "episode_architectures.json")
    architectures_payload = payload.get("episodes")
    if not isinstance(architectures_payload, list):
        raise RuntimeError(
            "Persisted episode_architectures.json must contain a top-level 'episodes' list."
        )
    return [
        EpisodeArchitecture.model_validate(architecture)
        for architecture in architectures_payload
    ]


async def _resume_from_synthesis_mapping(project_id: str) -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(project_dir)

    project = ThematicProject.model_validate(_load_json(project_dir / "thematic_project.json"))
    corpus = ThematicCorpus.model_validate(_load_json(project_dir / "thematic_corpus.json"))
    axes_from_file = _load_axes_from_file(project_dir)
    actor_metadata = ActorMetadata.model_validate(_load_json(project_dir / "actor_metadata.json"))

    if _axis_digest(axes_from_file) != _axis_digest(corpus.axes):
        raise RuntimeError(
            "thematic_axes.json and thematic_corpus.json disagree on axes; aborting strict resume."
        )

    orchestrator._bind_run_logger(project_dir)

    forced_config = project.config.model_copy(
        update={
            "skip_grounding": True,
            "skip_audio": True,
            "skip_spoken_delivery": False,
        }
    )
    project = project.model_copy(
        update={
            "config": forced_config,
            "status": ProjectStatus.ANALYZING,
        }
    )
    _save_json(project_dir / "thematic_project.json", project)

    actor_metrics: dict[str, Any] = {}
    try:
        synthesis_map, synthesis_actor_metrics = await orchestrator._map_synthesis(
            project=project,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["synthesis_primitives"] = synthesis_actor_metrics.get("primitives", {})

        strategy, strategy_actor_metrics = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=synthesis_map,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        episode_architectures, architecture_actor_metrics = await orchestrator._build_episode_architectures(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["episode_architecture"] = architecture_actor_metrics
        persisted_architectures = _load_persisted_architectures(project_dir)
        if len(persisted_architectures) != project.episode_count:
            raise RuntimeError(
                "Persisted episode architectures do not match the resolved episode count."
            )

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        episode_plans, planning_actor_metrics = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=persisted_architectures,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["episode_planning"] = planning_actor_metrics

        project = project.model_copy(update={"status": ProjectStatus.PRODUCING})
        _save_json(project_dir / "thematic_project.json", project)

        sem = asyncio.Semaphore(max(1, project.config.episode_write_concurrency))
        spoken_sem = asyncio.Semaphore(
            max(
                1,
                project.config.spoken_delivery_concurrency
                or project.config.episode_write_concurrency,
            )
        )
        ep_tasks = [
            orchestrator._produce_episode(
                plan,
                next(item for item in strategy.episodes if item.episode_number == plan.episode_number),
                next(item for item in persisted_architectures if item.episode_number == plan.episode_number),
                project,
                corpus,
                actor_metadata,
                project_dir,
                sem,
                spoken_sem,
            )
            for plan in episode_plans
        ]
        ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)

        spoken_scripts: list[tuple[int, SpokenScript]] = []
        production_errors: list[str] = []
        for plan, result in zip(episode_plans, ep_results, strict=True):
            if isinstance(result, Exception):
                production_errors.append(f"episode {plan.episode_number}: {result}")
                continue
            spoken_scripts.append(result)
        spoken_scripts.sort(key=lambda item: item[0])

        orchestrator._write_passage_utilization(
            project=project,
            corpus=corpus,
            episode_plans=episode_plans,
            project_dir=project_dir,
            episode_numbers=[episode_number for episode_number, _ in spoken_scripts],
        )
        actor_metrics["writing"] = orchestrator._build_writing_actor_metrics(
            project_dir,
            spoken_scripts,
        )
        orchestrator._write_actor_metadata_metrics(
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_plans=episode_plans,
            metrics=actor_metrics,
        )

        audio_errors: list[str] = []
        if spoken_scripts:
            audio_sem = asyncio.Semaphore(max(1, project.config.tts_concurrency))
            audio_tasks = [
                orchestrator._render_episode_audio(
                    episode_number,
                    spoken,
                    project.config,
                    project_dir,
                    audio_sem,
                    skip_audio=True,
                )
                for episode_number, spoken in spoken_scripts
            ]
            audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
            for episode_number, result in zip(
                (episode_number for episode_number, _ in spoken_scripts),
                audio_results,
                strict=True,
            ):
                if isinstance(result, Exception):
                    audio_errors.append(f"episode {episode_number}: {result}")

        if production_errors or audio_errors:
            details = production_errors + audio_errors
            raise RuntimeError(
                "Resume failed from synthesis mapping onward: " + "; ".join(details)
            )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an existing run from synthesis mapping onward using existing "
            "theme, corpus, and actor artifacts with strict integrity checks."
        )
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Run directory name under the configured artifact root.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_resume_from_synthesis_mapping(args.project_id))


if __name__ == "__main__":
    main()
