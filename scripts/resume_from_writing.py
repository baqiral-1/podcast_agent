from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_host_policy_payload,
    _flatten_synthesis_primitives,
    _save_json,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    EpisodeArchitecture,
    EpisodePlan,
    NarrativeStrategy,
    ProjectStatus,
    SpokenScript,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicCorpus,
    ThematicProject,
)


STRICT_TRACKED_FILES = (
    "thematic_axes.json",
    "thematic_corpus.json",
    "synthesis_primitives.json",
    "synthesis_map.json",
    "narrative_strategy.json",
    "episode_architectures.json",
    "series_plan.json",
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


def _load_episode_plans(project_dir: Path) -> list[EpisodePlan]:
    payload = _load_json(project_dir / "series_plan.json")
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise RuntimeError("series_plan.json must contain a non-empty episodes list")

    episode_plans = [EpisodePlan.model_validate(item) for item in episodes_payload]
    episode_numbers = [plan.episode_number for plan in episode_plans]
    if len(episode_numbers) != len(set(episode_numbers)):
        raise RuntimeError("series_plan.json contains duplicate episode_number values")
    return sorted(episode_plans, key=lambda plan: plan.episode_number)


def _load_episode_architectures(project_dir: Path) -> list[EpisodeArchitecture]:
    payload = _load_json(project_dir / "episode_architectures.json")
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise RuntimeError(
            "episode_architectures.json must contain a non-empty episodes list"
        )
    architectures = [
        EpisodeArchitecture.model_validate(item) for item in episodes_payload
    ]
    episode_numbers = [episode.episode_number for episode in architectures]
    if len(episode_numbers) != len(set(episode_numbers)):
        raise RuntimeError(
            "episode_architectures.json contains duplicate episode_number values"
        )
    return sorted(architectures, key=lambda episode: episode.episode_number)


def _load_existing_stage_metrics(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "actor_metadata_metrics.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    stage_metrics = payload.get("stage_metrics", {})
    if not isinstance(stage_metrics, dict):
        return {}
    return dict(stage_metrics)


async def _resume_from_writing(project_id: str) -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(project_dir)

    project = ThematicProject.model_validate(
        _load_json(project_dir / "thematic_project.json")
    )
    corpus = ThematicCorpus.model_validate(
        _load_json(project_dir / "thematic_corpus.json")
    )
    _ = SynthesisPrimitivesArtifact.model_validate(
        _load_json(project_dir / "synthesis_primitives.json")
    )
    synthesis_map = SynthesisMap.model_validate(
        _load_json(project_dir / "synthesis_map.json")
    )
    strategy = NarrativeStrategy.model_validate(
        _load_json(project_dir / "narrative_strategy.json")
    )
    episode_architectures = _load_episode_architectures(project_dir)
    episode_plans = _load_episode_plans(project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(project_dir / "actor_metadata.json")
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
            "episode_count": len(episode_plans),
            "recommended_episode_count": strategy.recommended_episode_count
            or project.recommended_episode_count,
            "status": ProjectStatus.PRODUCING,
        }
    )
    _save_json(project_dir / "thematic_project.json", project)

    architecture_by_number = {
        episode.episode_number: episode for episode in episode_architectures
    }
    strategy_by_number = {
        episode.episode_number: episode for episode in strategy.episodes
    }
    try:
        sem = asyncio.Semaphore(max(1, project.config.episode_write_concurrency))
        spoken_sem = asyncio.Semaphore(
            max(
                1,
                project.config.spoken_delivery_concurrency
                or project.config.episode_write_concurrency,
            )
        )
        host_policy = _build_host_policy_payload(strategy.narrator_profile)
        retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        ep_tasks = [
            orchestrator._produce_episode(
                plan,
                strategy_by_number[plan.episode_number],
                architecture_by_number[plan.episode_number],
                project,
                corpus,
                actor_metadata,
                project_dir,
                host_policy=host_policy,
                primitive_lookup=retained_primitive_lookup,
                semaphore=sem,
                spoken_semaphore=spoken_sem,
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

        completed_episode_numbers = [
            episode_number for episode_number, _ in spoken_scripts
        ]
        expected_episode_numbers = [plan.episode_number for plan in episode_plans]
        if completed_episode_numbers != expected_episode_numbers:
            production_errors.append(
                "completed episode numbers did not match series_plan.json "
                f"({completed_episode_numbers} != {expected_episode_numbers})"
            )

        orchestrator._write_passage_utilization(
            project=project,
            corpus=corpus,
            episode_plans=episode_plans,
            project_dir=project_dir,
            episode_numbers=completed_episode_numbers,
        )

        actor_metrics = _load_existing_stage_metrics(project_dir)
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
                "Resume failed from writing onward: " + "; ".join(details)
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
            "Resume an existing run from writing onward using existing series plan, "
            "synthesis artifacts, and actor metadata with strict integrity checks."
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
    asyncio.run(_resume_from_writing(args.project_id))


if __name__ == "__main__":
    main()
