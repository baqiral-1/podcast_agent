from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _save_json
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

T = TypeVar("T")


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
        raise RuntimeError("episode_architectures.json must contain a non-empty episodes list")
    architectures = [EpisodeArchitecture.model_validate(item) for item in episodes_payload]
    episode_numbers = [episode.episode_number for episode in architectures]
    if len(episode_numbers) != len(set(episode_numbers)):
        raise RuntimeError("episode_architectures.json contains duplicate episode_number values")
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


def _select_single_episode(
    items: list[T],
    *,
    episode_number: int,
    value_getter: Callable[[T], int],
    source_name: str,
) -> T:
    matches = [item for item in items if value_getter(item) == episode_number]
    if not matches:
        raise RuntimeError(
            f"{source_name} does not contain episode_number {episode_number}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"{source_name} contains duplicate episode_number {episode_number}"
        )
    return matches[0]


def _completed_episode_numbers(
    project_dir: Path,
    episode_plans: list[EpisodePlan],
) -> list[int]:
    completed: list[int] = []
    for plan in episode_plans:
        script_path = project_dir / "episodes" / str(plan.episode_number) / "episode_script.json"
        if script_path.exists():
            completed.append(plan.episode_number)
    return completed


def _load_completed_spoken_scripts(
    project_dir: Path,
    episode_plans: list[EpisodePlan],
) -> list[tuple[int, SpokenScript]]:
    spoken_scripts: list[tuple[int, SpokenScript]] = []
    for plan in episode_plans:
        spoken_path = project_dir / "episodes" / str(plan.episode_number) / "spoken_script.json"
        if not spoken_path.exists():
            continue
        spoken = SpokenScript.model_validate(_load_json(spoken_path))
        spoken_scripts.append((plan.episode_number, spoken))
    return spoken_scripts


async def _retry_single_episode_from_writing(project_id: str, episode_number: int) -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(project_dir)

    project = ThematicProject.model_validate(_load_json(project_dir / "thematic_project.json"))
    original_status = project.status
    corpus = ThematicCorpus.model_validate(_load_json(project_dir / "thematic_corpus.json"))
    _ = SynthesisPrimitivesArtifact.model_validate(
        _load_json(project_dir / "synthesis_primitives.json")
    )
    synthesis_map = SynthesisMap.model_validate(_load_json(project_dir / "synthesis_map.json"))
    strategy = NarrativeStrategy.model_validate(_load_json(project_dir / "narrative_strategy.json"))
    episode_architectures = _load_episode_architectures(project_dir)
    episode_plans = _load_episode_plans(project_dir)
    actor_metadata = ActorMetadata.model_validate(_load_json(project_dir / "actor_metadata.json"))

    target_plan = _select_single_episode(
        episode_plans,
        episode_number=episode_number,
        value_getter=lambda item: item.episode_number,
        source_name="series_plan.json",
    )
    target_architecture = _select_single_episode(
        episode_architectures,
        episode_number=episode_number,
        value_getter=lambda item: item.episode_number,
        source_name="episode_architectures.json",
    )
    target_strategy_episode = _select_single_episode(
        strategy.episodes,
        episode_number=episode_number,
        value_getter=lambda item: item.episode_number,
        source_name="narrative_strategy.json",
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

    ep_dir = project_dir / "episodes" / str(episode_number)
    ep_dir.mkdir(parents=True, exist_ok=True)

    try:
        script = await orchestrator._write_episode(
            target_plan,
            target_strategy_episode,
            target_architecture,
            project,
            corpus,
            ep_dir,
            project_dir,
            actor_metadata,
        )
        spoken = await orchestrator._rewrite_for_speech(
            episode_number,
            script,
            project,
            ep_dir,
            project_dir,
        )
        audio_sem = asyncio.Semaphore(max(1, project.config.tts_concurrency))
        await orchestrator._render_episode_audio(
            episode_number,
            spoken,
            project.config,
            project_dir,
            audio_sem,
            skip_audio=True,
        )

        completed_episode_numbers = _completed_episode_numbers(project_dir, episode_plans)
        orchestrator._write_passage_utilization(
            project=project,
            corpus=corpus,
            episode_plans=episode_plans,
            project_dir=project_dir,
            episode_numbers=completed_episode_numbers,
        )

        actor_metrics = _load_existing_stage_metrics(project_dir)
        completed_spoken_scripts = _load_completed_spoken_scripts(project_dir, episode_plans)
        actor_metrics["writing"] = orchestrator._build_writing_actor_metrics(
            project_dir,
            completed_spoken_scripts,
        )
        orchestrator._write_actor_metadata_metrics(
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_plans=episode_plans,
            metrics=actor_metrics,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        final_status = (
            original_status
            if original_status == ProjectStatus.COMPLETE
            else ProjectStatus.COMPLETE
        )
        project = project.model_copy(update={"status": final_status})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": original_status})
        _save_json(project_dir / "thematic_project.json", project)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry writing, spoken delivery, and render manifest generation for one "
            "episode in an existing run using persisted planning artifacts."
        )
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Run directory name under the configured artifact root.",
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        required=True,
        help="Episode number to retry from writing onward.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_retry_single_episode_from_writing(args.project_id, args.episode_number))


if __name__ == "__main__":
    main()
