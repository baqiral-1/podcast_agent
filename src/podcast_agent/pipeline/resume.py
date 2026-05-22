from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from podcast_agent.pipeline.orchestrator import (
    _build_host_policy_payload,
    _flatten_synthesis_primitives,
    _save_json,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    EpisodeArchitecture,
    EpisodePlan,
    NarrativeStrategy,
    PrimitiveFunctionTaggingArtifact,
    ProjectStatus,
    SpokenScript,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)

logger = logging.getLogger(__name__)


STRICT_TRACKED_FILES = (
    "thematic_axes.json",
    "thematic_corpus.json",
    "actor_metadata.json",
)
STRICT_TRACKED_FILES_WITH_SUBSTRATE_PRIMITIVES = (
    *STRICT_TRACKED_FILES,
    "substrate_primitives.json",
)
STRICT_TRACKED_FILES_WITH_SCENE_DISCOVERY_INPUTS = (
    *STRICT_TRACKED_FILES,
    "substrate_primitives.json",
    "tagged_primitives.json",
)
STRICT_TRACKED_FILES_WITH_RETAINED_PRIMITIVES = (
    *STRICT_TRACKED_FILES,
    "retained_primitives.json",
    "narrative_strategy.json",
    "episode_architectures.json",
    "series_plan.json",
)

SettingsFactory = Callable[[], Any]
OrchestratorFactory = Callable[[Any], Any]
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


def _capture_snapshots(
    project_dir: Path,
    *,
    tracked_files: tuple[str, ...] = STRICT_TRACKED_FILES,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for filename in tracked_files:
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
        raise RuntimeError(
            "Invalid thematic_axes.json: expected top-level 'axes' list."
        )
    return [ThematicAxis.model_validate(axis) for axis in axes_payload]


def _axis_digest(axes: list[ThematicAxis]) -> str:
    normalized = [axis.model_dump(mode="json") for axis in axes]
    serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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


def _primitive_id_list(
    artifact: SynthesisPrimitivesArtifact | PrimitiveFunctionTaggingArtifact,
) -> list[str]:
    return [primitive.id for primitive in artifact.primitives]


def _verify_primitive_ids_match(
    *,
    substrate_primitives: SynthesisPrimitivesArtifact,
    tagged_primitives: PrimitiveFunctionTaggingArtifact,
) -> None:
    substrate_ids = _primitive_id_list(substrate_primitives)
    tagged_ids = _primitive_id_list(tagged_primitives)
    if substrate_ids != tagged_ids:
        raise RuntimeError(
            "substrate_primitives.json and tagged_primitives.json disagree on "
            "primitive ids; aborting strict resume."
        )


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
        script_path = (
            project_dir / "episodes" / str(plan.episode_number) / "episode_script.json"
        )
        if script_path.exists():
            completed.append(plan.episode_number)
    return completed


def _load_completed_spoken_scripts(
    project_dir: Path,
    episode_plans: list[EpisodePlan],
) -> list[tuple[int, SpokenScript]]:
    spoken_scripts: list[tuple[int, SpokenScript]] = []
    for plan in episode_plans:
        spoken_path = (
            project_dir / "episodes" / str(plan.episode_number) / "spoken_script.json"
        )
        if not spoken_path.exists():
            continue
        spoken = SpokenScript.model_validate(_load_json(spoken_path))
        spoken_scripts.append((plan.episode_number, spoken))
    return spoken_scripts


async def resume_from_synthesis_stage(
    project_id: str,
    *,
    stage_label: str,
    settings_factory: SettingsFactory,
    orchestrator_cls: OrchestratorFactory,
) -> None:
    settings = settings_factory()
    orchestrator = orchestrator_cls(settings)
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
    axes_from_file = _load_axes_from_file(project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(project_dir / "actor_metadata.json")
    )

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
        (
            synthesis_primitives,
            synthesis_actor_metrics,
        ) = await orchestrator._map_synthesis(
            project=project,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["synthesis_primitives"] = synthesis_actor_metrics.get(
            "primitives", {}
        )

        if (
            synthesis_primitives.quality_score
            < project.config.synthesis_quality_threshold
        ):
            logger.warning(
                "Synthesis quality %.2f below threshold %.2f. "
                "Books may lack thematic overlap for strong synthesis.",
                synthesis_primitives.quality_score,
                project.config.synthesis_quality_threshold,
            )
            orchestrator.run_logger.log(
                "synthesis_quality_warning",
                score=synthesis_primitives.quality_score,
                threshold=project.config.synthesis_quality_threshold,
            )

        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=synthesis_primitives,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=synthesis_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            architecture_actor_metrics,
        ) = await orchestrator._build_episode_architectures(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["episode_architecture"] = architecture_actor_metrics

        episode_plans, planning_actor_metrics = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
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
        host_policy = _build_host_policy_payload(strategy.narrator_profile)
        retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        strategy_episode_by_number = {
            episode.episode_number: episode for episode in strategy.episodes
        }
        architecture_by_number = {
            episode.episode_number: episode for episode in episode_architectures
        }
        ep_tasks = [
            orchestrator._produce_episode(
                plan,
                strategy_episode_by_number[plan.episode_number],
                architecture_by_number[plan.episode_number],
                project,
                corpus,
                actor_metadata,
                project_dir,
                host_policy=host_policy,
                primitive_lookup=retained_primitive_lookup,
                semaphore=sem,
                spoken_semaphore=spoken_sem,
                series_explanation_registry=strategy.series_explanation_registry,
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
                f"Resume failed from {stage_label} onward: " + "; ".join(details)
            )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        raise


async def resume_single_episode_from_writing_stage(
    project_id: str,
    *,
    episode_number: int,
    settings_factory: SettingsFactory,
    orchestrator_cls: OrchestratorFactory,
) -> None:
    settings = settings_factory()
    orchestrator = orchestrator_cls(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(
        project_dir, tracked_files=STRICT_TRACKED_FILES_WITH_RETAINED_PRIMITIVES
    )

    project = ThematicProject.model_validate(
        _load_json(project_dir / "thematic_project.json")
    )
    original_status = project.status
    corpus = ThematicCorpus.model_validate(
        _load_json(project_dir / "thematic_corpus.json")
    )
    axes_from_file = _load_axes_from_file(project_dir)
    retained_primitives = SynthesisMap.model_validate(
        _load_json(project_dir / "retained_primitives.json")
    )
    strategy = NarrativeStrategy.model_validate(
        _load_json(project_dir / "narrative_strategy.json")
    )
    episode_architectures = _load_episode_architectures(project_dir)
    episode_plans = _load_episode_plans(project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(project_dir / "actor_metadata.json")
    )

    if _axis_digest(axes_from_file) != _axis_digest(corpus.axes):
        raise RuntimeError(
            "thematic_axes.json and thematic_corpus.json disagree on axes; aborting strict resume."
        )

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
        sem = asyncio.Semaphore(max(1, project.config.episode_write_concurrency))
        spoken_sem = asyncio.Semaphore(
            max(
                1,
                project.config.spoken_delivery_concurrency
                or project.config.episode_write_concurrency,
            )
        )
        host_policy = _build_host_policy_payload(strategy.narrator_profile)
        primitive_lookup = _flatten_synthesis_primitives(retained_primitives)
        _, spoken = await orchestrator._produce_episode(
            target_plan,
            target_strategy_episode,
            target_architecture,
            project,
            corpus,
            actor_metadata,
            project_dir,
            host_policy=host_policy,
            primitive_lookup=primitive_lookup,
            semaphore=sem,
            spoken_semaphore=spoken_sem,
            series_explanation_registry=strategy.series_explanation_registry,
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
        completed_spoken_scripts = _load_completed_spoken_scripts(
            project_dir, episode_plans
        )
        actor_metrics["writing"] = orchestrator._build_writing_actor_metrics(
            project_dir,
            completed_spoken_scripts,
        )
        orchestrator._write_actor_metadata_metrics(
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            synthesis_map=retained_primitives,
            strategy=strategy,
            episode_plans=episode_plans,
            metrics=actor_metrics,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": original_status})
        _save_json(project_dir / "thematic_project.json", project)
        raise


async def resume_from_substrate_function_tagging_stage(
    project_id: str,
    *,
    stage_label: str,
    settings_factory: SettingsFactory,
    orchestrator_cls: OrchestratorFactory,
) -> None:
    settings = settings_factory()
    orchestrator = orchestrator_cls(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(
        project_dir, tracked_files=STRICT_TRACKED_FILES_WITH_SUBSTRATE_PRIMITIVES
    )

    project = ThematicProject.model_validate(
        _load_json(project_dir / "thematic_project.json")
    )
    corpus = ThematicCorpus.model_validate(
        _load_json(project_dir / "thematic_corpus.json")
    )
    axes_from_file = _load_axes_from_file(project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(project_dir / "actor_metadata.json")
    )
    substrate_primitives = SynthesisPrimitivesArtifact.model_validate(
        _load_json(project_dir / "substrate_primitives.json")
    )

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

    actor_metrics: dict[str, Any] = {"synthesis_primitives": {}}
    try:
        synthesis_primitives, tagging_metrics = await orchestrator._tag_substrate_primitives(
            project=project,
            corpus=corpus,
            project_dir=project_dir,
            primitives=substrate_primitives,
            actor_metadata=actor_metadata,
        )
        actor_metrics["substrate_function_tagging"] = tagging_metrics

        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=synthesis_primitives,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=synthesis_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            architecture_actor_metrics,
        ) = await orchestrator._build_episode_architectures(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["episode_architecture"] = architecture_actor_metrics

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        episode_plans, planning_actor_metrics = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
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
        host_policy = _build_host_policy_payload(strategy.narrator_profile)
        retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        ep_tasks = [
            orchestrator._produce_episode(
                plan,
                next(
                    item
                    for item in strategy.episodes
                    if item.episode_number == plan.episode_number
                ),
                next(
                    item
                    for item in episode_architectures
                    if item.episode_number == plan.episode_number
                ),
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
                f"Resume failed from {stage_label} onward: " + "; ".join(details)
            )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        raise


async def resume_from_scene_discovery_stage(
    project_id: str,
    *,
    stage_label: str,
    settings_factory: SettingsFactory,
    orchestrator_cls: OrchestratorFactory,
) -> None:
    settings = settings_factory()
    orchestrator = orchestrator_cls(settings)
    project_dir = settings.pipeline.artifact_root / project_id
    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    strict_snapshots = _capture_snapshots(
        project_dir, tracked_files=STRICT_TRACKED_FILES_WITH_SCENE_DISCOVERY_INPUTS
    )

    project = ThematicProject.model_validate(
        _load_json(project_dir / "thematic_project.json")
    )
    corpus = ThematicCorpus.model_validate(
        _load_json(project_dir / "thematic_corpus.json")
    )
    axes_from_file = _load_axes_from_file(project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(project_dir / "actor_metadata.json")
    )
    substrate_primitives = SynthesisPrimitivesArtifact.model_validate(
        _load_json(project_dir / "substrate_primitives.json")
    )
    tagged_primitives = PrimitiveFunctionTaggingArtifact.model_validate(
        _load_json(project_dir / "tagged_primitives.json")
    )

    if _axis_digest(axes_from_file) != _axis_digest(corpus.axes):
        raise RuntimeError(
            "thematic_axes.json and thematic_corpus.json disagree on axes; aborting strict resume."
        )
    _verify_primitive_ids_match(
        substrate_primitives=substrate_primitives,
        tagged_primitives=tagged_primitives,
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

    actor_metrics = _load_existing_stage_metrics(project_dir)
    try:
        scene_discovery = await orchestrator._discover_scenes(
            project=project,
            synthesis_map=tagged_primitives,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )

        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=tagged_primitives,
            project_dir=project_dir,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=tagged_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            architecture_actor_metrics,
        ) = await orchestrator._build_episode_architectures(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        actor_metrics["episode_architecture"] = architecture_actor_metrics

        episode_plans, planning_actor_metrics = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
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
        host_policy = _build_host_policy_payload(strategy.narrator_profile)
        retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
        strategy_episode_by_number = {
            episode.episode_number: episode for episode in strategy.episodes
        }
        architecture_by_number = {
            episode.episode_number: episode for episode in episode_architectures
        }
        ep_tasks = [
            orchestrator._produce_episode(
                plan,
                strategy_episode_by_number[plan.episode_number],
                architecture_by_number[plan.episode_number],
                project,
                corpus,
                actor_metadata,
                project_dir,
                host_policy=host_policy,
                primitive_lookup=retained_primitive_lookup,
                semaphore=sem,
                spoken_semaphore=spoken_sem,
                series_explanation_registry=strategy.series_explanation_registry,
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
                f"Resume failed from {stage_label} onward: " + "; ".join(details)
            )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        raise
