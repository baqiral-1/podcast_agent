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
    _validate_architecture_transition,
    _validate_plan_transition,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    EpisodeArchitecture,
    EpisodePlan,
    ExcerptArtifact,
    NarrativeState,
    NarrativeStrategy,
    PrimitiveFunctionTaggingArtifact,
    ProjectStatus,
    SceneDiscoveryArtifact,
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
STRICT_TRACKED_FILES_WITH_NARRATIVE_STRATEGY_INPUTS = (
    *STRICT_TRACKED_FILES,
    "tagged_primitives.json",
    "scene_discovery.json",
)
STRICT_TRACKED_FILES_WITH_EPISODE_ARCHITECTURE_INPUTS = (
    *STRICT_TRACKED_FILES,
    "scene_discovery.json",
    "retained_primitives.json",
    "narrative_strategy.json",
)
STRICT_TRACKED_FILES_WITH_PLAN_INPUTS = (
    *STRICT_TRACKED_FILES,
    "scene_discovery.json",
    "retained_primitives.json",
    "narrative_strategy.json",
    "episode_architectures.json",
    "series_plan.json",
)
STRICT_TRACKED_FILES_WITH_EPISODE_PLANNING_INPUTS = (
    *STRICT_TRACKED_FILES,
    "scene_discovery.json",
    "retained_primitives.json",
    "narrative_strategy.json",
    "episode_architectures.json",
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


def _tracked_files_with_narrative_states(
    project_dir: Path,
    *,
    tracked_files: tuple[str, ...],
    episode_numbers: list[int],
) -> tuple[str, ...]:
    dynamic_files = list(tracked_files)
    if (project_dir / "narrative_state_timeline.json").exists():
        dynamic_files.append("narrative_state_timeline.json")
    if (project_dir / "narrative_state_latest.json").exists():
        dynamic_files.append("narrative_state_latest.json")
    for episode_number in episode_numbers:
        dynamic_files.append(
            f"episodes/{episode_number}/narrative_state_pre.json"
        )
        dynamic_files.append(
            f"episodes/{episode_number}/narrative_state_post.json"
        )
    return tuple(dynamic_files)


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


def _load_scene_discovery(project_dir: Path) -> SceneDiscoveryArtifact:
    return SceneDiscoveryArtifact.model_validate(
        _load_json(project_dir / "scene_discovery.json")
    )


def _load_narrative_strategy(project_dir: Path) -> NarrativeStrategy:
    strategy = NarrativeStrategy.model_validate(
        _load_json(project_dir / "narrative_strategy.json")
    )
    if not strategy.episodes:
        raise RuntimeError(
            "Persisted narrative_strategy.json must contain at least one episode."
        )
    return strategy


def _load_narrative_state(path: Path) -> NarrativeState:
    return NarrativeState.model_validate(_load_json(path))


def _load_episode_narrative_states(
    project_dir: Path,
    episode_numbers: list[int],
) -> tuple[dict[int, NarrativeState], dict[int, NarrativeState]]:
    pre_by_episode: dict[int, NarrativeState] = {}
    post_by_episode: dict[int, NarrativeState] = {}
    for episode_number in episode_numbers:
        ep_dir = project_dir / "episodes" / str(episode_number)
        pre_path = ep_dir / "narrative_state_pre.json"
        post_path = ep_dir / "narrative_state_post.json"
        if not pre_path.exists() or not post_path.exists():
            raise RuntimeError(
                "Resume requires narrative state artifacts for every episode. "
                f"Missing state snapshot(s) for episode {episode_number}: "
                f"{pre_path.name if not pre_path.exists() else ''} "
                f"{post_path.name if not post_path.exists() else ''}".strip()
            )
        pre_by_episode[episode_number] = _load_narrative_state(pre_path)
        post_by_episode[episode_number] = _load_narrative_state(post_path)
    return pre_by_episode, post_by_episode


def _load_existing_stage_metrics(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "actor_metadata_metrics.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    stage_metrics = payload.get("stage_metrics", {})
    if not isinstance(stage_metrics, dict):
        return {}
    return dict(stage_metrics)


async def _plan_series_for_resume(
    *,
    orchestrator: Any,
    project: ThematicProject,
    synthesis_map: SynthesisMap,
    strategy: NarrativeStrategy,
    corpus: ThematicCorpus,
    project_dir: Path,
    actor_metadata: ActorMetadata,
    scene_discovery: SceneDiscoveryArtifact | None = None,
    retained_excerpts: ExcerptArtifact | None = None,
) -> tuple[
    list[EpisodeArchitecture],
    list[EpisodePlan],
    dict[int, NarrativeState],
    dict[int, NarrativeState],
    dict[str, Any],
]:
    return await orchestrator._plan_series_with_narrative_state(
        project=project,
        synthesis_map=synthesis_map,
        strategy=strategy,
        corpus=corpus,
        project_dir=project_dir,
        actor_metadata=actor_metadata,
        scene_discovery=scene_discovery,
        retained_excerpts=retained_excerpts,
    )


def _load_excerpt_artifact_if_present(
    project_dir: Path, filename: str
) -> ExcerptArtifact | None:
    """Load an excerpt artifact from disk; return None when absent.

    Resume paths are gracefully tolerant of older runs that predate the
    excerpt stage. New runs always emit `excerpts.json` and
    `retained_excerpts.json` and resume points that depend on them will
    log/warn but not hard-fail when they are missing.
    """

    path = project_dir / filename
    if not path.exists():
        return None
    return ExcerptArtifact.model_validate(_load_json(path))


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


def _verify_strategy_scene_references(
    *,
    strategy: NarrativeStrategy,
    scene_discovery: SceneDiscoveryArtifact,
) -> None:
    candidate_ids = {
        candidate.candidate_id for candidate in scene_discovery.candidates
    }
    unknown_candidate_ids = sorted(
        {
            candidate_id
            for episode in strategy.episodes
            for beat in episode.promised_beats
            for candidate_id in beat.source_candidate_ids
            if candidate_id and candidate_id not in candidate_ids
        }
    )
    if unknown_candidate_ids:
        raise RuntimeError(
            "narrative_strategy.json references unknown scene discovery candidates: "
            f"{unknown_candidate_ids[:10]}"
        )


def _verify_architectures_against_strategy(
    *,
    strategy: NarrativeStrategy,
    episode_architectures: list[EpisodeArchitecture],
) -> None:
    strategy_by_number = {
        episode.episode_number: episode for episode in strategy.episodes
    }
    architecture_numbers = [episode.episode_number for episode in episode_architectures]
    strategy_numbers = sorted(strategy_by_number)
    if architecture_numbers != strategy_numbers:
        raise RuntimeError(
            "episode_architectures.json episode numbers do not match "
            f"narrative_strategy.json ({architecture_numbers} != {strategy_numbers})"
        )
    for architecture in episode_architectures:
        try:
            _validate_architecture_transition(
                strategy_episode=strategy_by_number[architecture.episode_number],
                architecture=architecture,
            )
        except Exception as exc:
            raise RuntimeError(
                "Persisted episode_architectures.json failed validation for "
                f"episode {architecture.episode_number}: {exc}"
            ) from exc


def _verify_plans_against_strategy_and_architecture(
    *,
    strategy: NarrativeStrategy,
    episode_architectures: list[EpisodeArchitecture],
    episode_plans: list[EpisodePlan],
) -> None:
    strategy_by_number = {
        episode.episode_number: episode for episode in strategy.episodes
    }
    architecture_by_number = {
        episode.episode_number: episode for episode in episode_architectures
    }
    plan_numbers = [plan.episode_number for plan in episode_plans]
    strategy_numbers = sorted(strategy_by_number)
    architecture_numbers = sorted(architecture_by_number)
    if plan_numbers != strategy_numbers or plan_numbers != architecture_numbers:
        raise RuntimeError(
            "series_plan.json episode numbers do not match persisted strategy and "
            "architecture artifacts"
        )
    for plan in episode_plans:
        try:
            _validate_plan_transition(
                strategy_episode=strategy_by_number[plan.episode_number],
                architecture=architecture_by_number[plan.episode_number],
                plan=plan,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Persisted series_plan.json failed validation for episode {plan.episode_number}: {exc}"
            ) from exc


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


async def _produce_from_persisted_plan(
    *,
    orchestrator: Any,
    project: ThematicProject,
    corpus: ThematicCorpus,
    actor_metadata: ActorMetadata,
    synthesis_map: SynthesisMap,
    strategy: NarrativeStrategy,
    episode_architectures: list[EpisodeArchitecture],
    episode_plans: list[EpisodePlan],
    project_dir: Path,
    actor_metrics: dict[str, Any],
    narrative_state_pre_by_episode: dict[int, NarrativeState] | None = None,
    narrative_state_post_by_episode: dict[int, NarrativeState] | None = None,
    retained_excerpts: ExcerptArtifact | None = None,
) -> None:
    project = project.model_copy(update={"status": ProjectStatus.PRODUCING})
    _save_json(project_dir / "thematic_project.json", project)

    architecture_by_number = {
        episode.episode_number: episode for episode in episode_architectures
    }
    strategy_by_number = {
        episode.episode_number: episode for episode in strategy.episodes
    }
    episode_numbers = [plan.episode_number for plan in episode_plans]
    if narrative_state_pre_by_episode is None or narrative_state_post_by_episode is None:
        (
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
        ) = _load_episode_narrative_states(project_dir, episode_numbers)
    sem = asyncio.Semaphore(max(1, project.config.episode_write_concurrency))
    spoken_sem = asyncio.Semaphore(
        max(
            1,
            project.config.spoken_delivery_concurrency
            or project.config.episode_write_concurrency,
        )
    )
    retained_primitive_lookup = _flatten_synthesis_primitives(synthesis_map)
    if retained_excerpts is None:
        retained_excerpts = _load_excerpt_artifact_if_present(
            project_dir, "retained_excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)
    retained_excerpt_by_id = retained_excerpts.excerpt_by_id()
    ep_tasks = [
        orchestrator._produce_episode(
            plan,
            strategy_by_number[plan.episode_number],
            architecture_by_number[plan.episode_number],
            project,
            corpus,
            actor_metadata,
            project_dir,
            host_policy=_build_host_policy_payload(
                strategy.narrator_profile,
                narrative_state_pre=narrative_state_pre_by_episode[plan.episode_number],
                narrative_state_post=narrative_state_post_by_episode[plan.episode_number],
            ),
            primitive_lookup=retained_primitive_lookup,
            semaphore=sem,
            spoken_semaphore=spoken_sem,
            series_explanation_registry=strategy.series_explanation_registry,
            narrative_state_pre=narrative_state_pre_by_episode[plan.episode_number],
            excerpt_by_id=retained_excerpt_by_id,
            narrative_state_post=narrative_state_post_by_episode[plan.episode_number],
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

    completed_episode_numbers = [episode_number for episode_number, _ in spoken_scripts]
    expected_episode_numbers = [plan.episode_number for plan in episode_plans]
    if not production_errors and completed_episode_numbers != expected_episode_numbers:
        production_errors.append(
            "completed episode numbers did not match planned episodes "
            f"({completed_episode_numbers} != {expected_episode_numbers})"
        )

    orchestrator._write_passage_utilization(
        project=project,
        corpus=corpus,
        episode_plans=episode_plans,
        project_dir=project_dir,
        episode_numbers=completed_episode_numbers,
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
        raise RuntimeError("; ".join(details))


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
            (synthesis_primitives, synthesis_actor_metrics),
            excerpts,
        ) = await asyncio.gather(
            orchestrator._map_synthesis(
                project=project,
                corpus=corpus,
                project_dir=project_dir,
                actor_metadata=actor_metadata,
            ),
            orchestrator._extract_excerpts(
                project=project,
                corpus=corpus,
                project_dir=project_dir,
                actor_metadata=actor_metadata,
            ),
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

        scene_discovery = await orchestrator._discover_scenes(
            project=project,
            synthesis_map=synthesis_primitives,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=synthesis_primitives,
            project_dir=project_dir,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
            excerpts=excerpts,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=synthesis_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )
        retained_excerpts = await orchestrator._materialize_selected_excerpts(
            project=project,
            excerpts=excerpts,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            episode_plans,
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
            planning_state_metrics,
        ) = await _plan_series_for_resume(
            orchestrator=orchestrator,
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            retained_excerpts=retained_excerpts,
        )
        actor_metrics["episode_architecture"] = planning_state_metrics.get(
            "episode_architecture", {}
        )
        actor_metrics["episode_planning"] = planning_state_metrics.get(
            "episode_planning", {}
        )

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
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
        (
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
        ) = _load_episode_narrative_states(project_dir, [episode_number])
        host_policy = _build_host_policy_payload(
            strategy.narrator_profile,
            narrative_state_pre=narrative_state_pre_by_episode[episode_number],
            narrative_state_post=narrative_state_post_by_episode[episode_number],
        )
        primitive_lookup = _flatten_synthesis_primitives(retained_primitives)
        retained_excerpts = _load_excerpt_artifact_if_present(
            project_dir, "retained_excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)
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
            narrative_state_pre=narrative_state_pre_by_episode[episode_number],
            narrative_state_post=narrative_state_post_by_episode[episode_number],
            excerpt_by_id=retained_excerpts.excerpt_by_id(),
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

        excerpts = _load_excerpt_artifact_if_present(
            project_dir, "excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)

        scene_discovery = await orchestrator._discover_scenes(
            project=project,
            synthesis_map=synthesis_primitives,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
        )
        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=synthesis_primitives,
            project_dir=project_dir,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
            excerpts=excerpts,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=synthesis_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )
        retained_excerpts = await orchestrator._materialize_selected_excerpts(
            project=project,
            excerpts=excerpts,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            episode_plans,
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
            planning_state_metrics,
        ) = await _plan_series_for_resume(
            orchestrator=orchestrator,
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            retained_excerpts=retained_excerpts,
        )
        actor_metrics["episode_architecture"] = planning_state_metrics.get(
            "episode_architecture", {}
        )
        actor_metrics["episode_planning"] = planning_state_metrics.get(
            "episode_planning", {}
        )

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
        raise


async def resume_from_narrative_strategy_stage(
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
        project_dir, tracked_files=STRICT_TRACKED_FILES_WITH_NARRATIVE_STRATEGY_INPUTS
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
    tagged_primitives = PrimitiveFunctionTaggingArtifact.model_validate(
        _load_json(project_dir / "tagged_primitives.json")
    )
    scene_discovery = _load_scene_discovery(project_dir)

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

    actor_metrics = _load_existing_stage_metrics(project_dir)
    try:
        excerpts = _load_excerpt_artifact_if_present(
            project_dir, "excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)

        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=tagged_primitives,
            project_dir=project_dir,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
            excerpts=excerpts,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        _verify_strategy_scene_references(
            strategy=strategy,
            scene_discovery=scene_discovery,
        )
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=tagged_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )
        retained_excerpts = await orchestrator._materialize_selected_excerpts(
            project=project,
            excerpts=excerpts,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            episode_plans,
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
            planning_state_metrics,
        ) = await _plan_series_for_resume(
            orchestrator=orchestrator,
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            retained_excerpts=retained_excerpts,
        )
        actor_metrics["episode_architecture"] = planning_state_metrics.get(
            "episode_architecture", {}
        )
        actor_metrics["episode_planning"] = planning_state_metrics.get(
            "episode_planning", {}
        )

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
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

        excerpts = _load_excerpt_artifact_if_present(
            project_dir, "excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)

        (
            strategy,
            strategy_actor_metrics,
        ) = await orchestrator._choose_narrative_strategy(
            project=project,
            synthesis_map=tagged_primitives,
            project_dir=project_dir,
            scene_discovery=scene_discovery,
            actor_metadata=actor_metadata,
            excerpts=excerpts,
        )
        actor_metrics["narrative_strategy"] = strategy_actor_metrics
        _verify_strategy_scene_references(
            strategy=strategy,
            scene_discovery=scene_discovery,
        )
        synthesis_map = await orchestrator._materialize_selected_primitives(
            project=project,
            synthesis_primitives=tagged_primitives,
            strategy=strategy,
            project_dir=project_dir,
        )
        retained_excerpts = await orchestrator._materialize_selected_excerpts(
            project=project,
            excerpts=excerpts,
            strategy=strategy,
            project_dir=project_dir,
        )

        project = orchestrator._resolve_episode_count_from_strategy(project, strategy)
        _save_json(project_dir / "thematic_project.json", project)

        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        (
            episode_architectures,
            episode_plans,
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
            planning_state_metrics,
        ) = await _plan_series_for_resume(
            orchestrator=orchestrator,
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            retained_excerpts=retained_excerpts,
        )
        actor_metrics["episode_architecture"] = planning_state_metrics.get(
            "episode_architecture", {}
        )
        actor_metrics["episode_planning"] = planning_state_metrics.get(
            "episode_planning", {}
        )

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
        raise


async def resume_from_episode_architecture_stage(
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
        project_dir,
        tracked_files=STRICT_TRACKED_FILES_WITH_EPISODE_ARCHITECTURE_INPUTS,
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
    scene_discovery = _load_scene_discovery(project_dir)
    synthesis_map = SynthesisMap.model_validate(
        _load_json(project_dir / "retained_primitives.json")
    )
    strategy = _load_narrative_strategy(project_dir)

    if _axis_digest(axes_from_file) != _axis_digest(corpus.axes):
        raise RuntimeError(
            "thematic_axes.json and thematic_corpus.json disagree on axes; aborting strict resume."
        )
    _verify_strategy_scene_references(
        strategy=strategy,
        scene_discovery=scene_discovery,
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
            "episode_count": len(strategy.episodes),
            "recommended_episode_count": strategy.recommended_episode_count
            or project.recommended_episode_count,
            "status": ProjectStatus.PLANNING,
        }
    )
    _save_json(project_dir / "thematic_project.json", project)

    actor_metrics = _load_existing_stage_metrics(project_dir)
    try:
        retained_excerpts = _load_excerpt_artifact_if_present(
            project_dir, "retained_excerpts.json"
        ) or ExcerptArtifact(project_id=project.project_id)
        (
            episode_architectures,
            episode_plans,
            narrative_state_pre_by_episode,
            narrative_state_post_by_episode,
            planning_state_metrics,
        ) = await _plan_series_for_resume(
            orchestrator=orchestrator,
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            scene_discovery=scene_discovery,
            retained_excerpts=retained_excerpts,
        )
        actor_metrics["episode_architecture"] = planning_state_metrics.get(
            "episode_architecture", {}
        )
        actor_metrics["episode_planning"] = planning_state_metrics.get(
            "episode_planning", {}
        )

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
        raise


async def resume_from_episode_planning_stage(
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
    scene_discovery = _load_scene_discovery(project_dir)
    synthesis_map = SynthesisMap.model_validate(
        _load_json(project_dir / "retained_primitives.json")
    )
    strategy = _load_narrative_strategy(project_dir)
    episode_architectures = _load_episode_architectures(project_dir)
    episode_numbers = [
        architecture.episode_number for architecture in episode_architectures
    ]
    strict_snapshots = _capture_snapshots(
        project_dir,
        tracked_files=_tracked_files_with_narrative_states(
            project_dir,
            tracked_files=STRICT_TRACKED_FILES_WITH_EPISODE_PLANNING_INPUTS,
            episode_numbers=episode_numbers,
        ),
    )
    # Strict resume rebuilds planning inputs from the canonical persisted stage
    # outputs plus per-episode narrative-state snapshots. The lightweight stage
    # summary under stage_artifacts/episode_planning/input.json is not the
    # authoritative planning payload.
    (
        narrative_state_pre_by_episode,
        narrative_state_post_by_episode,
    ) = _load_episode_narrative_states(project_dir, episode_numbers)

    if _axis_digest(axes_from_file) != _axis_digest(corpus.axes):
        raise RuntimeError(
            "thematic_axes.json and thematic_corpus.json disagree on axes; aborting strict resume."
        )
    _verify_strategy_scene_references(
        strategy=strategy,
        scene_discovery=scene_discovery,
    )
    _verify_architectures_against_strategy(
        strategy=strategy,
        episode_architectures=episode_architectures,
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
            "episode_count": len(episode_architectures),
            "recommended_episode_count": strategy.recommended_episode_count
            or project.recommended_episode_count,
            "status": ProjectStatus.PLANNING,
        }
    )
    _save_json(project_dir / "thematic_project.json", project)

    actor_metrics = _load_existing_stage_metrics(project_dir)
    try:
        episode_plans, planning_actor_metrics = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            corpus=corpus,
            project_dir=project_dir,
            actor_metadata=actor_metadata,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
        )
        actor_metrics["episode_planning"] = planning_actor_metrics

        await _produce_from_persisted_plan(
            orchestrator=orchestrator,
            project=project,
            corpus=corpus,
            actor_metadata=actor_metadata,
            synthesis_map=synthesis_map,
            strategy=strategy,
            episode_architectures=episode_architectures,
            episode_plans=episode_plans,
            project_dir=project_dir,
            actor_metrics=actor_metrics,
            narrative_state_pre_by_episode=narrative_state_pre_by_episode,
            narrative_state_post_by_episode=narrative_state_post_by_episode,
        )

        _verify_snapshots_unchanged(project_dir=project_dir, before=strict_snapshots)

        project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
        _save_json(project_dir / "thematic_project.json", project)
    except Exception as exc:
        project = project.model_copy(update={"status": ProjectStatus.FAILED})
        _save_json(project_dir / "thematic_project.json", project)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                f"Resume failed from {stage_label} onward: {exc}"
            ) from exc
        raise
