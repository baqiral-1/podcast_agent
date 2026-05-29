#!/usr/bin/env python3
"""Rerun a single episode's writing stage and compare it against baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

from podcast_agent.config import AgentConfig, Settings
from podcast_agent.langchain.llm import _supports_adaptive_thinking
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_host_policy_payload,
    _save_json,
)
from podcast_agent.prompts import (
    episode_writing_no_citations_instructions,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeScript,
    NarrativeStrategy,
    ThematicCorpus,
    ThematicProject,
)

VARIANT_LABEL = "no_citations_host_stance_v1"
DEFAULT_WRITING_MODEL_NAME = "claude-sonnet-4-6"
_WORD_RE = re.compile(r"\S+")
_PRONOUN_PATTERNS: dict[str, re.Pattern[str]] = {
    "i_me_my": re.compile(r"\b(?:i|me|my)\b", re.IGNORECASE),
    "we_us_our": re.compile(r"\b(?:we|us|our)\b", re.IGNORECASE),
    "you_your": re.compile(r"\b(?:you|your)\b", re.IGNORECASE),
}
_SURFACE_MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    "first_person_scene_camera": re.compile(
        r"\b(?:we|let(?:'s| us))\s+"
        r"(?:enter|step into|move into|arrive(?:\s+at)?|walk into|go to|follow)\b"
        r"|\bcome with me\b",
        re.IGNORECASE,
    ),
}

T = TypeVar("T")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_provider(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return "openai-compatible"
    return normalized


def _default_provider(settings: Settings) -> str:
    llm_provider = _normalize_provider(settings.llm.llm_provider)
    provider = _normalize_provider(settings.llm.provider)
    if llm_provider == "heuristic" or provider == "heuristic":
        return "heuristic"
    if llm_provider == "anthropic" or provider == "anthropic":
        return "anthropic"
    if llm_provider == "openai-compatible" or provider == "openai-compatible":
        return "openai-compatible"
    raise ValueError(f"Unsupported LLM provider '{settings.llm.llm_provider}'.")


def _provider_for_schema(settings: Settings, schema_name: str) -> str:
    overrides = {
        str(key): _normalize_provider(value)
        for key, value in settings.llm.provider_overrides.items()
    }
    return overrides.get(schema_name, _default_provider(settings))


def _settings_with_writing_model(settings: Settings, model_name: str) -> Settings:
    episode_writing_cfg = settings.llm.agent_configs.get("episode_writing")
    updated_episode_writing_cfg = (
        episode_writing_cfg.model_copy(update={"model_name": model_name})
        if episode_writing_cfg is not None
        else AgentConfig(model_name=model_name)
    )
    updated_agent_configs = dict(settings.llm.agent_configs)
    updated_agent_configs["episode_writing"] = updated_episode_writing_cfg
    return settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={"agent_configs": updated_agent_configs}
            )
        }
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


def _load_episode_plans(project_dir: Path) -> list[EpisodePlan]:
    payload = _load_json(project_dir / "series_plan.json")
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise RuntimeError("series_plan.json must contain a non-empty episodes list")
    return [EpisodePlan.model_validate(item) for item in episodes_payload]


def _load_episode_architectures(project_dir: Path) -> list[EpisodeArchitecture]:
    payload = _load_json(project_dir / "episode_architectures.json")
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise RuntimeError("episode_architectures.json must contain a non-empty episodes list")
    return [EpisodeArchitecture.model_validate(item) for item in episodes_payload]


def _writing_model_metadata(settings: Settings) -> dict[str, Any]:
    schema_name = "episode_writing"
    model_name = settings.llm.resolve_model(schema_name)
    provider = _provider_for_schema(settings, schema_name)
    adaptive_thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    if provider == "anthropic":
        if _supports_adaptive_thinking(str(model_name or "")):
            adaptive_thinking_effort = settings.llm.resolve_anthropic_thinking_effort(
                schema_name
            )
        else:
            thinking_budget_tokens = settings.llm.resolve_thinking_budget(schema_name)
    return {
        "schema_name": schema_name,
        "model_name": model_name,
        "provider": provider,
        "adaptive_thinking_effort": adaptive_thinking_effort,
        "thinking_budget_tokens": thinking_budget_tokens,
    }


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "").strip()))


def _pronoun_counts(text: str) -> dict[str, int]:
    normalized = str(text or "")
    return {
        family: len(pattern.findall(normalized))
        for family, pattern in _PRONOUN_PATTERNS.items()
    }


def _surface_marker_counts(text: str) -> dict[str, int]:
    normalized = str(text or "")
    return {
        marker: len(pattern.findall(normalized))
        for marker, pattern in _SURFACE_MARKER_PATTERNS.items()
    }


def _script_section_stats(script: EpisodeScript) -> list[dict[str, Any]]:
    return [
        {
            "section_id": section.section_id,
            "word_count": _word_count(section.text),
            "pronouns": _pronoun_counts(section.text),
            "surface_markers": _surface_marker_counts(section.text),
        }
        for section in script.prose_sections
    ]


def _script_summary(script: EpisodeScript) -> dict[str, Any]:
    combined_text = "\n".join(section.text for section in script.prose_sections)
    return {
        "word_count": script.total_word_count,
        "pronouns": _pronoun_counts(combined_text),
        "surface_markers": _surface_marker_counts(combined_text),
        "sections": _script_section_stats(script),
    }


def _build_section_deltas(
    baseline_sections: list[dict[str, Any]],
    variant_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {item["section_id"]: item for item in baseline_sections}
    variant_by_id = {item["section_id"]: item for item in variant_sections}
    ordered_section_ids: list[str] = []
    for section in baseline_sections + variant_sections:
        section_id = str(section["section_id"])
        if section_id not in ordered_section_ids:
            ordered_section_ids.append(section_id)

    rows: list[dict[str, Any]] = []
    for section_id in ordered_section_ids:
        baseline = baseline_by_id.get(section_id)
        variant = variant_by_id.get(section_id)
        delta_pronouns: dict[str, int] = {}
        delta_surface_markers: dict[str, int] = {}
        pronoun_keys = set()
        surface_marker_keys = set()
        if baseline is not None:
            pronoun_keys.update(baseline["pronouns"])
            surface_marker_keys.update(baseline["surface_markers"])
        if variant is not None:
            pronoun_keys.update(variant["pronouns"])
            surface_marker_keys.update(variant["surface_markers"])
        for key in sorted(pronoun_keys):
            delta_pronouns[key] = int((variant or {"pronouns": {}})["pronouns"].get(key, 0)) - int(
                (baseline or {"pronouns": {}})["pronouns"].get(key, 0)
            )
        for key in sorted(surface_marker_keys):
            delta_surface_markers[key] = int(
                (variant or {"surface_markers": {}})["surface_markers"].get(key, 0)
            ) - int(
                (baseline or {"surface_markers": {}})["surface_markers"].get(key, 0)
            )
        rows.append(
            {
                "section_id": section_id,
                "baseline": baseline,
                "variant": variant,
                "delta": {
                    "word_count": int((variant or {}).get("word_count", 0))
                    - int((baseline or {}).get("word_count", 0)),
                    "pronouns": delta_pronouns,
                    "surface_markers": delta_surface_markers,
                },
            }
        )
    return rows


def _build_comparison_summary(
    *,
    source_project_dir: Path,
    experiment_project_dir: Path,
    episode_number: int,
    baseline_script: EpisodeScript,
    variant_script: EpisodeScript,
    writing_model: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = _script_summary(baseline_script)
    variant_summary = _script_summary(variant_script)
    delta_pronouns = {
        key: int(variant_summary["pronouns"].get(key, 0))
        - int(baseline_summary["pronouns"].get(key, 0))
        for key in sorted(
            set(baseline_summary["pronouns"]).union(variant_summary["pronouns"])
        )
    }
    delta_surface_markers = {
        key: int(variant_summary["surface_markers"].get(key, 0))
        - int(baseline_summary["surface_markers"].get(key, 0))
        for key in sorted(
            set(baseline_summary["surface_markers"]).union(
                variant_summary["surface_markers"]
            )
        )
    }
    return {
        "variant_label": VARIANT_LABEL,
        "source_run_dir": str(source_project_dir),
        "experiment_run_dir": str(experiment_project_dir),
        "episode_number": episode_number,
        "writing_model": writing_model,
        "baseline": {
            "word_count": baseline_summary["word_count"],
            "pronouns": baseline_summary["pronouns"],
            "surface_markers": baseline_summary["surface_markers"],
        },
        "variant": {
            "word_count": variant_summary["word_count"],
            "pronouns": variant_summary["pronouns"],
            "surface_markers": variant_summary["surface_markers"],
        },
        "delta": {
            "word_count": int(variant_summary["word_count"])
            - int(baseline_summary["word_count"]),
            "pronouns": delta_pronouns,
            "surface_markers": delta_surface_markers,
        },
        "sections": _build_section_deltas(
            baseline_summary["sections"],
            variant_summary["sections"],
        ),
    }


async def _rerun_writing_variant_oneoff(
    project_id: str,
    episode_number: int,
    model_name: str = DEFAULT_WRITING_MODEL_NAME,
) -> None:
    settings = _settings_with_writing_model(Settings(), model_name)
    source_project_dir = settings.pipeline.artifact_root / project_id
    if not source_project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {source_project_dir}")

    project = ThematicProject.model_validate(
        _load_json(source_project_dir / "thematic_project.json")
    )
    if not project.config.skip_grounding:
        raise RuntimeError(
            "This one-off writing variant only supports runs with "
            "project.config.skip_grounding=true."
        )

    baseline_ep_dir = source_project_dir / "episodes" / str(episode_number)
    baseline_script_path = baseline_ep_dir / "episode_script.json"
    if not baseline_script_path.exists():
        raise RuntimeError(
            "Baseline episode_script.json is required before rerunning the writing "
            f"variant: {baseline_script_path}"
        )

    baseline_script = EpisodeScript.model_validate(_load_json(baseline_script_path))
    corpus = ThematicCorpus.model_validate(
        _load_json(source_project_dir / "thematic_corpus.json")
    )
    strategy = NarrativeStrategy.model_validate(
        _load_json(source_project_dir / "narrative_strategy.json")
    )
    episode_architectures = _load_episode_architectures(source_project_dir)
    episode_plans = _load_episode_plans(source_project_dir)
    actor_metadata = ActorMetadata.model_validate(
        _load_json(source_project_dir / "actor_metadata.json")
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

    experiment_project_dir = (
        source_project_dir / "writing_experiments" / VARIANT_LABEL
    )
    experiment_episode_dir = (
        experiment_project_dir / "episodes" / str(episode_number)
    )
    experiment_episode_dir.mkdir(parents=True, exist_ok=True)
    writing_model = _writing_model_metadata(settings)

    orchestrator = PipelineOrchestrator(settings)
    orchestrator._bind_run_logger(experiment_project_dir)
    orchestrator.writing_agent_no_citations.instructions = (
        episode_writing_no_citations_instructions()
    )

    _save_json(experiment_episode_dir / "baseline_episode_script.json", baseline_script)
    baseline_spine_path = baseline_ep_dir / "spine_diagnostics.json"
    if baseline_spine_path.exists():
        _save_json(
            experiment_episode_dir / "baseline_spine_diagnostics.json",
            _load_json(baseline_spine_path),
        )

    variant_script = await orchestrator._write_episode(
        target_plan,
        target_strategy_episode,
        target_architecture,
        project,
        corpus,
        experiment_episode_dir,
        experiment_project_dir,
        actor_metadata,
        _build_host_policy_payload(strategy.narrator_profile),
    )

    _save_json(
        experiment_episode_dir / "comparison_summary.json",
        _build_comparison_summary(
            source_project_dir=source_project_dir,
            experiment_project_dir=experiment_project_dir,
            episode_number=episode_number,
            baseline_script=baseline_script,
            variant_script=variant_script,
            writing_model=writing_model,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-id",
        required=True,
        help="Run directory name under the configured artifact root.",
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        required=True,
        help="Episode number to rerun with the writing prompt variant.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_WRITING_MODEL_NAME,
        help="Override the writing-stage model for this one-off run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        _rerun_writing_variant_oneoff(
            args.project_id,
            args.episode_number,
            model_name=args.model_name,
        )
    )


if __name__ == "__main__":
    main()
