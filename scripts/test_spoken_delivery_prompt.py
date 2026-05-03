#!/usr/bin/env python3
"""Run spoken delivery against an existing episode script with an optional prompt override."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import difflib
import statistics

from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.config import Settings
from podcast_agent.langchain.llm import build_llm_client
from podcast_agent.pipeline.orchestrator import (
    _save_json,
)
from podcast_agent.schemas.models import (
    EpisodeScript,
    PipelineConfig,
    SpokenScript,
    SpokenSection,
)


@dataclass(frozen=True)
class EpisodeInput:
    run_dir: Path
    episode_number: int
    script: EpisodeScript
    pipeline_config: PipelineConfig
    tts_provider: str
    existing_spoken_script: SpokenScript | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run spoken delivery on an existing episode script. "
            "Defaults to independence_v13 episode 1."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/independence_v13"),
        help="Run directory containing episodes/<n>/episode_script.json.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Episode number to use as input.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text file containing a prompt override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the generated spoken script JSON.",
    )
    parser.add_argument(
        "--spoken-max-retry-attempts",
        type=int,
        default=1,
        help="Retry attempts for the spoken delivery agent.",
    )
    return parser.parse_args()


def _default_output_path(run_dir: Path, episode_number: int) -> Path:
    return run_dir / "prompt_lab" / f"episode_{episode_number}_spoken_script.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pipeline_config(run_dir: Path) -> PipelineConfig:
    project_path = run_dir / "thematic_project.json"
    if not project_path.exists():
        return PipelineConfig()

    project_data = _load_json(project_path)
    config_data = project_data.get("config")
    if not isinstance(config_data, dict):
        return PipelineConfig()
    return PipelineConfig.model_validate(config_data)


def _load_existing_spoken_script(episode_dir: Path) -> SpokenScript | None:
    spoken_path = episode_dir / "spoken_script.json"
    if not spoken_path.exists():
        return None
    return SpokenScript.model_validate(_load_json(spoken_path))


def _resolve_tts_provider(
    existing_spoken_script: SpokenScript | None,
    pipeline_config: PipelineConfig,
) -> str:
    if existing_spoken_script is not None and existing_spoken_script.tts_provider:
        return existing_spoken_script.tts_provider
    return pipeline_config.tts_provider


def _load_episode_input(run_dir: Path, episode_number: int) -> EpisodeInput:
    episode_dir = run_dir / "episodes" / str(episode_number)
    script_path = episode_dir / "episode_script.json"
    if not script_path.exists():
        raise RuntimeError(f"Missing episode script: {script_path}")

    script = EpisodeScript.model_validate(_load_json(script_path))
    pipeline_config = _load_pipeline_config(run_dir)
    existing_spoken_script = _load_existing_spoken_script(episode_dir)
    tts_provider = _resolve_tts_provider(existing_spoken_script, pipeline_config)

    return EpisodeInput(
        run_dir=run_dir,
        episode_number=episode_number,
        script=script,
        pipeline_config=pipeline_config,
        tts_provider=tts_provider,
        existing_spoken_script=existing_spoken_script,
    )


def _chunk_stats(ratios: list[float]) -> list[tuple[str, float]]:
    if not ratios:
        return []
    third = (len(ratios) + 2) // 3
    chunks = [
        ("early", ratios[:third]),
        ("mid", ratios[third : 2 * third]),
        ("late", ratios[2 * third :]),
    ]
    stats: list[tuple[str, float]] = []
    for label, values in chunks:
        if values:
            stats.append((label, statistics.mean(values)))
    return stats


def _similarity_ratios(
    script: EpisodeScript,
    spoken_sections: list[Any],
) -> list[float]:
    if not spoken_sections:
        return []
    ratios: list[float] = []
    for source_section, spoken_section in zip(script.prose_sections, spoken_sections, strict=True):
        ratios.append(
            difflib.SequenceMatcher(
                None,
                source_section.text.strip(),
                spoken_section.text,
            ).ratio()
        )
    return ratios


def _print_similarity_summary(
    label: str,
    script: EpisodeScript,
    spoken_sections: list[Any],
) -> None:
    ratios = _similarity_ratios(script, spoken_sections)
    if not ratios:
        print(f"{label} similarity: skipped")
        return
    print(f"{label} average similarity: {statistics.mean(ratios):.3f}")
    for chunk_name, chunk_mean in _chunk_stats(ratios):
        print(f"{label} {chunk_name}: {chunk_mean:.3f}")


def main() -> int:
    args = _parse_args()
    episode_input = _load_episode_input(args.run_dir.resolve(), args.episode)

    settings = Settings()
    llm = build_llm_client(settings)
    agent = SpokenDeliveryAgent(
        llm,
        max_retry_attempts=args.spoken_max_retry_attempts,
    )
    if args.prompt_file is not None:
        agent.instructions = args.prompt_file.read_text(encoding="utf-8").strip()

    rewritten_sections: list[SpokenSection] = []
    previous_spoken_tail: str | None = None
    for prose_section in episode_input.script.prose_sections:
        payload = agent.build_payload(
            episode_number=episode_input.episode_number,
            section={
                "section_id": prose_section.section_id,
                "purpose": "",
                "anchor": "",
                "closure_mode": "",
                "movement_goal": prose_section.movement_goal,
                "text": prose_section.text,
            },
            max_words_per_segment=episode_input.pipeline_config.spoken_chunk_max_words,
            tts_provider=episode_input.tts_provider,
            previous_spoken_tail=previous_spoken_tail,
        )
        result = agent.run(payload)
        rewritten_sections.append(
            SpokenSection(
                section_id=prose_section.section_id,
                text=result.text,
                speech_hints=result.speech_hints,
            )
        )
        previous_spoken_tail = result.text.strip() or None
    spoken_script = SpokenScript(
        episode_number=episode_input.episode_number,
        title=episode_input.script.title,
        framing=episode_input.script.framing,
        sections=rewritten_sections,
        tts_provider=episode_input.tts_provider,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else _default_output_path(episode_input.run_dir, episode_input.episode_number)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(output_path, spoken_script)

    print(f"wrote {output_path}")
    print(f"episode: {episode_input.episode_number} {episode_input.script.title}")
    print(f"sections: {len(spoken_script.sections)}")
    _print_similarity_summary("new", episode_input.script, spoken_script.sections)
    if episode_input.existing_spoken_script is not None:
        _print_similarity_summary(
            "existing",
            episode_input.script,
            episode_input.existing_spoken_script.sections,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
