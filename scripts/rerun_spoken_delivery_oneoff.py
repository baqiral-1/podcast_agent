#!/usr/bin/env python3
"""Rerun spoken delivery from logged payloads and regenerate render manifests.

This one-off utility is intentionally strict:
- processes runs sequentially in the given order
- processes episodes in bounded parallelism per run
- fails fast on first error
- uses current spoken-delivery prompt from code
- uses per-episode payload from run.log (latest spoken_delivery llm_request)
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _spoken_delivery_batch_section_id,
    _save_json,
    build_render_manifest,
)
from podcast_agent.schemas.models import (
    EpisodeScript,
    RenderManifest,
    SpokenScript,
    SpokenSection,
)


DEFAULT_PROJECT_IDS = [
    "mughal_v9",
    "war_on_terror_29",
    "palestine_v3",
    "iranian_revolution_v16",
    "independence_v1",
]


@dataclass(frozen=True)
class RenderSettings:
    voice_id: str
    speed: float
    base_instructions: str | None
    words_per_minute: int


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_dirs(project_dir: Path) -> list[Path]:
    episodes_dir = project_dir / "episodes"
    if not episodes_dir.exists() or not episodes_dir.is_dir():
        raise RuntimeError(f"Missing episodes directory: {episodes_dir}")
    episode_dirs = sorted(
        [
            path
            for path in episodes_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=lambda path: int(path.name),
    )
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {episodes_dir}")
    return episode_dirs


def _extract_spoken_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("event_type") != "llm_request":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_name") != "spoken_delivery":
        return None
    user_text = payload.get("user_text")
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    wrapper = json.loads(user_text)
    if not isinstance(wrapper, dict):
        return None
    model_payload = wrapper.get("payload")
    if not isinstance(model_payload, dict):
        return None
    return model_payload


def _collect_logged_spoken_payloads(log_path: Path) -> dict[int, list[dict[str, Any]]]:
    payloads_by_episode: dict[int, list[dict[str, Any]]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        model_payload = _extract_spoken_payload(event)
        if model_payload is None:
            continue
        episode_number = model_payload.get("episode_number")
        if not isinstance(episode_number, int) or episode_number < 1:
            raise RuntimeError(
                f"Invalid spoken payload episode_number in log {log_path}: {episode_number!r}"
            )
        for required_field in (
            "episode_number",
            "script",
            "max_words_per_segment",
            "tts_provider",
        ):
            if required_field not in model_payload:
                raise RuntimeError(
                    f"Missing '{required_field}' in spoken payload for episode {episode_number}"
                )
        payloads_by_episode.setdefault(episode_number, []).append(model_payload)
    return payloads_by_episode


def _execute_in_bounded_parallel(
    item_ids: list[int],
    *,
    max_workers: int,
    worker: Callable[[int], Any],
) -> list[tuple[int, Any]]:
    if max_workers < 1:
        raise RuntimeError(f"max_workers must be >= 1, got {max_workers}")
    if not item_ids:
        return []

    results: list[tuple[int, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        remaining = iter(item_ids)
        in_flight: dict[Future[Any], int] = {}

        def _submit_next() -> bool:
            try:
                item_id = next(remaining)
            except StopIteration:
                return False
            future = executor.submit(worker, item_id)
            in_flight[future] = item_id
            return True

        for _ in range(min(max_workers, len(item_ids))):
            _submit_next()

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                item_id = in_flight.pop(future)
                try:
                    value = future.result()
                except Exception:
                    for pending in in_flight:
                        pending.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                results.append((item_id, value))
                _submit_next()

    return results


def _manifest_word_count(manifest: RenderManifest) -> int:
    return sum(len(segment.text.split()) for segment in manifest.segments)


def _infer_words_per_minute(manifests: list[RenderManifest]) -> int:
    candidates = set(range(1, 1001))
    for manifest in manifests:
        words = _manifest_word_count(manifest)
        est = manifest.estimated_duration_seconds
        if words <= 0:
            continue
        if est <= 0:
            raise RuntimeError(
                "Cannot infer words_per_minute: encountered non-positive estimated_duration_seconds"
            )
        valid = {
            w
            for w in candidates
            if (words * 60.0) / (est + 1) < w <= (words * 60.0) / est
        }
        candidates = valid
        if not candidates:
            raise RuntimeError(
                "Cannot infer words_per_minute: no integer satisfies manifest duration constraints"
            )
    if len(candidates) != 1:
        preview = sorted(candidates)
        raise RuntimeError(
            "Cannot infer words_per_minute uniquely; candidates="
            f"{preview[:12]}{'...' if len(preview) > 12 else ''}"
        )
    return next(iter(candidates))


def _extract_base_instruction_prefix(instruction: str) -> str:
    marker = " Segment guidance:"
    normalized = instruction.strip()
    if marker in normalized:
        return normalized.split(marker, 1)[0].strip()
    return normalized


def _infer_render_settings(manifests: list[RenderManifest]) -> RenderSettings:
    if not manifests:
        raise RuntimeError("Cannot infer render settings from empty manifest list")

    voice_ids: set[str] = set()
    speeds: set[float] = set()
    base_instruction_prefixes: set[str] = set()

    for manifest in manifests:
        framing = next(
            (segment for segment in manifest.segments if segment.segment_id == "framing_opening_image"),
            None,
        )
        if framing is None:
            raise RuntimeError(
                "Cannot infer render settings: missing framing_opening_image segment"
            )
        voice_ids.add(framing.voice_id)
        speeds.add(float(framing.speed))

        for segment in manifest.segments:
            if segment.instructions is None:
                continue
            prefix = _extract_base_instruction_prefix(segment.instructions)
            if prefix:
                base_instruction_prefixes.add(prefix)

    if len(voice_ids) != 1:
        raise RuntimeError(f"Ambiguous voice_id values in manifests: {sorted(voice_ids)}")
    if len(speeds) != 1:
        raise RuntimeError(f"Ambiguous base speed values in manifests: {sorted(speeds)}")

    if len(base_instruction_prefixes) > 1:
        raise RuntimeError(
            "Ambiguous base instructions inferred from manifests: "
            f"{sorted(base_instruction_prefixes)}"
        )

    words_per_minute = _infer_words_per_minute(manifests)
    base_instructions = (
        next(iter(base_instruction_prefixes)) if base_instruction_prefixes else None
    )

    return RenderSettings(
        voice_id=next(iter(voice_ids)),
        speed=next(iter(speeds)),
        base_instructions=base_instructions,
        words_per_minute=words_per_minute,
    )


def _build_spoken_section(payload: dict[str, Any], result: Any) -> SpokenSection:
    episode_number = payload["episode_number"]
    if not isinstance(episode_number, int) or episode_number < 1:
        raise RuntimeError(f"Invalid episode_number in payload: {episode_number!r}")

    script = EpisodeScript.model_validate(payload["script"])
    if not script.prose_sections:
        raise RuntimeError(
            "Spoken delivery rerun expects at least one prose section per payload; "
            f"episode {episode_number} payload contained {len(script.prose_sections)} sections."
        )
    tts_provider = payload.get("tts_provider")
    if not isinstance(tts_provider, str) or not tts_provider.strip():
        raise RuntimeError(f"Invalid tts_provider in payload for episode {episode_number}")

    batch_index = payload.get("batch_index")
    if isinstance(batch_index, int) and batch_index >= 1:
        section_id = _spoken_delivery_batch_section_id(batch_index)
    elif len(script.prose_sections) == 1:
        section_id = script.prose_sections[0].section_id
    else:
        raise RuntimeError(
            "Spoken delivery rerun requires batch_index when a payload contains multiple prose sections; "
            f"episode {episode_number} payload contained {len(script.prose_sections)} sections."
        )
    return SpokenSection(
        section_id=section_id,
        text=result.text,
        speech_hints=result.speech_hints,
    )


def _build_spoken_script(
    episode_script: EpisodeScript,
    *,
    tts_provider: str,
    sections: list[SpokenSection],
) -> SpokenScript:
    if not isinstance(tts_provider, str) or not tts_provider.strip():
        raise RuntimeError(f"Invalid tts_provider for episode {episode_script.episode_number}")
    return SpokenScript(
        episode_number=episode_script.episode_number,
        title=episode_script.title,
        framing=episode_script.framing,
        sections=sections,
        tts_provider=tts_provider,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun spoken delivery from logged payloads using current prompt, "
            "then regenerate render manifests."
        )
    )
    parser.add_argument(
        "--project-id",
        action="append",
        default=None,
        help=(
            "Run directory name under artifact root. Repeat for multiple runs. "
            f"Default order: {', '.join(DEFAULT_PROJECT_IDS)}"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional artifact root override (defaults to settings.pipeline.artifact_root).",
    )
    parser.add_argument(
        "--episode-concurrency",
        type=int,
        default=5,
        help="Maximum number of episodes to process in parallel per run (default: 5).",
    )
    parser.add_argument(
        "--spoken-max-retry-attempts",
        type=int,
        default=1,
        help=(
            "Maximum spoken_delivery retry attempts per episode (default: 1). "
            "Use 1 to keep active requests bounded by --episode-concurrency."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.episode_concurrency < 1:
        raise RuntimeError(
            f"--episode-concurrency must be >= 1, got {args.episode_concurrency}"
        )
    if args.spoken_max_retry_attempts < 1:
        raise RuntimeError(
            "--spoken-max-retry-attempts must be >= 1, "
            f"got {args.spoken_max_retry_attempts}"
        )

    settings = Settings()
    if args.artifact_root:
        settings = settings.model_copy(
            update={
                "pipeline": settings.pipeline.model_copy(
                    update={"artifact_root": Path(args.artifact_root)}
                )
            }
        )

    project_ids = args.project_id if args.project_id else list(DEFAULT_PROJECT_IDS)
    artifact_root = settings.pipeline.artifact_root
    orchestrator = PipelineOrchestrator(settings)
    orchestrator.spoken_delivery_agent.max_retry_attempts = (
        args.spoken_max_retry_attempts
    )

    total_episodes = 0
    for project_index, project_id in enumerate(project_ids, start=1):
        project_dir = artifact_root / project_id
        if not project_dir.exists() or not project_dir.is_dir():
            raise RuntimeError(f"Run directory not found: {project_dir}")

        run_log = project_dir / "run.log"
        if not run_log.exists():
            raise RuntimeError(f"Missing run.log: {run_log}")

        episode_dirs = _episode_dirs(project_dir)
        payloads_by_episode = _collect_logged_spoken_payloads(run_log)
        manifests: list[RenderManifest] = []

        for episode_dir in episode_dirs:
            episode_number = int(episode_dir.name)
            for required_name in (
                "episode_script.json",
                "spoken_script.json",
                "render_manifest.json",
            ):
                required_path = episode_dir / required_name
                if not required_path.exists():
                    raise RuntimeError(f"Missing required file: {required_path}")
            if episode_number not in payloads_by_episode:
                raise RuntimeError(
                    f"No spoken_delivery payload in run.log for {project_id} episode {episode_number}"
                )
            manifest = RenderManifest.model_validate(
                _load_json(episode_dir / "render_manifest.json")
            )
            manifests.append(manifest)

        render_settings = _infer_render_settings(manifests)

        print(
            f"[{project_index}/{len(project_ids)}] {project_id}: "
            f"episodes={len(episode_dirs)} voice={render_settings.voice_id} "
            f"speed={render_settings.speed} wpm={render_settings.words_per_minute} "
            f"episode_concurrency={args.episode_concurrency} "
            f"spoken_max_retry_attempts={args.spoken_max_retry_attempts}"
        )

        orchestrator._bind_run_logger(project_dir)

        episode_dir_by_number = {
            int(episode_dir.name): episode_dir for episode_dir in episode_dirs
        }

        def _process_episode(episode_number: int) -> int:
            episode_dir = episode_dir_by_number[episode_number]
            episode_script = EpisodeScript.model_validate(
                _load_json(episode_dir / "episode_script.json")
            )
            section_payloads = payloads_by_episode[episode_number]
            payload_records: list[tuple[int, list[str], dict[str, Any]]] = []
            tts_provider: str | None = None
            for payload_order, payload in enumerate(section_payloads, start=1):
                if payload["episode_number"] != episode_number:
                    raise RuntimeError(
                        f"Episode mismatch in payload for {project_id} episode {episode_number}"
                    )
                payload_script = EpisodeScript.model_validate(payload["script"])
                if not payload_script.prose_sections:
                    raise RuntimeError(
                        "Spoken delivery rerun expects at least one prose section per payload; "
                        f"episode {episode_number} payload contained {len(payload_script.prose_sections)} sections."
                    )
                batch_index = payload.get("batch_index")
                if batch_index is None:
                    batch_sort_key = payload_order
                elif isinstance(batch_index, int) and batch_index >= 1:
                    batch_sort_key = batch_index
                else:
                    raise RuntimeError(
                        f"Invalid batch_index in payload for episode {episode_number}: {batch_index!r}"
                    )
                payload_records.append((
                    batch_sort_key,
                    [section.section_id for section in payload_script.prose_sections],
                    payload,
                ))
                candidate_provider = payload.get("tts_provider")
                if not isinstance(candidate_provider, str) or not candidate_provider.strip():
                    raise RuntimeError(
                        f"Invalid tts_provider in payload for episode {episode_number}"
                    )
                if tts_provider is None:
                    tts_provider = candidate_provider
                elif tts_provider != candidate_provider:
                    raise RuntimeError(
                        f"Conflicting tts_provider values in payloads for episode {episode_number}"
                    )

            payload_records.sort(key=lambda record: record[0])
            covered_section_ids = [
                section_id
                for _, section_ids, _ in payload_records
                for section_id in section_ids
            ]
            expected_section_ids = [
                section.section_id for section in episode_script.prose_sections
            ]
            if covered_section_ids != expected_section_ids:
                raise RuntimeError(
                    "Spoken delivery payloads do not cover the episode script contiguously for episode "
                    f"{episode_number}: expected {expected_section_ids}, got {covered_section_ids}"
                )

            rewritten_sections: list[SpokenSection] = []
            for _, _, payload in payload_records:
                result = orchestrator.spoken_delivery_agent.run(payload)
                rewritten_sections.append(_build_spoken_section(payload, result))

            spoken_script = _build_spoken_script(
                episode_script,
                tts_provider=tts_provider or render_settings.voice_id,
                sections=rewritten_sections,
            )
            _save_json(episode_dir / "spoken_script.json", spoken_script)

            render_manifest = build_render_manifest(
                spoken_script,
                voice_id=render_settings.voice_id,
                speed=render_settings.speed,
                words_per_minute=render_settings.words_per_minute,
                base_instructions=render_settings.base_instructions,
            )
            _save_json(episode_dir / "render_manifest.json", render_manifest)
            return episode_number

        episode_numbers = sorted(episode_dir_by_number)
        completed_results = _execute_in_bounded_parallel(
            episode_numbers,
            max_workers=args.episode_concurrency,
            worker=_process_episode,
        )

        completed = 0
        for completed_episode, _ in completed_results:
            completed += 1
            total_episodes += 1
            print(
                f"  - [{completed}/{len(episode_numbers)}] episode {completed_episode}: "
                "spoken_script + render_manifest rewritten"
            )

    print(
        f"Completed spoken-delivery rerun for {len(project_ids)} run(s), "
        f"{total_episodes} episode(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
