from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from podcast_agent.schemas.models import RenderManifest, RenderSegment


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "rerun_spoken_delivery_oneoff.py"


spec = importlib.util.spec_from_file_location(
    "rerun_spoken_delivery_oneoff",
    SCRIPT_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load script module: {SCRIPT_PATH}")

script_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script_module
spec.loader.exec_module(script_module)


def _spoken_event(payload: dict) -> dict:
    return {
        "timestamp": "2026-04-12T00:00:00Z",
        "event_type": "llm_request",
        "payload": {
            "schema_name": "spoken_delivery",
            "user_text": json.dumps(
                {
                    "schema_name": "spoken_delivery",
                    "payload": payload,
                    "expected_schema": {},
                }
            ),
        },
    }


def test_collect_logged_spoken_payloads_uses_latest_per_episode(tmp_path: Path):
    log_path = tmp_path / "run.log"

    episode_one_old = {
        "episode_number": 1,
        "script": {"title": "old"},
        "max_words_per_segment": 200,
        "tts_provider": "openai-compatible",
    }
    episode_one_new = {
        "episode_number": 1,
        "script": {"title": "new"},
        "max_words_per_segment": 250,
        "tts_provider": "openai-compatible",
    }
    episode_two = {
        "episode_number": 2,
        "script": {"title": "two"},
        "max_words_per_segment": 250,
        "tts_provider": "openai-compatible",
    }

    events = [
        {"event_type": "stage_start", "payload": {"stage": "x"}},
        _spoken_event(episode_one_old),
        _spoken_event(episode_two),
        _spoken_event(episode_one_new),
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    payloads = script_module._collect_logged_spoken_payloads(log_path)
    assert sorted(payloads.keys()) == [1, 2]
    assert payloads[1]["script"]["title"] == "new"
    assert payloads[2]["script"]["title"] == "two"


def _make_manifest(
    *,
    episode_number: int,
    total_words: int,
    estimated_duration_seconds: int,
    instruction: str,
) -> RenderManifest:
    if total_words < 2:
        raise ValueError("total_words must be >= 2")

    section_words = total_words - 1
    section_text = " ".join(["word"] * section_words)
    return RenderManifest(
        episode_number=episode_number,
        segments=[
            RenderSegment(
                segment_id="framing_opening_image",
                text="opening",
                voice_id="ballad",
                speed=1.0,
                pause_before_ms=0,
                pause_after_ms=900,
            ),
            RenderSegment(
                segment_id="sec_1",
                text=section_text,
                voice_id="ballad",
                speed=1.0,
                pause_before_ms=300,
                pause_after_ms=300,
                instructions=instruction,
            ),
        ],
        total_segments=2,
        estimated_duration_seconds=estimated_duration_seconds,
    )


def test_infer_words_per_minute_unique():
    manifests = [
        _make_manifest(
            episode_number=1,
            total_words=281,
            estimated_duration_seconds=120,
            instruction="Narrate clearly. Segment guidance: Keep it measured.",
        ),
        _make_manifest(
            episode_number=2,
            total_words=421,
            estimated_duration_seconds=180,
            instruction="Narrate clearly. Segment guidance: Keep it steady.",
        ),
    ]

    assert script_module._infer_words_per_minute(manifests) == 140


def test_infer_words_per_minute_raises_when_not_unique():
    manifests = [
        _make_manifest(
            episode_number=1,
            total_words=140,
            estimated_duration_seconds=60,
            instruction="Narrate clearly. Segment guidance: Keep it measured.",
        )
    ]

    with pytest.raises(RuntimeError, match="uniquely"):
        script_module._infer_words_per_minute(manifests)


def test_infer_render_settings_uses_framing_and_instruction_prefix():
    manifests = [
        _make_manifest(
            episode_number=1,
            total_words=281,
            estimated_duration_seconds=120,
            instruction="Narrate clearly. Segment guidance: Keep it measured.",
        ),
        _make_manifest(
            episode_number=2,
            total_words=421,
            estimated_duration_seconds=180,
            instruction="Narrate clearly. Segment guidance: Keep it urgent.",
        ),
    ]

    settings = script_module._infer_render_settings(manifests)

    assert settings.voice_id == "ballad"
    assert settings.speed == 1.0
    assert settings.base_instructions == "Narrate clearly."
    assert settings.words_per_minute == 140


def test_parse_args_defaults_episode_concurrency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rerun_spoken_delivery_oneoff.py"],
    )
    args = script_module._parse_args()
    assert args.episode_concurrency == 5


def test_parse_args_accepts_episode_concurrency_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rerun_spoken_delivery_oneoff.py", "--episode-concurrency", "3"],
    )
    args = script_module._parse_args()
    assert args.episode_concurrency == 3


def test_parse_args_defaults_spoken_max_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rerun_spoken_delivery_oneoff.py"],
    )
    args = script_module._parse_args()
    assert args.spoken_max_retry_attempts == 1


def test_parse_args_accepts_spoken_max_retry_attempts_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rerun_spoken_delivery_oneoff.py",
            "--spoken-max-retry-attempts",
            "2",
        ],
    )
    args = script_module._parse_args()
    assert args.spoken_max_retry_attempts == 2


def test_execute_in_bounded_parallel_respects_max_workers():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def worker(item: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return item * 10

    results = script_module._execute_in_bounded_parallel(
        list(range(1, 13)),
        max_workers=3,
        worker=worker,
    )
    assert len(results) == 12
    assert max_active <= 3


def test_execute_in_bounded_parallel_fail_fast_stops_new_scheduling():
    started: list[int] = []
    lock = threading.Lock()

    def worker(item: int) -> int:
        with lock:
            started.append(item)
        if item == 1:
            # Keep item 1 running while item 2 fails so no additional tasks are scheduled.
            time.sleep(0.2)
            return item
        if item == 2:
            raise RuntimeError("boom")
        return item

    with pytest.raises(RuntimeError, match="boom"):
        script_module._execute_in_bounded_parallel(
            [1, 2, 3, 4, 5],
            max_workers=2,
            worker=worker,
        )

    assert set(started).issubset({1, 2})
