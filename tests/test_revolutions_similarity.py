from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.revolutions_similarity import (
    extract_feature_vector,
    load_episode_body_text,
    score_run_dir,
    score_text_against_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_revolutions_similarity.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_run(tmp_path: Path, *, intro_text: str, body_text: str) -> Path:
    run_dir = tmp_path / "runs" / "demo-run"
    episode_dir = run_dir / "episodes" / "1"
    episode_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        episode_dir / "episode_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": "Question",
                "handoff_scene_card_id": "scene_1",
            },
            "prose_sections": [
                {
                    "section_id": "intro",
                    "scene_card_ids": ["scene_1"],
                    "movement_goal": "orient",
                    "text": intro_text,
                },
                {
                    "section_id": "body",
                    "scene_card_ids": ["scene_2"],
                    "movement_goal": "explain",
                    "text": body_text,
                },
            ],
            "total_word_count": 10,
            "estimated_duration_seconds": 10,
        },
    )
    return run_dir


def test_score_text_against_benchmark_is_perfect_for_identical_text() -> None:
    benchmark_text = (
        "Now we left the west. In plain English, the problem was simple. "
        "That same logic returns here, which meant the result was clear."
    )
    benchmark = extract_feature_vector(benchmark_text)

    score, feature_scores, _ = score_text_against_benchmark(benchmark_text, benchmark)

    assert score == 100.0
    assert all(value == 100.0 for value in feature_scores.values())


def test_load_episode_body_text_ignores_intro_section_by_default(
    tmp_path: Path,
) -> None:
    benchmark_text = "Now we left the west. In plain English, the result was clear."
    run_dir = _build_run(
        tmp_path,
        intro_text="This episode tonight this series this hour.",
        body_text=benchmark_text,
    )
    script_path = run_dir / "episodes" / "1" / "episode_script.json"

    episode_number, title, body_text = load_episode_body_text(script_path)

    assert episode_number == 1
    assert title == "Episode One"
    assert body_text == benchmark_text


def test_score_run_dir_ignores_intro_section_by_default(tmp_path: Path) -> None:
    benchmark_text = "Now we left the west. In plain English, the result was clear."
    run_dir = _build_run(
        tmp_path,
        intro_text="This episode tonight this series this hour.",
        body_text=benchmark_text,
    )

    result = score_run_dir(run_dir, benchmark_text)

    assert result.episode_count == 1
    assert result.mean_episode_score == 100.0
    assert result.weighted_episode_score == 100.0
    assert result.corpus_score == 100.0


def test_score_revolutions_similarity_script_outputs_json(tmp_path: Path) -> None:
    benchmark_text = "Now we left the west. In plain English, the result was clear."
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text(benchmark_text, encoding="utf-8")
    run_dir = _build_run(
        tmp_path,
        intro_text="This episode tonight this series this hour.",
        body_text=benchmark_text,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-file",
            str(benchmark_path),
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ignore_intro_section"] is True
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["mean_episode_score"] == 100.0
