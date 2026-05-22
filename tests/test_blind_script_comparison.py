from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.blind_script_comparison import (
    build_blind_comparison_payload,
    render_blind_comparison_html,
    write_blind_comparison_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_blind_script_comparison.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _episode_text(run_name: str, episode_number: int) -> str:
    paragraphs: list[str] = []
    for paragraph_number in range(1, 10):
        paragraphs.append(
            (
                f"{run_name} episode {episode_number} paragraph {paragraph_number} keeps the host in the room, "
                "guides the listener through the scene, names the tension, and stays concrete enough "
                "to sound spoken instead of archival."
            )
        )
    return "\n\n".join(paragraphs)


def _build_run(tmp_path: Path, run_name: str) -> Path:
    run_dir = tmp_path / "runs" / run_name
    for episode_number in range(1, 5):
        episode_dir = run_dir / "episodes" / str(episode_number)
        episode_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            episode_dir / "episode_script.json",
            {
                "episode_number": episode_number,
                "title": f"Episode {episode_number}",
                "prose_sections": [
                    {
                        "section_id": f"sec-{episode_number:02d}",
                        "scene_card_ids": [],
                        "movement_goal": "host_presence",
                        "text": _episode_text(run_name, episode_number),
                    }
                ],
            },
        )
    return run_dir


def test_build_blind_comparison_payload_creates_expected_pairs(tmp_path: Path) -> None:
    left_run = _build_run(tmp_path, "iranian_revolution_v43")
    right_run = _build_run(tmp_path, "iranian_revolution_v44")

    payload = build_blind_comparison_payload(
        [left_run, right_run],
        title="Blind Comparison",
        counts_by_episode=(2, 2, 1, 1),
    )

    assert payload["title"] == "Blind Comparison"
    assert payload["episode_numbers"] == [1, 2, 3, 4]
    assert payload["total_comparisons"] == 6
    assert len(payload["comparisons"]) == 6
    for comparison in payload["comparisons"]:
        assert {comparison["left_run_id"], comparison["right_run_id"]} == {
            "iranian_revolution_v43",
            "iranian_revolution_v44",
        }
        assert comparison["left_source"]["episode_number"] == comparison["episode_number"]
        assert comparison["right_source"]["episode_number"] == comparison["episode_number"]


def test_render_blind_comparison_html_embeds_payload(tmp_path: Path) -> None:
    left_run = _build_run(tmp_path, "iranian_revolution_v43")
    right_run = _build_run(tmp_path, "iranian_revolution_v44")
    payload = build_blind_comparison_payload(
        [left_run, right_run],
        title="Blind <Comparison>",
        subtitle="Review both options",
        counts_by_episode=(1, 1, 1, 1),
    )

    html = render_blind_comparison_html(payload)

    assert "Blind &lt;Comparison&gt;" in html
    assert "comparison-payload" in html
    assert "blind-script-comparison::" in html


def test_write_blind_comparison_outputs_writes_files(tmp_path: Path) -> None:
    left_run = _build_run(tmp_path, "iranian_revolution_v43")
    right_run = _build_run(tmp_path, "iranian_revolution_v44")
    output_html = tmp_path / "blind-comparison.html"
    output_json = tmp_path / "blind-comparison.json"

    payload = write_blind_comparison_outputs(
        run_dirs=[left_run, right_run],
        output_html=output_html,
        output_json=output_json,
        title="Blind Comparison",
        counts_by_episode=(1, 1, 1, 1),
    )

    assert payload["total_comparisons"] == 4
    assert output_html.exists()
    assert output_json.exists()
    assert "Blind Comparison" in output_html.read_text(encoding="utf-8")


def test_build_blind_script_comparison_script_writes_outputs(tmp_path: Path) -> None:
    left_run = _build_run(tmp_path, "iranian_revolution_v43")
    right_run = _build_run(tmp_path, "iranian_revolution_v44")
    output_html = tmp_path / "comparison.html"
    output_json = tmp_path / "comparison.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Blind Comparison",
            "--counts-by-episode",
            "2,2,1,1",
            "--output-html",
            str(output_html),
            "--output-json",
            str(output_json),
            str(left_run),
            str(right_run),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_html.exists()
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["title"] == "Blind Comparison"
    assert payload["total_comparisons"] == 6
