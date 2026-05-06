from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.host_presence_review import build_review_payload, extract_host_snippets

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_host_presence_review.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "demo-run"
    episode_dir = run_dir / "episodes" / "1"
    episode_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "series_plan.json",
        {
            "episodes": [
                {
                    "episode_number": 1,
                    "scene_cards": [
                        {
                            "scene_id": "sc01_intro",
                            "section_id": "s01_open",
                            "title": "Opening frame",
                            "host_moves": {
                                "open": [
                                    {
                                        "move_type": "orient",
                                        "note": "Listen for the turn in the room and tell the listener what kind of story this is.",
                                        "address_mode": "you",
                                    }
                                ],
                                "pivot": [],
                                "close": [],
                            },
                        },
                        {
                            "scene_id": "sc02_body",
                            "section_id": "s01_open",
                            "title": "Body scene",
                            "host_moves": {
                                "open": [],
                                "pivot": [
                                    {
                                        "move_type": "clarify",
                                        "note": "Sharpen the pressure once the bodies are moving.",
                                    }
                                ],
                                "close": [],
                            },
                        },
                    ],
                }
            ]
        },
    )

    _write_json(
        episode_dir / "host_moves_script_diagnostics.json",
        {
            "phase_trace": [
                {
                    "scene_id": "sc01_intro",
                    "section_id": "s01_open",
                    "phase": "open",
                    "cue_count": 1,
                    "move_types": ["orient"],
                    "host_note": "Listen for the turn in the room and tell the listener what kind of story this is.",
                    "address_modes": ["you"],
                    "approx_realized": True,
                    "first_person_plural_present": False,
                }
            ]
        },
    )

    _write_json(
        episode_dir / "episode_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": "Question",
                "handoff_scene_card_id": "sc01_intro",
            },
            "prose_sections": [
                {
                    "section_id": "s01_open",
                    "scene_card_ids": ["sc01_intro", "sc02_body"],
                    "movement_goal": "orient",
                    "text": (
                        "Listen for the turn in the room. "
                        "This is the kind of story that looks local until you see the whole structure. "
                        "Then the crowd starts moving."
                    ),
                }
            ],
            "total_word_count": 20,
            "estimated_duration_seconds": 30,
        },
    )
    return run_dir


def test_extract_host_snippets_finds_expected_sentence(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    entries = extract_host_snippets(run_dir)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.highlight_text == "Listen for the turn in the room."
    assert entry.scene_title == "Opening frame"
    assert "imperative" in entry.cue_tags
    assert entry.phase == "open"
    assert entry.move_types == ["orient"]
    assert entry.script_artifact == "episode_script.json"


def test_build_review_payload_counts_entries(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    payload = build_review_payload([run_dir], title="Demo Review")

    assert payload["title"] == "Demo Review"
    assert payload["summary"]["total_entries"] == 1
    assert payload["summary"]["run_counts"] == {"demo-run": 1}
    assert payload["entries"][0]["highlight_text"] == "Listen for the turn in the room."


def test_build_host_presence_review_script_writes_outputs(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    output_html = tmp_path / "review.html"
    output_json = tmp_path / "review.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Demo Review",
            "--output-html",
            str(output_html),
            "--output-json",
            str(output_json),
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_html.exists()
    assert output_json.exists()
    assert "Demo Review" in output_html.read_text(encoding="utf-8")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["total_entries"] == 1
    assert payload["entries"][0]["scene_id"] == "sc01_intro"
