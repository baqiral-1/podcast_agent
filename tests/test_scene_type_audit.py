from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.scene_type_audit import (
    build_scene_type_audit_payload,
    render_scene_type_audit_html,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_scene_type_audit.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "demo-scene-audit"
    scene_cards = [
        {
            "scene_id": "sc_open_1",
            "section_id": "sec_open",
            "title": "A square fills at dawn",
            "scene_role": "action",
            "scene_job": "opening",
            "beat_change": "Crowds enter before the state can react.",
        },
        {
            "scene_id": "sc_open_2",
            "section_id": "sec_open",
            "title": "The strike shuts the ministry",
            "scene_role": "fallout",
            "scene_job": "build",
            "beat_change": "The action becomes an institutional consequence.",
        },
        {
            "scene_id": "sc_open_3",
            "section_id": "sec_open",
            "title": "The regime loses initiative",
            "scene_role": "implication",
            "scene_job": "turn",
            "beat_change": "The listener gets the meaning after the event.",
        },
        {
            "scene_id": "sc_flat_1",
            "section_id": "sec_flat",
            "title": "The planners' memo",
            "scene_role": "context_setup",
            "scene_job": "build",
            "beat_change": "The section opens by naming institutions and assumptions.",
        },
        {
            "scene_id": "sc_flat_2",
            "section_id": "sec_flat",
            "title": "The theory of the coalition",
            "scene_role": "implication",
            "scene_job": "build",
            "beat_change": "The prose shifts toward interpretation quickly.",
        },
        {
            "scene_id": "sc_flat_3",
            "section_id": "sec_flat",
            "title": "The theory hardens",
            "scene_role": "implication",
            "scene_job": "turn",
            "beat_change": "The section doubles down on argument without a fresh event beat.",
        },
        {
            "scene_id": "sc_close_1",
            "section_id": "sec_close",
            "title": "The inheritance remains",
            "scene_role": "implication",
            "scene_job": "close",
            "beat_change": "The episode closes by naming what persists.",
        },
    ]

    _write_json(
        run_dir / "series_plan.json",
        {
            "episodes": [
                {
                    "episode_number": 1,
                    "scene_cards": scene_cards,
                }
            ]
        },
    )
    _write_json(
        run_dir / "episodes" / "1" / "style_audited_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "prose_sections": [
                {
                    "section_id": "sec_open",
                    "scene_card_ids": ["sc_open_1", "sc_open_2", "sc_open_3"],
                    "movement_goal": "Move from street action into institutional consequence.",
                    "text": (
                        "At dawn the square fills before the police have a plan. By mid-morning the "
                        "strike has reached the ministry and the desks are empty.\n\nThat shift matters "
                        "because the regime is now reacting to the crowd rather than directing it."
                    ),
                },
                {
                    "section_id": "sec_flat",
                    "scene_card_ids": ["sc_flat_1", "sc_flat_2", "sc_flat_3"],
                    "movement_goal": "Explain why the coalition should matter.",
                    "text": (
                        "The memo names the institutions, the committees, and the assumptions behind the alliance.\n\n"
                        "From there the prose turns toward theory, and then theory again, without returning to a decisive "
                        "public act or confrontation."
                    ),
                },
                {
                    "section_id": "sec_close",
                    "scene_card_ids": ["sc_close_1"],
                    "movement_goal": "Leave the listener with the residue.",
                    "text": "What remains is the inheritance that the next episode must pick up.",
                },
            ],
        },
    )
    _write_json(
        run_dir / "episodes" / "1" / "spine_diagnostics.json",
        {
            "scene_order_preserved": True,
            "ending_alignment_pass": True,
            "new_load_bearing_question_detected": False,
            "second_ending_detected": False,
            "spine_drift_detected": False,
            "failure_labels": [],
        },
    )
    _write_json(
        run_dir / "episodes" / "1" / "continuity_script_diagnostics.json",
        {
            "episode_number": 1,
            "stage": "script",
            "warning_labels": [],
            "missed_item_ids": [],
        },
    )
    _write_json(
        run_dir / "episodes" / "1" / "host_moves_script_diagnostics.json",
        {
            "sections_with_host_phase_collapse": [],
            "sections_with_editorial_host_target_pressure": ["sec_flat"],
            "approx_unrealized_phase_ids": [],
        },
    )
    _write_json(
        run_dir / "episodes" / "1" / "style_audit_result.json",
        {
            "episode_number": 1,
            "episode_warnings": [],
            "sections": [
                {"section_id": "sec_open", "warnings": []},
                {"section_id": "sec_flat", "warnings": []},
                {"section_id": "sec_close", "warnings": []},
            ],
        },
    )
    return run_dir


def test_build_scene_type_audit_payload_classifies_sections(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    payload = build_scene_type_audit_payload(run_dir)

    assert payload["summary"]["episode_count"] == 1
    assert payload["summary"]["section_count"] == 3
    assert payload["summary"]["scene_card_count"] == 7

    sections = payload["episodes"][0]["sections"]
    verdicts = {section["section_id"]: section["verdict"] for section in sections}
    assert verdicts["sec_open"] in {"strong", "solid"}
    assert verdicts["sec_flat"] == "weak"
    assert verdicts["sec_close"] == "close-only"

    weak_sections = payload["summary"]["weak_sections"]
    assert weak_sections[0]["section_id"] == "sec_flat"


def test_render_scene_type_audit_html_includes_key_content(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    payload = build_scene_type_audit_payload(run_dir)
    html = render_scene_type_audit_html(payload)

    assert "Scene Type to Prose Audit" in html
    assert "Episode 1: Episode One" in html
    assert "sec_open" in html
    assert "sec_flat" in html
    assert "close-only" in html


def test_generate_scene_type_audit_script_writes_output(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    output_path = run_dir / "scene-type-audit.html"

    subprocess.run([sys.executable, str(SCRIPT), str(run_dir)], check=True)

    html = output_path.read_text(encoding="utf-8")
    assert "Scene Type to Prose Audit" in html
    assert "Episode 1: Episode One" in html
    assert "sec_close" in html
