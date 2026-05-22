from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.actor_explanation_audit import (
    build_actor_explanation_audit_payload,
    render_actor_explanation_audit_html,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_actor_explanation_audit.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_run(tmp_path: Path, *, late_intro: bool) -> Path:
    run_dir = tmp_path / "runs" / "demo-run"
    _write_json(
        run_dir / "actor_metadata.json",
        {
            "project_id": "demo-run",
            "actors": [
                {
                    "actor_id": "mossadeq",
                    "display_name": "Mohammad Mossadeq",
                    "actor_type": "person",
                    "aliases": ["Mossadeq"],
                }
            ],
        },
    )
    _write_json(
        run_dir / "narrative_strategy.json",
        {
            "strategy_type": "chronological",
            "justification": "One actor carries the turn.",
            "series_arc": "Arc",
            "series_actor_explanation_registry": [
                {
                    "actor_id": "mossadeq",
                    "introduction_episode_number": 1,
                    "first_background_depth": "appositive",
                    "later_episode_policy": "brief_reminder",
                }
            ],
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode One",
                    "authorial_contract": {
                        "introduce_actor_ids": ["mossadeq"],
                        "remind_actor_ids": [],
                    },
                }
            ],
        },
    )
    _write_json(
        run_dir / "episode_architectures.json",
        {
            "episodes": [
                {
                    "episode_number": 1,
                    "sections": [
                        {
                            "section_id": "s1",
                            "section_anchor": "The cabinet room",
                            "actor_explanations": [
                                {
                                    "actor_id": "mossadeq",
                                    "stage": "introduce",
                                    "background_depth": "appositive",
                                    "role_label": "prime minister and constitutional nationalist",
                                    "source_primitive_ids": ["primitive_1"],
                                    "source_passage_ids": ["passage_1"],
                                    "intro_facts": [
                                        "Mossadeq is prime minister.",
                                        "Oil nationalization is the live issue.",
                                    ],
                                    "why_now": "This is the listener's first concrete encounter with him in the crisis.",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )
    scene_cards = [
        {
            "scene_id": "scene_1",
            "section_id": "s1",
            "title": "Opening mention",
            "actors": [
                {
                    "name": "Mohammad Mossadeq",
                    "actor_id": "mossadeq",
                    "presence": "secondary",
                }
            ],
        },
        {
            "scene_id": "scene_2",
            "section_id": "s1",
            "title": "Tagged intro",
            "actors": [
                {
                    "name": "Mohammad Mossadeq",
                    "actor_id": "mossadeq",
                    "presence": "primary",
                    "explanation_stage": "introduce",
                    "background_depth": "appositive",
                    "role_label": "prime minister and constitutional nationalist",
                    "source_primitive_ids": ["primitive_1"],
                    "source_passage_ids": ["passage_1"],
                    "intro_facts": ["Mossadeq is prime minister."],
                    "why_now": "This is the listener's first concrete encounter with him in the crisis.",
                }
            ],
        },
    ]
    if not late_intro:
        scene_cards = [scene_cards[1]]
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
        run_dir / "episodes" / "1" / "episode_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "prose_sections": [
                {
                    "section_id": "s1",
                    "scene_card_ids": [scene["scene_id"] for scene in scene_cards],
                    "movement_goal": "discover",
                    "text": (
                        "Mohammad Mossadeq, the nationalist prime minister driving oil nationalization, enters the chamber. "
                        "The cabinet now has a public face."
                    ),
                    "actor_explanation_realizations": [
                        {
                            "actor_id": "mossadeq",
                            "scene_card_id": "scene_2" if late_intro else "scene_2",
                            "stage": "introduce",
                            "text_span": "Mohammad Mossadeq, the nationalist prime minister driving oil nationalization, enters the chamber.",
                            "source_passage_ids": ["passage_1"],
                            "realized_facts": ["Mossadeq is prime minister."],
                        }
                    ],
                }
            ],
        },
    )
    return run_dir


def test_build_actor_explanation_audit_payload_detects_late_intro(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, late_intro=True)

    payload = build_actor_explanation_audit_payload(run_dir)

    assert payload["summary"]["registry_actor_count"] == 1
    assert payload["summary"]["obligation_count"] == 1
    assert payload["summary"]["obligations_with_architecture"] == 1
    assert payload["summary"]["obligations_with_plan"] == 1
    assert payload["summary"]["obligations_with_script_hit"] == 1
    assert any(issue["kind"] == "late_intro_in_plan" for issue in payload["issues"])
    audit = payload["actors"][0]["obligations"][0]["script_audits"][0]
    assert audit["status"] == "mapped_section_realization"
    assert audit["source_passage_ids"] == ["passage_1"]
    assert "nationalist" in audit["snippet"]


def test_generate_actor_explanation_audit_html(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, late_intro=False)
    output_path = run_dir / "actor-explanation-audit.html"

    subprocess.run([sys.executable, str(SCRIPT), str(run_dir)], check=True)

    html = output_path.read_text(encoding="utf-8")
    assert "Actor Explanation Audit" in html
    assert "Mohammad Mossadeq" in html
    assert "prime minister and constitutional nationalist" in html
    assert "mapped_section_realization" in html

    payload = build_actor_explanation_audit_payload(run_dir)
    rendered = render_actor_explanation_audit_html(payload)
    assert "Episode 1" in rendered
