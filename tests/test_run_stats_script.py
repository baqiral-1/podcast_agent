from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_run_stats.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "demo-run"
    run_dir.mkdir(parents=True)

    _write_json(
        run_dir / "thematic_project.json",
        {
            "project_id": "demo-run",
            "theme": "War on Terror",
            "status": "complete",
            "episode_count": 1,
            "books": [],
        },
    )
    _write_json(run_dir / "thematic_axes.json", {"axes": [{"axis_id": "axis_1", "name": "Escalation"}]})
    _write_json(
        run_dir / "thematic_corpus.json",
        {
            "project_id": "demo-run",
            "axes": [],
            "total_passages": 2,
            "book_coverage": {},
            "cross_book_pairs": [],
            "passages_by_axis": {"axis_1": [{"passage_id": "p1"}, {"passage_id": "p2"}]},
        },
    )
    _write_json(
        run_dir / "synthesis_primitives.json",
        {
            "project_id": "demo-run",
            "primitives": [
                {
                    "id": "et_1",
                    "substrate": "events",
                    "title": "Turn",
                    "core_passage_ids": ["p1"],
                    "event_type": "political rupture",
                    "what_happened": "A decisive turn lands.",
                },
                {
                    "id": "cx_1",
                    "substrate": "readings",
                    "title": "Question",
                    "core_passage_ids": ["p2"],
                    "reading_type": "historiographical_dispute",
                    "reading_summary": "Two explanations compete.",
                },
            ],
            "quality_score": 0.6,
            "quality_notes": [],
        },
    )
    _write_json(
        run_dir / "synthesis_map.json",
        {
            "project_id": "demo-run",
            "primitives": [
                {
                    "id": "et_1",
                    "substrate": "events",
                    "title": "Turn",
                    "core_passage_ids": ["p1"],
                    "event_type": "political rupture",
                    "what_happened": "A decisive turn lands.",
                },
                {
                    "id": "cx_1",
                    "substrate": "readings",
                    "title": "Question",
                    "core_passage_ids": ["p2"],
                    "reading_type": "historiographical_dispute",
                    "reading_summary": "Two explanations compete.",
                },
            ],
            "quality_score": 0.7,
            "quality_notes": [],
        },
    )
    _write_json(
        run_dir / "narrative_strategy.json",
        {
            "strategy_type": "convergence",
            "justification": "One cluster anchors the first episode.",
            "series_arc": "The arc follows one local chain.",
            "episode_arc_outline": ["Episode One"],
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode One",
                    "thematic_focus": "Opening rupture",
                    "arc_summary": "Follow the local consequences.",
                    "unresolved_questions": [],
                    "episode_spine": {
                        "listener_question": "Why does the order land so hard?",
                        "argument": "Opening rupture clarifies the proposition.",
                        "core_primitive_ids": ["et_1"],
                        "support_primitive_roles": {},
                        "recall_primitive_ids": [],
                    },
                }
            ],
        },
    )
    _write_json(
        run_dir / "series_plan.json",
        {
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode One",
                    "scene_cards": [
                        {
                            "scene_id": "scene_1",
                            "dominant_primitive_id": "et_1",
                            "primitive_ids": ["et_1"],
                        }
                    ],
                }
            ]
        },
    )
    (run_dir / "run.log").write_text("run started\nrun finished\n", encoding="utf-8")

    episode_dir = run_dir / "episodes" / "1"
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
            "prose_sections": [{"section_id": "section_1", "scene_card_ids": ["scene_1"], "movement_goal": "discover", "text": "Narration"}],
            "total_word_count": 1,
            "estimated_duration_seconds": 1,
        },
    )
    _write_json(
        episode_dir / "spoken_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": "Question",
                "handoff_scene_card_id": "scene_1",
            },
            "sections": [{"section_id": "section_1", "text": "Narration"}],
        },
    )

    return run_dir


def test_generate_run_stats_html(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    output_path = run_dir / "run-stats.html"

    subprocess.run([sys.executable, str(SCRIPT), str(run_dir)], check=True)

    html = output_path.read_text(encoding="utf-8")
    assert "War on Terror" in html
    assert "events: 1" in html
    assert "readings: 1" in html
    assert "Synthesis Map" in html
    assert "Primitive Retention" in html
    assert "<td>events</td>" in html
    assert "100.0%" in html
    assert "Core primitives:</strong> 1" in html
    assert "Driving question:" in html


def test_generate_run_stats_fails_when_required_artifact_missing(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    (run_dir / "synthesis_primitives.json").unlink()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "synthesis_primitives.json" in result.stderr or "synthesis_primitives.json" in result.stdout
