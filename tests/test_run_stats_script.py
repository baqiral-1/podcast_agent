from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.schemas.models import SYNTHESIS_PRIMITIVE_FAMILIES

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
            "primitives_by_family": {
                **{family: [] for family in SYNTHESIS_PRIMITIVE_FAMILIES},
                "epochal_turns": [{"id": "et_1", "title": "Turn", "summary": "Summary", "core_passage_ids": ["p1"]}],
                "contested_explanations": [
                    {
                        "id": "cx_1",
                        "title": "Question",
                        "summary": "Summary",
                        "core_passage_ids": ["p2"],
                        "candidate_readings": [
                            {"label": "a", "summary": "A", "support_passage_ids": ["p2"]},
                            {"label": "b", "summary": "B", "support_passage_ids": ["p1"]},
                        ],
                    }
                ],
            },
            "quality_score": 0.6,
            "quality_notes": [],
        },
    )
    _write_json(
        run_dir / "synthesis_map.json",
        {
            "project_id": "demo-run",
            "primitives_by_family": {
                **{family: [] for family in SYNTHESIS_PRIMITIVE_FAMILIES},
                "epochal_turns": [{"id": "et_1", "title": "Turn", "summary": "Summary", "core_passage_ids": ["p1"]}],
                "contested_explanations": [
                    {
                        "id": "cx_1",
                        "title": "Question",
                        "summary": "Summary",
                        "core_passage_ids": ["p2"],
                        "candidate_readings": [
                            {"label": "a", "summary": "A", "support_passage_ids": ["p2"]},
                            {"label": "b", "summary": "B", "support_passage_ids": ["p1"]},
                        ],
                    }
                ],
            },
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
                    "driving_question": "Why does the order land so hard?",
                    "thematic_focus": "Opening rupture",
                    "arc_summary": "Follow the local consequences.",
                    "unresolved_questions": [],
                    "episode_spine": {
                        "listener_question": "Why does the order land so hard?",
                        "working_claim": "Opening rupture clarifies the proposition.",
                        "target_end_state": "The local consequences become legible.",
                        "verdict_mode": "constrain",
                        "primary_counterposition": "Another reading remains possible.",
                        "core_primitive_ids": ["primitive_1"],
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
                    "scene_cards": [{"scene_id": "scene_1"}],
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
    assert "epochal_turns: 1" in html
    assert "Synthesis Map" in html
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
