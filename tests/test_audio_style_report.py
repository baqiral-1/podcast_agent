from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from podcast_agent.eval.audio_style_report import (
    AFS_CALIBRATED_ANCHORS,
    AFS_RAW_ANCHORS,
    TSS_CALIBRATED_ANCHORS,
    TSS_RAW_ANCHORS,
    _piecewise_calibrate,
    build_audio_style_payload,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_audio_style_report.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_run(
    tmp_path: Path,
    *,
    run_name: str,
    artifact_name: str,
    text: str,
    section_container: str = "prose_sections",
) -> Path:
    run_dir = tmp_path / "runs" / run_name
    episode_dir = run_dir / "episodes" / "1"
    episode_dir.mkdir(parents=True, exist_ok=True)
    section_payload = [
        {
            "section_id": "intro",
            "scene_card_ids": ["scene_1"],
            "movement_goal": "open",
            "text": text,
        }
    ]
    _write_json(
        episode_dir / artifact_name,
        {
            "episode_number": 1,
            "title": f"{run_name} episode",
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": "Question",
                "handoff_scene_card_id": "scene_1",
            },
            section_container: section_payload,
            "total_word_count": len(text.split()),
            "estimated_duration_seconds": 180,
        },
    )
    return run_dir


def test_build_audio_style_payload_scores_oral_script_higher(tmp_path: Path) -> None:
    oral_text = (
        "Picture the room just before dawn. The doors are open. You can feel the crowd hesitate. "
        "Hold that. I mean, think about it like a trap snapping shut, wouldn't you? "
        "The point is that once those doors close, the whole argument changes.\n\n"
        "Now, if you were standing there, you'd feel it at once. You can see the turn in the crowd. "
        "In plain English, the leaders think they are buying time, but the result is the opposite. "
        "They are spending it, and you can almost watch the minutes burn."
    )
    dry_text = (
        "The constitutional arrangement established a revised governance framework for provincial "
        "administration. Representation, institutional capacity, and fiscal delegation interacted "
        "in ways that produced subsequent instability. The result was a procedural impasse.\n\n"
        "Political actors responded to the changed incentives through coalition breakdown, "
        "administrative realignment, and rhetorical escalation. These developments intensified the "
        "probability of further institutional fragmentation."
    )
    oral_run = _build_run(
        tmp_path,
        run_name="oral-run",
        artifact_name="episode_script.json",
        text=oral_text,
    )
    dry_run = _build_run(
        tmp_path,
        run_name="dry-run",
        artifact_name="episode_script.json",
        text=dry_text,
    )

    payload = build_audio_style_payload([oral_run, dry_run], title="Demo Audio Style Report")

    runs = {run["run_id"]: run for run in payload["runs"]}
    assert runs["oral-run"]["afs"] > runs["dry-run"]["afs"]
    assert runs["oral-run"]["tss"] > runs["dry-run"]["tss"]
    assert runs["oral-run"]["tss_v2"] is not None
    assert runs["dry-run"]["tss_v2"] is not None
    assert payload["comparison_target"]["label"] == "Bundled Revolutions directory-army sample"
    assert "raw_afs" in runs["oral-run"]
    assert "raw_tss" in runs["oral-run"]
    assert payload["summary"]["winner_by_afs"] == "oral-run"
    assert payload["summary"]["winner_by_tss"] == "oral-run"
    assert payload["summary"]["winner_by_tss_v2"] is not None


def test_build_audio_style_payload_emits_tss_v2_for_comparison_sample(tmp_path: Path) -> None:
    oral_text = (
        "Picture the bridge at first light. Hold that image. I mean, if you were standing there, "
        "you would feel the turn at once. The point is that the crowd changes before the leaders do."
    )
    dry_text = (
        "The revised constitutional framework altered the balance between provincial institutions "
        "and central authority, generating an unstable procedural outcome."
    )
    oral_run = _build_run(
        tmp_path,
        run_name="oral-run",
        artifact_name="episode_script.json",
        text=oral_text,
    )
    dry_run = _build_run(
        tmp_path,
        run_name="dry-run",
        artifact_name="episode_script.json",
        text=dry_text,
    )

    payload = build_audio_style_payload(
        [oral_run, dry_run],
        title="Demo Audio Style Report",
        comparison_target_snippet=oral_text,
        comparison_target_label="Oral reference",
    )

    runs = {run["run_id"]: run for run in payload["runs"]}
    assert payload["comparison_target"]["label"] == "Oral reference"
    assert runs["oral-run"]["tss_v2"] is not None
    assert runs["dry-run"]["tss_v2"] is not None
    assert runs["oral-run"]["tss_v2"] > runs["dry-run"]["tss_v2"]
    assert payload["summary"]["winner_by_tss_v2"] == "oral-run"


def test_build_audio_style_report_script_writes_outputs(tmp_path: Path) -> None:
    run_dir = _build_run(
        tmp_path,
        run_name="demo-run",
        artifact_name="spoken_script.json",
        text=(
            "Picture the bridge at first light. The trucks are already there. "
            "Hold that image in your head while the argument catches up."
        ),
        section_container="sections",
    )
    output_html = tmp_path / "audio-style-report.html"
    output_json = tmp_path / "audio-style-report.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Demo Audio Style Report",
            "--output-html",
            str(output_html),
            "--output-json",
            str(output_json),
            "--script-artifact",
            "spoken_script.json",
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_html.exists()
    assert output_json.exists()
    html_text = output_html.read_text(encoding="utf-8")
    assert "Demo Audio Style Report" in html_text
    assert "TSSv2" in html_text
    assert "comparison sample" in html_text.lower()
    assert "Raw AFS/TSS" in html_text
    assert "calibrated onto the editorial anchor scale" in html_text
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["run_count"] == 1
    assert payload["comparison_target"]["label"] == "Bundled Revolutions directory-army sample"
    assert payload["runs"][0]["episodes"][0]["script_artifact"] == "spoken_script.json"
    assert "raw_afs" in payload["runs"][0]
    assert "raw_tss" in payload["runs"][0]
    assert payload["runs"][0]["tss_v2"] is not None
    assert "raw_afs" in payload["runs"][0]["episodes"][0]
    assert "raw_tss" in payload["runs"][0]["episodes"][0]
    assert payload["runs"][0]["episodes"][0]["tss_v2"] is not None


def test_piecewise_calibration_hits_anchor_points() -> None:
    for raw_value, calibrated_value in zip(AFS_RAW_ANCHORS, AFS_CALIBRATED_ANCHORS, strict=True):
        assert _piecewise_calibrate(raw_value, AFS_RAW_ANCHORS, AFS_CALIBRATED_ANCHORS) == calibrated_value

    for raw_value, calibrated_value in zip(TSS_RAW_ANCHORS, TSS_CALIBRATED_ANCHORS, strict=True):
        assert _piecewise_calibrate(raw_value, TSS_RAW_ANCHORS, TSS_CALIBRATED_ANCHORS) == calibrated_value
