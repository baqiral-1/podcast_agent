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

    book_1 = {
        "book_id": "book-1",
        "title": "Book One",
        "author": "Author A",
        "source_path": "sample_books/book_one.txt",
        "source_type": "txt",
        "chunk_count": 80,
        "total_words": 12000,
        "chapters": [{"chapter_id": "ch-1"}],
    }
    book_2 = {
        "book_id": "book-2",
        "title": "Book Two",
        "author": "Author B",
        "source_path": "sample_books/book_two.txt",
        "source_type": "txt",
        "chunk_count": 80,
        "total_words": 10000,
        "chapters": [{"chapter_id": "ch-2"}],
    }
    _write_json(
        run_dir / "thematic_project.json",
        {
            "project_id": "demo-run",
            "theme": "Demo Theme",
            "theme_elaboration": "Demo elaboration.",
            "status": "complete",
            "episode_count": 1,
            "books": [book_1, book_2],
        },
    )

    _write_json(
        run_dir / "thematic_axes.json",
        {
            "axes": [
                {
                    "axis_id": "axis_01",
                    "name": "Axis One",
                    "description": "Primary axis.",
                    "keywords": ["alpha", "beta"],
                    "guiding_questions": ["What happens?"],
                    "relevance_by_book": {"book-1": 0.9, "book-2": 0.7},
                }
            ]
        },
    )

    passages = []
    for index in range(70):
        book_id = "book-1" if index < 35 else "book-2"
        passages.append(
            {
                "passage_id": f"p-{index:02d}",
                "book_id": book_id,
                "chapter_ref": f"{book_id}:chapter-1",
                "chunk_ids": [f"chunk-{index:02d}"],
                "text": f"Passage {index}",
                "trimmed_text": f"Passage {index}",
                "full_text": f"Passage {index}",
                "relevance_score": 1.0 - (index * 0.01),
                "quotability_score": 0.5 + (index * 0.001),
                "secondary_axes": [],
                "synthesis_tags": [],
            }
        )
    _write_json(
        run_dir / "thematic_corpus.json",
        {
            "project_id": "demo-run",
            "axes": ["axis_01"],
            "total_passages": len(passages),
            "book_coverage": {},
            "cross_book_pairs": [],
            "passages_by_axis": {"axis_01": passages},
        },
    )

    _write_json(
        run_dir / "synthesis_map.json",
        {
            "project_id": "demo-run",
            "insights": [
                {
                    "insight_id": "insight-1",
                    "insight_type": "latent_pattern",
                    "title": "Insight One",
                    "description": "Two anchor passages.",
                    "passage_ids": ["p-00", "p-01"],
                    "podcast_potential": 0.9,
                    "treatment": "contrast",
                    "axis_ids": ["axis_01"],
                }
            ],
            "merged_narratives": [
                {
                    "topic": "Merged Topic",
                    "narrative": "Merged narrative text.",
                    "source_passage_ids": ["p-00", "p-01"],
                    "points_of_consensus": ["Point A"],
                    "points_of_disagreement": ["Point B"],
                }
            ],
            "narrative_threads": [],
        },
    )

    _write_json(
        run_dir / "narrative_strategy.json",
        {
            "strategy_type": "convergence",
            "recommended_episode_count": 1,
            "series_arc": "A short arc.",
            "episode_assignments": [
                {
                    "episode_number": 1,
                    "title": "Episode One",
                    "driving_question": "What happened?",
                    "thematic_focus": "Focus",
                    "episode_strategy": "single-axis",
                    "axes": [{"axis_id": "axis_01", "description": "Primary axis."}],
                    "insight_ids": ["insight-1"],
                    "merged_narrative_id": "merged_narrative_001",
                    "tension_ids": [],
                }
            ],
            "episode_arc_details": [
                {
                    "episode_number": 1,
                    "arc_summary": "Arc summary.",
                    "narrative_stakes": "Stakes.",
                    "payoff_shape": "Payoff.",
                    "unresolved_questions": ["Question?"],
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
                    "beats": [
                        {
                            "beat_id": "beat-1",
                            "description": "Opening beat.",
                            "passage_ids": ["p-00", "p-10", "p-61"],
                            "primary_book_id": "book-1",
                            "supporting_book_ids": ["book-2"],
                            "synthesis_instruction": "Use passages.",
                        }
                    ],
                }
            ]
        },
    )

    _write_json(
        run_dir / "retrieval_metrics.json",
        {
            "summary": {"total_axes": 1, "total_passages": 70, "total_cross_book_pairs": 0},
            "per_axis": {
                "axis_01": {
                    "candidate_count": 80,
                    "post_rerank_count": 70,
                    "rehydrated_count": 70,
                    "full_text_count": 70,
                    "trimmed_text_count": 70,
                    "full_text_coverage_ratio": 1.0,
                    "selection_policy": "demo",
                    "avg_relevance_score": 0.75,
                    "avg_quotability_score": 0.55,
                    "relevance_distribution": {},
                    "books_represented": ["book-1", "book-2"],
                    "cross_pair_validation": {},
                }
            },
            "per_book": {
                "book-1": {"title": "Book One", "total_passages": 35, "axes_with_passages": 1, "avg_relevance": 0.8, "size_share": 0.5, "avg_axis_quota_share": 0.5, "quota_minus_size_share": 0.0},
                "book-2": {"title": "Book Two", "total_passages": 35, "axes_with_passages": 1, "avg_relevance": 0.7, "size_share": 0.5, "avg_axis_quota_share": 0.5, "quota_minus_size_share": 0.0},
            },
        },
    )

    _write_json(
        run_dir / "passage_utilization.json",
        {
            "summary": {"retained_count": 70, "planned_count": 3, "cited_count": 0},
            "per_axis": {
                "axis_01": {
                    "retained_count": 70,
                    "planned_count": 3,
                    "cited_count": 0,
                    "plan_utilization_ratio": 3 / 70,
                    "citation_utilization_ratio": 0.0,
                }
            },
            "per_book": {
                "book-1": {"title": "Book One", "retained_count": 35, "planned_count": 2, "cited_count": 0, "plan_utilization_ratio": 2 / 35, "citation_utilization_ratio": 0.0},
                "book-2": {"title": "Book Two", "retained_count": 35, "planned_count": 1, "cited_count": 0, "plan_utilization_ratio": 1 / 35, "citation_utilization_ratio": 0.0},
            },
        },
    )

    _write_json(
        run_dir / "episode_plan_realization.json",
        {
            "episodes": [
                {
                    "episode_number": 1,
                    "title": "Episode One",
                    "has_issues": False,
                    "problem_count": 0,
                    "insight_problem_count": 0,
                    "merged_narrative_problem_count": 0,
                    "insight_ids": ["insight-1"],
                    "insights": [
                        {
                            "insight_id": "insight-1",
                            "title": "Insight One",
                            "status": "ok",
                            "assigned_passage_count": 2,
                            "realized_count": 2,
                            "expected_min": 2,
                            "missing_passage_ids": [],
                        }
                    ],
                    "merged_narratives": [],
                }
            ]
        },
    )

    _write_json(
        run_dir / "stage_artifacts" / "passage_extraction" / "retrieval_candidates_axis_01.json",
        {
            "axis_id": "axis_01",
            "axis_name": "Axis One",
            "axis_candidate_budget_target": 80,
            "axis_candidate_budget_effective": 80,
            "passage_retrieval_min_per_book": 20,
            "passage_retrieval_max_per_book": 60,
            "soft_threshold": 0.2,
            "retrieval_relevance_power": 1.3,
            "allocation_policy": "demo-policy",
            "admission_quota_by_book": {"book-1": 35, "book-2": 35},
            "books": [
                {
                    "book_id": "book-1",
                    "title": "Book One",
                    "chunk_count": 80,
                    "retrieval_depth_budget": 40,
                    "admission_quota": 35,
                    "eligible_total_count": 35,
                    "eligible_above_threshold_count": 35,
                    "selected_above_threshold_count": 35,
                    "selected_backfill_count": 0,
                    "selected_spillover_count": 0,
                    "underfill_count": 0,
                    "candidates": [{"used": True, "selection_phase": "primary"}],
                },
                {
                    "book_id": "book-2",
                    "title": "Book Two",
                    "chunk_count": 80,
                    "retrieval_depth_budget": 40,
                    "admission_quota": 35,
                    "eligible_total_count": 35,
                    "eligible_above_threshold_count": 35,
                    "selected_above_threshold_count": 35,
                    "selected_backfill_count": 0,
                    "selected_spillover_count": 0,
                    "underfill_count": 0,
                    "candidates": [{"used": True, "selection_phase": "primary"}],
                },
            ],
        },
    )

    episode_dir = run_dir / "episodes" / "1"
    _write_json(
        episode_dir / "episode_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "segments": [{"segment_id": "s-1", "text": "Segment text."}],
            "total_word_count": 1234,
            "estimated_duration_seconds": 600,
            "citations": [],
        },
    )
    _write_json(
        episode_dir / "spoken_script.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "segments": [{"segment_id": "ss-1", "text": "Spoken segment text for the episode.", "max_words": 20, "speech_hints": []}],
            "arc_plan": "arc",
            "tts_provider": "demo",
        },
    )
    _write_json(
        episode_dir / "render_manifest.json",
        {
            "episode_number": 1,
            "segments": [{"segment_id": "r-1", "text": "Render text.", "voice_id": "v1", "speed": 1.0, "pause_before_ms": 0, "pause_after_ms": 100, "instructions": "", "hint_degradations": []}],
            "total_segments": 1,
            "estimated_duration_seconds": 540,
        },
    )
    _write_json(
        episode_dir / "plan_alignment_report.json",
        {
            "episode_number": 1,
            "title": "Episode One",
            "has_issues": False,
            "problem_count": 0,
            "thresholds": {},
            "insight_realization": {},
            "cross_references": {},
            "book_balance": {},
        },
    )
    _write_json(
        episode_dir / "episode_framing.json",
        {"episode_number": 1, "recap": None, "preview": "Preview", "cold_open": "Cold open"},
    )

    (run_dir / "run.log").write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-04-07T03:20:46.334759+00:00", "event_type": "stage_start", "payload": {"stage": "passage_extraction", "input_summary": {"axes": 1}}}),
                json.dumps({"timestamp": "2026-04-07T03:21:46.334759+00:00", "event_type": "stage_end", "payload": {"stage": "passage_extraction", "output_summary": {"retained": 70}}}),
            ]
        )
    )
    return run_dir


def _run_script(run_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), *extra_args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_run_stats_writes_html(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)

    result = _run_script(run_dir)

    assert result.returncode == 0, result.stderr
    output = run_dir / "run-stats.html"
    assert output.exists()
    html = output.read_text()
    assert "Run Statistics — demo-run" in html
    assert "planning exposures 62" in html
    assert "Retrieval Summary JSON and Raw Artifact Link" in html
    assert "stage_artifacts/passage_extraction/retrieval_candidates_axis_01.json" in html
    assert "per-book counts at each step" in html


def test_generate_run_stats_supports_custom_output(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    output = tmp_path / "custom-report.html"

    result = _run_script(run_dir, "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert output.exists()
    html = output.read_text()
    assert "episodes/1/episode_script.json" in html


def test_generate_run_stats_tolerates_missing_optional_artifacts(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    for rel in [
        "retrieval_metrics.json",
        "passage_utilization.json",
        "episode_plan_realization.json",
        "stage_artifacts/passage_extraction/retrieval_candidates_axis_01.json",
        "episodes/1/plan_alignment_report.json",
    ]:
        (run_dir / rel).unlink()

    result = _run_script(run_dir)

    assert result.returncode == 0, result.stderr
    html = (run_dir / "run-stats.html").read_text()
    assert "Run Statistics — demo-run" in html
    assert "Episode One" in html


def test_generate_run_stats_fails_when_core_artifact_missing(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    (run_dir / "thematic_corpus.json").unlink()

    result = _run_script(run_dir)

    assert result.returncode == 1
    assert "missing required artifacts" in result.stderr
