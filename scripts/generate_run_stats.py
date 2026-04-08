#!/usr/bin/env python3
"""Generate a standalone HTML statistics page from a pipeline run directory."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

CORE_ARTIFACTS = (
    "thematic_project.json",
    "thematic_axes.json",
    "thematic_corpus.json",
    "synthesis_map.json",
    "narrative_strategy.json",
    "series_plan.json",
    "run.log",
)


class RunStatsError(RuntimeError):
    """Raised when a run directory is missing required artifacts."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate run-stats.html for a completed run directory using persisted artifacts."
        )
    )
    parser.add_argument("run_dir", help="Path to runs/<project-id> directory")
    parser.add_argument(
        "--output",
        help="Optional output HTML path. Defaults to <run_dir>/run-stats.html.",
    )
    return parser.parse_args()


def _check_run_dir(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        raise RunStatsError(f"Run directory does not exist: {run_dir}")
    missing = [name for name in CORE_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise RunStatsError(f"Run directory is missing required artifacts: {joined}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _json_pretty(value: Any) -> str:
    return _escape(json.dumps(value, indent=2, ensure_ascii=False))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _fmt_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple, set, dict)):
        return f"{len(value):,}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f}s"
    total = int(round(value))
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’-]\w+)?\b", text))


def _parse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stage_family(name: str) -> str:
    for prefix in (
        "ingest_book_",
        "structure_",
        "write_episode_",
        "spoken_delivery_",
        "framing_",
        "audio_",
        "audio_resynthesis_",
    ):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return name


def _rel_link(base_output: Path, target: Path) -> str:
    return os.path.relpath(target, base_output.parent)


def _bar(value: int | float, max_value: int | float) -> str:
    pct = 0.0 if max_value <= 0 else max(0.0, min(100.0, (float(value) / float(max_value)) * 100.0))
    return f'<div class="bar"><span style="width:{pct:.2f}%"></span></div>'


def _load_stage_records(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dt.datetime | None, dt.datetime | None]:
    stage_records: list[dict[str, Any]] = []
    pending_starts: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for raw_line in (run_dir / "run.log").read_text(errors="ignore").splitlines():
        raw_line = raw_line.strip()
        if not raw_line.startswith("{"):
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        kind = event.get("event_type")
        payload = event.get("payload", {})
        timestamp = _parse_timestamp(event.get("timestamp"))
        if kind == "stage_start":
            pending_starts[payload.get("stage", "unknown")].append(
                {
                    "start": timestamp,
                    "input_summary": payload.get("input_summary"),
                }
            )
        elif kind == "stage_end":
            stage = payload.get("stage", "unknown")
            start_info = pending_starts[stage].pop(0) if pending_starts.get(stage) else None
            start_time = start_info["start"] if start_info else None
            elapsed = (timestamp - start_time).total_seconds() if timestamp and start_time else None
            stage_records.append(
                {
                    "stage": stage,
                    "family": _stage_family(stage),
                    "start": start_time,
                    "end": timestamp,
                    "elapsed_seconds": elapsed,
                    "input_summary": start_info["input_summary"] if start_info else None,
                    "output_summary": payload.get("output_summary"),
                }
            )
    for stage, starts in pending_starts.items():
        for start_info in starts:
            stage_records.append(
                {
                    "stage": stage,
                    "family": _stage_family(stage),
                    "start": start_info["start"],
                    "end": None,
                    "elapsed_seconds": None,
                    "input_summary": start_info["input_summary"],
                    "output_summary": None,
                }
            )
    stage_records.sort(
        key=lambda item: (
            item["start"] or dt.datetime.max.replace(tzinfo=dt.timezone.utc),
            item["stage"],
        )
    )
    family_summary: dict[str, dict[str, Any]] = collections.OrderedDict()
    for record in stage_records:
        family = record["family"]
        summary = family_summary.setdefault(
            family,
            {"family": family, "count": 0, "completed": 0, "elapsed_seconds": 0.0},
        )
        summary["count"] += 1
        if record["elapsed_seconds"] is not None:
            summary["completed"] += 1
            summary["elapsed_seconds"] += record["elapsed_seconds"]
    starts = [record["start"] for record in stage_records if record["start"] is not None]
    ends = [record["end"] for record in stage_records if record["end"] is not None]
    return stage_records, family_summary, min(starts) if starts else None, max(ends) if ends else None


def _chunk_key(passage: dict[str, Any]) -> tuple[str, ...]:
    chunk_ids = passage.get("chunk_ids") or []
    if chunk_ids:
        return tuple(chunk_ids)
    return ("passage", passage["passage_id"])


def _build_planning_selection(
    *,
    passages_by_axis: dict[str, list[dict[str, Any]]],
    assigned_axis_ids: list[str],
    selected_insight_passage_ids: set[str],
    supporting_passages_per_axis: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    chunk_axes: dict[tuple[str, ...], set[str]] = {}
    for axis_id in assigned_axis_ids:
        for passage in passages_by_axis.get(axis_id, []):
            key = _chunk_key(passage)
            chunk_axes.setdefault(key, set()).add(axis_id)

    selected_by_axis: dict[str, list[dict[str, Any]]] = {axis_id: [] for axis_id in assigned_axis_ids}
    for axis_id in assigned_axis_ids:
        insight_passages: list[dict[str, Any]] = []
        supporting_ranked: list[tuple[float, float, float, str, dict[str, Any]]] = []
        for passage in passages_by_axis.get(axis_id, []):
            if passage["passage_id"] in selected_insight_passage_ids:
                insight_passages.append(passage)
                continue
            key = _chunk_key(passage)
            is_multi_axis = len(chunk_axes.get(key, set())) > 1
            weighted_score = (
                (0.65 * passage["relevance_score"])
                + (0.35 * passage["quotability_score"])
                + (0.04 if is_multi_axis else 0.0)
            )
            supporting_ranked.append(
                (
                    -weighted_score,
                    -passage["relevance_score"],
                    -passage["quotability_score"],
                    passage["passage_id"],
                    passage,
                )
            )
        insight_passages.sort(
            key=lambda item: (-item["relevance_score"], -item["quotability_score"], item["passage_id"])
        )
        supporting_ranked.sort()
        supporting_passages = [item[-1] for item in supporting_ranked[:supporting_passages_per_axis]]
        selected_by_axis[axis_id] = insight_passages + supporting_passages
    return selected_by_axis


def _episode_spoken_word_count(episode_dir: Path) -> int | None:
    spoken = _load_optional_json(episode_dir / "spoken_script.json")
    if not spoken:
        return None
    text = "\n\n".join(
        segment.get("text", "").strip() for segment in spoken.get("segments", []) if segment.get("text")
    )
    return _count_words(text)


def _compute_reports(run_dir: Path, output_path: Path) -> dict[str, Any]:
    project = _load_json(run_dir / "thematic_project.json")
    axes_payload = _load_json(run_dir / "thematic_axes.json")
    corpus = _load_json(run_dir / "thematic_corpus.json")
    synthesis_map = _load_json(run_dir / "synthesis_map.json")
    strategy = _load_json(run_dir / "narrative_strategy.json")
    series_plan = _load_json(run_dir / "series_plan.json")
    retrieval_metrics = _load_optional_json(run_dir / "retrieval_metrics.json") or {}
    passage_utilization = _load_optional_json(run_dir / "passage_utilization.json") or {}
    realization = _load_optional_json(run_dir / "episode_plan_realization.json") or {}

    stage_records, stage_families, run_start, run_end = _load_stage_records(run_dir)
    run_elapsed_seconds = (
        (run_end - run_start).total_seconds() if run_start is not None and run_end is not None else None
    )

    book_by_id = {book["book_id"]: book for book in project.get("books", [])}
    axis_by_id = {axis["axis_id"]: axis for axis in axes_payload.get("axes", [])}
    axis_order = [axis["axis_id"] for axis in axes_payload.get("axes", [])]
    insight_by_id = {insight["insight_id"]: insight for insight in synthesis_map.get("insights", [])}
    merged_by_id = {
        f"merged_narrative_{index:03d}": item
        for index, item in enumerate(synthesis_map.get("merged_narratives", []), start=1)
    }
    assignment_by_episode = {
        item["episode_number"]: item for item in strategy.get("episode_assignments", [])
    }
    plan_by_episode = {item["episode_number"]: item for item in series_plan.get("episodes", [])}
    realization_by_episode = {
        item["episode_number"]: item for item in realization.get("episodes", [])
    }
    arc_detail_by_episode = {
        item["episode_number"]: item for item in strategy.get("episode_arc_details", [])
    }

    passages_by_axis = corpus.get("passages_by_axis", {})
    passage_lookup: dict[str, dict[str, Any]] = {}
    for axis_id, passages in passages_by_axis.items():
        for passage in passages:
            item = dict(passage)
            item["axis_id"] = axis_id
            passage_lookup[passage["passage_id"]] = item

    retrieval_artifacts: dict[str, dict[str, Any]] = {}
    passage_extraction_dir = run_dir / "stage_artifacts" / "passage_extraction"
    if passage_extraction_dir.exists():
        for artifact in sorted(passage_extraction_dir.glob("retrieval_candidates_axis_*.json")):
            payload = _load_json(artifact)
            retrieval_artifacts[payload["axis_id"]] = payload

    episode_reports: list[dict[str, Any]] = []
    planning_unique_ids_all: set[str] = set()
    beat_unique_ids_all: set[str] = set()
    missing_plan_passage_ids: set[str] = set()
    for episode_number in sorted(assignment_by_episode):
        assignment = assignment_by_episode[episode_number]
        episode_plan = plan_by_episode.get(episode_number)
        if episode_plan is None:
            continue
        axis_ids = [axis["axis_id"] if isinstance(axis, dict) else axis for axis in assignment.get("axes", [])]
        selected_insight_passage_ids = {
            passage_id
            for insight_id in assignment.get("insight_ids", [])
            for passage_id in insight_by_id.get(insight_id, {}).get("passage_ids", [])
        }
        planning_by_axis = _build_planning_selection(
            passages_by_axis=passages_by_axis,
            assigned_axis_ids=axis_ids,
            selected_insight_passage_ids=selected_insight_passage_ids,
        )
        planning_unique_ids = {
            passage["passage_id"]
            for items in planning_by_axis.values()
            for passage in items
        }
        planning_unique_ids_all.update(planning_unique_ids)

        beat_passage_refs = [
            passage_id
            for beat in episode_plan.get("beats", [])
            for passage_id in beat.get("passage_ids", [])
        ]
        beat_unique_ids = set(beat_passage_refs)
        beat_unique_ids_all.update(beat_unique_ids)
        missing_plan_passage_ids.update(
            passage_id for passage_id in beat_unique_ids if passage_id not in passage_lookup
        )
        beat_passages_known = [passage_lookup[p_id] for p_id in beat_passage_refs if p_id in passage_lookup]

        axis_rows: list[dict[str, Any]] = []
        axis_book_tables: list[dict[str, Any]] = []
        for axis_id in axis_ids:
            retained_axis = passages_by_axis.get(axis_id, [])
            planning_axis = planning_by_axis.get(axis_id, [])
            insight_axis = [item for item in retained_axis if item["passage_id"] in selected_insight_passage_ids]
            beat_axis_refs = [
                passage_id
                for passage_id in beat_passage_refs
                if passage_lookup.get(passage_id, {}).get("axis_id") == axis_id
            ]
            beat_axis_unique = set(beat_axis_refs)
            axis_rows.append(
                {
                    "axis_id": axis_id,
                    "axis_name": axis_by_id.get(axis_id, {}).get("name", axis_id),
                    "retained_count": len(retained_axis),
                    "insight_count": len(insight_axis),
                    "supporting_count": len(planning_axis) - len(insight_axis),
                    "planning_count": len(planning_axis),
                    "beat_ref_count": len(beat_axis_refs),
                    "beat_unique_count": len(beat_axis_unique),
                    "books_represented": len({item["book_id"] for item in planning_axis}),
                }
            )
            books_seen = {item["book_id"] for item in retained_axis} | {item["book_id"] for item in planning_axis}
            books_seen.update(
                passage_lookup[passage_id]["book_id"]
                for passage_id in beat_axis_unique
                if passage_id in passage_lookup
            )
            table_rows: list[dict[str, Any]] = []
            for book_id in sorted(books_seen, key=lambda value: book_by_id.get(value, {}).get("title", value)):
                retained_book = [item for item in retained_axis if item["book_id"] == book_id]
                insight_book = [item for item in insight_axis if item["book_id"] == book_id]
                planning_book = [item for item in planning_axis if item["book_id"] == book_id]
                beat_book_refs = [
                    passage_id
                    for passage_id in beat_axis_refs
                    if passage_lookup.get(passage_id, {}).get("book_id") == book_id
                ]
                row = {
                    "book_id": book_id,
                    "title": book_by_id.get(book_id, {}).get("title", book_id),
                    "retained_count": len(retained_book),
                    "insight_count": len(insight_book),
                    "supporting_count": len(planning_book) - len(insight_book),
                    "planning_count": len(planning_book),
                    "beat_ref_count": len(beat_book_refs),
                    "beat_unique_count": len(set(beat_book_refs)),
                }
                if row["retained_count"] or row["planning_count"] or row["beat_ref_count"]:
                    table_rows.append(row)
            axis_book_tables.append(
                {
                    "axis_id": axis_id,
                    "axis_name": axis_by_id.get(axis_id, {}).get("name", axis_id),
                    "rows": table_rows,
                }
            )

        books_seen = set()
        for axis_id in axis_ids:
            books_seen.update(item["book_id"] for item in passages_by_axis.get(axis_id, []))
            books_seen.update(item["book_id"] for item in planning_by_axis.get(axis_id, []))
        books_seen.update(item["book_id"] for item in beat_passages_known)
        book_rows: list[dict[str, Any]] = []
        for book_id in sorted(books_seen, key=lambda value: book_by_id.get(value, {}).get("title", value)):
            retained_count = sum(
                1
                for axis_id in axis_ids
                for item in passages_by_axis.get(axis_id, [])
                if item["book_id"] == book_id
            )
            insight_count = sum(
                1
                for axis_id in axis_ids
                for item in passages_by_axis.get(axis_id, [])
                if item["book_id"] == book_id and item["passage_id"] in selected_insight_passage_ids
            )
            planning_count = sum(
                1
                for axis_id in axis_ids
                for item in planning_by_axis.get(axis_id, [])
                if item["book_id"] == book_id
            )
            beat_book_refs = [item["passage_id"] for item in beat_passages_known if item["book_id"] == book_id]
            row = {
                "book_id": book_id,
                "title": book_by_id.get(book_id, {}).get("title", book_id),
                "retained_count": retained_count,
                "insight_count": insight_count,
                "supporting_count": planning_count - insight_count,
                "planning_count": planning_count,
                "beat_ref_count": len(beat_book_refs),
                "beat_unique_count": len(set(beat_book_refs)),
            }
            if row["retained_count"] or row["planning_count"] or row["beat_ref_count"]:
                book_rows.append(row)

        episode_dir = run_dir / "episodes" / str(episode_number)
        episode_script = _load_optional_json(episode_dir / "episode_script.json") or {}
        spoken_script = _load_optional_json(episode_dir / "spoken_script.json") or {}
        render_manifest = _load_optional_json(episode_dir / "render_manifest.json") or {}
        plan_alignment = _load_optional_json(episode_dir / "plan_alignment_report.json")
        episode_framing = _load_optional_json(episode_dir / "episode_framing.json")
        audio_manifest = _load_optional_json(episode_dir / "audio_manifest.json")
        realization_payload = realization_by_episode.get(episode_number)
        artifact_links = {}
        for name in (
            "episode_script.json",
            "spoken_script.json",
            "render_manifest.json",
            "plan_alignment_report.json",
            "episode_framing.json",
            "audio_manifest.json",
            "episode.mp3",
        ):
            target = episode_dir / name
            if target.exists():
                artifact_links[name] = _rel_link(output_path, target)
        spoken_word_count = _episode_spoken_word_count(episode_dir)
        episode_reports.append(
            {
                "episode_number": episode_number,
                "title": assignment.get("title", f"Episode {episode_number}"),
                "slug": f"episode-{episode_number}-{_slugify(assignment.get('title', f'episode-{episode_number}'))}",
                "assignment": assignment,
                "episode_plan": episode_plan,
                "arc_detail": arc_detail_by_episode.get(episode_number),
                "merged_narrative": merged_by_id.get(assignment.get("merged_narrative_id")),
                "realization": realization_payload,
                "plan_alignment": plan_alignment,
                "episode_framing": episode_framing,
                "axis_rows": axis_rows,
                "axis_book_tables": axis_book_tables,
                "book_rows": book_rows,
                "planning_total_exposures": sum(row["planning_count"] for row in axis_rows),
                "planning_unique_count": len(planning_unique_ids),
                "insight_unique_count": len(selected_insight_passage_ids),
                "beat_ref_count": len(beat_passage_refs),
                "beat_unique_count": len(beat_unique_ids),
                "unknown_beat_passage_count": len([p_id for p_id in beat_unique_ids if p_id not in passage_lookup]),
                "beat_count": len(episode_plan.get("beats", [])),
                "script_word_count": episode_script.get("total_word_count"),
                "spoken_word_count": spoken_word_count,
                "render_segments": render_manifest.get("total_segments"),
                "render_estimated_duration_seconds": render_manifest.get("estimated_duration_seconds"),
                "artifact_links": artifact_links,
                "realization_problems": realization_payload.get("problem_count") if realization_payload else None,
            }
        )

    axis_reports: list[dict[str, Any]] = []
    metrics_per_axis = retrieval_metrics.get("per_axis", {})
    util_per_axis = passage_utilization.get("per_axis", {})
    for axis_id in axis_order:
        retrieval = retrieval_artifacts.get(axis_id, {})
        metric = metrics_per_axis.get(axis_id, {})
        utilization = util_per_axis.get(axis_id, {})
        episode_rows = []
        for episode in episode_reports:
            row = next((item for item in episode["axis_rows"] if item["axis_id"] == axis_id), None)
            if row:
                episode_rows.append({"episode_number": episode["episode_number"], "title": episode["title"], **row})
        per_book_rows = []
        for book_payload in retrieval.get("books", []):
            used_candidates = [item for item in book_payload.get("candidates", []) if item.get("used")]
            phase_counts = collections.Counter(item.get("selection_phase") or "unknown" for item in used_candidates)
            retained_count = sum(
                1 for item in passages_by_axis.get(axis_id, []) if item["book_id"] == book_payload["book_id"]
            )
            per_book_rows.append(
                {
                    "title": book_payload.get("title", book_payload.get("book_id")),
                    "chunk_count": book_payload.get("chunk_count"),
                    "retrieval_depth_budget": book_payload.get("retrieval_depth_budget"),
                    "admission_quota": book_payload.get("admission_quota"),
                    "eligible_total_count": book_payload.get("eligible_total_count"),
                    "eligible_above_threshold_count": book_payload.get("eligible_above_threshold_count"),
                    "retained_count": retained_count,
                    "selected_above_threshold_count": book_payload.get("selected_above_threshold_count"),
                    "selected_backfill_count": book_payload.get("selected_backfill_count"),
                    "selected_spillover_count": book_payload.get("selected_spillover_count"),
                    "underfill_count": book_payload.get("underfill_count"),
                    "phase_counts": dict(phase_counts),
                }
            )
        axis_reports.append(
            {
                "axis_id": axis_id,
                "slug": f"axis-{axis_id.lower()}-{_slugify(axis_by_id.get(axis_id, {}).get('name', axis_id))}",
                "axis": axis_by_id.get(axis_id, {"axis_id": axis_id, "name": axis_id, "description": ""}),
                "retained_count": len(passages_by_axis.get(axis_id, [])),
                "retrieval_metric": metric,
                "utilization": utilization,
                "retrieval": retrieval,
                "episode_rows": episode_rows,
                "per_book_rows": per_book_rows,
                "raw_artifact_link": _rel_link(output_path, passage_extraction_dir / f"retrieval_candidates_{axis_id.lower()}.json")
                if (passage_extraction_dir / f"retrieval_candidates_{axis_id.lower()}.json").exists()
                else None,
            }
        )

    overview = {
        "project_id": project.get("project_id"),
        "theme": project.get("theme"),
        "theme_elaboration": project.get("theme_elaboration"),
        "status": project.get("status"),
        "book_count": len(project.get("books", [])),
        "axis_count": len(axis_order),
        "episode_count": project.get("episode_count"),
        "total_words": sum(book.get("total_words", 0) for book in project.get("books", [])),
        "total_chunks": sum(book.get("chunk_count", 0) for book in project.get("books", [])),
        "retained_passages": corpus.get("total_passages"),
        "planning_unique_passages": len(planning_unique_ids_all),
        "planning_total_exposures": sum(item["planning_total_exposures"] for item in episode_reports),
        "beat_unique_passages": len(beat_unique_ids_all),
        "beat_unique_passages_known": len([passage_id for passage_id in beat_unique_ids_all if passage_id in passage_lookup]),
        "insight_count": len(synthesis_map.get("insights", [])),
        "merged_narrative_count": len(synthesis_map.get("merged_narratives", [])),
        "narrative_thread_count": len(synthesis_map.get("narrative_threads", [])),
        "run_start": run_start,
        "run_end": run_end,
        "run_elapsed_seconds": run_elapsed_seconds,
        "missing_plan_passage_ids": sorted(missing_plan_passage_ids),
    }
    return {
        "run_dir": run_dir,
        "project": project,
        "overview": overview,
        "stage_records": stage_records,
        "stage_families": stage_families,
        "book_by_id": book_by_id,
        "axis_reports": axis_reports,
        "episode_reports": episode_reports,
        "retrieval_metrics": retrieval_metrics,
        "passage_utilization": passage_utilization,
        "strategy": strategy,
    }


def _render_html(report: dict[str, Any], output_path: Path) -> str:
    run_dir = report["run_dir"]
    project = report["project"]
    overview = report["overview"]
    axis_reports = report["axis_reports"]
    episode_reports = report["episode_reports"]
    stage_records = report["stage_records"]
    stage_families = report["stage_families"]
    retrieval_metrics = report["retrieval_metrics"]
    passage_utilization = report["passage_utilization"]
    strategy = report["strategy"]

    max_retained_axis = max((item["retained_count"] for item in axis_reports), default=0)
    max_planned_axis = max((item.get("utilization", {}).get("planned_count", 0) for item in axis_reports), default=0)
    max_episode_planning = max((item["planning_total_exposures"] for item in episode_reports), default=0)
    max_episode_usage = max((item["beat_unique_count"] for item in episode_reports), default=0)

    parts: list[str] = []
    add = parts.append
    add("<!doctype html><html lang='en'><head><meta charset='utf-8' />")
    add("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    add(f"<title>Run Statistics — {_escape(overview['project_id'])}</title>")
    add(
        """
<style>
:root {
  --paper: #fbf8f2;
  --paper-2: #fffdf8;
  --ink: #201b16;
  --muted: #6f655d;
  --line: #d8c9b9;
  --accent: #7c3f20;
  --accent-2: #b7672b;
  --accent-3: #245d66;
  --ok: #2f6a47;
  --warn: #9a5c12;
  --bad: #9b2f2f;
  --chip: #efe2d3;
  --shadow: 0 18px 40px rgba(64, 38, 18, 0.08);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  font-family: var(--sans);
  background:
    radial-gradient(circle at top left, rgba(255,255,255,0.9), rgba(255,255,255,0) 35%),
    linear-gradient(180deg, #f7f3ec 0%, #efe7dc 100%);
}
a { color: var(--accent); }
header {
  padding: 2.2rem 2.5rem 1.2rem;
  border-bottom: 1px solid var(--line);
  background: rgba(251, 248, 242, 0.92);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 10;
}
h1 { margin: 0; font-size: 2.3rem; line-height: 1.1; }
.subtitle { margin-top: 0.55rem; color: var(--muted); font-size: 1rem; }
nav { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
nav a { text-decoration: none; border: 1px solid var(--line); background: var(--paper-2); padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.9rem; }
main { max-width: 1600px; margin: 0 auto; padding: 1.5rem 2.5rem 3rem; }
section { margin-top: 2rem; }
h2 { margin: 0 0 0.8rem 0; font-size: 1.55rem; }
h3 { margin: 0 0 0.7rem 0; font-size: 1.15rem; color: var(--accent); }
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.two-col { display: grid; gap: 1rem; grid-template-columns: 1.3fr 1fr; }
.card, details.panel { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); }
.card { padding: 1rem 1.1rem; }
.stat { display: flex; justify-content: space-between; gap: 0.75rem; padding: 0.35rem 0; border-bottom: 1px dashed rgba(124, 63, 32, 0.18); }
.stat:last-child { border-bottom: 0; }
.label { color: var(--muted); }
.value { font-weight: 700; }
.kicker { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent-2); margin-bottom: 0.5rem; }
.table-wrap { overflow: auto; }
table { width: 100%; border-collapse: collapse; background: var(--paper); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: var(--shadow); }
th, td { padding: 0.58rem 0.7rem; border-bottom: 1px solid rgba(124, 63, 32, 0.14); text-align: left; vertical-align: top; font-size: 0.93rem; }
th { background: #efe3d6; color: var(--accent); text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.78rem; position: sticky; top: 0; }
tbody tr:hover { background: #fffdf8; }
.mono { font-family: var(--mono); font-size: 0.82rem; }
.chip { display: inline-block; padding: 0.16rem 0.5rem; border-radius: 999px; background: var(--chip); color: var(--accent); font-size: 0.76rem; margin: 0.1rem 0.2rem 0.1rem 0; border: 1px solid rgba(124,63,32,0.1); }
.chips { line-height: 1.7; }
.bar { width: 120px; height: 8px; background: #e8dccf; border-radius: 999px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent-2), var(--accent)); }
.callout { background: linear-gradient(135deg, rgba(183,103,43,0.10), rgba(36,93,102,0.06)); border: 1px solid rgba(124,63,32,0.18); border-radius: 14px; padding: 1rem 1.1rem; }
details.panel { padding: 0.85rem 1rem; margin-top: 1rem; }
details.panel summary { cursor: pointer; list-style: none; font-weight: 700; color: var(--accent); }
details.panel[open] summary { margin-bottom: 0.85rem; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; background: #f6eee6; border: 1px solid var(--line); border-radius: 12px; padding: 0.85rem; font-size: 0.82rem; line-height: 1.4; }
.small { font-size: 0.85rem; color: var(--muted); }
.ok { color: var(--ok); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
ul.flat { margin: 0.35rem 0 0 1rem; padding: 0; }
ul.flat li { margin: 0.18rem 0; }
@media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }
</style>
"""
    )
    add("</head><body>")
    add("<header>")
    add(f"<h1>Run Statistics — {_escape(overview['project_id'])}</h1>")
    add(
        f"<div class='subtitle'>{_escape(overview['theme'])} | {_escape(overview.get('theme_elaboration') or '')}<br />Generated from <span class='mono'>{_escape(str(run_dir))}</span></div>"
    )
    add("<nav><a href='#overview'>Overview</a><a href='#timeline'>Timeline</a><a href='#books'>Books</a><a href='#axes'>Axes</a><a href='#episodes'>Episodes</a><a href='#method'>Method</a></nav>")
    add("</header><main>")

    add("<section id='overview'><h2>Overview</h2><div class='grid'>")
    cards = [
        (
            "Run",
            [
                ("Project", overview["project_id"]),
                ("Status", overview.get("status")),
                ("Start", overview["run_start"].isoformat() if overview["run_start"] else "—"),
                ("End", overview["run_end"].isoformat() if overview["run_end"] else "—"),
                ("Elapsed", _fmt_seconds(overview.get("run_elapsed_seconds"))),
                ("Stages Logged", len(stage_records)),
            ],
        ),
        (
            "Corpus",
            [
                ("Books", overview.get("book_count")),
                ("Axes", overview.get("axis_count")),
                ("Episodes", overview.get("episode_count")),
                ("Book Words", overview.get("total_words")),
                ("Book Chunks", overview.get("total_chunks")),
                ("Retained Passages", overview.get("retained_passages")),
            ],
        ),
        (
            "Synthesis",
            [
                ("Insights", overview.get("insight_count")),
                ("Merged Narratives", overview.get("merged_narrative_count")),
                ("Narrative Threads", overview.get("narrative_thread_count")),
                ("Strategy Type", strategy.get("strategy_type")),
                ("Recommended Episodes", strategy.get("recommended_episode_count")),
                ("Series Arc", (strategy.get("series_arc", "")[:70] + "…") if strategy.get("series_arc") else "—"),
            ],
        ),
        (
            "Passage Funnel",
            [
                ("Retained Unique", overview.get("retained_passages")),
                ("Planning Unique", overview.get("planning_unique_passages")),
                ("Planning Exposures", overview.get("planning_total_exposures")),
                ("Beat Unique Used", overview.get("beat_unique_passages_known")),
                ("Utilization Planned", passage_utilization.get("summary", {}).get("planned_count")),
                ("Cited", passage_utilization.get("summary", {}).get("cited_count")),
            ],
        ),
    ]
    for title, stats in cards:
        add("<div class='card'>")
        add(f"<div class='kicker'>{_escape(title)}</div>")
        for label, value in stats:
            display = _fmt_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            add(f"<div class='stat'><span class='label'>{_escape(label)}</span><span class='value'>{_escape(display)}</span></div>")
        add("</div>")
    add("</div>")
    add("<div class='callout' style='margin-top:1rem'>")
    add("<strong>How planning counts are computed.</strong> The pipeline does not persist the per-episode planning payload. This page reconstructs it from <span class='mono'>thematic_corpus.json</span>, <span class='mono'>synthesis_map.json</span>, and <span class='mono'>narrative_strategy.json</span> using the same rule as <span class='mono'>_select_episode_planning_passages</span>: for each assigned axis, planning receives all selected insight passages plus up to 60 additional supporting passages ranked by relevance, quotability, and a small multi-axis bonus.")
    if overview.get("missing_plan_passage_ids"):
        add(f"<div class='small warn' style='margin-top:0.7rem'>One or more beat passage IDs appear in the series plan but not in the retained corpus: <span class='mono'>{_escape(', '.join(overview['missing_plan_passage_ids']))}</span>.</div>")
    add("</div></section>")

    add("<section id='timeline'><h2>Stage Timeline</h2><div class='grid'>")
    for family, summary in stage_families.items():
        add("<div class='card'>")
        add(f"<div class='kicker'>{_escape(family)}</div>")
        add(f"<div class='stat'><span class='label'>Logged</span><span class='value'>{_escape(_fmt_number(summary['count']))}</span></div>")
        add(f"<div class='stat'><span class='label'>Completed</span><span class='value'>{_escape(_fmt_number(summary['completed']))}</span></div>")
        add(f"<div class='stat'><span class='label'>Total Time</span><span class='value'>{_escape(_fmt_seconds(summary['elapsed_seconds']))}</span></div>")
        add("</div>")
    add("</div><div class='table-wrap'><table><thead><tr><th>Stage</th><th>Family</th><th>Start</th><th>End</th><th>Elapsed</th><th>Input Summary</th><th>Output Summary</th></tr></thead><tbody>")
    for record in stage_records:
        add("<tr>")
        add(f"<td class='mono'>{_escape(record['stage'])}</td>")
        add(f"<td>{_escape(record['family'])}</td>")
        add(f"<td class='mono'>{_escape(record['start'].isoformat() if record['start'] else '—')}</td>")
        add(f"<td class='mono'>{_escape(record['end'].isoformat() if record['end'] else '—')}</td>")
        add(f"<td>{_escape(_fmt_seconds(record['elapsed_seconds']))}</td>")
        add(f"<td><details><summary>view</summary><pre>{_json_pretty(record['input_summary'])}</pre></details></td>")
        add(f"<td><details><summary>view</summary><pre>{_json_pretty(record['output_summary'])}</pre></details></td>")
        add("</tr>")
    add("</tbody></table></div></section>")

    add("<section id='books'><h2>Books</h2><div class='table-wrap'><table><thead><tr><th>Book</th><th>Author</th><th>Words</th><th>Chunks</th><th>Chapters</th><th>Retained</th><th>Planned</th><th>Plan Util</th><th>Avg Relevance</th><th>Quota vs Size</th></tr></thead><tbody>")
    metrics_per_book = retrieval_metrics.get("per_book", {})
    util_per_book = passage_utilization.get("per_book", {})
    for book in sorted(project.get("books", []), key=lambda item: item.get("title", "")):
        metrics = metrics_per_book.get(book["book_id"], {})
        util = util_per_book.get(book["book_id"], {})
        add("<tr>")
        add(f"<td>{_escape(book.get('title'))}</td>")
        add(f"<td>{_escape(book.get('author'))}</td>")
        add(f"<td>{_escape(_fmt_number(book.get('total_words')))}</td>")
        add(f"<td>{_escape(_fmt_number(book.get('chunk_count')))}</td>")
        add(f"<td>{_escape(_fmt_number(len(book.get('chapters', []))))}</td>")
        add(f"<td>{_escape(_fmt_number(util.get('retained_count')))}</td>")
        add(f"<td>{_escape(_fmt_number(util.get('planned_count')))}</td>")
        add(f"<td>{_escape(_fmt_pct(util.get('plan_utilization_ratio')))}</td>")
        add(f"<td>{_escape(_fmt_number(metrics.get('avg_relevance')))}</td>")
        add(f"<td>{_escape(_fmt_number(metrics.get('quota_minus_size_share')))}</td>")
        add("</tr>")
    add("</tbody></table></div></section>")

    add("<section id='axes'><h2>Axes</h2><div class='table-wrap'><table><thead><tr><th>Axis</th><th>Name</th><th>Retained</th><th>Planned</th><th>Plan Util</th><th>Books</th><th>Candidates</th><th>Post-Rerank</th><th>Avg Relevance</th><th>Avg Quotability</th><th>Episodes</th></tr></thead><tbody>")
    for item in axis_reports:
        metric = item.get("retrieval_metric", {})
        utilization = item.get("utilization", {})
        books_represented = metric.get("books_represented")
        add("<tr>")
        add(f"<td><a href='#{_escape(item['slug'])}' class='mono'>{_escape(item['axis_id'])}</a></td>")
        add(f"<td>{_escape(item['axis'].get('name'))}</td>")
        add(f"<td>{_escape(_fmt_number(item['retained_count']))}<br />{_bar(item['retained_count'], max_retained_axis)}</td>")
        add(f"<td>{_escape(_fmt_number(utilization.get('planned_count')))}<br />{_bar(utilization.get('planned_count', 0), max_planned_axis)}</td>")
        add(f"<td>{_escape(_fmt_pct(utilization.get('plan_utilization_ratio')))}</td>")
        add(f"<td>{_escape(_fmt_number(books_represented))}</td>")
        add(f"<td>{_escape(_fmt_number(metric.get('candidate_count')))}</td>")
        add(f"<td>{_escape(_fmt_number(metric.get('post_rerank_count')))}</td>")
        add(f"<td>{_escape(_fmt_number(metric.get('avg_relevance_score')))}</td>")
        add(f"<td>{_escape(_fmt_number(metric.get('avg_quotability_score')))}</td>")
        add(f"<td>{_escape(' '.join(f"E{row['episode_number']}" for row in item['episode_rows']) or '—')}</td>")
        add("</tr>")
    add("</tbody></table></div>")
    for item in axis_reports:
        metric = item.get("retrieval_metric", {})
        utilization = item.get("utilization", {})
        retrieval = item.get("retrieval", {})
        axis = item["axis"]
        add(f"<details class='panel' id='{_escape(item['slug'])}'>")
        add(f"<summary>{_escape(item['axis_id'])} — {_escape(axis.get('name'))} | retained {_escape(_fmt_number(item['retained_count']))} | planned {_escape(_fmt_number(utilization.get('planned_count')))} | episodes {_escape(', '.join('E'+str(row['episode_number']) for row in item['episode_rows']) or 'none')}</summary>")
        add("<div class='grid'>")
        add("<div class='card'><div class='kicker'>Axis Summary</div>")
        add(f"<div class='small'>{_escape(axis.get('description', ''))}</div>")
        if axis.get("keywords"):
            add("<div class='chips' style='margin-top:0.6rem'>")
            for keyword in axis.get("keywords", []):
                add(f"<span class='chip'>{_escape(keyword)}</span>")
            add("</div>")
        add("</div>")
        add("<div class='card'><div class='kicker'>Retrieval Metrics</div>")
        for label, value in (
            ("Candidate Count", metric.get("candidate_count")),
            ("Post-Rerank", metric.get("post_rerank_count")),
            ("Rehydrated", metric.get("rehydrated_count")),
            ("Full Text Coverage", _fmt_pct(metric.get("full_text_coverage_ratio"))),
            ("Books Represented", metric.get("books_represented")),
            ("Plan Utilization", _fmt_pct(utilization.get("plan_utilization_ratio"))),
        ):
            add(f"<div class='stat'><span class='label'>{_escape(label)}</span><span class='value'>{_escape(_fmt_number(value) if not isinstance(value, str) else value)}</span></div>")
        add("</div>")
        add("<div class='card'><div class='kicker'>Retrieval Policy</div>")
        for label, value in (
            ("Candidate Budget Target", retrieval.get("axis_candidate_budget_target")),
            ("Candidate Budget Effective", retrieval.get("axis_candidate_budget_effective")),
            ("Min Per Book", retrieval.get("passage_retrieval_min_per_book")),
            ("Max Per Book", retrieval.get("passage_retrieval_max_per_book")),
            ("Soft Threshold", retrieval.get("soft_threshold")),
            ("Relevance Power", retrieval.get("retrieval_relevance_power")),
        ):
            add(f"<div class='stat'><span class='label'>{_escape(label)}</span><span class='value'>{_escape(_fmt_number(value) if not isinstance(value, str) else value)}</span></div>")
        add("</div></div>")
        if item["per_book_rows"]:
            add("<h3>Per-Book Retrieval for This Axis</h3><div class='table-wrap'><table><thead><tr><th>Book</th><th>Chunks</th><th>Depth Budget</th><th>Quota</th><th>Eligible</th><th>Eligible Above Threshold</th><th>Retained</th><th>Selected Above Threshold</th><th>Backfill</th><th>Spillover</th><th>Underfill</th><th>Selection Phases</th></tr></thead><tbody>")
            for row in sorted(item["per_book_rows"], key=lambda value: value["title"]):
                phases = ", ".join(f"{key}:{value}" for key, value in sorted(row["phase_counts"].items())) or "—"
                add("<tr>")
                add(f"<td>{_escape(row['title'])}</td>")
                add(f"<td>{_escape(_fmt_number(row['chunk_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['retrieval_depth_budget']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['admission_quota']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['eligible_total_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['eligible_above_threshold_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['retained_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['selected_above_threshold_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['selected_backfill_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['selected_spillover_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['underfill_count']))}</td>")
                add(f"<td class='mono'>{_escape(phases)}</td></tr>")
            add("</tbody></table></div>")
        if item["episode_rows"]:
            add("<h3>Episode Usage of This Axis</h3><div class='table-wrap'><table><thead><tr><th>Episode</th><th>Insight Selected</th><th>Supporting Added</th><th>Planning Available</th><th>Beat Unique Used</th><th>Beat Refs</th><th>Books in Planning</th></tr></thead><tbody>")
            for row in item["episode_rows"]:
                add("<tr>")
                add(f"<td><a href='#episode-{row['episode_number']}-{_slugify(row['title'])}'>E{row['episode_number']} — {_escape(row['title'])}</a></td>")
                add(f"<td>{_escape(_fmt_number(row['insight_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['supporting_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['planning_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['beat_unique_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['beat_ref_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['books_represented']))}</td></tr>")
            add("</tbody></table></div>")
        retrieval_summary = {
            "axis_id": item["axis_id"],
            "axis_name": axis.get("name"),
            "candidate_budget_target": retrieval.get("axis_candidate_budget_target"),
            "candidate_budget_effective": retrieval.get("axis_candidate_budget_effective"),
            "soft_threshold": retrieval.get("soft_threshold"),
            "relevance_power": retrieval.get("retrieval_relevance_power"),
            "allocation_policy": retrieval.get("allocation_policy"),
            "admission_quota_by_book": retrieval.get("admission_quota_by_book"),
            "per_book": item["per_book_rows"],
        }
        add("<details class='panel'><summary>Retrieval Summary JSON and Raw Artifact Link</summary>")
        if item.get("raw_artifact_link"):
            add(f"<div class='small' style='margin-bottom:0.7rem'><a class='mono' href='{_escape(item['raw_artifact_link'])}'>{_escape(item['raw_artifact_link'])}</a></div>")
        add(f"<pre>{_json_pretty(retrieval_summary)}</pre></details></details>")
    add("</section>")

    add("<section id='episodes'><h2>Episodes</h2><div class='table-wrap'><table><thead><tr><th>Episode</th><th>Axes</th><th>Insights</th><th>Planning Exposures</th><th>Planning Unique</th><th>Beat Unique</th><th>Beat Refs</th><th>Beats</th><th>Script Words</th><th>Spoken Words</th><th>Render Segments</th><th>Duration</th></tr></thead><tbody>")
    for item in episode_reports:
        add("<tr>")
        add(f"<td><a href='#{_escape(item['slug'])}'>E{item['episode_number']} — {_escape(item['title'])}</a></td>")
        add(f"<td>{_escape(_fmt_number(len(item['axis_rows'])))}</td>")
        add(f"<td>{_escape(_fmt_number(len(item['assignment'].get('insight_ids', []))))}</td>")
        add(f"<td>{_escape(_fmt_number(item['planning_total_exposures']))}<br />{_bar(item['planning_total_exposures'], max_episode_planning)}</td>")
        add(f"<td>{_escape(_fmt_number(item['planning_unique_count']))}</td>")
        add(f"<td>{_escape(_fmt_number(item['beat_unique_count']))}<br />{_bar(item['beat_unique_count'], max_episode_usage)}</td>")
        add(f"<td>{_escape(_fmt_number(item['beat_ref_count']))}</td>")
        add(f"<td>{_escape(_fmt_number(item['beat_count']))}</td>")
        add(f"<td>{_escape(_fmt_number(item['script_word_count']))}</td>")
        add(f"<td>{_escape(_fmt_number(item['spoken_word_count']))}</td>")
        add(f"<td>{_escape(_fmt_number(item['render_segments']))}</td>")
        add(f"<td>{_escape(_fmt_seconds(item['render_estimated_duration_seconds']))}</td></tr>")
    add("</tbody></table></div>")
    for item in episode_reports:
        add(f"<details class='panel' id='{_escape(item['slug'])}'>")
        add(f"<summary>E{item['episode_number']} — {_escape(item['title'])} | planning exposures {_escape(_fmt_number(item['planning_total_exposures']))} | beat unique {_escape(_fmt_number(item['beat_unique_count']))} | words {_escape(_fmt_number(item['spoken_word_count']))}</summary>")
        add("<div class='two-col'>")
        add("<div class='card'><div class='kicker'>Episode Assignment</div>")
        add(f"<div class='small'><strong>Driving question:</strong> {_escape(item['assignment'].get('driving_question', '—'))}</div>")
        add(f"<div class='small' style='margin-top:0.5rem'><strong>Thematic focus:</strong> {_escape(item['assignment'].get('thematic_focus', '—'))}</div>")
        add(f"<div class='small' style='margin-top:0.5rem'><strong>Episode strategy:</strong> {_escape(item['assignment'].get('episode_strategy', '—'))}</div>")
        add("<div class='chips' style='margin-top:0.7rem'>")
        for axis in item["assignment"].get("axes", []):
            axis_id = axis["axis_id"] if isinstance(axis, dict) else str(axis)
            desc = axis.get("description", axis_id) if isinstance(axis, dict) else axis_id
            add(f"<span class='chip'>{_escape(axis_id)}: {_escape(desc)}</span>")
        add("</div></div>")
        add("<div class='card'><div class='kicker'>Output Summary</div>")
        for label, value in (
            ("Beats", item["beat_count"]),
            ("Planning Exposures", item["planning_total_exposures"]),
            ("Planning Unique", item["planning_unique_count"]),
            ("Beat Unique Used", item["beat_unique_count"]),
            ("Beat Refs", item["beat_ref_count"]),
            ("Script Words", item["script_word_count"]),
            ("Spoken Words", item["spoken_word_count"]),
            ("Render Segments", item["render_segments"]),
            ("Render Duration", _fmt_seconds(item["render_estimated_duration_seconds"])),
        ):
            add(f"<div class='stat'><span class='label'>{_escape(label)}</span><span class='value'>{_escape(_fmt_number(value) if not isinstance(value, str) else value)}</span></div>")
        add("</div></div>")
        if item.get("merged_narrative") or item.get("arc_detail"):
            add("<div class='grid' style='margin-top:1rem'>")
            if item.get("merged_narrative"):
                merged = item["merged_narrative"]
                add("<div class='card'><div class='kicker'>Merged Narrative</div>")
                add(f"<div class='small'><strong>{_escape(merged.get('topic', ''))}</strong></div>")
                add(f"<div class='small' style='margin-top:0.5rem'>{_escape(merged.get('narrative', ''))}</div></div>")
            if item.get("arc_detail"):
                detail = item["arc_detail"]
                add("<div class='card'><div class='kicker'>Arc Detail</div>")
                for label, key in (
                    ("Arc Summary", "arc_summary"),
                    ("Narrative Stakes", "narrative_stakes"),
                    ("Payoff Shape", "payoff_shape"),
                    ("Unresolved Questions", "unresolved_questions"),
                ):
                    if detail.get(key):
                        add(f"<div class='small' style='margin-top:0.45rem'><strong>{_escape(label)}:</strong> {_escape(detail.get(key))}</div>")
                add("</div>")
            add("</div>")
        add("<h3>Axis Funnel Inside This Episode</h3><div class='table-wrap'><table><thead><tr><th>Axis</th><th>Retained in Corpus</th><th>Insight Selected</th><th>Supporting Added</th><th>Planning Available</th><th>Beat Unique Used</th><th>Beat Refs</th><th>Books in Planning</th></tr></thead><tbody>")
        for row in item["axis_rows"]:
            add("<tr>")
            add(f"<td><a href='#axis-{row['axis_id'].lower()}-{_slugify(row['axis_name'])}' class='mono'>{_escape(row['axis_id'])}</a><br />{_escape(row['axis_name'])}</td>")
            add(f"<td>{_escape(_fmt_number(row['retained_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['insight_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['supporting_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['planning_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['beat_unique_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['beat_ref_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['books_represented']))}</td></tr>")
        add("</tbody></table></div>")
        add("<h3>Book Funnel Across Assigned Axes</h3><div class='table-wrap'><table><thead><tr><th>Book</th><th>Retained Across Axes</th><th>Insight Selected</th><th>Supporting Added</th><th>Planning Available</th><th>Beat Unique Used</th><th>Beat Refs</th></tr></thead><tbody>")
        for row in item["book_rows"]:
            add("<tr>")
            add(f"<td>{_escape(row['title'])}</td>")
            add(f"<td>{_escape(_fmt_number(row['retained_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['insight_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['supporting_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['planning_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['beat_unique_count']))}</td>")
            add(f"<td>{_escape(_fmt_number(row['beat_ref_count']))}</td></tr>")
        add("</tbody></table></div>")
        for axis_table in item["axis_book_tables"]:
            add("<details class='panel'>")
            add(f"<summary>{_escape(axis_table['axis_id'])} — {_escape(axis_table['axis_name'])}: per-book counts at each step</summary>")
            add("<div class='table-wrap'><table><thead><tr><th>Book</th><th>Retained</th><th>Insight Selected</th><th>Supporting Added</th><th>Planning Available</th><th>Beat Unique Used</th><th>Beat Refs</th></tr></thead><tbody>")
            for row in axis_table["rows"]:
                add("<tr>")
                add(f"<td>{_escape(row['title'])}</td>")
                add(f"<td>{_escape(_fmt_number(row['retained_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['insight_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['supporting_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['planning_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['beat_unique_count']))}</td>")
                add(f"<td>{_escape(_fmt_number(row['beat_ref_count']))}</td></tr>")
            add("</tbody></table></div></details>")
        add("<h3>Realization and Artifacts</h3><div class='grid'>")
        issue_class = "bad" if (item.get("realization") or {}).get("has_issues") else "ok"
        add("<div class='card'><div class='kicker'>Plan Realization</div>")
        add(f"<div class='stat'><span class='label'>Has Issues</span><span class='value {issue_class}'>{_escape(str((item.get('realization') or {}).get('has_issues', False)))}</span></div>")
        add(f"<div class='stat'><span class='label'>Problem Count</span><span class='value'>{_escape(_fmt_number(item.get('realization_problems')))}</span></div>")
        add(f"<div class='stat'><span class='label'>Unknown Beat Passages</span><span class='value'>{_escape(_fmt_number(item['unknown_beat_passage_count']))}</span></div></div>")
        add("<div class='card'><div class='kicker'>Artifacts</div>")
        if item["artifact_links"]:
            for name, link in item["artifact_links"].items():
                add(f"<div class='stat'><span class='label'>{_escape(name)}</span><span class='value'><a class='mono' href='{_escape(link)}'>{_escape(link)}</a></span></div>")
        else:
            add("<div class='small'>Episode artifact files are unavailable.</div>")
        add("</div></div>")
        realization = item.get("realization")
        if realization and realization.get("insights"):
            add("<div class='table-wrap' style='margin-top:1rem'><table><thead><tr><th>Insight</th><th>Status</th><th>Assigned Passages</th><th>Realized</th><th>Expected Min</th><th>Missing Passage IDs</th></tr></thead><tbody>")
            for insight in realization.get("insights", []):
                add("<tr>")
                add(f"<td>{_escape(insight.get('title'))}</td>")
                add(f"<td>{_escape(insight.get('status'))}</td>")
                add(f"<td>{_escape(_fmt_number(insight.get('assigned_passage_count')))}</td>")
                add(f"<td>{_escape(_fmt_number(insight.get('realized_count')))}</td>")
                add(f"<td>{_escape(_fmt_number(insight.get('expected_min')))}</td>")
                add(f"<td class='mono'>{_escape(', '.join(insight.get('missing_passage_ids', [])) or '—')}</td></tr>")
            add("</tbody></table></div>")
        add("<details class='panel'><summary>Raw Assignment JSON</summary>")
        add(f"<pre>{_json_pretty(item['assignment'])}</pre></details>")
        if item.get("plan_alignment") is not None:
            add("<details class='panel'><summary>Raw Plan Alignment JSON</summary>")
            add(f"<pre>{_json_pretty(item['plan_alignment'])}</pre></details>")
        add("</details>")
    add("</section>")

    add("<section id='method'><h2>Method</h2><div class='callout'>")
    add("<p>This page is generated entirely from persisted run artifacts. It does not require changes to the main pipeline.</p>")
    add("<ul class='flat'>")
    add("<li><span class='mono'>thematic_project.json</span>, <span class='mono'>thematic_axes.json</span>, <span class='mono'>thematic_corpus.json</span>: core corpus and axis state.</li>")
    add("<li><span class='mono'>synthesis_map.json</span> and <span class='mono'>narrative_strategy.json</span>: selected insights and episode assignments.</li>")
    add("<li><span class='mono'>series_plan.json</span>: beat-level passage references for downstream usage counts.</li>")
    add("<li><span class='mono'>retrieval_metrics.json</span>, <span class='mono'>passage_utilization.json</span>, and <span class='mono'>stage_artifacts/passage_extraction/retrieval_candidates_axis_*.json</span>: retrieval and utilization details when present.</li>")
    add("<li><span class='mono'>episodes/&lt;N&gt;/*.json</span>: script, spoken, render, alignment, framing, and audio outputs when present.</li>")
    add("<li><span class='mono'>run.log</span>: stage timing and stage input/output summaries.</li>")
    add("</ul>")
    add("<p><strong>Important distinction:</strong> planning exposures count how many passages were offered to the planner across assigned axes. This is larger than the unique passage count eventually used in beats, because most available passages are never used.</p></div></section>")
    add("</main></body></html>")
    return "".join(parts)


def generate_run_stats(run_dir: Path, output_path: Path) -> Path:
    report = _compute_reports(run_dir, output_path)
    output_path.write_text(_render_html(report, output_path))
    return output_path


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir / "run-stats.html"
    try:
        _check_run_dir(run_dir)
        written = generate_run_stats(run_dir, output_path)
    except RunStatsError as exc:
        print(f"Run stats generation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"Run stats generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
