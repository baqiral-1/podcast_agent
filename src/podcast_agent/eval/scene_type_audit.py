"""Scene-type to prose audit helpers and standalone HTML rendering."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

DEFAULT_SCENE_AUDIT_ARTIFACTS = ("style_audited_script.json", "episode_script.json")

_ROLE_SCORES = {
    "shock": 3.0,
    "action": 2.0,
    "fallout": 1.0,
    "reaction": 1.0,
    "contestation": 1.0,
    "actor_setup": 0.0,
    "context_setup": 0.0,
    "implication": -0.5,
}
_VERDICT_ORDER = {
    "strong": 0,
    "solid": 1,
    "mixed": 2,
    "weak": 3,
    "close-only": 4,
}


class SceneTypeAuditError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _check_run_dir(run_dir: Path) -> None:
    if not (run_dir / "series_plan.json").exists():
        raise SceneTypeAuditError("Run directory is missing series_plan.json")
    if not (run_dir / "episodes").exists():
        raise SceneTypeAuditError("Run directory is missing episodes/")


def _choose_script_path(episode_dir: Path, script_artifacts: Sequence[str]) -> Path | None:
    for name in script_artifacts:
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    return None


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split("\n\n") if part.strip()]


def _excerpt(text: str, *, max_paragraphs: int = 2, max_chars: int = 700) -> str:
    parts = _paragraphs(text)
    if not parts:
        return ""
    excerpt = "\n\n".join(parts[:max_paragraphs]).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    clipped = excerpt[: max_chars - 1].rsplit(" ", 1)[0].rstrip()
    return f"{clipped}..."


def _render_paragraphs_html(text: str) -> str:
    parts = _paragraphs(text)
    if not parts:
        return "<p class=\"muted\">No text available.</p>"
    return "".join(f"<p>{escape(part)}</p>" for part in parts)


def _mean_or_zero(values: Iterable[float]) -> float:
    collected = list(values)
    return mean(collected) if collected else 0.0


def _evaluate_section(
    *,
    roles: Sequence[str],
    jobs: Sequence[str],
    word_count: int,
    host_phase_collapse: bool,
    section_style_warnings: Sequence[str],
    episode_warning_hits: Sequence[str],
) -> dict[str, Any]:
    score = 0.0
    strengths: list[str] = []
    weaknesses: list[str] = []

    for role in roles:
        score += _ROLE_SCORES.get(role, 0.0)

    if "shock" in roles:
        strengths.append("Shock card gives the prose a body-level turn instead of a purely analytic one.")
    if "action" in roles:
        strengths.append("Action card gives the section a concrete gesture, vote, march, arrest, or encounter to write from.")
    if "fallout" in roles:
        strengths.append("Fallout card helps the prose cash out consequences instead of only describing the event.")
    if "reaction" in roles:
        strengths.append("Reaction card helps the prose localize the event in a witness, institution, or audience response.")
    if "contestation" in roles:
        strengths.append("Contestation card keeps opposition or argument visible on the page.")

    if "implication" in roles and any(
        role in roles for role in ("action", "shock", "fallout", "reaction", "contestation")
    ):
        score += 1.0
        strengths.append("Implication arrives after concrete scene work, so the argument feels earned.")

    if roles and roles[-1] == "implication":
        score += 0.5
        strengths.append("Section closes with a clear interpretive landing.")

    if roles and roles[0] in ("action", "shock", "context_setup", "actor_setup"):
        score += 0.5

    if len(set(roles)) >= 3:
        score += 0.5
        strengths.append("Role variety helps the section move instead of circling one register.")

    if roles.count("implication") >= max(2, len(roles) - 1):
        score -= 1.5
        weaknesses.append("Implication stack risks turning the section into explanation before it refreshes the scene.")

    if roles and all(role == "implication" for role in roles):
        score -= 1.0
        weaknesses.append("Pure implication section has very little scenic material to work with.")

    if roles and roles[0] in ("context_setup", "actor_setup") and not any(
        role in roles for role in ("action", "shock")
    ):
        weaknesses.append("Section opens in setup mode and never gets a decisive action or shock beat.")

    if word_count > 2200:
        score -= 1.0
        weaknesses.append("Long section carries a lot of synthesis load and risks flattening its own momentum.")
    elif word_count > 1800:
        score -= 0.5
        weaknesses.append("Section is long enough that even good scene material has to fight explanatory drag.")

    if host_phase_collapse:
        score -= 1.0
        weaknesses.append("Host-move diagnostics flag phase collapse here, which matches a softer landing on the page.")

    if section_style_warnings:
        score -= 0.5
        weaknesses.append("Style audit flagged the section directly, so there is already downstream evidence of load or softness.")

    if episode_warning_hits:
        score -= 0.5
        weaknesses.append("Episode-level style warnings mention this section by name.")

    close_only = bool(jobs) and jobs[-1] == "close" and len(roles) == 1 and roles[0] == "implication"
    if close_only:
        verdict = "close-only"
        strengths.append("Short structural close works as a capstone, but it is not a good test case for scene-type performance.")
    elif score >= 4.0:
        verdict = "strong"
    elif score >= 2.0:
        verdict = "solid"
    elif score >= 0.0:
        verdict = "mixed"
    else:
        verdict = "weak"

    return {
        "score": round(score, 2),
        "verdict": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "close_only": close_only,
    }


def _role_presence_stats(sections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        for role in set(section["scene_role_sequence"]):
            buckets[role].append(section)
    stats = []
    for role, matches in buckets.items():
        stats.append(
            {
                "role": role,
                "section_count": len(matches),
                "average_score": round(_mean_or_zero(match["score"] for match in matches), 2),
                "strong_count": sum(match["verdict"] == "strong" for match in matches),
                "weak_count": sum(match["verdict"] == "weak" for match in matches),
            }
        )
    return sorted(stats, key=lambda item: (-item["average_score"], item["role"]))


def _start_role_stats(sections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        if section["close_only"]:
            continue
        first_role = section["scene_role_sequence"][0]
        buckets[first_role].append(section)
    stats = []
    for role, matches in buckets.items():
        stats.append(
            {
                "role": role,
                "section_count": len(matches),
                "average_score": round(_mean_or_zero(match["score"] for match in matches), 2),
            }
        )
    return sorted(stats, key=lambda item: (-item["average_score"], item["role"]))


def _combo_stats(sections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        combo = section["scene_role_combo"]
        buckets[combo].append(section)
    stats = []
    for combo, matches in buckets.items():
        stats.append(
            {
                "combo": combo,
                "section_count": len(matches),
                "average_score": round(_mean_or_zero(match["score"] for match in matches), 2),
                "verdict_counts": dict(Counter(match["verdict"] for match in matches)),
                "sample_sections": [
                    {
                        "episode_number": match["episode_number"],
                        "section_id": match["section_id"],
                        "verdict": match["verdict"],
                    }
                    for match in matches[:4]
                ],
            }
        )
    return sorted(
        stats,
        key=lambda item: (-item["section_count"], -item["average_score"], item["combo"]),
    )


def _aggregate_findings(
    *,
    sections: Sequence[dict[str, Any]],
    role_stats: Sequence[dict[str, Any]],
    start_role_stats: Sequence[dict[str, Any]],
    combo_stats: Sequence[dict[str, Any]],
) -> list[str]:
    active_sections = [section for section in sections if not section["close_only"]]
    implication_heavy = [
        section for section in active_sections if section["scene_role_sequence"].count("implication") >= 2
    ]
    setup_without_event = [
        section
        for section in active_sections
        if section["scene_role_sequence"][0] in ("context_setup", "actor_setup")
        and not any(role in section["scene_role_sequence"] for role in ("action", "shock"))
    ]
    shock_stat = next((item for item in role_stats if item["role"] == "shock"), None)
    action_stat = next((item for item in role_stats if item["role"] == "action"), None)
    implication_stat = next((item for item in role_stats if item["role"] == "implication"), None)

    findings: list[str] = []
    if shock_stat is not None:
        findings.append(
            f"Sections containing shock cards are the strongest recurring pattern in the series: "
            f"{shock_stat['section_count']} sections, {shock_stat['average_score']:.2f} average score."
        )
    if action_stat is not None:
        findings.append(
            f"Action cards are the workhorse of the run: {action_stat['section_count']} sections contain them, "
            f"and those sections average {action_stat['average_score']:.2f}."
        )
    if implication_stat is not None and implication_heavy:
        findings.append(
            f"Implication is everywhere, but implication-heavy sections are materially weaker: "
            f"{len(implication_heavy)} active sections with 2+ implication cards average "
            f"{_mean_or_zero(section['score'] for section in implication_heavy):.2f}."
        )
    if setup_without_event:
        findings.append(
            f"Sections that open in context or actor setup and never pick up an action/shock beat are the clearest weak structural family: "
            f"{len(setup_without_event)} sections, {_mean_or_zero(section['score'] for section in setup_without_event):.2f} average score."
        )
    recurring_good = [
        item for item in combo_stats if item["section_count"] >= 2 and item["average_score"] >= 5.0
    ][:3]
    if recurring_good:
        findings.append(
            "Recurring combinations that read cleanly: "
            + "; ".join(
                f"{item['combo']} ({item['section_count']} sections, {item['average_score']:.2f} avg)"
                for item in recurring_good
            )
            + "."
        )
    recurring_weak = [
        item for item in combo_stats if item["section_count"] >= 2 and item["average_score"] <= 1.5
    ][:3]
    if recurring_weak:
        findings.append(
            "Recurring combinations that tend to flatten into essay mode: "
            + "; ".join(
                f"{item['combo']} ({item['section_count']} sections, {item['average_score']:.2f} avg)"
                for item in recurring_weak
            )
            + "."
        )
    if start_role_stats:
        best_start = start_role_stats[0]
        worst_start = start_role_stats[-1]
        findings.append(
            f"Opening role matters. Best average starting role: {best_start['role']} "
            f"({best_start['section_count']} sections, {best_start['average_score']:.2f}). "
            f"Weakest: {worst_start['role']} ({worst_start['section_count']} sections, {worst_start['average_score']:.2f})."
        )
    return findings


def build_scene_type_audit_payload(
    run_dir: Path,
    *,
    script_artifacts: Sequence[str] = DEFAULT_SCENE_AUDIT_ARTIFACTS,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    _check_run_dir(run_dir)

    series_plan = _load_json(run_dir / "series_plan.json")
    episodes_payload: list[dict[str, Any]] = []
    all_sections: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    job_counts: Counter[str] = Counter()
    total_scene_cards = 0

    for episode_plan in series_plan.get("episodes", []) or []:
        if not isinstance(episode_plan, dict):
            continue
        episode_number = episode_plan.get("episode_number")
        scene_cards = episode_plan.get("scene_cards") or []
        if not isinstance(episode_number, int) or not isinstance(scene_cards, list):
            continue

        episode_dir = run_dir / "episodes" / str(episode_number)
        script_path = _choose_script_path(episode_dir, script_artifacts)
        if script_path is None:
            raise SceneTypeAuditError(
                f"Episode {episode_number} is missing all requested script artifacts: "
                + ", ".join(script_artifacts)
            )

        script_payload = _load_json(script_path)
        prose_sections = script_payload.get("prose_sections") or []
        if not isinstance(prose_sections, list):
            raise SceneTypeAuditError(
                f"Episode {episode_number} script payload at {script_path} does not contain prose_sections."
            )

        scene_map = {str(scene["scene_id"]): scene for scene in scene_cards if isinstance(scene, dict) and scene.get("scene_id")}
        for scene in scene_cards:
            if not isinstance(scene, dict):
                continue
            role_counts[str(scene.get("scene_role") or "unknown")] += 1
            job_counts[str(scene.get("scene_job") or "unknown")] += 1
            total_scene_cards += 1

        host_payload = _load_optional_json(episode_dir / "host_moves_script_diagnostics.json") or {}
        continuity_payload = _load_optional_json(episode_dir / "continuity_script_diagnostics.json") or {}
        spine_payload = _load_optional_json(episode_dir / "spine_diagnostics.json") or {}
        style_payload = _load_optional_json(episode_dir / "style_audit_result.json") or {}

        sections_with_host_phase_collapse = set(host_payload.get("sections_with_host_phase_collapse") or [])
        sections_with_editorial_pressure = set(
            host_payload.get("sections_with_editorial_host_target_pressure") or []
        )
        style_sections = style_payload.get("sections") or []
        style_warning_map = {
            str(section.get("section_id")): [
                str(item)
                for item in (section.get("warnings") or [])
                if str(item).strip()
            ]
            for section in style_sections
            if isinstance(section, dict) and section.get("section_id")
        }
        episode_warnings = [str(item) for item in (style_payload.get("episode_warnings") or []) if str(item).strip()]

        episode_sections: list[dict[str, Any]] = []
        for prose_section in prose_sections:
            if not isinstance(prose_section, dict):
                continue
            section_id = str(prose_section.get("section_id") or "")
            scene_card_ids = [str(item) for item in (prose_section.get("scene_card_ids") or []) if str(item)]
            planned_cards = [scene_map[scene_id] for scene_id in scene_card_ids if scene_id in scene_map]
            roles = [str(card.get("scene_role") or "unknown") for card in planned_cards]
            jobs = [str(card.get("scene_job") or "unknown") for card in planned_cards]
            text = str(prose_section.get("text") or "")
            word_count = len(text.split())
            section_style_warnings = style_warning_map.get(section_id, [])
            episode_warning_hits = [
                warning for warning in episode_warnings if section_id and section_id in warning
            ]
            evaluation = _evaluate_section(
                roles=roles,
                jobs=jobs,
                word_count=word_count,
                host_phase_collapse=section_id in sections_with_host_phase_collapse,
                section_style_warnings=section_style_warnings,
                episode_warning_hits=episode_warning_hits,
            )

            scene_card_summaries = []
            for card in planned_cards:
                scene_card_summaries.append(
                    {
                        "scene_id": str(card.get("scene_id") or ""),
                        "title": str(card.get("title") or ""),
                        "scene_role": str(card.get("scene_role") or ""),
                        "scene_job": str(card.get("scene_job") or ""),
                        "beat_change": str(card.get("beat_change") or ""),
                        "entry_image": str(card.get("entry_image") or ""),
                        "observable_detail": str(card.get("observable_detail") or ""),
                    }
                )

            flags = []
            if section_id in sections_with_host_phase_collapse:
                flags.append("Host phase collapse")
            if section_id in sections_with_editorial_pressure:
                flags.append("Editorial host-target pressure")
            if section_style_warnings:
                flags.append("Section-level style warning")
            if episode_warning_hits:
                flags.append("Referenced in episode-level style warning")

            section_summary = {
                "episode_number": episode_number,
                "section_id": section_id,
                "movement_goal": str(prose_section.get("movement_goal") or ""),
                "word_count": word_count,
                "scene_count": len(scene_card_ids),
                "scene_card_ids": scene_card_ids,
                "scene_role_sequence": roles,
                "scene_job_sequence": jobs,
                "scene_role_combo": " -> ".join(roles),
                "scene_job_combo": " -> ".join(jobs),
                "scene_cards": scene_card_summaries,
                "score": evaluation["score"],
                "verdict": evaluation["verdict"],
                "strengths": evaluation["strengths"],
                "weaknesses": evaluation["weaknesses"],
                "close_only": evaluation["close_only"],
                "flags": flags,
                "text_excerpt": _excerpt(text),
                "text": text,
                "style_warnings": section_style_warnings,
                "episode_warning_hits": episode_warning_hits,
            }
            episode_sections.append(section_summary)
            all_sections.append(section_summary)

        episode_verdict_counts = Counter(section["verdict"] for section in episode_sections)
        episodes_payload.append(
            {
                "episode_number": episode_number,
                "title": str(script_payload.get("title") or f"Episode {episode_number}"),
                "artifact_name": script_path.name,
                "artifact_path": str(script_path),
                "section_count": len(episode_sections),
                "scene_card_count": len(scene_cards),
                "average_score": round(_mean_or_zero(section["score"] for section in episode_sections), 2),
                "verdict_counts": dict(episode_verdict_counts),
                "diagnostics": {
                    "spine_drift_detected": bool(spine_payload.get("spine_drift_detected")),
                    "new_load_bearing_question_detected": bool(
                        spine_payload.get("new_load_bearing_question_detected")
                    ),
                    "failure_labels": [str(item) for item in (spine_payload.get("failure_labels") or [])],
                    "continuity_warning_labels": [
                        str(item) for item in (continuity_payload.get("warning_labels") or [])
                    ],
                    "missed_item_ids": [str(item) for item in (continuity_payload.get("missed_item_ids") or [])],
                    "host_phase_collapse_sections": sorted(sections_with_host_phase_collapse),
                    "host_editorial_pressure_sections": sorted(sections_with_editorial_pressure),
                    "host_unrealized_phase_ids": [
                        str(item) for item in (host_payload.get("approx_unrealized_phase_ids") or [])
                    ],
                    "episode_style_warnings": episode_warnings,
                },
                "sections": episode_sections,
            }
        )

    verdict_counts = Counter(section["verdict"] for section in all_sections)
    role_stats = _role_presence_stats(all_sections)
    start_role_stats = _start_role_stats(all_sections)
    combo_stats = _combo_stats(all_sections)
    findings = _aggregate_findings(
        sections=all_sections,
        role_stats=role_stats,
        start_role_stats=start_role_stats,
        combo_stats=combo_stats,
    )

    weak_sections = sorted(
        (
            section
            for section in all_sections
            if section["verdict"] == "weak"
        ),
        key=lambda item: (item["score"], item["episode_number"], item["section_id"]),
    )[:12]
    strong_sections = sorted(
        (
            section
            for section in all_sections
            if section["verdict"] == "strong"
        ),
        key=lambda item: (-item["score"], item["episode_number"], item["section_id"]),
    )[:12]

    summary = {
        "episode_count": len(episodes_payload),
        "section_count": len(all_sections),
        "scene_card_count": total_scene_cards,
        "average_score": round(_mean_or_zero(section["score"] for section in all_sections), 2),
        "verdict_counts": dict(verdict_counts),
        "role_counts": dict(sorted(role_counts.items())),
        "job_counts": dict(sorted(job_counts.items())),
        "findings": findings,
        "role_presence_stats": role_stats,
        "start_role_stats": start_role_stats,
        "combo_stats": combo_stats,
        "weak_sections": [
            {
                "episode_number": section["episode_number"],
                "section_id": section["section_id"],
                "verdict": section["verdict"],
                "score": section["score"],
                "scene_role_combo": section["scene_role_combo"],
            }
            for section in weak_sections
        ],
        "strong_sections": [
            {
                "episode_number": section["episode_number"],
                "section_id": section["section_id"],
                "verdict": section["verdict"],
                "score": section["score"],
                "scene_role_combo": section["scene_role_combo"],
            }
            for section in strong_sections
        ],
    }

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "title": f"Scene Type to Prose Audit: {run_dir.name}",
        "script_artifacts": list(script_artifacts),
        "summary": summary,
        "episodes": episodes_payload,
    }


def render_scene_type_audit_html(payload: dict[str, Any]) -> str:
    title = escape(str(payload.get("title") or "Scene Type Audit"))
    run_name = escape(str(payload.get("run_name") or "run"))
    summary = payload["summary"]
    episodes = payload["episodes"]

    verdict_counts = summary["verdict_counts"]
    verdict_cards_html = "".join(
        f"""
        <div class="metric-card">
          <div class="metric-label">{escape(verdict.replace('-', ' '))}</div>
          <div class="metric-value">{count}</div>
        </div>
        """
        for verdict, count in sorted(
            verdict_counts.items(), key=lambda item: _VERDICT_ORDER.get(item[0], 999)
        )
    )

    findings_html = "".join(f"<li>{escape(item)}</li>" for item in summary["findings"])

    role_rows_html = "".join(
        f"""
        <tr>
          <td><code>{escape(item['role'])}</code></td>
          <td>{item['section_count']}</td>
          <td>{item['average_score']:.2f}</td>
          <td>{item['strong_count']}</td>
          <td>{item['weak_count']}</td>
        </tr>
        """
        for item in summary["role_presence_stats"]
    )

    start_role_rows_html = "".join(
        f"""
        <tr>
          <td><code>{escape(item['role'])}</code></td>
          <td>{item['section_count']}</td>
          <td>{item['average_score']:.2f}</td>
        </tr>
        """
        for item in summary["start_role_stats"]
    )

    combo_rows_html = "".join(
        f"""
        <tr>
          <td><code>{escape(item['combo'])}</code></td>
          <td>{item['section_count']}</td>
          <td>{item['average_score']:.2f}</td>
          <td>{escape(', '.join(f"{k}:{v}" for k, v in sorted(item['verdict_counts'].items(), key=lambda pair: _VERDICT_ORDER.get(pair[0], 999))))}</td>
          <td>{escape('; '.join(f"E{sample['episode_number']} {sample['section_id']}" for sample in item['sample_sections']))}</td>
        </tr>
        """
        for item in summary["combo_stats"][:20]
    )

    strong_sections_html = "".join(
        f"<li><a href=\"#ep-{item['episode_number']}-{escape(item['section_id'])}\">Episode {item['episode_number']} / {escape(item['section_id'])}</a> <span class=\"inline-meta\">{item['score']:.2f} · {escape(item['scene_role_combo'])}</span></li>"
        for item in summary["strong_sections"]
    )
    weak_sections_html = "".join(
        f"<li><a href=\"#ep-{item['episode_number']}-{escape(item['section_id'])}\">Episode {item['episode_number']} / {escape(item['section_id'])}</a> <span class=\"inline-meta\">{item['score']:.2f} · {escape(item['scene_role_combo'])}</span></li>"
        for item in summary["weak_sections"]
    )

    episode_nav_html = "".join(
        f"<li><a href=\"#episode-{episode['episode_number']}\">Episode {episode['episode_number']}: {escape(episode['title'])}</a></li>"
        for episode in episodes
    )

    episode_blocks = []
    for episode in episodes:
        diagnostics = episode["diagnostics"]
        episode_verdict_chips = "".join(
            f"<span class=\"chip chip-{escape(verdict)}\">{escape(verdict)}: {count}</span>"
            for verdict, count in sorted(
                episode["verdict_counts"].items(), key=lambda item: _VERDICT_ORDER.get(item[0], 999)
            )
        )

        diagnostic_items = [
            f"spine drift: {'yes' if diagnostics['spine_drift_detected'] else 'no'}",
            f"new load-bearing question: {'yes' if diagnostics['new_load_bearing_question_detected'] else 'no'}",
            f"continuity warnings: {len(diagnostics['continuity_warning_labels'])}",
            f"host phase collapses: {len(diagnostics['host_phase_collapse_sections'])}",
            f"style warnings: {len(diagnostics['episode_style_warnings'])}",
        ]
        diagnostic_html = "".join(f"<li>{escape(item)}</li>" for item in diagnostic_items)

        section_blocks = []
        for section in episode["sections"]:
            scene_cards_html = "".join(
                f"""
                <li>
                  <div class="scene-line">
                    <code>{escape(card['scene_id'])}</code>
                    <strong>{escape(card['title'])}</strong>
                    <span class="chip role-chip">{escape(card['scene_role'])}</span>
                    <span class="chip job-chip">{escape(card['scene_job'])}</span>
                  </div>
                  <div class="scene-detail">{escape(card['beat_change'])}</div>
                </li>
                """
                for card in section["scene_cards"]
            )
            strengths_html = "".join(f"<li>{escape(item)}</li>" for item in section["strengths"]) or "<li>None.</li>"
            weaknesses_html = "".join(f"<li>{escape(item)}</li>" for item in section["weaknesses"]) or "<li>None.</li>"
            flags_html = (
                "".join(f"<span class=\"chip flag-chip\">{escape(flag)}</span>" for flag in section["flags"])
                if section["flags"]
                else "<span class=\"chip chip-muted\">No extra flags</span>"
            )
            section_blocks.append(
                f"""
                <details class="section-card" id="ep-{section['episode_number']}-{escape(section['section_id'])}">
                  <summary>
                    <div class="section-head">
                      <div>
                        <h3>{escape(section['section_id'])}</h3>
                        <div class="inline-meta">{section['word_count']} words · {section['scene_count']} scene cards</div>
                      </div>
                      <div class="section-badges">
                        <span class="badge badge-{escape(section['verdict'])}">{escape(section['verdict'])}</span>
                        <span class="score">{section['score']:.2f}</span>
                      </div>
                    </div>
                    <div class="combo-line">
                      <code>{escape(section['scene_role_combo'])}</code>
                    </div>
                  </summary>
                  <div class="section-body">
                    <p class="movement-goal">{escape(section['movement_goal'])}</p>
                    <div class="flag-row">{flags_html}</div>
                    <div class="grid-two">
                      <div>
                        <h4>Strength Drivers</h4>
                        <ul>{strengths_html}</ul>
                      </div>
                      <div>
                        <h4>Weakness Drivers</h4>
                        <ul>{weaknesses_html}</ul>
                      </div>
                    </div>
                    <h4>Planned Scene Cards</h4>
                    <ol class="scene-list">{scene_cards_html}</ol>
                    <h4>Excerpt</h4>
                    <blockquote>{_render_paragraphs_html(section['text_excerpt'])}</blockquote>
                    <details class="full-text">
                      <summary>Full section text</summary>
                      <div class="full-text-body">{_render_paragraphs_html(section['text'])}</div>
                    </details>
                  </div>
                </details>
                """
            )

        episode_blocks.append(
            f"""
            <details class="episode-card" id="episode-{episode['episode_number']}">
              <summary>
                <div class="episode-head">
                  <div>
                    <h2>Episode {episode['episode_number']}: {escape(episode['title'])}</h2>
                    <div class="inline-meta">{episode['section_count']} sections · {episode['scene_card_count']} scene cards · avg score {episode['average_score']:.2f}</div>
                  </div>
                  <div class="chip-row">{episode_verdict_chips}</div>
                </div>
              </summary>
              <div class="episode-body">
                <ul class="diagnostic-list">{diagnostic_html}</ul>
                {''.join(section_blocks)}
              </div>
            </details>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5efe6;
        --paper: #fffdf9;
        --ink: #1f1a17;
        --muted: #6a5e54;
        --line: #ddcdbb;
        --accent: #a44a1b;
        --accent-soft: #f0d8c7;
        --good: #1f6b3a;
        --good-soft: #d9f0df;
        --warn: #946200;
        --warn-soft: #faecbf;
        --bad: #8f2f2f;
        --bad-soft: #f8d6d6;
        --slate: #35526d;
        --slate-soft: #d9e7f3;
      }}

      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: smooth; }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, #f7d8bf 0%, transparent 28%),
          linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
      }}
      a {{ color: var(--accent); }}
      code {{
        font-family: "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 0.92em;
      }}
      .page {{
        max-width: 1320px;
        margin: 0 auto;
        padding: 28px 20px 64px;
      }}
      .hero {{
        background: linear-gradient(135deg, rgba(164,74,27,0.14), rgba(53,82,109,0.08));
        border: 1px solid rgba(164,74,27,0.18);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 40px rgba(75, 51, 32, 0.08);
      }}
      h1, h2, h3, h4 {{
        margin: 0 0 10px;
        line-height: 1.15;
      }}
      h1 {{
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -0.03em;
      }}
      .subhead {{
        margin-top: 10px;
        color: var(--muted);
        max-width: 920px;
        font-size: 1.05rem;
      }}
      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-top: 24px;
      }}
      .metric-card {{
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 18px;
      }}
      .metric-label {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
      }}
      .metric-value {{
        margin-top: 6px;
        font-size: 1.8rem;
        font-weight: 700;
      }}
      .layout {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 320px;
        gap: 24px;
        margin-top: 24px;
      }}
      .main-stack {{
        display: grid;
        gap: 24px;
      }}
      .panel {{
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 12px 24px rgba(75, 51, 32, 0.05);
      }}
      .sidebar {{
        position: sticky;
        top: 18px;
        align-self: start;
      }}
      .sidebar ul {{
        margin: 12px 0 0;
        padding-left: 18px;
      }}
      .sidebar li + li {{
        margin-top: 8px;
      }}
      .findings {{
        margin: 0;
        padding-left: 20px;
      }}
      .findings li + li {{
        margin-top: 10px;
      }}
      .table-wrap {{
        overflow-x: auto;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        border-bottom: 1px solid var(--line);
        padding: 10px 8px;
        text-align: left;
        vertical-align: top;
      }}
      th {{
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--muted);
      }}
      .list-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 18px;
      }}
      .list-grid ul {{
        margin: 0;
        padding-left: 18px;
      }}
      .inline-meta {{
        color: var(--muted);
        font-size: 0.94rem;
      }}
      .episode-card, .section-card, .full-text {{
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #fffdfa;
      }}
      .episode-card + .episode-card,
      .section-card + .section-card {{
        margin-top: 16px;
      }}
      summary {{
        cursor: pointer;
        list-style: none;
        padding: 18px 20px;
      }}
      summary::-webkit-details-marker {{
        display: none;
      }}
      .episode-head, .section-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: start;
      }}
      .chip-row, .flag-row, .section-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }}
      .chip, .badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.82rem;
        line-height: 1;
        border: 1px solid transparent;
      }}
      .chip {{
        background: #f3ece3;
        color: #463c35;
      }}
      .chip-muted {{
        background: #f1efe9;
        color: var(--muted);
      }}
      .chip-strong, .badge-strong {{ background: var(--good-soft); color: var(--good); border-color: rgba(31,107,58,0.15); }}
      .chip-solid, .badge-solid {{ background: var(--slate-soft); color: var(--slate); border-color: rgba(53,82,109,0.15); }}
      .chip-mixed, .badge-mixed {{ background: var(--warn-soft); color: var(--warn); border-color: rgba(148,98,0,0.15); }}
      .chip-weak, .badge-weak {{ background: var(--bad-soft); color: var(--bad); border-color: rgba(143,47,47,0.15); }}
      .chip-close-only, .badge-close-only {{ background: var(--accent-soft); color: var(--accent); border-color: rgba(164,74,27,0.15); }}
      .flag-chip {{ background: #ede6dc; color: #4e433a; }}
      .role-chip {{ background: #f6efe5; color: #654f3a; }}
      .job-chip {{ background: #edf3f8; color: #35526d; }}
      .score {{
        font-weight: 700;
        color: var(--accent);
      }}
      .combo-line {{
        margin-top: 8px;
        color: var(--muted);
      }}
      .episode-body, .section-body {{
        padding: 0 20px 20px;
      }}
      .movement-goal {{
        margin: 4px 0 14px;
        color: #473d36;
      }}
      .grid-two {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 18px;
      }}
      .scene-list {{
        margin: 0;
        padding-left: 20px;
      }}
      .scene-list li + li {{
        margin-top: 10px;
      }}
      .scene-line {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }}
      .scene-detail {{
        margin-top: 4px;
        color: var(--muted);
      }}
      blockquote {{
        margin: 0;
        padding: 16px 18px;
        border-left: 4px solid var(--accent);
        background: #f8f2eb;
        border-radius: 0 12px 12px 0;
      }}
      blockquote p:first-child {{
        margin-top: 0;
      }}
      blockquote p:last-child {{
        margin-bottom: 0;
      }}
      .full-text {{
        margin-top: 16px;
      }}
      .full-text-body {{
        padding: 0 18px 18px;
      }}
      .diagnostic-list {{
        margin: 0 0 18px;
        padding-left: 18px;
        color: var(--muted);
      }}
      .diagnostic-list li + li {{
        margin-top: 6px;
      }}
      .muted {{
        color: var(--muted);
      }}
      @media (max-width: 1080px) {{
        .layout {{
          grid-template-columns: 1fr;
        }}
        .sidebar {{
          position: static;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <h1>{title}</h1>
        <p class="subhead">
          Full-series audit of how planned scene-card roles in <code>series_plan.json</code> translate
          into the final prose of <code>style_audited_script.json</code> across {summary['episode_count']}
          episodes in <code>{run_name}</code>.
        </p>
        <div class="summary-grid">
          <div class="metric-card">
            <div class="metric-label">Episodes</div>
            <div class="metric-value">{summary['episode_count']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Sections</div>
            <div class="metric-value">{summary['section_count']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Scene Cards</div>
            <div class="metric-value">{summary['scene_card_count']}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value">{summary['average_score']:.2f}</div>
          </div>
          {verdict_cards_html}
        </div>
      </section>

      <div class="layout">
        <main class="main-stack">
          <section class="panel">
            <h2>Series Findings</h2>
            <ul class="findings">{findings_html}</ul>
          </section>

          <section class="panel">
            <h2>Role Presence</h2>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Role</th>
                    <th>Sections</th>
                    <th>Avg Score</th>
                    <th>Strong</th>
                    <th>Weak</th>
                  </tr>
                </thead>
                <tbody>{role_rows_html}</tbody>
              </table>
            </div>
          </section>

          <section class="panel">
            <h2>Opening Role Performance</h2>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Opening Role</th>
                    <th>Sections</th>
                    <th>Avg Score</th>
                  </tr>
                </thead>
                <tbody>{start_role_rows_html}</tbody>
              </table>
            </div>
          </section>

          <section class="panel">
            <h2>Recurring Role Combinations</h2>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Role Combo</th>
                    <th>Sections</th>
                    <th>Avg Score</th>
                    <th>Verdicts</th>
                    <th>Samples</th>
                  </tr>
                </thead>
                <tbody>{combo_rows_html}</tbody>
              </table>
            </div>
          </section>

          <section class="panel">
            <h2>Strongest and Weakest Sections</h2>
            <div class="list-grid">
              <div>
                <h3>Top Strong Sections</h3>
                <ul>{strong_sections_html}</ul>
              </div>
              <div>
                <h3>Primary Weak Spots</h3>
                <ul>{weak_sections_html}</ul>
              </div>
            </div>
          </section>

          <section class="panel">
            <h2>Episode Breakdown</h2>
            {''.join(episode_blocks)}
          </section>
        </main>

        <aside class="panel sidebar">
          <h2>Episode Index</h2>
          <p class="muted">Jump directly to an episode block.</p>
          <ul>{episode_nav_html}</ul>
        </aside>
      </div>
    </div>
  </body>
</html>
"""
