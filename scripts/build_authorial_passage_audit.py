from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "independence_v34"
ARCH_PATH = RUN_DIR / "episode_architectures.json"
OUTPUT_HTML = RUN_DIR / "authorial_passage_audit.html"
OUTPUT_JSON = RUN_DIR / "authorial_passage_audit.json"


MODE_DESCRIPTIONS = {
    "quote_then_gloss": "Uses a quoted line as a hinge, then cashes it out in plain language.",
    "doctrinal_unpack": "Turns a doctrine or theory into an operating rule with consequences.",
    "institutional_clarifier": "Explains machinery, rules, or formal structure the scene cannot assume.",
    "causal_compression": "Condenses a longer causal chain into a shorter audible argument.",
    "comparative_aside": "Carries a comparison out and then returns to the scene.",
    "verdict_landing": "Names the earned judgment or stakes after the evidence is already on the table.",
}

WORK_CLASS_LABELS = {
    5: "Load-bearing",
    4: "Strong enhancer",
    3: "Framing / secondary",
    2: "Phrase-level",
    1: "Negligible",
}

MODE_FINDINGS = {
    "quote_then_gloss": (
        "Best when the quoted line changes the logic of the room. In this run it is consistently substantive: "
        "the Dyer, snake-pit, Lahore, pistol, and accession lines all expand into real analysis rather than decorative quotation."
    ),
    "doctrinal_unpack": (
        "Only used once, but used well. The Lahore doctrine passage is one of the densest analytic blocks in the run."
    ),
    "institutional_clarifier": (
        "Useful when it names machinery the listener cannot be expected to carry in memory: separate electorates, Cabinet Mission grouping, "
        "paramountcy, dominion status, the Boundary Force, or a lashkar. Weaker when it only restates an already visible institution."
    ),
    "causal_compression": (
        "The broadest and most reliable workhorse. When it works, it does the section's real synthesis and often supplies the episode's actual argument."
    ),
    "comparative_aside": (
        "Most placement-sensitive mode in the run. The long Curzon and railway asides earn themselves; the short closing callbacks in episodes 2 and 4 are vivid but lighter."
    ),
    "verdict_landing": (
        "Usually high value here. Most verdicts are not mere taglines; they distill and stabilize what the preceding paragraphs have earned."
    ),
}

OVERALL_FINDINGS = [
    "This run's authorial passages are mostly substantive. None of the 44 audited passages collapse to pure phrase garnish.",
    "The strongest modes in realized prose are causal_compression, verdict_landing, quote_then_gloss, and the single doctrinal_unpack.",
    "Comparative_aside has the widest spread: two instances are genuinely load-bearing, two function more as elegant framing callbacks in short closing sections.",
    "Institutional_clarifier is valuable when it explains machinery the scene truly needs; it is less distinctive when it only labels something the prose has already made legible.",
    "The real risk in this run is not overuse of empty authorial phrasing. It is that some short closing sections front-load the authorial move, so the planned placement reads earlier and lighter than the architecture suggests.",
]


@dataclass(frozen=True)
class ManualAudit:
    passage_id: str
    paragraph_refs: tuple[int, ...]
    score: int
    judgment: str


MANUAL_AUDITS: tuple[ManualAudit, ...] = (
    ManualAudit(
        "ap_s1_jallianwala_compression",
        (12, 17),
        4,
        "Turns scene arithmetic into an explicit moral argument; the close lands harder because this compression names what the numbers mean.",
    ),
    ManualAudit(
        "ap_s2_dyer_quote",
        (3, 4, 5, 6, 7),
        5,
        "A quoted line becomes doctrine, then becomes the section's whole analytic hinge. This is load-bearing, not ornamental.",
    ),
    ManualAudit(
        "ap_s2_hunter_clarifier",
        (8, 9, 10),
        3,
        "Useful context that sharpens the inquiry's meaning, but the section's main force still comes from Dyer and Curzon rather than this explanation alone.",
    ),
    ManualAudit(
        "ap_s2_curzon_aside",
        (12, 13, 14, 15, 16),
        5,
        "A full comparison-with-return block. It extends Dyer into an older administrative habit and then snaps cleanly back to the Lahore room.",
    ),
    ManualAudit(
        "ap_s3_snake_pit",
        (24, 25, 26),
        5,
        "The quote is not merely memorable; it states the section's governing rule in a way the prose then fully cashes out.",
    ),
    ManualAudit(
        "ap_s3_bardoli_compression",
        (29, 32, 33),
        4,
        "Reframes withdrawal as the method discovering its own rule. Strong enhancement, though the section's scene work is already doing part of the lift.",
    ),
    ManualAudit(
        "ap_s4_dandi_compression",
        (10, 11, 12),
        4,
        "Converts the handful of salt from image into administrative problem; compact but materially clarifying.",
    ),
    ManualAudit(
        "ap_s4_method_verdict",
        (14, 15, 18, 19, 20),
        5,
        "This is the section's earned payoff: Bardoli's self-limitation is what makes Dandi work. Remove it and the section loses its final meaning.",
    ),
    ManualAudit(
        "ap_s5_swallow_anger",
        (9, 10, 11),
        4,
        "A short but real explanatory block. It makes Gandhi's line audible as cost rather than piety.",
    ),
    ManualAudit(
        "ap_s5_price_compression",
        (18, 19, 20, 21),
        5,
        "The Dandi-to-Malir comparison is doing section-level work, not just flourish; it names the price structure of the method.",
    ),
    ManualAudit(
        "ap_s6_verdict",
        (2, 3, 4),
        5,
        "The close of episode 1 depends on this synthesis. It turns a sequence of scenes into an inheritance the rest of the series can use.",
    ),
    ManualAudit(
        "ap_s1_separate_electorates",
        (7, 8, 9),
        4,
        "This translator-note move is essential background, and the episode keeps spending the concept later. Not flashy, but high-value.",
    ),
    ManualAudit(
        "ap_s2_electoral_math",
        (4, 5, 6, 7),
        4,
        "Compresses the electoral result into the episode's live question. Strong framing that genuinely advances the argument.",
    ),
    ManualAudit(
        "ap_s3_refusal_as_policy",
        (5, 6, 7, 10, 11, 12, 13),
        5,
        "This is the section's thesis. It upgrades a coalition failure into a deliberate constitutional refusal with downstream consequences.",
    ),
    ManualAudit(
        "ap_s3_moon_verdict",
        (14, 15, 16, 17, 18, 19, 20),
        4,
        "Strong closing frame that names a historical claim and then tests it. It shapes interpretation more than scene mechanics.",
    ),
    ManualAudit(
        "ap_s4_two_nations_quote",
        (9, 10, 11, 12, 13, 14),
        5,
        "A classic quote-then-gloss block: the quote changes the political grammar, and the gloss makes the change explicit.",
    ),
    ManualAudit(
        "ap_s4_doctrine_as_cause",
        (15, 16, 17, 18, 19, 20, 21),
        5,
        "The run's only doctrinal unpack is one of its strongest passages. It converts a slogan into a causal machine.",
    ),
    ManualAudit(
        "ap_s5_grouping_clarifier",
        (2, 3, 4, 6, 7, 8, 9),
        5,
        "This section effectively is the clarifier. Without it, the Cabinet Mission is a blur instead of the last workable united-India scheme.",
    ),
    ManualAudit(
        "ap_s6_pistol_quote",
        (4, 5, 6, 7),
        5,
        "The quote becomes a total change in political method. It does not decorate the scene; it defines what the scene now is.",
    ),
    ManualAudit(
        "ap_s6_direct_action_verdict",
        (15, 16, 17, 18, 19, 20, 21, 22),
        5,
        "This is not merely a tag at the end of Calcutta; it names authorship and changes the constitutional frame of the whole series.",
    ),
    ManualAudit(
        "ap_s7_guard_clarifier",
        (6, 7, 8),
        3,
        "Useful institutional definition with live statistics, but it is secondary support for the Noakhali-Bihar argument rather than the core argument itself.",
    ),
    ManualAudit(
        "ap_s7_bihar_compression",
        (9, 10, 11, 12, 13, 14, 15),
        4,
        "Strong causal hinge that shows how the cycle begins answering itself. Important to the section's conclusion.",
    ),
    ManualAudit(
        "ap_s8_lucknow_callback",
        (1, 2, 3, 4),
        3,
        "Elegant and memorable, but lighter than the heavier analytic blocks. It frames the close more than it carries fresh explanatory burden.",
    ),
    ManualAudit(
        "ap_s1_1",
        (8, 9, 10, 11, 12, 13, 14),
        5,
        "The calendar argument is the episode's engine, and this block is where it becomes explicit policy rather than anecdote.",
    ),
    ManualAudit(
        "ap_s2_1",
        (1, 2, 3, 4, 5, 6, 7),
        4,
        "Strong structural compression of insolvency into imperial exit logic. More analytic than scenic, and worthwhile.",
    ),
    ManualAudit(
        "ap_s2_2",
        (8, 9, 10, 11, 12, 13),
        4,
        "The dominion-status explanation is real work: it converts abstract constitutional procedure into operational consequence.",
    ),
    ManualAudit(
        "ap_s3_1",
        (11, 12, 13, 14, 15, 16, 17),
        5,
        "The safe is the section's governing image, and this verdict makes it politically legible. High-value synthesis.",
    ),
    ManualAudit(
        "ap_s4_1",
        (1, 2, 3, 4, 5, 6),
        5,
        "The Boundary Force explanation is load-bearing. Without it, the section's later collapse has no scale or institutional meaning.",
    ),
    ManualAudit(
        "ap_s4_2",
        (12, 13, 14, 15, 16, 17, 18, 19),
        5,
        "This block is doing the crucial collapse logic: the institution dissolves along the lines it was meant to police.",
    ),
    ManualAudit(
        "ap_s5_1",
        (13, 14, 15, 16, 17, 18, 19),
        4,
        "A strong cross-episode compression showing East Punjab as the cycle's third or fourth stop. It enhances more than it single-handedly carries the section.",
    ),
    ManualAudit(
        "ap_s6_1",
        (12, 13, 14, 15),
        5,
        "The corpse-train verdict is the section's analytic center of gravity. It gathers image, logistics, and calendar into one frame.",
    ),
    ManualAudit(
        "ap_s6_2",
        (16, 17, 18, 19, 20, 21),
        5,
        "One of the best comparative asides in the run. It starts from the loudspeaker, carries the railway benchmark, and lands the gift/instrument turn cleanly.",
    ),
    ManualAudit(
        "ap_s7_1",
        (6, 7, 8, 9, 14, 17),
        4,
        "Useful verdict that keeps the safe from becoming just a prop. It clarifies what the ceremony is concealing.",
    ),
    ManualAudit(
        "ap_s8_1",
        (2, 3, 4, 5, 6, 7, 8),
        4,
        "Short but effective closing montage. It compresses ceremony, administration, and concealment into a stable final image.",
    ),
    ManualAudit(
        "authorial_c59fc65a",
        (9, 10, 11, 12, 13, 14, 15, 16, 17),
        5,
        "This clarifier carries the whole princely-states problem. It is not a sidebar; it is the section's central explanatory burden.",
    ),
    ManualAudit(
        "authorial_540c0500",
        (21, 22),
        4,
        "A sharp verdict that crystallizes the accession machine after the evidence has been staged. Short, but it matters.",
    ),
    ManualAudit(
        "authorial_a917b3b3",
        (7, 8, 9, 15, 16, 17),
        5,
        "This compression is doing the real Junagadh work: the principle India needs here is the principle it cannot keep for Kashmir.",
    ),
    ManualAudit(
        "authorial_98b3a130",
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        5,
        "The war-as-decision frame is the whole point of the Lahore bedroom section. Remove it and the section turns back into mere chronology.",
    ),
    ManualAudit(
        "authorial_729ad8cd",
        (14, 15, 16),
        4,
        "Compact but important definition. The section needs the listener to hear lashkar as a deniable instrument rather than a generic army.",
    ),
    ManualAudit(
        "authorial_5b815540",
        (21, 22),
        5,
        "A strong doubled verdict: atrocity and strategic delay are inseparable here. This is one of the run's best short landings.",
    ),
    ManualAudit(
        "authorial_cc8727ad",
        (6, 7, 8),
        4,
        "Efficient quote-then-gloss. It turns Menon's line into the legal-to-military conversion point.",
    ),
    ManualAudit(
        "authorial_acd15f36",
        (15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 28),
        5,
        "This is the section's true engine: the signed page becomes an army, then a status quo. It is absolutely load-bearing.",
    ),
    ManualAudit(
        "authorial_be3f3c36",
        (7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        5,
        "The inversion at Junagadh versus Srinagar is the episode's deepest constitutional point, and this verdict block fully owns it.",
    ),
    ManualAudit(
        "authorial_4403797a",
        (2, 3, 4),
        3,
        "Vivid and apt, but comparatively lighter. It frames the close through Lahore-to-Lahore rather than doing the heaviest new analytical lift.",
    ),
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_sentences(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return len([part for part in parts if part])


def first_paragraph_band(first_ref: int, total_paragraphs: int) -> str:
    ratio = first_ref / max(total_paragraphs, 1)
    if ratio <= 0.25:
        return "early"
    if ratio <= 0.65:
        return "middle"
    return "late"


def paragraph_ref_label(refs: tuple[int, ...]) -> str:
    ranges: list[str] = []
    start = refs[0]
    prev = refs[0]
    for ref in refs[1:]:
        if ref == prev + 1:
            prev = ref
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = ref
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def placement_note(planned: str, band: str) -> str:
    if planned == "open":
        if band == "early":
            return "planned open; realized early"
        return f"planned open; realized {band}"
    if planned == "mid":
        if band == "middle":
            return "planned mid; realized mid"
        return f"planned mid; realized {band}"
    if band == "late":
        return "planned close; realized late"
    return f"planned close; realized {band}"


def mode_rank_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (row["avg_score"], row["avg_sentences"], -row["count"])


def build_dataset() -> dict[str, Any]:
    manual_by_id = {item.passage_id: item for item in MANUAL_AUDITS}
    architecture = json.loads(ARCH_PATH.read_text())["episodes"]
    passage_rows: list[dict[str, Any]] = []

    for episode in architecture:
        episode_number = episode["episode_number"]
        script_path = RUN_DIR / "episodes" / str(episode_number) / "episode_script.json"
        script = json.loads(script_path.read_text())
        prose_by_section = {
            section["section_id"]: section["text"].split("\n\n")
            for section in script["prose_sections"]
        }
        for section in episode["sections"]:
            section_id = section["section_id"]
            paragraphs = prose_by_section[section_id]
            total_paragraphs = len(paragraphs)
            for passage in section.get("authorial_passages", []):
                passage_id = passage["authorial_passage_id"]
                if passage_id not in manual_by_id:
                    raise ValueError(f"Missing manual audit entry for {passage_id}")
                audit = manual_by_id[passage_id]
                if max(audit.paragraph_refs) > total_paragraphs:
                    raise ValueError(
                        f"Paragraph ref out of range for {passage_id}: "
                        f"{audit.paragraph_refs} vs {total_paragraphs}"
                    )
                selected_paragraphs = [
                    {"index": ref, "text": paragraphs[ref - 1]} for ref in audit.paragraph_refs
                ]
                combined_text = "\n\n".join(item["text"] for item in selected_paragraphs)
                sentence_count = count_sentences(combined_text)
                band = first_paragraph_band(audit.paragraph_refs[0], total_paragraphs)
                row = {
                    "episode_number": episode_number,
                    "section_id": section_id,
                    "passage_id": passage_id,
                    "mode": passage["mode"],
                    "placement": passage["placement"],
                    "budget_sentences": passage.get("budget_sentences"),
                    "claim": passage["claim"],
                    "quote_anchor": passage.get("quote_anchor", ""),
                    "gloss_seed": passage.get("gloss_seed", ""),
                    "score": audit.score,
                    "work_class": WORK_CLASS_LABELS[audit.score],
                    "judgment": audit.judgment,
                    "paragraph_refs": list(audit.paragraph_refs),
                    "paragraph_ref_label": paragraph_ref_label(audit.paragraph_refs),
                    "selected_paragraphs": selected_paragraphs,
                    "selected_sentence_count": sentence_count,
                    "total_paragraphs_in_section": total_paragraphs,
                    "realized_band": band,
                    "placement_note": placement_note(passage["placement"], band),
                }
                passage_rows.append(row)

    if len(passage_rows) != len(MANUAL_AUDITS):
        raise ValueError(f"Expected {len(MANUAL_AUDITS)} passages, found {len(passage_rows)}")

    score_counts = Counter(row["score"] for row in passage_rows)
    load_bearing_count = sum(1 for row in passage_rows if row["score"] == 5)
    strong_or_better_count = sum(1 for row in passage_rows if row["score"] >= 4)
    phrase_level_count = sum(1 for row in passage_rows if row["score"] <= 2)

    mode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in passage_rows:
        mode_groups[row["mode"]].append(row)

    mode_summary = []
    for mode, rows in mode_groups.items():
        summary_row = {
            "mode": mode,
            "count": len(rows),
            "avg_score": round(mean(row["score"] for row in rows), 2),
            "avg_sentences": round(mean(row["selected_sentence_count"] for row in rows), 2),
            "load_bearing_count": sum(1 for row in rows if row["score"] == 5),
            "strong_or_better_count": sum(1 for row in rows if row["score"] >= 4),
            "framing_count": sum(1 for row in rows if row["score"] == 3),
            "description": MODE_DESCRIPTIONS[mode],
            "finding": MODE_FINDINGS[mode],
        }
        mode_summary.append(summary_row)
    mode_summary.sort(key=mode_rank_key, reverse=True)

    episode_summary = []
    episode_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in passage_rows:
        episode_groups[row["episode_number"]].append(row)
    for episode_number, rows in sorted(episode_groups.items()):
        episode_summary.append(
            {
                "episode_number": episode_number,
                "count": len(rows),
                "avg_score": round(mean(row["score"] for row in rows), 2),
                "load_bearing_count": sum(1 for row in rows if row["score"] == 5),
                "strong_or_better_count": sum(1 for row in rows if row["score"] >= 4),
            }
        )

    return {
        "run_id": "independence_v34",
        "generated_from": {
            "architecture": str(ARCH_PATH.relative_to(ROOT)),
            "episode_scripts": "runs/independence_v34/episodes/*/episode_script.json",
        },
        "overall_findings": OVERALL_FINDINGS,
        "summary": {
            "passage_count": len(passage_rows),
            "load_bearing_count": load_bearing_count,
            "strong_or_better_count": strong_or_better_count,
            "phrase_level_count": phrase_level_count,
            "avg_score": round(mean(row["score"] for row in passage_rows), 2),
            "avg_selected_sentences": round(
                mean(row["selected_sentence_count"] for row in passage_rows), 2
            ),
            "score_counts": dict(sorted(score_counts.items(), reverse=True)),
        },
        "mode_summary": mode_summary,
        "episode_summary": episode_summary,
        "passages": passage_rows,
    }


def render_bar(value: float, max_value: float) -> str:
    width = 0 if max_value == 0 else round((value / max_value) * 100, 1)
    return f'<div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>'


def render_mode_table(mode_summary: list[dict[str, Any]]) -> str:
    max_count = max(row["count"] for row in mode_summary)
    rows = []
    for row in mode_summary:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['mode'])}</code></td>"
            f"<td>{row['count']}{render_bar(row['count'], max_count)}</td>"
            f"<td>{row['avg_score']}</td>"
            f"<td>{row['avg_sentences']}</td>"
            f"<td>{row['load_bearing_count']}</td>"
            f"<td>{row['framing_count']}</td>"
            f"<td>{html.escape(row['finding'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_episode_table(episode_summary: list[dict[str, Any]]) -> str:
    rows = []
    for row in episode_summary:
        rows.append(
            "<tr>"
            f"<td>Episode {row['episode_number']}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{row['avg_score']}</td>"
            f"<td>{row['load_bearing_count']}</td>"
            f"<td>{row['strong_or_better_count']}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_passage_card(row: dict[str, Any]) -> str:
    paragraphs_html = "\n".join(
        (
            '<p class="excerpt-paragraph">'
            f'<span class="paragraph-index">[{item["index"]}]</span> '
            f"{html.escape(item['text'])}"
            "</p>"
        )
        for item in row["selected_paragraphs"]
    )
    quote_html = ""
    if row["quote_anchor"]:
        quote_html = (
            '<div class="meta-block"><span class="meta-label">Quote anchor</span>'
            f"<blockquote>{html.escape(row['quote_anchor'])}</blockquote></div>"
        )
    gloss_html = ""
    if row["gloss_seed"]:
        gloss_html = (
            '<div class="meta-block"><span class="meta-label">Planned gloss seed</span>'
            f"<p>{html.escape(row['gloss_seed'])}</p></div>"
        )
    return (
        f'<article class="passage-card" data-mode="{html.escape(row["mode"])}" '
        f'data-episode="{row["episode_number"]}" data-score="{row["score"]}">'
        '<div class="passage-header">'
        f'<div class="passage-kicker">Episode {row["episode_number"]} / {html.escape(row["section_id"])}</div>'
        f"<h3>{html.escape(row['passage_id'])}</h3>"
        '<div class="chip-row">'
        f'<span class="chip mode-chip"><code>{html.escape(row["mode"])}</code></span>'
        f'<span class="chip">{html.escape(row["work_class"])}</span>'
        f'<span class="chip">score {row["score"]}/5</span>'
        f'<span class="chip">planned {html.escape(row["placement"])}</span>'
        f'<span class="chip">{html.escape(row["placement_note"])}</span>'
        f'<span class="chip">paras {html.escape(row["paragraph_ref_label"])}</span>'
        f'<span class="chip">~{row["selected_sentence_count"]} selected sentences</span>'
        "</div>"
        "</div>"
        '<div class="claim-block">'
        '<span class="meta-label">Planned claim</span>'
        f"<p>{html.escape(row['claim'])}</p>"
        "</div>"
        f"{quote_html}"
        f"{gloss_html}"
        '<div class="meta-block"><span class="meta-label">Audit verdict</span>'
        f"<p>{html.escape(row['judgment'])}</p></div>"
        "<details>"
        "<summary>Show realized prose paragraphs</summary>"
        f"{paragraphs_html}"
        "</details>"
        "</article>"
    )


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    passage_cards = "\n".join(render_passage_card(row) for row in report["passages"])
    overall_findings_html = "\n".join(
        f"<li>{html.escape(item)}</li>" for item in report["overall_findings"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Authorial Passage Audit: independence_v34</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1b1b18;
      --muted: #696552;
      --line: #d7cfbf;
      --accent: #a2461d;
      --accent-soft: #ead2c5;
      --good: #1f5e3b;
      --mid: #6c5416;
      --shadow: 0 18px 40px rgba(51, 36, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ea 0, transparent 34%),
        linear-gradient(180deg, #f4efe6 0%, #efe7d7 100%);
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 24px 80px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(162, 70, 29, 0.14), rgba(130, 91, 32, 0.06));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{
      font-size: clamp(32px, 4vw, 54px);
      line-height: 0.98;
      margin: 0 0 12px;
      max-width: 980px;
    }}
    .hero p {{
      max-width: 980px;
      font-size: 18px;
      line-height: 1.5;
      margin: 0;
      color: #2c2922;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-top: 22px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 26px;
    }}
    h3 {{
      margin: 0;
      font-size: 22px;
    }}
    p, li {{
      line-height: 1.58;
      font-size: 16px;
    }}
    .finding-list {{
      margin: 0;
      padding-left: 20px;
      display: grid;
      gap: 8px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .summary-card {{
      background: #fff9f1;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .summary-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .summary-value {{
      font-size: 34px;
      line-height: 1;
      margin-bottom: 6px;
    }}
    .summary-note {{
      font-size: 14px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-top: 1px solid var(--line);
      padding: 12px 10px;
      font-size: 15px;
      line-height: 1.45;
    }}
    th {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-top: none;
      padding-top: 0;
    }}
    .bar-track {{
      background: #efe6d8;
      border-radius: 999px;
      height: 8px;
      margin-top: 8px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #d87b4a);
    }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .method-card {{
      padding: 16px 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #fffaf4;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    select {{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      font-size: 15px;
      min-width: 190px;
    }}
    .passage-grid {{
      display: grid;
      gap: 16px;
    }}
    .passage-card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 16px;
      background: white;
    }}
    .passage-kicker {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      margin-bottom: 14px;
    }}
    .chip {{
      border-radius: 999px;
      padding: 7px 10px;
      background: #f4ecdf;
      border: 1px solid #e0d5c1;
      font-size: 13px;
      color: #413b31;
    }}
    .mode-chip {{
      background: #f3e5dd;
      border-color: #dfc2b5;
    }}
    .meta-block {{
      margin-top: 14px;
    }}
    .meta-label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    blockquote {{
      margin: 0;
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      background: #fff7ef;
    }}
    details {{
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
      color: #2f2a22;
    }}
    .excerpt-paragraph {{
      margin: 12px 0 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: #fbf7ef;
      border: 1px solid #ece2d2;
    }}
    .paragraph-index {{
      color: var(--accent);
      font-weight: 700;
    }}
    .footer-note {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 14px;
    }}
    .hidden {{
      display: none !important;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">independence_v34 / authorial passage audit</div>
      <h1>Which authorial passage types are doing real work, and which ones are mostly framing?</h1>
      <p>
        This page audits all 44 planned <code>authorial_passages</code> in the realized
        <code>episode_script.json</code> output for <code>independence_v34</code>. The scoring is intentionally pragmatic:
        score 5 means the passage is load-bearing for the section's argument, score 4 means it materially strengthens the prose,
        and score 3 means it is useful but more secondary or framing-led. Scores 2 and 1 are reserved for phrase-level garnish or near-no-op passages.
      </p>
    </section>

    <section class="section">
      <h2>Top Findings</h2>
      <ul class="finding-list">
        {overall_findings_html}
      </ul>
      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-label">Total passages</div>
          <div class="summary-value">{summary["passage_count"]}</div>
          <div class="summary-note">All planned authorial passages in the run.</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Load-bearing</div>
          <div class="summary-value">{summary["load_bearing_count"]}</div>
          <div class="summary-note">Score 5: section argument would materially weaken without them.</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Strong or better</div>
          <div class="summary-value">{summary["strong_or_better_count"]}</div>
          <div class="summary-note">Scores 4-5: not just phrasework.</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Phrase-level</div>
          <div class="summary-value">{summary["phrase_level_count"]}</div>
          <div class="summary-note">Scores 1-2. In this audit, none fall here.</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Average score</div>
          <div class="summary-value">{summary["avg_score"]}</div>
          <div class="summary-note">Across all 44 passages.</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Avg selected sentences</div>
          <div class="summary-value">{summary["avg_selected_sentences"]}</div>
          <div class="summary-note">Approximate realized sentence footprint in audited paragraphs.</div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Method</h2>
      <div class="method-grid">
        <div class="method-card">
          <h3>Source of truth</h3>
          <p>
            Planned passage metadata comes from <code>runs/independence_v34/episode_architectures.json</code>.
            Realized prose comes from <code>runs/independence_v34/episodes/*/episode_script.json</code>.
          </p>
        </div>
        <div class="method-card">
          <h3>Placement audit</h3>
          <p>
            Each card shows the planned placement (<code>open</code>, <code>mid</code>, <code>close</code>) and the actual paragraph refs where the passage's work appears.
            The realized band is a coarse positional read: early, middle, or late within the section.
          </p>
        </div>
        <div class="method-card">
          <h3>What counts as “real work”</h3>
          <p>
            A passage scores highest when it changes the argument of the section, clarifies machinery the section depends on, or lands a synthesis the prose has truly earned.
            A passage scores lower when it mainly frames, echoes, or tidies what is already obvious.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Mode Ranking</h2>
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Count</th>
            <th>Avg score</th>
            <th>Avg selected sentences</th>
            <th>Load-bearing</th>
            <th>Framing / secondary</th>
            <th>Mode verdict</th>
          </tr>
        </thead>
        <tbody>
          {render_mode_table(report["mode_summary"])}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Episode Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Episode</th>
            <th>Passages</th>
            <th>Avg score</th>
            <th>Load-bearing</th>
            <th>Strong or better</th>
          </tr>
        </thead>
        <tbody>
          {render_episode_table(report["episode_summary"])}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Passage-by-Passage Audit</h2>
      <div class="filters">
        <label>
          Mode
          <select id="modeFilter">
            <option value="all">All modes</option>
            <option value="causal_compression">causal_compression</option>
            <option value="comparative_aside">comparative_aside</option>
            <option value="doctrinal_unpack">doctrinal_unpack</option>
            <option value="institutional_clarifier">institutional_clarifier</option>
            <option value="quote_then_gloss">quote_then_gloss</option>
            <option value="verdict_landing">verdict_landing</option>
          </select>
        </label>
        <label>
          Episode
          <select id="episodeFilter">
            <option value="all">All episodes</option>
            <option value="1">Episode 1</option>
            <option value="2">Episode 2</option>
            <option value="3">Episode 3</option>
            <option value="4">Episode 4</option>
          </select>
        </label>
        <label>
          Minimum score
          <select id="scoreFilter">
            <option value="1">1+</option>
            <option value="3">3+</option>
            <option value="4">4+</option>
            <option value="5">5 only</option>
          </select>
        </label>
      </div>
      <div id="passageCount" class="footer-note"></div>
      <div class="passage-grid" id="passageGrid">
        {passage_cards}
      </div>
      <p class="footer-note">
        Paragraph refs are section-local. Example: <code>paras 12-16</code> means paragraphs 12 through 16 inside that section's realized prose string.
      </p>
    </section>
  </div>

  <script>
    const modeFilter = document.getElementById("modeFilter");
    const episodeFilter = document.getElementById("episodeFilter");
    const scoreFilter = document.getElementById("scoreFilter");
    const cards = Array.from(document.querySelectorAll(".passage-card"));
    const countNode = document.getElementById("passageCount");

    function applyFilters() {{
      const mode = modeFilter.value;
      const episode = episodeFilter.value;
      const minScore = Number(scoreFilter.value);
      let visible = 0;
      for (const card of cards) {{
        const okMode = mode === "all" || card.dataset.mode === mode;
        const okEpisode = episode === "all" || card.dataset.episode === episode;
        const okScore = Number(card.dataset.score) >= minScore;
        const show = okMode && okEpisode && okScore;
        card.classList.toggle("hidden", !show);
        if (show) visible += 1;
      }}
      countNode.textContent = `${{visible}} passage${{visible === 1 ? "" : "s"}} visible`;
    }}

    for (const node of [modeFilter, episodeFilter, scoreFilter]) {{
      node.addEventListener("change", applyFilters);
    }}
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    report = build_dataset()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    OUTPUT_HTML.write_text(render_html(report))
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
