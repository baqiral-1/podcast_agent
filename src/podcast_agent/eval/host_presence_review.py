"""Build host-presence review payloads and standalone HTML pages."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9']+")

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "i",
    "it",
    "its",
    "me",
    "my",
    "mine",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "what",
    "which",
    "why",
    "with",
    "you",
    "your",
}

TAG_PATTERNS: dict[str, tuple[re.Pattern[str], float]] = {
    "i": (re.compile(r"\b(?:i|me|my|mine|myself|i'm|i've|i'd|i'll)\b", re.IGNORECASE), 2.2),
    "we": (re.compile(r"\b(?:we|us|our)\b", re.IGNORECASE), 2.2),
    "you": (re.compile(r"\b(?:you|your)\b", re.IGNORECASE), 2.2),
    "question": (re.compile(r"\?|\bthe question\b", re.IGNORECASE), 1.8),
    "imperative": (
        re.compile(
            r"\b(?:listen|look|notice|picture|imagine|remember|follow|hold|hear|mark|"
            r"begin|start|take|slow|keep|watch)\b",
            re.IGNORECASE,
        ),
        1.7,
    ),
    "meta": (
        re.compile(
            r"\b(?:this episode|this hour|this story|this series|tonight|next episode|"
            r"for the next)\b",
            re.IGNORECASE,
        ),
        1.5,
    ),
    "callback": (
        re.compile(
            r"\b(?:carry forward|hold .* forward|remember|again|returns here|comes back|"
            r"we left|as we saw|same logic|back in)\b",
            re.IGNORECASE,
        ),
        1.6,
    ),
    "contrast": (
        re.compile(r"\bnot\b.{0,60}\bbut\b|\binstead\b|\brather than\b", re.IGNORECASE),
        1.6,
    ),
    "evaluation": (
        re.compile(
            r"\b(?:the point|the trouble|the result|the reason|what matters|what registered|"
            r"what follows|what happened|what this means|the fact is)\b",
            re.IGNORECASE,
        ),
        1.5,
    ),
    "naming": (
        re.compile(r"\b(?:call|name|word|phrase|term|means|let that word sit)\b", re.IGNORECASE),
        1.4,
    ),
}

MOVE_PATTERNS: dict[str, tuple[re.Pattern[str], float]] = {
    "orient": (
        re.compile(
            r"\b(?:for the next|this episode|the road to|the story of|we will follow|"
            r"the question .* is)\b",
            re.IGNORECASE,
        ),
        2.0,
    ),
    "clarify": (
        re.compile(
            r"\b(?:to understand|what .* means|it is not|begin with|slow .* down|"
            r"note the word|plainly)\b",
            re.IGNORECASE,
        ),
        1.8,
    ),
    "naming_note": (
        re.compile(r"\b(?:note the word|name the|call it|the word|the phrase|the term)\b", re.IGNORECASE),
        1.8,
    ),
    "callback": (
        re.compile(r"\b(?:carry forward|remember|again|returns here|we left|same)\b", re.IGNORECASE),
        1.8,
    ),
    "contrast": (
        re.compile(r"\bnot\b.{0,60}\bbut\b|\binstead\b|\brather than\b", re.IGNORECASE),
        1.8,
    ),
    "evaluate": (
        re.compile(
            r"\b(?:the point|the result|the reason|what matters|the trouble|the lesson|"
            r"what this means)\b",
            re.IGNORECASE,
        ),
        1.8,
    ),
}

PLACEMENT_TARGETS = {"open": 0.15, "pivot": 0.5, "close": 0.85}


@dataclass(frozen=True)
class SnippetSentence:
    text: str
    highlight: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HostSnippetEntry:
    uid: str
    run_id: str
    episode_number: int
    episode_title: str
    section_id: str
    scene_id: str
    scene_title: str
    phase: str
    cue_count: int
    move_types: list[str]
    host_note: str
    address_modes: list[str]
    approx_realized: bool
    first_person_plural_present: bool | None
    confidence: str
    cue_tags: list[str]
    script_artifact: str
    source_path: str
    scene_index: int | None
    section_scene_total: int | None
    window: list[SnippetSentence]
    highlight_text: str
    snippet_sentence_indexes: list[int]
    section_sentence_count: int
    score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["window"] = [sentence.to_dict() for sentence in self.window]
        return payload


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentence_spans(text: str) -> list[str]:
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text.strip()) if segment.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def _keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall(text.lower())
        if len(token) >= 4 and token not in COMMON_STOPWORDS
    }


def _scene_cards_by_episode(run_dir: Path) -> dict[int, list[dict[str, object]]]:
    plan_path = run_dir / "series_plan.json"
    if not plan_path.exists():
        return {}
    plan = _load_json(plan_path)
    episodes = plan.get("episodes")
    if not isinstance(episodes, list):
        return {}
    mapping: dict[int, list[dict[str, object]]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_number = episode.get("episode_number")
        scene_cards = episode.get("scene_cards")
        if isinstance(episode_number, int) and isinstance(scene_cards, list):
            mapping[episode_number] = [card for card in scene_cards if isinstance(card, dict)]
    return mapping


def _choose_script_path(
    episode_dir: Path, script_artifacts: Sequence[str]
) -> Path | None:
    for name in script_artifacts:
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_section_texts(script_path: Path) -> dict[str, str]:
    payload = _load_json(script_path)
    section_map: dict[str, str] = {}
    if isinstance(payload.get("prose_sections"), list):
        for section in payload["prose_sections"]:
            if isinstance(section, dict):
                section_id = section.get("section_id")
                text = section.get("text")
                if isinstance(section_id, str) and isinstance(text, str):
                    section_map[section_id] = text
    elif isinstance(payload.get("sections"), list):
        for section in payload["sections"]:
            if isinstance(section, dict):
                section_id = section.get("section_id")
                text = section.get("text")
                if isinstance(section_id, str) and isinstance(text, str):
                    section_map[section_id] = text
    return section_map


def _load_episode_title(script_path: Path) -> str:
    payload = _load_json(script_path)
    title = payload.get("title")
    return title if isinstance(title, str) else script_path.parent.name


def _load_phase_trace(episode_dir: Path) -> list[dict[str, object]]:
    for file_name in (
        "host_moves_script_diagnostics.json",
        "spoken_host_moves_diagnostics.json",
    ):
        path = episode_dir / file_name
        if path.exists():
            diagnostics = _load_json(path)
            phase_trace = diagnostics.get("phase_trace")
            if isinstance(phase_trace, list):
                return [row for row in phase_trace if isinstance(row, dict)]
    return []


def _section_scene_order(scene_cards: list[dict[str, object]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for card in scene_cards:
        section_id = card.get("section_id")
        scene_id = card.get("scene_id")
        if isinstance(section_id, str) and isinstance(scene_id, str):
            mapping.setdefault(section_id, []).append(scene_id)
    return mapping


def _host_tags(sentence: str) -> list[str]:
    tags: list[str] = []
    for name, (pattern, _weight) in TAG_PATTERNS.items():
        if pattern.search(sentence):
            tags.append(name)
    return tags


def _position_score(index: int, total: int, phase: str) -> float:
    if total <= 1:
        return 1.0
    target = PLACEMENT_TARGETS.get(phase, 0.5)
    position = index / max(total - 1, 1)
    distance = abs(position - target)
    return max(0.0, 1.0 - (distance / 0.85))


def _sentence_score(
    sentence: str,
    *,
    note_keywords: set[str],
    move_types: Sequence[str],
    phase: str,
    index: int,
    total: int,
    address_modes: Sequence[str],
) -> tuple[float, list[str]]:
    lowered = sentence.lower()
    score = 0.0
    tags = _host_tags(sentence)
    for tag in tags:
        score += TAG_PATTERNS[tag][1]

    for move_type in move_types:
        if move_type in MOVE_PATTERNS and MOVE_PATTERNS[move_type][0].search(lowered):
            score += MOVE_PATTERNS[move_type][1]

    sentence_keywords = _keyword_tokens(sentence)
    overlap = note_keywords & sentence_keywords
    score += 1.35 * len(overlap)

    if "i" in address_modes and "i" in tags:
        score += 1.0
    if "we" in address_modes and "we" in tags:
        score += 1.0
    if "you" in address_modes and "you" in tags:
        score += 1.0
    if "implicit" in address_modes and ("evaluation" in tags or "contrast" in tags):
        score += 0.7

    score += 1.6 * _position_score(index=index, total=total, phase=phase)
    return score, tags


def _pick_highlight_indexes(anchor_index: int) -> list[int]:
    return [anchor_index]


def _window_from_indexes(
    sentences: list[str], highlight_indexes: list[int]
) -> list[SnippetSentence]:
    if not sentences:
        return []
    start = max(min(highlight_indexes) - 1, 0)
    end = min(max(highlight_indexes) + 1, len(sentences) - 1)
    return [
        SnippetSentence(text=sentences[index], highlight=index in highlight_indexes)
        for index in range(start, end + 1)
    ]


def _confidence_for(score: float, tags: list[str], note_overlap: int) -> str:
    if score >= 6.2 or note_overlap >= 2 or len(tags) >= 3:
        return "high"
    if score >= 4.0 or len(tags) >= 2:
        return "medium"
    return "low"


def extract_host_snippets(
    run_dir: Path,
    *,
    script_artifacts: Sequence[str] = ("episode_script.json",),
) -> list[HostSnippetEntry]:
    scene_cards_by_episode = _scene_cards_by_episode(run_dir)
    entries: list[HostSnippetEntry] = []

    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return entries

    for episode_dir in sorted(
        [path for path in episodes_dir.iterdir() if path.is_dir()],
        key=lambda path: int(path.name),
    ):
        try:
            episode_number = int(episode_dir.name)
        except ValueError:
            continue

        script_path = _choose_script_path(episode_dir, script_artifacts)
        if script_path is None:
            continue

        section_texts = _load_section_texts(script_path)
        if not section_texts:
            continue

        episode_title = _load_episode_title(script_path)
        scene_cards = scene_cards_by_episode.get(episode_number, [])
        scene_card_map = {
            card["scene_id"]: card
            for card in scene_cards
            if isinstance(card.get("scene_id"), str)
        }
        section_scene_ids = _section_scene_order(scene_cards)

        for trace in _load_phase_trace(episode_dir):
            approx_realized = bool(trace.get("approx_realized", True))
            if not approx_realized:
                continue

            section_id = trace.get("section_id")
            scene_id = trace.get("scene_id")
            phase = trace.get("phase")
            if (
                not isinstance(section_id, str)
                or not isinstance(scene_id, str)
                or not isinstance(phase, str)
            ):
                continue

            section_text = section_texts.get(section_id)
            if not section_text:
                continue

            scene_card = scene_card_map.get(scene_id, {})
            host_note = ""
            move_types: list[str] = []
            address_modes: list[str] = []
            cue_count = trace.get("cue_count")
            if not isinstance(cue_count, int) or cue_count <= 0:
                cue_count = 1
            phase_move_types = trace.get("move_types")
            if isinstance(phase_move_types, list):
                move_types = [
                    move_type
                    for move_type in phase_move_types
                    if isinstance(move_type, str) and move_type
                ]
            phase_address_modes = trace.get("address_modes")
            if isinstance(phase_address_modes, list):
                address_modes = [
                    address_mode
                    for address_mode in phase_address_modes
                    if isinstance(address_mode, str) and address_mode
                ]
            note = trace.get("host_note")
            if isinstance(note, str):
                host_note = note
            if not move_types or not address_modes or not host_note:
                phase_bucket = {}
                host_moves = scene_card.get("host_moves")
                if isinstance(host_moves, dict):
                    maybe_bucket = host_moves.get(phase)
                    if isinstance(maybe_bucket, list):
                        phase_bucket = maybe_bucket
                if isinstance(phase_bucket, list):
                    if not move_types:
                        move_types = [
                            move_type
                            for cue in phase_bucket
                            if isinstance(cue, dict)
                            for move_type in [cue.get("move_type")]
                            if isinstance(move_type, str) and move_type
                        ]
                    if not address_modes:
                        address_modes = [
                            address_mode
                            for cue in phase_bucket
                            if isinstance(cue, dict)
                            for address_mode in [cue.get("address_mode")]
                            if isinstance(address_mode, str) and address_mode
                        ]
                    if not host_note:
                        host_note = " ".join(
                            note_text
                            for cue in phase_bucket
                            if isinstance(cue, dict)
                            for note_text in [cue.get("note")]
                            if isinstance(note_text, str) and note_text
                        )
                    cue_count = max(cue_count, len(phase_bucket))
            if not move_types:
                move_types = ["clarify"]
            if not address_modes:
                address_modes = ["implicit"]

            sentences = _sentence_spans(section_text)
            if not sentences:
                continue

            note_keywords = _keyword_tokens(host_note)
            scored: list[tuple[float, list[str], int, int]] = []
            for index, sentence in enumerate(sentences):
                score, tags = _sentence_score(
                    sentence,
                    note_keywords=note_keywords,
                    move_types=move_types,
                    phase=phase,
                    index=index,
                    total=len(sentences),
                    address_modes=address_modes,
                )
                note_overlap = len(note_keywords & _keyword_tokens(sentence))
                scored.append((score, tags, index, note_overlap))

            best_score, best_tags, anchor_index, note_overlap = max(
                scored, key=lambda row: (row[0], -abs(row[2] - math.floor(len(sentences) / 2)))
            )
            highlight_indexes = _pick_highlight_indexes(anchor_index)
            window = _window_from_indexes(sentences, highlight_indexes)
            scene_id_order = section_scene_ids.get(section_id, [])
            scene_index = scene_id_order.index(scene_id) + 1 if scene_id in scene_id_order else None
            title = scene_card.get("title")
            scene_title = title if isinstance(title, str) else scene_id
            confidence = _confidence_for(best_score, best_tags, note_overlap)
            highlight_text = " ".join(sentences[index] for index in highlight_indexes)

            entries.append(
                HostSnippetEntry(
                    uid=f"{run_dir.name}::ep{episode_number}::{scene_id}:{phase}::{script_path.name}",
                    run_id=run_dir.name,
                    episode_number=episode_number,
                    episode_title=episode_title,
                    section_id=section_id,
                    scene_id=scene_id,
                    scene_title=scene_title,
                    phase=phase,
                    cue_count=cue_count,
                    move_types=move_types,
                    host_note=host_note,
                    address_modes=address_modes,
                    approx_realized=approx_realized,
                    first_person_plural_present=(
                        trace.get("first_person_plural_present")
                        if isinstance(trace.get("first_person_plural_present"), bool)
                        else None
                    ),
                    confidence=confidence,
                    cue_tags=sorted(best_tags),
                    script_artifact=script_path.name,
                    source_path=str(script_path),
                    scene_index=scene_index,
                    section_scene_total=len(scene_id_order) if scene_id_order else None,
                    window=window,
                    highlight_text=highlight_text,
                    snippet_sentence_indexes=highlight_indexes,
                    section_sentence_count=len(sentences),
                    score=round(best_score, 3),
                )
            )

    return entries


def build_review_payload(
    run_dirs: Iterable[Path],
    *,
    title: str | None = None,
    script_artifacts: Sequence[str] = ("episode_script.json",),
) -> dict[str, object]:
    resolved_runs = [Path(run_dir) for run_dir in run_dirs]
    entries: list[HostSnippetEntry] = []
    for run_dir in resolved_runs:
        entries.extend(extract_host_snippets(run_dir, script_artifacts=script_artifacts))

    entries.sort(
        key=lambda entry: (
            entry.run_id,
            entry.episode_number,
            entry.section_id,
            entry.scene_index if entry.scene_index is not None else 999,
            entry.scene_id,
        )
    )

    if title is None:
        run_labels = ", ".join(run_dir.name for run_dir in resolved_runs)
        title = f"Host Presence RLHF Review: {run_labels}"

    move_counts = Counter(
        move_type for entry in entries for move_type in entry.move_types
    )
    phase_counts = Counter(entry.phase for entry in entries)
    run_counts = Counter(entry.run_id for entry in entries)
    artifact_counts = Counter(entry.script_artifact for entry in entries)
    tag_counts = Counter(tag for entry in entries for tag in entry.cue_tags)

    return {
        "title": title,
        "run_ids": [run_dir.name for run_dir in resolved_runs],
        "script_artifacts": list(script_artifacts),
        "summary": {
            "total_entries": len(entries),
            "run_count": len(run_counts),
            "move_counts": dict(move_counts),
            "phase_counts": dict(phase_counts),
            "run_counts": dict(run_counts),
            "artifact_counts": dict(artifact_counts),
            "tag_counts": dict(tag_counts),
        },
        "entries": [entry.to_dict() for entry in entries],
    }


def render_review_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    title = str(payload["title"])
    subtitle = (
        "Rate each extracted host-presence landing from the selected runs on a 1–5 scale. "
        "Ratings stay in local storage, can be exported, and the page surfaces live best/worst "
        "patterns by run, move type, and cue tag."
    )
    template = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>__TITLE__</title>
    <style>
      :root {
        --bg: #efe5d6;
        --paper: #fff9f1;
        --panel: #f8efe1;
        --card: #fffdf8;
        --ink: #211811;
        --muted: #6e6257;
        --line: #d9c6ae;
        --accent: #9b431d;
        --accent-soft: #f4dfcb;
        --good: #23593e;
        --good-soft: #e4efe6;
        --warn: #8e5b00;
        --warn-soft: #f5ecd9;
        --bad: #8a2d22;
        --bad-soft: #f4dfdc;
        --highlight: #ffe6a7;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, #fff8ee 0, #efe5d6 42%, #e2d0bb 100%);
        font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
        line-height: 1.58;
      }
      .shell {
        width: min(1360px, calc(100vw - 26px));
        margin: 20px auto 44px;
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        gap: 18px;
      }
      .panel,
      .main {
        background: var(--paper);
        border: 1px solid var(--line);
        box-shadow: 0 16px 40px rgba(45, 29, 14, 0.08);
      }
      .panel {
        position: sticky;
        top: 18px;
        align-self: start;
        overflow: hidden;
      }
      .panel-head,
      .main-head {
        padding: 24px 24px 18px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(155, 67, 29, 0.1), rgba(155, 67, 29, 0.02));
      }
      .panel-body { padding: 18px 24px 24px; }
      .main-body { padding: 24px 28px 34px; }
      h1 {
        margin: 0 0 10px;
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.02;
        letter-spacing: -0.035em;
      }
      h2 {
        margin: 0 0 8px;
        font-size: 1.28rem;
        line-height: 1.1;
      }
      h3 {
        margin: 22px 0 10px;
        font-size: 1rem;
        line-height: 1.2;
      }
      p { margin: 10px 0; }
      code {
        padding: 1px 5px;
        background: #f1e8db;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.9em;
      }
      .subhead,
      .small,
      .tiny {
        color: var(--muted);
      }
      .tiny {
        font-size: 0.86rem;
      }
      .small {
        font-size: 0.94rem;
      }
      .stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin: 16px 0 12px;
      }
      .stat {
        padding: 12px 12px 10px;
        background: var(--panel);
        border: 1px solid var(--line);
      }
      .stat .label {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 6px;
      }
      .stat .value {
        font-size: 1.42rem;
        line-height: 1.04;
      }
      .controls {
        display: grid;
        gap: 10px;
      }
      .control-row {
        display: grid;
        gap: 6px;
      }
      label {
        color: var(--muted);
        font-size: 0.88rem;
      }
      select,
      button,
      input[type="checkbox"] {
        font: inherit;
      }
      select,
      .btn {
        padding: 9px 11px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--card);
        color: var(--ink);
      }
      .btn {
        cursor: pointer;
        transition: background 120ms ease, transform 120ms ease;
      }
      .btn:hover {
        background: #fff7ec;
      }
      .btn:active {
        transform: translateY(1px);
      }
      .btn.primary {
        background: var(--accent);
        color: #fff8f1;
        border-color: #7a3618;
      }
      .btn.ghost {
        background: transparent;
      }
      .toggle {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .queue {
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
      }
      .chip-list {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
      }
      .pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: #efe5d9;
        font-size: 0.82rem;
      }
      .main-head .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }
      .card {
        border: 1px solid var(--line);
        background: var(--card);
        overflow: hidden;
      }
      .card-head {
        padding: 16px 18px 14px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(155, 67, 29, 0.07), rgba(155, 67, 29, 0.01));
      }
      .card-body {
        padding: 18px;
      }
      .blind {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
      }
      .blind-mask {
        display: inline-block;
        min-width: 180px;
        padding: 4px 8px;
        border-radius: 999px;
        background: #efe7da;
        color: var(--muted);
        font-size: 0.84rem;
      }
      .snippet {
        margin: 16px 0 18px;
        padding: 18px 20px;
        background: #fbf4e8;
        border-left: 4px solid var(--accent);
      }
      .snippet p {
        margin: 10px 0;
        font-size: 1.06rem;
      }
      mark {
        background: var(--highlight);
        color: inherit;
        padding: 0 2px;
      }
      .rating-strip {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin: 18px 0 8px;
      }
      .rating-btn {
        padding: 14px 10px 12px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: var(--card);
        cursor: pointer;
        text-align: center;
      }
      .rating-btn strong {
        display: block;
        font-size: 1.3rem;
        line-height: 1;
        margin-bottom: 6px;
      }
      .rating-btn span {
        display: block;
        font-size: 0.88rem;
        color: var(--muted);
      }
      .rating-btn.active[data-rating="1"] { background: var(--bad-soft); border-color: #d8b2ab; color: var(--bad); }
      .rating-btn.active[data-rating="2"] { background: #f6e7d7; border-color: #ddc3a2; color: #8a4e18; }
      .rating-btn.active[data-rating="3"] { background: var(--warn-soft); border-color: #ddca98; color: var(--warn); }
      .rating-btn.active[data-rating="4"] { background: #e7f1e6; border-color: #c1d7c2; color: #2c6b49; }
      .rating-btn.active[data-rating="5"] { background: var(--good-soft); border-color: #b9d5c1; color: var(--good); }
      .jump-row,
      .tool-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
      }
      details.meta-block {
        margin-top: 18px;
        border: 1px solid var(--line);
        background: #fff8ef;
      }
      details.meta-block > summary {
        cursor: pointer;
        padding: 12px 14px;
        list-style: none;
      }
      details.meta-block > summary::-webkit-details-marker { display: none; }
      .meta-body {
        padding: 0 14px 14px;
        border-top: 1px solid var(--line);
      }
      .detail-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .mini-card {
        padding: 10px 12px;
        border: 1px solid var(--line);
        background: var(--card);
      }
      .lists {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 20px;
      }
      .list-card {
        padding: 14px;
        border: 1px solid var(--line);
        background: var(--panel);
      }
      .list-card ul {
        margin: 10px 0 0 18px;
        padding: 0;
      }
      .list-card li {
        margin: 6px 0;
      }
      .empty {
        padding: 24px;
        border: 1px dashed var(--line);
        background: #fff8ef;
      }
      .recent-list {
        margin: 10px 0 0;
        padding: 0;
        list-style: none;
      }
      .recent-list li {
        margin: 8px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(217, 198, 174, 0.65);
      }
      @media (max-width: 1100px) {
        .shell {
          grid-template-columns: 1fr;
        }
        .panel {
          position: static;
        }
      }
      @media (max-width: 820px) {
        .main-body,
        .panel-head,
        .panel-body,
        .main-head {
          padding-left: 18px;
          padding-right: 18px;
        }
        .rating-strip,
        .detail-grid,
        .lists,
        .stats {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <div class="panel-head">
          <h2>Review Controls</h2>
          <p class="small">Rate with keys <code>1</code> to <code>5</code>. Use arrow keys to move.</p>
        </div>
        <div class="panel-body">
          <div class="stats">
            <div class="stat">
              <div class="label">Rated</div>
              <div class="value" id="rated-count">0</div>
            </div>
            <div class="stat">
              <div class="label">Remaining</div>
              <div class="value" id="remaining-count">0</div>
            </div>
            <div class="stat">
              <div class="label">Mean Rating</div>
              <div class="value" id="mean-rating">–</div>
            </div>
            <div class="stat">
              <div class="label">Current</div>
              <div class="value" id="current-position">0 / 0</div>
            </div>
          </div>

          <div class="controls">
            <div class="control-row">
              <label for="run-filter">Run</label>
              <select id="run-filter"></select>
            </div>
            <div class="control-row">
              <label for="move-filter">Move Type</label>
              <select id="move-filter"></select>
            </div>
            <div class="control-row">
              <label for="status-filter">Status</label>
              <select id="status-filter">
                <option value="all">All</option>
                <option value="unrated">Unrated only</option>
                <option value="rated">Rated only</option>
              </select>
            </div>
            <div class="control-row">
              <label for="order-mode">Order</label>
              <select id="order-mode">
                <option value="chronological">Chronological</option>
                <option value="random">Random</option>
                <option value="score-desc">Highest-confidence first</option>
              </select>
            </div>
            <label class="toggle">
              <input id="blind-toggle" type="checkbox" checked />
              <span>Blind review for unrated snippets</span>
            </label>
          </div>

          <div class="tool-row">
            <button class="btn" id="prev-btn" type="button">Previous</button>
            <button class="btn primary" id="next-btn" type="button">Next</button>
          </div>
          <div class="tool-row">
            <button class="btn ghost" id="clear-rating-btn" type="button">Clear Rating</button>
            <button class="btn ghost" id="jump-unrated-btn" type="button">Jump to Unrated</button>
          </div>
          <div class="tool-row">
            <button class="btn ghost" id="export-json-btn" type="button">Export JSON</button>
            <button class="btn ghost" id="export-csv-btn" type="button">Export CSV</button>
          </div>

          <div class="queue">
            <h3>Recent Ratings</h3>
            <ul class="recent-list tiny" id="recent-ratings"></ul>
          </div>
        </div>
      </aside>

      <main class="main">
        <div class="main-head">
          <h1>__TITLE__</h1>
          <p class="subhead">__SUBTITLE__</p>
          <div class="meta" id="top-meta"></div>
        </div>
        <div class="main-body">
          <div id="card-root"></div>
          <section class="lists">
            <div class="list-card">
              <h3>Best Cue Tags So Far</h3>
              <ul id="best-tags"></ul>
            </div>
            <div class="list-card">
              <h3>Worst Cue Tags So Far</h3>
              <ul id="worst-tags"></ul>
            </div>
            <div class="list-card">
              <h3>Run Averages</h3>
              <ul id="run-averages"></ul>
            </div>
          </section>
          <section class="lists">
            <div class="list-card">
              <h3>Move-Type Averages</h3>
              <ul id="move-averages"></ul>
            </div>
            <div class="list-card">
              <h3>Best Rated Snippets</h3>
              <ul id="best-snippets"></ul>
            </div>
            <div class="list-card">
              <h3>Worst Rated Snippets</h3>
              <ul id="worst-snippets"></ul>
            </div>
          </section>
        </div>
      </main>
    </div>

    <script id="review-payload" type="application/json">
__PAYLOAD_JSON__
    </script>
    <script>
      const payload = JSON.parse(document.getElementById('review-payload').textContent);
      const storageKey = `host-presence-ratings::${payload.run_ids.join('|')}::${payload.script_artifacts.join('|')}`;
      const labels = {
        1: 'dead',
        2: 'weak',
        3: 'okay',
        4: 'good',
        5: 'strong',
      };
      let ratings = loadRatings();
      let currentIndex = 0;
      let randomOrder = [];

      const runFilter = document.getElementById('run-filter');
      const moveFilter = document.getElementById('move-filter');
      const statusFilter = document.getElementById('status-filter');
      const orderMode = document.getElementById('order-mode');
      const blindToggle = document.getElementById('blind-toggle');
      const cardRoot = document.getElementById('card-root');
      const topMeta = document.getElementById('top-meta');

      function loadRatings() {
        try {
          const raw = localStorage.getItem(storageKey);
          return raw ? JSON.parse(raw) : {};
        } catch (_error) {
          return {};
        }
      }

      function persistRatings() {
        localStorage.setItem(storageKey, JSON.stringify(ratings));
      }

      function buildSelect(select, options, includeAll = true) {
        select.innerHTML = '';
        if (includeAll) {
          const option = document.createElement('option');
          option.value = 'all';
          option.textContent = 'All';
          select.appendChild(option);
        }
        options.forEach((value) => {
          const option = document.createElement('option');
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        });
      }

      function shuffledIndexes(length) {
        const values = Array.from({ length }, (_value, index) => index);
        for (let index = values.length - 1; index > 0; index -= 1) {
          const swapIndex = Math.floor(Math.random() * (index + 1));
          const current = values[index];
          values[index] = values[swapIndex];
          values[swapIndex] = current;
        }
        return values;
      }

      function ratingFor(entry) {
        return ratings[entry.uid] ? ratings[entry.uid].rating : null;
      }

      function filteredEntries() {
        const entries = payload.entries.filter((entry) => {
          if (runFilter.value !== 'all' && entry.run_id !== runFilter.value) return false;
          if (moveFilter.value !== 'all' && !entry.move_types.includes(moveFilter.value)) return false;
          const rating = ratingFor(entry);
          if (statusFilter.value === 'rated' && rating == null) return false;
          if (statusFilter.value === 'unrated' && rating != null) return false;
          return true;
        });
        if (orderMode.value === 'random') {
          if (randomOrder.length !== entries.length) {
            randomOrder = shuffledIndexes(entries.length);
          }
          return randomOrder.map((index) => entries[index]);
        }
        if (orderMode.value === 'score-desc') {
          return [...entries].sort((left, right) => right.score - left.score);
        }
        return entries;
      }

      function visibleEntry() {
        const entries = filteredEntries();
        if (entries.length === 0) return null;
        currentIndex = Math.max(0, Math.min(currentIndex, entries.length - 1));
        return entries[currentIndex];
      }

      function setRating(entry, rating) {
        ratings[entry.uid] = {
          rating,
          rated_at: new Date().toISOString(),
        };
        persistRatings();
        const entries = filteredEntries();
        if (currentIndex < entries.length - 1) currentIndex += 1;
        render();
      }

      function clearRating(entry) {
        delete ratings[entry.uid];
        persistRatings();
        render();
      }

      function windowHtml(entry) {
        return entry.window.map((item) => (
          item.highlight ? `<p><mark>${item.text}</mark></p>` : `<p>${item.text}</p>`
        )).join('');
      }

      function metaPills(entry, concealed) {
        if (concealed) {
          return '<span class="blind-mask">Blind review active until rated</span>';
        }
        const pills = [
          `<span class="pill">${entry.run_id}</span>`,
          `<span class="pill">episode ${entry.episode_number}</span>`,
          `<span class="pill">${entry.move_types.join(' / ')}</span>`,
          `<span class="pill">${entry.phase}</span>`,
          `<span class="pill">${entry.confidence} confidence</span>`,
        ];
        if (entry.script_artifact) {
          pills.push(`<span class="pill">${entry.script_artifact}</span>`);
        }
        return pills.join('');
      }

      function ratingButtons(entry) {
        const active = ratingFor(entry);
        return [1, 2, 3, 4, 5].map((value) => `
          <button class="rating-btn${active === value ? ' active' : ''}" data-rating="${value}" type="button">
            <strong>${value}</strong>
            <span>${labels[value]}</span>
          </button>
        `).join('');
      }

      function topMetaHtml() {
        topMeta.innerHTML = `
          <span class="pill">${payload.summary.total_entries} snippets</span>
          <span class="pill">${payload.run_ids.length} runs</span>
          ${payload.script_artifacts.map((artifact) => `<span class="pill">artifact: ${artifact}</span>`).join('')}
        `;
      }

      function renderCard() {
        const entry = visibleEntry();
        if (!entry) {
          cardRoot.innerHTML = '<div class="empty"><p>No snippets match the current filters.</p></div>';
          return;
        }
        const rating = ratingFor(entry);
        const concealed = blindToggle.checked && rating == null;
        cardRoot.innerHTML = `
          <section class="card">
            <div class="card-head">
              <div class="small">Snippet ${currentIndex + 1} of ${filteredEntries().length}</div>
              <h2>${concealed ? 'Blind Snippet Review' : entry.episode_title}</h2>
              <div class="chip-list">${metaPills(entry, concealed)}</div>
            </div>
            <div class="card-body">
              <div class="snippet">${windowHtml(entry)}</div>
              <div class="small">Rate the highlighted host-presence landing, not the historical claim.</div>
              <div class="rating-strip">${ratingButtons(entry)}</div>
              <div class="jump-row">
                <span class="small">Current rating: <strong>${rating == null ? 'unrated' : `${rating} · ${labels[rating]}`}</strong></span>
              </div>
              <details class="meta-block"${concealed ? '' : ' open'}>
                <summary><strong>${concealed ? 'Reveal source and plan details' : 'Source and plan details'}</strong></summary>
                <div class="meta-body">
                  <div class="detail-grid">
                    <div class="mini-card">
                      <strong>Source</strong>
                      <p class="tiny">${entry.run_id} · episode ${entry.episode_number} · <code>${entry.section_id}</code></p>
                      <p class="tiny"><code>${entry.scene_id}</code>${entry.scene_index ? ` · scene ${entry.scene_index} of ${entry.section_scene_total}` : ''}</p>
                    </div>
                    <div class="mini-card">
                      <strong>Extractor</strong>
                      <p class="tiny">moves: ${entry.move_types.join(', ')} · phase: ${entry.phase} · confidence: ${entry.confidence}</p>
                      <p class="tiny">tags: ${entry.cue_tags.length ? entry.cue_tags.join(', ') : 'none'}</p>
                    </div>
                  </div>
                  <p><strong>Scene title:</strong> ${entry.scene_title}</p>
                  <p><strong>Planned host note:</strong> ${entry.host_note || 'No host note found in series_plan.'}</p>
                  <p class="tiny"><strong>File:</strong> <code>${entry.source_path}</code></p>
                </div>
              </details>
            </div>
          </section>
        `;
        cardRoot.querySelectorAll('.rating-btn').forEach((button) => {
          button.addEventListener('click', () => setRating(entry, Number(button.dataset.rating)));
        });
      }

      function updateSidebarStats() {
        const ratedEntries = payload.entries.filter((entry) => ratingFor(entry) != null);
        const ratingsOnly = ratedEntries.map((entry) => ratingFor(entry));
        document.getElementById('rated-count').textContent = String(ratedEntries.length);
        document.getElementById('remaining-count').textContent = String(payload.entries.length - ratedEntries.length);
        document.getElementById('mean-rating').textContent = ratingsOnly.length
          ? (ratingsOnly.reduce((sum, value) => sum + value, 0) / ratingsOnly.length).toFixed(2)
          : '–';
        document.getElementById('current-position').textContent = `${filteredEntries().length === 0 ? 0 : currentIndex + 1} / ${filteredEntries().length}`;

        const recent = payload.entries
          .filter((entry) => ratingFor(entry) != null)
          .sort((left, right) => ratings[right.uid].rated_at.localeCompare(ratings[left.uid].rated_at))
          .slice(0, 8);
        document.getElementById('recent-ratings').innerHTML = recent.length
          ? recent.map((entry) => `<li><strong>${ratings[entry.uid].rating}</strong> · ${entry.run_id} · ep ${entry.episode_number} · ${entry.scene_title}</li>`).join('')
          : '<li>No ratings yet.</li>';
      }

      function groupedAverage(items, keyFn, minCount = 1) {
        const groups = new Map();
        items.forEach((item) => {
          const rating = ratingFor(item);
          if (rating == null) return;
          const key = keyFn(item);
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(rating);
        });
        return [...groups.entries()]
          .map(([key, values]) => ({
            key,
            count: values.length,
            mean: values.reduce((sum, value) => sum + value, 0) / values.length,
          }))
          .filter((row) => row.count >= minCount)
          .sort((left, right) => right.mean - left.mean || right.count - left.count);
      }

      function renderList(id, rows, formatter, emptyMessage) {
        const root = document.getElementById(id);
        root.innerHTML = rows.length
          ? rows.map(formatter).join('')
          : `<li>${emptyMessage}</li>`;
      }

      function updatePatternPanels() {
        const ratedEntries = payload.entries.filter((entry) => ratingFor(entry) != null);
        const tagRows = groupedAverage(
          ratedEntries.flatMap((entry) => entry.cue_tags.map((tag) => ({ entry, tag }))),
          (row) => row.tag,
          3,
        );
        renderList(
          'best-tags',
          tagRows.slice(0, 6),
          (row) => `<li><strong>${row.key}</strong> · ${row.mean.toFixed(2)} <span class="tiny">(${row.count})</span></li>`,
          'Rate at least three snippets sharing a tag.'
        );
        renderList(
          'worst-tags',
          [...tagRows].reverse().slice(0, 6),
          (row) => `<li><strong>${row.key}</strong> · ${row.mean.toFixed(2)} <span class="tiny">(${row.count})</span></li>`,
          'Rate at least three snippets sharing a tag.'
        );

        const runRows = groupedAverage(ratedEntries, (entry) => entry.run_id, 2);
        renderList(
          'run-averages',
          runRows,
          (row) => `<li><strong>${row.key}</strong> · ${row.mean.toFixed(2)} <span class="tiny">(${row.count})</span></li>`,
          'Need at least two ratings in a run.'
        );

        const moveRows = groupedAverage(
          ratedEntries.flatMap((entry) => entry.move_types.map((moveType) => ({ entry, moveType }))),
          (row) => row.moveType,
          2
        );
        renderList(
          'move-averages',
          moveRows,
          (row) => `<li><strong>${row.key}</strong> · ${row.mean.toFixed(2)} <span class="tiny">(${row.count})</span></li>`,
          'Need at least two ratings for a move type.'
        );

        const bestSnippets = [...ratedEntries]
          .sort((left, right) => ratingFor(right) - ratingFor(left) || left.uid.localeCompare(right.uid))
          .slice(0, 6);
        const worstSnippets = [...ratedEntries]
          .sort((left, right) => ratingFor(left) - ratingFor(right) || left.uid.localeCompare(right.uid))
          .slice(0, 6);
        renderList(
          'best-snippets',
          bestSnippets,
          (entry) => `<li><strong>${ratingFor(entry)}</strong> · ${entry.run_id} · ep ${entry.episode_number}<br /><span class="tiny">${entry.scene_title}</span></li>`,
          'No ratings yet.'
        );
        renderList(
          'worst-snippets',
          worstSnippets,
          (entry) => `<li><strong>${ratingFor(entry)}</strong> · ${entry.run_id} · ep ${entry.episode_number}<br /><span class="tiny">${entry.scene_title}</span></li>`,
          'No ratings yet.'
        );
      }

      function render() {
        topMetaHtml();
        renderCard();
        updateSidebarStats();
        updatePatternPanels();
      }

      function jumpToFirstUnrated() {
        const entries = filteredEntries();
        const index = entries.findIndex((entry) => ratingFor(entry) == null);
        currentIndex = index === -1 ? 0 : index;
        render();
      }

      function exportData(format) {
        const enriched = payload.entries.map((entry) => ({
          ...entry,
          rating: ratingFor(entry),
          rated_at: ratings[entry.uid] ? ratings[entry.uid].rated_at : null,
        }));
        if (format === 'json') {
          const blob = new Blob(
            [JSON.stringify({ title: payload.title, script_artifacts: payload.script_artifacts, entries: enriched }, null, 2)],
            { type: 'application/json' },
          );
          downloadBlob(blob, 'host-presence-ratings.json');
          return;
        }
        const headers = [
          'uid', 'run_id', 'episode_number', 'episode_title', 'scene_id', 'scene_title',
          'move_types', 'phase', 'confidence', 'cue_tags', 'script_artifact', 'rating',
          'rated_at', 'highlight_text',
        ];
        const rows = [headers.join(',')];
        enriched.forEach((entry) => {
          rows.push(headers.map((header) => csvCell(entry[header])).join(','));
        });
        const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
        downloadBlob(blob, 'host-presence-ratings.csv');
      }

      function csvCell(value) {
        const rendered = Array.isArray(value) ? value.join('|') : (value ?? '');
        return `"${String(rendered).replaceAll('"', '""')}"`;
      }

      function downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
      }

      function attachEvents() {
        document.getElementById('prev-btn').addEventListener('click', () => {
          currentIndex = Math.max(0, currentIndex - 1);
          render();
        });
        document.getElementById('next-btn').addEventListener('click', () => {
          currentIndex = Math.min(Math.max(filteredEntries().length - 1, 0), currentIndex + 1);
          render();
        });
        document.getElementById('clear-rating-btn').addEventListener('click', () => {
          const entry = visibleEntry();
          if (entry) clearRating(entry);
        });
        document.getElementById('jump-unrated-btn').addEventListener('click', jumpToFirstUnrated);
        document.getElementById('export-json-btn').addEventListener('click', () => exportData('json'));
        document.getElementById('export-csv-btn').addEventListener('click', () => exportData('csv'));

        [runFilter, moveFilter, statusFilter, orderMode, blindToggle].forEach((control) => {
          control.addEventListener('change', () => {
            randomOrder = [];
            currentIndex = 0;
            render();
          });
        });

        document.addEventListener('keydown', (event) => {
          if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
            return;
          }
          const entry = visibleEntry();
          if (!entry) return;
          if (/^[1-5]$/.test(event.key)) {
            setRating(entry, Number(event.key));
          } else if (event.key === 'ArrowRight') {
            currentIndex = Math.min(Math.max(filteredEntries().length - 1, 0), currentIndex + 1);
            render();
          } else if (event.key === 'ArrowLeft') {
            currentIndex = Math.max(0, currentIndex - 1);
            render();
          } else if (event.key.toLowerCase() === 'u') {
            clearRating(entry);
          } else if (event.key.toLowerCase() === 'j') {
            jumpToFirstUnrated();
          }
        });
      }

      buildSelect(runFilter, payload.run_ids);
      buildSelect(
        moveFilter,
        [...new Set(payload.entries.flatMap((entry) => entry.move_types))].sort()
      );
      statusFilter.value = 'unrated';
      attachEvents();
      render();
    </script>
  </body>
</html>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__PAYLOAD_JSON__", payload_json)
    )
