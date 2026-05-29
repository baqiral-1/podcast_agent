"""Build blind side-by-side script comparison payloads and standalone HTML."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

WORD_RE = re.compile(r"[a-z0-9']+")
DEFAULT_COUNTS_BY_EPISODE = (13, 13, 12, 12)
DEFAULT_MIN_WORDS = 55
MAX_PARAGRAPHS_PER_SNIPPET = 2


@dataclass(frozen=True)
class SnippetSource:
    run_id: str
    episode_number: int
    paragraph_start: int
    paragraph_end: int
    word_count: int
    text: str
    source_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BlindComparisonEntry:
    uid: str
    order_index: int
    episode_number: int
    stage_label: str
    checkpoint_index: int
    checkpoint_total: int
    left_text: str
    right_text: str
    left_run_id: str
    right_run_id: str
    left_source: SnippetSource
    right_source: SnippetSource

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["left_source"] = self.left_source.to_dict()
        payload["right_source"] = self.right_source.to_dict()
        return payload


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text.lower()))


def _episode_paragraphs(run_dir: Path, episode_number: int) -> list[str]:
    script_path = run_dir / "episodes" / str(episode_number) / "episode_script.json"
    payload = _load_json(script_path)
    sections = payload.get("prose_sections")
    if not isinstance(sections, list):
        raise ValueError(f"{script_path} does not contain prose_sections")

    paragraphs: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        text = section.get("text")
        if not isinstance(text, str):
            continue
        paragraphs.extend(part.strip() for part in text.split("\n\n") if part.strip())
    if not paragraphs:
        raise ValueError(f"{script_path} does not contain any prose paragraphs")
    return paragraphs


def _common_episode_numbers(run_dirs: Sequence[Path]) -> list[int]:
    episode_sets: list[set[int]] = []
    for run_dir in run_dirs:
        episodes_dir = run_dir / "episodes"
        episode_sets.append(
            {
                int(path.name)
                for path in episodes_dir.iterdir()
                if path.is_dir() and (path / "episode_script.json").exists()
            }
        )
    shared = set.intersection(*episode_sets) if episode_sets else set()
    return sorted(shared)


def _fractions_for_count(count: int) -> list[float]:
    return [(index + 1) / (count + 1) for index in range(count)]


def _stage_label(fraction: float) -> str:
    if fraction < 0.25:
        return "opening"
    if fraction < 0.5:
        return "early middle"
    if fraction < 0.75:
        return "late middle"
    return "closing"


def _build_snippet_source(
    run_dir: Path,
    run_id: str,
    episode_number: int,
    fraction: float,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
) -> SnippetSource:
    paragraphs = _episode_paragraphs(run_dir, episode_number)
    target_index = min(len(paragraphs) - 1, max(0, round(fraction * (len(paragraphs) - 1))))
    start_index = target_index
    end_index = target_index
    chosen = [paragraphs[target_index]]
    total_words = _word_count(chosen[0])

    while (
        total_words < min_words
        and len(chosen) < MAX_PARAGRAPHS_PER_SNIPPET
        and end_index + 1 < len(paragraphs)
    ):
        end_index += 1
        chosen.append(paragraphs[end_index])
        total_words = sum(_word_count(part) for part in chosen)

    if total_words < min_words and len(chosen) < MAX_PARAGRAPHS_PER_SNIPPET and start_index > 0:
        start_index -= 1
        chosen.insert(0, paragraphs[start_index])
        total_words = sum(_word_count(part) for part in chosen)

    source_path = run_dir / "episodes" / str(episode_number) / "episode_script.json"
    return SnippetSource(
        run_id=run_id,
        episode_number=episode_number,
        paragraph_start=start_index + 1,
        paragraph_end=end_index + 1,
        word_count=total_words,
        text="\n\n".join(chosen),
        source_path=str(source_path),
    )


def _swap_left_right(uid: str) -> bool:
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return bool(digest[0] & 1)


def build_blind_comparison_payload(
    run_dirs: Sequence[Path],
    *,
    title: str,
    subtitle: str | None = None,
    comparison_prompt: str | None = None,
    counts_by_episode: Sequence[int] = DEFAULT_COUNTS_BY_EPISODE,
) -> dict[str, object]:
    if len(run_dirs) != 2:
        raise ValueError("blind comparison payloads require exactly two run directories")

    resolved_runs = [run_dir.resolve() for run_dir in run_dirs]
    run_ids = [run_dir.name for run_dir in resolved_runs]
    shared_episodes = _common_episode_numbers(resolved_runs)
    if len(shared_episodes) != len(counts_by_episode):
        raise ValueError(
            "counts_by_episode length must match the number of common episodes "
            f"({len(shared_episodes)})"
        )

    comparisons: list[BlindComparisonEntry] = []
    order_index = 1
    for episode_number, count in zip(shared_episodes, counts_by_episode, strict=True):
        fractions = _fractions_for_count(count)
        for checkpoint_index, fraction in enumerate(fractions, start=1):
            left_source = _build_snippet_source(
                resolved_runs[0],
                run_ids[0],
                episode_number,
                fraction,
            )
            right_source = _build_snippet_source(
                resolved_runs[1],
                run_ids[1],
                episode_number,
                fraction,
            )

            uid = f"ep{episode_number:02d}-slot{checkpoint_index:02d}"
            if _swap_left_right(uid):
                display_left = right_source
                display_right = left_source
            else:
                display_left = left_source
                display_right = right_source

            comparisons.append(
                BlindComparisonEntry(
                    uid=uid,
                    order_index=order_index,
                    episode_number=episode_number,
                    stage_label=_stage_label(fraction),
                    checkpoint_index=checkpoint_index,
                    checkpoint_total=count,
                    left_text=display_left.text,
                    right_text=display_right.text,
                    left_run_id=display_left.run_id,
                    right_run_id=display_right.run_id,
                    left_source=display_left,
                    right_source=display_right,
                )
            )
            order_index += 1

    prompt = comparison_prompt or (
        "Which passage better sustains a live host presence, feels more personable, "
        "and avoids slipping into historical prose?"
    )
    resolved_subtitle = subtitle or (
        "Fifty blind A/B comparisons sampled from corresponding episode positions. "
        "Choose the stronger passage, let the page persist your decisions in local storage, "
        "and reveal the run winner only after all comparisons are complete."
    )
    run_labels = {run_id: run_id.removeprefix("iranian_revolution_") for run_id in run_ids}
    return {
        "title": title,
        "subtitle": resolved_subtitle,
        "comparison_prompt": prompt,
        "run_ids": run_ids,
        "run_labels": run_labels,
        "episode_numbers": shared_episodes,
        "total_comparisons": len(comparisons),
        "counts_by_episode": list(counts_by_episode),
        "comparisons": [entry.to_dict() for entry in comparisons],
    }


def _payload_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False).replace("</", "<\\/")


def render_blind_comparison_html(payload: dict[str, object]) -> str:
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
        --panel: #f7ede0;
        --card: #fffdf8;
        --ink: #201710;
        --muted: #6d6157;
        --line: #d7c3aa;
        --accent: #92401e;
        --accent-soft: #f4ddca;
        --good: #23593e;
        --good-soft: #e3efe6;
        --bad: #8b2f24;
        --bad-soft: #f3dfdb;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, #fff8ee 0, #efe5d6 44%, #e0cfbb 100%);
        font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
        line-height: 1.58;
      }
      .shell {
        width: min(1380px, calc(100vw - 28px));
        margin: 22px auto 42px;
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        gap: 18px;
      }
      .sidebar,
      .main {
        background: var(--paper);
        border: 1px solid var(--line);
        box-shadow: 0 16px 40px rgba(45, 29, 14, 0.08);
      }
      .sidebar {
        position: sticky;
        top: 18px;
        align-self: start;
        overflow: hidden;
      }
      .sidebar-head,
      .main-head {
        padding: 24px 24px 18px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(146, 64, 30, 0.1), rgba(146, 64, 30, 0.02));
      }
      .sidebar-body { padding: 18px 24px 24px; }
      .main-body { padding: 24px 28px 34px; }
      h1 {
        margin: 0 0 10px;
        font-size: clamp(2rem, 3vw, 3.15rem);
        line-height: 1.02;
        letter-spacing: -0.035em;
      }
      h2 {
        margin: 0 0 8px;
        font-size: 1.28rem;
        line-height: 1.1;
      }
      h3 {
        margin: 0 0 10px;
        font-size: 1rem;
        line-height: 1.2;
      }
      p { margin: 10px 0; }
      code {
        padding: 1px 5px;
        border-radius: 4px;
        background: #f1e7da;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.9em;
      }
      .subhead,
      .small,
      .tiny { color: var(--muted); }
      .small { font-size: 0.94rem; }
      .tiny { font-size: 0.86rem; }
      .stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 16px;
      }
      .stat {
        padding: 12px 12px 10px;
        border: 1px solid var(--line);
        background: var(--panel);
      }
      .stat .label {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 6px;
      }
      .stat .value {
        font-size: 1.45rem;
        line-height: 1.04;
      }
      .chip-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: #efe5d9;
        font-size: 0.82rem;
      }
      .controls {
        display: grid;
        gap: 10px;
      }
      .tool-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      button {
        font: inherit;
      }
      .btn {
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--card);
        color: var(--ink);
        cursor: pointer;
      }
      .btn:hover {
        background: #fff6eb;
      }
      .btn.primary {
        background: var(--accent);
        color: #fff9f2;
        border-color: #773519;
      }
      .btn.primary:hover {
        background: #a24820;
      }
      .btn.good {
        background: var(--good-soft);
        color: var(--good);
        border-color: #bdd4c1;
      }
      .btn.bad {
        background: var(--bad-soft);
        color: var(--bad);
        border-color: #d8b5ae;
      }
      .card {
        border: 1px solid var(--line);
        background: var(--card);
        overflow: hidden;
      }
      .card-head {
        padding: 16px 18px 14px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(146, 64, 30, 0.07), rgba(146, 64, 30, 0.01));
      }
      .card-body {
        padding: 18px;
      }
      .prompt {
        margin: 14px 0 0;
        padding: 12px 14px;
        background: #fbf4e8;
        border-left: 4px solid var(--accent);
      }
      .arena {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
      }
      .snippet-card {
        border: 1px solid var(--line);
        background: #fffbf5;
      }
      .snippet-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--line);
        background: #f7eddf;
      }
      .snippet-label {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.02em;
      }
      .snippet-body {
        padding: 16px 16px 18px;
      }
      .snippet-body p {
        margin: 0 0 12px;
        font-size: 1rem;
      }
      .snippet-body p:last-child {
        margin-bottom: 0;
      }
      .decision-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 18px;
      }
      .decision-row .btn {
        padding: 14px 12px;
        font-size: 1rem;
      }
      .status-line {
        margin-top: 14px;
        color: var(--muted);
      }
      .winner-card,
      .history-card {
        margin-top: 18px;
        padding: 16px 18px;
        border: 1px solid var(--line);
        background: var(--panel);
      }
      .winner-card.hidden,
      .history-card.hidden {
        display: none;
      }
      .winner-line {
        font-size: 1.2rem;
        line-height: 1.2;
        margin: 8px 0 0;
      }
      .breakdown {
        margin: 14px 0 0;
        padding-left: 18px;
      }
      .history-list {
        margin: 10px 0 0;
        padding: 0;
        list-style: none;
      }
      .history-list li {
        margin: 8px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(215, 195, 170, 0.65);
      }
      .empty {
        padding: 28px;
        border: 1px dashed var(--line);
        background: #fff8ef;
      }
      @media (max-width: 1100px) {
        .shell {
          grid-template-columns: 1fr;
        }
        .sidebar {
          position: static;
        }
      }
      @media (max-width: 840px) {
        .arena,
        .decision-row,
        .stats {
          grid-template-columns: 1fr;
        }
        .main-body,
        .sidebar-head,
        .sidebar-body,
        .main-head {
          padding-left: 18px;
          padding-right: 18px;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="sidebar">
        <div class="sidebar-head">
          <h2>Blind Review</h2>
          <p class="small">Pick the stronger snippet with keys <code>A</code> and <code>B</code>. Use the arrow keys to move.</p>
        </div>
        <div class="sidebar-body">
          <div class="stats">
            <div class="stat">
              <div class="label">Answered</div>
              <div class="value" id="answered-count">0</div>
            </div>
            <div class="stat">
              <div class="label">Remaining</div>
              <div class="value" id="remaining-count">0</div>
            </div>
            <div class="stat">
              <div class="label">Current</div>
              <div class="value" id="current-position">0 / 0</div>
            </div>
            <div class="stat">
              <div class="label">Episodes</div>
              <div class="value" id="episode-progress">–</div>
            </div>
          </div>

          <div class="controls">
            <div class="small">Results persist in local storage. The run winner stays hidden until every comparison has a choice.</div>
          </div>

          <div class="tool-row" style="margin-top: 16px;">
            <button class="btn" id="prev-btn" type="button">Previous</button>
            <button class="btn primary" id="next-btn" type="button">Next</button>
          </div>
          <div class="tool-row">
            <button class="btn" id="jump-unanswered-btn" type="button">Jump to Unanswered</button>
            <button class="btn" id="export-json-btn" type="button">Export Results</button>
          </div>
          <div class="tool-row">
            <button class="btn bad" id="reset-btn" type="button">Reset All Results</button>
          </div>

          <div class="history-card" id="recent-card">
            <h3>Recent Choices</h3>
            <ul class="history-list tiny" id="recent-choices"></ul>
          </div>
        </div>
      </aside>

      <main class="main">
        <div class="main-head">
          <h1>__TITLE__</h1>
          <p class="subhead">__SUBTITLE__</p>
          <div class="chip-list" id="top-meta"></div>
        </div>
        <div class="main-body">
          <div id="comparison-root"></div>
          <section class="winner-card hidden" id="winner-card"></section>
          <section class="history-card hidden" id="audit-card">
            <h3>Completed Mapping Audit</h3>
            <p class="small">Revealed only after all decisions are complete.</p>
            <ul class="history-list tiny" id="audit-list"></ul>
          </section>
        </div>
      </main>
    </div>

    <script id="comparison-payload" type="application/json">
__PAYLOAD_JSON__
    </script>
    <script>
      const payload = JSON.parse(document.getElementById('comparison-payload').textContent);
      const storageKey = `blind-script-comparison::${payload.run_ids.join('|')}::${payload.total_comparisons}::v1`;
      let votes = loadVotes();
      let currentIndex = firstUnansweredIndex();

      const comparisonRoot = document.getElementById('comparison-root');
      const winnerCard = document.getElementById('winner-card');
      const auditCard = document.getElementById('audit-card');
      const auditList = document.getElementById('audit-list');

      function loadVotes() {
        try {
          const raw = localStorage.getItem(storageKey);
          return raw ? JSON.parse(raw) : {};
        } catch (_error) {
          return {};
        }
      }

      function saveVotes() {
        localStorage.setItem(storageKey, JSON.stringify(votes));
      }

      function answeredEntries() {
        return payload.comparisons.filter((entry) => votes[entry.uid]);
      }

      function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, (character) => {
          const replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
          };
          return replacements[character] || character;
        });
      }

      function answeredCount() {
        return answeredEntries().length;
      }

      function firstUnansweredIndex() {
        const index = payload.comparisons.findIndex((entry) => !votes[entry.uid]);
        return index === -1 ? 0 : index;
      }

      function currentEntry() {
        if (payload.comparisons.length === 0) return null;
        currentIndex = Math.max(0, Math.min(currentIndex, payload.comparisons.length - 1));
        return payload.comparisons[currentIndex];
      }

      function choose(entry, side) {
        const chosenRunId = side === 'left' ? entry.left_run_id : entry.right_run_id;
        votes[entry.uid] = {
          choice: side,
          chosen_run_id: chosenRunId,
          rated_at: new Date().toISOString(),
        };
        saveVotes();
        const nextUnanswered = payload.comparisons.findIndex(
          (item, index) => index > currentIndex && !votes[item.uid]
        );
        if (nextUnanswered !== -1) {
          currentIndex = nextUnanswered;
        } else if (currentIndex < payload.comparisons.length - 1) {
          currentIndex += 1;
        }
        render();
      }

      function clearAllVotes() {
        votes = {};
        localStorage.removeItem(storageKey);
        currentIndex = 0;
        render();
      }

      function paragraphsHtml(text) {
        return text
          .split(/\\n\\n+/)
          .map((part) => `<p>${escapeHtml(part)}</p>`)
          .join('');
      }

      function setStat(id, value) {
        document.getElementById(id).textContent = value;
      }

      function updateTopMeta() {
        const meta = document.getElementById('top-meta');
        meta.innerHTML = `
          <span class="pill">${payload.total_comparisons} comparisons</span>
          <span class="pill">2 blind variants</span>
          <span class="pill">same episode number · same relative checkpoint</span>
        `;
      }

      function updateStats() {
        const answered = answeredCount();
        setStat('answered-count', String(answered));
        setStat('remaining-count', String(payload.total_comparisons - answered));
        const currentPosition = payload.total_comparisons === 0
          ? '0 / 0'
          : `${currentIndex + 1} / ${payload.total_comparisons}`;
        setStat('current-position', currentPosition);
        const progressByEpisode = payload.episode_numbers
          .map((episodeNumber, index) => {
            const count = payload.counts_by_episode[index];
            const answeredInEpisode = answeredEntries().filter((entry) => entry.episode_number === episodeNumber).length;
            return `E${episodeNumber} ${answeredInEpisode}/${count}`;
          })
          .join(' · ');
        setStat('episode-progress', progressByEpisode);
      }

      function updateRecentChoices() {
        const root = document.getElementById('recent-choices');
        const items = answeredEntries()
          .map((entry) => ({ entry, vote: votes[entry.uid] }))
          .sort((left, right) => new Date(right.vote.rated_at) - new Date(left.vote.rated_at))
          .slice(0, 8);
        if (items.length === 0) {
          root.innerHTML = '<li>No choices yet.</li>';
          return;
        }
        root.innerHTML = items.map(({ entry, vote }) => {
          const label = vote.choice === 'left' ? 'A' : 'B';
          return `<li>Comparison ${entry.order_index} · Episode ${entry.episode_number} · chose ${label}</li>`;
        }).join('');
      }

      function renderComparison() {
        const entry = currentEntry();
        if (!entry) {
          comparisonRoot.innerHTML = '<div class="empty"><p>No comparisons available.</p></div>';
          return;
        }
        const vote = votes[entry.uid] || null;
        comparisonRoot.innerHTML = `
          <section class="card">
            <div class="card-head">
              <div class="small">Comparison ${entry.order_index} of ${payload.total_comparisons}</div>
              <h2>Episode ${entry.episode_number} · ${escapeHtml(entry.stage_label)}</h2>
              <div class="chip-list">
                <span class="pill">checkpoint ${entry.checkpoint_index} of ${entry.checkpoint_total}</span>
                <span class="pill">blind A/B</span>
              </div>
            </div>
            <div class="card-body">
              <div class="prompt">${escapeHtml(payload.comparison_prompt)}</div>
              <div class="arena">
                <article class="snippet-card">
                  <div class="snippet-head">
                    <div class="snippet-label">A</div>
                    <div class="tiny">${entry.left_source.word_count} words</div>
                  </div>
                  <div class="snippet-body">${paragraphsHtml(entry.left_text)}</div>
                </article>
                <article class="snippet-card">
                  <div class="snippet-head">
                    <div class="snippet-label">B</div>
                    <div class="tiny">${entry.right_source.word_count} words</div>
                  </div>
                  <div class="snippet-body">${paragraphsHtml(entry.right_text)}</div>
                </article>
              </div>
              <div class="decision-row">
                <button class="btn good" id="pick-left-btn" type="button">A is better</button>
                <button class="btn good" id="pick-right-btn" type="button">B is better</button>
              </div>
              <div class="status-line">Current choice: <strong>${vote ? (vote.choice === 'left' ? 'A' : 'B') : 'unanswered'}</strong></div>
            </div>
          </section>
        `;
        document.getElementById('pick-left-btn').addEventListener('click', () => choose(entry, 'left'));
        document.getElementById('pick-right-btn').addEventListener('click', () => choose(entry, 'right'));
      }

      function tallyByRun() {
        const totals = {};
        payload.run_ids.forEach((runId) => {
          totals[runId] = 0;
        });
        answeredEntries().forEach((entry) => {
          const vote = votes[entry.uid];
          totals[vote.chosen_run_id] = (totals[vote.chosen_run_id] || 0) + 1;
        });
        return totals;
      }

      function updateWinnerCard() {
        if (answeredCount() < payload.total_comparisons) {
          winnerCard.classList.add('hidden');
          auditCard.classList.add('hidden');
          winnerCard.innerHTML = '';
          auditList.innerHTML = '';
          return;
        }
        const totals = tallyByRun();
        const ranked = Object.entries(totals).sort((left, right) => right[1] - left[1]);
        const [firstRun, firstScore] = ranked[0];
        const [secondRun, secondScore] = ranked[1];
        const winnerLabel = escapeHtml(payload.run_labels[firstRun] || firstRun);
        const loserLabel = escapeHtml(payload.run_labels[secondRun] || secondRun);
        const isTie = firstScore === secondScore;
        const breakdown = payload.run_ids.map((runId) => {
          const label = escapeHtml(payload.run_labels[runId] || runId);
          return `<li>${label}: ${totals[runId]} wins</li>`;
        }).join('');
        winnerCard.classList.remove('hidden');
        winnerCard.innerHTML = `
          <h2>${isTie ? 'Blind Review Complete: Tie' : 'Blind Review Complete'}</h2>
          <p class="winner-line">${isTie ? `${winnerLabel} and ${loserLabel} tied at ${firstScore}-${secondScore}.` : `Winner: ${winnerLabel} by ${firstScore} to ${secondScore}.`}</p>
          <p class="small">The winner is revealed only after all ${payload.total_comparisons} comparisons have a stored choice.</p>
          <ul class="breakdown">${breakdown}</ul>
        `;

        auditCard.classList.remove('hidden');
        auditList.innerHTML = payload.comparisons.map((entry) => {
          const vote = votes[entry.uid];
          const leftLabel = escapeHtml(payload.run_labels[entry.left_run_id] || entry.left_run_id);
          const rightLabel = escapeHtml(payload.run_labels[entry.right_run_id] || entry.right_run_id);
          const chosenLabel = vote.choice === 'left' ? 'A' : 'B';
          const winner = escapeHtml(payload.run_labels[vote.chosen_run_id] || vote.chosen_run_id);
          return `
            <li>
              Comparison ${entry.order_index} · Episode ${entry.episode_number} ·
              A=${leftLabel} · B=${rightLabel} · chose ${chosenLabel} · winner ${winner}
            </li>
          `;
        }).join('');
      }

      function exportResults() {
        const exportPayload = {
          title: payload.title,
          run_ids: payload.run_ids,
          run_labels: payload.run_labels,
          total_comparisons: payload.total_comparisons,
          answered: answeredCount(),
          totals: tallyByRun(),
          votes,
        };
        const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'blind-script-comparison-results.json';
        link.click();
        URL.revokeObjectURL(url);
      }

      function render() {
        updateTopMeta();
        updateStats();
        updateRecentChoices();
        renderComparison();
        updateWinnerCard();
      }

      document.getElementById('prev-btn').addEventListener('click', () => {
        currentIndex = Math.max(0, currentIndex - 1);
        render();
      });
      document.getElementById('next-btn').addEventListener('click', () => {
        currentIndex = Math.min(payload.total_comparisons - 1, currentIndex + 1);
        render();
      });
      document.getElementById('jump-unanswered-btn').addEventListener('click', () => {
        currentIndex = firstUnansweredIndex();
        render();
      });
      document.getElementById('export-json-btn').addEventListener('click', exportResults);
      document.getElementById('reset-btn').addEventListener('click', () => {
        if (window.confirm('Clear all stored comparison results for this page?')) {
          clearAllVotes();
        }
      });

      document.addEventListener('keydown', (event) => {
        if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
        const entry = currentEntry();
        if (!entry) return;
        if (event.key === 'a' || event.key === 'A') {
          choose(entry, 'left');
        } else if (event.key === 'b' || event.key === 'B') {
          choose(entry, 'right');
        } else if (event.key === 'ArrowLeft') {
          currentIndex = Math.max(0, currentIndex - 1);
          render();
        } else if (event.key === 'ArrowRight') {
          currentIndex = Math.min(payload.total_comparisons - 1, currentIndex + 1);
          render();
        }
      });

      render();
    </script>
  </body>
</html>
"""
    return (
        template.replace("__TITLE__", html.escape(str(payload["title"])))
        .replace("__SUBTITLE__", html.escape(str(payload["subtitle"])))
        .replace("__PAYLOAD_JSON__", _payload_json(payload))
    )


def write_blind_comparison_outputs(
    *,
    run_dirs: Sequence[Path],
    output_html: Path,
    output_json: Path,
    title: str,
    subtitle: str | None = None,
    comparison_prompt: str | None = None,
    counts_by_episode: Sequence[int] = DEFAULT_COUNTS_BY_EPISODE,
) -> dict[str, object]:
    payload = build_blind_comparison_payload(
        run_dirs,
        title=title,
        subtitle=subtitle,
        comparison_prompt=comparison_prompt,
        counts_by_episode=counts_by_episode,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(render_blind_comparison_html(payload), encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_COUNTS_BY_EPISODE",
    "build_blind_comparison_payload",
    "render_blind_comparison_html",
    "write_blind_comparison_outputs",
]
