"""Deterministic narrative-state fold over the enrichment agenda.

The episode-architecture stage used to run serially because each episode's
architecture consumed the *reconciled* post-state of the previous episode. The
enrichment `narrative_agenda` already plans the full cross-episode progression
(question / thread / host moves + takeaways for every episode), so each episode's
entry state can be derived deterministically by folding those planned moves —
no LLM, no backward dependency. That lets architectures run in parallel.

The status-transition maps here are the single source of truth shared with the
heuristic reconciler (`llm/heuristic.py`).
"""

from __future__ import annotations

from podcast_agent.schemas.models import (
    EpisodeNarrativeAgenda,
    NarrativeState,
    NarrativeStrategy,
)

# Canonical move -> status maps. Questions and host mysteries share one map.
QUESTION_STATUS_BY_ACTION: dict[str, str] = {
    "open": "open",
    "advance": "advanced",
    "resolve": "resolved",
    "reframe": "reframed",
}
MEMORY_THREAD_STATUS_BY_ACTION: dict[str, str] = {
    "open": "open",
    "refresh": "refreshed",
    "payoff": "paid_off",
    "retire": "retired",
}
ASSUMPTION_STATUS_BY_ACTION: dict[str, str] = {
    "introduce": "active",
    "weaken": "weakened",
    "revise": "revised",
    "drop": "dropped",
}
THEORY_STATUS_BY_ACTION: dict[str, str] = {
    "propose": "proposed",
    "strengthen": "strengthened",
    "replace": "replaced",
    "retire": "retired",
}


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def apply_agenda_moves(
    state: NarrativeState, agenda: EpisodeNarrativeAgenda
) -> NarrativeState:
    """Apply one episode's planned agenda moves to ``state``, returning the
    resulting state. Mirrors the heuristic reconciler's transition semantics
    (preserve prior entries; overlay the latest action; never silently un-resolve
    an item this episode does not move)."""

    s = state.model_dump(mode="json")
    listener = s.setdefault("listener", {})
    host = s.setdefault("host", {})
    lis = agenda.listener
    hst = agenda.host

    # Known sets accumulate.
    listener["known_explanation_item_ids"] = _dedupe(
        list(listener.get("known_explanation_item_ids", []) or [])
        + list(lis.introduce_explanation_item_ids)
    )
    listener["known_actor_ids"] = _dedupe(
        list(listener.get("known_actor_ids", []) or [])
        + list(lis.introduce_actor_ids)
    )

    # Listener questions.
    questions = {
        str(q.get("question_id", "")): dict(q)
        for q in (listener.get("questions", []) or [])
        if str(q.get("question_id", "")).strip()
    }
    for move in lis.question_moves:
        qid = move.question_id.strip()
        if not qid:
            continue
        entry = questions.get(
            qid, {"question_id": qid, "text": move.text or qid, "status": "open"}
        )
        if move.text:
            entry["text"] = move.text
        entry["status"] = QUESTION_STATUS_BY_ACTION.get(
            move.action, entry.get("status", "open")
        )
        questions[qid] = entry
    listener["questions"] = list(questions.values())

    # Listener memory threads.
    threads = {
        str(t.get("thread_id", "")): dict(t)
        for t in (listener.get("memory_threads", []) or [])
        if str(t.get("thread_id", "")).strip()
    }
    for move in lis.memory_thread_moves:
        tid = move.thread_id.strip()
        if not tid:
            continue
        entry = threads.get(
            tid,
            {
                "thread_id": tid,
                "label": move.label or tid,
                "status": "open",
                "source_promised_beat_id": move.source_promised_beat_id,
            },
        )
        if move.label:
            entry["label"] = move.label
        entry["status"] = MEMORY_THREAD_STATUS_BY_ACTION.get(
            move.action, entry.get("status", "open")
        )
        threads[tid] = entry
    listener["memory_threads"] = list(threads.values())

    # Host mysteries.
    mysteries = {
        str(m.get("mystery_id", "")): dict(m)
        for m in (host.get("mysteries", []) or [])
        if str(m.get("mystery_id", "")).strip()
    }
    for move in hst.mystery_moves:
        mid = move.mystery_id.strip()
        if not mid:
            continue
        entry = mysteries.get(
            mid, {"mystery_id": mid, "text": move.text or mid, "status": "open"}
        )
        if move.text:
            entry["text"] = move.text
        entry["status"] = QUESTION_STATUS_BY_ACTION.get(
            move.action, entry.get("status", "open")
        )
        mysteries[mid] = entry
    host["mysteries"] = list(mysteries.values())

    # Host assumptions.
    assumptions = {
        str(a.get("assumption_id", "")): dict(a)
        for a in (host.get("assumptions", []) or [])
        if str(a.get("assumption_id", "")).strip()
    }
    revisions: list[str] = []
    for move in hst.assumption_moves:
        aid = move.assumption_id.strip()
        if not aid:
            continue
        entry = assumptions.get(
            aid, {"assumption_id": aid, "statement": move.statement or aid, "status": "active"}
        )
        if move.revised_statement:
            entry["statement"] = move.revised_statement
        elif move.statement:
            entry["statement"] = move.statement
        entry["status"] = ASSUMPTION_STATUS_BY_ACTION.get(
            move.action, entry.get("status", "active")
        )
        assumptions[aid] = entry
        if move.action in {"weaken", "revise", "drop"}:
            revisions.append(move.revised_statement or move.statement or aid)
    host["assumptions"] = list(assumptions.values())

    # Host working theories.
    theories = {
        str(t.get("theory_id", "")): dict(t)
        for t in (host.get("working_theories", []) or [])
        if str(t.get("theory_id", "")).strip()
    }
    for move in hst.theory_moves:
        tid = move.theory_id.strip()
        if not tid:
            continue
        entry = theories.get(
            tid, {"theory_id": tid, "statement": move.statement or tid, "status": "proposed"}
        )
        if move.statement:
            entry["statement"] = move.statement
        entry["status"] = THEORY_STATUS_BY_ACTION.get(
            move.action, entry.get("status", "proposed")
        )
        theories[tid] = entry
        if move.action in {"strengthen", "replace", "retire"}:
            revisions.append(move.statement or tid)
    host["working_theories"] = list(theories.values())

    # Carry-forward + takeaways come from this episode's agenda (replace, not accumulate).
    listener["carry_forward_memory"] = [
        item.model_dump(mode="json") for item in lis.carry_forward_memory
    ]
    if lis.episode_takeaway is not None:
        listener["last_episode_takeaway"] = lis.episode_takeaway.model_dump(mode="json")
    if hst.episode_takeaway is not None:
        host["last_episode_takeaway"] = hst.episode_takeaway.model_dump(mode="json")

    # Recent revisions reflect this episode's host motion; confidence is derived.
    host["recent_revisions"] = revisions[-4:]
    unresolved = any(
        m.get("status") in {"open", "advanced"} for m in mysteries.values()
    )
    destabilized = any(
        a.get("status") in {"weakened", "revised", "dropped"} for a in assumptions.values()
    )
    stabilizing = any(
        t.get("status") in {"strengthened", "replaced"} for t in theories.values()
    )
    if unresolved or (destabilized and not stabilizing):
        host["confidence_posture"] = "tentative"
    elif not unresolved and stabilizing:
        host["confidence_posture"] = "grounded"
    else:
        host["confidence_posture"] = "mixed"

    return NarrativeState.model_validate(s)


def fold_planned_pre_states(
    strategy: NarrativeStrategy, project_id: str
) -> dict[int, NarrativeState]:
    """Deterministically fold the enrichment agenda into each episode's entry
    (`pre`) state. ``pre[n]`` is the cumulative planned state entering episode n;
    no LLM and no dependence on realized architecture, so all episodes can be
    built in parallel."""

    episodes = sorted(strategy.episodes, key=lambda ep: ep.episode_number)
    state = NarrativeState(project_id=project_id, next_episode_number=1)
    pre_by_episode: dict[int, NarrativeState] = {}
    for episode in episodes:
        pre_by_episode[episode.episode_number] = state.model_copy(deep=True)
        state = apply_agenda_moves(state, episode.narrative_agenda)
        state.next_episode_number = episode.episode_number + 1
    return pre_by_episode
