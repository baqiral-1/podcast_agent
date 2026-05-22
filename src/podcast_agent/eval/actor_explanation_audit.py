"""Actor-explanation audit helpers and standalone HTML rendering."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any, Sequence

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_BLOCKED_SURNAME_TOKENS = {
    "ali",
    "ayatollah",
    "imam",
    "mehdi",
    "mohammad",
    "reza",
    "ruhollah",
    "shah",
}
_STOPWORDS = {
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
    "had",
    "has",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "who",
    "with",
}


class ActorExplanationAuditError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _check_run_dir(run_dir: Path) -> None:
    required = (
        "actor_metadata.json",
        "narrative_strategy.json",
        "episode_architectures.json",
        "series_plan.json",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ActorExplanationAuditError(
            f"Run directory is missing required artifacts: {', '.join(missing)}"
        )


def _sentence_spans(text: str) -> list[str]:
    parts = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(str(text or "").strip())
        if sentence.strip()
    ]
    return parts if parts else ([str(text).strip()] if str(text).strip() else [])


def _keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _scene_index_by_episode(series_plan: dict[str, Any]) -> dict[int, dict[str, int]]:
    mapping: dict[int, dict[str, int]] = {}
    for episode in series_plan.get("episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        episode_number = episode.get("episode_number")
        scene_cards = episode.get("scene_cards")
        if not isinstance(episode_number, int) or not isinstance(scene_cards, list):
            continue
        mapping[episode_number] = {
            str(scene.get("scene_id")): idx
            for idx, scene in enumerate(scene_cards)
            if isinstance(scene, dict) and scene.get("scene_id")
        }
    return mapping


def _choose_script_path(
    episode_dir: Path,
    script_artifacts: Sequence[str],
) -> Path | None:
    for name in script_artifacts:
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_script_context(
    run_dir: Path,
    *,
    script_artifacts: Sequence[str],
) -> dict[int, dict[str, Any]]:
    contexts: dict[int, dict[str, Any]] = {}
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return contexts
    for episode_dir in sorted(
        (path for path in episodes_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    ):
        try:
            episode_number = int(episode_dir.name)
        except ValueError:
            continue
        script_path = _choose_script_path(episode_dir, script_artifacts)
        if script_path is None:
            contexts[episode_number] = {
                "artifact_name": None,
                "artifact_path": None,
                "sections": [],
                "section_text_by_id": {},
                "section_ids_by_scene_id": {},
            }
            continue
        payload = _load_json(script_path)
        raw_sections = payload.get("prose_sections")
        if not isinstance(raw_sections, list):
            raw_sections = payload.get("sections")
        sections = [section for section in raw_sections or [] if isinstance(section, dict)]
        section_text_by_id = {
            str(section["section_id"]): str(section.get("text", ""))
            for section in sections
            if section.get("section_id")
        }
        section_by_id = {
            str(section["section_id"]): section
            for section in sections
            if section.get("section_id")
        }
        section_ids_by_scene_id: dict[str, list[str]] = {}
        for section in sections:
            section_id = str(section.get("section_id", "") or "")
            for scene_id in section.get("scene_card_ids", []) or []:
                if not scene_id:
                    continue
                section_ids_by_scene_id.setdefault(str(scene_id), []).append(section_id)
        contexts[episode_number] = {
            "artifact_name": script_path.name,
                "artifact_path": str(script_path),
                "sections": sections,
                "section_by_id": section_by_id,
                "section_text_by_id": section_text_by_id,
                "section_ids_by_scene_id": section_ids_by_scene_id,
            }
    return contexts


def _actor_search_phrases(
    actor_profile: dict[str, Any],
    plan_names: Sequence[str],
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        cleaned = str(value or "").strip()
        if not cleaned:
            return
        lowered = cleaned.casefold()
        if lowered in seen:
            return
        seen.add(lowered)
        phrases.append(cleaned)

    _add(str(actor_profile.get("display_name", "") or ""))
    for alias in actor_profile.get("aliases", []) or []:
        _add(str(alias))
    for name in plan_names:
        _add(name)

    for phrase in list(phrases):
        tokens = [token for token in re.split(r"\s+", phrase) if token]
        if len(tokens) < 2:
            continue
        last_token = tokens[-1].casefold().strip(".,;:!?")
        if len(last_token) < 4 or last_token in _BLOCKED_SURNAME_TOKENS:
            continue
        _add(tokens[-1])
    phrases.sort(key=len, reverse=True)
    return phrases


def _phrase_match(sentence: str, phrases: Sequence[str]) -> str | None:
    for phrase in phrases:
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        if pattern.search(sentence):
            return phrase
    return None


def _find_script_hit(
    *,
    section_text: str,
    phrases: Sequence[str],
) -> dict[str, Any] | None:
    sentences = _sentence_spans(section_text)
    for idx, sentence in enumerate(sentences):
        matched_phrase = _phrase_match(sentence, phrases)
        if matched_phrase is None:
            continue
        start = max(idx - 1, 0)
        end = min(idx + 1, len(sentences) - 1)
        return {
            "matched_phrase": matched_phrase,
            "sentence_index": idx,
            "snippet": " ".join(sentences[start : end + 1]),
            "sentence": sentence,
        }
    return None


def _find_script_hit_in_episode(
    *,
    section_text_by_id: dict[str, str],
    phrases: Sequence[str],
) -> tuple[str, dict[str, Any]] | None:
    for section_id, text in section_text_by_id.items():
        hit = _find_script_hit(section_text=text, phrases=phrases)
        if hit is not None:
            return section_id, hit
    return None


def _gloss_overlap(preferred_plain_gloss: str, snippet: str, actor_phrases: Sequence[str]) -> list[str]:
    gloss_tokens = _keyword_tokens(preferred_plain_gloss)
    actor_tokens = set()
    for phrase in actor_phrases:
        actor_tokens.update(_keyword_tokens(phrase))
    candidate_tokens = sorted((gloss_tokens - actor_tokens) & _keyword_tokens(snippet))
    return candidate_tokens


def _find_actor_realization(
    *,
    section: dict[str, Any] | None,
    actor_id: str,
    stage: str,
    scene_id: str,
) -> dict[str, Any] | None:
    if not isinstance(section, dict):
        return None
    for realization in section.get("actor_explanation_realizations", []) or []:
        if not isinstance(realization, dict):
            continue
        if (
            realization.get("actor_id") == actor_id
            and realization.get("stage") == stage
            and realization.get("scene_card_id") == scene_id
        ):
            return realization
    return None


def build_actor_explanation_audit_payload(
    run_dir: Path,
    *,
    script_artifacts: Sequence[str] = (
        "style_audited_script.json",
        "spoken_script.json",
        "episode_script.json",
    ),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    _check_run_dir(run_dir)
    actor_metadata = _load_json(run_dir / "actor_metadata.json")
    strategy = _load_json(run_dir / "narrative_strategy.json")
    architectures = _load_json(run_dir / "episode_architectures.json")
    series_plan = _load_json(run_dir / "series_plan.json")

    actor_by_id = {
        actor["actor_id"]: actor
        for actor in actor_metadata.get("actors", []) or []
        if isinstance(actor, dict) and actor.get("actor_id")
    }
    architecture_by_episode = {
        episode["episode_number"]: episode
        for episode in architectures.get("episodes", []) or []
        if isinstance(episode, dict) and isinstance(episode.get("episode_number"), int)
    }
    plan_by_episode = {
        episode["episode_number"]: episode
        for episode in series_plan.get("episodes", []) or []
        if isinstance(episode, dict) and isinstance(episode.get("episode_number"), int)
    }
    scene_indexes = _scene_index_by_episode(series_plan)
    script_contexts = _load_script_context(run_dir, script_artifacts=script_artifacts)

    issues: list[dict[str, Any]] = []
    obligations_total = 0
    obligations_with_arch = 0
    obligations_with_plan = 0
    obligations_with_script = 0

    actor_cards: list[dict[str, Any]] = []
    registry = strategy.get("series_actor_explanation_registry", []) or []
    strategy_episodes = strategy.get("episodes", []) or []

    for registry_item in registry:
        if not isinstance(registry_item, dict):
            continue
        actor_id = str(registry_item.get("actor_id", "") or "")
        if not actor_id:
            continue
        actor_profile = actor_by_id.get(actor_id, {})
        actor_title = (
            str(actor_profile.get("display_name", "") or "")
            or actor_id.replace("_", " ").title()
        )

        obligations: list[dict[str, Any]] = []
        plan_names: set[str] = set()
        for episode_payload in plan_by_episode.values():
            for scene in episode_payload.get("scene_cards", []) or []:
                for actor in scene.get("actors", []) or []:
                    if isinstance(actor, dict) and actor.get("actor_id") == actor_id:
                        name = str(actor.get("name", "") or "").strip()
                        if name:
                            plan_names.add(name)
        actor_phrases = _actor_search_phrases(actor_profile, sorted(plan_names))

        for strategy_episode in strategy_episodes:
            if not isinstance(strategy_episode, dict):
                continue
            episode_number = strategy_episode.get("episode_number")
            if not isinstance(episode_number, int):
                continue
            contract = strategy_episode.get("authorial_contract", {}) or {}
            for stage_key, stage_name in (
                ("introduce_actor_ids", "introduce"),
                ("remind_actor_ids", "reminder"),
            ):
                actor_ids = contract.get(stage_key, []) or []
                if actor_id not in actor_ids:
                    continue
                obligations_total += 1
                architecture_episode = architecture_by_episode.get(episode_number, {})
                architecture_matches = []
                for section in architecture_episode.get("sections", []) or []:
                    if not isinstance(section, dict):
                        continue
                    for explanation in section.get("actor_explanations", []) or []:
                        if (
                            isinstance(explanation, dict)
                            and explanation.get("actor_id") == actor_id
                            and explanation.get("stage") == stage_name
                        ):
                            architecture_matches.append(
                                {
                                    "section_id": str(section.get("section_id", "") or ""),
                                    "section_anchor": str(
                                        section.get("section_anchor", "") or ""
                                    ),
                                    "background_depth": str(
                                        explanation.get("background_depth", "") or ""
                                    ),
                                    "role_label": str(
                                        explanation.get("role_label", "") or ""
                                    ),
                                    "source_passage_ids": [
                                        str(passage_id)
                                        for passage_id in (
                                            explanation.get("source_passage_ids", []) or []
                                        )
                                        if str(passage_id).strip()
                                    ],
                                    "intro_facts": [
                                        str(fact)
                                        for fact in (
                                            explanation.get("intro_facts", []) or []
                                        )
                                        if str(fact).strip()
                                    ],
                                    "why_now": str(explanation.get("why_now", "") or ""),
                                    "preferred_plain_gloss": str(
                                        explanation.get("preferred_plain_gloss", "") or ""
                                    ),
                                }
                            )
                if architecture_matches:
                    obligations_with_arch += 1
                else:
                    issues.append(
                        {
                            "severity": "error",
                            "episode_number": episode_number,
                            "actor_id": actor_id,
                            "stage": stage_name,
                            "kind": "architecture_missing",
                            "message": f"{actor_title}: strategy {stage_name} obligation has no architecture actor_explanations placement.",
                        }
                    )

                plan_episode = plan_by_episode.get(episode_number, {})
                plan_scene_cards = plan_episode.get("scene_cards", []) or []
                all_plan_mentions = []
                plan_matches = []
                first_seen_index: int | None = None
                first_intro_index: int | None = None
                for scene in plan_scene_cards:
                    if not isinstance(scene, dict):
                        continue
                    scene_id = str(scene.get("scene_id", "") or "")
                    scene_index = scene_indexes.get(episode_number, {}).get(scene_id)
                    for actor in scene.get("actors", []) or []:
                        if not isinstance(actor, dict) or actor.get("actor_id") != actor_id:
                            continue
                        mention = {
                            "scene_id": scene_id,
                            "scene_index": scene_index,
                            "section_id": str(scene.get("section_id", "") or ""),
                            "scene_title": str(scene.get("title", "") or ""),
                            "name": str(actor.get("name", "") or ""),
                            "presence": str(actor.get("presence", "") or ""),
                            "explanation_stage": actor.get("explanation_stage"),
                            "background_depth": str(actor.get("background_depth", "") or ""),
                            "role_label": str(actor.get("role_label", "") or ""),
                            "source_passage_ids": [
                                str(passage_id)
                                for passage_id in (actor.get("source_passage_ids", []) or [])
                                if str(passage_id).strip()
                            ],
                            "intro_facts": [
                                str(fact)
                                for fact in (actor.get("intro_facts", []) or [])
                                if str(fact).strip()
                            ],
                            "why_now": str(actor.get("why_now", "") or ""),
                            "preferred_plain_gloss": str(
                                actor.get("preferred_plain_gloss", "") or ""
                            ),
                        }
                        all_plan_mentions.append(mention)
                        if first_seen_index is None or (
                            scene_index is not None and scene_index < first_seen_index
                        ):
                            first_seen_index = scene_index
                        if actor.get("explanation_stage") == "introduce" and (
                            first_intro_index is None
                            or (scene_index is not None and scene_index < first_intro_index)
                        ):
                            first_intro_index = scene_index
                        if actor.get("explanation_stage") == stage_name:
                            plan_matches.append(mention)

                if plan_matches:
                    obligations_with_plan += 1
                else:
                    issues.append(
                        {
                            "severity": "error",
                            "episode_number": episode_number,
                            "actor_id": actor_id,
                            "stage": stage_name,
                            "kind": "plan_missing",
                            "message": f"{actor_title}: architecture {stage_name} obligation never lands on a scene actor in the plan.",
                        }
                    )

                if stage_name == "introduce" and first_seen_index is not None and (
                    first_intro_index is None or first_seen_index < first_intro_index
                ):
                    issues.append(
                        {
                            "severity": "warn",
                            "episode_number": episode_number,
                            "actor_id": actor_id,
                            "stage": stage_name,
                            "kind": "late_intro_in_plan",
                            "message": f"{actor_title}: actor is mentioned in an earlier scene before the tagged introduction scene.",
                        }
                    )

                script_context = script_contexts.get(episode_number, {})
                section_by_id = script_context.get("section_by_id", {}) or {}
                section_text_by_id = script_context.get("section_text_by_id", {}) or {}
                section_ids_by_scene_id = (
                    script_context.get("section_ids_by_scene_id", {}) or {}
                )
                script_audits = []
                script_found = False
                for plan_match in plan_matches:
                    mapped_section_ids = section_ids_by_scene_id.get(
                        plan_match["scene_id"], []
                    )
                    realization_payload: dict[str, Any] | None = None
                    hit_payload: dict[str, Any] | None = None
                    status = "missing"
                    mapped_section_id = mapped_section_ids[0] if mapped_section_ids else ""
                    if mapped_section_id:
                        realization_payload = _find_actor_realization(
                            section=section_by_id.get(mapped_section_id),
                            actor_id=actor_id,
                            stage=stage_name,
                            scene_id=plan_match["scene_id"],
                        )
                        if realization_payload is not None:
                            status = "mapped_section_realization"
                            script_found = True
                        else:
                            mapped_text = str(
                                section_text_by_id.get(mapped_section_id, "") or ""
                            )
                            hit_payload = _find_script_hit(
                                section_text=mapped_text,
                                phrases=actor_phrases,
                            )
                            if hit_payload is not None:
                                status = "mapped_section_hit"
                                script_found = True
                    if realization_payload is None and hit_payload is None:
                        other_hit = _find_script_hit_in_episode(
                            section_text_by_id=section_text_by_id,
                            phrases=actor_phrases,
                        )
                        if other_hit is not None:
                            other_section_id, hit_payload = other_hit
                            if not mapped_section_id or other_section_id != mapped_section_id:
                                status = "episode_hit_elsewhere"
                            else:
                                status = "mapped_section_hit"
                            script_found = True
                            if not mapped_section_id:
                                mapped_section_id = other_section_id

                    script_audits.append(
                        {
                            "scene_id": plan_match["scene_id"],
                            "mapped_section_id": mapped_section_id,
                            "status": status,
                            "matched_phrase": (
                                hit_payload["matched_phrase"] if hit_payload else ""
                            ),
                            "snippet": (
                                str(realization_payload.get("text_span", "") or "")
                                if realization_payload is not None
                                else hit_payload["snippet"]
                                if hit_payload
                                else ""
                            ),
                            "source_passage_ids": [
                                str(passage_id)
                                for passage_id in (
                                    realization_payload.get("source_passage_ids", [])
                                    if realization_payload is not None
                                    else []
                                )
                                if str(passage_id).strip()
                            ],
                            "realized_facts": [
                                str(fact)
                                for fact in (
                                    realization_payload.get("realized_facts", [])
                                    if realization_payload is not None
                                    else []
                                )
                                if str(fact).strip()
                            ],
                        }
                    )
                    if status not in {"mapped_section_realization", "mapped_section_hit"}:
                        issues.append(
                            {
                                "severity": "warn",
                                "episode_number": episode_number,
                                "actor_id": actor_id,
                                "stage": stage_name,
                                "kind": "script_drift",
                                "message": f"{actor_title}: planned {stage_name} scene does not cleanly surface in the mapped script section.",
                            }
                        )
                if script_found:
                    obligations_with_script += 1

                obligations.append(
                    {
                        "episode_number": episode_number,
                        "stage": stage_name,
                        "strategy": {
                            "background_depth": (
                                str(registry_item.get("first_background_depth", "") or "")
                                if stage_name == "introduce"
                                else "appositive"
                            ),
                            "later_episode_policy": str(
                                registry_item.get("later_episode_policy", "") or ""
                            ),
                            "preferred_plain_gloss": str(
                                registry_item.get("preferred_plain_gloss", "") or ""
                            ),
                        },
                        "architecture_matches": architecture_matches,
                        "plan_matches": plan_matches,
                        "all_plan_mentions": all_plan_mentions,
                        "script_audits": script_audits,
                        "final_script_artifact": script_context.get("artifact_name"),
                    }
                )

        actor_cards.append(
            {
                "actor_id": actor_id,
                "display_name": actor_title,
                "actor_type": str(actor_profile.get("actor_type", "") or ""),
                "registry": registry_item,
                "obligations": obligations,
            }
        )

    issues.sort(
        key=lambda issue: (
            0 if issue["severity"] == "error" else 1,
            issue["episode_number"],
            issue["actor_id"],
            issue["stage"],
            issue["kind"],
        )
    )
    actor_cards.sort(key=lambda card: card["display_name"].casefold())

    return {
        "title": f"Actor Explanation Audit: {run_dir.name}",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "script_artifacts_preference": list(script_artifacts),
        "summary": {
            "registry_actor_count": len(actor_cards),
            "obligation_count": obligations_total,
            "obligations_with_architecture": obligations_with_arch,
            "obligations_with_plan": obligations_with_plan,
            "obligations_with_script_hit": obligations_with_script,
            "issue_count": len(issues),
        },
        "issues": issues,
        "actors": actor_cards,
    }


def render_actor_explanation_audit_html(payload: dict[str, Any]) -> str:
    def _pill(text: str, klass: str = "") -> str:
        class_attr = f"pill {klass}".strip()
        return f"<span class='{class_attr}'>{escape(text)}</span>"

    summary = payload["summary"]
    issue_items = "".join(
        f"<li><strong>Episode {issue['episode_number']}</strong> · <code>{escape(issue['actor_id'])}</code> · {escape(issue['stage'])}: {escape(issue['message'])}</li>"
        for issue in payload["issues"]
    ) or "<li>No flagged issues.</li>"

    actor_cards_html: list[str] = []
    for actor in payload["actors"]:
        obligation_blocks: list[str] = []
        for obligation in actor["obligations"]:
            architecture_html = "".join(
                "<li><code>{section_id}</code> · {stage} · {depth}<br><span class='muted'>{role}</span><br><span class='tiny'>Passages: {passages}</span><br><span class='tiny'>Facts: {facts}</span><br><span class='tiny'>Why now: {why_now}</span></li>".format(
                    section_id=escape(match["section_id"]),
                    stage=escape(obligation["stage"]),
                    depth=escape(match["background_depth"]),
                    role=escape(match["role_label"] or match["preferred_plain_gloss"] or "No role label"),
                    passages=escape(", ".join(match["source_passage_ids"]) or "none"),
                    facts=escape(" | ".join(match["intro_facts"]) or "none"),
                    why_now=escape(match["why_now"] or "none"),
                )
                for match in obligation["architecture_matches"]
            ) or "<li>Missing architecture placement.</li>"
            plan_html = "".join(
                "<li>Scene {scene_index} · <code>{scene_id}</code> · {title} · {presence}{stage}<br><span class='tiny'>{role}</span><br><span class='tiny'>Passages: {passages}</span></li>".format(
                    scene_index=escape(
                        str(match["scene_index"] + 1)
                        if isinstance(match["scene_index"], int)
                        else "?"
                    ),
                    scene_id=escape(match["scene_id"]),
                    title=escape(match["scene_title"]),
                    presence=escape(match["presence"] or "unknown"),
                    stage=(
                        f" · {escape(str(match['explanation_stage']))}"
                        if match.get("explanation_stage")
                        else ""
                    ),
                    role=escape(match["role_label"] or match["preferred_plain_gloss"] or "No role label"),
                    passages=escape(", ".join(match["source_passage_ids"]) or "none"),
                )
                for match in obligation["plan_matches"]
            ) or "<li>Missing plan scene placement.</li>"
            script_html = "".join(
                "<li><code>{scene_id}</code> → <code>{section_id}</code> · {status}<br><blockquote>{snippet}</blockquote><div class='tiny'>Matched: {phrase} · Passages: {passages} · Realized facts: {facts}</div></li>".format(
                    scene_id=escape(audit["scene_id"]),
                    section_id=escape(audit["mapped_section_id"] or "missing"),
                    status=escape(audit["status"]),
                    snippet=escape(audit["snippet"] or "No actor mention found in the chosen script artifact."),
                    phrase=escape(audit["matched_phrase"] or "none"),
                    passages=escape(", ".join(audit["source_passage_ids"]) or "none"),
                    facts=escape(" | ".join(audit["realized_facts"]) or "none"),
                )
                for audit in obligation["script_audits"]
            ) or "<li>No script audit rows because no plan placement was found.</li>"
            obligation_blocks.append(
                """
                <details class='obligation' open>
                  <summary>
                    <strong>Episode {episode_number}</strong>
                    {stage_pill}
                    {artifact_pill}
                  </summary>
                  <div class='obligation-grid'>
                    <section>
                      <h4>Strategy</h4>
                      <p><strong>Depth:</strong> {depth}</p>
                      <p class='muted'>Later policy: {policy}</p>
                    </section>
                    <section>
                      <h4>Architecture</h4>
                      <ul>{architecture_html}</ul>
                    </section>
                    <section>
                      <h4>Plan</h4>
                      <ul>{plan_html}</ul>
                    </section>
                    <section>
                      <h4>Final Script</h4>
                      <ul>{script_html}</ul>
                    </section>
                  </div>
                </details>
                """.format(
                    episode_number=escape(str(obligation["episode_number"])),
                    stage_pill=_pill(obligation["stage"], obligation["stage"]),
                    artifact_pill=_pill(
                        str(obligation.get("final_script_artifact") or "no script"),
                        "artifact",
                    ),
                    depth=escape(obligation["strategy"]["background_depth"]),
                    policy=escape(obligation["strategy"]["later_episode_policy"]),
                    architecture_html=architecture_html,
                    plan_html=plan_html,
                    script_html=script_html,
                )
            )
        actor_cards_html.append(
            """
            <section class='actor-card'>
              <header>
                <h3>{display_name}</h3>
                <div class='meta'>
                  {actor_id_pill}
                  {type_pill}
                  {intro_pill}
                  {policy_pill}
                </div>
                <p class='muted'>{gloss}</p>
              </header>
              {obligations}
            </section>
            """.format(
                display_name=escape(actor["display_name"]),
                actor_id_pill=_pill(actor["actor_id"], "actor-id"),
                type_pill=_pill(actor["actor_type"] or "unknown", "type"),
                intro_pill=_pill(
                    f"intro ep {actor['registry'].get('introduction_episode_number')}",
                    "intro-ep",
                ),
                policy_pill=_pill(
                    str(actor["registry"].get("later_episode_policy", "") or ""),
                    "policy",
                ),
                gloss=escape(
                    str(actor["registry"].get("preferred_plain_gloss", "") or "")
                    or "Strategy no longer owns the intro prose; see section briefs below."
                ),
                obligations="".join(obligation_blocks) or "<p>No episode obligations found.</p>",
            )
        )

    payload_json = escape(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(str(payload['title']))}</title>
  <style>
    :root {{
      --bg: #f2eadf;
      --paper: #fffaf4;
      --ink: #251b14;
      --muted: #6f6357;
      --line: #ddcbb7;
      --accent: #8f3b1d;
      --accent-soft: #f4e4d7;
      --warn: #8e5b00;
      --warn-soft: #f7edd9;
      --error: #8b2d22;
      --error-soft: #f5dfdc;
      --ok: #28563f;
      --ok-soft: #e6efe9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff8ef 0, #f2eadf 45%, #e6d6c5 100%);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1320px, calc(100vw - 28px));
      margin: 20px auto 44px;
      display: grid;
      gap: 18px;
    }}
    section, header {{
      display: block;
    }}
    .hero, .panel, .actor-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 16px 40px rgba(45, 29, 14, 0.08);
    }}
    .hero {{
      padding: 28px 30px 24px;
      background: linear-gradient(180deg, rgba(143, 59, 29, 0.10), rgba(143, 59, 29, 0.02));
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 1.35rem;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 1.5rem;
    }}
    h4 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    p {{ margin: 10px 0; }}
    code {{
      background: #f0e8dd;
      border-radius: 4px;
      padding: 1px 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.9em;
    }}
    .muted, .tiny {{
      color: var(--muted);
    }}
    .tiny {{
      font-size: 0.88rem;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      padding: 14px 14px 12px;
      background: #fbf2e8;
      border: 1px solid var(--line);
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 1.5rem;
      line-height: 1.05;
    }}
    .panel {{
      padding: 22px 26px;
    }}
    .issues {{
      margin: 0;
      padding-left: 20px;
    }}
    .issues li {{
      margin: 8px 0;
    }}
    .actor-card {{
      padding: 22px 24px 24px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 12px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #efe5d7;
      font-size: 0.82rem;
    }}
    .pill.introduce {{
      background: var(--ok-soft);
      border-color: #bfd3c5;
      color: var(--ok);
    }}
    .pill.reminder {{
      background: var(--warn-soft);
      border-color: #dbc79a;
      color: var(--warn);
    }}
    .pill.artifact {{
      background: var(--accent-soft);
      border-color: #e0bca4;
      color: var(--accent);
    }}
    .obligation {{
      margin-top: 16px;
      border: 1px solid var(--line);
      background: #fffdf9;
    }}
    .obligation > summary {{
      cursor: pointer;
      padding: 12px 14px;
      background: #f9f1e7;
      border-bottom: 1px solid var(--line);
    }}
    .obligation-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      padding: 16px 16px 6px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li {{
      margin: 8px 0;
    }}
    blockquote {{
      margin: 8px 0 6px;
      padding: 10px 12px;
      background: #fbf4ea;
      border-left: 3px solid var(--accent);
    }}
    details.data {{
      margin-top: 18px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f7efe4;
      border: 1px solid var(--line);
      padding: 14px;
      overflow: auto;
    }}
    @media (max-width: 880px) {{
      .obligation-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{escape(str(payload["title"]))}</h1>
      <p class="muted">Run: <code>{escape(str(payload["run_id"]))}</code></p>
      <div class="summary-grid">
        <div class="stat"><div class="label">Registry actors</div><div class="value">{summary["registry_actor_count"]}</div></div>
        <div class="stat"><div class="label">Strategy obligations</div><div class="value">{summary["obligation_count"]}</div></div>
        <div class="stat"><div class="label">Arch placements</div><div class="value">{summary["obligations_with_architecture"]}</div></div>
        <div class="stat"><div class="label">Plan placements</div><div class="value">{summary["obligations_with_plan"]}</div></div>
        <div class="stat"><div class="label">Script hits</div><div class="value">{summary["obligations_with_script_hit"]}</div></div>
        <div class="stat"><div class="label">Flagged issues</div><div class="value">{summary["issue_count"]}</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Flagged Issues</h2>
      <ul class="issues">{issue_items}</ul>
    </section>
    {''.join(actor_cards_html)}
    <section class="panel">
      <details class="data">
        <summary><strong>Embedded Payload</strong></summary>
        <pre>{payload_json}</pre>
      </details>
    </section>
  </main>
</body>
</html>"""
