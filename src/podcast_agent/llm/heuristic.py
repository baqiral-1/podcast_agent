"""Heuristic LLM implementation used for local development and tests."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient, PromptPayload, prompt_log_metadata
from podcast_agent.narrative_state import (
    ASSUMPTION_STATUS_BY_ACTION,
    MEMORY_THREAD_STATUS_BY_ACTION,
    QUESTION_STATUS_BY_ACTION,
    THEORY_STATUS_BY_ACTION,
)
from podcast_agent.schemas.models import (
    PodcastMode,
    authorial_passage_target_for_mode,
    scene_discovery_candidate_range_for_mode,
    scene_job_budget_for_mode,
)


class HeuristicLLMClient(LLMClient):
    """Deterministic stand-in that produces minimal valid JSON for any schema."""

    def __init__(self) -> None:
        super().__init__()

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
        attempt: int | None = None,
        max_attempts: int | None = None,
    ) -> BaseModel:
        if self.run_logger is not None:
            self.run_logger.log(
                "llm_request",
                client="heuristic",
                schema_name=schema_name,
                **prompt_log_metadata(instructions, payload),
            )
        try:
            generator = getattr(self, f"_generate_{schema_name}", None)
            if generator is None:
                response = self._generate_default(schema_name, payload)
            else:
                response = generator(payload)
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_response",
                    client="heuristic",
                    schema_name=schema_name,
                    response=response,
                )
            return response_model.model_validate(response)
        except Exception as exc:
            if self.run_logger is not None:
                self.run_logger.log(
                    "llm_error",
                    client="heuristic",
                    schema_name=schema_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    instructions=instructions,
                    payload=payload,
                )
            raise

    def _generate_default(
        self, schema_name: str, payload: PromptPayload
    ) -> dict[str, Any]:
        raise ValueError(f"No heuristic generator for schema '{schema_name}'.")

    def _generate_chapter_summary(self, payload: PromptPayload) -> dict[str, Any]:
        chapter_title = str(payload.get("chapter_title", "")).strip() or "This chapter"
        return {
            "analysis": {
                "themes_touched": [chapter_title],
                "major_actors": [],
                "key_events_or_arguments": [
                    f"{chapter_title} contributes material relevant to the project theme."
                ],
            },
        }

    def _generate_book_summary(self, payload: PromptPayload) -> dict[str, Any]:
        theme = str(payload.get("theme", "")).strip()
        title = str(payload.get("title", "")).strip() or "this book"
        return {
            "summary": f"{title} addresses {theme or 'the project theme'} through its chapter structure."
        }

    def _generate_theme_decomposition(self, payload: PromptPayload) -> dict[str, Any]:
        books = payload.get("books", [])
        book_ids = [b.get("book_id", f"b{i}") for i, b in enumerate(books)]
        relevance = {bid: 0.7 for bid in book_ids}
        axis_label = str(payload.get("theme", "unknown"))
        axis_count_min = int(payload.get("axis_count_min", 12))
        actors = self._generate_actor_metadata_actors(books)
        actor_ids = [actor["actor_id"] for actor in actors]
        axes = []
        for idx in range(1, axis_count_min + 1):
            importance = max(0.0, 1.0 - ((idx - 1) * 0.06))
            axes.append(
                {
                    "axis_id": f"axis_{idx:02d}",
                    "name": f"{axis_label} axis {idx}",
                    "description": f"Heuristic thematic axis {idx} for {axis_label}.",
                    "theme_importance_score": round(importance, 3),
                    "guiding_questions": [
                        f"How does axis {idx} appear across the books?",
                        f"What changes become legible through axis {idx}?",
                    ],
                    "relevance_by_book": relevance,
                    "keywords": ["theme", f"axis_{idx}"],
                    "actor_ids": actor_ids[:3],
                }
            )
        return {
            "axes": axes,
            "actor_metadata": {
                "actors": actors,
                "relationships": [],
                "unresolved_mentions": [],
                "quality_notes": ["Heuristic actor metadata."],
            },
        }

    def _generate_actor_metadata_actors(
        self, books: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        actors: list[dict[str, Any]] = []
        seen: set[str] = set()
        for book in books:
            book_id = str(book.get("book_id", ""))
            for chapter in book.get("chapters", []) or []:
                analysis = chapter.get("analysis") or {}
                candidates = [
                    *(analysis.get("major_actors", []) or []),
                ]
                for raw_name in candidates:
                    name = str(raw_name or "").strip()
                    if not name:
                        continue
                    actor_id = self._slug_actor_id(name)
                    if not actor_id or actor_id in seen:
                        continue
                    seen.add(actor_id)
                    actors.append(
                        {
                            "actor_id": actor_id,
                            "display_name": name,
                            "aliases": [],
                            "actor_type": "person",
                            "description": f"Heuristic actor derived from {chapter.get('title', 'chapter analysis')}.",
                            "book_ids": [book_id] if book_id else [],
                            "chapter_refs": [
                                {
                                    "book_id": book_id,
                                    "chapter_id": str(chapter.get("chapter_id", "")),
                                    "chapter_title": str(chapter.get("title", "")),
                                }
                            ],
                            "narrative_functions": ["other"],
                            "goals_or_motivational_pressures": [],
                            "constraints": [],
                            "stakes": [],
                            "transformations": [],
                            "uncertainty_notes": "",
                            "evidence_confidence": "medium",
                            "narrative_tier": "supporting",
                            "series_scope": "local",
                        }
                    )
                    if len(actors) >= 20:
                        return actors
        if not actors:
            actors.append(
                {
                    "actor_id": "project_actor",
                    "display_name": "Project actor",
                    "aliases": [],
                    "actor_type": "other",
                    "description": "Heuristic fallback actor for local runs.",
                    "book_ids": book_ids
                    if (
                        book_ids := [
                            str(book.get("book_id", ""))
                            for book in books
                            if book.get("book_id")
                        ]
                    )
                    else [],
                    "chapter_refs": [],
                    "narrative_functions": ["other"],
                    "goals_or_motivational_pressures": [],
                    "constraints": [],
                    "stakes": [],
                    "transformations": [],
                    "uncertainty_notes": "No chapter actor candidates were available.",
                    "evidence_confidence": "low",
                    "narrative_tier": "minor",
                    "series_scope": "local",
                }
            )
        return actors

    def _slug_actor_id(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if not slug:
            return ""
        if slug[0].isdigit():
            slug = f"actor_{slug}"
        return slug[:64]

    def _generate_passage_extraction(self, payload: PromptPayload) -> dict[str, Any]:
        candidates = payload.get("candidate_passages", [])
        passages = []
        for candidate in candidates[:5]:
            passages.append(
                {
                    "passage_id": candidate.get("passage_id", uuid4().hex),
                    "relevance_score": 0.7,
                    "quotability_score": 0.6,
                    "synthesis_tags": ["independent"],
                }
            )
        return {"passages": passages, "cross_book_pairs": []}

    def _collect_passage_ids(self, passages_by_axis: dict[str, Any]) -> list[str]:
        passage_ids: list[str] = []
        for axis_passages in passages_by_axis.values():
            for item in axis_passages:
                if not isinstance(item, dict):
                    continue
                if "passages" not in item:
                    passage_ids.append(str(item.get("passage_id", uuid4().hex)))
                    continue
                for passage in item.get("passages", []):
                    if not isinstance(passage, dict):
                        continue
                    passage_ids.append(str(passage.get("passage_id", uuid4().hex)))
        return passage_ids

    def _build_base_primitive(
        self,
        *,
        substrate: str,
        title: str,
        passage_ids: list[str],
        actor_ids: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actor_ids = actor_ids or []
        primitive: dict[str, Any] = {
            "substrate": substrate,
            "title": title,
            "core_passage_ids": passage_ids[:1] or [uuid4().hex],
            "timeframe": None,
            "geography": None,
            "functions": [],
            "salience": None,
        }
        support_passage_ids = passage_ids[1:3]
        if support_passage_ids:
            primitive["support_passage_ids"] = support_passage_ids
        compact_actor_ids = actor_ids[:2]
        if compact_actor_ids:
            primitive["actor_ids"] = compact_actor_ids
        if extra_fields:
            primitive.update(extra_fields)
        return primitive

    def _generate_primitive_substrate_extraction(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        passages_by_axis = payload.get("passages_by_axis", {})
        passage_ids = self._collect_passage_ids(passages_by_axis) or [
            uuid4().hex,
            uuid4().hex,
        ]
        actor_metadata = payload.get("actor_metadata", {}) or {}
        actor_ids = [
            str(actor.get("actor_id", ""))
            for actor in actor_metadata.get("actors", [])
            if actor.get("actor_id")
        ]
        primitives = [
            self._build_base_primitive(
                substrate="events",
                title="Heuristic event",
                passage_ids=passage_ids[:3],
                actor_ids=actor_ids[:1],
                extra_fields={
                    "event_type": "crisis",
                    "what_happened": "A decisive event concentrates pressure in one place and time.",
                },
            ),
            self._build_base_primitive(
                substrate="acts",
                title="Heuristic act",
                passage_ids=passage_ids[:3],
                actor_ids=actor_ids[:1],
                extra_fields={
                    "act_type": "decision",
                    "acting_subject": actor_ids[0] if actor_ids else "A named actor",
                    "act_summary": "A concrete choice or refusal narrows the next set of options.",
                },
            ),
            self._build_base_primitive(
                substrate="actor_portraits",
                title="Heuristic actor pressure",
                passage_ids=passage_ids[:3],
                actor_ids=actor_ids[:1],
                extra_fields={
                    "focus_actor_id": actor_ids[0] if actor_ids else None,
                    "actor_label": "A pressure-bearing actor",
                    "goal_or_project": "Protect a vulnerable position.",
                    "stakes_or_fears": "Loss of leverage or safety.",
                },
            ),
            self._build_base_primitive(
                substrate="mechanisms",
                title="Heuristic mechanism",
                passage_ids=passage_ids[:3],
                extra_fields={
                    "mechanism_name": "A political-administrative chain",
                    "failure_mode": "The chain distorts under stress.",
                },
            ),
            self._build_base_primitive(
                substrate="conditions",
                title="Heuristic condition",
                passage_ids=passage_ids[:3],
                extra_fields={
                    "condition_type": "fault_line",
                    "condition_summary": "A standing political condition narrows later choices.",
                },
            ),
            self._build_base_primitive(
                substrate="artifacts",
                title="Heuristic artifact",
                passage_ids=passage_ids[:3],
                extra_fields={
                    "artifact_type": "object",
                    "artifact_label": "A visible material object or place",
                },
            ),
            self._build_base_primitive(
                substrate="readings",
                title="Heuristic reading",
                passage_ids=passage_ids[:3],
                extra_fields={
                    "reading_type": "interpretive_claim",
                    "subject_of_reading": "The central development",
                    "attributed_to": "A visible interpreter",
                    "reading_summary": "The same evidence can bear more than one plausible reading.",
                },
            ),
        ]
        grouped = {
            "events": [],
            "acts": [],
            "actor_portraits": [],
            "mechanisms": [],
            "conditions": [],
            "artifacts": [],
            "readings": [],
        }
        for primitive in primitives:
            grouped[str(primitive["substrate"])].append(
                {key: value for key, value in primitive.items() if key != "substrate"}
            )
        return {
            "project_id": payload.get("project_id", "project"),
            **grouped,
            "quality_score": 0.5,
            "quality_notes": ["Heuristic primitives artifact."],
        }

    def _generate_synthesis_primitives(self, payload: PromptPayload) -> dict[str, Any]:
        return self._generate_primitive_substrate_extraction(payload)

    def _generate_primitive_function_tagging_batch(
        self, payload: PromptPayload, *, substrate: str
    ) -> dict[str, Any]:
        base_primitives = payload.get("base_primitives", []) or []
        tagged_primitives: list[dict[str, Any]] = []
        for primitive in base_primitives:
            updated = dict(primitive)
            updated["salience"] = {
                "score": 0.7 if substrate in {"events", "acts"} else 0.55,
                "justification": "Heuristic: this primitive is useful for downstream selection.",
            }
            if substrate == "events":
                updated.setdefault(
                    "event_result",
                    "Actors now have to respond to the changed situation.",
                )
                updated["functions"] = ["pivot", "texture"]
                updated["pivot"] = {
                    "what_changed": "The event forces the next phase of the story into a new channel.",
                    "irreversibility": "med",
                }
                updated["texture"] = {
                    "what_it_anchors": "A visible moment the listener can step into.",
                }
            elif substrate == "acts":
                updated.setdefault(
                    "immediate_result",
                    "The move forces a new response.",
                )
                updated["functions"] = ["pivot", "stake"]
                updated["pivot"] = {
                    "what_changed": "The act narrows what the actors can now do next.",
                    "irreversibility": "med",
                }
                updated["stake"] = {
                    "whose": primitive.get("acting_subject") or "The acting subject",
                    "what_at_stake": "Authority and immediate leverage.",
                }
            elif substrate == "actor_portraits":
                updated.setdefault(
                    "operating_pressure",
                    "Institutions and rivals narrow the available room to move.",
                )
                updated["functions"] = ["stake", "complication"]
                updated["stake"] = {
                    "whose": primitive.get("actor_label") or "The actor",
                    "what_at_stake": primitive.get("stakes_or_fears")
                    or "Status and survival.",
                }
                updated["complication"] = {
                    "what_is_compromised": "A clean reading of the actor's available options.",
                    "why_no_clean_option": "Every live move damages something binding or risky.",
                }
            elif substrate == "mechanisms":
                updated.setdefault(
                    "operating_chain",
                    [
                        "Orders move downward through an institution.",
                        "Intermediaries translate them into local pressure.",
                    ],
                )
                updated.setdefault("inputs", ["orders"])
                updated.setdefault("outputs", ["compliance"])
                updated["functions"] = ["complication", "recurrence"]
                updated["complication"] = {
                    "what_is_compromised": "Any simple explanation that ignores the operating chain.",
                    "why_no_clean_option": "The machinery keeps producing pressure even when individual actors hesitate.",
                }
                updated["recurrence"] = {
                    "connects_to": [],
                    "meaning_accrued": "The same machinery can reappear later as a callback or explanatory spine.",
                }
            elif substrate == "conditions":
                updated.setdefault(
                    "active_tension",
                    "The condition is stable only until direct pressure arrives.",
                )
                updated["functions"] = ["stake", "cost"]
                updated["stake"] = {
                    "whose": "Actors moving inside the condition",
                    "what_at_stake": "Room to maneuver under a standing pressure field.",
                }
                updated["cost"] = {
                    "who_paid": "Those closest to the unstable condition",
                    "what_was_paid": "Security and future options.",
                }
            elif substrate == "artifacts":
                updated.setdefault(
                    "artifact_detail",
                    "The carrier itself becomes historically memorable.",
                )
                updated["functions"] = ["texture", "recurrence"]
                updated["texture"] = {
                    "what_it_anchors": "A concrete carrier the listener can remember and revisit.",
                }
                updated["recurrence"] = {
                    "connects_to": [],
                    "meaning_accrued": "The image or object can return later with added meaning.",
                }
            else:
                updated["functions"] = ["contest", "complication"]
                updated["contest"] = {
                    "candidate_readings": [
                        {
                            "reading": "One interpretation emphasizes institutional logic.",
                            "support": "It fits the available structural evidence.",
                            "weakness": "It may underweight immediate tactical calculation.",
                        },
                        {
                            "reading": "Another interpretation emphasizes tactical improvisation.",
                            "support": "It explains the short-term behavior more directly.",
                            "weakness": "It may underweight longer structural constraints.",
                        },
                    ]
                }
                updated["complication"] = {
                    "what_is_compromised": "A clean single-cause explanation.",
                    "why_no_clean_option": "The evidence can plausibly be read in more than one direction.",
                }
            updated["narration_hooks"] = {
                "concrete_detail": "A concrete detail makes the primitive speakable.",
                "host_lens": "One narrow lens helps orient the listener to the pressure here.",
                "carry_forward": "This primitive leaves residue that can matter again later.",
                "quote_anchor": "",
                "plain_gloss": "Plainspoken gloss for oral use.",
                "why_it_matters": "Why this matters to the listener-facing pressure line.",
                "best_use": "Best used as build material unless elevated by later structure.",
                "natural_host_move": "orient",
                "listener_confusion": "What this is doing in the story.",
                "authorial_move": "none",
            }
            tagged_primitives.append(updated)
        return {
            "project_id": payload.get("project_id", "project"),
            "primitives": tagged_primitives,
            "quality_score": 0.6,
            "quality_notes": ["Heuristic primitive function tagging artifact."],
        }

    def _generate_primitive_function_tagging_events(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="events")

    def _generate_primitive_function_tagging_acts(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="acts")

    def _generate_excerpt_extraction(self, payload: PromptPayload) -> dict[str, Any]:
        passages_by_axis = payload.get("passages_by_axis", {})
        passage_ids = self._collect_passage_ids(passages_by_axis) or [uuid4().hex]
        actor_metadata = payload.get("actor_metadata", {}) or {}
        actor_ids = [
            str(actor.get("actor_id", ""))
            for actor in actor_metadata.get("actors", [])
            if actor.get("actor_id")
        ]
        return {
            "project_id": payload.get("project_id", "project"),
            "excerpts": [
                {
                    "excerpt_type": "speech",
                    "title": "Heuristic excerpt",
                    "passage_ids": passage_ids[:2] or ["p0"],
                    "actor_ids": actor_ids[:1],
                    "speaker": "A visible speaker",
                    "audience": "A public audience",
                    "summary": "A line makes the dispute audible.",
                    "verbatim_excerpt": "This changes what can now be said aloud.",
                    "plain_gloss": "In plain terms, the rules just shifted.",
                    "quotability": 0.7,
                }
            ],
            "quality_score": 0.5,
            "quality_notes": ["Heuristic excerpt artifact."],
        }

    def _generate_primitive_function_tagging_actor_portraits(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(
            payload, substrate="actor_portraits"
        )

    def _generate_primitive_function_tagging_mechanisms(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="mechanisms")

    def _generate_primitive_function_tagging_conditions(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="conditions")

    def _generate_primitive_function_tagging_artifacts(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="artifacts")

    def _generate_primitive_function_tagging_readings(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        return self._generate_primitive_function_tagging_batch(payload, substrate="readings")

    def _generate_scene_discovery(self, payload: PromptPayload) -> dict[str, Any]:
        project = payload.get("project", {}) or {}
        if not isinstance(project, dict):
            project = {}
        try:
            mode = PodcastMode(project.get("podcast_mode", PodcastMode.FULL.value))
        except ValueError:
            mode = PodcastMode.FULL
        target_min, _target_max = scene_discovery_candidate_range_for_mode(mode)
        synthesis_map = payload.get("synthesis_map", {}) or {}
        primitives = list(synthesis_map.get("primitives", []) or [])
        passage_list = list(payload.get("passage_list", []) or [])
        candidates: list[dict[str, Any]] = []
        for idx in range(target_min):
            primitive = primitives[min(idx, len(primitives) - 1)] if primitives else {}
            primitive_id = str(primitive.get("id", f"primitive_{idx + 1:03d}"))
            passage = passage_list[min(idx, len(passage_list) - 1)] if passage_list else {}
            passage_id = str(passage.get("passage_id", f"passage_{idx + 1:03d}"))
            actor_ids = list(primitive.get("actors", primitive.get("actor_ids", [])) or [])
            if idx == 0:
                scene_jobs = ["opening"]
            elif idx == target_min - 3:
                scene_jobs = ["turn"]
            elif idx == target_min - 2:
                scene_jobs = ["answer"]
            else:
                scene_jobs = ["build" if idx % 4 == 0 else "opening"]
            normalized_scene_jobs = [
                role for role in scene_jobs if role in {"opening", "build", "turn", "answer"}
            ] or ["opening"]
            candidates.append(
                {
                    "candidate_id": f"candidate_{idx + 1:03d}",
                    "primitive_ids": [primitive_id],
                    "passage_ids": [passage_id],
                    "scene_sketch": f"Candidate scene {idx + 1} stages one load-bearing historical beat.",
                    "scene_jobs": normalized_scene_jobs[:2],
                    "anchor_image": "A concrete room, object, or public act anchors the beat.",
                    "why_sceneable": "The beat has a visible carrier and a clear oral payoff.",
                    "quote_anchor": "A short line can survive on air." if idx % 3 == 0 else "",
                    "actor_ids": [str(actor_id) for actor_id in actor_ids[:2] if actor_id],
                }
            )
        return {"candidates": candidates}

    def _generate_narrative_strategy_skeleton(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        requested_episode_count = payload.get("requested_episode_count")
        recommended_episode_count_min = int(
            payload.get("recommended_episode_count_min", 8)
        )
        recommended_episode_count_max = int(
            payload.get("recommended_episode_count_max", 12)
        )
        if recommended_episode_count_max < recommended_episode_count_min:
            recommended_episode_count_max = recommended_episode_count_min
        project = payload.get("project", {})
        if not isinstance(project, dict):
            project = {}
        try:
            podcast_mode = PodcastMode(
                project.get("podcast_mode", PodcastMode.FULL.value)
            )
        except ValueError:
            podcast_mode = PodcastMode.FULL
        core_target_min = int(
            project.get("episode_spine_core_primitive_target_min", 6)
        )
        support_target_min = int(
            project.get("episode_spine_support_primitive_target_min", 7)
        )
        synthesis_map = payload.get("synthesis_map", {})
        primitives = list(synthesis_map.get("primitives", []) or [])
        primitive_ids = [
            str(item.get("id", ""))
            for item in primitives
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        if not primitive_ids:
            primitive_ids = ["primitive_001", "primitive_002", "primitive_003"]
        while len(primitive_ids) < 96:
            primitive_ids.append(f"primitive_{len(primitive_ids) + 1:03d}")
        pivot_ids = [
            str(item.get("id", ""))
            for item in primitives
            if isinstance(item, dict)
            and "pivot" in list(item.get("functions", []) or [])
            and str(item.get("id", "")).strip()
        ]
        event_or_act_ids = [
            str(item.get("id", ""))
            for item in primitives
            if isinstance(item, dict)
            and str(item.get("substrate", "")) in {"events", "acts"}
            and str(item.get("id", "")).strip()
        ]
        if requested_episode_count is None:
            recommended_episode_count = recommended_episode_count_min
        else:
            recommended_episode_count = int(requested_episode_count)
        episodes = []
        scene_discovery = payload.get("scene_discovery", {}) or {}
        discovered_candidates = list(scene_discovery.get("candidates", []) or [])
        for idx in range(recommended_episode_count):
            listener_problem = (
                "What local turn best explains the series?"
                if idx == 0
                else f"What does episode {idx + 1} newly reveal?"
            )
            core_start = idx * 13
            core_ids = primitive_ids[core_start : core_start + core_target_min]
            if len(core_ids) < core_target_min:
                core_ids = primitive_ids[:core_target_min]
            if pivot_ids:
                preferred_cores = [primitive_id for primitive_id in pivot_ids if primitive_id not in core_ids]
                core_ids = list(dict.fromkeys([*pivot_ids[:2], *core_ids, *preferred_cores]))[:core_target_min]
            if event_or_act_ids and not any(primitive_id in event_or_act_ids for primitive_id in core_ids):
                core_ids = [event_or_act_ids[0], *core_ids]
                core_ids = list(dict.fromkeys(core_ids))[:core_target_min]
            support_candidates = primitive_ids[
                core_start + core_target_min : core_start + core_target_min + 10
            ]
            support_ids = [
                primitive_id
                for primitive_id in support_candidates
                if primitive_id not in core_ids
            ][:support_target_min]
            if len(support_ids) < support_target_min:
                support_ids = [
                    primitive_id
                    for primitive_id in primitive_ids
                    if primitive_id not in core_ids
                ][:support_target_min]
            episodes.append(
                {
                    "episode_number": idx + 1,
                    "title": f"Episode {idx + 1}",
                    "thematic_focus": "Heuristic focus",
                    "arc_summary": f"Episode {idx + 1} follows a single listener-facing pressure line.",
                    "unresolved_questions": [],
                    "actor_arc_directives": [],
                    "authorial_contract": {
                        "analysis_weight": "medium",
                        "priority_moves": ["causal_compression"],
                        "governing_lenses": [
                            "The episode tightens one governing pressure line."
                        ],
                        "must_clarify_terms": [],
                        "must_clarify_institutions": [],
                    },
                    "episode_spine": {
                        "listener_problem": listener_problem,
                        "episode_answer": "Selected primitives carry the episode's answer.",
                        "pressure_line": "The listener should feel one pressure line tightening across the episode.",
                        "core_primitive_ids": core_ids,
                        "support_primitive_roles": {
                            primitive_id: "mechanism" if primitive_id not in event_or_act_ids else "stakes"
                            for primitive_id in support_ids
                            if primitive_id not in core_ids
                        },
                        "recall_primitive_ids": pivot_ids[2:3] if idx > 0 else [],
                    },
                    "negative_scope": {
                        "boundary": "Keep the episode on one pressure line.",
                        "excluded_topics": ["adjacent background not required for this episode"],
                        "tempting_but_out": ["parallel but non-binding subplot"],
                        "omission_logic": "Leave neighboring material out unless it directly advances the current episode's answer pressure.",
                    },
                }
            )
        return {
            "strategy_type": "convergence",
            "justification": "Heuristic: defaulting to convergence strategy.",
            "series_arc": "Books converge on shared themes.",
            "episode_arc_outline": [
                f"Episode {idx + 1}" for idx in range(recommended_episode_count)
            ],
            "recommended_episode_count": recommended_episode_count,
            "episodes": episodes,
        }

    def _generate_narrative_strategy_enrichment(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        strategy_skeleton = payload.get("strategy_skeleton", {}) or {}
        episodes = list(strategy_skeleton.get("episodes", []) or [])
        project = payload.get("project", {})
        if not isinstance(project, dict):
            project = {}
        try:
            podcast_mode = PodcastMode(
                project.get("podcast_mode", PodcastMode.FULL.value)
            )
        except ValueError:
            podcast_mode = PodcastMode.FULL
        episode_scene_candidates = {
            int(item.get("episode_number")): list(item.get("candidates", []) or [])
            for item in list(payload.get("episode_scene_candidates", []) or [])
            if isinstance(item, dict) and isinstance(item.get("episode_number"), int)
        }
        enriched_episodes: list[dict[str, Any]] = []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            episode_number = int(episode.get("episode_number", 0) or 0)
            candidates = episode_scene_candidates.get(episode_number, [])
            promised_beats: list[dict[str, Any]] = []
            if candidates:
                opening_candidate = candidates[0]
                promised_beats.append(
                    {
                        "beat_id": f"episode_{episode_number:02d}_opening",
                        "label": f"Episode {episode_number} opening promise",
                        "intended_job": "opening",
                        "source_candidate_ids": [
                            str(opening_candidate.get("candidate_id", ""))
                        ],
                        "source_primitive_ids": list(
                            opening_candidate.get("primitive_ids", []) or []
                        )[:2],
                        "why_load_bearing": "The episode needs one concrete opening obligation.",
                    }
                )
                answer_candidate = candidates[min(1, len(candidates) - 1)]
                promised_beats.append(
                    {
                        "beat_id": f"episode_{episode_number:02d}_answer",
                        "label": f"Episode {episode_number} answer promise",
                        "intended_job": "answer",
                        "source_candidate_ids": [
                            str(answer_candidate.get("candidate_id", ""))
                        ],
                        "source_primitive_ids": list(
                            answer_candidate.get("primitive_ids", []) or []
                        )[:2],
                        "why_load_bearing": "The episode needs one explicit answer-bearing scene commitment.",
                    }
                )
            enriched_episodes.append(
                {
                    "episode_number": episode_number,
                    "narrator_contract": {
                        "analysis_weight": "medium",
                        "priority_moves": ["causal_compression"],
                        "governing_lenses": [
                            "The episode tightens one governing pressure line."
                        ],
                    },
                    "authorial_contract": {
                        "analysis_weight": "medium",
                        "priority_moves": ["causal_compression"],
                        "governing_lenses": [
                            "The episode tightens one governing pressure line."
                        ],
                        "must_clarify_terms": [],
                        "must_clarify_institutions": [],
                    },
                    "narrative_agenda": {
                        "listener": {
                            "question_moves": [],
                            "memory_thread_moves": [],
                            "carry_forward_memory": [],
                            "episode_takeaway": {
                                "inherited_condition": "A prior structural condition has narrowed the field.",
                                "proximate_contingency": "A choice still on the table tips it this way.",
                            },
                        },
                        "host": {
                            "mystery_moves": [],
                            "assumption_moves": [],
                            "theory_moves": [],
                            "intended_revision_beats": [],
                            "episode_takeaway": {
                                "inherited_condition": "The governing pressure was set up earlier.",
                                "proximate_contingency": "What the host cannot yet settle is how it was used.",
                            },
                        },
                    },
                    "promised_beats": promised_beats,
                }
            )
        return {
            "narrator_profile": {
                "presence_mode": "visible_host",
                "baseline_tone": "plainspoken",
                "spoken_style_contract": "anti_academic_oral",
                "allowed_moves": [
                    "orient",
                    "clarify",
                    "evaluate",
                    "contrast",
                    "callback",
                    "light_aside",
                    "naming_note",
                ],
                "forbidden_moves": [
                    "unsupported_psychology",
                    "cheap_joke",
                    "fake_banter",
                    "teaser_hype",
                ],
                "target_full_phase_scene_coverage_min": 0.60,
                "target_full_phase_scene_coverage_target": 0.75,
                "analysis_mode": "hybrid",
                "analysis_density": "medium",
                "quote_gloss_preference": "allow",
                "clarifier_tolerance": "medium",
                "wit_ceiling": "dry",
                "target_authorial_passages_per_episode": (
                    authorial_passage_target_for_mode(podcast_mode)
                ),
            },
            "episodes": enriched_episodes,
        }

    def _generate_episode_architecture(self, payload: PromptPayload) -> dict[str, Any]:
        episode = payload.get("episode", {})
        episode_spine = episode.get("episode_spine", {})
        narrative_agenda = dict(episode.get("narrative_agenda", {}) or {})
        listener_agenda = dict(narrative_agenda.get("listener", {}) or {})
        host_agenda = dict(narrative_agenda.get("host", {}) or {})
        project = payload.get("project", {})
        if not isinstance(project, dict):
            project = {}
        primitive_ids = list(episode_spine.get("assigned_primitive_ids") or [])
        if not primitive_ids:
            primitive_ids = list(
                episode_spine.get("core_primitive_ids") or ["primitive_001"]
            )
        section_target_min = int(project.get("architecture_section_target_min", 12))
        section_target_max = int(project.get("architecture_section_target_max", 18))
        section_count = (
            min(section_target_max, max(section_target_min, len(primitive_ids)))
            if primitive_ids
            else section_target_min
        )
        closing_minutes = 2.0
        non_closing_count = max(1, section_count - 1)
        target_runtime_minutes = float(project.get("min_episode_minutes", 108.0))
        non_closing_minutes = max(
            1.0, (target_runtime_minutes - closing_minutes) / non_closing_count
        )
        sections = []
        answer_section_idx = max(0, section_count - 3)
        residue_section_idx = max(0, section_count - 2)
        turn_section_idx = min(2, section_count - 2)
        question_moves = list(listener_agenda.get("question_moves", []) or [])
        memory_thread_moves = list(listener_agenda.get("memory_thread_moves", []) or [])
        mystery_moves = list(host_agenda.get("mystery_moves", []) or [])
        assumption_moves = list(host_agenda.get("assumption_moves", []) or [])
        theory_moves = list(host_agenda.get("theory_moves", []) or [])
        for idx in range(section_count):
            section_id = f"section_{idx + 1:02d}"
            local_primitive_ids = [primitive_ids[min(idx, len(primitive_ids) - 1)]]
            is_closing = idx == section_count - 1
            is_turn = idx == turn_section_idx
            if is_closing:
                section_stage = "close"
            elif idx == answer_section_idx:
                section_stage = "answer"
            elif idx == residue_section_idx:
                section_stage = "afterpressure"
            elif idx == 0:
                section_stage = "setup"
            else:
                section_stage = "advance"
            section_question_moves = []
            section_memory_thread_moves = []
            section_host_mystery_moves = []
            section_host_assumption_moves = []
            section_host_theory_moves = []
            if idx == 0:
                section_question_moves.extend(
                    move for move in question_moves if move.get("action") == "open"
                )
                section_memory_thread_moves.extend(
                    move
                    for move in memory_thread_moves
                    if move.get("action") in {"open", "refresh"}
                )
                section_host_mystery_moves.extend(
                    move for move in mystery_moves if move.get("action") == "open"
                )
                section_host_assumption_moves.extend(
                    move
                    for move in assumption_moves
                    if move.get("action") == "introduce"
                )
                section_host_theory_moves.extend(
                    move for move in theory_moves if move.get("action") == "propose"
                )
            if idx == turn_section_idx:
                section_question_moves.extend(
                    move
                    for move in question_moves
                    if move.get("action") in {"advance", "reframe"}
                )
                section_host_mystery_moves.extend(
                    move
                    for move in mystery_moves
                    if move.get("action") in {"advance", "reframe"}
                )
                section_host_assumption_moves.extend(
                    move
                    for move in assumption_moves
                    if move.get("action") == "weaken"
                )
            if idx == answer_section_idx:
                section_question_moves.extend(
                    move for move in question_moves if move.get("action") == "resolve"
                )
                section_host_mystery_moves.extend(
                    move for move in mystery_moves if move.get("action") == "resolve"
                )
                section_host_assumption_moves.extend(
                    move
                    for move in assumption_moves
                    if move.get("action") == "revise"
                )
                section_host_theory_moves.extend(
                    move
                    for move in theory_moves
                    if move.get("action") in {"strengthen", "replace"}
                )
            if idx == residue_section_idx or is_closing:
                section_memory_thread_moves.extend(
                    move
                    for move in memory_thread_moves
                    if move.get("action") in {"payoff", "retire"}
                )
                section_host_mystery_moves.extend(
                    move
                    for move in mystery_moves
                    if move.get("action") == "reframe"
                )
                section_host_assumption_moves.extend(
                    move
                    for move in assumption_moves
                    if move.get("action") == "drop"
                )
                section_host_theory_moves.extend(
                    move
                    for move in theory_moves
                    if move.get("action") == "retire"
                )
            sections.append(
                {
                    "section_id": section_id,
                    "purpose": (
                        "opening"
                        if idx == 0
                        else "closing"
                        if is_closing
                        else "turn"
                        if is_turn
                        else "setup"
                    ),
                    "approx_runtime_minutes": closing_minutes
                    if is_closing
                    else non_closing_minutes,
                    "primitive_ids": local_primitive_ids,
                    "open_mode": "scene_anchor",
                    "section_anchor": "A concrete section anchor keeps the listener oriented.",
                    "must_stage_beats": [
                        f"Stage the visible move that defines section {idx + 1}.",
                        f"Show the immediate consequence that makes section {idx + 1} matter.",
                    ],
                    "listener_tension": f"What pressure does section {idx + 1} put on the listener?",
                    "section_turn": f"Section {idx + 1} changes the listener's understanding.",
                    "transition_logic": "Move by clarifying the next structural pressure.",
                    "depends_on_section_ids": [f"section_{idx:02d}"] if idx > 0 else [],
                    "sets_up_section_ids": [f"section_{idx + 2:02d}"]
                    if idx < section_count - 1
                    else [],
                    "recurrence_role": "plant"
                    if idx == 0
                    else "payoff"
                    if idx == section_count - 1
                    else "deepen",
                    "priority_core_passage_ids": [],
                    "section_progression": {
                        "stage": section_stage,
                        "becomes_obvious": (
                            f"By the end of section {idx + 1} the listener can see "
                            "the next move in the episode's answer."
                        ),
                        "answer_contribution": (
                            f"Section {idx + 1} advances the episode answer at the "
                            f"{section_stage} stage."
                        ),
                        "theme_link": (
                            "Ties this section's pressure to the episode's overall theme."
                        ),
                        "what_remains_live": (
                            "The episode exits without reopening the answer."
                            if is_closing
                            else f"Section {idx + 1} hands its live pressure forward."
                        ),
                        "state_effects": {
                            "question_moves": section_question_moves,
                            "memory_thread_moves": section_memory_thread_moves,
                            "host_mystery_moves": section_host_mystery_moves,
                            "host_assumption_moves": section_host_assumption_moves,
                            "host_theory_moves": section_host_theory_moves,
                        },
                    },
                    "analysis_goal": (
                        "Cash out the main pressure line in plain terms."
                        if idx in (1, section_count - 1)
                        else ""
                    ),
                    "key_terms": [],
                    "authorial_passages": (
                        [
                            {
                                "passage_id": f"ap_{idx + 1:02d}",
                                "mode": "causal_compression",
                                "placement": "mid"
                                if idx != section_count - 1
                                else "close",
                                "claim": "The narrator briefly compresses what the section proves.",
                                "source_primitive_ids": local_primitive_ids,
                                "source_passage_ids": [],
                                "quote_anchor": "",
                                "gloss_seed": "Say plainly what changes here.",
                                "must_name_terms": [],
                                "budget_sentences": 3,
                            }
                        ]
                        if idx in (1, section_count - 1)
                        else []
                    ),
                }
            )
        promised_beat_decisions = []
        for beat in list(episode.get("promised_beats", []) or []):
            beat_id = str(beat.get("beat_id", "") or "")
            intended_job = str(beat.get("intended_job", "") or "")
            if intended_job == "answer":
                section_id = sections[answer_section_idx]["section_id"]
            else:
                section_id = sections[min(1, len(sections) - 1)]["section_id"]
            promised_beat_decisions.append(
                {
                    "beat_id": beat_id,
                    "decision": "stage",
                    "section_id": section_id,
                    "reason": "",
                }
            )
        return {
            "episode_number": int(episode.get("episode_number", 1)),
            "major_turn_section_id": sections[turn_section_idx]["section_id"],
            "allowed_recurring_primitive_ids": primitive_ids[:2],
            "forbidden_redundancies": [],
            "promised_beat_decisions": promised_beat_decisions,
            "sections": sections,
            "architecture_notes": ["Heuristic architecture output."],
        }

    def _generate_episode_planning(self, payload: PromptPayload) -> dict[str, Any]:
        architecture = payload.get("architecture", {})
        strategy_episode = payload.get("strategy_episode", {})
        project = payload.get("project", {})
        if not isinstance(project, dict):
            project = {}
        episode_number = int(architecture.get("episode_number", 1))
        available_passages = payload.get("available_passages", [])
        passage_ids = [
            str(passage.get("passage_id", uuid4().hex))
            for passage in available_passages[:3]
        ]
        sections = architecture.get("sections") or [{"section_id": "section_01"}]
        scene_cards = []
        default_duration = 180
        budget = payload.get("scene_job_budget") or scene_job_budget_for_mode(
            project.get("podcast_mode", PodcastMode.FULL.value)
        )
        opening_count = int(budget.get("opening_min", 2))
        build_count = int(budget.get("build_min", 11))
        turn_count = int(budget.get("turn_min", 2))
        job_sequence = (
            [("opening", "context_setup")] * opening_count
            + [("build", "action")] * build_count
            + [("turn", "shock")] * turn_count
            + [("answer", "implication")]
            + [("close", "implication")]
        )
        scene_count = len(job_sequence)

        def _section_id_for_stage(stage: str) -> str | None:
            for section in sections:
                progression = section.get("section_progression") or {}
                if progression.get("stage") == stage:
                    return str(section.get("section_id"))
            return None

        answer_section_id = str(
            _section_id_for_stage("answer")
            or sections[max(0, len(sections) - 3)].get("section_id", "section_01")
        )
        close_section_id = str(sections[-1].get("section_id", "section_01"))
        answer_scene_card_id = None
        for idx in range(scene_count):
            scene_job, scene_role = job_sequence[idx]
            source_section = sections[min(idx, len(sections) - 1)]
            section_id = str(source_section.get("section_id", "section_01"))
            if scene_job == "answer":
                section_id = answer_section_id
            elif scene_job == "close":
                section_id = close_section_id
            section_primitive_ids = list(source_section.get("primitive_ids", []) or [])
            is_closing = scene_job == "close"
            scene_id = f"scene_{idx + 1:02d}"
            if scene_job == "answer":
                answer_scene_card_id = scene_id
            scene_cards.append(
                {
                    "scene_id": scene_id,
                    "section_id": section_id,
                    "title": f"Heuristic scene {idx + 1}",
                    "scene_role": scene_role,
                    "scene_job": scene_job,
                    "beat_change": "The listener enters the episode's opening pressure."
                    if scene_job == "opening"
                    else "The next beat changes the situation in a concrete way."
                    if scene_job in {"build", "turn"}
                    else "The answer becomes clear without abstraction."
                    if scene_job == "answer"
                    else "The close exits the episode without reopening the answer.",
                    "must_land_facts": {
                        "required": [
                            "A concrete change becomes visible in the beat."
                        ],
                        "strongly_preferred": [],
                        "if_room": [],
                    },
                    "entry_image": "A concrete opening image."
                    if scene_job == "opening"
                    else "A grounded detail keeps the scene moving.",
                    "observable_detail": "A visible consequence lands in the scene.",
                    "timeframe": None,
                    "location": None,
                    "actors": [],
                    "primitive_ids": section_primitive_ids[:2],
                    "passage_ids": passage_ids,
                    "authorial_passage_ids": [],
                    "word_count_priority": "default",
                    "estimated_duration_seconds": 120
                    if is_closing
                    else default_duration,
                    "host_moves": {
                        "open": (
                            [
                                {
                                    "move_type": "orient",
                                    "target": "pressure first",
                                    "surface_mode": "woven",
                                    "address_mode": "implicit",
                                }
                            ]
                            if scene_job in {"opening", "build"}
                            else []
                        ),
                        "pivot": (
                            [
                                {
                                    "move_type": "clarify",
                                    "target": "what changes",
                                    "surface_mode": "woven",
                                    "address_mode": "implicit",
                                }
                            ]
                            if scene_job in {"turn", "answer"}
                            else []
                        ),
                        "close": (
                            [
                                {
                                    "move_type": "callback",
                                    "target": "clean exit",
                                    "surface_mode": "mixed",
                                    "address_mode": "implicit",
                                }
                            ]
                            if scene_job == "close"
                            else []
                        ),
                    },
                }
            )
        return {
            "episode_number": episode_number,
            "framing": {
                "opening_image": "A listener-facing opening image.",
                "threat_or_unresolved_action": "A threat remains active as the episode starts.",
                "opening_question": (
                    (strategy_episode.get("episode_spine", {}) or {}).get(
                        "listener_problem", "What changes here?"
                    )
                    or "What changes here?"
                ),
                "handoff_scene_card_id": "scene_01",
                "recap": None,
                "preview": None,
            },
            "scene_cards": scene_cards,
            "answer_scene_card_id": answer_scene_card_id or "scene_01",
            "dropped_support_primitive_reasons": {},
        }

    def _generate_episode_writing(self, payload: PromptPayload) -> dict[str, Any]:
        plan = payload.get("plan", {})
        scene_cards = plan.get("scene_cards", [])
        prose_sections: list[dict[str, Any]] = []
        current_section_id: str | None = None
        current_scene_ids: list[str] = []
        for scene in scene_cards:
            if not isinstance(scene, dict):
                continue
            section_id = str(scene.get("section_id", "") or "")
            scene_id = str(scene.get("scene_id", "") or "")
            if not section_id or not scene_id:
                continue
            if current_section_id is not None and section_id != current_section_id:
                prose_sections.append(
                    {
                        "section_id": current_section_id,
                        "scene_card_ids": list(current_scene_ids),
                        "movement_goal": "discover",
                        "text": "Heuristic narration content.",
                        "citations": [],
                        "source_book_ids": [],
                    }
                )
                current_scene_ids = []
            current_section_id = section_id
            current_scene_ids.append(scene_id)
        if current_section_id is not None:
            prose_sections.append(
                {
                    "section_id": current_section_id,
                    "scene_card_ids": list(current_scene_ids),
                    "movement_goal": "discover",
                    "text": "Heuristic narration content.",
                    "citations": [],
                    "source_book_ids": [],
                }
            )
        if not prose_sections:
            prose_sections = [
                {
                    "section_id": "section_01",
                    "scene_card_ids": ["scene_01"],
                    "movement_goal": "discover",
                    "text": "Heuristic narration content.",
                    "citations": [],
                    "source_book_ids": [],
                }
            ]
        return {
            "prose_sections": prose_sections,
        }

    def _generate_quality_judge(self, payload: PromptPayload) -> dict[str, Any]:
        """Fabricate a passing EpisodeQualityScore for local heuristic runs.

        Returns identical 92s across all six criteria, no remediation
        targets — just enough shape for downstream tests/runs to proceed.
        """
        episode_number = int(payload.get("episode_number", 1) or 1)
        prose_sections = list(payload.get("prose_sections", []) or [])
        criteria = [
            "narrative_quality",
            "listener_engagement",
            "podcast_fidelity",
            "prose_polish",
            "frame_discipline",
            "excerpt_staging",
        ]
        # Stagger so prose_polish is the unique minimum and the validator
        # accepts weakest_criterion="prose_polish".
        score_overrides = {"prose_polish": 90}
        criterion_scores = [
            {
                "criterion": c,
                "score": score_overrides.get(c, 92),
                "rationale": "Heuristic uniform score for local runs.",
                "weaknesses": [],
            }
            for c in criteria
        ]
        overall = round(sum(cs["score"] for cs in criterion_scores) / 6)
        section_scores = []
        for section in prose_sections:
            section_id = str(section.get("section_id", "")).strip() or "section_01"
            section_scores.append({
                "section_id": section_id,
                "overall_score": overall,
                "criterion_scores": criterion_scores,
                "weakest_criterion": "prose_polish",
                "remediation_hints": [],
                "tic_families_detected": [],
                "second_ending_detected": False,
                "thesis_at_open_or_close_detected": False,
            })
        return {
            "episode_number": episode_number,
            "overall_score": 92,
            "criterion_scores": criterion_scores,
            "section_scores": section_scores,
            "weakest_sections": [],
            "judge_model_name": "heuristic",
            "judge_temperature": 0.0,
        }

    def _generate_spoken_delivery(self, payload: PromptPayload) -> dict[str, Any]:
        """Emit the new segment-aware SpokenDeliveryResponse shape (Change 4)."""
        script = payload.get("script", {})
        prose_sections = list(script.get("prose_sections", []) or [])
        return {
            "sections": [
                {
                    "section_id": str(section.get("section_id", f"section_{idx + 1}")),
                    "segments": [
                        {
                            "segment_id": (
                                f"{section.get('section_id', f'section_{idx + 1}')}_seg1"
                            ),
                            "text": str(section.get("text", "")).strip()
                            or "Spoken delivery text.",
                            "speaker_role": "primary",
                            "tonal_register": "neutral",
                        }
                    ],
                    "tonal_register": "neutral",
                    "speech_hints": {
                        "style": "neutral",
                        "intensity": "none",
                        "pause_before_ms": 300,
                        "pause_after_ms": 300,
                        "pace": "normal",
                        "pronunciation_hints": [],
                        "emphasis_targets": [],
                        "render_strategy": "plain",
                    },
                }
                for idx, section in enumerate(prose_sections)
            ],
        }

    def _generate_style_audit(self, payload: PromptPayload) -> dict[str, Any]:
        sections = payload.get("sections", [])
        return {
            "episode_number": payload.get("episode_number", 1),
            "sections": [
                {
                    "section_id": str(section.get("section_id", f"section_{idx + 1}")),
                    "edited_text": str(section.get("text", "")).strip()
                    or "Audited text.",
                    "edit_notes": [],
                }
                for idx, section in enumerate(sections)
            ],
            "episode_warnings": [],
        }

    def _generate_narrative_state_reconciliation(
        self, payload: PromptPayload
    ) -> dict[str, Any]:
        episode_number = int(payload.get("episode_number", 1) or 1)
        project_id = str(payload.get("project_id", "") or "project")
        state_pre = dict(payload.get("narrative_state_pre", {}) or {})
        listener_pre = dict(state_pre.get("listener", {}) or {})
        host_pre = dict(state_pre.get("host", {}) or {})
        strategy_episode = dict(payload.get("strategy_episode", {}) or {})
        narrative_agenda = dict(strategy_episode.get("narrative_agenda", {}) or {})
        listener_agenda = dict(narrative_agenda.get("listener", {}) or {})
        host_agenda = dict(narrative_agenda.get("host", {}) or {})
        architecture = dict(payload.get("architecture", {}) or {})

        known_explanation_item_ids = list(
            dict.fromkeys(
                list(listener_pre.get("known_explanation_item_ids", []) or [])
                + list(listener_agenda.get("introduce_explanation_item_ids", []) or [])
            )
        )
        known_actor_ids = list(
            dict.fromkeys(
                list(listener_pre.get("known_actor_ids", []) or [])
                + list(listener_agenda.get("introduce_actor_ids", []) or [])
            )
        )

        questions_by_id = {
            str(item.get("question_id", "")): dict(item)
            for item in list(listener_pre.get("questions", []) or [])
            if str(item.get("question_id", "")).strip()
        }
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("question_moves", []) or []):
                question_id = str(move.get("question_id", "") or "").strip()
                action = str(move.get("action", "") or "").strip()
                text = str(move.get("text", "") or "").strip()
                if not question_id:
                    continue
                entry = questions_by_id.get(
                    question_id,
                    {"question_id": question_id, "text": text or question_id, "status": "open"},
                )
                if text:
                    entry["text"] = text
                entry["status"] = QUESTION_STATUS_BY_ACTION.get(
                    action, entry.get("status", "open")
                )
                questions_by_id[question_id] = entry

        memory_threads_by_id = {
            str(item.get("thread_id", "")): dict(item)
            for item in list(listener_pre.get("memory_threads", []) or [])
            if str(item.get("thread_id", "")).strip()
        }
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("memory_thread_moves", []) or []):
                thread_id = str(move.get("thread_id", "") or "").strip()
                action = str(move.get("action", "") or "").strip()
                label = str(move.get("label", "") or "").strip()
                if not thread_id:
                    continue
                entry = memory_threads_by_id.get(
                    thread_id,
                    {
                        "thread_id": thread_id,
                        "label": label or thread_id,
                        "status": "open",
                        "source_promised_beat_id": move.get("source_promised_beat_id"),
                    },
                )
                if label:
                    entry["label"] = label
                entry["status"] = MEMORY_THREAD_STATUS_BY_ACTION.get(
                    action, entry.get("status", "open")
                )
                memory_threads_by_id[thread_id] = entry

        mysteries_by_id = {
            str(item.get("mystery_id", "")): dict(item)
            for item in list(host_pre.get("mysteries", []) or [])
            if str(item.get("mystery_id", "")).strip()
        }
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("host_mystery_moves", []) or []):
                mystery_id = str(move.get("mystery_id", "") or "").strip()
                action = str(move.get("action", "") or "").strip()
                text = str(move.get("text", "") or "").strip()
                if not mystery_id:
                    continue
                entry = mysteries_by_id.get(
                    mystery_id,
                    {
                        "mystery_id": mystery_id,
                        "text": text or mystery_id,
                        "status": "open",
                    },
                )
                if text:
                    entry["text"] = text
                entry["status"] = QUESTION_STATUS_BY_ACTION.get(
                    action, entry.get("status", "open")
                )
                mysteries_by_id[mystery_id] = entry

        assumptions_by_id = {
            str(item.get("assumption_id", "")): dict(item)
            for item in list(host_pre.get("assumptions", []) or [])
            if str(item.get("assumption_id", "")).strip()
        }
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("host_assumption_moves", []) or []):
                assumption_id = str(move.get("assumption_id", "") or "").strip()
                action = str(move.get("action", "") or "").strip()
                statement = str(move.get("statement", "") or "").strip()
                revised_statement = str(move.get("revised_statement", "") or "").strip()
                if not assumption_id:
                    continue
                entry = assumptions_by_id.get(
                    assumption_id,
                    {
                        "assumption_id": assumption_id,
                        "statement": statement or assumption_id,
                        "status": "active",
                    },
                )
                if revised_statement:
                    entry["statement"] = revised_statement
                elif statement:
                    entry["statement"] = statement
                entry["status"] = ASSUMPTION_STATUS_BY_ACTION.get(
                    action, entry.get("status", "active")
                )
                assumptions_by_id[assumption_id] = entry

        theories_by_id = {
            str(item.get("theory_id", "")): dict(item)
            for item in list(host_pre.get("working_theories", []) or [])
            if str(item.get("theory_id", "")).strip()
        }
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("host_theory_moves", []) or []):
                theory_id = str(move.get("theory_id", "") or "").strip()
                action = str(move.get("action", "") or "").strip()
                statement = str(move.get("statement", "") or "").strip()
                if not theory_id:
                    continue
                entry = theories_by_id.get(
                    theory_id,
                    {
                        "theory_id": theory_id,
                        "statement": statement or theory_id,
                        "status": "proposed",
                    },
                )
                if statement:
                    entry["statement"] = statement
                entry["status"] = THEORY_STATUS_BY_ACTION.get(
                    action, entry.get("status", "proposed")
                )
                theories_by_id[theory_id] = entry

        recent_revisions = list(host_pre.get("recent_revisions", []) or [])
        for section in list(architecture.get("sections", []) or []):
            for move in list(section.get("host_mystery_moves", []) or []):
                action = str(move.get("action", "") or "").strip()
                text = str(move.get("text", "") or "").strip()
                mystery_id = str(move.get("mystery_id", "") or "").strip()
                if action in {"advance", "resolve", "reframe"} and (text or mystery_id):
                    recent_revisions.append(text or mystery_id)
            for move in list(section.get("host_assumption_moves", []) or []):
                action = str(move.get("action", "") or "").strip()
                statement = str(move.get("statement", "") or "").strip()
                revised_statement = str(move.get("revised_statement", "") or "").strip()
                assumption_id = str(move.get("assumption_id", "") or "").strip()
                if action in {"weaken", "revise", "drop"}:
                    recent_revisions.append(
                        revised_statement or statement or assumption_id
                    )
            for move in list(section.get("host_theory_moves", []) or []):
                action = str(move.get("action", "") or "").strip()
                statement = str(move.get("statement", "") or "").strip()
                theory_id = str(move.get("theory_id", "") or "").strip()
                if action in {"strengthen", "replace", "retire"}:
                    recent_revisions.append(statement or theory_id)
        recent_revisions = recent_revisions[-4:]

        listener_takeaway = listener_agenda.get("episode_takeaway") or None
        host_takeaway = host_agenda.get("episode_takeaway") or None
        unresolved_mysteries = [
            item
            for item in mysteries_by_id.values()
            if item.get("status") in {"open", "advanced"}
        ]
        destabilized_assumptions = any(
            item.get("status") in {"weakened", "revised", "dropped"}
            for item in assumptions_by_id.values()
        )
        stabilizing_theories = any(
            item.get("status") in {"strengthened", "replaced"}
            for item in theories_by_id.values()
        )
        if unresolved_mysteries or (destabilized_assumptions and not stabilizing_theories):
            host_confidence_posture = "tentative"
        elif not unresolved_mysteries and stabilizing_theories:
            host_confidence_posture = "grounded"
        else:
            host_confidence_posture = "mixed"

        state_post = {
            "project_id": project_id,
            "next_episode_number": episode_number + 1,
            "listener": {
                "known_explanation_item_ids": known_explanation_item_ids,
                "known_actor_ids": known_actor_ids,
                "questions": list(questions_by_id.values()),
                "memory_threads": list(memory_threads_by_id.values()),
                "carry_forward_memory": list(
                    listener_agenda.get("carry_forward_memory", []) or []
                ),
                "last_episode_takeaway": listener_takeaway,
            },
            "host": {
                "mysteries": list(mysteries_by_id.values()),
                "assumptions": list(assumptions_by_id.values()),
                "working_theories": list(theories_by_id.values()),
                "recent_revisions": recent_revisions,
                "confidence_posture": host_confidence_posture,
                "last_episode_takeaway": host_takeaway,
            },
        }

        return {
            "episode_number": episode_number,
            "state_post": state_post,
            "delta": {
                "introduced_explanation_item_ids": list(
                    listener_agenda.get("introduce_explanation_item_ids", []) or []
                ),
                "introduced_actor_ids": list(
                    listener_agenda.get("introduce_actor_ids", []) or []
                ),
                "listener_question_updates": list(questions_by_id.values()),
                "listener_memory_thread_updates": list(memory_threads_by_id.values()),
                "host_mystery_updates": list(mysteries_by_id.values()),
                "host_assumption_updates": list(assumptions_by_id.values()),
                "host_theory_updates": list(theories_by_id.values()),
                "listener_carry_forward_memory": list(
                    listener_agenda.get("carry_forward_memory", []) or []
                ),
                "listener_episode_takeaway": listener_takeaway,
                "host_recent_revisions": recent_revisions,
                "host_episode_takeaway": host_takeaway,
                "host_confidence_posture": host_confidence_posture,
            },
            "warnings": [],
            "rationale": "Heuristic reconciliation from agenda and architecture state moves.",
        }
