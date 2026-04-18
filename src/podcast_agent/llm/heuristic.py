"""Heuristic LLM implementation used for local development and tests."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient, PromptPayload, prompt_log_metadata
from podcast_agent.schemas.models import SYNTHESIS_PRIMITIVE_FAMILIES


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

    def _generate_default(self, schema_name: str, payload: PromptPayload) -> dict[str, Any]:
        raise ValueError(f"No heuristic generator for schema '{schema_name}'.")

    def _generate_chapter_summary(self, payload: PromptPayload) -> dict[str, Any]:
        chapter_title = str(payload.get("chapter_title", "")).strip() or "This chapter"
        return {
            "analysis": {
                "themes_touched": [chapter_title],
                "major_actors": [],
                "key_places": [],
                "key_institutions": [],
                "timeframe": "",
                "key_events_or_arguments": [
                    f"{chapter_title} contributes material relevant to the project theme."
                ],
                "major_tensions": [],
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
        actors = self._generate_actor_metadata_actors(books)
        actor_ids = [actor["actor_id"] for actor in actors]
        axes = []
        for idx in range(1, 11):
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

    def _generate_actor_metadata_actors(self, books: list[dict[str, Any]]) -> list[dict[str, Any]]:
        actors: list[dict[str, Any]] = []
        seen: set[str] = set()
        for book in books:
            book_id = str(book.get("book_id", ""))
            for chapter in book.get("chapters", []) or []:
                analysis = chapter.get("analysis") or {}
                candidates = [
                    *(analysis.get("major_actors", []) or []),
                    *(analysis.get("key_institutions", []) or []),
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
                            "actor_type": "institution"
                            if name in (analysis.get("key_institutions", []) or [])
                            else "person",
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
                            "narrative_importance_score": max(0.1, 1.0 - (len(actors) * 0.03)),
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
                    "book_ids": book_ids if (book_ids := [str(book.get("book_id", "")) for book in books if book.get("book_id")]) else [],
                    "chapter_refs": [],
                    "narrative_functions": ["other"],
                    "goals_or_motivational_pressures": [],
                    "constraints": [],
                    "stakes": [],
                    "transformations": [],
                    "uncertainty_notes": "No chapter actor candidates were available.",
                    "evidence_confidence": "low",
                    "narrative_importance_score": 0.1,
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

    def _build_primitive(
        self,
        *,
        primitive_id: str,
        title: str,
        summary: str,
        axis_ids: list[str],
        passage_ids: list[str],
        actor_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        actor_ids = actor_ids or []
        return {
            "id": primitive_id,
            "title": title,
            "summary": summary,
            "axis_ids": axis_ids,
            "core_passage_ids": passage_ids[:1],
            "support_passage_ids": passage_ids[1:3],
            "timeframe": None,
            "geography": None,
            "primary_actor_ids": actor_ids[:1],
            "affected_actor_ids": [],
            "actor_ids": actor_ids[:1],
            "actor_tags": [],
            "institution_tags": [],
            "unresolved_actor_tags": [],
            "narrative_importance_score": 0.5,
        }

    def _generate_synthesis_primitives(self, payload: PromptPayload) -> dict[str, Any]:
        passages_by_axis = payload.get("passages_by_axis", {})
        axis_ids = list(passages_by_axis.keys()) or ["axis_01"]
        passage_ids = self._collect_passage_ids(passages_by_axis) or [uuid4().hex, uuid4().hex]
        actor_metadata = payload.get("actor_metadata", {}) or {}
        actor_ids = [
            str(actor.get("actor_id", ""))
            for actor in actor_metadata.get("actors", [])
            if actor.get("actor_id")
        ]
        family_templates: dict[str, tuple[str, str, str]] = {
            "turning_points": (
                "tp_001",
                "Heuristic turning point",
                "A threshold crossing changes what becomes possible next.",
            ),
            "scene_worthy_consequences": (
                "sc_001",
                "Heuristic consequence",
                "A visible consequence follows from the turn.",
            ),
            "causal_mechanisms": (
                "cm_001",
                "Heuristic mechanism",
                "A process explains how local change propagates.",
            ),
            "live_questions": (
                "lq_001",
                "Heuristic live question",
                "The evidence supports more than one plausible reading.",
            ),
            "misperceptions": (
                "mp_001",
                "Heuristic misperception",
                "Actors misread the situation in ways that shape later choices and outcomes.",
            ),
            "reversals": (
                "rv_001",
                "Heuristic reversal",
                "A development flips expectations or stated intent.",
            ),
            "motivations_dilemmas": (
                "md_001",
                "Heuristic motivation dilemma",
                "Actors face pressure between competing goals and constraints.",
            ),
            "perspective_shifts": (
                "ps_001",
                "Heuristic perspective shift",
                "Interpretation changes when viewed through a different actor lens.",
            ),
            "moral_ambiguities": (
                "ma_001",
                "Heuristic moral ambiguity",
                "The evidence supports conflicting judgments about responsibility.",
            ),
            "personal_stakes": (
                "st_001",
                "Heuristic personal stake",
                "Individual risk and cost become decisive for later outcomes.",
            ),
            "trauma_legacies": (
                "tl_001",
                "Heuristic trauma legacy",
                "Past violence continues shaping present choices and fears.",
            ),
        }
        primitives_by_family: dict[str, list[dict[str, Any]]] = {}
        for family in SYNTHESIS_PRIMITIVE_FAMILIES:
            primitive_id, title, summary = family_templates[family]
            primitive = self._build_primitive(
                primitive_id=primitive_id,
                title=title,
                summary=summary,
                axis_ids=axis_ids[:1],
                passage_ids=passage_ids[:3],
                actor_ids=actor_ids[:1],
            )
            if family == "live_questions":
                primitive["candidate_readings"] = [
                    {
                        "label": "reading_a",
                        "summary": "A cautious interpretation of the evidence.",
                        "support_passage_ids": passage_ids[:1],
                    },
                    {
                        "label": "reading_b",
                        "summary": "A competing interpretation that remains plausible.",
                        "support_passage_ids": passage_ids[1:2] or passage_ids[:1],
                    },
                ]
            primitives_by_family[family] = [primitive]
        return {
            "project_id": payload.get("project_id", "project"),
            "primitives_by_family": primitives_by_family,
            "quality_score": 0.5,
            "quality_notes": ["Heuristic primitives artifact."],
        }

    def _generate_synthesis_consolidation(self, payload: PromptPayload) -> dict[str, Any]:
        primitives = payload.get("primitives", {})
        primitives_by_family = primitives.get("primitives_by_family", {})
        actor_metadata = payload.get("actor_metadata", {}) or {}
        actor_ids = [
            str(actor.get("actor_id", ""))
            for actor in actor_metadata.get("actors", [])
            if actor.get("actor_id")
        ]
        member_ids: list[str] = []
        primitive_ids_by_family: dict[str, list[str]] = {}
        for family in SYNTHESIS_PRIMITIVE_FAMILIES:
            family_items = primitives_by_family.get(family, [])
            family_ids = [str(item.get("id", uuid4().hex)) for item in family_items]
            primitive_ids_by_family[family] = family_ids
            for item in family_items[:1]:
                member_ids.append(str(item.get("id", uuid4().hex)))
        primary_member_id = member_ids[0] if member_ids else "tp_001"
        return {
            "project_id": payload.get("project_id", "project"),
            "episode_candidate_clusters": [
                {
                    "cluster_id": "cluster_001",
                    "title": "Heuristic cluster",
                    "summary": "A compact local causal chain.",
                    "primary_member_id": primary_member_id,
                    "member_ids": member_ids or [primary_member_id],
                    "actor_ids": actor_ids[:1],
                    "primary_actor_id": actor_ids[0] if actor_ids else None,
                    "actor_tension": "A heuristic actor pressure shapes the local chain." if actor_ids else "",
                    "narrative_importance_score": 0.5,
                    "coverage_policy": "anchor",
                    "local_question": "What changes the stakes locally?",
                    "local_payoff_shape": "reveal",
                }
            ],
            "primitive_ids_by_family": primitive_ids_by_family,
            "quality_score": float(primitives.get("quality_score", 0.5)),
            "quality_notes": ["Heuristic consolidated synthesis map."],
        }

    def _generate_narrative_strategy(self, payload: PromptPayload) -> dict[str, Any]:
        requested_episode_count = payload.get("requested_episode_count")
        synthesis_map = payload.get("synthesis_map", {})
        clusters = synthesis_map.get("episode_candidate_clusters", [])
        if not clusters:
            clusters = [{"cluster_id": "cluster_001", "title": "Heuristic cluster"}]
        if requested_episode_count is None:
            recommended_episode_count = max(6, min(10, len(clusters)))
        else:
            recommended_episode_count = max(6, min(10, int(requested_episode_count)))
        while len(clusters) < recommended_episode_count:
            idx = len(clusters) + 1
            clusters.append(
                {
                    "cluster_id": f"cluster_{idx:03d}",
                    "title": f"Heuristic cluster {idx}",
                }
            )
        episodes = []
        for idx in range(recommended_episode_count):
            cluster = clusters[idx % len(clusters)]
            cluster_actor_ids = list(cluster.get("actor_ids", []))
            actor_arc_directives = [
                {
                    "actor_id": actor_id,
                    "arc_refs": [
                        {
                            "ref_id": f"{actor_id}_role_1",
                            "arc_type": "role",
                            "label": "episode role",
                            "premise": "This actor provides the heuristic character spine for the episode.",
                            "pressure": "Episode events constrain the actor's available choices.",
                            "movement": "The actor's position becomes clearer as the cluster path develops.",
                            "payoff": "The episode lands the actor's function as part of the local turn.",
                        },
                        {
                            "ref_id": f"{actor_id}_tracking_1",
                            "arc_type": "tracking",
                            "label": "listener tracking",
                            "premise": "Track how the actor's position becomes clearer across the episode.",
                            "pressure": "New evidence or consequences should sharpen what is at stake.",
                            "movement": "Each relevant scene should add pressure rather than repeat the same role.",
                            "payoff": "The listener should understand why this actor mattered to the episode.",
                        },
                        {
                            "ref_id": f"{actor_id}_tension_1",
                            "arc_type": "tension",
                            "label": "tension line",
                            "premise": "External pressure constrains the actor's available choices.",
                            "pressure": "The cluster path narrows the actor's options.",
                            "movement": "Pressure should become more concrete across scenes.",
                            "payoff": "The tension should clarify the episode's local consequence.",
                        },
                        {
                            "ref_id": f"{actor_id}_progression_1",
                            "arc_type": "turn",
                            "label": "arc progression",
                            "premise": "The actor's position changes as the cluster path develops.",
                            "pressure": "A decision or consequence should alter what the actor can do.",
                            "movement": "Move from setup toward consequence.",
                            "payoff": "The actor's changed position should make the episode turn legible.",
                        },
                        {
                            "ref_id": f"{actor_id}_scene_job_1",
                            "arc_type": "payoff",
                            "label": "scene job",
                            "premise": "Use this actor where pressure turns into consequence.",
                            "pressure": "The scene should not merely name the actor.",
                            "movement": "Bind actor presence to visible consequence.",
                            "payoff": "The payoff should connect actor pressure to episode meaning.",
                        },
                        {
                            "ref_id": f"{actor_id}_guardrail_1",
                            "arc_type": "guardrail",
                            "label": "repetition guardrail",
                            "premise": "Do not restate the same actor function unless the scene changes or pays it off.",
                            "pressure": "Repeated appearances can flatten the actor into a label.",
                            "movement": "Vary actor use by scene.",
                            "payoff": "The episode should preserve actor continuity without repetition.",
                        },
                    ],
                }
                for actor_id in cluster_actor_ids[:1]
            ]
            episodes.append(
                {
                    "episode_number": idx + 1,
                    "title": f"Episode {idx + 1}",
                    "driving_question": (
                        "What local turn best explains the series?"
                        if idx == 0
                        else f"What does episode {idx + 1} newly reveal?"
                    ),
                    "thematic_focus": cluster.get("title", "Heuristic focus"),
                    "arc_summary": f"Episode {idx + 1} follows a discovery-ordered cluster path.",
                    "unresolved_questions": [],
                    "actor_arc_directives": actor_arc_directives,
                    "cluster_path": [
                        {
                            "occurrence_id": f"occ_{idx + 1:03d}",
                            "cluster_id": cluster.get("cluster_id", "cluster_001"),
                            "usage": "primary",
                            "emphasis": cluster.get("coverage_policy", "anchor"),
                            "transition_note": "",
                            "chronology_break": None,
                        }
                    ],
                }
            )
        return {
            "strategy_type": "convergence",
            "justification": "Heuristic: defaulting to convergence strategy.",
            "series_arc": "Books converge on shared themes.",
            "episode_arc_outline": [f"Episode {idx + 1}" for idx in range(recommended_episode_count)],
            "recommended_episode_count": recommended_episode_count,
            "episodes": episodes,
        }

    def _generate_episode_planning(self, payload: PromptPayload) -> dict[str, Any]:
        episode = payload.get("episode", {})
        episode_number = int(episode.get("episode_number", 1))
        cluster_path = episode.get("cluster_path", [])
        available_passages = payload.get("available_passages", [])
        passage_ids = [
            str(passage.get("passage_id", uuid4().hex))
            for passage in available_passages[:3]
        ]
        occurrence_id = (
            str(cluster_path[0].get("occurrence_id", "occ_001"))
            if cluster_path
            else "occ_001"
        )
        scene_cards = [
            {
                "scene_id": "scene_01",
                "title": "Heuristic scene 1",
                "card_kind": "normal",
                "scene_role": "setup",
                "dominant_cluster_occurrence_id": occurrence_id,
                "entry_image": "A concrete opening image.",
                "local_question": "What changes here?",
                "observable_detail": "A visible consequence lands in the scene.",
                "intended_move": "Move the listener into the next discovery step.",
                "timeframe": None,
                "location": None,
                "actors": [],
                "primitive_ids": [],
                "passage_ids": passage_ids,
                "estimated_duration_seconds": 4200,
                "coverage_depth": "deep",
            }
        ]
        return {
            "episode_number": episode_number,
            "title": episode.get("title", f"Episode {episode_number}"),
            "driving_question": episode.get("driving_question", "What changes here?"),
            "thematic_focus": episode.get("thematic_focus", "Heuristic focus"),
            "arc_summary": episode.get("arc_summary", "A heuristic episode arc."),
            "unresolved_questions": episode.get("unresolved_questions", []),
            "actor_arc_directives": episode.get("actor_arc_directives", []),
            "framing": {
                "opening_image": "A listener-facing opening image.",
                "threat_or_unresolved_action": "A threat remains active as the episode starts.",
                "opening_question": episode.get("driving_question", "What changes here?"),
                "handoff_scene_card_id": "scene_01",
                "recap": None,
                "preview": None,
            },
            "scene_cards": scene_cards,
            "target_duration_minutes": 90.0,
        }

    def _generate_episode_writing(self, payload: PromptPayload) -> dict[str, Any]:
        batch_id = str(payload.get("batch_id", "batch_1"))
        active_scene_card_ids = [
            str(scene_id) for scene_id in payload.get("active_scene_card_ids", [])
        ] or ["scene_01"]
        return {
            "batch_id": batch_id,
            "prose_sections": [
                {
                    "section_id": f"section_{batch_id}",
                    "scene_card_ids": active_scene_card_ids,
                    "movement_goal": "discover",
                    "text": "Heuristic narration content.",
                    "citations": [],
                    "source_book_ids": [],
                }
            ],
            "transitions": [],
            "window_map": [
                {
                    "batch_id": batch_id,
                    "section_ids": [f"section_{batch_id}"],
                    "transition_ids": [],
                }
            ],
        }

    def _generate_grounding_validation(self, payload: PromptPayload) -> dict[str, Any]:
        return {
            "episode_number": payload.get("episode_number", 1),
            "claim_assessments": [],
            "cross_book_claims": [],
            "overall_status": "PASSED",
            "grounding_score": 1.0,
            "attribution_accuracy": 1.0,
            "fairness_flags": [],
        }

    def _generate_repair(self, payload: PromptPayload) -> dict[str, Any]:
        return {"repaired_sections": [], "repaired_transitions": []}

    def _generate_spoken_delivery(self, payload: PromptPayload) -> dict[str, Any]:
        script = payload.get("script", {})
        return {
            "sections": [
                {
                    "section_id": str(section.get("section_id", uuid4().hex)),
                    "text": str(section.get("text", "Spoken delivery text.")),
                    "speech_hints": {
                        "style": "neutral",
                        "intensity": "none",
                        "pause_before_ms": 300,
                        "pause_after_ms": 300,
                        "pace": "normal",
                    },
                }
                for section in script.get("prose_sections", [])
            ],
            "transitions": [
                {
                    "transition_id": str(transition.get("transition_id", uuid4().hex)),
                    "text": str(transition.get("text", "Transition.")),
                    "speech_hints": {
                        "style": "neutral",
                        "intensity": "none",
                        "pause_before_ms": 300,
                        "pause_after_ms": 300,
                        "pace": "normal",
                    },
                }
                for transition in script.get("transitions", [])
            ],
        }
