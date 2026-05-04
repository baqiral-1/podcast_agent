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
        family: str,
        primitive_id: str,
        title: str,
        summary: str,
        axis_ids: list[str],
        passage_ids: list[str],
        actor_ids: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actor_ids = actor_ids or []
        primitive = {
            "id": primitive_id,
            "family": family,
            "title": title,
            "summary": summary,
            "axis_ids": axis_ids,
            "core_passage_ids": passage_ids[:1],
            "support_passage_ids": passage_ids[1:3],
            "timeframe": None,
            "geography": None,
            "actor_ids": actor_ids[:1],
            "narrative_importance_score": 0.5,
        }
        if extra_fields:
            primitive.update(extra_fields)
        return primitive

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
            "epochal_turns": (
                "et_001",
                "Heuristic epochal turn",
                "A large irreversible pivot changes the rules of the story.",
            ),
            "decisions_and_nondecisions": (
                "dn_001",
                "Heuristic decision or nondecision",
                "A choice, refusal, delay, or failure to act redirects events.",
            ),
            "set_piece_scenes": (
                "ss_001",
                "Heuristic set piece",
                "A major playable scene concentrates the story's pressure.",
            ),
            "telling_details": (
                "td_001",
                "Heuristic telling detail",
                "A concrete memorable detail makes the history locally vivid.",
            ),
            "human_costs": (
                "hc_001",
                "Heuristic human cost",
                "A development lands as lived damage for people on the ground.",
            ),
            "character_engines": (
                "cg_001",
                "Heuristic character engine",
                "A person's motive, fear, or ambition helps drive the story.",
            ),
            "coalitions_and_fault_lines": (
                "cf_001",
                "Heuristic coalition fault line",
                "An alliance holds, strains, or fractures under pressure.",
            ),
            "systems_and_operating_logics": (
                "so_001",
                "Heuristic operating logic",
                "A system or institutional logic explains how pressure propagates.",
            ),
            "misreadings_and_fantasies": (
                "mf_001",
                "Heuristic misreading",
                "Actors misread reality or cling to a flattering illusion.",
            ),
            "contested_explanations": (
                "cx_001",
                "Heuristic contested explanation",
                "The evidence supports more than one plausible explanation.",
            ),
            "perspective_windows": (
                "pw_001",
                "Heuristic perspective window",
                "Meaning changes when seen from a different vantage.",
            ),
            "moral_traps": (
                "mt_001",
                "Heuristic moral trap",
                "Every available option carries compromise or stain.",
            ),
            "afterlives": (
                "al_001",
                "Heuristic afterlife",
                "An earlier rupture keeps shaping the present.",
            ),
            "recurring_images_and_symbols": (
                "rs_001",
                "Heuristic recurring image",
                "An image or symbol can recur across the series and gather meaning.",
            ),
            "ironies_and_reversals": (
                "ir_001",
                "Heuristic irony or reversal",
                "A development lands opposite to its intended or expected result.",
            ),
        }
        primitives_by_family: dict[str, list[dict[str, Any]]] = {}
        for family in SYNTHESIS_PRIMITIVE_FAMILIES:
            primitive_id, title, summary = family_templates[family]
            primitive = self._build_primitive(
                family=family,
                primitive_id=primitive_id,
                title=title,
                summary=summary,
                axis_ids=axis_ids[:1],
                passage_ids=passage_ids[:3],
                actor_ids=actor_ids[:1],
            )
            primitives_by_family[family] = [primitive]
        return {
            "project_id": payload.get("project_id", "project"),
            "primitives_by_family": primitives_by_family,
            "quality_score": 0.5,
            "quality_notes": ["Heuristic primitives artifact."],
        }

    def _generate_primitive_enrichment(self, payload: PromptPayload) -> dict[str, Any]:
        family = str(payload.get("family", "")).strip() or "epochal_turns"
        base_primitives = payload.get("base_primitives", []) or []
        enriched_primitives: list[dict[str, Any]] = []
        for primitive in base_primitives:
            primitive_id = str(primitive.get("id", uuid4().hex))
            actor_ids = list(primitive.get("actor_ids", []) or [])
            scoped_actor_ids = actor_ids[:2] or ["actor_001"]
            primary_actor_id = scoped_actor_ids[0]
            delta: dict[str, Any] = {"id": primitive_id, "family": family}
            hooks = {
                "concrete_detail": "A concrete detail crystallizes the turn.",
                "host_lens": "The pressure is now visible.",
                "carry_forward": "The residue shapes what follows.",
            }
            if family == "epochal_turns":
                delta.update(
                    {
                        "before_state": "The old balance still holds.",
                        "after_state": "A new political reality is now in force.",
                        "change_driver": "A decisive turn forces the transition.",
                        "why_no_return": "The institutional and political costs cannot be unwound quickly.",
                        "proof_of_change": "The new balance becomes publicly undeniable.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "decisions_and_nondecisions":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "decision_trigger": "A fresh shock forces the actor to choose.",
                        "decision_question": "Should the actor force a move now or hold back?",
                        "decision_mode": "decision",
                        "options_considered": ["Act immediately.", "Hold position and wait."],
                        "next_result": "The choice redirects the next phase of events.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "set_piece_scenes":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "scene_anchor": "A public confrontation concentrates pressure in one place.",
                        "hinge_action": "One visible move breaks the standoff.",
                        "scene_outcome": "The confrontation makes the next turn unavoidable.",
                        "location": "capital",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "human_costs":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "affected_group": "ordinary families near the center of events",
                        "cost_type": "displacement and fear",
                        "concrete_marker": "Households carry loss into the street.",
                        "lived_consequence": "People closest to the rupture absorb the damage first.",
                        "who_saw_it": "public but unevenly acknowledged",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "character_engines":
                delta.update(
                    {
                        "actor_id": primary_actor_id,
                        "goal": "Protect a fragile position.",
                        "pressure_box": "Institutions, rivals, and fear narrow the available choices.",
                        "risk_if_it_breaks": "The actor's status and safety both depend on the outcome.",
                        "tell": "The actor keeps justifying the same risky move.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "coalitions_and_fault_lines":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "alignment_type": "tactical",
                        "coalition_phase": "holding",
                        "alignment_shape": "A temporary alliance of convenience.",
                        "alignment_basis": "Each side needs the other for the moment.",
                        "fracture_trigger": "The alliance weakens when pressure rises.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "systems_and_operating_logics":
                delta.update(
                    {
                        "system_name": "A coercive political-administrative machine",
                        "operating_chain": [
                            "Orders move downward through patronage and command.",
                            "Local intermediaries translate pressure into compliance.",
                        ],
                        "inputs": ["orders", "money"],
                        "outputs": ["compliance", "distortion"],
                        "where_it_shows_up": "Officials and intermediaries enforce it face to face.",
                        "failure_mode": "The system becomes brittle when pressure outruns coordination.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "contested_explanations":
                base_passage_ids = list(primitive.get("core_passage_ids", []) or [])
                delta.update(
                    {
                        "candidate_readings": [
                            {
                                "label": "reading_a",
                                "claim": "A cautious interpretation of the evidence.",
                                "emphasizes": "institutional caution",
                                "downplays": "short-term opportunism",
                                "support_passage_ids": base_passage_ids[:1],
                            },
                            {
                                "label": "reading_b",
                                "claim": "A competing interpretation that remains plausible.",
                                "emphasizes": "short-term opportunism",
                                "downplays": "institutional caution",
                                "support_passage_ids": base_passage_ids[:1],
                            },
                        ],
                        "narration_hooks": hooks,
                    }
                )
            elif family == "moral_traps":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "competing_obligations": ["Protect allies.", "Preserve institutional order."],
                        "compromised_options": ["Act and deepen the damage.", "Wait and allow the damage to spread."],
                        "trap_structure": "Every plausible move imposes a visible cost.",
                        "narration_hooks": hooks,
                    }
                )
            elif family == "ironies_and_reversals":
                delta.update(
                    {
                        "actor_ids": scoped_actor_ids,
                        "expected_outcome": "The move should stabilize the situation.",
                        "actual_outcome": "It instead accelerates the breakdown.",
                        "flip_cause": "The same tactic triggers the opposite effect under pressure.",
                        "narration_hooks": hooks,
                    }
                )
            else:
                continue
            enriched_primitives.append(delta)
        return {
            "project_id": payload.get("project_id", "project"),
            "family": family,
            "enriched_primitives": enriched_primitives,
        }

    def _generate_narrative_strategy(self, payload: PromptPayload) -> dict[str, Any]:
        requested_episode_count = payload.get("requested_episode_count")
        recommended_episode_count_min = int(payload.get("recommended_episode_count_min", 8))
        recommended_episode_count_max = int(payload.get("recommended_episode_count_max", 12))
        if recommended_episode_count_max < recommended_episode_count_min:
            recommended_episode_count_max = recommended_episode_count_min
        synthesis_map = payload.get("synthesis_map", {})
        primitive_ids = []
        for family_items in (synthesis_map.get("primitives_by_family", {}) or {}).values():
            for item in family_items:
                primitive_id = str(item.get("id", ""))
                if primitive_id:
                    primitive_ids.append(primitive_id)
        if not primitive_ids:
            primitive_ids = ["primitive_001", "primitive_002", "primitive_003"]
        while len(primitive_ids) < 96:
            primitive_ids.append(f"primitive_{len(primitive_ids) + 1:03d}")
        if requested_episode_count is None:
            recommended_episode_count = recommended_episode_count_min
        else:
            recommended_episode_count = int(requested_episode_count)
        episodes = []
        for idx in range(recommended_episode_count):
            listener_problem = (
                "What local turn best explains the series?"
                if idx == 0
                else f"What does episode {idx + 1} newly reveal?"
            )
            core_start = idx * 13
            core_ids = primitive_ids[core_start:core_start + 5]
            if len(core_ids) < 5:
                core_ids = primitive_ids[:5]
            support_candidates = primitive_ids[core_start + 5:core_start + 15]
            support_ids = [
                primitive_id
                for primitive_id in support_candidates
                if primitive_id not in core_ids
            ][:5]
            if len(support_ids) < 5:
                support_ids = [
                    primitive_id
                    for primitive_id in primitive_ids
                    if primitive_id not in core_ids
                ][:5]
            episodes.append(
                {
                    "episode_number": idx + 1,
                    "title": f"Episode {idx + 1}",
                    "thematic_focus": "Heuristic focus",
                    "arc_summary": f"Episode {idx + 1} follows a single listener-facing pressure line.",
                    "unresolved_questions": [],
                    "actor_arc_directives": [],
                    "episode_spine": {
                        "listener_problem": listener_problem,
                        "episode_answer": "Selected primitives carry the episode's answer.",
                        "pressure_line": "The listener should feel one pressure line tightening across the episode.",
                        "core_primitive_ids": core_ids,
                        "support_primitive_roles": {
                            primitive_id: "mechanism"
                            for primitive_id in support_ids
                            if primitive_id not in core_ids
                        },
                        "recall_primitive_ids": [],
                    },
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

    def _generate_episode_architecture(self, payload: PromptPayload) -> dict[str, Any]:
        episode = payload.get("episode", {})
        episode_spine = episode.get("episode_spine", {})
        primitive_ids = list(episode_spine.get("assigned_primitive_ids") or [])
        if not primitive_ids:
            primitive_ids = list(episode_spine.get("core_primitive_ids") or ["primitive_001"])
        section_count = min(8, max(6, len(primitive_ids))) if primitive_ids else 6
        closing_minutes = 2.0
        non_closing_count = max(1, section_count - 1)
        non_closing_minutes = max(1.0, (85.0 - closing_minutes) / non_closing_count)
        sections = []
        for idx in range(section_count):
            section_id = f"section_{idx + 1:02d}"
            local_primitive_ids = [primitive_ids[min(idx, len(primitive_ids) - 1)]]
            is_closing = idx == section_count - 1
            sections.append(
                {
                    "section_id": section_id,
                    "purpose": (
                        "opening"
                        if idx == 0
                        else "closing" if is_closing else "setup"
                    ),
                    "approx_runtime_minutes": closing_minutes if is_closing else non_closing_minutes,
                    "primitive_ids": local_primitive_ids,
                    "section_anchor": "A concrete section anchor keeps the listener oriented.",
                    "must_stage_beats": [
                        f"Stage the visible move that defines section {idx + 1}.",
                        f"Show the immediate consequence that makes section {idx + 1} matter.",
                    ],
                    "listener_tension": f"What pressure does section {idx + 1} put on the listener?",
                    "section_turn": f"Section {idx + 1} changes the listener's understanding.",
                    "transition_logic": "Move by clarifying the next structural pressure.",
                    "depends_on_section_ids": [f"section_{idx:02d}"] if idx > 0 else [],
                    "sets_up_section_ids": [f"section_{idx + 2:02d}"] if idx < section_count - 1 else [],
                    "recurrence_role": "plant" if idx == 0 else "payoff" if idx == section_count - 1 else "deepen",
                    "closure_mode": "final_answer" if idx == section_count - 1 else "residue",
                    "priority_core_passage_ids": [],
                }
            )
        return {
            "episode_number": int(episode.get("episode_number", 1)),
            "major_turn_section_id": sections[min(2, len(sections) - 1)]["section_id"],
            "allowed_recurring_primitive_ids": primitive_ids[:2],
            "forbidden_redundancies": [],
            "sections": sections,
            "architecture_notes": ["Heuristic architecture output."],
        }

    def _generate_episode_planning(self, payload: PromptPayload) -> dict[str, Any]:
        architecture = payload.get("architecture", {})
        strategy_episode = payload.get("strategy_episode", {})
        episode_number = int(architecture.get("episode_number", 1))
        episode_spine = strategy_episode.get("episode_spine", {})
        available_passages = payload.get("available_passages", [])
        passage_ids = [
            str(passage.get("passage_id", uuid4().hex))
            for passage in available_passages[:3]
        ]
        sections = architecture.get("sections") or [{"section_id": "section_01"}]
        scene_cards = []
        default_duration = 180
        for idx in range(18):
            source_section = sections[min(idx, len(sections) - 1)]
            section_id = str(source_section.get("section_id", "section_01"))
            is_closing = idx == 17
            scene_cards.append(
                {
                    "scene_id": f"scene_{idx + 1:02d}",
                    "section_id": section_id if not is_closing else str(sections[-1].get("section_id", section_id)),
                    "title": f"Heuristic scene {idx + 1}",
                    "scene_role": "setup" if idx == 0 else "consequence" if is_closing else "action",
                    "beat_change": "The listener enters the episode's opening pressure." if idx == 0 else "The next beat changes the situation in a concrete way." if not is_closing else "The closing beat leaves a constrained answer in view.",
                    "must_land_facts": ["A concrete change becomes visible in the beat."],
                    "entry_image": "A concrete opening image." if idx == 0 else "A grounded detail keeps the scene moving.",
                    "observable_detail": "A visible consequence lands in the scene.",
                    "timeframe": None,
                    "location": None,
                    "actors": [],
                    "passage_ids": passage_ids,
                    "estimated_duration_seconds": 120 if is_closing else default_duration,
                    "host_move": {
                        "move_type": "none",
                        "note": "",
                        "max_sentences": 1,
                        "placement": "close",
                    },
                }
            )
        return {
            "episode_number": episode_number,
            "framing": {
                "opening_image": "A listener-facing opening image.",
                "threat_or_unresolved_action": "A threat remains active as the episode starts.",
                "opening_question": (
                    (
                        strategy_episode.get("episode_spine", {}) or {}
                    ).get("listener_problem", "What changes here?")
                    or "What changes here?"
                ),
                "handoff_scene_card_id": "scene_01",
                "recap": None,
                "preview": None,
            },
            "scene_cards": scene_cards,
            "dropped_support_primitive_reasons": {},
        }

    def _generate_episode_writing(self, payload: PromptPayload) -> dict[str, Any]:
        plan = payload.get("plan", {})
        scene_cards = plan.get("scene_cards", [])
        scene_card_ids = [
            str(scene.get("scene_id"))
            for scene in scene_cards
            if isinstance(scene, dict) and scene.get("scene_id")
        ] or ["scene_01"]
        return {
            "scene_prose": [
                {
                    "scene_card_id": scene_id,
                    "movement_goal": "discover",
                    "text": "Heuristic narration content.",
                    "citations": [],
                    "source_book_ids": [],
                }
                for scene_id in scene_card_ids
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
        return {"repaired_sections": []}

    def _generate_spoken_delivery(self, payload: PromptPayload) -> dict[str, Any]:
        script = payload.get("script", {})
        prose_sections = list(script.get("prose_sections", []) or [])
        return {
            "sections": [
                {
                    "section_id": str(section.get("section_id", f"section_{idx + 1}")),
                    "text": str(section.get("text", "")).strip() or "Spoken delivery text.",
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
                    "edited_text": str(section.get("text", "")).strip() or "Audited text.",
                    "edit_notes": [],
                }
                for idx, section in enumerate(sections)
            ],
            "episode_warnings": [],
        }
