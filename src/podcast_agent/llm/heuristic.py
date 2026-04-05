"""Heuristic LLM implementation used for local development and tests."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient, PromptPayload, prompt_log_metadata


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
            if generator is not None:
                response = generator(payload)
            else:
                response = self._generate_default(schema_name, payload)
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
        """Produce a minimal valid response for unknown schemas."""
        raise ValueError(f"No heuristic generator for schema '{schema_name}'.")

    def _generate_structuring(self, payload: PromptPayload) -> dict[str, Any]:
        text_window = payload.get("text_window", "")
        offset = payload.get("window_character_offset", 0)
        word_count = len(text_window.split())
        return {
            "chapters": [
                {
                    "chapter_id": uuid4().hex,
                    "title": "Chapter 1",
                    "start_index": 0,
                    "end_index": len(text_window),
                    "word_count": word_count,
                    "summary": "Heuristic chapter summary.",
                }
            ]
        }

    def _generate_chapter_summary(self, payload: PromptPayload) -> dict[str, Any]:
        chapter_title = str(payload.get("chapter_title", "")).strip() or "This chapter"
        chapter_text = str(payload.get("chapter_text", "")).strip()
        words = chapter_text.split()
        first_terms = [word.strip(".,;:!?()[]\"'") for word in words[:40]]
        keywords: list[str] = []
        for term in first_terms:
            cleaned = term.strip()
            if len(cleaned) < 5:
                continue
            if cleaned.lower() in {item.lower() for item in keywords}:
                continue
            keywords.append(cleaned)
            if len(keywords) >= 5:
                break
        return {
            "summary": f"{chapter_title} is summarized heuristically from the source text.",
            "analysis": {
                "themes_touched": [chapter_title] if chapter_title else [],
                "major_actors": [],
                "key_places": [],
                "key_institutions": [],
                "timeframe": "",
                "key_events_or_arguments": [
                    f"{chapter_title} contributes material relevant to the project theme."
                ],
                "major_tensions": [],
                "causal_shifts": [],
                "narrative_hooks": [f"Return to {chapter_title} for episode construction."],
                "retrieval_keywords": keywords,
            },
        }

    def _generate_book_summary(self, payload: PromptPayload) -> dict[str, Any]:
        theme = str(payload.get("theme", "")).strip()
        sub_themes = [
            str(item).strip()
            for item in payload.get("sub_themes", [])
            if str(item).strip()
        ]
        title = str(payload.get("title", "")).strip() or "this book"
        chapters = payload.get("chapters", [])
        chapter_titles = [
            chapter.get("title", "")
            for chapter in chapters[:3]
            if isinstance(chapter, dict) and chapter.get("title")
        ]
        coverage = ", ".join(chapter_titles) if chapter_titles else "its chapters"
        if theme:
            if sub_themes:
                summary = (
                    f"{title} addresses {theme} ({', '.join(sub_themes[:2])}) "
                    f"through {coverage}."
                )
            else:
                summary = f"{title} addresses {theme} through {coverage}."
        else:
            summary = f"{title} is summarized through {coverage}."
        return {"summary": summary}

    def _generate_theme_decomposition(self, payload: PromptPayload) -> dict[str, Any]:
        books = payload.get("books", [])
        book_ids = [b.get("book_id", f"b{i}") for i, b in enumerate(books)]
        relevance = {bid: 0.7 for bid in book_ids}
        axis_label = str(payload.get("theme", "unknown"))
        sub_themes = [
            str(item).strip()
            for item in payload.get("sub_themes", [])
            if str(item).strip()
        ]
        if sub_themes:
            axis_label = f"{axis_label} + {sub_themes[0]}"
        return {
            "axes": [
                {
                    "axis_id": uuid4().hex,
                    "name": f"Axis based on: {axis_label}",
                    "description": "Heuristic thematic axis.",
                    "guiding_questions": ["How does this theme manifest?"],
                    "relevance_by_book": relevance,
                    "keywords": ["theme"],
                }
            ]
        }

    def _generate_passage_extraction(self, payload: PromptPayload) -> dict[str, Any]:
        candidates = payload.get("candidate_passages", [])
        passages = []
        for c in candidates[:5]:
            passages.append({
                "passage_id": c.get("passage_id", uuid4().hex),
                "relevance_score": 0.7,
                "quotability_score": 0.6,
                "synthesis_tags": ["independent"],
            })
        return {"passages": passages, "cross_book_pairs": []}

    def _generate_synthesis_mapping(self, payload: PromptPayload) -> dict[str, Any]:
        books = payload.get("books", [])
        passages_by_axis = payload.get("passages_by_axis", {})

        # Collect passage IDs across books
        passage_ids: list[str] = []
        for axis_passages in passages_by_axis.values():
            for p in axis_passages:
                passage_ids.append(p.get("passage_id", uuid4().hex))

        insights = []
        if len(passage_ids) >= 2:
            insights.append({
                "insight_id": uuid4().hex,
                "insight_type": "synchronicity",
                "title": "Shared perspective",
                "description": "Authors share a common perspective on this topic.",
                "passage_ids": passage_ids[:2],
                "podcast_potential": 0.7,
                "treatment": "build",
            })

        return {
            "insights": insights,
            "narrative_threads": [],
            "book_relationship_matrix": {},
            "unresolved_tensions": [],
            "quality_score": 0.5,
            "merged_narratives": [
                {
                    "topic": "Heuristic topic",
                    "narrative": "There is a compelling case to be made that a shared pattern emerges here ("
                    + (passage_ids[0] if passage_ids else "passage_id")
                    + ").",
                    "source_passage_ids": passage_ids[:3],
                    "points_of_consensus": [],
                    "points_of_disagreement": [],
                }
            ],
        }

    def _generate_narrative_strategy(self, payload: PromptPayload) -> dict[str, Any]:
        requested_episode_count = payload.get("requested_episode_count")
        synthesis_map = payload.get("synthesis_map", {})
        thematic_axes = payload.get("thematic_axes", [])
        insights = synthesis_map.get("insights", [])
        if requested_episode_count is None:
            insight_count = int(synthesis_map.get("insight_count", len(insights)))
            thread_count = int(synthesis_map.get("thread_count", 0))
            quality_score = float(synthesis_map.get("quality_score", 0.0))
            base = max(2, (insight_count // 5) + (thread_count // 2))
            if quality_score >= 0.75:
                base += 1
            recommended_episode_count = max(2, min(8, base))
        else:
            recommended_episode_count = max(2, min(8, int(requested_episode_count)))
        episode_assignments = []
        axis_ids = [axis.get("axis_id", uuid4().hex) for axis in thematic_axes] or [uuid4().hex]
        insight_ids = [insight.get("insight_id", uuid4().hex) for insight in insights]
        for i in range(recommended_episode_count):
            axis_id = axis_ids[i % len(axis_ids)]
            assigned_insights = []
            if insight_ids:
                assigned_insights = [insight_ids[i % len(insight_ids)]]
            episode_assignments.append(
                {
                    "episode_number": i + 1,
                    "title": f"Episode {i + 1}",
                    "thematic_focus": f"Focus on axis {axis_id[:8]}",
                    "axis_ids": [axis_id],
                    "insight_ids": assigned_insights,
                    "merged_narrative_ids": [],
                    "tension_ids": [],
                    "episode_strategy": "advance main thread",
                }
            )
        return {
            "strategy_type": "convergence",
            "justification": "Heuristic: defaulting to convergence strategy.",
            "series_arc": "Books converge on shared themes.",
            "episode_arc_outline": [
                f"Episode {i + 1}" for i in range(recommended_episode_count)
            ],
            "recommended_episode_count": recommended_episode_count,
            "episode_assignments": episode_assignments,
        }

    def _generate_episode_planning(self, payload: PromptPayload) -> dict[str, Any]:
        assignment = payload.get("episode_assignment", {})
        episode_number = int(assignment.get("episode_number", 1))
        axis_ids = assignment.get("axis_ids", [])
        insight_ids = assignment.get("insight_ids", [])
        available_passages = payload.get("available_passages", {})
        synthesis_map = payload.get("synthesis_map", {})
        first_axis = axis_ids[0] if axis_ids else ""
        passage_pool = available_passages.get(first_axis, []) if first_axis else []
        selected_passage_ids = [
            passage.get("passage_id", uuid4().hex)
            for passage in passage_pool[:3]
        ]
        beats = []
        for i in range(40):
            beats.append(
                {
                    "beat_id": uuid4().hex,
                    "description": f"Beat {i + 1} for episode {episode_number}",
                    "passage_ids": selected_passage_ids,
                    "narrative_instruction": "advance_events",
                    "attribution_level": "none",
                    "estimated_duration_seconds": 150,
                }
            )
        return {
            "episode_number": episode_number,
            "title": assignment.get("title", f"Episode {episode_number}"),
            "thematic_focus": assignment.get("thematic_focus", "Heuristic focus"),
            "axis_ids": axis_ids,
            "insight_ids": insight_ids,
            "attribution_budget": 0.2,
            "beats": beats,
            "narrative_spine": {
                "episode_number": episode_number,
                "spine_segments": [
                    {
                        "segment_id": uuid4().hex,
                        "narrative_text": f"Heuristic spine segment {episode_number}.",
                        "source_passages": selected_passage_ids,
                        "segment_function": "context",
                        "era_or_moment": "",
                    }
                ],
                "attribution_moments": [],
                "narrative_voice": "omniscient narrator telling a story",
            },
            "synthesis_context": synthesis_map or None,
            "target_duration_minutes": 100.0,
            "episode_strategy": assignment.get("episode_strategy", ""),
        }

    def _generate_episode_writing(self, payload: PromptPayload) -> dict[str, Any]:
        ep_num = payload.get("episode_number", 1)
        return {
            "title": f"Episode {ep_num}",
            "segments": [
                {
                    "segment_id": uuid4().hex,
                    "text": "Heuristic narration content.",
                    "segment_type": "body",
                    "attribution_level": "none",
                }
            ],
            "citations": [],
        }

    def _generate_source_weaving(self, payload: PromptPayload) -> dict[str, Any]:
        return {
            "text": "Here the story splits, and the answer depends on which evidence you trust.",
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
        return {"repaired_segments": []}

    def _generate_spoken_delivery(self, payload: PromptPayload) -> dict[str, Any]:
        segments = payload.get("script_segments", [])
        spoken_segments = []
        for seg in segments:
            spoken_segments.append({
                "segment_id": seg.get("segment_id", uuid4().hex),
                "text": seg.get("text", "Spoken delivery text."),
                "max_words": payload.get("max_words_per_segment", 250),
                "speech_hints": {
                    "style": "neutral",
                    "intensity": "none",
                    "pause_before_ms": 300,
                    "pause_after_ms": 300,
                    "pace": "normal",
                },
            })
        if not spoken_segments:
            spoken_segments = [{
                "segment_id": uuid4().hex,
                "text": "Heuristic spoken delivery.",
                "max_words": 250,
                "speech_hints": {
                    "style": "neutral",
                    "intensity": "none",
                    "pause_before_ms": 300,
                    "pause_after_ms": 300,
                    "pace": "normal",
                },
            }]
        return {"segments": spoken_segments, "arc_plan": None}

    def _generate_episode_framing(self, payload: PromptPayload) -> dict[str, Any]:
        ep_num = payload.get("episode_number", 1)
        total = payload.get("total_episodes", 1)
        return {
            "episode_number": ep_num,
            "recap": "Previously, we explored..." if ep_num > 1 else None,
            "preview": "Next time, we'll discover..." if ep_num < total else None,
            "cold_open": None,
        }
