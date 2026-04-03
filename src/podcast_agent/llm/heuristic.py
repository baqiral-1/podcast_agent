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

    def _generate_theme_decomposition(self, payload: PromptPayload) -> dict[str, Any]:
        books = payload.get("books", [])
        book_ids = [b.get("book_id", f"b{i}") for i, b in enumerate(books)]
        relevance = {bid: 0.7 for bid in book_ids}
        return {
            "axes": [
                {
                    "axis_id": uuid4().hex,
                    "name": f"Axis based on: {payload.get('theme', 'unknown')}",
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
                "insight_type": "agreement",
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
                    "narrative": "Heuristic merged narrative.",
                    "source_passage_ids": passage_ids[:3],
                    "points_of_consensus": [],
                    "points_of_disagreement": [],
                }
            ],
        }

    def _generate_narrative_strategy(self, payload: PromptPayload) -> dict[str, Any]:
        return {
            "strategy_type": "convergence",
            "justification": "Heuristic: defaulting to convergence strategy.",
            "series_arc": "Books converge on shared themes.",
            "episode_arc_outline": [
                f"Episode {i + 1}" for i in range(payload.get("episode_count", 3))
            ],
        }

    def _generate_series_planning(self, payload: PromptPayload) -> dict[str, Any]:
        episode_count = payload.get("episode_count", 3)
        episodes = []
        for i in range(1, episode_count + 1):
            episodes.append({
                "episode_number": i,
                "title": f"Episode {i}",
                "thematic_focus": "Heuristic focus",
                "attribution_budget": payload.get("attribution_budget", 0.2),
                "narrative_spine": {
                    "episode_number": i,
                    "spine_segments": [
                        {
                            "segment_id": uuid4().hex,
                            "narrative_text": f"Heuristic spine segment {i}.",
                            "source_passages": [],
                            "segment_function": "context",
                            "era_or_moment": "",
                        }
                    ],
                    "attribution_moments": [],
                    "narrative_voice": "omniscient narrator telling a story",
                },
                "beats": [
                    {
                        "beat_id": uuid4().hex,
                        "description": f"Beat for episode {i}",
                        "narrative_instruction": "advance_events",
                        "attribution_level": "none",
                    }
                ],
            })
        return {"episodes": episodes}

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
            })
        if not spoken_segments:
            spoken_segments = [{
                "segment_id": uuid4().hex,
                "text": "Heuristic spoken delivery.",
                "max_words": 250,
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
