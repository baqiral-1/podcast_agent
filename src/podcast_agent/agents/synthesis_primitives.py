"""Stage 7a: synthesis primitives agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import synthesis_primitives_instructions
from podcast_agent.schemas.models import SynthesisPrimitivesArtifact


class SynthesisPrimitivesAgent(Agent):
    """Extracts grounded synthesis primitives from the full evidence surface."""

    schema_name = "synthesis_primitives"
    response_model = SynthesisPrimitivesArtifact
    instructions = synthesis_primitives_instructions()

    def build_payload(
        self,
        project_id: str,
        axes_summary: list[dict],
        passages_by_axis: dict[str, list[dict]],
        cross_book_pairs: list[dict],
        book_metadata: list[dict],
        synthesis_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "project_id": project_id,
            "axes": axes_summary,
            "passages_by_axis": passages_by_axis,
            "cross_book_pairs": cross_book_pairs,
            "books": book_metadata,
        }
        if synthesis_feedback is not None:
            payload["synthesis_feedback"] = synthesis_feedback
        return payload
