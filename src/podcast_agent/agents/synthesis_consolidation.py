"""Stage 7b: synthesis consolidation agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import synthesis_consolidation_instructions
from podcast_agent.schemas.models import SynthesisConsolidationResult


class SynthesisConsolidationAgent(Agent):
    """Consolidates primitives into the cluster-first synthesis artifact."""

    schema_name = "synthesis_consolidation"
    response_model = SynthesisConsolidationResult
    instructions = synthesis_consolidation_instructions()

    def build_payload(
        self,
        project_id: str,
        primitives: dict,
        axes_summary: list[dict],
        book_metadata: list[dict],
        series_size_hint: int | None,
        actor_metadata: dict | None = None,
        consolidation_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "project_id": project_id,
            "primitives": primitives,
            "axes": axes_summary,
            "books": book_metadata,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if series_size_hint is not None:
            payload["series_size_hint"] = series_size_hint
        if consolidation_feedback is not None:
            payload["consolidation_feedback"] = consolidation_feedback
        return payload
