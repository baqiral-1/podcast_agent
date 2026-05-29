"""Stage 7b: excerpt extraction agent.

Extracts the standalone excerpt record — the spoken and documentary source
material whose payload is the words themselves. Runs in parallel with primitive
substrate extraction and is fully self-contained (no later enrichment pass).
"""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import excerpt_extraction_instructions
from podcast_agent.schemas.models import ExcerptArtifact, RawExcerptArtifact

_EXCERPT_ID_PREFIX = "x"


class ExcerptExtractionAgent(Agent):
    """Extracts the series' excerpt record from quote-biased evidence."""

    schema_name = "excerpt_extraction"
    response_model = RawExcerptArtifact
    instructions = excerpt_extraction_instructions()

    @staticmethod
    def _count_bounds(payload: dict) -> tuple[int, int]:
        count_min = payload.get("excerpt_count_min")
        count_max = payload.get("excerpt_count_max")
        try:
            return int(count_min), int(count_max)
        except (TypeError, ValueError):
            return 140, 220

    def build_instructions(self, payload: dict) -> str:
        count_min, count_max = self._count_bounds(payload)
        return excerpt_extraction_instructions(count_min=count_min, count_max=count_max)

    def build_payload(
        self,
        project_id: str,
        podcast_mode: str,
        axes_summary: list[dict],
        passages_by_axis: dict[str, list[dict]],
        book_metadata: list[dict],
        excerpt_count_min: int,
        excerpt_count_max: int,
        actor_metadata: dict | None = None,
        excerpt_feedback: dict | None = None,
    ) -> dict:
        payload: dict = {
            "project_id": project_id,
            "podcast_mode": podcast_mode,
            "axes": axes_summary,
            "passages_by_axis": passages_by_axis,
            "books": book_metadata,
            "excerpt_count_min": excerpt_count_min,
            "excerpt_count_max": excerpt_count_max,
        }
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if excerpt_feedback is not None:
            payload["excerpt_feedback"] = excerpt_feedback
        return payload

    def validate_result(
        self,
        result: RawExcerptArtifact,
        payload: dict,
    ) -> ExcerptArtifact:
        excerpts = []
        for index, excerpt in enumerate(result.excerpts, start=1):
            data = excerpt.model_dump(mode="json")
            data["id"] = f"{_EXCERPT_ID_PREFIX}{index}"
            excerpts.append(data)
        return ExcerptArtifact.model_validate(
            {
                "project_id": result.project_id,
                "excerpts": excerpts,
                "quality_score": result.quality_score,
                "quality_notes": result.quality_notes,
            }
        )
