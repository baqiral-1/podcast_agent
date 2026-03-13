"""Grounding validation agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodeScript, GroundingReport, RetrievalHit


class GroundingValidationAgent(Agent):
    """Agent that validates script claims against retrieved evidence."""

    schema_name = "grounding_report"
    instructions = "Validate each claim against the cited evidence and classify grounding strength."
    response_model = GroundingReport

    def build_payload(self, script: EpisodeScript, retrieval_hits: list[RetrievalHit]) -> dict:
        return {
            "script": script.model_dump(mode="python"),
            "retrieval_hits": [hit.model_dump(mode="python") for hit in retrieval_hits],
        }

    def validate(self, script: EpisodeScript, retrieval_hits: list[RetrievalHit]) -> GroundingReport:
        """Produce a claim-level grounding report."""

        return self.run(self.build_payload(script, retrieval_hits))
