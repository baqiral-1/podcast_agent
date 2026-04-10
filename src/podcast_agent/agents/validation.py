"""Stage 11: grounding validation agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import grounding_validation_instructions
from podcast_agent.schemas.models import GroundingReport


class GroundingValidationAgent(Agent):
    """Verifies grounded support for section-based episode drafts."""

    schema_name = "grounding_validation"
    response_model = GroundingReport
    instructions = grounding_validation_instructions()

    def build_payload(
        self,
        episode_number: int,
        script: dict,
        passages: dict[str, dict],
    ) -> dict:
        return {
            "episode_number": episode_number,
            "script": script,
            "cited_passages": passages,
        }
