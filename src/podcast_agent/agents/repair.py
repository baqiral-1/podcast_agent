"""Stage 12: repair agent for grounding failures."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import repair_instructions
from podcast_agent.schemas.models import ProseSection


class RepairResponse(BaseModel):
    repaired_sections: list[ProseSection] = Field(default_factory=list)


class RepairAgent(Agent):
    """Fixes grounding failures without rewriting the entire episode."""

    schema_name = "repair"
    response_model = RepairResponse
    instructions = repair_instructions()

    def build_payload(
        self,
        failing_sections: list[dict],
        failure_reasons: list[dict],
        passages: dict[str, dict],
    ) -> dict:
        return {
            "failing_sections": failing_sections,
            "failure_reasons": failure_reasons,
            "cited_passages": passages,
        }
