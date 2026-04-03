"""Stage 12: Repair agent for grounding failures."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import ScriptSegment


class RepairResponse(BaseModel):
    repaired_segments: list[ScriptSegment] = Field(default_factory=list)


class RepairAgent(Agent):
    """Fixes grounding failures without rewriting the entire episode."""

    schema_name = "repair"
    response_model = RepairResponse
    instructions = (
        "You are a script editor. Fix the following claims that failed grounding validation.\n\n"
        "For misattributions: correct which author said what.\n"
        "For unsupported claims: rewrite using the cited passage.\n"
        "For fairness issues: represent the author's position more accurately.\n\n"
        "Change only what's necessary — preserve the surrounding narrative flow.\n"
        "Maintain all existing citations and add new ones where needed.\n\n"
        "Return a JSON object with a 'repaired_segments' array containing the fixed segments."
    )

    def build_payload(
        self,
        failing_segments: list[dict],
        failure_reasons: list[dict],
        passages: dict[str, dict],
    ) -> dict:
        return {
            "failing_segments": failing_segments,
            "failure_reasons": failure_reasons,
            "cited_passages": passages,
        }
