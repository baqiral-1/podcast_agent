"""Repair agent for weak or unsupported script segments."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodeScript, GroundingReport, RepairResult


class RepairAgent(Agent):
    """Agent that rewrites failing segments while preserving grounded claims."""

    schema_name = "episode_repair"
    instructions = "Repair only the failed script segments and keep the output JSON-valid."
    response_model = RepairResult

    def build_payload(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> dict:
        return {
            "script": script.model_dump(mode="python"),
            "report": report.model_dump(mode="python"),
            "attempt": attempt,
        }

    def repair(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> RepairResult:
        """Repair failing script segments."""

        return self.run(self.build_payload(script, report, attempt))
