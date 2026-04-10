"""Stage 14: style audit agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import style_audit_instructions
from podcast_agent.schemas.models import StyleAuditReport


class StyleAuditAgent(Agent):
    """Runs a warnings-only style audit over the spoken script."""

    schema_name = "style_audit"
    response_model = StyleAuditReport
    instructions = style_audit_instructions()

    def build_payload(self, episode_number: int, script: dict) -> dict:
        return {"episode_number": episode_number, "script": script}
