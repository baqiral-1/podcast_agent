"""Episode-level subtractive style audit agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import style_audit_instructions
from podcast_agent.schemas.models import StyleAuditResponse


class StyleAuditAgent(Agent):
    """Trims repetitive or over-resolved prose before spoken delivery."""

    schema_name = "style_audit"
    response_model = StyleAuditResponse
    instructions = style_audit_instructions()

    def build_payload(
        self,
        episode_number: int,
        title: str,
        sections: list[dict],
        host_policy: dict | None = None,
        series_explanation_registry: list[dict] | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "title": title,
            "sections": sections,
        }
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if series_explanation_registry is not None:
            payload["series_explanation_registry"] = series_explanation_registry
        return payload
