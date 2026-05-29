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
        narrative_state_pre: dict | None = None,
        narrative_state_post: dict | None = None,
        continuity_contract_pre: dict | None = None,
        continuity_contract_post: dict | None = None,
        series_explanation_registry: list[dict] | None = None,
        field_semantics: dict | None = None,
        lint_flags: dict | None = None,
        quality_judgment: dict | None = None,
        series_carryover_counts: dict[str, int] | None = None,
    ) -> dict:
        payload = {
            "episode_number": episode_number,
            "title": title,
            "sections": sections,
        }
        if lint_flags is not None:
            payload["lint_flags"] = lint_flags
        if quality_judgment is not None:
            payload["quality_judgment"] = quality_judgment
        if series_carryover_counts is not None:
            payload["series_carryover_counts"] = series_carryover_counts
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if narrative_state_pre is not None:
            payload["narrative_state_pre"] = narrative_state_pre
        if narrative_state_post is not None:
            payload["narrative_state_post"] = narrative_state_post
        if continuity_contract_pre is not None:
            payload["continuity_contract_pre"] = continuity_contract_pre
        if continuity_contract_post is not None:
            payload["continuity_contract_post"] = continuity_contract_post
        if series_explanation_registry is not None:
            payload["series_explanation_registry"] = series_explanation_registry
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        return payload
