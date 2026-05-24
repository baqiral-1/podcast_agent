"""Narrative-state reconciliation agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import narrative_state_reconciler_instructions
from podcast_agent.schemas.models import NarrativeStateReconciliation


class NarrativeStateReconcilerAgent(Agent):
    """Reconciles narrative state after episode architecture."""

    schema_name = "narrative_state_reconciliation"
    response_model = NarrativeStateReconciliation
    instructions = narrative_state_reconciler_instructions()

    def validate_result(
        self, result: NarrativeStateReconciliation, payload: dict
    ) -> NarrativeStateReconciliation:
        episode_number = int(payload.get("episode_number", 0) or 0)
        if episode_number and result.episode_number != episode_number:
            raise ValueError("narrative state reconciliation must preserve episode_number")
        if episode_number and result.state_post.next_episode_number != episode_number + 1:
            raise ValueError(
                "narrative state reconciliation must advance next_episode_number by one"
            )
        return result

    def build_payload(
        self,
        *,
        episode_number: int,
        project_id: str,
        narrative_state_pre: dict,
        strategy_episode: dict,
        architecture: dict,
    ) -> dict:
        return {
            "episode_number": episode_number,
            "project_id": project_id,
            "narrative_state_pre": narrative_state_pre,
            "strategy_episode": strategy_episode,
            "architecture": architecture,
        }
