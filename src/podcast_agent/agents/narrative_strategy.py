"""Stage 8: Narrative strategy agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import NarrativeStrategy, SynthesisMap


class NarrativeStrategyAgent(Agent):
    """Chooses the macro-level creative structure for the podcast series."""

    schema_name = "narrative_strategy"
    response_model = NarrativeStrategy
    instructions = (
        "You are a podcast series architect. Based on the synthesis map of cross-book "
        "insights, choose the best narrative strategy for the series.\n\n"
        "Available strategies:\n"
        "- thesis_driven: Build a single overarching argument from all books. Best when "
        "the synthesis reveals strong convergence.\n"
        "- debate: Each episode stages a structured argument between books. Best when "
        "the map reveals strong disagreements.\n"
        "- chronological: Ideas presented in historical order, showing intellectual "
        "evolution. Best for EVOLUTION insights.\n"
        "- convergence: Starts with apparent disagreements, gradually reveals shared "
        "foundations. Best for maps with both DISAGREEMENT and AGREEMENT insights.\n"
        "- mosaic: Each episode explores a different axis independently; the final "
        "episode weaves them together. Best for maps with many independent but "
        "interesting axes.\n\n"
        "Select based on the distribution of insight types and thread arc types.\n\n"
        "Also choose a recommended_episode_count between 2 and 8 based on thematic density, "
        "number of insights, thread complexity, and narrative pacing.\n"
        "If project.requested_episode_count is provided, use it as a planning hint for arc shape.\n\n"
        "Return a JSON object with strategy_type, justification, series_arc, and "
        "episode_arc_outline (one line per episode showing planned progression), plus "
        "recommended_episode_count."
    )

    def build_payload(
        self,
        synthesis_map_summary: dict,
        project_metadata: dict,
        episode_count: int | None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map_summary,
            "project": project_metadata,
        }
        if episode_count is not None:
            payload["requested_episode_count"] = episode_count
        return payload
