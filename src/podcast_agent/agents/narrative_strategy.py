"""Stage 8: Narrative strategy agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import NarrativeStrategy


class NarrativeStrategyAgent(Agent):
    """Chooses the macro-level creative structure for the podcast series."""

    schema_name = "narrative_strategy"
    response_model = NarrativeStrategy
    instructions = (
        "You are a podcast series architect. Based on the synthesis map of cross-book "
        "insights and thematic axes, choose the best narrative strategy for the series and "
        "turn that strategy into a concrete episode assignment plan.\n\n"
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
        "Choose the strategy using:\n"
        "- distribution of insight types\n"
        "- thread arc types\n"
        "- number and cohesion of thematic axes\n"
        "- how easily the strongest insights can be shaped into a compelling "
        "multi-episode arc\n\n"
        "Then build the episode assignment plan that expresses that strategy.\n\n"
        "You must also assign each episode's axis_ids and insight_ids. "
        "These assignments are required and should follow the selected strategy arc.\n"
        "Every SynthesisInsight with podcast_potential > 0.5 must be assigned to at least one episode.\n"
        "Keep each episode coherent: usually 1-2 axes and 2-4 insights.\n\n"
        "Prefer assignments that maximize narrative progression, contrast, and payoff "
        "across the series rather than evenly distributing material.\n\n"
        "Also choose a recommended_episode_count between 2 and 8 based on thematic density, "
        "number of insights, thread complexity, and narrative pacing.\n"
        "If project.requested_episode_count is provided, use it as a planning hint for arc shape.\n\n"
        "Output schema (strict):\n"
        "- strategy_type: one of thesis_driven, debate, chronological, convergence, mosaic\n"
        "- justification: string\n"
        "- series_arc: string\n"
        "- episode_arc_outline: array of strings (one line per episode)\n"
        "- recommended_episode_count: integer between 2 and 8\n"
        "- episode_assignments: array of objects\n"
        "  each episode_assignments item must include:\n"
        "  - episode_number: integer >= 1\n"
        "  - title: string\n"
        "  - thematic_focus: string\n"
        "  - axis_ids: array of strings\n"
        "  - insight_ids: array of strings\n"
        "  - episode_strategy: string\n\n"
        "Assignment consistency rules:\n"
        "- Every SynthesisInsight with podcast_potential > 0.5 must appear in at least one "
        "episode_assignments[*].insight_ids entry.\n\n"
        "Return only a JSON object matching this schema. Do not include markdown or prose."
    )

    def build_payload(
        self,
        synthesis_map: dict,
        thematic_axes: list[dict],
        project_metadata: dict,
        episode_count: int | None,
    ) -> dict:
        payload = {
            "synthesis_map": synthesis_map,
            "thematic_axes": thematic_axes,
            "project": project_metadata,
        }
        if episode_count is not None:
            payload["requested_episode_count"] = episode_count
        return payload
