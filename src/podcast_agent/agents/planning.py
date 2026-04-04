"""Stage 9: Episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodePlan


class EpisodePlanningAgent(Agent):
    """Produces a detailed structural plan for a single episode assignment."""

    schema_name = "episode_planning"
    response_model = EpisodePlan
    instructions = (
        "You are a podcast episode planner. Produce a narrative-first episode plan for exactly one episode.\n\n"
        "Constraints you must satisfy:\n"
        "1. Keep coherence to the assignment's axis_ids and insight_ids.\n"
        "2. Target runtime must be 75-90 minutes.\n"
        "3. Include 30-36 beats, each designed for 2-3 minutes.\n"
        "4. Include at least one cross-reference to previous episodes when available.\n\n"
        "Plan episodes around a NarrativeSpine, not around author-by-author comparison. "
        "The listener should feel like they are hearing history unfold, not a book review. "
        "Organize beats around what happened and why it matters.\n\n"
        "Each EpisodeBeat must include:\n"
        "- narrative_instruction: set_the_scene, advance_events, explain_context, "
        "build_tension, reveal_consequence, or pivot_to_new_thread\n"
        "- attribution_level: none (default, majority), light (rare), full (very rare)\n"
        "Attribution budget: no more than 20% of beats should be light or full. "
        "If you exceed the budget, merge or cut attribution moments.\n\n"
        "Narrative spine rules:\n"
        "- Provide enough spine segments to support the beat count and 75-90 minute target\n"
        "- Include 3-5 attribution moments per episode\n"
        "- Spine segments must not include author names\n\n"
        "available_passages entries include mixed source detail:\n"
        "- insight-linked entries include full_text\n"
        "- supporting entries include summary_text\n"
        "Use full_text when available for high-fidelity beat planning, and use summary_text for "
        "supporting context and cross-axis comparisons.\n\n"
        "Assign specific passage_ids from the thematic corpus to each beat. "
        "Use primary_book_id to indicate the best source material, not an author lead.\n\n"
        "Return a JSON object matching the EpisodePlan schema."
    )

    def build_payload(
        self,
        episode_assignment: dict,
        narrative_strategy: dict,
        synthesis_map: dict,
        project_metadata: dict,
        available_passages: dict,
        previous_episode: dict | None,
        next_episode: dict | None,
    ) -> dict:
        return {
            "episode_assignment": episode_assignment,
            "narrative_strategy": narrative_strategy,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "available_passages": available_passages,
            "previous_episode": previous_episode,
            "next_episode": next_episode,
        }
