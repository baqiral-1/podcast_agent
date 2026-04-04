"""Stage 9: Series planning agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodePlan


class SeriesPlanResponse(BaseModel):
    episodes: list[EpisodePlan] = Field(default_factory=list)


class SeriesPlanningAgent(Agent):
    """Produces detailed episode-by-episode plans with assigned insights and beats."""

    schema_name = "series_planning"
    response_model = SeriesPlanResponse
    instructions = (
        "You are a podcast series planner. Produce a narrative-first episode plan.\n\n"
        "Constraints you must satisfy:\n"
        "1. Every SynthesisInsight with podcast_potential > 0.5 must appear in at least one episode.\n"
        "2. Book balance: across the series, no book gets less than (1/N × 0.5) share of "
        "assigned passages, unless book_weights overrides this.\n"
        "3. Per-episode coherence: each episode focuses on 1-2 axes and 2-4 insights. "
        "Don't try to cover everything in one episode.\n"
        "4. Narrative arc: episode ordering follows the strategy's arc type. Early episodes "
        "set up, middle episodes develop, final episodes resolve or synthesize.\n"
        "5. Cross-references: every episode after the first includes at least one CrossReference "
        "connecting to a previous episode.\n\n"
        "Plan episodes around a NarrativeSpine, not around author-by-author comparison. "
        "The listener should feel like they are hearing history unfold, not a book review. "
        "Organize beats around what happened and why it matters.\n\n"
        "Each EpisodeBeat must include:\n"
        "- narrative_instruction: set_the_scene, advance_events, explain_context, "
        "build_tension, reveal_consequence, or pivot_to_new_thread\n"
        "- attribution_level: none (default, majority), light (rare), full (very rare)\n"
        "Attribution budget: no more than 20% of beats may be light or full. "
        "If you exceed the budget, merge or cut attribution moments.\n\n"
        "Narrative spine rules:\n"
        "- Provide 15-20 spine segments for a 75-100 minute episode\n"
        "- Include 3-5 attribution moments per episode\n"
        "- Spine segments must not include author names\n\n"
        "Assign specific passage_ids from the thematic corpus to each beat. "
        "Use primary_book_id to indicate the best source material, not an author lead.\n\n"
        "Return a JSON object with an 'episodes' array of EpisodePlan objects."
    )

    def build_payload(
        self,
        synthesis_map_summary: dict,
        narrative_strategy: dict,
        project_metadata: dict,
        episode_count: int,
        passages_summary: dict,
    ) -> dict:
        return {
            "synthesis_map": synthesis_map_summary,
            "narrative_strategy": narrative_strategy,
            "project": project_metadata,
            "episode_count": episode_count,
            "available_passages": passages_summary,
        }
