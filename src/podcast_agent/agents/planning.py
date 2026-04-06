"""Stage 9: Episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodePlanDraft


class EpisodePlanningAgent(Agent):
    """Produces a detailed structural plan for a single episode assignment."""

    schema_name = "episode_planning"
    response_model = EpisodePlanDraft
    instructions = (
        "You are a podcast episode planner. Produce a narrative-first episode plan for exactly one episode.\n\n"
        "Constraints you must satisfy:\n"
        "1. Keep coherence to the assignment's axis_ids and insight_ids.\n"
        "2. Materially realize every assigned insight in the beat plan using that insight's passage_ids.\n"
        "3. Target runtime is 140 minutes with a hard planning floor of 125 minutes.\n"
        "4. Include 50-60 beats, each designed to sustain long-form pacing.\n"
        "5. Include at least one cross-reference to previous episodes when available.\n\n"
        "Plan episodes around a NarrativeSpine, not around author-by-author comparison. "
        "The listener should feel like they are hearing history unfold, not a book review. "
        "Organize beats around what happened and why it matters.\n\n"
        "Treat episode_assignment.driving_question as the binding rhetorical anchor for the episode. "
        "The opening beats should dramatize or pose it, the middle beats should narrow, test, or "
        "complicate it, and the ending should answer, partially answer, or deepen it in the shape "
        "required by narrative_strategy.episode_arc_detail.payoff_shape.\n\n"
        "Use the selected synthesis context actively:\n"
        "- Treat narrative_strategy.episode_arc_detail as binding arc architecture\n"
        "- Use assigned insights to shape the core episode argument\n"
        "- Use selected merged narratives to anchor long-arc synthesis and payoff\n"
        "- Use narrative_strategy.episode_arc_detail.unresolved_questions and selected unresolved "
        "tensions to shape open questions, pivots, or endings\n\n"
        "Each EpisodeBeat must include:\n"
        "- narrative_instruction: set_the_scene, advance_events, explain_context, "
        "build_tension, reveal_consequence, or pivot_to_new_thread\n"
        "- attribution_level: none (default, majority), light (rare), full (very rare)\n"
        "Attribution budget: no more than 20% of beats should be light or full. "
        "If you exceed the budget, merge or cut attribution moments.\n\n"
        "Narrative spine rules:\n"
        "- Provide enough spine segments to support the beat count and 140-minute target\n"
        "- Include 3-5 attribution moments per episode\n"
        "- Spine segments must not include author names\n\n"
        "available_passages entries include summary_text only.\n"
        "Use chapter_context_by_ref[book_id][chapter_ref] when chapter-level context is needed "
        "(themes, tensions, causal shifts, hooks).\n"
        "insight_passages entries contain the full_text passages for assigned insights, even when "
        "those passages live outside the episode's assigned axes.\n"
        "Use insight_passages for assigned-insight realization, and use available_passages for "
        "supporting context and cross-axis comparisons.\n\n"
        "Set book_balance using project.book_size_share_by_id as a starting prior, then adjust "
        "based on assigned axis_ids, selected insight evidence, and narrative needs.\n\n"
        "Assign specific passage_ids from the thematic corpus to each beat. "
        "Use primary_book_id to indicate the best source material, not an author lead.\n\n"
        "If planning_feedback is provided, treat it as a correction request and fix the cited "
        "insight-realization gaps in the revised plan.\n\n"
        "Return a JSON object matching the EpisodePlanDraft schema."
    )

    def build_payload(
        self,
        episode_assignment: dict,
        narrative_strategy: dict,
        synthesis_map: dict,
        project_metadata: dict,
        available_passages: dict,
        insight_passages: list[dict] | None = None,
        chapter_context_by_ref: dict[str, dict[str, dict]] | None = None,
        previous_episode: dict | None = None,
        next_episode: dict | None = None,
        planning_feedback: dict | None = None,
    ) -> dict:
        payload = {
            "episode_assignment": episode_assignment,
            "narrative_strategy": narrative_strategy,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "available_passages": available_passages,
            "insight_passages": insight_passages or [],
            "chapter_context_by_ref": chapter_context_by_ref or {},
            "previous_episode": previous_episode,
            "next_episode": next_episode,
        }
        if planning_feedback is not None:
            payload["planning_feedback"] = planning_feedback
        return payload
