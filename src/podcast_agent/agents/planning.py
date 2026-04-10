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
        "1. Keep coherence to the assignment's axes and insight_ids.\n"
        "2. Materially realize every assigned insight in the scene and beat plan using that insight's passage_ids.\n"
        "3. Materially realize each assigned merged narrative in the scene and beat plan using its source_passage_ids.\n"
        "4. Target runtime is 140 minutes with a hard planning floor of 125 minutes.\n"
        "5. Include 70-80 beats, each designed to sustain long-form pacing. Do not exceed 80 beats.\n"
        "6. Include at least one cross-reference to previous episodes when available.\n\n"
        "Plan episodes around a NarrativeSpine realized through scene cards, not around author-by-author comparison. "
        "The listener should feel like they are hearing history unfold, not a book review. "
        "Scene cards are the primary structural units. Beats are pacing and grounding units inside scenes. "
        "Organize beats around what happened and why it matters without turning beats into mini-script paragraphs.\n\n"
        "Treat episode_assignment.driving_question as the binding rhetorical anchor for the episode. "
        "The opening scenes and beats should dramatize or pose it, the middle scenes and beats should narrow, test, or "
        "complicate it, and the ending should answer, partially answer, or deepen it in the shape "
        "required by narrative_strategy.episode_arc_detail.payoff_shape.\n\n"
        "Use the selected synthesis context actively:\n"
        "- Treat narrative_strategy.episode_arc_detail as binding arc architecture\n"
        "- Use narrative_strategy.episode_arc_detail.episode_inquiries as secondary questions to"
        " help contextualize and structure scenes and beats\n"
        "- Use the assigned insights to shape the core episode argument\n"
        "- Use selected merged narratives to anchor long-arc synthesis and payoff\n"
        "- Use narrative_strategy.episode_arc_detail.unresolved_questions and selected unresolved "
        "tensions to shape open questions, pivots, or endings\n\n"
        "Narrative spine rules:\n"
        "- Provide enough spine segments to support the beat count and 140-minute target\n"
        "- Include 3-5 attribution moments per episode\n"
        "- Spine segments must not include author names\n"
        "- Each spine segment must include spine_segment_id, title, and summary\n\n"
        "Scene card rules:\n"
        "- Produce 12-18 scene_cards\n"
        "- Every scene_card must belong to exactly one narrative spine segment via spine_segment_id\n"
        "- A spine segment may own multiple scene_cards\n"
        "- Scene cards should capture dramatic situations, causal turns, or analytical turns that can sustain multiple beats\n"
        "- Keep scene_cards concise, structural, and scene-led rather than prose-heavy\n"
        "- Produce anchor_scene_ids as a list of 3-6 unique scene_ids from the scene_cards\n"
        "- Every anchor_scene_id must exist in scene_cards and should mark a load-bearing opening, reversal, escalation, or payoff scene\n\n"
        "Each SceneCard must include:\n"
        "- scene_id: stable unique id\n"
        "- spine_segment_id: id of the narrative spine segment this scene realizes\n"
        "- title: short scene label\n"
        "- narrative_purpose: one-line statement of what the scene does for the episode\n"
        "- timeframe: concise temporal marker\n"
        "- location: concise place marker\n"
        "- actors: optional list of up to 4 objects with {name, role_in_scene, affiliation?}\n"
        "- entry_image: concrete opening image or moment, 1-2 lines\n"
        "- exit_turn: what changes, sharpens, or becomes newly legible by the end of the scene, 1-2 lines\n"
        "- insight_ids: explicit list of assigned insight IDs materially advanced by this scene\n"
        "- passage_ids: explicit grounding passage IDs for this scene\n"
        "- estimated_duration_seconds: scene budget\n\n"
        "Each EpisodeBeat must include:\n"
        "- scene_id: the scene this beat belongs to\n"
        "- narrative_instruction: set_the_scene, advance_events, explain_context, "
        "build_tension, reveal_consequence, or pivot_to_new_thread\n"
        "- attribution_level: none (default, majority), light (rare), full (very rare)\n"
        "- insight_ids: explicit list of assigned insight IDs materially advanced by this beat\n"
        "- passage_ids: explicit grounding passage IDs for this beat\n"
        "- primary_book_id and supporting_book_ids for source alignment\n"
        "- estimated_duration_seconds for runtime pacing\n"
        "- If beat.description names an insight like ins_38, include it in that beat's insight_ids\n"
        "- Beats must appear in scene order, and all beats for a scene must form one contiguous block\n"
        "Attribution budget: no more than 20% of beats should be light or full. "
        "If you exceed the budget, merge or cut attribution moments.\n\n"
        "available_passages is a flat list.\n"
        "Each entry contains either summary_text (non-insight support passages) or full_text "
        "(passages linked to assigned insights).\n"
        "Use full_text entries for assigned-insight realization and use summary_text entries for "
        "supporting context and cross-axis comparisons.\n\n"
        "For merged narrative realization, use synthesis_map.merged_narratives[*].source_passage_ids "
        "as the grounding set for that episode's assigned merged_narrative_id.\n\n"
        "Set book_balance using project.book_size_share_by_id as a starting prior, then adjust "
        "based on assigned axes, selected insight evidence, and narrative needs.\n\n"
        "Assign specific passage_ids from the thematic corpus to scene_cards and beats. "
        "Use primary_book_id to indicate the best source material, not an author lead.\n\n"
        "If planning_feedback is provided, treat it as a correction request and fix the cited "
        "insight-realization gaps in the revised plan.\n\n"
        "Return a JSON object matching the EpisodePlanDraft schema, including scene_cards, anchor_scene_ids, and beat-level scene_id."
    )

    def build_payload(
        self,
        episode_assignment: dict,
        narrative_strategy: dict,
        synthesis_map: dict,
        project_metadata: dict,
        available_passages: list[dict],
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
            "previous_episode": previous_episode,
            "next_episode": next_episode,
        }
        if planning_feedback is not None:
            payload["planning_feedback"] = planning_feedback
        return payload
