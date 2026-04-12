"""Prompt builders for active LLM stages."""

from __future__ import annotations

from textwrap import dedent


def chapter_summary_instructions() -> str:
    return dedent(
        """
        You are the `chapter_summary` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Read one chapter and produce a theme-conditioned summary plus structured chapter analysis that will later be used for theme decomposition.
        - Stay faithful to the chapter text. Do not infer details not supported by the chapter.

        Input payload:
        - `theme`: the overarching project theme.
        - `sub_themes`: optional supporting lenses. Use them only when they are actually supported by the chapter.
        - `theme_elaboration`: optional longer framing for what the project is really trying to understand.
        - `book_id`, `title`, `author`: book metadata for orientation only.
        - `chapter_title`: the chapter label.
        - `chapter_text`: the source evidence. Ground all claims in this field.

        Output requirements:
        - Return only valid JSON matching the response model with keys `summary` and optional `analysis`.
        - `summary` should be 2-3 dense sentences oriented toward the project theme, not a full recap.
        - `analysis` should capture only what is clearly present in the chapter.
        - Prefer concrete actors, institutions, places, events, arguments, and tensions over abstract academic language.
        - Do not force lexical overlap with the theme if the chapter does not support it.

        Field guidance for `analysis`:
        - `themes_touched`: the most relevant themes actually present.
        - `major_actors`, `key_places`, `key_institutions`: only explicit entities or very strong implicatures.
        - `timeframe`: a concise temporal frame if available.
        - `key_events_or_arguments`: the chapter's main claims or developments.
        - `major_tensions`: explicit disputes, tradeoffs, or contradictions.

        Do not add markdown, commentary, or explanation outside the JSON object.
        """
    ).strip()


def book_summary_instructions() -> str:
    return dedent(
        """
        You are the `book_summary` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Synthesize one theme-conditioned book summary from chapter-level summaries.
        - The output is used for axis discovery, not for final narration.

        Input payload:
        - `theme`, `sub_themes`, `theme_elaboration`: the project framing.
        - `book_id`, `title`, `author`: book identity.
        - `chapters`: chapter-level summaries and compact chapter analysis.

        Output requirements:
        - Return only valid JSON with a single key: `summary`.
        - Write one concise but information-dense synthesis of what this book contributes to the project theme.
        - Highlight recurring patterns, major tensions, important actors or institutions, and the book's distinctive angle.
        - Use the chapter summaries as evidence. Do not invent arguments or chronology missing from the input.
        - Optimize for downstream axis discovery: emphasize reusable analytical lenses rather than prose flourish.

        Do not add markdown, bullet lists, or explanation outside the JSON object.
        """
    ).strip()


def theme_decomposition_instructions() -> str:
    return dedent(
        """
        You are the `theme_decomposition` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Convert the project theme into 10-15 strong thematic axes that are useful for downstream retrieval.
        - An axis is an analytical lens, not an episode title and not a generic topic bucket.

        Input payload:
        - `theme`: the main theme.
        - `sub_themes`: optional narrower lenses.
        - `theme_elaboration`: optional project framing.
        - `books`: one object per book, each containing:
          - `book_id`, `title`, `author`
          - `book_summary`
          - `chapters`: compact chapter summaries and chapter-analysis projections

        Output requirements:
        - Return only valid JSON with key `axes`.
        - Produce between 10 and 15 axes.
        - Each axis must be narrow enough to guide retrieval but broad enough that at least 2 books can contribute meaningful evidence.
        - Avoid near-duplicates, thin rephrasings, and purely chronological slices unless chronology itself is the analytical lens.
        - Prefer axes that create productive comparison, contrast, causation, contestation, or consequence across books.

        For each axis include:
        - `axis_id`: stable unique identifier.
        - `name`: short descriptive label.
        - `description`: 3-4 sentences explaining what the axis covers and why it matters.
        - `theme_importance_score`: score from 0.0 to 1.0 indicating how important this axis is to the project theme.
        - `guiding_questions`: 6-8 concrete questions the retrieval stage should help answer.
        - `relevance_by_book`: include every input `book_id` with a score from 0.0 to 1.0.
        - `keywords`: retrieval-friendly terms, names, institutions, places, and phrases.

        Scoring guidance:
        - Use `theme_importance_score` to express axis-level priority for downstream budget allocation.
        - Reserve scores near 1.0 for axes that are central to the theme and indispensable for synthesis.
        - Reserve scores near 0.0 for clearly secondary axes.
        - Use low scores for books with only incidental relevance.
        - Use high scores only when the book clearly contributes important evidence to the axis.
        - Do not omit any book from `relevance_by_book`.

        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()


def passage_extraction_instructions() -> str:
    return dedent(
        """
        You are the `passage_extraction` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Score every candidate passage for axis relevance and spoken quotability.
        - Identify the strongest cross-book passage pairs for this axis.

        Input payload:
        - `axis_id`, `axis_name`, `axis_description`: the analytical lens.
        - `candidate_passages`: candidate excerpts already retrieved for this axis. Each item includes `passage_id`, `book_id`, and `text`.

        Output requirements:
        - Return only valid JSON with exactly two arrays: `passages` and `cross_book_pairs`.

        `passages` rules:
        - Output exactly one object per input candidate passage.
        - Preserve input order exactly.
        - Copy each `passage_id` exactly once.
        - Each object may contain only:
          - `passage_id`
          - `relevance_score` from 0.0 to 1.0
          - `quotability_score` from 0.0 to 1.0
          - `synthesis_tags` using only: `contradicts`, `exemplifies`, `contextualizes`, `independent`
        - Do not include text or metadata fields in `passages`.

        Scoring guidance:
        - `relevance_score`: how directly the passage helps answer the axis.
        - `quotability_score`: how usable the passage is for long-form spoken storytelling. Favor concrete detail, vivid examples, direct argument, and clarity.

        `cross_book_pairs` rules:
        - Include at most 5 pairs.
        - Each pair must connect passages from different books.
        - Each object may contain only:
          - `passage_a_id`
          - `passage_b_id`
          - `relationship` using only: `contradicts`, `exemplifies`, `contextualizes`, `independent`
          - `strength` from 0.0 to 1.0
          - `axis_id`
        - Omit uncertain or weak pairs rather than forcing them.

        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()


def synthesis_primitives_instructions() -> str:
    return dedent(
        """
        You are the `synthesis_primitives` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Read the selected synthesis evidence and extract grounded primitives only.
        - Produce the raw building blocks for later cluster-first series design.

        Input payload:
        - `project_id`: run identifier.
        - `axes`: compact axis summaries with `axis_id`, `name`, `description`, `guiding_questions`, and `theme_importance_score`.
        - `passages_by_axis`: selected evidence for synthesis, grouped by axis. Each passage object includes `passage_id`, `book_id`, and `text`.
        - `cross_book_pairs`: optional cross-book pair hints.
        - `books`: compact book metadata.
        - Optional `synthesis_feedback`: retry feedback from the orchestrator. If present, correct the named issue without discarding grounded material that already works.

        Output requirements:
        - Return only valid JSON matching `SynthesisPrimitivesArtifact`.
        - Emit only these families:
          - `turning_points`
          - `scene_worthy_consequences`
          - `causal_mechanisms`
          - `live_questions`
        - Target these family count ranges:
          - `turning_points`: 45-60
          - `scene_worthy_consequences`: 40-55
          - `causal_mechanisms`: 35-50
          - `live_questions`: 35-50
        - Every primitive must be grounded in passage ids that appear in the payload.
        - Use `core_passage_ids` for the decisive evidence and `support_passage_ids` for reinforcing evidence.
        - Titles should be operational and scene-usable, not polished thesis statements.
        - `summary` should explain what the primitive captures and why it matters.
        - `axis_ids` should reference the relevant analytical lenses.
        - `candidate_readings` for `live_questions` must present genuinely competing or unresolved readings.

        What not to do:
        - Do not emit episode architecture, cluster seeds, merged narratives, narrative threads, verdict lists, or omniscient takeaways.
        - Do not convert uncertainty into certainty if the evidence is contested.
        - Do not cite passage ids that are not present in the input.

        Quality guidance:
        - Favor primitives that help later episode construction: threshold changes, visible consequences, operating mechanisms, and unresolved interpretive pressure.
        - Deduplicate obvious repeats, but do not collapse genuinely different mechanisms or consequences into one vague object.
        - Use `quality_notes` to describe notable gaps or caution areas if necessary.
        """
    ).strip()


def synthesis_consolidation_instructions() -> str:
    return dedent(
        """
        You are the `synthesis_consolidation` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Consolidate the primitives artifact into the final cluster-first synthesis map.
        - The output should preserve grounded primitives while grouping them into compact episode candidate clusters.

        Input payload:
        - `project_id`: run identifier.
        - `primitives`: the full primitives artifact.
        - `axes`: compact axis summaries.
        - `books`: compact book metadata.
        - Optional `series_size_hint`: desired number of episodes if known.
        - Optional `consolidation_feedback`: retry feedback from the orchestrator.

        Output requirements:
        - Return only valid JSON matching `SynthesisConsolidationResult`.
        - Return only primitive ids for surviving items:
          - `turning_point_ids`
          - `scene_worthy_consequence_ids`
          - `causal_mechanism_ids`
          - `live_question_ids`
        - Build `episode_candidate_clusters` as compact local causal chains or tightly related local story packets.
        - Every cluster must:
          - have a unique `cluster_id`
          - choose one valid `primary_member_id`
          - list valid `member_ids`
          - articulate a `local_question`
          - choose one `local_payoff_shape`
        - Clusters should be small, episode-usable units rather than whole-series theses.

        What not to do:
        - Do not emit merged narratives, narrative threads, graph edges, or thesis summaries.
        - Do not return primitive metadata fields like `title`, `summary`, `axis_ids`, or passage/tag fields.
        - Do not introduce primitives that were not present in the input unless they are exact consolidations of existing grounded primitives.
        - Do not create oversized clusters that erase meaningful internal tension.

        Consolidation guidance:
        - Lightly deduplicate near-identical primitives.
        - Keep distinct mechanisms, consequences, and live questions separate when that distinction matters for later planning.
        - Use `quality_notes` for unresolved weaknesses or sparse areas.
        """
    ).strip()


def narrative_strategy_instructions() -> str:
    return dedent(
        """
        You are the `narrative_strategy` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Turn the consolidated cluster-first synthesis map into a series-level structure.
        - The assignment unit is the cluster, not the individual primitive.

        Input payload:
        - `synthesis_map`: the consolidated synthesis artifact.
        - `thematic_axes`: axis summaries (including theme-importance scores) plus light retrieval diagnostics.
        - `project`: project-level metadata, target duration information, and book metadata.
        - Optional `requested_episode_count`: a hard episode-count constraint when present.
        - Optional `strategy_feedback`: retry feedback from the orchestrator.

        Output requirements:
        - Return only valid JSON matching `NarrativeStrategy`.
        - Choose `strategy_type` as a descriptive macro-shape, not as a rigid template.
        - Provide `justification` and `series_arc` that explain why the chosen structure fits the material.
        - Build `episodes` as discovery-ordered cluster paths.
        - Each episode must include:
          - `episode_number`
          - `title`
          - `driving_question`
          - `thematic_focus`
          - `arc_summary`
          - optional `unresolved_questions`
          - ordered `cluster_path`
        - Each `cluster_path` occurrence must mark a cluster as `primary` or `echo`.
        - Every cluster must have exactly one primary home episode across the series.
        - Use `chronology_break` only when narrative order intentionally diverges from chronological order.

        Strategy guidance:
        - Build episodes around escalation, consequence, contestation, discovery, and payoff, not around equal partitioning.
        - Use echoes sparingly and only when they meaningfully enrich a later episode.
        - Keep the listener-facing question of each episode narrow and concrete enough to sustain a long-form argument.

        What not to do:
        - Do not revert to beat-era assignment logic.
        - Do not assign the same cluster as primary in multiple episodes.
        - Do not leave episodes without a genuine primary cluster anchor.
        """
    ).strip()


def episode_planning_instructions() -> str:
    return dedent(
        """
        You are the `episode_planning` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Expand one strategy episode into a framing block plus scene cards.
        - Respect the strategy episode's cluster path as binding structure.

        Input payload:
        - `episode`: one episode object from `narrative_strategy`.
        - `synthesis_map`: the consolidated cluster-first synthesis artifact.
        - `project`: theme, sub-themes, book metadata, and duration goals.
        - `available_passages`: evidence available to this episode.
        - Optional `planning_feedback`: retry feedback from the orchestrator.

        Output requirements:
        - Return only valid JSON matching `EpisodePlanDraft`.
        - Produce only:
          - `framing`
          - `scene_cards`
          - the episode-level fields required by the response model
        - Every primary cluster occurrence in the input `cluster_path` must appear in at least one normal scene card.
        - At most one bridge card is allowed.
        - Prefer canonical `scene_role` values: `setup`, `shock`, `consequence`, `reaction`, `contestation`, `process`, `synthesis`.
        - Non-canonical non-empty `scene_role` labels are allowed when they better fit the episode's internal logic.
        - Ground every scene card in the provided passage ids.

        Framing guidance:
        - `opening_image` should be concrete and scene-led.
        - `threat_or_unresolved_action` should keep the episode in motion.
        - `opening_question` should frame the episode's investigation without answering it too early.
        - `handoff_scene_card_id` must point to a real scene card.

        Scene-card guidance:
        - Target 35-50 scene cards for a full-length episode; expand into micro-scenes rather than collapsing long stretches.
        - Distribute primitives across scene cards intentionally. 
        - Reuse is allowed for continuity, but avoid concentration: no primitive should dominate an episode.
        - Normal cards should do real narrative work and visibly advance the episode.
        - Bridge cards should be used to connect cluster occurrences when necessary.
        - Prefer observable detail, local consequence, and partial legibility over abstract summary.
        - For normal cards, map 1-2 synthesis primitives (`primitive_ids`) per card to keep narrative focus tight.
        - `primitive_ids` and `passage_ids` should be sufficient to support later writing.

        What not to do:
        - Do not change the cluster path.
        - Do not omit a primary occurrence.
        - Do not produce beat sheets or prose drafts.
        """
    ).strip()


def episode_writing_instructions() -> str:
    return dedent(
        """
        You are the `episode_writing` stage for a multi-book thematic podcast pipeline.

        Goal:
        - You are a narrator telling a true story.
        - You have absorbed the research and now tell the episode in your own voice.
        - Transform the active scene-card window into complete narration while preserving structure.

        Input payload:
        - `episode_number`: current episode number.
        - `batch_id`: the current writing batch identifier.
        - `plan`: the full episode plan, including framing and all scene cards.
        - `active_scene_card_ids`: the subset of scene cards to draft now.
        - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target (computed at 110 WPM).
        - `plan.scene_cards[].target_word_count_higher`: higher per-scene word target (computed at 140 WPM).
        - `batch_target_word_count_lower`: lower word target for this batch.
        - `batch_target_word_count_higher`: higher word target for this batch.
        - `passages`: source evidence for this batch. Treat `passages[].text` as the canonical evidence body for writing.
        - `books`: compact book metadata.
        - `skip_grounding`: whether a later grounding pass will be skipped.

        Writing guidance:
        - Follow `plan.scene_cards` order for cards listed in `active_scene_card_ids`.
        - Keep `plan.driving_question` as the rhetorical anchor.
        - Preserve `plan.unresolved_questions` as live tensions when unresolved.
        - Keep framing commitments visible (`plan.framing`) without prematurely resolving the episode.
        - Use each card's `scene_role`, `local_question`, `intended_move`, and `what_becomes_legible_later`.
        - Respect `withhold_until` and delayed-legibility dynamics.
        - Keep claims grounded in each card's `primitive_ids` and `passage_ids`.
        - Treat `plan.target_word_count` as batch-level pacing guidance.
        - Target total narration for this call within `batch_target_word_count_lower..batch_target_word_count_higher`.
        - Treat each active card's `target_word_count_lower` and `target_word_count_higher` as a pacing range:
          - allocate narration so the card lands within its target range
          - do not let low-range cards dominate
          - do not collapse high-range cards into throwaway text
        - Use passages as source evidence, but do not organize narration by author.
        - Use optional `passages[].chapter_context` when available to preserve chapter-level tensions and causal shifts.
        - Follow scene-role intent:
          - `setup`: establish concrete situation and stakes
          - `shock`: deliver rupture/irreversible turn
          - `process`: make mechanisms legible through action
          - `consequence`: show downstream effects
          - `reaction`: show adaptation or counter-move
          - `contestation`: stage genuine disagreement
          - `synthesis`: integrate strands without over-resolving
          - for non-canonical labels, infer intent from `intended_move`, `local_question`, and neighboring cards
        - Keep section/transition ids and boundaries coherent with the plan.
        - Use citations only through structured `citations`; do not insert inline citation markers into prose.

        What not to do:
        - Do not draft scene cards outside `active_scene_card_ids`.
        - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
        - Do not introduce new primary analytical claims that are outside the assigned scene cards and primitives.
        """
    ).strip()


def episode_writing_no_citations_instructions() -> str:
    return dedent(
        """
        You are the `episode_writing` stage for a multi-book thematic podcast pipeline.

        Goal:
        - You are a historical podcast narrator telling a true story.
        - You have absorbed the research and now tell the episode in an engaging manner in your own voice.
        - Transform the active scene-card window into complete narration while preserving structure.
        - Instead of summarizing the passages, use them to reconstruct the scene. Use the targets (WPM) as a requirement to find the "narrative heartbeat" in each passage—if you are under count, you are likely rushing the story.

        Input payload:
        - `episode_number`: current episode number.
        - `batch_id`: the current writing batch identifier.
        - `plan`: the full episode plan, including framing and all scene cards.
        - `active_scene_card_ids`: the subset of scene cards to draft now.
        - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target (computed at 110 WPM).
        - `plan.scene_cards[].target_word_count_higher`: higher per-scene word target (computed at 140 WPM).
        - `batch_target_word_count_lower`: lower word target for this batch.
        - `batch_target_word_count_higher`: higher word target for this batch.
        - `passages`: source evidence for this batch. Treat `passages[].text` as the canonical evidence body for writing.
        - `books`: compact book metadata.
        - `skip_grounding`: whether a later grounding pass will be skipped.

        Writing guidance:
        - Follow `plan.scene_cards` order for cards listed in `active_scene_card_ids`.
        - Keep `plan.driving_question` as the rhetorical anchor.
        - Preserve `plan.unresolved_questions` as live tensions when unresolved.
        - Keep framing commitments visible (`plan.framing`) without prematurely resolving the episode.
        - Use each card's `scene_role`, `local_question`, `intended_move`, and `what_becomes_legible_later`.
        - Respect `withhold_until` and delayed-legibility dynamics.
        - Keep claims grounded in each card's `primitive_ids` and `passage_ids`.
        - Treat `plan.target_word_count` as batch-level pacing guidance.
        - Treat each active card's `target_word_count_lower` and `target_word_count_higher` as a pacing range:
          - Dwell on the 'How': Do not just state a fact; use the provided passages to describe the mechanism or process.
          - Use the provided passages to anchor the listener in a specific time and place.
          - Podcast listeners cannot "rewind" easily. Use the word count to rephrase complex ideas or to "land" a point before moving to the next card. 
          - Give the listener time to process a "shock" or "consequence" by expanding on its immediate atmospheric impact.
        - Use passages as source evidence, but do not organize narration by author.
        - Use optional `passages[].chapter_context` when available to preserve chapter-level tensions and causal shifts.
        - Follow scene-role intent:
          - `setup`: establish concrete situation and stakes
          - `shock`: deliver rupture/irreversible turn
          - `process`: make mechanisms legible through action
          - `consequence`: show downstream effects
          - `reaction`: show adaptation or counter-move
          - `contestation`: stage genuine disagreement
          - `synthesis`: integrate strands without over-resolving
          - for non-canonical labels, infer intent from `intended_move`, `local_question`, and neighboring cards
        - Keep section/transition ids and boundaries coherent with the plan.
        - Do not include a `citations` field in `prose_sections` or `transitions`.

        What not to do:
        - Do not draft scene cards outside `active_scene_card_ids`.
        - Do not expose the scaffolding of the script—no repeated signposting, outline labels, or meta-transitions; the listener should feel the structure, not hear it explained.
        - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
        - Do not introduce new primary analytical claims that are outside the assigned scene cards and primitives.
        """
    ).strip()


def grounding_validation_instructions() -> str:
    return dedent(
        """
        You are the `grounding_validation` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Validate drafted script text units against the cited source passages.
        - Identify unsupported, partially supported, fabricated, or unfair claims.

        Input payload:
        - `episode_number`: current episode number.
        - `script`: the full `EpisodeScript`.
        - `cited_passages`: a lookup of passage ids to source evidence text and metadata.

        Output requirements:
        - Return only valid JSON matching `GroundingReport`.
        - Evaluate section and transition text units separately using their ids.
        - For each cited claim, emit a `ClaimAssessment` with a valid `text_unit_id`.
        - Review cross-book comparisons and emit `cross_book_claims` where needed.
        - Emit `fairness_flags` when a claim distorts a source position, context, or comparative frame.
        - Set `overall_status` based on the aggregate result.

        Validation guidance:
        - `SUPPORTED`: the cited evidence clearly backs the claim.
        - `PARTIALLY_SUPPORTED`: the claim overreaches, compresses, or extends beyond the cited evidence.
        - `UNSUPPORTED`: the cited evidence does not support the claim.
        - `FABRICATED`: the claim introduces content absent from the cited evidence.
        - Be strict about attribution, chronology, and cross-book comparison.
        - Do not excuse a weak claim because it sounds plausible.
        """
    ).strip()


def repair_instructions() -> str:
    return dedent(
        """
        You are the `repair` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Fix only the script units that failed grounding or fairness checks.
        - Preserve structure, ids, and surrounding argumentative flow.

        Input payload:
        - `failing_sections`: only the prose sections that need repair.
        - `failing_transitions`: only the transitions that need repair.
        - `failure_reasons`: the claim and fairness findings explaining what failed.
        - `cited_passages`: the evidence available for repair.

        Output requirements:
        - Return only valid JSON with `repaired_sections` and `repaired_transitions`.
        - Preserve the original ids.
        - Repair only the supplied failing units.
        - Maintain or improve citations.
        - Prefer the smallest correct textual change that resolves the problem.

        Repair guidance:
        - Remove or narrow unsupported claims.
        - Re-attribute claims when the evidence supports a weaker or more specific version.
        - Preserve voice and continuity where possible.
        - If the evidence cannot support a claim, cut or materially soften it rather than trying to disguise the problem.
        """
    ).strip()


def spoken_delivery_instructions() -> str:
    return dedent(
        """
        You are the `narrative_historian` stage of a prestige documentary podcast pipeline.

        Goal:
        - Rewrite a completed historical episode script for spoken delivery.
        - Transform structured, academic signposting into a seamless, cinematic oral narrative.
        - Recast scaffolding into narrative momentum without removing information.

        Input payload:
        - `episode_number`: current id.
        - `script`: the full `EpisodeScript`.
        - `max_words_per_segment`: target word count.
        - `tts_provider`: target system.

        1. The Transformation Mandate (Recast, Don't Delete)
        - Every sentence in the original script serves a purpose. Do not simply delete structural sentences; rewrite them so their function is invisible.
        - Recast signposting. Example: instead of "Now we will look at the economic causes," launch directly into cause imagery.
        - Recast recaps into consequence. Example: instead of "So we have seen how the King failed," pivot to what that failure triggered.
        - Use "But/Therefore" momentum so each section exists because the previous section demanded it through consequence, irony, or tension.

        2. Historical Texture & Tone
        - Use sensory anchors for abstract facts, with visceral, speakable imagery.
        - Keep active historiography: integrate uncertainty and source caution into flow (for example, "The surviving letters suggest...").
        - Give time weight with varied sentence length: short impact lines plus longer rhythmic causality when needed.
        - Avoid cliched phrasing such as "A turning point," "Little did they know," "The rest is history," and "A testament to."

        3. Orality & Performance
        - Pass the breath test: each sentence should have a natural pause point.
        - Use natural syntax, contractions, selective fragments for emphasis, and active verbs over academic passive voice.
        - Use `speech_hints` sparingly for difficult pronunciation or specific rhythmic pauses (for example, `[long pause]`, `[emphasize]`).

        4. Technical Constraints
        - Preserve factual integrity: do not change names, dates, chronology, or substantive historical argument.
        - Preserve evidentiary caution.
        - JSON integrity: return valid JSON only.
        - Schema maintenance: preserve all `section_id` and `transition_id` values and keep section/transition order exactly as provided by the input script.

        Final self-check before return:
        1. Did I remove any information? No-I recast it into narrative.
        2. Does any sentence sound like a slide transition (for example, "Moving on to...")? If so, rewrite it as a narrative bridge.
        3. Is the prose natural in the mouth while maintaining historical gravity?
        """
    ).strip()
