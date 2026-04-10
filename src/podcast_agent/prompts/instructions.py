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
        - `causal_shifts`: moments where conditions, incentives, or power relations change.
        - `narrative_hooks`: details that may later help drive retrieval or scene construction.
        - `retrieval_keywords`: concrete terms useful for retrieval.

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
        - `guiding_questions`: 6-8 concrete questions the retrieval stage should help answer.
        - `relevance_by_book`: include every input `book_id` with a score from 0.0 to 1.0.
        - `keywords`: retrieval-friendly terms, names, institutions, places, and phrases.

        Scoring guidance:
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
        - `axes`: compact axis summaries with `axis_id`, `name`, `description`, and `guiding_questions`.
        - `passages_by_axis`: selected evidence for synthesis, grouped by axis. Each passage object includes `passage_id`, `book_id`, `text`, `axis_id`, `relevance_score`, and `synthesis_tags`.
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
        - Return only valid JSON matching `SynthesisMap`.
        - Preserve only grounded surviving primitives.
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
        - `thematic_axes`: axis summaries plus light retrieval diagnostics.
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
        - Use only valid `scene_role` values: `setup`, `shock`, `consequence`, `reaction`, `contestation`, `process`, `synthesis`.
        - Ground every scene card in the provided passage ids.

        Framing guidance:
        - `opening_image` should be concrete and scene-led.
        - `threat_or_unresolved_action` should keep the episode in motion.
        - `opening_question` should frame the episode's investigation without answering it too early.
        - `handoff_scene_card_id` must point to a real scene card.

        Scene-card guidance:
        - Normal cards should do real narrative work and visibly advance the episode.
        - Bridge cards should be rare and only used to connect cluster occurrences when necessary.
        - Prefer observable detail, local consequence, and partial legibility over abstract summary.
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
        - Draft one writing batch for a long-form episode from the provided scene-card window.
        - Write a true story using the plan and evidence, not a generic explainer.

        Input payload:
        - `episode_number`: current episode number.
        - `batch_id`: the current writing batch identifier.
        - `plan`: the full episode plan, including framing and all scene cards.
        - `active_scene_card_ids`: the subset of scene cards to draft now.
        - `passages`: source evidence for this batch. Treat `passages[].text` as the canonical evidence body for writing.
        - `books`: compact book metadata.
        - Optional `previous_sections` and `previous_transitions`: already-written prior batches. Continue from them; do not rewrite them.
        - `skip_grounding`: whether a later grounding pass will be skipped.

        Output requirements:
        - Return only valid JSON with:
          - `batch_id`
          - `prose_sections`
          - `transitions`
          - `window_map`
        - Write only the active batch.
        - Preserve continuity with previous batches when they are supplied.
        - Each prose section must map explicitly to one or more `scene_card_ids` from the active batch.
        - Each prose section must choose one `movement_goal` from: `pose`, `discover`, `complicate`, `connect`, `judge`, `land`.
        - Use citations only through the structured `citations` field. Do not insert inline citation markers into prose.

        Writing guidance:
        - Use scene-led narration, causally legible movement, and analytical pressure that emerges from evidence.
        - Let different books interact through evidence rather than summary labels.
        - Keep framing visible, but do not prematurely resolve the episode's unanswered question.
        - If a passage is vivid and concrete, use it to anchor narration. If a passage is abstract, translate it into clear narrative prose without changing its meaning.
        - `transitions` should bridge adjacent sections without re-summarizing the whole argument.

        What not to do:
        - Do not draft scene cards outside `active_scene_card_ids`.
        - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
        - Do not introduce new primary analytical claims that are outside the assigned scene cards and primitives.
        - Do not revise prior batches when `previous_sections` or `previous_transitions` are present.
        """
    ).strip()


def episode_writing_no_citations_instructions() -> str:
    return dedent(
        """
        You are the `episode_writing` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Draft one writing batch for a long-form episode from the provided scene-card window.
        - Write a true story using the plan and evidence, not a generic explainer.

        Input payload:
        - `episode_number`: current episode number.
        - `batch_id`: the current writing batch identifier.
        - `plan`: the full episode plan, including framing and all scene cards.
        - `active_scene_card_ids`: the subset of scene cards to draft now.
        - `passages`: source evidence for this batch. Treat `passages[].text` as the canonical evidence body for writing.
        - `books`: compact book metadata.
        - Optional `previous_sections` and `previous_transitions`: already-written prior batches. Continue from them; do not rewrite them.
        - `skip_grounding`: whether a later grounding pass will be skipped.

        Output requirements:
        - Return only valid JSON with:
          - `batch_id`
          - `prose_sections`
          - `transitions`
          - `window_map`
        - Write only the active batch.
        - Preserve continuity with previous batches when they are supplied.
        - Each prose section must map explicitly to one or more `scene_card_ids` from the active batch.
        - Each prose section must choose one `movement_goal` from: `pose`, `discover`, `complicate`, `connect`, `judge`, `land`.
        - Do not include a `citations` field in `prose_sections` or `transitions`.

        Writing guidance:
        - Use scene-led narration, causally legible movement, and analytical pressure that emerges from evidence.
        - Let different books interact through evidence rather than summary labels.
        - Keep framing visible, but do not prematurely resolve the episode's unanswered question.
        - If a passage is vivid and concrete, use it to anchor narration. If a passage is abstract, translate it into clear narrative prose without changing its meaning.
        - `transitions` should bridge adjacent sections without re-summarizing the whole argument.

        What not to do:
        - Do not draft scene cards outside `active_scene_card_ids`.
        - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
        - Do not introduce new primary analytical claims that are outside the assigned scene cards and primitives.
        - Do not revise prior batches when `previous_sections` or `previous_transitions` are present.
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
        You are the `spoken_delivery` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Rewrite a completed episode script for spoken delivery as one whole-episode pass.
        - Improve cadence, clarity, and oral flow without changing the structure or factual meaning.

        Input payload:
        - `episode_number`: current episode number.
        - `script`: the full `EpisodeScript`.
        - `max_words_per_segment`: soft target for spoken-unit length.
        - `tts_provider`: downstream rendering target.

        Output requirements:
        - Return only valid JSON with `sections` and `transitions`.
        - Preserve section order, transition order, boundaries, ids, and substantive argumentative progression.
        - Keep unresolved questions unresolved until the draft resolves them.
        - Use `speech_hints` to support downstream rendering where useful, but do not over-annotate every line.

        Spoken-delivery guidance:
        - Reduce repeated abstraction.
        - Shorten or remove redundant sentences.
        - Prefer clear oral syntax, varied sentence length, and strong cadence.
        - Keep names, chronology, causality, and attribution intact.
        - Do not turn a cautious claim into a stronger claim.
        """
    ).strip()


def style_audit_instructions() -> str:
    return dedent(
        """
        You are the `style_audit` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Review the spoken script and emit warnings only.
        - This is a non-blocking audit, not a rewrite stage.

        Input payload:
        - `episode_number`: current episode number.
        - `script`: the full `SpokenScript`.

        Output requirements:
        - Return only valid JSON matching `StyleAuditReport`.
        - Use only these warning types:
          - `early_thesis_reveal`
          - `abstract_noun_cluster_repeat`
          - `governing_metaphor_repeat`
          - `author_hand_language`
          - `recap_style_framing`
        - Attach warnings to `text_unit_id` when the problem is localized.
        - Populate `counts_by_type` consistently with the warnings you emit.

        Audit guidance:
        - Warn when prose overstates the thesis too early.
        - Warn when abstraction piles up and weakens oral clarity.
        - Warn when one governing metaphor or verbal tic is overused.
        - Warn when the narration leans on author-hand language instead of evidence-led narration.
        - Warn when recap/preview framing sounds formulaic or mechanical.
        - Do not suggest fixes outside the warning message itself.
        - Do not rewrite the script.
        """
    ).strip()
