"""Prompt builders for active LLM stages."""

from __future__ import annotations

from textwrap import dedent


def chapter_summary_instructions() -> str:
    return dedent(
        """
        You are the `chapter_summary` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Read one chapter and produce structured chapter analysis.
        - Stay faithful to the chapter text. Do not infer details not supported by the chapter.

        Input payload:
        - `theme`: the overarching project theme.
        - `sub_themes`: optional supporting lenses. Use them only when they are actually supported by the chapter.
        - `theme_elaboration`: optional longer framing for what the project is really trying to understand.
        - `book_id`, `title`, `author`: book metadata for orientation only.
        - `chapter_title`: the chapter label.
        - `chapter_text`: the source evidence. Ground all claims in this field.

        Output requirements:
        - Return only valid JSON matching the response model with key `analysis` (or `null` when the chapter has no usable signal).
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
        - Synthesize one theme-conditioned book summary from chapter-level structured analysis.
        - The output is used for axis discovery, not for final narration.

        Input payload:
        - `theme`, `sub_themes`, `theme_elaboration`: the project framing.
        - `book_id`, `title`, `author`: book identity.
        - `chapters`: chapter-level structured analysis objects.

        Output requirements:
        - Return only valid JSON with a single key: `summary`.
        - Write one concise but information-dense synthesis of what this book contributes to the project theme.
        - Highlight recurring patterns, major tensions, important actors or institutions, and the book's distinctive angle.
        - Use the chapter analysis objects as evidence. Do not invent arguments or chronology missing from the input.
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
          - `chapters`: chapter-analysis objects

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
        - Emit primitives only under `primitives_by_family`.
        - Emit only these family keys:
          - `turning_points`
          - `scene_worthy_consequences`
          - `causal_mechanisms`
          - `live_questions`
          - `reversals`
          - `motivations_dilemmas`
          - `perspective_shifts`
          - `moral_ambiguities`
          - `personal_stakes`
          - `trauma_legacies`
        - Target these family count ranges:
          - `turning_points`: 25-40
          - `scene_worthy_consequences`: 30-45
          - `causal_mechanisms`: 20-35
          - `live_questions`: 20-30
          - `reversals`: 20-30
          - `motivations_dilemmas`: 20-30
          - `perspective_shifts`: 15-30
          - `moral_ambiguities`: 15-30
          - `personal_stakes`: 15-25
          - `trauma_legacies`: 10-20
        - Every primitive must be grounded in passage ids that appear in the payload.
        - Do not reuse the same passage across primitives unless it is uniquely decisive.
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
        - Return only primitive ids for surviving items under `primitive_ids_by_family`.
        - `primitive_ids_by_family` must use the same family keys as `primitives_by_family`.
        - Build `episode_candidate_clusters` as compact local causal chains or tightly related local story packets.
        - Every cluster must:
          - have a unique `cluster_id`
          - choose one valid `primary_member_id`
          - list valid `member_ids`
          - articulate a `local_question`
          - choose one `local_payoff_shape`
        - Aim for 3-8 members per cluster.
        - Clusters should be small and focused episode-usable units rather than whole-series theses.

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
        - Target 30-45 scene cards for a full-length episode; expand into micro-scenes rather than collapsing long stretches.
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
        - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target (computed at 120 WPM).
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
        - Instead of summarizing the passages, use them to reconstruct the scene.

        Input payload:
        - `episode_number`: current episode number.
        - `batch_id`: the current writing batch identifier.
        - `plan`: the full episode plan, including framing and all scene cards.
        - `active_scene_card_ids`: the subset of scene cards to draft now.
        - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target (computed at 120 WPM).
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
          - Do not exceed the target_word_count_higher for the scene unless absolutely necessary.
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
        You are the `spoken_delivery` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Rewrite a completed historical episode script for spoken delivery as one whole-episode pass.
        - Improve cadence, clarity, and oral flow without changing the structure, chronology, or factual meaning.
        - Make the episode sound stronger by deleting redundancy, not by preserving every sentence.

        Use the following narration style (this is an excerpt from a altogether different script):
        Two other quick points worth making about the Doolittle raid. The first is that there is another way of looking at this and people who score more aggressively on the willing to take risk military commanders spectrum that I do, will say that in addition to the, you know, everyone agrees upon morale and psychology boosts and effects that there was a real world goal in this whole thing and that it was achieved in the real world goal was the equivalent of what a boxer does when he faints an opponent in the ring. 
        The Japanese flinched, they reacted to the blow by changing their strategy, moving their assets around, doing things differently and perhaps deciding different places to strike and maneuver in order to prevent anything like this from happening again. 
        So in other words, a real world military affect something that's hard to measure in terms of how much of an effect and how much that helps. But something different from the psychological or morale side of things.
        I tend to think it's a little bit more like, you know, you spin the wheel of what happens when we bombed Japan and something good came up. But everybody's got a different opinion.
        Now, something that is not a question of opinion is the real world effects of the Doolittle raid on our Allies. If you're in an allied country, the Chinese because they were so obviously in on this plan, they were to provide the basis that the, the pilots and the plane crews landed at.
        But there's a harrowing personal experiences thing on the Chinese side that's a little known in  the United States.
        The Japanese took their anger out on the only allied people that they could get their hands on and take their anger out on, they punished the Chinese and the numbers are incredible. Most people think that Chiang kai shek was exaggerating but cut him in half if you want Chang says 250,000 Chinese civilians paid with their lives in reprisals for the Doolittle raid.
        In his book Hirohito's war author Frances, pike puts it this way, Chang would later notify Roosevelt that in southern China the Japanese Army slaughtered 250,000 Chinese civilians in a campaign of vengeance.
        Even allowing for some exaggeration on Chang's part, the Doolittle raid thus caused the death of more than twice the number of Chinese than the United States military suffered during the entire Pacific war End, I'm fascinated by human experiences
        It's part of the reason we focus so much of it in these conversations and in the same way I can't help but think About those B- 25 crews that have just bombed Japan and whose planes are running out of fuel and now they're you know, going to bail out. I can't help but wonder, oh my God, what is that like?
        And you think about that human experience and that is one kind of human experience, but at the very opposite End of the scale with Franklin Roosevelt experiences when his ally Chiang kai shek tells him that resulting from decisions that he made, 250,000 civilians are brutally killed. That's a different kind of human experience.
        How'd you like to be in Franklin Roosevelt shoes right then.
        Now disclaimer, there are well known leaders of countries on both the Axis and allied side who seem to have not a ton of sympathy for human suffering. One of them is alleged to have said that one death is a tragedy, a millionaire statistic another sets up human extermination camps that run like a factory Operation.
        So let's not attribute the same sense of gut punch when you find out what results downstream sometimes from your decision making to all of them. But I do not believe many people would think franklin. Delano Roosevelt was one of those people.
        And yes, war results and terrible things happen. But in this case Roosevelt is on record as saying what he was trying to do here with a raid that would not have killed tons of people anyway, was bring the war home to the Japanese and show them what this was like. You know, here's what you bargained for.
        This is what's going to happen because you start, you know, in other words, in his way, this was a measured sort of a response to that sent a message. And yet as a result of that decision, 250,000 civilians are killed. I cannot imagine that Roosevelt could have reacted to that.
        Well, I wonder what that night's attempt to sleep was like. That's not a human experience many people ever have to have either knowing that any decision you make could in many cases un foreseeably result in the deaths of huge numbers of human beings, would paralyze most of us.

        Input payload:
        - `episode_number`: current episode number.
        - `script`: the full `EpisodeScript`.
        - `max_words_per_segment`: soft target for spoken-unit length.
        - `tts_provider`: downstream rendering target.

        Output requirements:
        - Return only valid JSON with `sections` and `transitions`.
        - Preserve section order, transition order, section boundaries, transition boundaries, all `section_id` and `transition_id` values, and substantive argumentative progression.
        - Keep unresolved questions unresolved until the draft itself resolves them.
        - Use `speech_hints` only where they materially help delivery; do not annotate every unit.

        Core rewrite rules:
        - You may delete redundant structural language when its function is already achieved elsewhere.
        - Do not preserve repeated thesis statements, repeated recaps, or repeated rhetorical framing just because they appeared in the draft.
        - If a sentence only restates what the surrounding scene already made clear, cut it.
        - Prefer subtraction over paraphrased duplication.

        Listener-first guidance:
        - Podcast listeners experience this linearly. They cannot skim. Trust them.
        - Say important things once, cleanly.
        - Let vivid facts and scenes carry weight without telling the listener that they are important.
        - Avoid narrator tics such as: "Pause on that," "Read that again," "Let that sink in," and "Think about what this means."
        - Avoid repeated rhetorical questions unless a later question materially changes the frame.

        Tone guidance:
        - Use plain reportorial tone as the default register.
        - Reserve heightened diction for genuine irreversible turns.
        - Reduce repeated abstraction, repeated emphasis, and prestige-documentary inflation.
        - Avoid overusing words like "devastating," "extraordinary," "staggering," "remarkable," "breathtaking," and "fateful."
        - Do not turn every paragraph into a climax.

        Historical discipline:
        - Keep names, dates, chronology, causality, and attribution intact.
        - Do not turn a cautious or interpretive claim into a stronger claim.
        - When the draft presents a hot historical claim, preserve or sharpen the caution rather than smoothing it into certainty.
        - Do not overstate single-cause explanations for partition or Pakistan's emergence.

        Spoken-delivery guidance:
        - Prefer clear oral syntax, varied sentence length, and strong cadence.
        - Use short sentences for turns and longer sentences for causality only when they remain speakable.
        - Make transitions feel inevitable, not announced.
        - Convert analytical scaffolding into narrative flow where possible.
        - If a transition or section is empty or functionally redundant, keep the id and return the leanest viable text.

        Pronunciation guidance:
        - For names, places, and non-English terms that may be misread, add `speech_hints.pronunciation_hints`.
        - Keep pronunciation hints sparse; add them only when they materially improve delivery.
        - In each hint, keep `text` exactly as it appears in the segment text.
        - Use concise, speakable `spoken_as` values.

        Final self-check before return:
        1. Did I preserve the structure and factual meaning? If not, fix it.
        2. Did I cut duplicated framing where the scene already did the work? If not, cut it.
        3. Did I keep the prose speakable without flattening the history? If not, rewrite it.
        """
    ).strip()
