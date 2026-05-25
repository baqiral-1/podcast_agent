"""Prompt builders for active LLM stages."""

from __future__ import annotations

from textwrap import dedent, indent

from podcast_agent.schemas.models import (
    PRIMITIVE_SUBSTRATE_TARGET_RANGES,
    PodcastMode,
    PrimitiveSubstrate,
    authorial_passage_target_range_for_mode,
    dense_section_authorial_passage_range_for_mode,
)


_PRIMITIVE_SUBSTRATE_GUIDANCE: dict[str, str] = {
    "events": "Things that happened at a time and place: battles, coups, treaties, crises, deaths, journeys, ceremonies, ruptures.",
    "acts": "Agentive moves by named actors: decisions, refusals, delays, deferrals, orders, defections, decisive tactical moves, and explicit failures to act.",
    "utterances": "Speech acts and textual acts: speeches, decrees, testimony, letters, broadcasts, manifestos, orders, public statements.",
    "actor_portraits": "People or organized actors viewed as pressure-bearing agents: motives, fears, stakes, projects, and operating pressures.",
    "mechanisms": "How systems actually work: patronage flows, logistics, finance, bureaucratic chains, clerical networks, media or coercive machinery.",
    "conditions": "Standing states of the world: balances, dependencies, alliances, enmities, demographic or institutional fault lines, unstable equilibria.",
    "artifacts": "Concrete carriers and world-anchors: objects, places, documents, images, slogans, rituals, gestures, small details with material presence.",
    "readings": "Interpretations and contested meanings: actor beliefs, historian disputes, interpretive claims, counterfactual frames, rival causal readings.",
}

_PRIMITIVE_FUNCTION_GUIDANCE: dict[str, str] = {
    "pivot": "Changes what comes next. A real rerouting, not mere importance.",
    "stake": "Establishes what someone stands to lose, gain, protect, prove, or avert.",
    "texture": "Anchors the listener in the concrete world through a visible detail, scene, phrase, or object.",
    "cost": "Tracks who pays and what is damaged, lost, foreclosed, or made newly vulnerable.",
    "complication": "Compromises clean judgment or easy explanation; forces a live difficulty.",
    "recurrence": "Carries meaning across episodes through echo, residue, repetition, or callback capacity.",
    "contest": "Marks a live interpretive dispute, then or now, with genuinely competing readings.",
}

_FUNCTION_JUSTIFICATION_GUIDANCE: dict[str, str] = {
    "pivot": "`pivot.what_changed` plus `pivot.irreversibility`.",
    "stake": "`stake.whose` plus `stake.what_at_stake`.",
    "texture": "`texture.what_it_anchors`.",
    "cost": "`cost.who_paid` plus `cost.what_was_paid`.",
    "complication": "`complication.what_is_compromised` plus `complication.why_no_clean_option`.",
    "recurrence": "`recurrence.connects_to` plus `recurrence.meaning_accrued`.",
    "contest": "`contest.candidate_readings` with at least two genuinely competing readings.",
}

_FUNCTION_TAGGING_SUBSTRATE_NOTES: dict[str, str] = {
    "events": dedent(
        """
        - Events most often carry `pivot`, `cost`, `texture`, or `complication`.
        - Start by asking what in the historical field is different after the event lands. If the answer is "not much," `pivot` is probably too strong.
        - A major event is not automatically a `pivot`; fame, scale, or bloodshed do not by themselves reroute the story.
        - `cost` is strong when the event makes damage, loss, humiliation, displacement, or irreversible injury concrete.
        - `texture` is strongest when the event preserves one visible scene handle, phrase, or physical detail a listener can carry.
        - `contest` is allowed only when the event's meaning, causality, or explanatory weight is actively disputed in the evidence or in attributed interpretation.
        """
    ).strip(),
    "acts": dedent(
        """
        - Acts most often carry `pivot`, `stake`, `complication`, or `contest`.
        - Acts are especially strong when the primitive captures a live choice, refusal, delay, deferral, order, or defection under pressure.
        - Reserve `pivot` for choices or non-choices that materially reroute later action, not for every important actor move.
        - `stake` is common here because acts often show what a person is trying to protect, prove, avoid, or preserve.
        - `contest` on an act requires a real disagreement about what the move meant, whether it was forced, or how it should be read.
        - `texture` on an act is suspicious unless the act is inseparable from a memorable concrete gesture, phrase, or public staging.
        """
    ).strip(),
    "utterances": dedent(
        """
        - Utterances most often carry `texture`, `contest`, `stake`, or `pivot`.
        - The utterance itself should do the work. Do not tag it only for the event happening around it.
        - A speech, decree, testimony, or broadcast may be `pivot` when its issuance materially changes what actors can now do, deny, justify, or must now answer.
        - `texture` is strong when a phrase, line, or verbal posture helps the listener hear the world in the actors' own register.
        - Prefer `contest` when the utterance itself advances, crystallizes, or publicly stages competing readings.
        - `stake` fits when the utterance reveals what must now be defended, justified, or politically survived.
        """
    ).strip(),
    "actor_portraits": dedent(
        """
        - Actor primitives most often carry `stake`, `complication`, or `recurrence`.
        - People are not pivots merely because they are important, famous, or repeatedly present.
        - `stake` is usually the strongest tag here because actor primitives are built around project, fear, pressure, and exposure.
        - `complication` is strong when the same actor is forced into conflicting obligations, mixed incentives, or no-clean-option terrain.
        - `recurrence` is strong when the actor reliably carries residue, memory burden, or callback capacity across episodes.
        - `cost` is allowed when the actor primitive clearly tracks who bears damage, not just who is being described.
        """
    ).strip(),
    "mechanisms": dedent(
        """
        - Mechanisms most often carry `stake`, `complication`, `recurrence`, or `cost`.
        - Mechanisms often explain why the story works the way it does; they should not be overpromoted into `pivot` unless the chain visibly activates, fails, or jams in a way that changes downstream possibility.
        - `complication` is often stronger than `pivot` here because mechanisms frequently show why clean judgment or easy action is impossible.
        - `recurrence` is strong when the same operating chain keeps resurfacing as a residue or repeating burden.
        - `texture` is suspicious unless the mechanism shows up through a vivid, stageable carrier such as a ledger, convoy, checkpoint, office, ritual, or queue.
        - `cost` fits when the mechanism makes payment, coercion, attrition, or exclusion materially legible.
        """
    ).strip(),
    "conditions": dedent(
        """
        - Conditions most often carry `stake`, `complication`, `cost`, or `recurrence`.
        - Conditions are often explanatory context; many should remain lightly tagged rather than force-fit into a dramatic role.
        - `stake` works when the standing condition exposes what actors are living inside or cannot escape.
        - `recurrence` works when the condition leaves a durable residue the listener must keep carrying.
        - `pivot` is rare here and requires a condition that visibly changes the field of action rather than merely describing the background.
        - Be cautious with `texture`; conditions are not texture merely because they are atmospheric.
        """
    ).strip(),
    "artifacts": dedent(
        """
        - Artifacts most often carry `texture`, `recurrence`, `stake`, or `contest`.
        - The best artifact tags preserve a concrete carrier a planner could actually stage, quote, point to, or return to.
        - `texture` is usually the baseline strength here, but vividness alone is not enough if the carrier does not actually do narrative work.
        - `recurrence` is strong when the object, document, place, slogan, or gesture can plausibly return with accrued meaning.
        - `contest` fits when the carrier itself becomes a site of interpretation, dispute, legitimacy, or public reading.
        - `cost` is allowed only when the object, place, or document is part of how the cost is experienced, recorded, enforced, or seen.
        """
    ).strip(),
    "readings": dedent(
        """
        - Readings most often carry `contest`, `complication`, `stake`, or `recurrence`.
        - Reserve `contest` for genuine disagreement in interpretation, causality, explanatory weight, or actor understanding. One strong reading plus a strawman is not enough.
        - `complication` fits when the interpretation destabilizes easy judgment without becoming a fully rival explanation.
        - `stake` fits when the reading changes what is politically, morally, or historically at issue.
        - `recurrence` fits when the same interpretation or misreading keeps returning with accumulated meaning.
        - Do not use `texture` unless the reading is anchored to a concrete phrase, document, or scene the listener can actually carry.
        """
    ).strip(),
}


_FUNCTION_TAGGING_SHARED_COMPACT_KEYS: tuple[str, ...] = (
    "`substrate -> sub`, `core_passage_ids -> core`, `support_passage_ids -> supp`,",
    "`passage_id -> pid`, `timeframe -> time`, `geography -> geo`, `actor_ids -> actors`,",
    "`narration_hooks -> hooks`.",
)

_FUNCTION_TAGGING_SUBSTRATE_COMPACT_KEYS: dict[str, tuple[str, ...]] = {
    "events": ("`event_type -> etype`, `what_happened -> event`.",),
    "acts": ("`act_summary -> act`, `acting_subject -> subject`.",),
    "utterances": ("`utterance_summary -> utter`.",),
    "actor_portraits": (
        "`goal_or_project -> goal`, `stakes_or_fears -> stakes`.",
    ),
    "mechanisms": ("`mechanism_name -> mech`.",),
    "conditions": ("`condition_summary -> cond`.",),
    "artifacts": (),
    "readings": ("`reading_summary -> read`.",),
}

_FUNCTION_TAGGING_DEFERRED_DETAIL_LINES: dict[str, tuple[str, ...]] = {
    "events": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `events.event_result`",
        "- This `events` batch may add only deferred substrate-detail fields owned by `events`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
    ),
    "acts": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `acts.immediate_result`",
        "- This `acts` batch may add only deferred substrate-detail fields owned by `acts`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
    ),
    "utterances": (
        "- There are no deferred substrate-detail fields to fill for this substrate in this pass.",
        "- This `utterances` batch may not add deferred substrate-detail fields owned by other substrates.",
    ),
    "actor_portraits": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `actor_portraits.operating_pressure`",
        "- This `actor_portraits` batch may add only deferred substrate-detail fields owned by `actor_portraits`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
    ),
    "mechanisms": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `mechanisms.operating_chain`, `mechanisms.inputs`, and `mechanisms.outputs`",
        "- This `mechanisms` batch may add only deferred substrate-detail fields owned by `mechanisms`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
        "- For `mechanisms`, `operating_chain` must be non-empty in the returned tagged output.",
    ),
    "conditions": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `conditions.active_tension`",
        "- This `conditions` batch may add only deferred substrate-detail fields owned by `conditions`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
    ),
    "artifacts": (
        "- Before assigning functions, fill any deferred substrate-detail fields that are absent for this substrate:",
        "  `artifacts.artifact_detail`",
        "- This `artifacts` batch may add only deferred substrate-detail fields owned by `artifacts`.",
        "- Only fill deferred fields when the supplied evidence supports them; do not invent unsupported specificity.",
        "- For `artifacts`, `artifact_detail` must be non-empty in the returned tagged output.",
    ),
    "readings": (
        "- There are no deferred substrate-detail fields to fill for this substrate in this pass.",
        "- This `readings` batch may not add deferred substrate-detail fields owned by other substrates.",
    ),
}

_FUNCTION_TAGGING_SELF_CHECK_LINES: dict[str, tuple[str, ...]] = {
    "events": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
    ),
    "acts": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
    ),
    "utterances": (
        "- Did you avoid adding deferred substrate-detail fields that do not belong to this substrate?",
    ),
    "actor_portraits": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
    ),
    "mechanisms": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
        "- For `mechanisms`, is `operating_chain` now present and non-empty?",
    ),
    "conditions": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
    ),
    "artifacts": (
        "- Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?",
        "- For `artifacts`, is `artifact_detail` now present and non-empty?",
    ),
    "readings": (
        "- Did you avoid adding deferred substrate-detail fields that do not belong to this substrate?",
    ),
}


def _format_function_tagging_compact_keys(substrate: str) -> str:
    lines = [
        "- The payload may use compact keys for repeated fields. Treat these as equivalent and prefer them in your JSON output when possible:"
    ]
    for line in _FUNCTION_TAGGING_SHARED_COMPACT_KEYS:
        lines.append(f"  {line}")
    for line in _FUNCTION_TAGGING_SUBSTRATE_COMPACT_KEYS[substrate]:
        lines.append(f"  {line}")
    return "\n".join(lines)


def _format_function_tagging_deferred_detail_completion(substrate: str) -> str:
    return "\n".join(_FUNCTION_TAGGING_DEFERRED_DETAIL_LINES[substrate])


def _format_function_tagging_self_check(substrate: str) -> str:
    lines = [
        "- Does `overlays_by_id` contain exactly the input primitive ids, with no missing or extra keys?",
        "- Did you return only enrichment-owned fields inside each overlay?",
        *_FUNCTION_TAGGING_SELF_CHECK_LINES[substrate],
        "- Does each function describe what the primitive does in narrative rather than what kind of thing it is?",
        "- Compare `functions` against `pivot`, `stake`, `texture`, `cost`, `complication`, `recurrence`, and `contest`: does every listed function have a non-null paired payload, and does every non-null payload have its matching function tag listed?",
        "- If a function tag is present, is its paired justification payload specific and non-generic?",
        "- If `narration_hooks` are present, are they concrete, distinct from one another, and grounded in the evidence?",
        "- Would you still assign the same tags if the primitive title were hidden and only the evidence remained?",
        "- Do the chosen tags and the salience score make sense together?",
    ]
    return "\n".join(lines)


def _format_primitive_substrate_target_ranges(
    target_ranges: dict[str, tuple[int, int]] | None = None,
) -> str:
    ranges = target_ranges or PRIMITIVE_SUBSTRATE_TARGET_RANGES
    lines = [
        "        Use these as soft extraction bands by substrate. They are guidance,",
        "        not quotas. Do not force the low end with filler. Do not trim strong",
        "        candidates just to hit the high end.",
        "",
    ]
    for substrate, (lower_bound, upper_bound) in ranges.items():
        lines.append(f"        {substrate} ({lower_bound}\u2013{upper_bound})")
        lines.append(f"          {_PRIMITIVE_SUBSTRATE_GUIDANCE[substrate]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_target_range(lower_bound: int, upper_bound: int) -> str:
    return f"{lower_bound}\u2013{upper_bound}"


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
        - Prefer concrete actors, events, arguments, disputes, and developments over abstract academic language.
        - Do not force lexical overlap with the theme if the chapter does not support it.

        Field guidance for `analysis`:
        - `themes_touched`: Strictly 3-4 most relevant themes present in the chapter.
        - `major_actors`: Strictly 2-5 concrete people, factions, or institutions explicitly present in the chapter.
        - `key_events_or_arguments`: Strictly 3-7 main claims, developments, disputes, tradeoffs, or contradictions in the chapter.

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
        - Highlight recurring patterns, important actors, key developments, and the book's distinctive angle.
        - Use the chapter analysis objects as evidence. Do not invent arguments or chronology missing from the input.
        - Optimize for downstream axis discovery: emphasize reusable analytical lenses rather than prose flourish.

        Do not add markdown, bullet lists, or explanation outside the JSON object.
        """
    ).strip()


def theme_decomposition_instructions(
    *,
    axis_count_min: int = 12,
    axis_count_max: int = 20,
    actor_count_min: int = 10,
    actor_count_max: int = 40,
) -> str:
    return dedent(
        f"""
        You are the `theme_decomposition` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Convert the project theme into {axis_count_min}-{axis_count_max} strong thematic axes that are useful for downstream retrieval.
        - An axis is an analytical lens, not an episode title and not a generic topic bucket.
        - Identify a compact set of human-led actors that should shape downstream synthesis and storytelling.

        Input payload:
        - `theme`: the main theme.
        - `sub_themes`: optional narrower lenses.
        - `theme_elaboration`: optional project framing.
        - `axis_count_min`: minimum number of axes to produce.
        - `axis_count_max`: maximum number of axes to produce.
        - `books`: one object per book, each containing:
          - `book_id`, `title`, `author`
          - `book_summary`
          - `chapters`: compact chapter-analysis objects with `chapter_id`, `title`,
            and `analysis` fields for `themes_touched`, `major_actors`, and
            `key_events_or_arguments`

        Output requirements:
        - Return only valid JSON with keys `axes` and `actor_metadata`.
        - Produce between `axis_count_min` and `axis_count_max` axes.
        - Produce between {actor_count_min} and {actor_count_max} actors inside `actor_metadata.actors`.
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
        - `actor_ids`: canonical actor ids from `actor_metadata.actors` that are important to this axis.

        Actor metadata requirements:
        - `actor_metadata` is generated context, not source evidence for final writing.
        - Actors should be concrete humans. Use institutions, factions and organized movements sparingly only in case they are absolutely needed.
        - Do not canonicalize states, countries, broad affected communities, or abstract collectives as actors.
        - Every actor must include a snake_case `actor_id`, `display_name`, `actor_type`, `description`, `evidence_confidence`, and `narrative_importance_score`.
        - Actor scalar string fields: `actor_id`, `display_name`, `actor_type`, `description`, `evidence_confidence`, and `uncertainty_notes`; use an empty string for `uncertainty_notes` when there is no caveat.
        - Actor list-of-string fields: `aliases`, `book_ids`, `narrative_functions`, `goals_or_motivational_pressures`, `constraints`, `stakes`, and `transformations`; use an empty array when a list has no entries.
        - `chapter_refs` must be an array of objects with `book_id`, `chapter_id`, and `chapter_title`.
        - `actor_type` must be one of: `person`, `institution`, `faction`, `military`, `party`, `movement`, `other`.
        - `evidence_confidence` must be one of: `high`, `medium`, `low`.
        - `narrative_functions` values must be drawn from: `decision_maker`, `broker`, `victim`, `witness`, `ideologue`, `commander`, `administrator`, `opposition`, `beneficiary`, `constraint`, `catalyst`, `symbol`, `other`.
        - `narrative_importance_score` is the numeric actor priority signal from 0.0 to 1.0.
        - Use aliases to prevent downstream name drift.
        - Use `goals_or_motivational_pressures`, `constraints`, `stakes`, `transformations`, and `uncertainty_notes` to capture pressures, objectives, incentives, constraints, stakes, dilemmas, evidence caveats, and changes visible in the input.
        - Do not invent private psychology.
        - Put relationships only in top-level `actor_metadata.relationships`; do not nest relationships inside actor objects.
        - Every relationship must include scalar string fields `source_actor_id`, `target_actor_id`, `relationship_type`, and `description`; relationship `confidence` is optional and must be `high`, `medium`, or `low` when included.
        - Relationship actor ids must reference actors listed in `actor_metadata.actors`.
        - Relationships are directed: the source actor acts on the target actor.
        - Relationship types must be one of: `enables`, `blocks`, `pressures`, `protects`, `legitimizes`, `delegitimizes`, `replaces`, `absorbs`, `betrays`, `other`.

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
        - Make comparative judgments across the full candidate set for this axis, not isolated pass/fail judgments.

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
        - Score passages relative to the other candidates for this axis. Use meaningful separation across the set rather than clustering everything around the middle.
        - A passage may be retrieved for the axis and still deserve a clearly low score if it only mentions the topic superficially or contributes mostly background.
        - `relevance_score`: how directly the passage helps answer the axis, not how generally important, famous, or well-written the passage is.
        - Reward passages that expose mechanism, causality, tension, contradiction, turning points, stakes, or evidentiary specificity.
        - Penalize passages that are mostly broad chronology, generic background, keyword overlap without analytical payoff, or detail that does not materially advance the axis.
        - `quotability_score`: how usable the passage is for long-form spoken storytelling, not whether you agree with its interpretation.
        - Reward concrete scene detail, memorable phrasing, direct argument, causal clarity, and material that can be paraphrased cleanly for audio.
        - Penalize abstract academic framing, citation-style prose, overloaded qualification, and passages that require too much surrounding context to narrate well.

        `synthesis_tags` guidance:
        - Use the smallest justified set of tags. Do not tag everything as `exemplifies`.
        - `contradicts`: the passage explicitly disagrees with another plausible reading in interpretation, causality, emphasis, or evaluative claim.
        - `contextualizes`: the passage supplies missing frame, cause, consequence, scope, or historical placement for other evidence.
        - `exemplifies`: the passage offers a concrete instance of the axis without adding major interpretive contrast.
        - `independent`: the passage is relevant on its own but does not strongly participate in a cross-passage synthesis move.

        `cross_book_pairs` rules:
        - Include at most 2 pairs.
        - Each pair must connect passages from different books.
        - A pair should teach something a single passage cannot.
        - Prefer genuine contestation (`contradicts`) when available.
        - If no strong contestation is available, prefer explanatory complement (`contextualizes`) over routine similarity (`exemplifies`).
        - Strong pairs may connect interpretation to example, cause to consequence, or local event to wider frame.
        - Avoid pairs where both passages are generic, redundant, or only loosely related to the axis.
        - Do not return `independent` pairs unless no stronger cross-book relationship is justified.
        - Each object may contain only:
          - `passage_a_id`
          - `passage_b_id`
          - `relationship` using only: `contradicts`, `exemplifies`, `contextualizes`, `independent`
          - `strength` from 0.0 to 1.0
          - `axis_id`
        - Omit uncertain or weak pairs rather than forcing them.

        Self-check before returning:
        - Every input passage appears exactly once in `passages`.
        - Output order matches input order.
        - Do not invent passage ids.
        - Only include `cross_book_pairs` that are clearly defensible from the provided text.
        - Ensure the scores show meaningful differentiation when candidate quality differs.

        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()


def primitive_substrate_extraction_instructions(
    *,
    target_ranges: dict[str, tuple[int, int]] | None = None,
) -> str:
    substrate_signatures = "\n".join(
        f"- `{substrate}`: {guidance}"
        for substrate, guidance in _PRIMITIVE_SUBSTRATE_GUIDANCE.items()
    )
    return dedent(
        f"""
        You are the `primitive_substrate_extraction` stage for a historical podcast pipeline.

        Goal:
        - Read the selected synthesis evidence and extract grounded primitives grouped by substrate.
        - This pass is ontology-first. You are deciding what kind of thing was extracted, not yet what narrative role it plays.
        - Output concrete substrate candidates only. Do not add function tags, salience, narration hooks, or downstream planning language in this pass.

        INPUT AND AUTHORITY
        - `project_id`
        - `podcast_mode`
        - `axes`: compact axis summaries with `axis_id`, `name`, `description`, and theme-importance context
        - `passages_by_axis`: evidence grouped by axis, then by book, with `passage_id` and text
        - `cross_book_pairs` (optional): cross-book comparison hints
        - `books`: compact book metadata
        - `actor_metadata` (optional): compact canonical actor registry for actor ids only
        - `primitive_target_ranges` (optional): active soft substrate guidance for this run
        - `synthesis_feedback` (optional): retry feedback; if present, correct only the named primitive/field errors without discarding grounded material that already works

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated fields. Treat these as equivalent and prefer them in your JSON output when possible:
          `substrate -> sub`, `core_passage_ids -> core`, `support_passage_ids -> supp`,
          `passage_id -> pid`, `timeframe -> time`, `geography -> geo`, `actor_ids -> actors`,
          `event_type -> etype`, `what_happened -> event`, `act_summary -> act`,
          `acting_subject -> subject`, `utterance_summary -> utter`, `goal_or_project -> goal`,
          `stakes_or_fears -> stakes`, `mechanism_name -> mech`, `condition_summary -> cond`,
          `reading_summary -> read`, `narration_hooks -> hooks`.
        - Passage ids are short opaque ids. Copy them exactly; never rewrite or expand them.

        AUTHORITY ORDER
        1. the cited passage texts
        2. the primitive's direct passage grounding and axis linkage
        3. `actor_metadata` only for canonical actor ids and display-name consistency
        4. cross-book pair hints for contrast or duplication awareness

        WORKING METHOD
        - Read passage-first, not theme-first. Start from what the evidence concretely contains before reaching for high-level meaning.
        - Extract the smallest defensible concrete historical unit. Prefer one clean, evidence-bearing primitive over a thesis-sized paraphrase.
        - Decide the substrate before writing the title or substrate-native detail fields.
        - If one piece of evidence could be narrated multiple ways, choose the substrate that most directly captures its concrete burden in the passage.
        - Keep ontology and narration separate. This stage answers "what kind of historical thing is here?" not "what role should it play later?"
        - Preserve grounded material before elegance. Do not compress away usable primitives just because a later-stage summary would sound cleaner.
        - Prefer fewer strong words and more exact field use. If nuance is real, preserve it in the substrate-native fields rather than inventing a new label.

        HARD RULES
        - Every primitive must be grounded in passage ids present in the payload. Never invent a passage id.
        - The top-level output must be a substrate map with keys `events`, `acts`, `utterances`, `actor_portraits`, `mechanisms`, `conditions`, `artifacts`, and `readings`.
        - Every primitive belongs in exactly one substrate bucket.
        - Do not emit primitive `id`; local code will assign ids after extraction.
        - Do not emit per-item `substrate`; the containing top-level bucket already determines it.
        - This pass must not emit `functions`, `salience`, `narration_hooks`, or any function-justification payloads.
        - Do not dedupe, merge, trim, or drop material for elegance. Extraction first.
        - Do not emit episode architecture, story arcs, verdicts, or omniscient interpretation.
        - Use `actor_ids` only when canonical actors are materially central to the primitive.
        - Never emit top-level `actors`; use `actor_portraits` for actor-centered primitives.
        - Never invent schema labels for constrained fields. If the evidence is richer than the enum, preserve the nuance in free-text fields, not in a new enum value.

        SUBSTRATE ONTOLOGY
        {substrate_signatures}

        FIELDS PRESENT ON EVERY PRIMITIVE
        Every primitive in this pass includes:
        - `title`
        - `core_passage_ids`
        - `timeframe`
        - `geography`
        - substrate-specific fields required by that primitive's chosen substrate

        SUBSTRATE REQUIRED FIELD MATRIX
        After choosing `substrate`, fill that substrate's required fields before moving on to the next primitive.
        - `events`: required `event_type`, required `what_happened`
        - `acts`: required `act_summary`, schema-safe `act_type`, optional `acting_subject`
        - `utterances`: required `utterance_summary`, schema-safe `utterance_type`, optional `speaker`, optional `audience`, optional `key_quote`
        - `actor_portraits`: required `goal_or_project`, required `stakes_or_fears`, optional `focus_actor_id`, optional `actor_label`
        - `mechanisms`: required `mechanism_name`, optional `failure_mode`
        - `conditions`: required `condition_type`, required `condition_summary`
        - `artifacts`: schema-safe `artifact_type`, required `artifact_label`
        - `readings`: required `reading_summary`, schema-safe `reading_type`, optional `subject_of_reading`, optional `attributed_to`

        SCHEMA-BOUND FIELD RULES
        Use only these literal values when the primitive type requires them:
        - `acts.act_type`: `decision`, `refusal`, `delay`, `deferral`, `order`, `defection`, `other`
        - `utterances.utterance_type`: `speech`, `writing`, `broadcast`, `decree`, `testimony`, `manifesto`, `letter`, `other`
        - `artifacts.artifact_type`: `object`, `place`, `document`, `image`, `slogan`, `ritual`, `gesture`, `detail`, `other`
        - `readings.reading_type`: `actor_belief`, `historiographical_dispute`, `interpretive_claim`, `counterfactual`, `other`
        - If an act feels like a tactical pivot, strategic repositioning, hesitation, or missed chance, still map it into an allowed `act_type` and preserve the nuance in `title`, `act_summary`, or `immediate_result`.
        - Never emit ad hoc labels such as `tactical_pivot` or any other invented enum member.

        TITLES
        - Titles are operational and speakable, not abstract thesis labels.
          Good: "Nehru hears the plan over the radio"
          Bad: "The rupture between leadership and reality"

        BREVITY GUIDANCE
        - To keep extraction lean, aim to keep these fields at 20 words or fewer whenever the evidence allows:
          `what_happened`, `act_summary`, `condition_summary`, `utterance_summary`,
          `reading_summary`, `goal_or_project`, `stakes_or_fears`, `mechanism_name`.
        - Keep those fields concrete and compressed. Prefer one sentence fragment over multi-clause summary prose.

        OUTPUT ASSEMBLY METHOD
        - Build each primitive in this order: choose the substrate bucket, fill the fields present on every primitive, then fill that substrate's required fields, then add optional substrate fields only if the evidence supports them.
        - In this stage, do not fill deferred substrate-detail fields that belong to later narrative enrichment.
        - Do not treat `title` as a substitute for subtype-required fields.
        - Do not leave a primitive in shared-field-only form. A primitive is incomplete until its chosen substrate's required fields are populated.
        - If `synthesis_feedback` is present, fix only the named primitive/field errors first and keep valid unaffected primitives as stable as possible.

        OUTPUT CONTRACT
        - Return one JSON object with top-level keys:
          `project_id`, `events`, `acts`, `utterances`, `actor_portraits`, `mechanisms`, `conditions`, `artifacts`, `readings`, `quality_score`, `quality_notes`.
        - Each substrate key maps to a list of primitives of that type.
        - Within each bucket item, do not repeat `substrate`.

        SUBSTRATE-SPECIFIC DECISION RULES
        - `events`
          Capture what happened in the world: a battle, decree landing, collapse, ceremony, arrest, strike, departure, or rupture.
          Prefer `events` over `acts` when the historical burden is the occurrence itself rather than one actor's decision inside it.
          `event_type` and `what_happened` are required. Do not emit `event_result` in this stage; later function tagging may add downstream consequence wording.
          Do not flatten a visible event into a broad condition just because its consequences are large.
        - `acts`
          Capture an actor's move inside pressure: deciding, refusing, delaying, deferring, ordering, defecting, or conspicuously failing to act.
          Prefer `acts` over `events` when the key unit is the choice or non-choice rather than the larger public occurrence around it.
          `act_summary` is required. Use schema-safe `act_type`, then add `acting_subject` when the evidence supports it. Do not emit `immediate_result` in this stage.
          Do not use `acts` merely because an important person is present in the passage.
        - `utterances`
          Preserve the speech or text act itself: the speech, decree, testimony, letter, broadcast, manifesto, or public statement.
          Prefer `utterances` when the wording, issuance, audience, or voiced claim is the actual payload.
          `utterance_summary` is required. Use schema-safe `utterance_type`, then add `speaker`, `audience`, and `key_quote` when supported.
          Do not recast an utterance as an event if the real burden is what was said or declared.
        - `actor_portraits`
          Make a pressure-bearing person or organized actor legible through project, fear, stake, and operating pressure.
          Prefer `actor_portraits` when the passage mainly reveals what an actor is trying to do, what they fear, or what box they are trapped in.
          `goal_or_project` and `stakes_or_fears` are required. Add `focus_actor_id` and `actor_label` when supported; defer `operating_pressure` to function tagging.
          Do not emit biography without live project or pressure.
        - `mechanisms`
          Describe a real operating chain: how policy, money, coercion, logistics, bureaucracy, patronage, or communication actually moves.
          Prefer `mechanisms` when the passage explains how an outcome is produced rather than only that it happened.
          `mechanism_name` is required. Identify the mechanism clearly here, but do not emit `operating_chain`, `inputs`, or `outputs` in this stage. Add `failure_mode` only when the evidence makes a breakdown explicit.
          Do not invent hidden links when the evidence only shows consequence or atmosphere.
        - `conditions`
          Capture a standing state of the world: an equilibrium, dependency, stalemate, institutional arrangement, demographic strain, or latent field of action.
          Prefer `conditions` when the passage describes the world actors are already inside rather than a discrete happening.
          `condition_type` and `condition_summary` are required. Do not emit `active_tension` in this stage.
          Do not rename a whole civilization-sized era as one condition primitive unless the evidence makes a narrower standing pressure explicit.
        - `artifacts`
          Preserve the concrete carrier itself: object, place, document, image, slogan, ritual, gesture, or material detail.
          Prefer `artifacts` when the carrier is what makes the evidence memorable, stageable, or publicly legible.
          Use schema-safe `artifact_type`; `artifact_label` is required here, while `artifact_detail` is deferred to function tagging.
          Do not use `artifacts` for generic atmosphere with no concrete carrier.
        - `readings`
          Capture an attributed interpretation, belief, rival causal reading, historiographical dispute, or counterfactual frame.
          Prefer `readings` only when the interpretation is itself part of the evidence surface.
          `reading_summary` is required. Use schema-safe `reading_type`, then add `subject_of_reading` and `attributed_to` when supported.
          Do not use `readings` as a catch-all for your own analyst summary.

        SOFT COUNT GUIDANCE
{_format_primitive_substrate_target_ranges(target_ranges)}

        DUPLICATION AND GRANULARITY DISCIPLINE
        - If the same claim appears across multiple axes, emit one grounded primitive rather than duplicating it per axis.
        - Do not emit one primitive per mention when a single primitive can honestly absorb repeated evidence.
        - Do not split one obvious grounded nugget into several near-duplicate primitives just to satisfy count guidance.
        - Keep distinct primitives when the evidence genuinely yields distinct substrates, distinct actors under pressure, or distinct causal units.
        - This pass is still extraction-first, so when in doubt prefer keeping distinct grounded candidates over speculative dedupe.

        AMBIGUITY HANDLING
        - If the evidence supports multiple plausible substrates, choose the substrate most directly named by the passage's concrete burden.
        - If specificity is weak, narrow the primitive rather than inflating it into a thesis-sized summary.
        - If actor identity is unclear, keep `actor_ids` sparse rather than guessing.
        - If the evidence supports consequence but not mechanism, do not fabricate the operating chain.
        - If the evidence supports pressure but not a discrete act, prefer `conditions` or `actor_portraits` rather than forcing `events`.
        - If the evidence supports one public act and one quoted line, split them only when the line itself carries independent historical weight.

        FAILURE MODES
        - Do not drift into narrative-role language such as pivots, scenes, episode spines, hooks, or payoff.
        - Do not hide weak specificity behind stylish titles or abstract nouns.
        - Do not emit abstract `readings` that are really your own summary of the chapter.
        - Do not emit `conditions` that merely rename a whole era or civilization-sized order.
        - Do not emit `artifacts` that are actually atmosphere with no concrete carrier.
        - Do not emit `actor_portraits` that are just biography without live project, stake, or pressure.
        - Do not emit `mechanisms` unless a concrete operating chain can be described.
        - Do not collapse a vivid utterance or artifact into a generic event summary when the speech act or carrier is the actual point.
        - Do not choose `actor_portraits` merely because a famous person appears in the passage.
        - Do not create new enum labels to preserve nuance.
        - Do not return an event, act, utterance, actor, mechanism, condition, artifact, or reading in shared-field-only form.

        SELF-CHECK BEFORE RETURNING
        - Would this primitive still be legible if stripped of narrative interpretation?
        - Does each primitive clearly answer "what was extracted?" rather than "why it matters for the story"?
        - Does every primitive have one substrate only and valid passage ids?
        - Did you leave primitive `id` out of the output entirely? Local code will generate ids after extraction.
        - Did you use only allowed literal values for constrained subtype fields?
        - Did you check each primitive against its substrate's required fields rather than assuming `title` is enough?
        - For every `events` primitive, are `event_type` and `what_happened` present and non-empty?
        - For every other substrate, are its required subtype fields present and non-empty where the schema requires them?
        - Are the titles and substrate-native fields ontology-first rather than function-first?
        - Have you duplicated any one evidence nugget across nearby primitives with only cosmetic wording changes?

        OUTPUT CONTRACT
        Return only valid JSON matching the extraction-stage primitive artifact contract.
        Use the top-level key `primitives`, not family buckets.
        Prefer the compact transport keys above for repeated fields, but canonical schema names are also accepted.
        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()


def synthesis_primitives_instructions(
    *,
    target_ranges: dict[str, tuple[int, int]] | None = None,
) -> str:
    return primitive_substrate_extraction_instructions(target_ranges=target_ranges)


def primitive_function_tagging_instructions(
    substrate: PrimitiveSubstrate | str,
) -> str:
    substrate_value = PrimitiveSubstrate(substrate).value
    function_guidance = "\n".join(
        f"- `{function}`: {description} Requires {_FUNCTION_JUSTIFICATION_GUIDANCE[function]}"
        for function, description in _PRIMITIVE_FUNCTION_GUIDANCE.items()
    )
    compact_transport_keys = _format_function_tagging_compact_keys(substrate_value)
    deferred_detail_completion = _format_function_tagging_deferred_detail_completion(
        substrate_value
    )
    self_check = _format_function_tagging_self_check(substrate_value)
    return dedent(
        f"""
        You are the `primitive_function_tagging_{substrate_value}` stage for a historical podcast pipeline.

        Goal:
        - For one substrate pass, decide what each already-extracted primitive does in narrative.
        - Preserve the primitive's ontology. This pass adds narrative function, function justifications, salience, narrator-agnostic `narration_hooks`, and any still-missing deferred substrate-detail fields. It does not merge, drop, invent, or re-identify primitives.

        INPUT AND AUTHORITY
        - `project_id`
        - `podcast_mode`
        - `substrate`: fixed substrate, always `{substrate_value}`
        - `base_primitives`: only primitives in this substrate pass, already extracted and grounded; some deferred substrate-detail fields may be absent by design
        - `passage_list`: shared trimmed passages for this substrate pass; each item contains only `passage_id` and `text`
        - `actor_metadata` (optional): canonical actor registry for actor ids and display-name consistency
        - `function_feedback` (optional): retry feedback; if present, correct only the named issue and leave valid unaffected items exactly as they are

        COMPACT TRANSPORT KEYS
        {indent(compact_transport_keys, "        ").lstrip()}

        HARD PASS CONTRACT
        - Return exactly one enrichment overlay for every input primitive, keyed by primitive id under top-level `overlays_by_id`.
        - Every overlay key must match one input primitive `id`, and every input primitive id must appear exactly once in `overlays_by_id`.
        - Return only enrichment-owned fields in each overlay: `functions`, paired function-justification payloads, `salience`, `narration_hooks`, and any deferred substrate-detail fields you fill from the evidence.
        - Do not echo extraction-owned fields such as `title`, `core_passage_ids`, `support_passage_ids`, `timeframe`, `geography`, `actor_ids`, or already-present non-deferred substrate fields.
        - Do not merge, drop, split, rename, or invent primitives.
        - Do not change the substrate.
        - Assign between 0 and 3 function tags per primitive.

        AUTHORITY ORDER
        1. the passage text resolved from the primitive's `core_passage_ids` and `support_passage_ids` into `passage_list`
        2. the primitive's existing shared and substrate-specific fields
        3. `actor_metadata` only for canonicalization and compact context

        WORKING METHOD
        - Resolve each primitive's `core_passage_ids` and `support_passage_ids` against `passage_list` before trusting the title. The evidence outranks elegant phrasing.
        - Treat `core_passage_ids` as the higher-authority grounding link and `support_passage_ids` as supplementary grounding.
        - First resolve any deferred substrate-detail fields for this substrate from `passage_list` and the primitive's existing fields. Then assign functions from the enriched primitive.
        - Decide what narrative work the primitive performs, not what kind of thing it is. Substrate is ontology; functions are narrative jobs.
        - Prefer fewer strong tags over padded breadth. Zero strong tags is better than three inflated ones.
        - Assign functions from concrete evidence and downstream structural pressure already visible in the primitive, not from fame, scale, or rhetorical force.
        - Choose the smallest justified `functions` set first. Then write paired payloads only for the functions you actually chose.
        - Add salience after function reasoning, not before. First decide the work, then decide how indispensable that work is.
        - If a tag feels almost right but the evidence is thin, drop it and keep the primitive lean.

        FUNCTION DEFINITIONS
        {function_guidance}

        FUNCTION-JUSTIFICATION DISCIPLINE
        - Every function tag must earn its paired justification payload. Do not rely on the tag name alone.
        - The contract is bidirectional: if a function tag is present, its paired payload must be present and non-null; if a payload is present, its matching function tag must appear in `functions`.
        - Strong justifications name the specific change, burden, damage, dispute, residue, or scene anchor this primitive carries.
        - Weak justifications are generic, circular, or interchangeable across many primitives in the substrate pass.
        - Do not copy the same justification pattern across unrelated primitives.
        - If the evidence cannot support a substrate-native, primitive-specific justification, drop the tag rather than padding the payload.
        - Leave unused justification fields null. Do not leave behind a draft payload from an earlier line of thought.

        DEFERRED SUBSTRATE DETAIL COMPLETION
        {indent(deferred_detail_completion, "        ").lstrip()}

        OUTPUT ASSEMBLY METHOD
        - Build each overlay in this order: resolve deferred substrate-detail fields, choose `functions`, fill the paired payloads for those functions, add `salience`, add `narration_hooks`, then confirm every unchosen function payload field remains null.
        - Do not emit a justification payload speculatively and decide the function tag later.
        - Do not keep a function tag as a placeholder if its paired payload is still generic, empty, or missing.
        - When `function_feedback` is present, fix only the named issue and keep all other valid unaffected overlays unchanged.

        SALIENCE SCORING METHOD
        Add `salience.score` and `salience.justification` for every primitive.
        Use these score bands:
        - `0.90-1.00`: indispensable load-bearing primitive; later series structure materially weakens without it.
        - `0.70-0.89`: high-value primitive with strong spine, scene, or explanatory leverage.
        - `0.50-0.69`: solid support primitive; useful but replaceable.
        - `0.30-0.49`: narrow or local primitive; worth keeping only for distinct texture, context, or contrast.
        - below `0.30`: weak or highly local primitive that survives mainly because this stage does not trim.
        - Compare primitives within this substrate pass rather than grading each one in isolation.
        - Reward primitives that are structurally reusable, explanatory, turn-bearing, or likely to anchor later planning.
        - Penalize primitives that are vivid but narrowly local unless the vividness itself does real narrative work.
        - Do not confuse fame, spectacle, or moral weight with actual structural indispensability.

        NARRATION HOOKS
        - Add narrator-agnostic `narration_hooks` when the evidence supports a reusable, concrete hook; keep hooks null only when the evidence is too thin to ground them.
        - `concrete_detail` should preserve the most speakable detail, image, gesture, phrase, or operating sign in the evidence.
        - `host_lens` should give one narrow orienting angle a narrator could use without turning the primitive into thesis prose.
        - `carry_forward` should name the residue, callback burden, or later pressure this primitive can carry.
        - `quote_anchor`, `plain_gloss`, and `listener_confusion` should stay sparse and only appear when the evidence clearly warrants them.
        - `authorial_move` must stay within the schema enum and should remain narrow rather than verdict-heavy.
        - Hooks must stay grounded in the supplied evidence and primitive fields; do not use them to invent new claims.

        SUBSTRATE-SPECIFIC FUNCTION LOGIC
        {_FUNCTION_TAGGING_SUBSTRATE_NOTES[substrate_value]}

        ANTI-OVERREACH RULES
        - Do not assign `pivot` to a primitive merely because it is important.
        - Do not assign `contest` to tonal disagreement, emphasis drift, or vague uncertainty.
        - Do not assign `recurrence` to generic thematic resemblance; there must be a plausible later return or residue.
        - Do not assign `complication` to ordinary sadness or complexity unless a clean judgment or option is genuinely compromised.
        - Do not assign `texture` unless the primitive can concretely anchor the listener in a world, scene, object, phrase, or visible situation.
        - Do not use substrate identity as a proxy for function. A primitive is not automatically entitled to a tag just because of its substrate.

        BORDERLINE CASES AND TIE-BREAKS
        - If choosing between `stake` and `cost`, use `stake` for exposure, risk, or what can still be lost; use `cost` for damage, payment, injury, humiliation, or foreclosed possibility already borne.
        - If choosing between `complication` and `contest`, use `contest` only when there are genuinely competing readings; otherwise use `complication` for a clean judgment that will not hold.
        - If choosing between `pivot` and high salience, reserve `pivot` for a primitive that materially changes later possibility rather than one that is merely central.
        - If tempted to assign `recurrence`, require plausible callback, residue, repetition, or memory burden beyond conceptual resemblance.
        - If a primitive is vivid but not structurally meaningful, `texture` may fit while salience stays modest.

        ANCHOR ENFORCEMENT
        - If the primitive's specificity is too thin, do not compensate by inflating functions.
        - Strip weak function claims rather than padding their justifications.
        - A function justification must be specific to this primitive and grounded in its supplied evidence.

        FAILURE MODES
        - Do not tag for rhetorical importance instead of narrative function.
        - Do not inflate weak primitives with extra tags to compensate for thin evidence.
        - Do not assign all major events as pivots or all vivid artifacts as texture.
        - Do not turn `salience.justification` into a second summary when it should name structural leverage.
        - Do not turn `narration_hooks` into script lines, verdicts, or a second layer of summary prose.
        - Do not preserve a tag if its paired justification payload has to stay generic.
        - Do not return a payload without its matching tag, or a tag without its matching payload.
        - Do not restate extraction-owned fields inside the overlay.

        SELF-CHECK BEFORE RETURNING
        {indent(self_check, "        ").lstrip()}

        OUTPUT CONTRACT
        Return only valid JSON matching `PrimitiveFunctionTaggingOverlayArtifact`, with top-level `overlays_by_id`.
        Prefer the compact transport keys above for repeated fields, but canonical schema names are also accepted.
        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()

def scene_discovery_instructions(
    *,
    candidate_target_min: int | None = None,
    candidate_target_max: int | None = None,
    podcast_mode: PodcastMode | str = PodcastMode.FULL,
) -> str:
    mode = PodcastMode(podcast_mode)
    if candidate_target_min is None or candidate_target_max is None:
        candidate_target_min, candidate_target_max = (
            16,
            24,
        ) if mode == PodcastMode.MINIFIED else (48, 72)
    candidate_range = _format_target_range(candidate_target_min, candidate_target_max)
    mode_label = "minified" if mode == PodcastMode.MINIFIED else "full"
    return dedent(
        f"""
        You are the `scene_discovery` stage for a historical podcast pipeline.

        TASK
        Discover a ranked, series-wide pool of concrete, playable historical
        moments before narrative strategy assigns episodes.

        This is a selection-and-synthesis stage, not a summarization stage.
        You are finding which evidence clusters can actually become audible,
        stageable scenes downstream.

        INPUT PAYLOAD
        - `project`: compact runtime and mode metadata
        - `actor_metadata`: compact canonical actor registry
        - `synthesis_map`: compact primitive payload with narration hooks and sceneable spoken gloss
        - `passage_list`: one shared deduped trimmed snippet list, each item only `passage_id` and `text`
        - `scene_discovery_feedback` (optional): retry feedback; if present, correct only the named contract failure and keep valid candidates unchanged

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated primitive fields. Treat these as equivalent:
          `sub`, `core`, `supp`, `pid`, `time`, `geo`, `actors`,
          `etype`, `event`, `act`, `subject`, `utter`, `goal`, `stakes`,
          `mech`, `cond`, `read`, `hooks`.

        MODE TARGET
        - This is a `{mode_label}` run.
        - Return {candidate_range} candidates.
        - Bias toward the lower-middle of that range unless the evidence surface is unusually rich and clearly supports more distinct playable moments.

        OUTPUT CONTRACT
        Return only valid JSON matching `SceneDiscoveryArtifact`.
        Output only `candidates`.

        Each candidate must include:
        - `candidate_id`
        - `primitive_ids`
        - `passage_ids`
        - `scene_sketch`
        - `scene_jobs`
        - `anchor_image`
        - `why_sceneable`
        - optional `quote_anchor`
        - optional `actor_ids`

        WHAT YOU ARE DISCOVERING
        - A candidate is not an episode assignment.
        - A candidate is not a section plan.
        - A candidate is not one primitive copied forward.
        - A candidate is not one passage restated in prose.
        - A candidate is not a heuristic coverage checklist.
        - A candidate is a derived, stageable scene hypothesis over referenced evidence.
        - A strong candidate has a concrete carrier and a clear oral payoff.

        SCENEABILITY CRITERIA
        A strong candidate usually has several of:
        - a visible carrier: room, object, document, gesture, public act, bodily consequence, or staged confrontation
        - active pressure: collision, dilemma, reveal, visible consequence, irreversible turn, or immediate aftermath
        - oral handle: easy to say, easy to remember, easy to re-enter later
        - bounded explanatory burden: it can be staged before it has to be explained
        - callback or residue potential
        - actor concentration: one or two focal actors rather than diffuse background

        Weak candidates usually fail because they are:
        - pure abstraction
        - mechanism without visible surface
        - summary without a moment
        - multiple unrelated primitives stapled together
        - a near-duplicate of a stronger candidate

        DISCOVERY WORKFLOW
        Internally work in this order:
        1. inventory sceneable surfaces in the primitives and snippets
        2. cluster evidence into distinct playable moments
        3. test each cluster for sceneability, not just importance
        4. assign likely scene jobs
        5. prune overlap and rank the survivors strongest-first

        MERGE VS SEPARATE
        - Merge when multiple primitives or passages clearly describe the same room, object, confrontation, immediate aftermath, or public act.
        - Separate when the evidence supports genuinely different playable moments.
        - Do not create two candidates that are the same moment with different rhetoric.
        - Separate only when the discovered scene itself is different, not merely because it could serve different explanatory purposes.

        ROLE DIVERSITY
        - `scene_jobs` may only use these structural scene roles: `opening`, `build`, `turn`, `answer`, `residue`. Never use primitive-function labels such as `cost`, `complication`, `stake`, `pivot`, `contest`, or `recurrence` here.
        - Aim for a varied pool, not a substrate quota:
          - multiple opening-capable candidates
          - at least one answer-capable candidate
          - at least one residue-capable candidate
          - multiple build-capable candidates when the evidence supports visible system surfaces
          - turn-capable candidates only when the moment visibly redirects the episode

        FIELD GUIDANCE
        - `scene_sketch`: 1-3 sentences, concrete, staged, and free of primitive jargon.
        - `anchor_image`: short and physical; not a thesis phrase.
        - `why_sceneable`: explain the oral payoff, not just the historical importance.
        - `quote_anchor`: include only when there is a genuinely airtime-worthy line or phrase.
        - `actor_ids`: include focal actors only, not an exhaustive cast list.
        - `primitive_ids` and `passage_ids`: include enough traceability to support the merged moment without overstuffing.

        USE OF SPOKEN GLOSS
        - Use `plain_gloss`, `why_it_matters`, `best_use`, and `natural_host_move` as routing help, not as evidence authority.
        - Those fields may help elevate, demote, or route a candidate.
        - Passages remain authoritative. If gloss and passage evidence conflict, follow the passages.

        RANKING AND PRUNING
        - Order candidates strongest-first.
        - Prefer candidates that can carry more than one plausible job without becoming vague.
        - Prefer candidates with distinct images, distinct actors, or distinct causal pressure.
        - Prefer overlap-light candidates. If two candidates are materially the same moment, keep the stronger one.
        - A build candidate must still open from something visible.
        - A residue candidate should leave after-pressure, irony, damage, or unresolved burden, not a second answer.

        FAILURE MODES
        - Do not return a heuristic coverage grid.
        - Do not emit one candidate per primitive by reflex.
        - Do not restate the passage snippet as summary prose if you can name the actual staged moment.
        - Do not produce episode sequencing.
        - Do not repeat passage text across multiple candidates.
        - Do not output abstract theme moments.
        - Do not emit build candidates with no visible entry handle.
        - Do not distribute candidates evenly across substrates by reflex.
        - Do not create a second primitive layer that merely renames the evidence.

        SELF-CHECK BEFORE RETURNING
        - Is each candidate a real moment rather than a primitive summary?
        - Does every `scene_jobs` entry exactly match one of: `opening`, `build`, `turn`, `answer`, `residue`?
        - Are answer and residue represented distinctly?
        - Are build candidates still visible and stageable?
        - Did you remove near-duplicate candidates?
        - Are the strongest candidates first?
        - Is the total count inside {candidate_range}?

        Return only the JSON object.
        """
    ).strip()


def narrative_strategy_skeleton_instructions(
    *,
    core_primitive_target_min: int = 8,
    core_primitive_target_max: int = 11,
    support_primitive_target_min: int = 9,
    support_primitive_target_max: int = 13,
    recall_primitive_target_max: int = 3,
) -> str:
    core_range = _format_target_range(
        core_primitive_target_min, core_primitive_target_max
    )
    support_range = _format_target_range(
        support_primitive_target_min, support_primitive_target_max
    )
    recall_limit = (
        "1 primitive"
        if recall_primitive_target_max == 1
        else f"{recall_primitive_target_max} primitives"
    )
    return dedent(
        f"""
        You are the `narrative_strategy_skeleton` stage for a historical podcast pipeline.

        Turn the flat primitive synthesis map into a series skeleton.

        This stage decides the structural series shape:
        - how many episodes there should be
        - what each episode is actually about
        - which primitives are core, support, or recall inside each episode
        - which actors need explicit arc directives
        - what each episode is consciously leaving out

        This stage does NOT decide:
        - narrator profile
        - explanation registries
        - actor introduction registry
        - authorial contract
        - narrator contract
        - listener/host progression agenda
        - promised beats

        Those belong to a later enrichment stage. Your job is to produce a structurally
        correct, evidence-backed series skeleton that a later stage can safely enrich
        without changing the partition.

        INPUT PAYLOAD
        - `synthesis_map`: primitive-only synthesis artifact with a flat `primitives` list
        - `project`: project metadata, runtime bounds, book metadata
        - `scene_discovery` (optional): compact global sceneability pool used only to pressure-test what is concretely stageable
        - `actor_metadata` (optional): canonical actor context
        - `requested_episode_count` (optional): hard episode-count constraint
        - `recommended_episode_count_min`: lower bound when no explicit count is requested
        - `recommended_episode_count_max`: upper bound when no explicit count is requested
        - `strategy_skeleton_feedback` (optional): retry feedback from the orchestrator; when present, it is binding corrective guidance for the retry
        - When `strategy_skeleton_feedback` says an episode is underfull, strengthen, merge, or reduce episode count rather than forcing a thin partition.
        - When `strategy_skeleton_feedback` says an episode is overfull, trim or demote primitives rather than adding more.

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated fields. Treat these as equivalent:
          `episode_spine -> spine`, `core_primitive_ids -> core_prims`,
          `support_primitive_roles -> support_roles`, `recall_primitive_ids -> recall_prims`,
          `scene_discovery -> scenes`,
          plus the primitive-field aliases `sub`, `core`, `supp`, `pid`, `time`, `geo`, `actors`,
          `etype`, `event`, `act`, `subject`, `utter`, `goal`, `stakes`, `mech`, `cond`, `read`, `hooks`.
        - In your JSON output, prefer canonical field names: `episode_spine`, `core_primitive_ids`,
          `support_primitive_roles`, `recall_primitive_ids`, and `negative_scope`.

        EPISODE COUNT
        - If `requested_episode_count` is present, treat it as binding.
        - Otherwise, produce between `recommended_episode_count_min` and `recommended_episode_count_max` episodes, inclusive.

        PRIORITY RULES
        - Every episode must have exactly one `episode_spine`.
        - `listener_problem` is the listener-facing problem the episode carries.
        - `episode_answer` is the concise answer the episode earns.
        - `pressure_line` is the live contradiction or pressure the listener should feel while moving through the episode.
        - `core_primitive_ids` are the binding episode contract.
        - Support primitives must be typed with exactly one role each:
          `stakes`, `mechanism`, `counterpressure`, `consequence`, or `texture`.
        - Infer support role and recall eligibility from primitive substrate, functions,
          title, substrate-specific fields, salience, passage grounding, actor context, and time/place context.
        - Use `scene_discovery` only to pressure-test whether an episode is concrete enough to survive downstream audio realization.
        - Do not produce section topology, scene-card ordering, prose, narrator method, registries, or promised beats.

        SUPPORT ROLE DEFINITIONS
        - `stakes`: raises what can be lost, protected, or irreversibly changed
        - `mechanism`: explains how the episode’s proposition works in practice
        - `counterpressure`: supplies the force, contradiction, or rival logic resisting it
        - `consequence`: shows what the proposition causes, unlocks, or damages downstream
        - `texture`: adds grounded lived detail or context that strengthens the same proposition without carrying it alone

        FIRST-PASS GROUPING WORKFLOW
        - Draft the series using primitives first.
        - Build provisional episodes from primitive title, substrate-specific fields, substrate, functions,
          actor context, time/place context, causal sequence, and passage grounding.
        - Infer thematic coverage from the primitive set itself rather than from external thematic scaffolds.

        PRIMITIVE-FIRST DISCIPLINE
        - Do not recreate abstract thematic buckets as episode titles.
        - Do not let repeated conceptual vocabulary become the organizing outline unless the primitives independently force it.
        - Prefer concrete historical problems, reversals, collisions, decision chains, and causal transformations over abstract thematic bucket labels.

        SERIES SHAPE
        Choose one `strategy_type`:
        - `thesis_driven`
        - `debate`
        - `chronological`
        - `convergence`
        - `mosaic`

        You must actively compare at least:
        - `chronological`
        - `thesis_driven`
        - one other plausible shape

        Do not default to `thesis_driven`.

        SERIES SHAPE RULES
        - Build episodes around escalation, consequence, contestation, discovery, and payoff.
        - Do not partition material evenly.
        - Do not organize episodes as biographies unless the evidence genuinely warrants it.
        - Keep each episode centered on one main problem.
        - Secondary pressures should sharpen that problem, not open a neighboring one.

        EVIDENCE BALANCE
        Balance evidence across episodes by sufficiency, not equality.
        - Do not let one episode become evidence-thin while another absorbs reusable support with much deeper grounding.
        - Prefer support sets that introduce new passage grounding or materially broaden interpretive coverage instead of stacking near-duplicate argumentative work.
        - If an episode’s primitive mix is too concentrated on one local cluster or one local job, reassign support before accepting the partition.
        - If two episodes could plausibly host the same non-core primitive, prefer the assignment that strengthens the weaker episode’s evidence base.
        - Prefer the smallest support set that materially broadens grounding, pressure, or consequence.
        - Do not stack multiple support primitives that perform the same argumentative job unless they add genuinely distinct grounding.
        - If an episode is elegant but under-grounded, strengthen grounding first.

        SCENEABILITY PRESSURE
        When `scene_discovery` is present:
        - Use it only as a reality check on whether an episode has enough concrete carriers to survive downstream audio realization.
        - Prefer partitions whose core can plausibly yield an opening image, a turn-bearing moment, and an answer-bearing visible carrier.
        - Do not convert scene candidates into promised beats here.
        - Do not reshape the series around scene discovery alone if the primitive logic says otherwise.
        - If an episode is structurally smart but starved of anything stageable, fix the primitive partition now rather than expecting a later stage to rescue it.

        EPISODES
        Each episode must include:
        - `episode_number`
        - `title`
        - `thematic_focus`
        - `arc_summary`
        - `unresolved_questions`
        - `episode_spine`
        - `actor_arc_directives`
        - `negative_scope`

        Do NOT include:
        - `narrator_contract`
        - `authorial_contract`
        - `narrative_agenda`
        - `promised_beats`
        - top-level narrator or registry fields

        NEGATIVE SCOPE
        `negative_scope` tells downstream stages what this episode is consciously not trying to carry.
        It must include:
        - `boundary`
        - `excluded_topics`
        - `tempting_but_out`
        - `omission_logic`

        EPISODE SPINE RULES
        - `core_primitive_ids` must contain {core_range} primitives.
        - `core_primitive_ids` is the episode's thesis-bearing contract, not a holding area for every later-useful primitive.
        - Each episode core must be pivot-led.
        - At least one core primitive must use substrate `events` or `acts`.
        - Prefer at least two core primitives tagged `pivot` whenever the evidence supports it.
        - `support_primitive_roles` must contain {support_range} primitives.
        - Each support primitive gets exactly one support role.
        - Support primitives cannot also appear in the core.
        - `recall_primitive_ids` are optional and must contain at most {recall_limit}.
        - Use recall only when it is explicitly justified and materially helps the listener carry accumulated meaning forward.
        - Prefer recall primitives carrying `recurrence`. Do not use recall as recap.
        - Do not use `core_primitive_ids` to preserve optional downstream material for enrichment.
        - If an episode needs more than {core_range} core primitives to feel viable, the partition is wrong; rebalance, merge, or reduce episode count.
        - If both core and support are overfull, reduce episode scope before adding more primitives.

        ACTOR ARC DIRECTIVES
        Actor arc directives are episode-specific planning guidance for how selected actors function across scenes.
        - `actor_arc_directives` must contain only the 2-4 actors whose episode function needs explicit planning guidance.
        - Choose actors who give the episode a usable character spine.
        - Do not include an actor just because they appear in clusters or primitives.

        QUALITY
        - Keep the listener-facing problem narrow and concrete.
        - Build each episode around one controlling proposition expressed through one explicit set of core primitives.
        - Keep support subordinate, but do not confuse subordination with piling up multiple primitives that make the same point.
        - If a later episode becomes too dense, demote non-thesis material into support before expanding core.
        - If a proposed episode is weak, fix the partition rather than compensating by inflating core.
        - Do not let tail episodes become everything left over.
        - Let `pivot` drive cores, `stake` and `cost` drive human grounding, `mechanisms` and `conditions` drive explanation, `contest` and `complication` drive pressure, and `recurrence` drive callbacks and later memory burden.
        - `arc_summary` should explain the episode shape, but it should not become a disguised narrator script.
        - `unresolved_questions` should name live uncertainty or deferred consequence, not generic teaser copy.
        - Use `negative_scope` to prevent neighboring attractive material from bleeding into an episode that already has enough load-bearing work.
        - If a proposed episode sounds like an essay topic instead of an oral historical problem, rescope it now.

        OUTPUT
        Return only valid JSON matching `NarrativeStrategySkeleton`.
        - `strategy_type` must be one of the schema values.
        - If `recommended_episode_count` is present, it must match the number of episodes produced.
        - If `episode_arc_outline` is present, it must align in length with `episodes`.
        - Each episode must include `negative_scope`.
        - Do not add markdown or commentary.
        """
    ).strip()


def narrative_strategy_enrichment_instructions(
    *,
    authorial_passage_target_min: int = authorial_passage_target_range_for_mode(
        PodcastMode.FULL
    )[0],
    authorial_passage_target_max: int = authorial_passage_target_range_for_mode(
        PodcastMode.FULL
    )[1],
    podcast_mode: PodcastMode | str = PodcastMode.FULL,
) -> str:
    mode = PodcastMode(podcast_mode)
    authorial_range = _format_target_range(
        authorial_passage_target_min, authorial_passage_target_max
    )
    if mode == PodcastMode.MINIFIED:
        authorial_guidance = (
            f"For minified episodes, `target_authorial_passages_per_episode` should "
            f"usually land around {authorial_range}."
        )
    else:
        authorial_guidance = (
            "For full-length episodes running about 130-150 minutes at the pipeline's spoken-rate targets, "
            f"`target_authorial_passages_per_episode` should usually land around {authorial_range}."
        )
    return dedent(
        f"""
        You are the `narrative_strategy_enrichment` stage for a historical podcast pipeline.

        Take an already-fixed series skeleton and enrich it into a complete narrative strategy.

        This stage does NOT decide the partition. The skeleton is binding.
        You may not:
        - change episode count
        - change episode order
        - change titles
        - change thematic focus
        - change arc summaries
        - change unresolved questions
        - change episode spines
        - change actor arc directives
        - change negative scope
        - reassign primitives across episodes

        Your job is to add the layers that make the skeleton playable as a podcast season:
        - narrator profile
        - reusable term and institution registry
        - reusable actor introduction registry
        - per-episode narrator contract
        - per-episode authorial contract
        - per-episode listener/host progression agenda
        - sparse promised beats anchored in episode-scoped scene candidates

        INPUT PAYLOAD
        - `strategy_skeleton`: binding structural series skeleton
        - `synthesis_map`: skeleton-selected primitive subset only
        - `project`: project metadata, runtime bounds, book metadata
        - `episode_scene_candidates`: per-episode scene-candidate pools already filtered from global scene discovery
        - `actor_metadata` (optional): canonical actor context
        - `strategy_enrichment_feedback` (optional): retry feedback from the orchestrator; when present, it is binding corrective guidance for the retry

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated fields. Treat these as equivalent and prefer them in your JSON output when possible:
          `strategy_skeleton -> skeleton`, `episode_scene_candidates -> episode_scenes`,
          `episode_spine -> spine`, `core_primitive_ids -> core_prims`,
          `support_primitive_roles -> support_roles`, `recall_primitive_ids -> recall_prims`,
          `series_explanation_registry -> term_registry`,
          `series_actor_explanation_registry -> actor_registry`,
          `source_candidate_ids -> source_candidates`,
          `source_primitive_ids -> source_prims`,
          `promised_beats -> promised`,
          plus the primitive-field aliases `sub`, `core`, `supp`, `pid`, `time`, `geo`, `actors`,
          `etype`, `event`, `act`, `subject`, `utter`, `goal`, `stakes`, `mech`, `cond`, `read`, `hooks`.

        BINDING IMMUTABILITY RULES
        - Treat `strategy_skeleton` as authoritative structure.
        - Do not silently rewrite a weak episode by changing its primitive assignment or implied scope.
        - Do not introduce a new top-level strategy type or series arc.

        ENRICHMENT OBJECTIVE
        You are converting a structural partition into a podcast-usable editorial contract.

        The enriched output must answer:
        - what kind of host mind is carrying the season
        - what explanatory burdens must be explicit versus assumed
        - which actors need a real first introduction and where
        - which questions, mysteries, assumptions, and theories are moving across the season
        - which concrete beats the episode is now explicitly promising to stage or pay off

        You are not drafting sections or scene cards. You are deciding what downstream architecture and planning must honor.

        NARRATOR METHOD
        Return a top-level `narrator_profile`. Derive a usable narrator method from:
        - primitive density
        - chronology
        - institutional load
        - actor pressure
        - doctrinal load
        - the skeleton's pressure lines and answer shapes

        Infer:
        - whether the season is mostly scene-led or needs more explicit explanation
        - which episodes need stronger doctrinal unpacking, institutional clarification, quote-then-gloss, or harder verdict landings
        - how much explicit host presence the material can carry
        - how much authorial passage density a listener needs to stay oriented in one hearing

        NARRATOR PROFILE RULES
        - `presence_mode` should default to `visible_host`.
        - `spoken_style_contract` should default to `anti_academic_oral`.
        - Prefer `baseline_tone = plainspoken` unless the material is overwhelmingly atrocity-led or testimonial in a way that truly requires graver surface phrasing.
        - `allowed_moves` must contain only:
          `orient`, `clarify`, `evaluate`, `contrast`, `callback`, `light_aside`,
          `naming_note`, `uncertainty`, `revision`, `surprise`.
        - If `wit_ceiling` is `dry` or `wry`, include `light_aside` in `allowed_moves`.
        - The narrator profile must define method, not just tone.
        - {authorial_guidance}

        EXPLANATION REGISTRY
        Return a top-level `series_explanation_registry` for only the most reusable terms or institutions.
        - Use it sparingly.
        - Give each item one canonical introduction episode.
        - Later episodes should usually remind rather than fully redefine.
        - `preferred_plain_gloss` must sound like something the host could actually say aloud.

        ACTOR EXPLANATION REGISTRY
        Return a top-level `series_actor_explanation_registry` for only the few people whose first naked mention would likely confuse the listener.
        - Use it sparingly.
        - Give each selected actor one canonical introduction episode.
        - Use `first_background_depth = appositive` when one clause is enough and `full` only when the listener truly needs a short background sentence.

        EPISODE ENRICHMENTS
        Return one enrichment record per skeleton episode. Each enriched episode must include:
        - `episode_number`
        - `narrator_contract`
        - `authorial_contract`
        - `narrative_agenda`
        - `promised_beats`

        Do not repeat structural skeleton fields in the output.

        PROMISED BEATS
        `promised_beats` are the sparse set of concrete historical obligations this episode is explicitly promising to stage or pay off downstream.

        Each item must include:
        - `beat_id`
        - `label`
        - `intended_job`: `opening`, `build`, `turn`, `answer`, `residue`, or `close`
        - `source_candidate_ids`
        - `source_primitive_ids`
        - `why_load_bearing`

        PROMISED BEAT RULES
        - Use the per-episode `episode_scene_candidates` pool first.
        - Every promised beat must cite at least one source candidate or source primitive.
        - Prefer source candidates when a concrete carrier exists.
        - Make `promised_beats` sparse.
        - Usually land around 2-4 promised beats per episode unless the episode truly needs more structural commitments.
        - Use at most one promised beat with `intended_job = answer`.
        - Use at most one promised beat with `intended_job = residue`.

        NARRATIVE AGENDA
        Use `narrative_agenda` as the episode-level listener/host progression contract.

        LISTENER AGENDA
        The listener side must answer:
        - what the listener is newly allowed to understand
        - which terms/institutions get introduced versus merely reminded
        - which actors get a real first introduction versus only a reminder
        - what question is opened, advanced, resolved, or reframed
        - what memory thread is opened, refreshed, paid off, or retired
        - what carry-forward memory burden the next episode should inherit
        - what the listener’s episode takeaway should be

        HOST AGENDA
        The host side must answer:
        - what the host is still trying to understand
        - what assumption is being weakened or revised
        - what working theory is being proposed, strengthened, replaced, or retired
        - what explicit revision beats should be felt across the season
        - what the host’s episode takeaway is

        HOST MOVE OUTPUT RULES
        - `mystery_moves.open` must include `text`
        - `assumption_moves.introduce` must include `statement`
        - `assumption_moves.revise` must include both `statement` and `revised_statement`
        - `theory_moves.propose` and `theory_moves.replace` must include `statement`

        QUALITY
        - Keep the enrichment layer faithful to the skeleton’s actual episode problem.
        - Prefer a host who sounds like a distinct spoken mind carrying pressure forward, not polished page prose softened for audio.
        - Keep registries sparse and reusable.
        - Make promised beats concrete enough to matter and few enough to remain binding.
        - If a beat, registry item, or agenda move would not materially change downstream architecture or planning, leave it out.
        - If `strategy_enrichment_feedback` is present, revise only the invalid episodes or invalid items and keep unaffected content unchanged.

        OUTPUT
        Return only valid JSON matching `NarrativeStrategyEnrichment`.
        - `narrator_profile` must define narrator method, not just tone.
        - `series_explanation_registry` is top-level output, not per-episode output.
        - `series_actor_explanation_registry` is top-level output, not per-episode output.
        - Every episode must include `promised_beats`.
        - Before returning JSON, verify that every actor id used in any episode's `introduce_actor_ids` or `remind_actor_ids` appears exactly once in the top-level `series_actor_explanation_registry`.
        - Do not restate skeleton fields that are already fixed upstream.
        - Do not add markdown or commentary.
        """
    ).strip()


def episode_planning_instructions(
    *,
    scene_card_target_min: int = 40,
    scene_card_target_max: int = 48,
) -> str:
    scene_card_range = _format_target_range(
        scene_card_target_min, scene_card_target_max
    )
    return dedent(
        f"""
        You are the `episode_planning` stage of a historical podcast pipeline.

        Your job: turn one episode architecture into a framing block plus a sequence
        of concrete, playable scene cards. The architecture's section topology and
        `episode_spine` are binding structure. You are giving that architecture
        scene-level shape, not reconsidering proposition selection.

        ==============================================================================
        INPUT PAYLOAD
        ==============================================================================
        - `strategy_episode`        one episode object from `narrative_strategy`
        - `architecture`             one episode architecture object
        - `synthesis_map`            primitive-first synthesis map filtered to this episode
        - `project`                  theme, sub-themes, book metadata, duration goals
        - `scene_job_budget`         explicit scene-job allocation contract for this mode
        - `available_passages`       evidence available to this episode
        - `host_policy`              series-level narrator policy for host moves
        - `narrative_state_pre`      authoritative listener/host state entering this episode, already reconciled after the previous episode's architecture
        - `continuity_contract_pre`  compact continuity obligations derived from the incoming state
        - `actor_metadata`           episode-relevant canonical actor context
        - `planning_feedback`        optional retry feedback from the orchestrator; if present, correct only the named contract failure and preserve valid structure
        - Optional `field_semantics` explicit definitions for `closure_mode`,
          fact tiers, withholding, and word-count priority

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated fields. Treat these as equivalent and prefer them in your JSON output when possible:
          `passage_id -> pid`, `passage_ids -> passages`, `priority_core_passage_ids -> priority_core`,
          `source_passage_ids -> source_passages`, `source_primitive_ids -> source_prims`,
          `must_land_facts -> facts`, `scene_cards -> scenes`,
          `dropped_support_primitive_reasons -> dropped_support`,
          plus the primitive-field aliases `sub`, `core`, `supp`, `time`, `geo`,
          `etype`, `event`, `act`, `subject`, `utter`, `goal`, `stakes`, `mech`, `cond`, `read`, `hooks`.
        - For compact scene-plan output, prefer these aliases when possible:
          `episode_number -> ep`, `framing -> frame`, `scene_id -> sid`,
          `section_id -> sec`, `title -> ttl`, `scene_role -> role`,
          `scene_job -> job`, `beat_change -> beat`, `required -> req`,
          `strongly_preferred -> pref`, `if_room -> room`,
          `entry_image -> img`, `observable_detail -> detail`,
          `estimated_duration_seconds -> dur`, `host_moves -> moves`,
          `move_type -> type`, `target -> tgt`, `surface_mode -> surf`,
          `address_mode -> addr`, `opening_image -> open_img`,
          `opening_question -> open_q`, `handoff_scene_card_id -> handoff`,
          `answer_scene_card_id -> answer_sid`,
          `residue_scene_card_id -> residue_sid`.

        ==============================================================================
        PRIORITY RULES — WHAT IS BINDING vs. WHAT YOU OWN
        ==============================================================================
        BINDING (do not restate, rewrite, reorder, or reconsider):
        - `strategy_episode` fields: title, thematic focus,
          arc summary, unresolved questions, `episode_spine`, actor arc directives,
          and `narrative_agenda`.
        - `architecture.section_id` order.
        - `architecture.major_turn_section_id`.
        - Per-section primitive groupings (`primitive_ids` lists).
        - Per-section state obligations: `question_moves`, `memory_thread_moves`,
          `host_mystery_moves`, `host_assumption_moves`, and `host_theory_moves`.

        YOU OWN:
        - Framing block (opening image, threat, opening question, handoff target).
        - Scene cards: count, ordering within sections, titles, dramatic roles,
          scene jobs, durations, evidence selection, scene actors, lean beat changes,
          tiered must-land facts, scene-local authorial assignments, withholding,
          word-count priority, and host-move permissions with explicit placement.
        - `answer_scene_card_id` and `residue_scene_card_id`.
        - The `dropped_support_primitive_reasons` register.
        - Staging scenes so planned section-level explanatory passages are earned by the surrounding evidence.

        OUTPUT: only `episode_number`, `framing`, `scene_cards`,
        `answer_scene_card_id`, `residue_scene_card_id`, and
        `dropped_support_primitive_reasons`. Nothing else.
        Every scene card must be grounded in provided `passage_ids`.

        ==============================================================================
        FRAMING
        ==============================================================================
        - `opening_image` — concrete and scene-led; a thing the listener can see.
        - `threat_or_unresolved_action` — keeps the episode in motion; something not
          yet resolved.
        - `opening_question` — should create curiosity in the same territory as
          `strategy_episode.episode_spine.listener_problem`. Do not paraphrase it.
        - `handoff_scene_card_id` — must point to a real scene card you produce.
        - `recap` — for Episode 1, set this to null. For later episodes, when
          `continuity_contract_pre.recap_items` is non-empty, write a 1-2
          sentence spoken recap that recalls inherited pressure or memory
          without turning into teaser copy.
        - The framing should orient the listener without pre-explaining the episode.
          Do not preview the thesis, the turn, or the closing.

        ==============================================================================
        SCENE CARDS — STRUCTURE
        ==============================================================================
        COUNTS
        - Target {scene_card_range} scene cards for this episode.
        - Treat `scene_job_budget` as binding. Allocate cards against that budget,
          not just the total count.
        - Expand into playable micro-scenes. Do not collapse long stretches into one
          card.
        - Use the extra cards to separate mechanism from consequence, definition
          from payoff, and host reorientation from factual pressure.
        - Do not split for its own sake; prefer one clean job per card.

        SECTION BOUNDARIES
        - Use `architecture.section_id` as the only grouping boundary.
        - All scene cards for a given `section_id` must be contiguous.
        - Every architecture section must yield at least one scene card.
        - Treat `architecture.section_anchor` as the section-local opening handle.
          It is distinct from the episode framing `opening_image`.
        - Treat each section as a binding local brief: its scenes must collectively
          realize that section's `must_stage_beats`, `closure_mode`, and any
          planned `authorial_passages`, `term_explanations`, `actor_explanations`,
          `question_moves`, `memory_thread_moves`, `host_mystery_moves`,
          `host_assumption_moves`, `host_theory_moves`, or `key_terms`.
        - Build each section through accumulation. The first scene should open
          from the anchor and the final beat should cash out the last
          `must_stage_beats` item or land the assigned `closure_mode`.

        CLOSING SECTION (special rules)
        - The final architecture `closing` section must expand to exactly one scene card.
        - That scene card must be the episode's last scene card and must use `scene_job = close`.
        - It must keep `estimated_duration_seconds` ≤ 120.
        - It may land verdict, payoff, or consequence.
        - It must NOT introduce a fresh mechanism, counterpressure chain, parallel
          argument, new institution, or new actor thread. (This is the two-endings
          failure.)
        - The closing card may exit, constrain, or reframe. It may not perform the answer job again.

        ==============================================================================
        SCENE CARDS — REQUIRED FIELDS
        ==============================================================================
        Every scene card must set:

          `section_id`                  one architecture section
          `title`                       short, concrete card title; usually 2-5 words
          `scene_role`                  dramatic beat type (see canonical values)
          `scene_job`                   structural job (see canonical values)
          `beat_change`                 short operational statement of what materially
                                        changes in this beat; usually 4-8 words
          `must_land_facts.required`    usually 1–3 load-bearing facts the scene fails without
          `must_land_facts.strongly_preferred`
                                        optional; usually 0–1 extra fact when it materially sharpens the scene
          `must_land_facts.if_room`     optional; usually omitted unless one contextual fact clearly helps
          `word_count_priority`         `default` unless this scene truly needs `tight`
          `estimated_duration_seconds`  integer; you allocate
          `passage_ids`                 enough evidence to support later writing

        Treat `must_land_facts` as the card's factual spine.
        Treat `must_land_facts.required` as the card's factual spine.
        Omit `strongly_preferred` unless the scene genuinely needs one extra fact.
        Omit `if_room` by default; use it only when dropping the fact would
        materially reduce flexibility under length pressure. `host_moves` may prioritize, clarify, contrast,
        or land those facts, but they may not replace them or become a second
        fact list. If the beat grows denser than that default, keep it playable and orally manageable rather than stuffing one card by reflex.

        Optional scene-level fields:
          actors[], entry_image, observable_detail, timeframe,
          location, withhold_until, host_moves,
          primitive_ids, authorial_passage_ids
        - Omit optional fields entirely instead of returning blank strings or empty arrays.
        - When present, keep `entry_image` and `observable_detail` to one short clause each.

        When a section has `actor_explanations`, pin each one to the earliest
        concrete scene where that actor materially lands on-mic.
        - Put the obligation on the matching `actors[]` item for that scene.
        - Use `explanation_stage = introduce` or `reminder`.
        - Carry the section plan's `background_depth`, `role_label`,
          `source_primitive_ids`, `source_passage_ids`, `intro_facts`, and
          `why_now` onto that scene actor as scene-local control metadata.
        - Do not add copied registry glosses or lifted prose when placing the
          scene actor; keep the actor entry specific to the scene's own burden.
        - Do not leave actor-introduction work as loose section intent once you can
          place it on a concrete scene actor.
        - When a section has `authorial_passages`, assign every
          `authorial_passage_id` to exactly one scene in that same section using
          `authorial_passage_ids[]`.
        - A scene carrying `comparative_aside` should usually have one concrete anchor,
          one clear return path, and `word_count_priority = default`.
        - If a scene carries both `comparative_aside` and another heavy explanatory
          passage such as `quote_then_gloss`, `doctrinal_unpack`,
          `institutional_clarifier`, or `verdict_landing`, split them into adjacent
          scenes unless this is an intentionally benchmarked answer scene with room.
        - For `scene_job = answer`, use `comparative_aside` only when it sharpens the
          answer and returns immediately to the scene's own pressure.
        - Use structured `withhold_until` only when delayed legibility improves
          the scene. Set `subject`, `reveal_phase`, optional
          `surrogate_label`, and `reveal_scene_id` only when the reveal belongs
          to a later scene.

        CANONICAL `scene_role` VALUES:
          context_setup, actor_setup, action, shock, contestation, reaction,
          fallout, implication

        Legacy scene-function vocabulary you may still see in older notes:
          scene, hinge, mechanism, turn, landing, callback, afterlife

        CANONICAL `scene_job` VALUES:
          opening, build, turn, answer, residue, close

        FIELD DISTINCTION
        - `scene_role` is the dramatic beat type.
        - `scene_job` is the coarse structural job.
        - Valid example: `scene_role = "contestation"` with `scene_job = "build"`
        - Invalid swap: `scene_role = "mechanism"` or `scene_job = "contestation"`
        - Exactly one card must use `scene_job = answer`.
        - Exactly one later card must use `scene_job = residue`.
        - Exactly one final card must use `scene_job = close`.
        - `answer_scene_card_id` must point to the `answer` card.
        - `residue_scene_card_id` must point to the `residue` card.

        HOST MOVES
        - `host_policy` is binding narrator policy for density, tone, and pronouns.
        - `narrative_state_pre` is read-only continuity context. Use it to preserve
          carry-forward memory and open host pressure, not to invent fresh season-state
          changes.
        - `continuity_contract_pre` is the compact version of that burden.
          Prefer it when deciding what must surface early, especially in
          `framing.recap` and the opening scenes.
        - Planning may realize state commitments but may not invent new
          listener-question, memory-thread, or host-state progression not already
          present in strategy or architecture.
        - `host_moves` are required scene design, not optional garnish.
        - Each scene card must include `host_moves` with phase buckets:
          - `open`: how the scene enters and what is foregrounded first
          - `pivot`: what becomes clearer after concrete material lands
          - `close`: what residue, verdict, callback, or pressure remains
        - Ordinary cards should usually use one populated phase. Heavier cards may use two or all three only when the scene clearly needs them.
        - Default to one populated phase and one cue per populated phase.
        - Fresh plans should use at most one cue per phase. If two ideas compete, compress them into one better cue.
        - Return only populated phase buckets; omit empty `open`, `pivot`, and `close` keys.
        - Major turns, openings, closings, and explanation-heavy cards may use
          more than one populated phase when they clearly need it, but every scene
          must populate at least one phase.
        - If a section carries any `host_mystery_moves`, `host_assumption_moves`, or
          `host_theory_moves`, at least one scene in that section must carry non-empty
          `host_moves`.
        - Use `allowed_moves` from `host_policy` as binding. Do not emit a move
          type the narrator policy does not allow.
        - Every cue must include:
          - `move_type`
          - `target`
          - `surface_mode`: `woven`, `distinct`, or `mixed`
          - `address_mode`: `implicit`, `we`, `you`, or `i`
        - `surface_mode = woven` means the host mostly shapes diction, emphasis,
          and residue from inside the narration.
        - `surface_mode = distinct` means one clearly audible host line or phrase
          should survive in that phase.
        - `surface_mode = mixed` means the cue may surface distinctly, but should
          still shape the whole phase.
        - `target` is a compressed structural permission, not final copy.
        - `target` should usually be 1-4 words, must never exceed 6, and should compress the move into a short phrase rather than planning prose.
        - `address_mode = you` is useful for guide-like explanation.
        - `address_mode = we` is useful for shared inference, companionable
          guidance, callbacks, reorientation, and closings.
        - `address_mode = i` is useful for taste-bearing evaluation, candid
          uncertainty, quick comparison, and light asides that keep the
          narrator inhabited.
        - Prefer `clarify` after complexity, `contrast` when killing a false
          reading, `evaluate` when consequence needs a clean landing, and
          `callback` only when real distance or memory pressure makes the return sharper.
        - Prefer `surface_mode = woven` by default. Use `distinct` only when the
          surviving host clause would still sound natural if spoken aloud.
        - First- and second-person language is allowed throughout when it stays
          brief, scene-rooted, and earns its keep.

        ==============================================================================
        SCENE CARDS — DURATION ALLOCATION
        ==============================================================================
        Allocate duration so the proposition chain carries the runtime.

        Classify a scene by what it does for the listener:
        - opens pressure
        - sharpens pressure
        - redirects pressure
        - lands a consequence
        - carries a turn

        The `closing` scene's ≤120s cap still applies.

        ==============================================================================
        SCENE CONSTRUCTION CRAFT
        ==============================================================================
        - Most scene cards should do one job cleanly: establish, pressure, reveal, decide, rupture, react, or show consequence.
        - Prefer scene cards that can be narrated from something the listener can see: a person, room, object, document, journey, or dated moment.
        - Leave something live for the next card: a question, threat, expectation, or consequence.
        - `actor_setup`, `action`, and `fallout` scenes normally have at least one actor.
        - Use `contestation` only when the disagreement can be staged through actors, texts, councils, trials, letters, accusations, or rival actions, never as a narrator-side literature review.

        SCENE JOBS
        - `opening` enters pressure quickly and visibly.
        - `build` carries most setup, mechanism, consequence, contest, callback, and explanatory load.
        - `turn` marks the rerouting beat where the balance materially changes.
        - `answer` is the one card where the listener problem is actually resolved.
        - `residue` leaves the cost, irony, after-pressure, or unresolved burden alive after the answer.
        - `close` exits cleanly without reopening the answer.
        - A `build` card may still stage mechanism or callback work, but it must remain concrete.
        - `turn`, `answer`, `residue`, and `close` cards must still open from something visible and set both `entry_image` and `observable_detail`.
        - Do not let structural cards turn into free-floating thesis cards or descriptive backgrounding.

        TEXTURE-ONLY CARDS
        Allowed only when they still serve the same proposition the spine is advancing.

        HUMAN GROUNDING
        If the episode leans heavily on structural cards, ensure at least one `build` or `turn` card centers lived pressure, cost, fear, choice, or bodily consequence through named actors plus concrete image/detail.

        ==============================================================================
        OPENING SCENES (first 2–3 cards)
        ==============================================================================
        The opening must SHOW, not FRAME.

        Prefer anomaly, pressure, risk, visible change, or a person doing something the listener would not have predicted.
        Avoid debate-setup openings, historiography/framework openings, baseline-plus-theory openings, or any scene whose only job is "the listener needs to know X before we begin."

        The opening scene that the framing's `handoff_scene_card_id` points to
        should be one the listener can step into immediately.

        ==============================================================================
        PRIMITIVES AND EVIDENCE
        ==============================================================================
        SOURCING
        - Ground every scene in the provided `passage_ids`.
        - Use architecture `primitive_ids` as binding planning context.
        - Scene cards should usually carry `primitive_ids`.
        - Only setup/connective cards such as `context_setup`, `actor_setup`, or comparable transitional scenes may omit `primitive_ids`, and even then only when the card genuinely exists to bridge already-selected material.

        PRIORITY
        - When a section has `priority_core_passage_ids`, prefer those passages
          first when they fit the scene's job.

        WHICH PRIMITIVES TO PICK
        - Prefer concrete substrates for scene entry: `events`, `acts`, `utterances`, and `artifacts`, especially when tagged `texture` or `pivot`.
        - Use `cost`, `stake`, and actor-centered primitives to keep scenes from going purely abstract or institutional.
        - Use `mechanisms`, `conditions`, and `readings` to structure explanation only when the scene still opens from something visible.
        - Use substrate-specific fields directly:
          `what_happened`/`event_result` for event payoff,
          `act_summary`/`immediate_result` for choice framing,
          `utterance_summary`/`key_quote` for speech-driven scenes,
          `goal_or_project`/`stakes_or_fears` for actor pressure,
          `operating_chain` for mechanism staging,
          `condition_summary`/`active_tension` for standing-pressure scenes,
          `artifact_label`/`artifact_detail` for entry images and callbacks,
          `reading_summary` for contest or interpretation scenes.
        - Prefer `narration_hooks.concrete_detail` for `observable_detail` when present.
        - Use `recurrence`-tagged primitives for openings, callbacks, handoffs, and closings when the evidence supports explicit return.

        DISTRIBUTION
        - Do not distribute scenes evenly by default.
        - Do not pack multiple unrelated beats into one scene by reflex. Split when density makes the scene stop feeling playable, visible, or orally manageable.

        DROPPING SUPPORT PRIMITIVES
        - A support primitive may be dropped only if it remains available inside
          the architecture-defined section set and does not fit the spine chain.
        - Every dropped primitive needs an explicit reason in
          `dropped_support_primitive_reasons`.

        ==============================================================================
        BEAT CHANGE (beat_change)
        ==============================================================================
        `beat_change` is what's NEWLY IN PLAY by the scene's end: a fact, pressure,
        contradiction, alignment, loss, risk, or consequence the next scene can
        pick up.

        Good (concrete, short, advances state):
          "The viceroy now has written authorization but no troops."
          "The two factions are publicly aligned for the first time."
          "The price ceiling has triggered a black market in the capital."

        Bad (verdict, mini-thesis, summary):
          "By the end of the scene, the listener understands the deeper paradox..."
          "This shows that the revolution was inevitable."
          "The episode reveals a contradiction at the heart of the regime."

        ==============================================================================
        ACTORS IN SCENES
        ==============================================================================
        SCENE ACTORS
        - Use `actors[]` for scene actors.
        - Include `actor_id` only when the listed actor exists in `actor_metadata`.
        - Each actor's `presence` is `primary`, `secondary`, or `background`.
        - Not every scene needs actors.

        Each `actors[]` item may include:
          name
          actor_id
          affiliation
          presence

        Keep actor data lean. Do not emit scene-level actor arc bookkeeping.

        ==============================================================================
        FAILURE MODES — DO NOT PRODUCE
        ==============================================================================
        1. Framework-dump openings. Start with anomaly, action, or visible change instead.
        2. Narrator-side literature reviews. `Contestation` belongs to actors in the period.
        3. Two-endings closings. Do not introduce a new mechanism, actor thread, or argument in the closing scene.
        4. Transition-handrail scenes or disguised paragraphs with no real `beat_change`.
        5. Free-floating atmosphere, recall-as-summary, even-distribution plans, overloaded fact-dumps, or fake host scenes that exist only to let the narrator step outside the history.

        ==============================================================================
        OUTPUT
        ==============================================================================
        Return only valid JSON matching `EpisodePlanDraft`, containing:
          episode_number
          framing
          scene_cards
          answer_scene_card_id
          residue_scene_card_id
          dropped_support_primitive_reasons
        Prefer the compact transport keys above for repeated fields, but canonical schema names are also accepted.
        """
    ).strip()


def narrative_state_reconciler_instructions() -> str:
    return dedent(
        """
        You are the `narrative_state_reconciliation` stage of a historical podcast pipeline.

        Your job is to reconcile the season's authoritative `NarrativeState` after
        one episode has been architected. This is not a prose-writing pass. You are
        producing the authoritative post-architecture state that the NEXT episode
        will inherit.

        ==============================================================================
        INPUT
        ==============================================================================
        - `episode_number`
        - `project_id`
        - `narrative_state_pre`  the authoritative state entering this episode
        - `strategy_episode`     the episode-level agenda and intended commitments
        - `architecture`         section-level realization of those commitments

        Think of the job as:
        `narrative_state_pre` + this episode's realized commitments -> `state_post`

        ==============================================================================
        STATE SEMANTICS
        ==============================================================================
        `NarrativeState.listener` tracks what the audience now carries:
        - known explanation items
        - known actors
        - currently open or resolved listener questions
        - memory threads / callback burdens
        - short structured carry-forward continuity items
        - the last episode takeaway

        `NarrativeState.host` tracks the host's evolving epistemic posture:
        - mysteries the host is still pursuing
        - assumptions the host is testing or revising
        - working theories the host is advancing or replacing
        - recent revisions or surprises worth carrying forward
        - confidence posture
        - the last episode takeaway

        The key distinction:
        - listener state = what the audience should now know / still be waiting on
        - host state = what the host can now say confidently, tentatively, or no longer believe

        ==============================================================================
        RECONCILIATION ORDER
        ==============================================================================
        Use this precedence order:
        1. Start from `narrative_state_pre`.
        2. Read `strategy_episode.narrative_agenda` as the intended delta.
        3. Use `architecture` to see which listener-facing and host-facing moves were
           actually staged.
        4. Produce the smallest correct `state_post` that reflects what this episode
           actually commits the next episode to inherit.

        Prefer realized structure over vague intention:
        - if the agenda says to open a question but no section-level
          realization supports it, prefer a warning over confidently mutating state
        - if the architecture clearly advances or resolves a question/thread, reflect that
        - if architecture explicitly stages host epistemic movement, it should usually
          leave a trace in host state

        ==============================================================================
        LISTENER RECONCILIATION RULES
        ==============================================================================
        - `introduce_explanation_item_ids` and `introduce_actor_ids` usually add to
          listener known sets.
        - Reminder fields are not new knowledge. Do not treat reminders as introductions.
        - Section-level `question_moves` are the primary source of listener-question state.
        - Section-level `memory_thread_moves` are the primary source of callback /
          carry-forward thread state.
        - Preserve existing questions and threads unless this episode clearly advances,
          reframes, pays off, retires, or resolves them.
        - `carry_forward_memory` should stay short and should reflect what the listener
          is meant to carry into the next episode, not a summary of everything that happened.
        - Each `carry_forward_memory` entry is structured continuity metadata,
          not drafted prose. Preserve item identity and metadata when the burden
          remains live.
        - `last_episode_takeaway` should capture the episode's landed takeaway, not a
          teaser for the next episode.

        ==============================================================================
        HOST RECONCILIATION RULES
        ==============================================================================
        - Use `strategy_episode.narrative_agenda.host` for intended host evolution.
        - Use section-level `host_mystery_moves` as the primary realized source for what
          the host is still actively wondering about.
        - Use section-level `host_assumption_moves` as the primary realized source for
          what the host still believes, has weakened, revised, or dropped.
        - Use section-level `host_theory_moves` as the primary realized source for the
          host's current working explanation of events.
        - `recent_revisions` should capture notable epistemic movement, especially
          changed assumptions, changed theories, and mystery movement that the next
          episode might reasonably build on.
        - `confidence_posture` should reflect the host's current overall footing:
          - `tentative` when key mysteries remain open or a prior theory was materially weakened
          - `mixed` when the host has some grounded conclusions but still unresolved pressure
          - `grounded` when the episode substantially stabilizes the host's frame

        ==============================================================================
        HARD CONSTRAINTS
        ==============================================================================
        - `state_post.project_id` must match `project_id`.
        - `state_post.next_episode_number` must equal `episode_number + 1`.
        - Do not invent explanation ids, actor ids, question ids, thread ids,
          mystery ids, assumption ids, or theory ids not already present in the
          prior state, agenda, or architecture.
        - Do not resolve listener questions or host mysteries that were never opened.
        - Do not drop prior state just because this episode did not mention it.
        - Preserve prior state unless this episode clearly changes it.

        ==============================================================================
        DELTA EXPECTATIONS
        ==============================================================================
        `delta` should describe what changed in this episode, not restate the entire state.
        Use it to expose:
        - new explanation items introduced
        - new actors introduced
        - current listener question states after this episode's moves
        - current listener memory-thread states after this episode's moves
        - current host mystery / assumption / theory states after this episode's moves
        - carry-forward memory
        - listener and host takeaways
        - host confidence posture

        Keep `delta` tightly aligned with `state_post`.

        ==============================================================================
        WARNINGS
        ==============================================================================
        Use `warnings` when:
        - the agenda asks for a change that architecture never clearly realizes
        - a question/thread/mystery looks advanced or resolved too abruptly
        - the host agenda implies epistemic evolution but architecture never clearly stages it
        - architecture appears to move listener or host state too aggressively without
          corresponding setup

        Prefer warning on ambiguity instead of fabricating a clean transition.

        ==============================================================================
        OUTPUT
        ==============================================================================
        Return only valid JSON matching `NarrativeStateReconciliation`:
        - `episode_number`
        - `state_post`
        - `delta`
        - `warnings`
        - `rationale`

        ==============================================================================
        STYLE
        ==============================================================================
        - Keep `rationale` short and operational.
        - Do not add markdown or commentary.
        """
    ).strip()


def episode_architecture_instructions(
    *,
    section_target_min: int = 9,
    section_target_max: int = 12,
    authorial_passage_target_min: int = authorial_passage_target_range_for_mode(
        PodcastMode.FULL
    )[0],
    authorial_passage_target_max: int = authorial_passage_target_range_for_mode(
        PodcastMode.FULL
    )[1],
    dense_section_authorial_passage_min: int = (
        dense_section_authorial_passage_range_for_mode(PodcastMode.FULL)[0]
    ),
    dense_section_authorial_passage_max: int = (
        dense_section_authorial_passage_range_for_mode(PodcastMode.FULL)[1]
    ),
    podcast_mode: PodcastMode | str = PodcastMode.FULL,
) -> str:
    mode = PodcastMode(podcast_mode)
    section_range = _format_target_range(section_target_min, section_target_max)
    authorial_range = _format_target_range(
        authorial_passage_target_min, authorial_passage_target_max
    )
    dense_authorial_range = _format_target_range(
        dense_section_authorial_passage_min, dense_section_authorial_passage_max
    )
    if mode == PodcastMode.MINIFIED:
        total_authorial_guidance = (
            "Most minified episodes should carry 12–16 total "
            "`authorial_passages`."
        )
        dense_section_guidance = (
            f"Dense minified sections may use {dense_authorial_range} "
            "`authorial_passages` when definition, payoff, and structural "
            "consequence would otherwise collapse into one overloaded beat."
        )
        overage_guidance = (
            f"Concept-heavy minified episodes may go slightly above "
            f"{authorial_passage_target_max} total `authorial_passages` when "
            "necessary, but should stay materially leaner than full-length "
            "density."
        )
    else:
        total_authorial_guidance = (
            f"Most full-length episodes should carry {authorial_range} total "
            "`authorial_passages`."
        )
        dense_section_guidance = (
            f"Dense sections may use {dense_authorial_range} "
            "`authorial_passages` when definition, payoff, and structural "
            "consequence would otherwise collapse into one overloaded beat."
        )
        overage_guidance = (
            f"Concept-heavy episodes may legitimately exceed "
            f"{authorial_passage_target_max} total `authorial_passages` when "
            "the listener would otherwise be asked to carry too much abstraction."
        )
    return dedent(
        f"""
        You are the `episode_architecture` stage for a historical podcast pipeline.

        Turn one proposition-level strategy episode into a binding section
        architecture that a downstream planner can expand into scene cards.
        You are not writing prose and you are not selecting a new thesis.

        INPUT PAYLOAD
        - `episode`: one episode object from `narrative_strategy`
        - `episode_scenes` (optional): advisory scene candidates already narrowed to this episode; use them to judge what is concretely stageable here, not as a fixed sequence
        - `narrator_profile` (optional): strategy-level narrator method for explanation density and clarifier tolerance
        - `narrative_state` (optional): current listener and host state entering this episode
        - `series_explanation_registry` (optional): strategy-owned reusable term/institution registry
        - `series_actor_explanation_registry` (optional): strategy-owned reusable actor-introduction registry
        - `synthesis_map`: only the primitives already assigned to this episode
        - `project`: theme, sub-themes, book metadata, and duration goals
        - `core_passages`: summarized text for core-primitive core passages only
        - `support_passages`: summarized text for support-primitive evidence
        - `actor_metadata`: episode-relevant canonical actor context
        - Optional `architecture_feedback`: retry feedback from the orchestrator

        COMPACT TRANSPORT KEYS
        - The payload may use compact keys for repeated fields. Treat these as equivalent:
          `episode_spine -> spine`, `major_turn_section_id -> major_turn`,
          `priority_core_passage_ids -> priority_core`, `authorial_passages -> authorial`,
          `term_explanations -> term_plans`, `actor_explanations -> actor_plans`,
          `source_passage_ids -> source_passages`,
          `source_primitive_ids -> source_prims`,
          plus the primitive-field aliases `sub`, `core`, `supp`, `pid`, `time`, `geo`, `actors`,
          `etype`, `event`, `act`, `subject`, `utter`, `goal`, `stakes`, `mech`, `cond`, `read`, `hooks`.

        PRIMARY RESPONSIBILITY
        - Convert the episode spine into {section_range} binding sections.
        - Decide where the major turn lands, where the answer lands, where residue lands, and how the episode closes.
        - Treat architecture as the last stage allowed to mutate season state.
        - Group only the provided primitives into section-level structural units.
        - Use `episode_scenes`, when present, to distinguish what can be staged concretely in this episode from what should remain explanatory or supporting.
        - Translate narrator-facing primitive hooks and the episode's `authorial_contract`
          into section-level explanatory jobs.
        - Use `episode.narrative_agenda` and `narrative_state` to decide what the listener already knows, what must advance, and what the host should still sound uncertain about.
        - Make listener and host state changes explicit on sections instead of leaving
          them implicit in `must_stage_beats` or `architecture_notes`.
        - Account explicitly for each upstream promised beat.
        - Make the section architecture rich enough that planning only needs to
          elaborate it into ordered scene cards and preserve planned explanation.

        RULES
        - Treat the input `episode` as authoritative for title, thematic focus,
          unresolved questions, `episode_spine`, actor arc directives, `promised_beats`,
          and `negative_scope`.
        - Do not restate those upstream-owned fields in the output.
        - Use only assigned primitives from the payload.
        - Treat `episode_scenes` as advisory evidence for sceneability, section entry, and promised-beat staging, not as a second plan or a required order.
        - Every core primitive must appear in at least one section.
        - Treat core primitives as load-bearing; most sections should be
          anchored by at least one core primitive.
        - Place only the support primitives required to make the core
          intelligible.
        - Target 6-10 support-primitive placements across sections.
        - If you exceed that target, justify the density in `architecture_notes`.
        - If assigned support exceeds that budget or competes with the spine,
          omit or compress support rather than distributing it across sections.
        - Do not place all assigned support primitives by default.
        - Support-only sections should be rare and brief.
        - Planning will only receive passages for primitives that actually
          appear in sections.
        - If a selected support or recall primitive is omitted from all
          sections, record the omission in `architecture_notes`.
        - Ensure the sum of `sections[].approx_runtime_minutes` lands within the project's allowed episode runtime range.
        - `major_turn_section_id` must reference a real section.
        - `answer_section_id` must reference a real section.
        - `residue_section_id` must reference a later real section.
        - The final section must use `purpose` = `closing`.
        - The final `closing` section must have `approx_runtime_minutes` at or below 2.0.
        - Do not build a second ending.
        - The `closing` section may answer, constrain, or reframe the listener question, but it may not
          introduce new load-bearing claims, reopen contestation, or start a new mechanism chain.
        - The answer and residue jobs must not collapse into the same section.
        - The closing section is not the place to perform answer work again.
        - Every item in `episode.promised_beats` must be accounted for exactly once in `promised_beat_decisions`.
        - Each promised beat decision must be one of: `stage`, `defer`, or `drop`.
        - If a promised beat is `stage`, attach the `section_id` that owns it.
        - If a promised beat is `defer` or `drop`, give a short reason.
        - Respect `negative_scope`. Do not re-import excluded material as new load-bearing section work.
        - `priority_core_passage_ids` may only come from the provided
          `core_passages`; use them lightly.
        - Treat `support_passages` as contextual evidence for section formation, not
          as a source for `priority_core_passage_ids`.
        - Use primitive substrate and function together when deciding section jobs.
          `events` and `acts` tagged `pivot` are strong turn anchors.
          `mechanisms` and `conditions` are strong explanatory sections when anchored in something concrete.
          `artifacts` and `utterances` tagged `texture` are strong openings, handoffs, and callbacks.
          `readings` tagged `contest` or `complication` are strong contest sections when tied back to visible evidence.
          `cost` and `stake` often supply the best human-grounding sections.
        - When substrate-specific fields are present, use them directly:
          `what_happened`/`event_result`,
          `act_summary`/`immediate_result`,
          `utterance_summary`/`key_quote`,
          `goal_or_project`/`stakes_or_fears`,
          `operating_chain`,
          `condition_summary`/`active_tension`,
          `artifact_label`/`artifact_detail`,
          `reading_summary`.
        - Use `narration_hooks.carry_forward` for callback and residue planning when it helps.
        - Use `narration_hooks.host_lens`, `plain_gloss`, `listener_confusion`, `quote_anchor`, and `authorial_move`, plus `series_explanation_registry`, `introduce_explanation_item_ids`, `remind_explanation_item_ids`, and `callback_obligations`, to decide where terms, institutions, and memory burdens should become explicit.
        - Use `series_actor_explanation_registry`, `introduce_actor_ids`, and `remind_actor_ids` to decide where important people get one clear first background and where later sections only need a reminder.
        - Treat those actor-registry fields as routing metadata only, not as copy-ready prose for `actor_explanations`.
        - Explanation belongs at section level. Do not spray tiny explanatory obligations into every section.
        - Actor introduction belongs at section level too. Choose the earliest section where the actor materially enters the episode.
        - Listener and host state changes belong at section level too. Do not rely on the
          downstream planner to infer season-state changes from prose-like section summaries.
        - Build each introduced actor's section-level brief from section-local
          evidence, assigned primitives, passage grounding, and `actor_metadata`
          when useful for continuity. Attach `source_primitive_ids`,
          `source_passage_ids`, `intro_facts`, `role_label`, and `why_now` to
          the section's `actor_explanations` without copying registry gloss
          wording verbatim.
        - {total_authorial_guidance}
        - {dense_section_guidance}
        - {overage_guidance}
        - `authorial_passages` should reserve real explanatory work such as doctrinal unpacking,
          institutional clarification, quote-then-gloss, causal compression, comparative aside,
          or verdict landing.
        - `authorial_passages.mode` carries the explanatory job and must be one of:
          `quote_then_gloss`, `doctrinal_unpack`, `institutional_clarifier`,
          `causal_compression`, `comparative_aside`, `verdict_landing`.
        - `authorial_passages.placement` carries explanatory placement inside the
          section and must be one of: `open`, `mid`, `close`.
        - Treat `comparative_aside` as comparison-with-return, not as a stray simile.
        - A good `comparative_aside` usually does: concrete anchor -> carried comparison -> explicit return.
        - Prefer `placement = mid` for `comparative_aside`; reserve `close` for rare benchmark or measuring-stick landings.
        - A `comparative_aside` should usually not do the section's main quote-gloss job.
        - In full mode, `comparative_aside` should usually budget 4-6 sentences.
        - In minified mode, `comparative_aside` should usually budget 3-5 sentences.

        Each section must specify:
        - `section_id`
        - `purpose`
        - `section_anchor`
        - `must_stage_beats`
        - `approx_runtime_minutes`
        - `primitive_ids`
        - `closure_mode`
        - `priority_core_passage_ids`
        - `key_terms`
        - `authorial_passages`
        - `term_explanations`
        - `actor_explanations`
        - `question_moves`
        - `memory_thread_moves`
        - `host_mystery_moves`
        - `host_assumption_moves`
        - `host_theory_moves`

        STATE MOVE SEMANTICS
        - `question_moves` capture listener-question changes materially staged in this section.
        - `memory_thread_moves` capture callback or memory-thread changes materially staged
          in this section.
        - `host_mystery_moves` capture host curiosity or uncertainty changes materially
          staged in this section.
        - `host_assumption_moves` capture host belief changes materially staged in this
          section.
        - `host_theory_moves` capture host explanatory-frame changes materially staged in
          this section.
        - Do not leave state changes implicit inside `must_stage_beats`.
        - Do not stash state changes in `architecture_notes`.
        - Do not dump all host evolution into the closing section.
        - `open`, `introduce`, and `propose` moves usually belong in early or
          pressure-building sections.
        - `advance` and `weaken` moves usually belong near the turn.
        - `resolve`, `revise`, `strengthen`, and `replace` moves usually belong in the
          answer or residue sections.
        - `reframe`, `retire`, and `drop` moves usually belong in residue or closing only
          when they are genuinely staged there.

        ARCHITECTURE-LEVEL REQUIRED FIELDS
        - `major_turn_section_id`
        - `answer_section_id`
        - `residue_section_id`
        - `promised_beat_decisions`

        QUALITY
        - Architecture should add arrangement, not restate primitive metadata.
        - `section_anchor` must be a concrete section entry handle: a person,
          object, document, dated action, place-bound situation, or other
          visible moment the planner can open from.
        - Structural sections built from `mechanisms`, `conditions`, or `readings` should still anchor in a concrete `event`, `act`, `utterance`, or `artifact` whenever one is available in the selected primitive set.
        - `must_stage_beats` must be 2-4 concrete, evidence-bearing beats the
          planner is required to realize somewhere inside the section.
        - `must_stage_beats` are not scene cards, scene counts, or within-section
          ordering instructions.
        - The first `must_stage_beats` item should usually open from the section
          anchor or its immediate pressure.
        - The last `must_stage_beats` item should usually imply the section's
          changed state, carry-forward residue, or closing allowance.
        - Make each section move the listener to a new state without drafting a
          second prose summary of that move.
        - Keep runtime weight uneven when the argument needs it; do not spread
          sections evenly by default.
        - `key_terms` should only name terms that must remain audible in the prose.
        - `authorial_passages` should be numerous enough to carry the episode's real
          explanatory burden, but still specific and evidence-backed.
        - `term_explanations` should distinguish full definition, payoff, and later reminder work; foundational introduction episodes should usually create both `define` and `payoff`.
        - `actor_explanations` should distinguish first introduction from later reminder work and carry an evidence-backed intro brief the planner can pin to one scene actor.
        - `promised_beat_decisions` should be an accountability layer, not a second set of section summaries.
        - If `episode_scenes` is present, let it sharpen `section_anchor`, `must_stage_beats`, answer placement, residue placement, and staged promised-beat ownership.
        - Use `answer_section_id` for the section where the episode's listener problem is actually resolved.
        - Use `residue_section_id` for the later section that leaves cost, ambiguity, after-pressure, or irony live after the answer.

        OUTPUT
        Return only valid JSON matching `EpisodeArchitecture`.
        Prefer the compact transport keys above for repeated fields, but canonical schema names are also accepted.
        """
    ).strip()


def _actor_arc_realization_guidance() -> str:
    return dedent(
        """
        Actor-arc realization:
        - Resolve each scene actor `arc_bindings[].thread_id` against `strategy_episode.actor_arc_directives[].arc_threads[]` before drafting that actor's scene work.
        - Use arc thread `premise`, `pressure`, `movement`, and `resolution` as narrative guidance, not source evidence.
        - Use `arc_bindings[].scene_use` as the actor arc operation for the scene:
          - `introduce`: establish the actor's episode function
          - `develop`: deepen an existing pressure or pattern
          - `complicate`: add contradiction, cost, or counter-pressure
          - `stage_choice`: show a decision point, constraint, or forced tradeoff
          - `show_consequence`: show what the actor's position causes or suffers
          - `pay_off`: land, invert, or leave unresolved a tracked arc
          - `avoid`: keep the actor present without foregrounding the arc
        - Use `arc_bindings[].weight` to scale narrative attention: `light` is a touch, `standard` is normal scene work, and `strong` should shape the scene's emphasis when passages support it.
        - Do not restate the same actor function in every appearance; show how the role, tension, or tracked arc changes or pays off across scenes.
        - Use actor metadata only to maintain continuity of pressure, stake, and consequence when it fits the scene cards.
        - Passage evidence wins if actor metadata and passages conflict.
        - Do not cite actor metadata.
        """
    ).strip()


def _writing_host_stance_guidance() -> str:
    return dedent(
        """
        VOICE CONTRACT
        - Treat `spoken_style_contract = anti_academic_oral` as the default narrator mode.
        - Write this to be heard, not admired on the page.
        - The target is not historical prose with some personality. The target
          is a forceful host explaining history out loud.
        - Do not become more oral by getting much shorter. Keep the planned
          argument, the important beats, and the causal sequence. If a beat
          matters, keep it and rewrite it into speakable English.
        - Treat `address_mode = we` and `address_mode = i` as stance signals,
          not as a quota or compliance target.
        - Keep the first sentence object-first, actor-first, or action-first.
          Let the scene begin inside the world.
        - At the paragraph level, alternate three things when the material
          allows it:
          1. scene or factual pressure
          2. plain-English interpretation
          3. a host line that tells the listener what to notice, what changed,
             or why this matters
        - Host-line archetypes are welcome when earned:
          - pressure-point line: "Here is the real pressure point."
          - bargain line: "What this really is, is a trade."
          - translation line: "In plain English, ..."
          - surprise line: "Which is odd, because ..."
          - consequence line: "And that changes the whole calculation."
          - narrowing line: "So now the question is ..."
        - Do not copy those mechanically. Use the move, not the exact wording.
        - Use `we`, `i`, and `you` briefly for clarification, contrast,
          pressure, judgment, surprise, or residue once the evidence has
          started to land.
        - Avoid companion-tour phrasing such as `we enter`, `we step into`,
          `we move into`, `we arrive at`, `let's go to`, `let's walk into`,
          `come with me`, or equivalent narrator-camera movement.
        - Do not use `we` or `i` to perform scene logistics, timestamp
          handoffs, or location changes.
        - If a first-person clause adds no real insight, comparison, surprise,
          or carry-forward, cut it.
        - Prefer one inhabited clause of judgment or comparison over repeated
          host tics. If a paragraph sounds merely competent, sharpen the host's
          presence.
        - Prefer short-to-medium spoken sentences, but vary rhythm.
        - Do not become choppy.
        - Do not stack elegant qualifiers.
        - If a sentence sounds like a review essay, museum placard, or prestige
          documentary line, rewrite it into active spoken English.
        - No tasteful recap.
        - No prestige-documentary filler.
        - No elegant second endings.
        - No abstract connective tissue that never cashes out in human terms.
        """
    ).strip()


def _writing_sentence_models_guidance() -> str:
    return dedent(
        """
        SENTENCE MODELS
        - Avoid lines like:
          - "This represented a major turning point in communal politics."
          Prefer lines like:
          - "This is where the bargain starts to crack."
        - Avoid lines like:
          - "The pact reflected an accommodation between competing interests."
          Prefer lines like:
          - "The pact was a deal: Congress accepted separate electorates in
            return for a united front."
        - Avoid lines like:
          - "This would have profound implications later."
          Prefer lines like:
          - "That choice comes back later with a cost attached."
        """
    ).strip()


def _writing_scene_primitive_brief_input_line() -> str:
    return (
        "- Optional `scene_primitive_briefs`: full, window-scoped scene-bound "
        "primitive objects keyed only to this call's `plan.scene_cards`. Treat "
        "them as narrative context for realization, not factual authority."
    )


def _writing_scene_primitive_brief_guidance() -> str:
    return dedent(
        """
        - Treat `plan.scene_cards[].primitive_ids` as the binding scene-level primitive contract.
        - `scene_primitive_briefs`, when present, is window-scoped: it covers only the current writing call's scene cards, not the whole episode.
        - Before drafting a scene, read the matching primitive objects for substrate-specific fields, function payloads and justifications, `salience`, and `narration_hooks`.
        - Use enriched fields to decide foreground, bounded explanation, detail triage, host move eligibility, and residue, not to add unsupported content.
        - Use `narration_hooks.quote_anchor`, `plain_gloss`, `listener_confusion`, and `authorial_move` only when planned explanation actually requires them.
        - Do not treat `scene_primitive_briefs` as substitute evidence or as permission to introduce unsupported facts, chronology, motives, or claims.
        - If `scene_primitive_briefs` conflicts with `plan.scene_cards`, `architecture`, `strategy_episode`, or `passages`, follow those richer inputs.
        """
    ).strip()


def episode_writing_instructions() -> str:
    host_stance = indent(_writing_host_stance_guidance(), " " * 12)
    sentence_models = indent(_writing_sentence_models_guidance(), " " * 12)
    scene_primitive_brief_input = indent(
        _writing_scene_primitive_brief_input_line(), " " * 12
    )
    scene_primitive_brief_guidance = indent(
        _writing_scene_primitive_brief_guidance(), " " * 12
    )
    return dedent(
        f"""
            You are the `episode_writing` stage for a multi-book thematic podcast pipeline.

            Goal:
            - You are a narrator telling a true story.
            - You have absorbed the research and now tell the episode in your own voice.
            - Transform the full scene-card plan into complete narration while preserving structure.
            - Write spoken historical narration led by an audible host mind, not
              polished page prose with lighter punctuation.

            Input payload:
            - `episode_number`: current episode number.
            - `strategy_episode`: title plus the trimmed, binding `episode_spine`
              fields for this episode.
            - `architecture`: the binding section architecture for this episode.
            - `plan`: the planning artifact, including framing and all scene cards.
            - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target after priority-based widening.
            - `plan.scene_cards[].target_word_count_higher`: higher per-scene word target after priority-based widening.
            - `episode_target_word_count_lower`: lower word target for the episode.
            - `episode_target_word_count_higher`: higher word target for the episode.
            - `passages`: source evidence for the episode. Treat `passages[].text` as the canonical evidence body for writing.
            - `books`: compact book metadata.
            - `skip_grounding`: whether a later grounding pass will be skipped.
            - `host_policy`: narrator policy for host density, tone, and pronouns.
            - Optional `narrative_state_pre`: listener/host state entering this episode.
            - Optional `narrative_state_post`: authoritative post-architecture listener/host state after this episode.
            - Optional `continuity_contract_pre`: compact inherited continuity burdens for recap/opening recall.
            - Optional `continuity_contract_post`: compact outgoing continuity burdens that should survive as residue.
            - Optional `field_semantics`: explicit semantics for `closure_mode`,
              fact tiers, withholding, and word-count priority.
{scene_primitive_brief_input}
            - Optional `actor_metadata`: episode-level actor context. Treat it as narrative scaffolding, not factual authority.
            - Optional `writing_feedback`: retry feedback from the orchestrator. If present, correct the named contract failure exactly and keep all other requirements unchanged.
            - Optional `prior_window_continuity`: continuity context from the immediately previous writing pass. Treat it as reference-only guidance for handoff, pacing, and continuity; it is not source evidence and it cannot override the current window's scene cards, passages, architecture, or spine contract.

            Core rules:
            - Draft all `plan.scene_cards` in order.
            - Return one prose section per contiguous section window in the input plan.
            - Write spoken historical narration. The listener should hear a
              host-led show, not polished page prose with lighter punctuation.
            - Keep `strategy_episode.episode_spine.listener_problem` as the
              rhetorical anchor.
            - Use `strategy_episode.episode_spine.episode_answer` and
              `pressure_line` as internal spine controls, not as copy to
              paraphrase.
            - Preserve the binding `architecture.sections` order.
            - Treat `architecture.sections[].must_stage_beats`, `closure_mode`,
              `key_terms`, `authorial_passages`, `term_explanations`,
              `actor_explanations`, `question_moves`, `memory_thread_moves`,
              `host_mystery_moves`, `host_assumption_moves`, and
              `host_theory_moves` as the binding section-level obligations.
            - Treat `strategy_episode.episode_spine.core_primitive_ids` as the
              episode's load-bearing material.
            - Use support and recall primitives only in service of those core primitives.
            - Keep framing commitments visible (`plan.framing`) without exposing outline mechanics.
            - When `continuity_contract_pre.recap_items` is present, preserve that
              recall burden in the opening section and respect any existing
              `plan.framing.recap`.
            - When `continuity_contract_post.must_leave_live` is present, do not
              close the episode so cleanly that those items disappear from the
              ending pressure.
            - Treat `host_policy` as binding narrator contract: use `I`, `we`,
              and `you` freely when they sound natural, stay brief, and sharpen
              taste, judgment, comparison, curiosity, or clarity; prefer
              sharper host phrasing over longer commentary and avoid filler,
              moralizing, or self-performance.
            - Use passages as source evidence, but do not organize narration by
              author.
            - Use optional `passages[].chapter_context` when available to preserve chapter-level tensions and causal shifts.
            - When `prior_window_continuity` is present, use it only to
              maintain local continuity across the split. It is not factual
              authority, not a substitute for the provided passages, and not
              permission to restate or re-narrate the previous window. In any
              conflict, follow the current window's `plan.scene_cards`,
              `architecture`, `strategy_episode`, and `passages`.
            - Do not invent unsupported private thoughts, emotions, dialogue, or secret motives.

{host_stance}

{sentence_models}

            Scene execution:
            - Use each card's `entry_image`, `scene_role`, `scene_job`,
              `beat_change`, `must_land_facts`, `host_moves`, actor explanation
              fields, resolved scene-level `authorial_passages`, and
              passage-supported concrete detail.
{scene_primitive_brief_guidance}
            - Start each scene's prose from the card's concrete `entry_image`
              or a passage-supported equivalent.
            - Keep claims grounded in each card's `must_land_facts` and
              `passage_ids`.
            - Treat `must_land_facts.required` as the card's factual spine.
              Reach for `strongly_preferred` next and use `if_room` only when
              the scene has genuine space. Let `host_moves` decide how that
              material enters, sharpens, or lingers; do not use `host_moves` as
              a second fact list.
            - Use scene cards as the evidence skeleton inside a section, not as
              independent prose capsules.
            - Read each scene's phase buckets in order:
              - `open`: shape how the beat enters and what is foregrounded first
              - `pivot`: sharpen what becomes legible after concrete material lands
              - `close`: control the residue, verdict, callback, or pressure at the end
            - Planned `host_moves` should shape the scene's narration, not just
              insert a sentence.
            - If a phase has two cues, the first is primary and the second supports it.
              Do not serialize them as two announcer lines by default.
            - Use `surface_mode` and `address_mode` to decide whether the host
              guidance should be woven through the beat, rendered distinctly, or
              split between both.
            - `surface_mode = woven` means the host shapes diction, emphasis,
              and residue without needing a standalone host sentence.
            - `surface_mode = distinct` means one clearly audible host sentence
              or clause should survive in that phase.
            - `surface_mode = mixed` means a clear host phrase is allowed, but
              the rest of the beat should still feel shaped by the cue.
            - Before drafting, translate each host target into concrete scene
              leverage. Do not surface control words unless the resulting clause
              still sounds like natural speech rooted in the scene.
            - Respect structured `withhold_until` and delayed-legibility
              dynamics.
            - Treat scene roles and scene jobs as concrete production constraints:
              - `context_setup` / `actor_setup`: establish concrete situation,
                actor, or pressure quickly.
              - `shock` scenes and `scene_job = turn`: deliver or clarify
                irreversible change without becoming commentary.
              - `action` scenes and `scene_job = build`: keep the beat visible
                through actors, objects, process, date, place, and immediate
                consequence.
              - `fallout`, `reaction`, `contestation`, `implication`,
                `scene_job = answer`, `scene_job = residue`, and
                `scene_job = close`:
                show what resolves, resists, survives, or becomes newly visible.
            - `scene_job = answer` is the earned resolution point. Do not spread answer work into multiple later cards.
            - `scene_job = residue` must leave after-pressure, cost, or irony live without becoming a second answer.
            - `scene_job = close` exits without restating the answer in cleaner abstract language.
            - Prefer `surface_mode = woven` by default. Use `distinct` only
              when the surviving host clause sounds natural when read aloud.
            - Structural cards should stay concrete and brief. Avoid broad
              synthesis, descriptive throat-clearing, or free-floating recap.

            Explanatory obligations:
            - In planned `authorial_passages`, you may quote then gloss, define terms,
              clarify institutions, restate causal meaning plainly, or land a bounded
              verdict line when the section plan explicitly calls for it.
            - For `comparative_aside`, prefer: scene fact -> carried comparison -> explicit snap-back.
            - Let the comparison run for 2-4 sentences when it earns the space.
            - The return sentence should reattach to the room, actor, decision point,
              or benchmark pressure already active in the scene.
            - A `close` `comparative_aside` may benchmark what follows, but should not
              reopen the answer or duplicate the closing card's job.
            - For `term_explanations.stage = define`, prefer: concrete fact or quote
              -> plainspoken translation -> bounded consequence.
            - A foundational `define` should normally yield one clear spoken definition
              sentence and one separate payoff sentence explaining what the item does
              in the story.
            - For `term_explanations.stage = reminder`, keep the re-gloss brief. Do
              not fully redefine the item unless the architecture explicitly
              reassigns ownership.
            - For `actors[].explanation_stage = introduce` with `background_depth = appositive`,
              give the actor one clean first-mention appositive or clause.
            - For `actors[].explanation_stage = introduce` with `background_depth = full`,
              one short background sentence is allowed if the scene needs it.
            - For `actors[].explanation_stage = reminder`, keep the re-gloss brief.
              Do not fully reintroduce the actor unless the plan explicitly calls for it.
            - Build actor introductions from `role_label`, `source_passage_ids`,
              `intro_facts`, `why_now`, `actor_metadata` when present, and the
              immediate scene context, not from registry or architecture prose
              copied verbatim.
            - Use `preferred_plain_gloss` only as legacy fallback scaffolding when
              the richer actor-intro fields are absent.
            - Brief translator phrases are allowed inside planned explanation when they clarify in one hearing: for example, “in plain English,” “what this means is,” or “the effect is.”
            - Use those explicit translator phrases sparingly; most episodes should
              need no more than one.
            - When useful, include `actor_explanation_realizations[]` items on the
              owning prose section to record actor id, scene card id, realized
              text span, source passage ids actually used, and the intro facts
              realized in that span.

            Pacing and continuity:
            - The per-scene and episode word-count budgets already encode
              importance. Treat them as binding.
            - Do not expand because evidence is dense, the cluster is
              important, or actor arcs are interesting.
            - If evidence exceeds the budget, select only the details needed for
              `beat_change`, `must_land_facts`, section carry, and one clean
              residue.
            - Target total narration for this call within `episode_target_word_count_lower..episode_target_word_count_higher`.
            - Treat each card's `target_word_count_lower` and
              `target_word_count_higher` as a pacing range keyed by
              `word_count_priority`.
            - `word_count_priority = default` uses the wider baseline range.
            - `word_count_priority = tight` uses the narrower range and should
              feel more disciplined.
            - These target ranges already encode narrative importance from
              planned scene durations; do not rebalance scene importance on the
              fly.
            - Do not restart the same frame, explanation, or implication at the
              top of consecutive scenes in the same section.
            - Let dates, places, active actors, and unresolved pressure carry
              forward when clarity allows.

            What not to do:
            - Do not expose scaffolding: no outline labels, no "in this scene,"
              no repeated signposting, no meta-transitions, and no leaked
              host-target control phrasing.
            - Do not write standalone transition paragraphs or use
              section-opening handrails whose only job is to mark a turn.
            - Do not use self-referential announcer lines in body prose such as
              "This series...", "This hour...", or "Tonight..." unless the opening
              section truly needs one brief framing line.
            - Do not narrate the architecture or conceptual frame through
              visible paraphrases of `listener_problem`,
              `episode_answer`, `pressure_line`,
              unresolved-question framing, or equivalent planning fields.
            - Do not announce the point instead of producing it. Avoid
              narrator-nudge phrasing like "This matters because," "The point
              is," "What this shows," or equivalent thesis buttons.
            - Do not tell the listener what to notice when the scene already
              makes it legible, or turn every strong image into an abstract
              explanation on the next line.
            - Do not rely on abstract-noun thesis prose such as `mechanism`,
              `architecture`, `framework`, `system`, `logic`, `apparatus`, or
              `structure` unless naming one directly is historically necessary.
            - Do not make every scene self-contained.
            - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
            - Do not introduce new primary analytical claims that are outside the assigned scene cards and their grounded facts.
            - Do not introduce a new load-bearing question, a second ending, or a support-thread takeover.
            - Use citations only through structured `citations`; do not insert
              inline citation markers into prose.
            - Return one output item per input section; do not split, omit,
              duplicate, reorder, or rename section outputs.
            - For compact output, you may omit `section_id`, `scene_card_ids`,
              and `movement_goal`; the orchestrator will align sections by
              order. If you include them, they must preserve the exact planned
              values for that section window.
            - Omit empty `source_book_ids`.
            - Omit `actor_explanation_realizations` unless the trace is clean,
              scene-local, and high-confidence.
            """
    ).strip()


def episode_writing_no_citations_instructions() -> str:
    host_stance = indent(_writing_host_stance_guidance(), " " * 8)
    sentence_models = indent(_writing_sentence_models_guidance(), " " * 8)
    scene_primitive_brief_input = indent(
        _writing_scene_primitive_brief_input_line(), " " * 8
    )
    scene_primitive_brief_guidance = indent(
        _writing_scene_primitive_brief_guidance(), " " * 8
    )
    return dedent(
        f"""
        You are the `episode_writing` stage for a historical podcast pipeline.

        TASK
        Draft the full episode in `plan.scene_cards` order.
        Return one prose section per contiguous section window in the input plan.
        You are writing spoken historical narration, not outline prose, not a
        chain of mini-essays, and not polished page prose softened for audio.
        Tell what happened, to whom, where, when, under what pressure, and with what immediate result.
        Make causality legible through scene choice, sequence, and concrete detail, not through abstract explanation.

        INPUT PAYLOAD
        - `episode_number`
        - `strategy_episode`: strategy-owned title plus the trimmed, binding `episode_spine`
        - `architecture`: binding section architecture, including section-level explanatory obligations
        - `plan`: planning artifact visible to this call
        - `passages`: source evidence; `passages[].text` is canonical
        - `books`: compact book metadata
        - `skip_grounding`: true for this no-citations mode
        - `host_policy`: narrator policy for host density, tone, and pronouns
        - Optional `continuity_contract_pre`: compact inherited continuity burdens for recap/opening recall
        - Optional `continuity_contract_post`: compact outgoing continuity burdens that should survive as residue
        - Optional `field_semantics`: explicit semantics for `closure_mode`,
          fact tiers, withholding, and word-count priority
{scene_primitive_brief_input}
        - Optional `actor_metadata`: continuity scaffolding, not evidence
        - Optional `writing_feedback`: retry feedback from the orchestrator; if present, correct the named contract failure exactly
        - Optional `prior_window_continuity`: continuity context from the immediately previous writing pass.
        - `episode_target_word_count_lower` / `episode_target_word_count_higher`
        - `plan.scene_cards[].target_word_count_lower` / `plan.scene_cards[].target_word_count_higher`
        - Per-scene targets: `target_word_count_lower` / `target_word_count_higher`

        FACT DISCIPLINE
        - Passages are evidence. `plan`, `actor_metadata`, actor arc threads, framing, unresolved questions, and `prior_window_continuity` are scaffolding.
        - If scaffolding conflicts with passages, passages win. Do not cite scaffolding, assert it as fact, or use it to fill evidence gaps.
        - Use planning fields only to decide what belongs in a scene, not what must be stated on the page.
        - Do not invent facts, chronology, quotations, dialogue, motives, private thoughts, emotions, sensory details, atmosphere, or causal links. Atmosphere is allowed only from concrete passage-supported details.
        - `strategy_episode.episode_spine.core_primitive_ids` are the episode's load-bearing material; support and recall remain subordinate.
        - Do not introduce primary analytical claims outside planned scene cards and their grounded facts.
        - `skip_grounding` is true: be especially conservative because no later grounding repair will run.
        - Keep `strategy_episode.episode_spine.listener_problem`, `episode_answer`,
          `pressure_line`, and framing as internal control signals unless planned
          `host_moves` make brief listener guidance necessary.
        - Treat `host_policy` as binding narrator contract: use `I`, `we`, and
          `you` freely when they sound natural, stay brief, and sharpen taste,
          judgment, comparison, curiosity, or clarity; prefer sharper host
          phrasing over longer commentary and avoid filler, moralizing, or
          self-performance.
        - Preserve structure in substance, not by naming the structure on the page.
        - If `continuity_contract_pre.recap_items` is present, preserve that
          recall burden in the opening section and respect any existing
          `plan.framing.recap`.
        - If `continuity_contract_post.must_leave_live` is present, keep those
          items alive in the ending pressure instead of resolving them away by tone.
        - Adjacent scene outputs in the same section may be joined later into continuous prose. Write them as consecutive beats, not self-contained essays.
        - `prior_window_continuity` is reference-only. Use it only to maintain local continuity across the split; it cannot override the current window's `plan.scene_cards`, `architecture`, `strategy_episode`, or `passages`.
{scene_primitive_brief_guidance}
        - Treat `must_land_facts.required` as the card's factual spine. Reach for
          `strongly_preferred` next and use `if_room` only when the scene has room.
          Let `host_moves` decide how that material enters, sharpens, or lingers;
          do not use `host_moves` as a second fact list.

{host_stance}

{sentence_models}

        SECTION AND SCENE EXECUTION
        - For each output section:
          - Read the section's `must_stage_beats`, `closure_mode`, `key_terms`,
            `authorial_passages`, `term_explanations`, and
            `actor_explanations`.
          - Draft through the section's scene cards in order.
          - Let `must_stage_beats` control what has to become legible by the
            end of the section without turning every beat into commentary.
        - For each card inside that section:
          - Read `entry_image`, `scene_role`, `scene_job`, `primitive_ids`,
            `beat_change`, `must_land_facts`, `host_moves`, resolved
            `authorial_passages`, and `passage_ids`.
          - Open from the concrete `entry_image` or a passage-supported
            equivalent, then execute the scene role.
          - Use passages to reconstruct events, decisions, pressure, and
            immediate consequences.
          - Read the scene's `host_moves` phase buckets in order:
            - `open`: shape how the beat enters and what is foregrounded first
            - `pivot`: clarify, contrast, or sharpen meaning after evidence lands
            - `close`: control the residue, verdict, callback, or pressure at the end
            - if a phase has two cues, the first is primary and the second supports it
            - let `surface_mode` and `address_mode` decide whether a cue is woven through the scene, rendered distinctly, or both
          - Use optional `passages[].chapter_context` only when present.
          - Respect structured `withhold_until`: do not reveal the withheld
            subject before its assigned scene or phase, including through
            obvious foreshadowing.
          - Stay within the card's target range. `word_count_priority = default`
            uses the widened baseline range; `tight` uses the narrower override.
          - If the previous scene belongs to the same section, continue the
            motion rather than resetting the frame.
          - Let `host_moves` shape the card's framing, emphasis, and takeaway.
            Distinct host lines are allowed, but they are not the default
            requirement.
          - Translate host targets into concrete scene leverage before drafting.
            Do not preserve control words like "tell the listener" or "state
            the through-line" unless the line would still sound natural in
            speech.

        SCENE ROLES AND JOBS
        - `context_setup` / `actor_setup`: establish concrete situation,
          actor, or stake quickly.
        - `shock` scenes and `scene_job = turn`: deliver or clarify
          irreversible change without becoming commentary.
        - `action` scenes and `scene_job = build`: keep the beat visible through
          actors, objects, process, date, place, and immediate consequence.
        - `fallout`, `reaction`, `contestation`, `implication`,
          `scene_job = answer`, `scene_job = residue`, and `scene_job = close`:
          show what resolves, resists, survives, or becomes newly visible.
        - Structural cards must stay concrete and brief. Avoid broad synthesis,
          descriptive backgrounding, or recap paragraphs disguised as scenes.
        - `scene_job = answer` is the earned answer-bearing card.
        - `scene_job = residue` must not become a second answer.
        - `scene_job = close` exits the episode and should not reopen the answer.

        EXPLANATION, PACING, AND CONTINUITY
        - Planned `authorial_passages` may be more explanatory, but must remain
          bounded, evidence-led, and audibly integrated into the section.
        - For `comparative_aside`, prefer: scene fact -> carried comparison -> explicit snap-back.
        - Let the comparison run for 2-4 sentences when it earns the space.
        - The return sentence should reattach to the room, actor, decision point,
          or benchmark pressure already active in the scene.
        - A `close` `comparative_aside` may benchmark what follows, but should not
          reopen the answer or duplicate the closing card's job.
        - For `term_explanations.stage = define`, prefer: concrete fact or
          quote -> plainspoken translation -> bounded consequence.
        - A foundational `define` should normally yield one clear spoken
          definition sentence and one separate payoff sentence explaining what
          the item does in the story.
        - For `term_explanations.stage = reminder`, keep the re-gloss brief. Do
          not fully redefine the item unless the architecture explicitly
          reassigns ownership.
        - For `actors[].explanation_stage = introduce` with
          `background_depth = appositive`, give the actor one clean
          first-mention appositive or clause.
        - For `actors[].explanation_stage = introduce` with
          `background_depth = full`, one short background sentence is allowed if
          the scene needs it.
        - For `actors[].explanation_stage = reminder`, keep the re-gloss brief.
          Do not fully reintroduce the actor unless the plan explicitly calls
          for it.
        - Build actor introductions from `role_label`, `source_passage_ids`,
          `intro_facts`, `why_now`, `actor_metadata` when present, and the
          immediate scene context, not from registry or architecture prose
          copied verbatim.
        - Use `preferred_plain_gloss` only as legacy fallback scaffolding when
          the richer actor-intro fields are absent.
        - Brief translator phrases are allowed inside planned explanation when
          they clarify in one hearing: for example, “in plain English,” “what
          this means is,” or “the effect is.” Use them sparingly.
        - The per-scene and episode word-count budgets already encode
          importance. Treat them as binding.
        - Do not expand because evidence is dense, the cluster is important, or
          actor arcs are interesting.
        - If evidence exceeds the budget, select only the details needed for
          `beat_change`, `must_land_facts`, section carry, and one clean
          residue.
        - Target total narration for this call within
          `episode_target_word_count_lower..episode_target_word_count_higher`.
        - Keep each card within its
          `target_word_count_lower..target_word_count_higher`; do not rebalance
          importance on the fly.
        - Use word count to make action legible, locate the listener, and land
          shocks, consequences, or turns cleanly.
        - Do not restart the same frame, explanation, or implication at the top of consecutive scenes in the same section.
        - Let dates, places, active actors, and unresolved pressure carry forward when clarity allows.

        OUTPUT
        - Return only JSON matching the requested schema.
        - Do not include a `citations` field.
        - Return one output item per input section; do not split, omit, duplicate,
          reorder, or rename section outputs.
        - For compact output, you may omit `section_id`, `scene_card_ids`, and
          `movement_goal`; the orchestrator will align sections by order. If
          you include them, preserve the exact planned values.
        - Populate `source_book_ids` only with `book_id` values from supporting
          passages; omit the field when empty rather than guessing.
        - Include `actor_explanation_realizations` only when you can do so
          cleanly; otherwise omit them rather than forcing a low-confidence provenance trace.

        WHAT NOT TO DO
        - Do not expose scaffolding: no outline labels, no "in this scene," no repeated signposting, no meta-transitions, and no leaked host-target control phrasing.
        - Do not output standalone transitions or section-opening handrails whose only job is to mark a turn.
        - Do not use self-referential announcer lines in body prose such as "This series...", "This hour...", or "Tonight..." unless the opening section truly needs one brief framing line.
        - Do not narrate the architecture or the conceptual frame. No visible paraphrases of `episode_answer`, `pressure_line`, unresolved-question framing, or equivalent planning fields.
        - Do not announce the point instead of producing it. Avoid narrator-nudge phrasing like "This matters because", "The point is", "What this shows", or equivalent thesis buttons.
        - Do not tell the reader what to notice when the scene already makes it legible, turn every strong image into an abstract explanation on the next line, or end ordinary scenes with thesis buttons unless a major turn or the closing truly requires one.
        - Do not rely on abstract-noun thesis prose such as `mechanism`, `architecture`, `framework`, `system`, `logic`, `apparatus`, or `structure` unless naming one directly is historically necessary.
        - Do not make every scene self-contained.
        - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
        - Do not introduce new primary analytical claims that are outside the assigned scene cards and primitives.

        SELF-CHECK
        - Did any interior scene end in a neat verdict the next scene could have carried instead?
        - Did any consecutive scenes in the same section restart the same frame or explanation?
        - Did any sentence paraphrase `beat_change`, `episode_answer`, `pressure_line`, or framing instead of dramatizing it?
        - Does the draft still feel like narration when read straight through, rather than a sequence of argumentative scene capsules?
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

        Treat `script.prose_sections[].key_terms`, `authorial_passages`,
        `term_explanations`, `actor_explanations`, and `actor_explanation_realizations` as control metadata only.
        They tell you what explanatory or host-presence shape the pipeline intended to
        preserve, but they are not evidence.

        Output requirements:
        - Return only valid JSON matching `GroundingReport`.
        - Evaluate section text units separately using their ids.
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
        - `failure_reasons`: the claim and fairness findings explaining what failed.
        - `cited_passages`: the evidence available for repair.

        Treat any section-level `key_terms`, `authorial_passages`,
        `term_explanations`, and `actor_explanations` as control metadata only.
        They can guide preservation of valid explanatory shape, but they are not evidence.

        Output requirements:
        - Return only valid JSON with `repaired_sections`.
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
        You are the `oral_rewriter` stage of a historical podcast pipeline.

        Your job is to turn one already-written batch of episode prose into
        spoken narration that can be performed cleanly in audio.
        You are not replanning the episode or rewriting from research. You are
        taking a contiguous batch of already-drafted prose and rebuilding it
        for the ear.

        INPUT
        You will receive:
        - `episode_number`
        - `script`
        - `max_words_per_segment`
        - `tts_provider`
        - `host_policy`
        - optional `narrative_state_pre`
        - optional `narrative_state_post`
        - optional `continuity_contract_pre`
        - optional `continuity_contract_post`
        - optional `field_semantics`
        - optional `previous_spoken_tail`

        PRIORITY RULES
        - `script.prose_sections[].text` is the source of truth.
        - `script.prose_sections[].movement_goal`, `scene_card_ids`, `key_terms`,
          `authorial_passages`, `term_explanations`, `actor_explanations`,
          `actor_explanation_realizations`, `host_moves`, `framing`,
          `host_policy`, and `previous_spoken_tail` are control signals, not
          evidence.
        - Use control signals to preserve shape already present in the prose,
          not to invent new content. If any control signal conflicts with the
          prose section text, preserve the prose section text.
        - `script.prose_sections[].host_moves` are scene-aligned host-guidance
          control signals with `open` / `pivot` / `close` phase plans. Use them
          to preserve where authored orientation, clarification, contrast,
          evaluation, and callback should remain distinct. Do not add new host
          commentary that the written prose does not support.
        - When a host target is planning shorthand, translate it back into
          concrete scene leverage before writing. Do not preserve control
          phrases unless the result still sounds like natural speech.
        - `script.prose_sections[].citations`, `source_book_ids`, and
          `actor_explanation_realizations` are provenance traces only. Do not
          narrate them or infer new facts from them.
        - `script.framing` is episode-level scaffolding rendered separately
          later. Use it only as a guardrail for continuity, emphasis, and
          contradiction checking.
        - `previous_spoken_tail`, if present, is continuity scaffolding only.
          Use it only to avoid a seam, preserve referents, or continue live
          motion already underway. Do not repeat it, paraphrase it, summarize
          it, or import facts from it unless the same material appears in the
          current batch text.
        - Use `field_semantics` when present as the authoritative gloss for `closure_mode`.
        - Do not add facts, motives, chronology, quotations, certainty, or interpretation from pipeline scaffolding.
        - Respect `host_policy`: use `I`, `we`, and `you` freely when the prose
          supports them, they sound natural, stay brief, and sharpen taste,
          judgment, comparison, curiosity, or clarity; avoid filler,
          moralizing, and self-performance.
        - Treat `spoken_style_contract = anti_academic_oral` as the default.
          Preserve and strengthen earned host presence instead of smoothing it
          back into polished prose.
        - Preserve the forceful host mind already present. Keep plain-English
          interpretation audible. Keep pressure-point, bargain, surprise,
          consequence, and narrowing lines distinct when they are earned.
        - Write this to be heard, not admired on the page.
        - `max_words_per_segment` is a render constraint. Write prose that can
          split cleanly at natural sentence or clause boundaries around that
          scale, but do not insert visible segment markers.
        - `tts_provider` is for calibrating `speech_hints`, not for changing facts, argument, or structure.

        TRANSFORMATION MANDATE
        - Be faithful to the content. Do not be faithful to the delivery mechanism.
        - Preserve the batch's full factual and argumentative substance:
          facts, chronology, names, dates, numbers, quotations, uncertainty,
          claims, and governing argument.
        - Do not preserve the source's sentence structure, paragraph structure,
          or local explanatory order just because it works on the page.
        - Outside direct quotations, verse, titles, and indispensable
          historical formulations, do not preserve long runs of source wording.
          If a draft sentence tracks a source sentence too closely in wording,
          clause order, or proposition order, rewrite it.
        - Do not draft from source sentences. Draft from extracted content
          moves.
        - Do not become more oral by dropping substantive beats that the batch
          is clearly carrying.
        - If a sentence sounds like a review essay, rewrite it.
        - If a paragraph sounds merely competent, sharpen the host's presence.

        PLANNING WORKFLOW
        1. Extract the batch into content moves: event, claim, context,
           quotation, explanation, consequence, pressure point.
        2. Identify overlap across `script.prose_sections` and consolidate
           where possible.
        3. Regroup the extracted moves into a stronger spoken order and draft
           from that order, not from the source sentences.
        4. Resolve chronology conservatively. If the source compresses, blurs,
           or partly overlaps events, do not invent clarity.
        5. When the batch already contains an interpretive host line, preserve
           its force unless it sounds managerial or unsupported.

        CONTINUITY
        - `script.prose_sections` contains the full current batch. Rewrite all
          of it for the ear while preserving continuity across the batch.
        - If `previous_spoken_tail` is present, continue rather than restart.
          Do not manufacture a new cold open or repeat/paraphrase the previous
          tail unless the same material is also present in the current batch.
        - If `previous_spoken_tail` ends later in time than the current batch's
          main scene, preserve continuity of pressure, theme, or contradiction
          rather than pretending chronology moves straight forward.
        - If this batch begins mid-argument, mid-scene, or mid-pressure, pick
          up the live motion already in progress.

        DELIVERY TARGET
        - The listener should hear a distinctive host mind carrying thought
          forward in pressure and consequence, not page prose with lighter
          punctuation.
        - Prefer paragraph movement like:
          concrete scene or fact -> plain-English translation -> host
          proposition or consequence.
        - When the batch already contains a planned host line, preserve its
          distinctness instead of smoothing it into generic exposition.
        - Make the most important turn, loss, contradiction, decision, or
          consequence easy to hear.
        - Write for a voice that must carry the sentence in one pass. If a
          sentence would likely require rereading on the page, reshape it for
          the ear.
        - Do not overload a sentence with too many new names, titles, places,
          or claims at once. If a sentence asks too much memory work of the
          listener, redistribute the information across adjacent sentences.
        - If the material turns coercive, humiliating, or irreversible, allow
          the prose to become barer and more percussive.
        - If the source is exact, sound exact. If it is approximate,
          contested, or open, keep it that way.

        WHAT NOT TO WRITE
        - Do not tell the listener that a moment matters before the material
          has made it matter.
        - Do not announce hinges, pivots, turning points, or the weight of what
          is coming.
        - Do not use visible planning language or paraphrase `movement_goal`,
          `scene_card_ids`, `framing`, or other pipeline scaffolding into
          audible prose.
        - Do not leak host-target control phrasing such as "tell the listener,"
          "state the through-line," "mark the math," "name the lens," "for the
          rest of the hour," or "the next section."
        - Do not write cold-open resets at later batch boundaries.
        - Do not use narrator nudges, thesis stamps, rhetorical filler, or
          abstract-noun crutches as a substitute for movement.
        - Do not flatten an authored host callback or evaluation into bland
          connective tissue.
        - Avoid topic-announcing transitions and prestige-documentary phrasing
          that sounds composed for admiration on the page rather than for
          one-hearing clarity in the ear.

        TTS AND SPEECH HINTS
        `speech_hints` should help rendering, not compensate for weak prose.
        Add `speech_hints.pronunciation_hints` only for names or terms likely
        to be misread. Keep `spoken_as` concise, keep the hint set small, and
        use `render_strategy`, emphasis, and pacing conservatively.
        Add at most 8 pronunciation hints unless the batch genuinely cannot be
        rendered intelligibly without more. Prefer only high-frequency
        recurring names, unusual transliterations, or terms likely to be
        mangled by TTS.
        If there is no strong reason to do otherwise, prefer restrained delivery:
        - style: measured
        - intensity: light
        - pace: normal
        - render_strategy: plain
        - If the prose still contains several hard-beat lines after rewriting,
          `split_sentences` is acceptable.
        - Preserve the cadence around clarifiers, contrastive turns,
          evaluative closes, and one-sentence dry asides instead of smoothing
          them into neutral exposition.

        OUTPUT
        Return only valid JSON matching `expected_schema` exactly.
        Return one rewritten item per input `script.prose_sections[]`, in the same order.
        Return exactly one top-level key:
        - `sections`

        Each `sections[]` item must include:
        - `section_id`
        - `text`
        - `speech_hints`

        No extra wrapper keys.
        No markdown.
        No commentary.

        SELF-CHECK BEFORE RETURNING
        1. Did you rebuild the spoken architecture rather than paraphrase paragraph by paragraph?
        2. Did you preserve all facts, chronology, names, dates, numbers,
           quotations, and certainty?
        3. If the source timeline or logic was ambiguous, did you handle it
           conservatively instead of smoothing it away?
        4. If `previous_spoken_tail` was present, did it shape continuity
           without being repeated or used as extra source material?
        5. Is the most important turn easy to hear, and are any accurate
           sentences still overloaded?
        6. Do paragraphs now serve spoken logic and land on consequence,
           contradiction, decision, or stake?
        7. Does the narration sound like a serious host carrying thought
           forward in real time rather than a polished essay paragraph?
        8. Does `speech_hints` remain minimal and does the JSON match `expected_schema` exactly?

        Return only the JSON object.
        """
    ).strip()


def style_audit_instructions() -> str:
    return dedent(
        """
        You are the `style_audit` stage for a historical podcast pipeline.

        TASK
        Edit a fully drafted episode script for listener quality before spoken delivery.

        You are not replanning the episode.
        You are not grounding facts.
        You are not adding research.
        You are not changing chronology, argument, or section order.

        Your job is to remove prose patterns that make the script sound like
        pipeline output instead of finished narration, while preserving planned
        host guidance when it is doing real listener work.

        INPUT
        You will receive:
        - `episode_number`
        - `title`
        - `host_policy`
        - optional `narrative_state_pre`
        - optional `narrative_state_post`
        - optional `continuity_contract_pre`
        - optional `continuity_contract_post`
        - optional `field_semantics`
        - optional `series_explanation_registry`
        - `sections[]`

        Each `sections[]` item contains:
        - `section_id`
        - `purpose`
        - `anchor`
        - `closure_mode`
        - `scene_card_count`
        - `projected_word_count`
        - `structural_card_count`
        - `host_moves`
        - `key_terms`
        - `authorial_passages`
        - `term_explanations`
        - `actor_explanations`
        - `text`

        PRIORITY RULES
        - `sections[].text` is the factual source of truth for this stage.
        - `purpose`, `anchor`, `closure_mode`, `host_moves`, `key_terms`, `authorial_passages`, `term_explanations`, `actor_explanations`, `actor_explanation_realizations`, `host_policy`, and optional `series_explanation_registry` are control signals, not evidence.
        - Preserve facts, chronology, names, dates, quotations, uncertainty, and core claims.
        - Keep every section id and section order unchanged. Do not merge or split sections.
        - Do not add new claims, interpretations, motives, or scene material.
        - Prefer cutting repetition before flattening a strong natural host line.
        - Preserve planned explanatory passages when they are doing real listener work.
        - If `continuity_contract_pre.recap_items` is present, do not cut the
          only recap line that realizes that burden.
        - If `continuity_contract_pre.must_surface_early` or
          `continuity_contract_post.must_leave_live` is present, do not delete
          the sole explicit callback, payoff, or residue line carrying a
          high-priority continuity item.
        - Treat compressed host targets as lower-authority than section shape, closure logic, and natural speech.
        - Use `field_semantics` when present as the authoritative gloss for `closure_mode`.
        - Treat `spoken_style_contract = anti_academic_oral` as the default.
          Remove prestige-documentary residue without neutralizing earned
          direct address, plain-English translation, or audible host
          propositions.

        PRIMARY FAILURE MODES TO FIX
        1. Repeated interpretive landing across adjacent sections.
        2. Interior sections should usually end on scene residue, consequence, or open pressure, not explicit explanation.
        3. Abstract-noun thesis drift such as `mechanism`, `architecture`, `system`, `logic`, `structure`, or `framework` unless historically necessary.
        4. Seam handrails such as "which brings us to", "the pattern is", "that is", or other summary-reset lines.
        5. Second endings, including a close that restates the answer instead of exiting.
        6. Repeated causal or pressure restatement in slightly different words.
        7. Overloaded sections whose prose keeps explaining after the point is already clear.
        8. Structural beats drifting into broad synthesis or descriptive backgrounding.
        9. Weak section openings that lose audible orientation.
        10. Planned host phase cues diluted by adjacent explanation, flattened into generic connective tissue, or neutralized into impersonal exposition.
        11. Full redefinition of a foundational term in a later episode when a reminder would do.
        12. Cutting the only clear payoff sentence attached to a first definition.
        13. Visible production-frame phrasing such as "This series...", "This hour...", or "Tonight..." surviving in body prose.
        14. Tasteful recap, prestige-documentary filler, elegant second endings,
            or abstract connective tissue that never cashes out in human terms.

        ALLOWED EDITS
        - Delete repeated sentences.
        - Compress repeated explanation.
        - Remove transition handrails.
        - Replace abstract recap with a more concrete opening anchored in the section's existing material.
        - Soften interior mini-conclusions into residue.
        - Tighten endings so only `closure_mode = final_answer` gets the strongest explicit landing.
        - Shorten over-explained interpretive lines when the scene already carries the point.
        - In sections that already have high `scene_card_count`, high `projected_word_count`, or many structural cards, prefer pruning repeated explanation before preserving descriptive setup.
        - Prefer deletion to paraphrase when both preserve meaning.
        - Preserve a planned host move when it offers earned orientation,
          clarification, evaluation, contrast, callback, naming support, or one
          brief `I`/`we`/`you` aside.
        - Preserve a strong oral pressure line, bargain line, surprise line,
          consequence line, or narrowing line when it is grounded and does real
          listener work.
        - Sharpen one existing line when needed so a planned host move lands
          cleanly at its intended opening, pivot, or closing position.
        - Cut reverse-expanded planning cues when they survive as managerial
          prose instead of natural narration.
        - Preserve planned quote-then-gloss, doctrinal unpacking,
          institutional clarification, causal compression, comparative aside,
          or verdict landing when they are clearly anchored in the existing
          prose.
        - Prefer cutting visible production-frame phrasing before cutting a
          real clarifier or payoff line.
        - Rewrite review-essay, prestige-documentary, or elegant-recap residue
          into active spoken English when a small line-level change can
          preserve meaning and improve oral force.
        - Prefer cutting repetition before softening a hard spoken line that
          is already doing real listener work.
        - If a line merely sounds polished, make it more speakable without
          changing the underlying claim.

        NOT ALLOWED
        - No new facts, chronology, quotations, or framing language not already supported by the text.
        - Do not delete valid host guidance just because it resembles a handrail.
        - Do not add unsupported personal material, fake intimacy, or host performance.
        - No rewriting that changes the meaning of a contested claim.

        OUTPUT
        Return only valid JSON with:
        - `episode_number`
        - `sections[]`
        - `episode_warnings[]`

        Each `sections[]` item must contain:
        - `section_id`
        - `edited_text`
        - `edit_notes[]`
        """
    ).strip()
