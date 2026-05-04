"""Prompt builders for active LLM stages."""

from __future__ import annotations

from textwrap import dedent

from podcast_agent.schemas.models import SYNTHESIS_PRIMITIVE_TARGET_RANGES


def _format_synthesis_primitive_target_ranges() -> str:
    lines = [
        "        Emit only these family keys under primitives_by_family, with the",
        "        soft target ceiling range for each. These are not quotas. Omit weak or",
        "        filler primitives rather than forcing the low end of a range.",
        "",
    ]
    family_notes = {
        "epochal_turns": "Big irreversible pivots that change the rules of the story.",
        "decisions_and_nondecisions": (
            "Choices, refusals, hesitations, delays, and failures to act that\n"
            "          redirect what happens next."
        ),
        "set_piece_scenes": (
            "Major playable scenes: battles, coups, trials, raids, funerals,\n"
            "          rallies, flights, negotiations, ceremonies."
        ),
        "telling_details": (
            "Concrete, memorable local details: gestures, objects, absurdities,\n"
            "          vanity, bureaucratic weirdness, anecdotal texture."
        ),
        "human_costs": (
            "What events cost ordinary people, families, soldiers, prisoners,\n"
            "          refugees, and local communities."
        ),
        "character_engines": "What specific people are trying to protect, prove, gain, avoid, conceal, or survive.",
        "coalitions_and_fault_lines": "How alliances form, hold, strain, split, or quietly rot.",
        "systems_and_operating_logics": (
            "The machinery under the story: institutions, logistics, patronage,\n"
            "          finance, clerical networks, bureaucracy, media ecosystems."
        ),
        "misreadings_and_fantasies": (
            "What people got wrong, wanted to believe, or needed to believe in\n"
            "          real time."
        ),
        "contested_explanations": (
            "Live historical disputes and unresolved competing readings of why\n"
            "          events happened or what they meant.\n"
            "          candidate_readings must present genuinely competing readings."
        ),
        "perspective_windows": "Vantage shifts that materially change the meaning of the story.",
        "moral_traps": (
            "Situations where every real option is compromised and clean judgment\n"
            "          would flatten the reality."
        ),
        "afterlives": "The residues of earlier rupture: traumas, precedents, humiliations, remembered betrayals, inherited fears.",
        "recurring_images_and_symbols": (
            "Images, places, slogans, rituals, documents, buildings, and objects\n"
            "          that can recur across episodes and gather meaning."
        ),
        "ironies_and_reversals": "Backfires, inversions, and cruel flips where actions land opposite to the intended result.",
    }
    for family, (lower_bound, upper_bound) in SYNTHESIS_PRIMITIVE_TARGET_RANGES.items():
        lines.append(f"        {family} ({lower_bound}\u2013{upper_bound})")
        lines.append(f"          {family_notes[family]}")
        lines.append("")
    return "\n".join(lines).rstrip()


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


def theme_decomposition_instructions() -> str:
    return dedent(
        """
        You are the `theme_decomposition` stage for a multi-book thematic podcast pipeline.

        Goal:
        - Convert the project theme into 12-20 strong thematic axes that are useful for downstream retrieval.
        - An axis is an analytical lens, not an episode title and not a generic topic bucket.
        - Identify a compact set of human-led actors that should shape downstream synthesis and storytelling.

        Input payload:
        - `theme`: the main theme.
        - `sub_themes`: optional narrower lenses.
        - `theme_elaboration`: optional project framing.
        - `books`: one object per book, each containing:
          - `book_id`, `title`, `author`
          - `book_summary`
          - `chapters`: compact chapter-analysis objects with `chapter_id`, `title`,
            and `analysis` fields for `themes_touched`, `major_actors`, and
            `key_events_or_arguments`

        Output requirements:
        - Return only valid JSON with keys `axes` and `actor_metadata`.
        - Produce between 12 and 20 axes.
        - Produce between 10 and 40 actors inside `actor_metadata.actors`.
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


def synthesis_primitives_instructions() -> str:
    return dedent(
        f"""
        You are the synthesis_primitives stage for a historical podcast pipeline.
        Read the selected synthesis evidence and extract grounded base primitives.
        These are the raw building blocks for later series design. They are not
        episode architecture, not narrative threads, not verdicts, and not the
        place to add rich family-specific enrichment fields.

        INPUT PAYLOAD
        - project_id
        - axes: compact axis summaries (axis_id, name, description,
          theme_importance_score)
        - passages_by_axis: evidence grouped by axis, then by book
          (book_id, passages with passage_id and text)
        - cross_book_pairs (optional): cross-book pair hints
        - books: compact book metadata
        - actor_metadata (optional): compact canonical actor registry
        - synthesis_feedback (optional): retry feedback; correct the named
          issue without discarding grounded material that already works

        PRIORITY RULES (govern everything below)
        - Ground every primitive in passage_ids present in the input.
          core_passage_ids for decisive evidence, support_passage_ids for
          reinforcing evidence. Never cite a passage_id not in the payload.
        - Every primitive must include its own `family` field, and that value
          must match the `primitives_by_family` bucket key it appears under.
        - Emit only the shared base primitive fields in this pass.
          Do not add family-specific enrichment fields yet.
        - Do not convert contested or uncertain evidence into certainty.
        - Do not emit episode architecture, cluster seeds, merged narratives,
          narrative threads, verdict lists, or omniscient takeaways.
          Primitives only.
        - Use actor_ids only when canonical actors are central to the primitive.
        - Do not force actor ids onto structural primitives.

        PRIMITIVE FAMILIES
{_format_synthesis_primitive_target_ranges()}

        TITLES AND SUMMARIES
        - Titles are operational and narratively usable, not polished thesis
          statements. Think: what would a scene card or planning note call
          this primitive?
          Good: "Nehru learns of partition plan from the radio"
          Bad: "The tragic disconnection between leadership and reality"
        - summary explains what the primitive captures and why it matters.

        NARRATIVE IMPORTANCE SCORING
        
        Score every primitive on 0.0–1.0 using five distinct questions:

        - Causal necessity: If this primitive vanished, how much of the later history would become harder to explain?
        - Scene value: How strongly can this primitive carry a memorable narrated beat on its own?
        - Recurrence value: How useful is this primitive as a callback, motif, or later point of return across episodes?
        - Human/interpretive load: How much does this primitive carry stakes, lived cost, contradiction, or pressure that the series would otherwise flatten?
        - Context value: How much does this primitive help later events feel properly grounded rather than generic or underexplained?
        
        Scoring guide:
        - 0.85-1.00: indispensable load-bearing primitive; either the causal map breaks without it, or the series loses a major scene/
          motif it cannot replace, or later material becomes materially less grounded without it.
        - 0.70-0.84: high-value primitive; strongly useful in at least two of the four dimensions.
        - 0.55-0.69: solid supporting primitive; clearly useful, but replaceable.
        - 0.40-0.54: narrow or local primitive; worth keeping only if it provides distinctive scene fuel, texture, or contrast.
        - below 0.40: marginal primitive; low leverage, low memorability, or duplicative.

        Do not score by family prestige.

        SHARED BASE FIELDS ONLY
        Every primitive in this pass includes only:
        - `id`
        - `family`
        - `title`
        - `summary`
        - `axis_ids`
        - `core_passage_ids`
        - `support_passage_ids`
        - `timeframe`
        - `geography`
        - `actor_ids`
        - `narrative_importance_score`

        Do not emit `primary_actor_ids`, `affected_actor_ids`, `actor_tags`,
        `institution_tags`, `unresolved_actor_tags`, or any family-specific
        rich fields in this pass.

        AXIS LINKING
        - axis_ids references the relevant analytical lenses.

        QUALITY
        - Favor primitives that help later episode construction: spine,
          scene fuel, human grounding, operating logic, unresolved
          interpretive pressure, recurring memory hooks and contextual grounding.
        - Preserve primitives that are likely to survive into episode cores,
          recalls, or later callbacks. Do not treat importance as
          identical to likely spine status.
        - When trimming mentally, remove duplicates before removing
          distinctive texture or useful contextual grounding
        - Protect local texture. `telling_details` and
          `recurring_images_and_symbols` are not filler and should not be
          collapsed into generic set pieces.
        - Preserve both scene utility and series utility. A healthy pool has
          playable scenes, analytical structure, and motifs that can recur.
        - Prefer actor-linked primitives where evidence supports pressure,
          decision, conflict, misreading, consequence, or stakes.
        - Deduplicate obvious repeats. Do not collapse genuinely different
          mechanisms or consequences into one vague object.
        - Do not relabel the same claim across families unless the new
          primitive adds a distinct actor-level insight.
        - Limit passage reuse across primitives unless necessary.
        - Use quality_notes to describe notable gaps or caution areas.
        """
    ).strip()


_PRIMITIVE_ENRICHMENT_PROMPT_CONTEXT: dict[str, dict[str, object]] = {
    "epochal_turns": {
        "type_label": "epochal-turn primitives",
        "task_line": "You are enriching epochal-turn primitives.",
        "required_fields": "`before_state`, `after_state`, `change_driver`, `proof_of_change`, `why_no_return`, `narration_hooks`",
        "good_looks_like": [
            "A strong epochal turn marks a genuine break in the rules of the story, not just an important event inside the existing order.",
            "The best outputs make the prior order, the new order, the proof that the break landed, and the reason return failed all legible in separate moves.",
        ],
        "field_distinctions": [
            "`before_state` names the old order, not setup chronology.",
            "`after_state` names the new order, not later downstream history.",
            "`change_driver` identifies the hinge cause that forces the transition.",
            "`proof_of_change` gives the concrete sign that made the turn undeniable to actors on the ground.",
            "`why_no_return` explains why the old arrangement could not simply be restored.",
            "`narration_hooks.concrete_detail` gives the most speakable detail, image, gesture, or phrase in the evidence.",
            "`narration_hooks.host_lens` gives a narrow, evidence-grounded orienting angle a narrator could use.",
            "`narration_hooks.carry_forward` gives the residue this turn leaves for later scenes or episodes.",
        ],
        "failure_modes": [
            "Do not upgrade ordinary escalation, battle intensity, or local drama into an epochal break unless the evidence shows a real shift in the governing situation.",
            "Bad pattern: `change_driver` says the empire weakens and `why_no_return` says the empire becomes weak.",
            "Do not let `proof_of_change` turn into a second `after_state` sentence.",
        ],
        "stay_narrow": [
            "If the passage only shows acceleration inside an existing trajectory, describe that acceleration narrowly rather than declaring a new era.",
        ],
        "self_check": [
            "Did you identify a real rule-changing break rather than a merely important episode?",
            "Do `before_state`, `after_state`, `proof_of_change`, and `why_no_return` do different jobs?",
        ],
    },
    "decisions_and_nondecisions": {
        "type_label": "decision and nondecision primitives",
        "task_line": "You are enriching decision and nondecision primitives.",
        "required_fields": "`actor_ids`, `decision_trigger`, `decision_question`, `decision_mode`, `options_considered`, `next_result`, `narration_hooks`",
        "good_looks_like": [
            "A strong output names who faced the choice, what forced the hinge, what live question they were resolving, what options were live, and what changed next.",
            "The best outputs distinguish active choice from refusal, delay, drift, or failure to decide.",
        ],
        "field_distinctions": [
            "`decision_trigger` is the immediate pressure or event that forces the hinge now.",
            "`decision_question` should name the live hinge the actor is resolving, not a historian's retrospective thesis about the entire episode.",
            "`decision_mode` must distinguish active decision, refusal, delay, and nondecision carefully.",
            "`options_considered` should be 2-4 short live options visible in the evidence, not a historian's full menu of hypotheticals.",
            "`next_result` must be the first visible consequence, not a long downstream chain.",
            "Use `actor_ids` when canonical actors clearly own or carry the choice; leave `actor_ids` empty when the choice remains institutionally or structurally attributed in the evidence.",
            "`narration_hooks.concrete_detail` should name the most speakable detail attached to the choice point.",
            "`narration_hooks.host_lens` should help the narrator orient the listener to why this hinge matters now.",
            "`narration_hooks.carry_forward` should capture what this decision leaves in motion.",
        ],
        "failure_modes": [
            "Do not collapse later regime consequences, wars, or dynastic outcomes into `next_result`.",
            "Bad pattern: a ruler hesitates, but the output labels it as a fully executed decision because the later outcome was dramatic.",
            "Do not invent actor linkage just to make the primitive feel more personalized.",
        ],
        "stay_narrow": [
            "If the evidence only supports a constrained pause, refusal, or drift, keep the primitive at that scale instead of inflating it into grand strategy.",
        ],
        "self_check": [
            "Does `decision_mode` match what the actor actually did or failed to do?",
            "Are `decision_trigger`, `decision_question`, `options_considered`, and `next_result` functionally distinct?",
        ],
    },
    "set_piece_scenes": {
        "type_label": "set-piece scene primitives",
        "task_line": "You are enriching set-piece scene primitives.",
        "required_fields": "`actor_ids`, `scene_anchor`, `hinge_action`, `scene_outcome`, `location`, `narration_hooks`",
        "good_looks_like": [
            "A strong set-piece scene is playable: you can picture the setup, the visible turning act, and the landed result.",
            "The best outputs feel stageable, not like a chapter summary cut into fragments.",
        ],
        "field_distinctions": [
            "`scene_anchor` should name the setup image or where pressure concentrates before the turn.",
            "`hinge_action` should name the visible turning act.",
            "`scene_outcome` should name the visible result, not the whole later history.",
            "`location` should identify where the scene materially happens, not the larger region unless the evidence stays broad.",
            "Use `actor_ids` when canonical actors materially anchor the scene; leave `actor_ids` empty when the scene is principally event-, crowd-, or place-anchored in the evidence.",
            "`narration_hooks.concrete_detail` should give the most speakable detail in the scene.",
            "`narration_hooks.host_lens` should help the narrator orient the listener without turning the scene into commentary.",
            "`narration_hooks.carry_forward` should name what residue this scene leaves behind.",
        ],
        "failure_modes": [
            "Do not turn broad chronology, campaign background, or multi-month drift into a pseudo-scene.",
            "Bad pattern: `scene_anchor` says the empire faces crisis, `hinge_action` is vague, and `scene_outcome` just says the crisis continues.",
        ],
        "stay_narrow": [
            "If the evidence gives atmosphere and consequence but no clear hinge moment, keep the scene anchor tightly framed around what the passage actually stages.",
        ],
        "self_check": [
            "Could a listener picture setup, act, and result distinctly?",
        ],
    },
    "human_costs": {
        "type_label": "human-cost primitives",
        "task_line": "You are enriching human-cost primitives.",
        "required_fields": "`actor_ids`, `affected_group`, `cost_type`, `concrete_marker`, `lived_consequence`, `who_saw_it`, `narration_hooks`",
        "good_looks_like": [
            "A strong human-cost primitive makes the harm land on a concrete group in lived terms rather than describing suffering at the level of abstract social damage.",
            "The best outputs distinguish who is hurt, what kind of harm it is, how it is experienced, and who can or cannot ignore it.",
        ],
        "field_distinctions": [
            "`affected_group` names who absorbs the harm.",
            "`cost_type` is the category of harm; `lived_consequence` is how that harm lands in lived terms.",
            "`concrete_marker` is the most speakable physical or social sign of that harm.",
            "`who_saw_it` is about what was public, buried, deniable, or legible only later.",
            "Use `actor_ids` when a canonical actor clearly causes, directs, personifies, or administers the harm.",
            "Leave `actor_ids` empty when the harm is diffuse, structural, collective, or not cleanly attributable to canonical actors.",
            "`narration_hooks.concrete_detail` should not duplicate `concrete_marker`; it should sharpen the same harm from another speakable angle when possible.",
            "`narration_hooks.host_lens` should help a narrator frame the human meaning without moralizing beyond the evidence.",
            "`narration_hooks.carry_forward` should name the residue the harm leaves in later political or social life.",
        ],
        "failure_modes": [
            "Do not invent actor linkage just to satisfy schema pressure.",
            "Do not let `affected_group`, `concrete_marker`, and `lived_consequence` collapse into the same sentence in slightly different words.",
        ],
        "stay_narrow": [
            "If the evidence only supports generalized hardship, keep the claim specific to what is actually described instead of inferring a broader social totality.",
        ],
        "self_check": [
            "Does the output show the cost as lived experience rather than abstract commentary?",
            "Is `who_saw_it` about visibility and denial, not a second harm description?",
        ],
    },
    "character_engines": {
        "type_label": "character-engine primitives",
        "task_line": "You are enriching character-engine primitives.",
        "required_fields": "`actor_id`, `goal`, `pressure_box`, `risk_if_it_breaks`, `tell`, `narration_hooks`",
        "good_looks_like": [
            "A strong character engine makes one actor's operating pressure legible through distinct objective, compression, risk, and observable pattern.",
            "The best outputs help explain why the actor keeps moving as they do without inventing private psychology.",
        ],
        "field_distinctions": [
            "Use `actor_id` when one canonical actor clearly carries the pressure profile; return `null` when the evidence supports the pressure but not a clean canonical actor mapping.",
            "`goal` is what the actor is trying to secure, gain, prove, protect, or avoid.",
            "`pressure_box` is the combined trap, squeeze, or limiting box the actor is operating inside.",
            "`risk_if_it_breaks` is what the actor loses if the engine fails or the pressure snaps.",
            "`tell` is the observable behavior, phrase, reflex, or self-justifying pattern that makes the engine legible in scenes.",
            "`narration_hooks.concrete_detail` should give the best speakable detail attached to the pressure profile.",
            "`narration_hooks.host_lens` should help the narrator explain the engine cleanly without pretending to know private interiority.",
            "`narration_hooks.carry_forward` should identify what this engine keeps pushing into later scenes.",
        ],
        "failure_modes": [
            "Bad pattern: `goal`, `pressure_box`, and `risk_if_it_breaks` all say 'retain power' in slightly different words.",
            "Better pattern: distinct objective, trap, consequence, and tell.",
            "Do not turn general biography into pressure unless the evidence shows it operating in this moment.",
        ],
        "stay_narrow": [
            "If the evidence only supports one clear pressure, keep the rest restrained instead of inventing a fully rounded interior profile.",
        ],
        "self_check": [
            "Are `goal`, `pressure_box`, `risk_if_it_breaks`, and `tell` functionally different?",
        ],
    },
    "coalitions_and_fault_lines": {
        "type_label": "coalition and fault-line primitives",
        "task_line": "You are enriching coalition and fault-line primitives.",
        "required_fields": "`actor_ids`, `alignment_type`, `coalition_phase`, `alignment_shape`, `alignment_basis`, `fracture_trigger`, `narration_hooks`",
        "good_looks_like": [
            "A strong coalition primitive shows why actors align, what holds them together for now, what contradiction sits underneath, and where that contradiction starts to bite.",
            "The best outputs distinguish durable shared interest from opportunistic overlap.",
        ],
        "field_distinctions": [
            "`alignment_type` classifies the kind of alignment: tactical, strategic, institutional, or situational.",
            "`coalition_phase` says whether the evidence shows the coalition forming, holding, fracturing, or breaking.",
            "Do not invent a coalition if the evidence only shows temporary alignment, parallel incentives, or one-off overlap.",
            "`alignment_shape` says what the alignment actually is.",
            "`alignment_basis` is the practical reason it holds for now.",
            "`fracture_trigger` is the condition, event, or contradiction that activates the break.",
            "Leave `actor_ids` empty when the alignment is real but the evidence does not support a clean canonical actor set.",
            "`narration_hooks.concrete_detail` should give the best speakable sign of the alignment or its strain.",
            "`narration_hooks.host_lens` should help the narrator explain what kind of alliance this is without inflating it.",
            "`narration_hooks.carry_forward` should name the later stress or residue the alignment leaves behind.",
        ],
        "failure_modes": [
            "Do not confuse simple coexistence, shared enemies, or momentary simultaneity with a meaningful coalition.",
            "Bad pattern: `alignment_basis` says both sides want order and `fracture_trigger` just repeats that they disagree.",
        ],
        "stay_narrow": [
            "If the evidence only supports a narrow tactical alignment, keep the coalition limited to that alignment rather than implying a deeper political bloc.",
        ],
        "self_check": [
            "Did you identify a real alignment basis and a distinct fracture trigger?",
        ],
    },
    "systems_and_operating_logics": {
        "type_label": "systems and operating-logics primitives",
        "task_line": "You are enriching systems and operating-logics primitives.",
        "required_fields": "`system_name`, `operating_chain`, `inputs`, `outputs`, `where_it_shows_up`, `failure_mode`, `narration_hooks`",
        "good_looks_like": [
            "A strong systems primitive describes a concrete operating chain: something goes in, something moves through specific channels, something comes out, and something breaks or distorts under pressure.",
            "The best outputs describe machinery that historical actors could actually experience, manipulate, or suffer from.",
        ],
        "field_distinctions": [
            "Describe a real operating chain, not abstract thesis prose.",
            "`system_name` should name the actual system, not an entire civilization-sized order.",
            "`operating_chain` should give 2-4 short ordered concrete steps inside the chain.",
            "`inputs` and `outputs` should be concrete, historically legible items or effects, not abstractions.",
            "`where_it_shows_up` should identify where a listener would actually see this system work.",
            "`failure_mode` should say how the system distorts, stalls, breaks, or backfires.",
            "`narration_hooks.concrete_detail` should give the best speakable sign of the system in action.",
            "`narration_hooks.host_lens` should help the narrator orient the listener to the operating logic without thesis prose.",
            "`narration_hooks.carry_forward` should name what later residue this system leaves behind.",
        ],
        "failure_modes": [
            "Bad pattern: 'The political system processes pressure through institutions.'",
            "Better pattern: a concrete flow of orders, incentives, bottlenecks, patronage, logistics, rumor, coercion, or information.",
            "Do not use ideology or atmosphere alone as a system unless the evidence shows an operating chain.",
        ],
        "stay_narrow": [
            "If the passage only shows one bottleneck or one distorted channel, describe that chain specifically instead of naming an entire civilization-sized system.",
        ],
        "self_check": [
            "Is the operating chain concrete rather than abstract filler?",
            "Could a planner stage `where_it_shows_up` without inventing new evidence?",
        ],
    },
    "contested_explanations": {
        "type_label": "contested-explanation primitives",
        "task_line": "You are enriching contested-explanation primitives.",
        "required_fields": "`candidate_readings` with at least two genuinely competing readings, plus `narration_hooks`",
        "good_looks_like": [
            "A strong contested explanation sets two real explanations against each other, not two tones of the same explanation.",
            "The best outputs make clear what each reading emphasizes, what each downplays, and why both remain plausible from the supplied evidence.",
        ],
        "field_distinctions": [
            "The readings must disagree in interpretation, causality, emphasis, or explanatory weight.",
            "Each reading's `claim` states the explanation itself.",
            "Each reading's `emphasizes` says what evidence, cause, or logic it weights heavily.",
            "Each reading's `downplays` says what it minimizes or treats as secondary.",
            "Each reading's `support_passage_ids` should point only to passages that really support that reading.",
            "Do not return two phrasings of the same reading.",
            "`narration_hooks.concrete_detail` should give the best speakable detail that keeps the dispute grounded.",
            "`narration_hooks.host_lens` should help the narrator name the dispute cleanly without deciding it prematurely.",
            "`narration_hooks.carry_forward` should name why this dispute matters later.",
        ],
        "failure_modes": [
            "Bad pattern: one reading says leaders miscalculated; the other says leaders made a mistaken calculation.",
            "Do not manufacture disagreement by relabeling one side as moral and the other as strategic if the underlying claim stays the same.",
        ],
        "stay_narrow": [
            "If the evidence supports only a narrow dispute over emphasis, state that narrow dispute clearly instead of pretending the readings are fully incompatible.",
        ],
        "self_check": [
            "Do the candidate readings genuinely disagree?",
            "Does each reading specify `claim`, `emphasizes`, and `downplays` distinctly?",
        ],
    },
    "moral_traps": {
        "type_label": "moral-trap primitives",
        "task_line": "You are enriching moral-trap primitives.",
        "required_fields": "`actor_ids`, `competing_obligations`, `compromised_options`, `trap_structure`, `narration_hooks`",
        "good_looks_like": [
            "A strong moral-trap primitive shows an actor bound by real duties or loyalties, with available actions that each violate something serious.",
            "The best outputs make the trap structural, not just emotionally difficult.",
        ],
        "field_distinctions": [
            "`competing_obligations` should be actual obligations, loyalties, or duties.",
            "`compromised_options` should be live available actions, not abstract values or retrospective judgments.",
            "`compromised_options` should be actions available to the actor, not abstract values.",
            "`trap_structure` should explain why every live move damages something binding.",
            "Leave `actor_ids` empty when the trap is structurally real but not cleanly attributable to canonical actors in the supplied evidence.",
            "Do not let `competing_obligations` and `compromised_options` collapse into the same list in different wording.",
            "`narration_hooks.concrete_detail` should give the best speakable sign of the trap on the ground.",
            "`narration_hooks.host_lens` should help the narrator frame why there is no clean path without moral inflation.",
            "`narration_hooks.carry_forward` should name the later residue the trap leaves behind.",
        ],
        "failure_modes": [
            "Do not confuse a hard preference, policy disagreement, or sad outcome with a moral trap unless all available paths are compromised.",
            "Bad pattern: one option is clearly clean and the output still calls the situation a trap.",
        ],
        "stay_narrow": [
            "If the evidence only shows a difficult choice without conflicting obligations, keep the claim at the level of pressure or dilemma rather than calling it a moral trap.",
        ],
        "self_check": [
            "Would every available move force the actor to violate or damage something they are bound to protect?",
            "Is `trap_structure` explaining the structure of compromise rather than repeating the options list?",
        ],
    },
    "ironies_and_reversals": {
        "type_label": "irony and reversal primitives",
        "task_line": "You are enriching irony and reversal primitives.",
        "required_fields": "`actor_ids`, `expected_outcome`, `actual_outcome`, `flip_cause`, `narration_hooks`",
        "good_looks_like": [
            "A strong irony or reversal shows actors aiming at one result and producing a meaningfully inverted result through a visible mechanism.",
            "The best outputs capture backfire, inversion, or cruel flip, not just ordinary disappointment.",
        ],
        "field_distinctions": [
            "`expected_outcome` should be what actors plausibly aimed for or assumed.",
            "`actual_outcome` should materially invert or undercut that expectation.",
            "`flip_cause` should identify why the result flipped.",
            "Leave `actor_ids` empty when the reversal is clear but the evidence does not support a stable canonical actor set.",
            "`narration_hooks.concrete_detail` should give the best speakable sign of the backfire.",
            "`narration_hooks.host_lens` should help the narrator frame the inversion cleanly.",
            "`narration_hooks.carry_forward` should name the residue the reversal leaves behind.",
        ],
        "failure_modes": [
            "Do not label every failed plan a reversal; the result should land opposite to or corrosively undercut the intended effect.",
            "Bad pattern: the expected outcome is success, the actual outcome is partial failure, and no inversion is visible.",
        ],
        "stay_narrow": [
            "If the evidence shows only setback or friction, describe that setback narrowly rather than forcing a full irony arc.",
        ],
        "self_check": [
            "Is the result genuinely inverted or backfiring, rather than merely weaker than hoped?",
            "Do `expected_outcome`, `actual_outcome`, and `flip_cause` form a real reversal rather than a simple setback?",
        ],
    },
    "telling_details": {
        "type_label": "telling-detail primitives",
        "task_line": "You are enriching telling-detail primitives.",
        "required_fields": "`narration_hooks`",
        "good_looks_like": [
            "A strong telling detail gives the narrator one speakable image and one plain reason it matters.",
        ],
        "field_distinctions": [
            "`narration_hooks.concrete_detail` should preserve the detail itself.",
            "`narration_hooks.host_lens` should say what kind of pressure or meaning this detail helps orient.",
            "`narration_hooks.carry_forward` should name the residue this detail leaves for later scenes or callbacks.",
            "`narration_hooks.plain_gloss` should offer one plain-English translation when the detail needs unpacking.",
            "`narration_hooks.authorial_move` should stay narrow and default to compression or clarifier moves, not verdict inflation.",
        ],
        "failure_modes": [
            "Do not replace the detail with generic symbolism.",
        ],
        "stay_narrow": [
            "If the evidence only supports local texture, keep the hook local.",
        ],
        "self_check": [
            "Did you preserve the detail rather than abstracting it away?",
        ],
    },
    "misreadings_and_fantasies": {
        "type_label": "misreading and fantasy primitives",
        "task_line": "You are enriching misreading and fantasy primitives.",
        "required_fields": "`narration_hooks`",
        "good_looks_like": [
            "A strong output gives the narrator one clean way to explain the misread without overclaiming motive.",
        ],
        "field_distinctions": [
            "`narration_hooks.listener_confusion` should name what a listener might otherwise misread here.",
            "`narration_hooks.plain_gloss` should say plainly what the misreading gets wrong.",
        ],
        "failure_modes": [
            "Do not turn a narrow misreading into a total civilizational delusion.",
        ],
        "stay_narrow": [
            "If the evidence only supports one mistaken expectation, keep the hook at that scale.",
        ],
        "self_check": [
            "Is the hook clarifying the mistake without inflating its scope?",
        ],
    },
    "perspective_windows": {
        "type_label": "perspective-window primitives",
        "task_line": "You are enriching perspective-window primitives.",
        "required_fields": "`narration_hooks`",
        "good_looks_like": [
            "A strong output helps the narrator explain what this angle lets the listener newly see.",
        ],
        "field_distinctions": [
            "`narration_hooks.host_lens` should orient the listener to what this perspective clarifies.",
            "`narration_hooks.authorial_move` should usually stay `comparative_aside` or `causal_compression`.",
        ],
        "failure_modes": [
            "Do not turn a perspective window into a complete explanation of the episode.",
        ],
        "stay_narrow": [
            "If the evidence only opens one side window, keep the hook narrow.",
        ],
        "self_check": [
            "Does the hook preserve this as one angle, not the whole map?",
        ],
    },
    "afterlives": {
        "type_label": "afterlife primitives",
        "task_line": "You are enriching afterlife primitives.",
        "required_fields": "`narration_hooks`",
        "good_looks_like": [
            "A strong output helps the narrator say what survives and why that survival matters later.",
        ],
        "field_distinctions": [
            "`narration_hooks.carry_forward` should be especially concrete here.",
            "`narration_hooks.plain_gloss` should say what later pressure or memory this survival keeps alive.",
        ],
        "failure_modes": [
            "Do not smuggle a full concluding verdict into the hook.",
        ],
        "stay_narrow": [
            "If only one residue survives, name that residue rather than summarizing the entire legacy.",
        ],
        "self_check": [
            "Did you identify one real residue rather than a generic legacy line?",
        ],
    },
    "recurring_images_and_symbols": {
        "type_label": "recurring-image primitives",
        "task_line": "You are enriching recurring-image primitives.",
        "required_fields": "`narration_hooks`",
        "good_looks_like": [
            "A strong output keeps the image concrete and gives the narrator a reusable callback angle.",
        ],
        "field_distinctions": [
            "`narration_hooks.concrete_detail` should preserve the image itself.",
            "`narration_hooks.carry_forward` should make recurrence usable later without over-explaining the symbol.",
        ],
        "failure_modes": [
            "Do not flatten the image into generic symbolism detached from the evidence.",
        ],
        "stay_narrow": [
            "If the image only recurs weakly, keep the hook practical rather than grand.",
        ],
        "self_check": [
            "Could a later callback reuse this hook without needing a new explanation?",
        ],
    },
}


def _primitive_enrichment_prompt_context(family: str) -> dict[str, object]:
    try:
        return _PRIMITIVE_ENRICHMENT_PROMPT_CONTEXT[family]
    except KeyError as exc:
        raise ValueError(f"Unknown primitive enrichment family: {family}") from exc


def _primitive_enrichment_type_specific_block(family: str) -> str:
    context = _primitive_enrichment_prompt_context(family)
    good_looks_like = "\n".join(
        f"- {line}" for line in context["good_looks_like"]
    )
    field_distinctions = "\n".join(
        f"- {line}" for line in context["field_distinctions"]
    )
    failure_modes = "\n".join(f"- {line}" for line in context["failure_modes"])
    stay_narrow = "\n".join(f"- {line}" for line in context["stay_narrow"])
    if failure_modes:
        failure_modes = f"\n\nFAILURE MODES\n{failure_modes}"
    if stay_narrow:
        stay_narrow = f"\n\nWHEN TO STAY NARROW\n{stay_narrow}"
    return dedent(
        f"""
        TASK
        - {context["task_line"]}

        REQUIRED FIELDS
        - Return these fields for every selected primitive: {context["required_fields"]}

        WHAT GOOD LOOKS LIKE
        {good_looks_like}

        FIELD DISTINCTIONS
        {field_distinctions}{failure_modes}{stay_narrow}
        """
    ).strip()


def _primitive_enrichment_additional_self_check(family: str) -> str:
    context = _primitive_enrichment_prompt_context(family)
    lines = [f"- {line}" for line in context["self_check"]]
    return "\n".join(lines)


def primitive_enrichment_instructions(family: str | None = None) -> str:
    if family is None:
        return dedent(
            """
            You are the `primitive_enrichment` stage for a historical podcast pipeline.

            Runtime dispatch supplies a type-native prompt for one primitive type per batch.
            Use the family-specific runtime prompt rather than this generic fallback when
            generating enrichment output.
            """
        ).strip()

    context = _primitive_enrichment_prompt_context(family)
    return dedent(
        f"""
        You are the `primitive_enrichment` stage for a historical podcast pipeline.

        Goal:
        You receive already-selected {context["type_label"]} in one batch.
        Enrich them into a stronger downstream-ready shape by filling only the
        required fields that are still missing.

        INPUT AND AUTHORITY
        - `project_id`
        - `family`: fixed output-shape value for this batch
        - `base_primitives`: only the selected primitives for this batch
        - `evidence_by_primitive_id`: primitive-scoped trimmed evidence with `core_passages` and `support_passages`
        - `narrator_profile` (optional): strategy-level narrator method; use it to calibrate hook shape, not to invent new claims
        - `actor_metadata` (optional): slim canonical actor context with `one_line_role`
        - `enrichment_feedback` (optional): retry feedback; if present, correct the named issue exactly and keep all other valid output unchanged

        Authority order:
        1. each primitive's `core_passages` and `support_passages` under `evidence_by_primitive_id`
        2. each primitive's existing `title`, `summary`, passage ids, and context fields
        3. `actor_metadata` for canonical ids and compact actor context

        If these conflict, preserve the supplied primitive claim and the passage evidence.

        PRIORITY RULES
        - Return only deltas: `id`, `family`, and the required fields below.
        - Do not add new primitives, merge primitives, split primitives, change type,
          or replace the primitive's underlying claim.
        - Ground every enriched field in the primitive's provided evidence bundle.
          Do not invent passage ids or outside evidence.
        - Treat core passages as decisive evidence. Support passages may sharpen
          or contextualize a claim, but they should not override the primitive's
          underlying claim.
        - Be conservative. If the evidence is thin, choose the narrowest
          supportable formulation rather than a sweeping one.
        - Respect the already-selected primitive titles and summaries. Enrichment
          should sharpen them, not replace them with a different claim.
        - Use canonical actor ids from `actor_metadata` when the evidence supports them.
        - Multiple primitives may share a passage. Enrich each primitive according
          to its own title, summary, and evidence role, not according to the entire
          shared passage.
        - If `enrichment_feedback` is present, fix those named issues directly
          without reshaping unaffected primitives or adding new keys.
        - If `enrichment_feedback` names omitted selected primitive ids, return those
          missing ids in the next full batch response while preserving all valid
          unaffected deltas.
        - Avoid generic abstraction drift. Do not hide weak evidence behind vague
          nouns such as `mechanism`, `system`, `logic`, `pressure`, `constraint`,
          or `stakes` unless you make them historically concrete.
        - Keep fields short. Do not turn any field into a mini-summary of the whole chapter.
        - `narration_hooks` are narrator-facing affordances for downstream architecture and writing. They are not script lines. Keep them compact, concrete, and reusable.

        WORKING METHOD
        - First identify what the base primitive is already claiming.
        - Then identify which parts of that primitive's evidence bundle actually
          justify the missing enriched fields for that primitive.
        - Fill only what the evidence can carry.
        - If two enriched fields risk collapsing into the same sentence in different
          words, separate them by function rather than paraphrasing.
        - If one field can be made specific and another cannot, keep the weak field
          narrow instead of inflating both.

        {_primitive_enrichment_type_specific_block(family)}

        FAILURE AND AMBIGUITY HANDLING
        - If the evidence supports only part of the enriched shape, still return the
          narrowest valid formulation for every required field rather than escalating
          into a different claim.
        - Do not over-resolve local ambiguity by importing wider series framing.
        - Do not smuggle episode planning, thesis language, or retrospective omniscience
          into the enriched fields.
        - Do not let any field become a paragraph, a chapter summary, or a second version
          of another field.

        OUTPUT CONTRACT
        Return a JSON object with:
        - `project_id`
        - `family`
        - `enriched_primitives`

        Return one enriched delta for each selected primitive in this batch.
        Preserve each primitive `id` and the fixed `family` value for the batch.
        Do not emit shared base fields unless the required fields above include them.

        SELF-CHECK BEFORE RETURNING
        - Does every returned item preserve the batch's fixed `family` value?
        - Did you preserve the primitive's underlying claim instead of replacing it?
        - Are the enriched fields distinct rather than paraphrases of one another?
        - Are actor ids canonical where you used them?
        {_primitive_enrichment_additional_self_check(family)}

        Do not add markdown or commentary outside the JSON object.
        """
    ).strip()


def narrative_strategy_instructions() -> str:
    return dedent(
        """
        You are the `narrative_strategy` stage for a historical podcast pipeline.
        
        Turn the primitive synthesis map into a series structure.
        The assignment unit is the individual primitive.
        You are deciding which primitives form each episode’s load-bearing core, which
        primitives are support with typed subordinate roles, and which earlier
        primitives may be recalled later.
        You are not drafting scenes.

        INPUT PAYLOAD
        - `synthesis_map`: primitive-only synthesis artifact
        - `project`: project metadata, runtime bounds, book metadata
        - `actor_metadata` (optional): canonical actor context
        - `requested_episode_count` (optional): hard episode-count constraint
        - `recommended_episode_count_min`: lower bound when no explicit count is requested
        - `recommended_episode_count_max`: upper bound when no explicit count is requested
        - `strategy_feedback` (optional): retry feedback from the orchestrator

        EPISODE COUNT
        - If `requested_episode_count` is present, treat it as binding.
        - Otherwise, produce between `recommended_episode_count_min` and `recommended_episode_count_max` episodes, inclusive.

        PRIORITY RULES
        - Every episode must have exactly one `episode_spine`.
        - `listener_problem` is the listener-facing problem the episode carries.
        - `episode_answer` is the concise answer the episode earns.
        - `pressure_line` is the live pressure or contradiction the listener should feel while moving through the episode.
        - `core_primitive_ids` are the binding episode contract.
        - Support primitives must be typed with exactly one role each:
          `stakes`, `mechanism`, `counterpressure`, `consequence`, or `texture`.
        - Infer support role and recall eligibility from primitive title, summary,
          family, score, passage grounding, actor context, and time/place context.
        - Strategy assigns primitives, not scenes. Do not produce scene-level detail,
          pacing, beats, or architecture.

        SUPPORT ROLE DEFINITIONS
        - `stakes`: raises what can be lost, protected, or irreversibly changed
        - `mechanism`: explains how the episode’s proposition works in practice
        - `counterpressure`: supplies the force, contradiction, or rival logic resisting it
        - `consequence`: shows what the proposition causes, unlocks, or damages downstream
        - `texture`: adds grounded lived detail or context that strengthens the same proposition without carrying it alone

        FIRST-PASS GROUPING WORKFLOW
        - Draft the series using primitives first.
        - Build provisional episodes from primitive title, summary, family,
          actor context, time/place context, causal sequence, and passage grounding.
        - Infer thematic coverage from the primitive set itself rather than from
          external thematic scaffolds.

        PRIMITIVE-FIRST DISCIPLINE
        - Do not recreate abstract thematic buckets as episode titles.
        - Do not let repeated conceptual vocabulary become the organizing outline
          unless the primitives independently force it.
        - Prefer concrete historical problems, reversals, collisions, and decision
          chains over abstract thematic bucket labels.

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

        Choose `chronological` when:
        - sequence itself is the main explanatory engine
        - later episodes inherit institutions, pressures, or consequences from earlier ones
        - reversals and transformations make most sense in time order
        - the listener’s understanding should come from watching the process unfold
        - compressing time into propositions would flatten causality or erase distinct eras

        Choose `thesis_driven` only when:
        - one proposition clearly organizes the series better than temporal unfolding
        - episodes are strongest as arguments rather than phases
        - the same causal claim must be examined from several angles that chronology alone would not organize well

        Choose `convergence` when:
        - separate tracks matter independently for a while and then collide
        - the hinge event is best understood as a meeting point rather than a linear progression

        Choose `debate` when:
        - rival explanations or rival political logics can structure multiple episodes
        - disagreement is genuinely load-bearing and not just a support note

        Choose `mosaic` only when:
        - juxtaposed partial views are genuinely the best explanatory shape
        - linear causality is not the dominant organizing truth

        Use `justification` to describe the actual macro-shape, not a generic preference.

        SERIES SHAPE RULES
        Build episodes around escalation, consequence, contestation, discovery, and payoff.
        Do not partition material evenly.
        Do not organize episodes as biographies unless the evidence genuinely warrants it.
        Keep each episode centered on one main problem; secondary pressures should sharpen
        that problem, not open a neighboring one.

        EVIDENCE BALANCE
        Balance evidence across episodes by sufficiency, not equality.

        Rules:
        - Do not let one episode become evidence-thin while another absorbs materially reusable support primitives with a much deeper passage pool.
        - When multiple candidate primitives perform similar episode work, prefer the combination that broadens passage grounding or interpretive coverage rather than stacking near-duplicate argumentative work.
        - Avoid assigning several primitives that rely on the same small passage cluster or do the same local argumentative job unless that concentration is necessary to the episode’s core claim.
        - Prefer support sets that add evidentiary or interpretive coverage rather than repeating the same passage-grounded claim or adjacent argumentative function in different language.
        - If an episode’s assigned primitive set is heavily concentrated on a few shared passages or one narrow evidentiary bundle, prefer lower-overlap primitives when they can do comparable episode work.
        - If two episodes could plausibly host the same non-core primitive, prefer the assignment that strengthens the weaker episode’s evidence base.
        - If an episode cannot support its assigned primitive load with a distinct enough evidence base, reassign support primitives before accepting the partition.
        - Base primitives are intentionally lean in this stage. Infer episode work
          from title, summary, family, score, actor_ids, timeframe, geography,
          and passage grounding rather than expecting rich family-specific fields.

        EPISODE BALANCE CHECK
        For each episode, silently check:
        - Does this episode have enough distinct passage grounding to sustain its assigned primitive load?
        - Is its support set broadening the evidence base or just stacking adjacent claims?
        - Is this episode obviously thinner than its neighbors while carrying a similar primitive count?
        - Is there a comparable support primitive that would improve this episode’s grounding?
        - Is this episode strong because of evidence, or only because of structural elegance?

        If an episode is elegant but under-grounded, strengthen grounding first.

        NARRATOR METHOD
        Alongside the episode arc, derive a usable narrator contract from the base primitives.
        You are not placing explanatory passages or writing gloss lines. You are choosing how explanation should behave at the episode level.

        Use the primitive mix, chronology, institutional density, actor pressure, and doctrinal load to infer:
        - whether the season is mostly scene-led or needs more explicit explanation
        - which episodes need stronger doctrinal unpacking, institutional clarification, quote-then-gloss, or harder verdict landings
        - which terms or institutions the listener must not be left to inference alone

        EPISODES
        Each episode must include:
        - `episode_number`
        - `title`
        - `thematic_focus`
        - `arc_summary`
        - `unresolved_questions`
        - `episode_spine`
        - `actor_arc_directives`
        - `authorial_contract`

        EPISODE SPINE
        Each `episode_spine` only includes:
        - `listener_problem`
        - `episode_answer`
        - `pressure_line`
        - `core_primitive_ids`
        - `support_primitive_roles`
        - `recall_primitive_ids`

        EPISODE SPINE RULES
        - `core_primitive_ids` must contain 5-7 primitives.
        - At least two core primitives must come from `epochal_turns` or `decisions_and_nondecisions`.
        - `support_primitive_roles` must contain 5-7 primitives.
        - Each support primitive gets exactly one support role.
        - Support primitives cannot also appear in the core.
        - `recall_primitive_ids` are optional and must contain at most 2 primitives.
        - Use recall only when it is explicitly justified and materially helps the listener carry accumulated meaning forward.
        - Prefer the smallest support set that materially broadens grounding, pressure, or consequence.
        - Do not stack multiple support primitives that rely on the same passage cluster or perform the same argumentative job unless the core claim truly needs that density.

        STRATEGY RULES
        - `listener_problem` should name the episode's actual listener problem.
        - `episode_answer` should say what becomes clear by the end rather than what each scene must prove.
        - `pressure_line` should be short, concrete, and live enough to carry an oral narrative.
        - Allow later recalls only when explicitly justified.
        - Fail rather than auto-rescoping if no dominant listener problem can be formed.
        - Do not use later payoff material to artificially prop up an earlier weak episode unless recall or consequence logic clearly justifies it.

        ACTOR ARC DIRECTIVES
        Actor arc directives are episode-specific planning guidance for how selected
        actors function across scenes.
        They are not synthesis primitives, evidence summaries, or copied metadata.

        Selection:
        - `actor_arc_directives` must contain only the 2-4 actors whose episode function needs explicit planning guidance.
        - Choose actors who give the episode a usable character spine.
        - Do not include an actor just because they appear in clusters or primitives.

        Each `actor_arc_directives[]` item has:
        - `actor_id`
        - `arc_threads`

        Each `arc_threads[]` item has:
        - `thread_id`
        - `arc_type`: `role`, `tracking`, `tension`, `turn`, `payoff`, or `guardrail`
        - `label`
        - `premise`
        - `pressure`
        - `movement`
        - `resolution`

        QUALITY
        - Keep the listener-facing question narrow and concrete.
        - Build each episode around one controlling proposition expressed through one explicit set of core primitives.
        - Keep support subordinate, but do not confuse subordination with piling up multiple primitives that make the same point.
        - Prefer episodes whose evidence, proposition, and causal sequence all reinforce one another.
        - Keep narrator method episode-scaled. Strategy owns explanatory posture and obligations, not section placement or line drafting.
        - `authorial_contract.governing_lenses` should be concise, speakable interpretive pressures, not mini-essays.
        - `must_clarify_terms` and `must_clarify_institutions` should be sparse. Name only what the listener truly needs help carrying.

        OUTPUT
        Return only valid JSON matching `NarrativeStrategy`.
        - `strategy_type` must be one of the schema values.
        - If `recommended_episode_count` is present, it must match the number of episodes produced.
        - If `episode_arc_outline` is present, it must align in length with `episodes`.
        - `narrator_profile` must define narrator method, not just tone.
        - Do not add markdown or commentary.
        """
    ).strip()


def episode_planning_instructions() -> str:
    return dedent(
        """
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
        - `available_passages`       evidence available to this episode
        - `host_policy`              series-level narrator policy for host moves
        - `actor_metadata`           episode-relevant canonical actor context
        - `planning_feedback`        optional retry feedback from the orchestrator

        ==============================================================================
        PRIORITY RULES — WHAT IS BINDING vs. WHAT YOU OWN
        ==============================================================================
        BINDING (do not restate, rewrite, reorder, or reconsider):
        - `strategy_episode` fields: title, thematic focus,
          arc summary, unresolved questions, `episode_spine`, actor arc directives.
        - `architecture.section_id` order.
        - `architecture.major_turn_section_id`.
        - Per-section primitive groupings (`primitive_ids` lists).

        YOU OWN:
        - Framing block (opening image, threat, opening question, handoff target).
        - Scene cards: count, ordering within sections, titles, dramatic roles,
          structural functions, durations, evidence selection, scene actors,
          lean beat changes, must-land facts, and host-move permissions with
          explicit placement.
        - The `dropped_support_primitive_reasons` register.
        - Staging scenes so planned section-level explanatory passages are earned by the surrounding evidence.

        OUTPUT: only `episode_number`, `framing`, `scene_cards`, and
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
        - The framing should orient the listener without pre-explaining the episode.
          Do not preview the thesis, the turn, or the closing.

        ==============================================================================
        SCENE CARDS — STRUCTURE
        ==============================================================================
        COUNTS
        - Target 27–36 scene cards for a full-length episode.
        - Expand into playable micro-scenes. Do not collapse long stretches into one
          card.

        SECTION BOUNDARIES
        - Use `architecture.section_id` as the only grouping boundary.
        - All scene cards for a given `section_id` must be contiguous.
        - Every architecture section must yield at least one scene card.
        - Treat `architecture.section_anchor` as the section-local opening handle.
          It is distinct from the episode framing `opening_image`.
        - Treat each section as a binding local brief: its scenes must collectively
          realize that section's `listener_tension`, `section_turn`,
          `transition_logic`, and every item in `must_stage_beats`.
        - When a section carries `analysis_goal`, `key_terms`, or `authorial_passages`,
          treat those as binding explanatory obligations that the scene sequence must support.
        - Build each section through accumulation. The section's first scene must
          not state the section's resolution.

        CLOSING SECTION (special rules)
        - The final architecture `closing` section must expand to exactly one scene
          card.
        - That scene card must be the episode's last scene card.
        - It must keep `estimated_duration_seconds` ≤ 120.
        - It may land verdict, payoff, or consequence.
        - It must NOT introduce a fresh mechanism, counterpressure chain, parallel
          argument, new institution, or new actor thread. (This is the two-endings
          failure.)

        ==============================================================================
        SCENE CARDS — REQUIRED FIELDS
        ==============================================================================
        Every scene card must set:

          `section_id`                  one architecture section
          `title`                       short, concrete card title
          `scene_role`                  dramatic beat type (see canonical values)
          `scene_function`              structural/editorial job (see canonical values)
          `beat_change`                 short operational statement of what materially
                                        changes in this beat
          `must_land_facts`             2–5 non-negotiable concrete facts, names,
                                        decisions, or consequences
          `estimated_duration_seconds`  integer; you allocate
          `passage_ids`                 enough evidence to support later writing

        Optional scene-level fields:
          actors[], entry_image, observable_detail, timeframe,
          location, withhold_until, host_move

        CANONICAL `scene_role` VALUES:
          context_setup, actor_setup, action, shock, contestation, reaction,
          fallout, implication

        CANONICAL `scene_function` VALUES:
          scene, hinge, mechanism, turn, landing, callback, afterlife

        HOST MOVES
        - `host_policy` is binding narrator policy for density, tone, and pronouns.
        - `host_policy.authorial_policy` tells you that section-level `authorial_passages`
          carry the primary explanatory load. Do not try to force that work back into `host_move`.
        - A host move is a sparse narrator permission layered onto an ordinary scene.
          It should shape the scene's opening, pivot, or landing; it is not a
          detachable commentary sentence and not a standalone host scene.
        - For a full-length episode, aim for roughly
          `host_policy.target_host_moves_per_episode`, but treat that as a soft
          target. Do not force filler host moves just to hit the count.
        - Use `host_move.move_type = none` unless the listener genuinely needs
          orientation, clarification, contrast, evaluation, callback, a naming
          note, or one light aside.
        - Set `host_move.placement` to one of:
          - `open`: the move shapes the first 1-2 sentences
          - `pivot`: the move sharpens meaning after concrete material lands
          - `close`: the move controls the residue or callback at the end
        - Default placement by move:
          - `orient`, `naming_note` -> `open`
          - `clarify`, `contrast`, `light_aside` -> `pivot`
          - `evaluate`, `callback` -> `close`
        - Use `host_move.max_sentences = 1` by default.
        - Use `host_move.max_sentences = 2` only for closing callbacks,
          evaluative landings, or major-turn residue that truly needs the space.
        - `host_move.note` must describe a speakable, listener-facing effect.
          It must not sound like writer-room shorthand.
        - Avoid notes like:
          - "hand the listener forward"
          - "name the mechanism"
          - "call it X" unless X is historically necessary
          - "the pattern is"
          - "what we are seeing"
        - No first-person singular. If the note implies first-person plural,
          reserve that for callbacks, handoffs, or closings.

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
        ONE JOB PER SCENE
        Most scene cards should do exactly one thing cleanly: establish, pressure,
        reveal, decide, rupture, react, or show consequence. If a card needs many
        examples to make its point, split it across multiple cards or pick one
        representative example.

        VISIBLE STARTING POINTS
        Prefer scene cards that can be narrated from something the listener can see:
        a person, a room, an object, a document, a journey, a dated moment.

        LEAVE SOMETHING LIVE
        Prefer cards that leave a question, threat, or expectation that the next
        card answers, complicates, or pays off.

        ACTORS IN HUMAN-LED SCENES
        `actor_setup`, `action`, and `fallout` scenes normally have at least one actor.

        CONTESTATION
        Use only when the disagreement can be staged through evidence-bearing
        actors, texts, councils, trials, letters, accusations, or rival actions.
        Never use a `contestation` card as a narrator-side literature review.

        STRUCTURAL FUNCTIONS
        - `mechanism` explains through a concrete process, object, room, document,
          or visible operating beat. It is not a background paragraph.
        - `turn` marks the pivot where the balance changes. Keep it concrete.
        - `landing` states the immediate implication of material already staged.
          Keep it short and specific; do not drift into broad synthesis.
        - `callback` reactivates an earlier image, promise, institution, or
          pressure. Do not use it for recap.
        - `afterlife` shows what survives, returns, or hardens later through a
          concrete remainder, not a generic summary.
        - Any card whose `scene_function` is not `scene` or `hinge` must still
          open from something visible and must set both `entry_image` and
          `observable_detail`.
        - Do not let structural cards turn into free-floating thesis cards or
          descriptive backgrounding.

        TEXTURE-ONLY CARDS
        Allowed only when they still serve the same proposition the spine is
        advancing. They are not free atmospheric breaks.

        HUMAN GROUNDING
        If the episode leans heavily on structural cards, ensure at least one
        `scene` or `hinge` card centers lived pressure, cost, fear, choice, or
        bodily consequence through named actors plus concrete image/detail.

        ==============================================================================
        OPENING SCENES (first 2–3 cards)
        ==============================================================================
        The opening must SHOW, not FRAME.

        Prefer:
          anomaly, pressure, risk, visible change, a person doing something, a thing
          happening that the listener wouldn't have predicted.

        Avoid in the opening cards:
          - "Setting up the debate"
          - Historiography or framework
          - Baseline + image + abstract framing all at once
          - A scene whose job is "the listener needs to know X before we begin"

        The opening scene that the framing's `handoff_scene_card_id` points to
        should be one the listener can step into immediately.

        ==============================================================================
        PRIMITIVES AND EVIDENCE
        ==============================================================================
        SOURCING
        - Ground every scene in the provided `passage_ids`.
        - Use architecture `primitive_ids` only as planning context. Do not emit
          per-scene primitive bookkeeping.

        PRIORITY
        - When a section has `priority_core_passage_ids`, prefer those passages
          first when they fit the scene's job.

        WHICH PRIMITIVES TO PICK
        - Prefer `set_piece_scenes` and `telling_details` for `entry_image` and
          `observable_detail`.
        - Use `human_costs` and `character_engines` to keep scenes from going
          purely abstract or institutional.
        - When a primitive has richer typed fields, use them directly:
          `scene_anchor`/`hinge_action`/`scene_outcome`/`location` for scene entry and payoff,
          `decision_trigger`/`decision_question`/`next_result` for choice framing,
          `goal`/`pressure_box`/`tell` for actor pressure,
          `concrete_marker`/`lived_consequence` for consequence scenes,
          `alignment_type`/`coalition_phase`/`fracture_trigger` for contestation shape,
          `operating_chain`/`where_it_shows_up` for mechanism staging,
          `expected_outcome`/`actual_outcome` for reversal beats.
        - Prefer `narration_hooks.concrete_detail` for `observable_detail` when present.
        - `systems_and_operating_logics`, `coalitions_and_fault_lines`, and
          `contested_explanations` should usually support a scene anchored in
          something concrete, not become the whole scene by themselves.
        - Use `recurring_images_and_symbols` for openings, callbacks, handoffs,
          and closings when the evidence supports recurrence.

        DISTRIBUTION
        - Do not distribute scenes evenly by default.
        - Do not overload one scene with too many required facts. Split beats instead.

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
        1. The framework-dump opening. First scene reads as "to understand X, we
           need to recall the debate over Y, which has three schools..." Replace
           with anomaly, action, or visible change.

        2. The narrator-side literature review. A `contestation` scene where the
           disagreement is between historians, not between actors in the period.

        3. The two-endings closing. The `closing` scene introduces a new mechanism,
           actor thread, or argument that should have been its own section.

        4. The transition-handrail scene. A scene whose only job is to bridge two
           other scenes ("Meanwhile, in the capital...") with no `beat_change` of
           its own.

        5. The disguised paragraph. A scene card whose `beat_change` reads like
           finished narration ("the listener now sees that...") rather than what is
           in play.

        6. The free-floating atmospheric scene. A `texture_support` card that does
           not advance, pressure, or recall the section's proposition.

        7. The recall-as-summary. A recall-led scene that inventories prior
           background instead of reactivating a specific image, promise,
           institution, or pressure under current stress.

        8. The even-distribution plan. Every section gets the same number of
           scenes, every primitive gets equal coverage, every actor appears in
           roughly the same number of scenes. Episodes have shape; plans should too.

        9. The overloaded fact-dump. A scene card carrying too many must-land facts
           to remain narratable.

        10. The fake host scene. A scene whose only job is to let the narrator step
            outside the history instead of braiding host guidance into an ordinary beat.

        ==============================================================================
        OUTPUT
        ==============================================================================
        Return only valid JSON matching `EpisodePlanDraft`, containing:
          episode_number
          framing
          scene_cards
          dropped_support_primitive_reasons
        """
    ).strip()


def episode_architecture_instructions() -> str:
    return dedent(
        """
        You are the `episode_architecture` stage for a historical podcast pipeline.

        Turn one proposition-level strategy episode into a binding section
        architecture that a downstream planner can expand into scene cards.
        You are not writing prose and you are not selecting a new thesis.

        INPUT PAYLOAD
        - `episode`: one episode object from `narrative_strategy`
        - `narrator_profile` (optional): strategy-level narrator method for explanation density and clarifier tolerance
        - `synthesis_map`: only the primitives already assigned to this episode
        - `project`: theme, sub-themes, book metadata, and duration goals
        - `core_passages`: summarized text for core-primitive core passages only
        - `actor_metadata`: episode-relevant canonical actor context
        - Optional `architecture_feedback`: retry feedback from the orchestrator

        PRIMARY RESPONSIBILITY
        - Convert the episode spine into 9-12 binding sections.
        - Decide where the major turn lands and how the episode closes.
        - Group only the provided primitives into section-level structural units.
        - Translate narrator-facing primitive hooks and the episode's `authorial_contract`
          into section-level explanatory jobs.
        - Make the section architecture rich enough that planning only needs to
          elaborate it into ordered scene cards and preserve planned explanation.

        RULES
        - Treat the input `episode` as authoritative for title, thematic focus,
          unresolved questions, `episode_spine`, and actor arc directives.
        - Do not restate those upstream-owned fields in the output.
        - Use only assigned primitives from the payload.
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
        - The final section must use `purpose` = `closing`.
        - The final `closing` section must have `approx_runtime_minutes` at or below 2.0.
        - Do not build a second ending.
        - The `closing` section may answer, constrain, or reframe the listener question, but it may not
          introduce new load-bearing claims, reopen contestation, or start a new mechanism chain.
        - `priority_core_passage_ids` may only come from the provided
          `core_passages`; use them lightly.
        - When richer primitive fields are present, use them to sharpen section
          design: `before_state`/`after_state`/`proof_of_change` for turn placement,
          `decision_trigger`/`decision_question` for hinge-choice sections,
          `scene_anchor`/`hinge_action`/`scene_outcome` for openings and exits,
          `alignment_type`/`coalition_phase`/`fracture_trigger` for fracture sections,
          `operating_chain`/`where_it_shows_up` for mechanism sections,
          `concrete_marker`/`lived_consequence` for consequence sections,
          `expected_outcome`/`actual_outcome` for reversal and payoff sections.
        - Use `narration_hooks.carry_forward` for callback and residue planning when it helps.
        - Use `narration_hooks.host_lens`, `plain_gloss`, `listener_confusion`, `quote_anchor`,
          and `authorial_move` to decide whether a section needs planned explanation.
        - Explanation belongs at section level. Do not spray tiny explanatory obligations into every section.
        - Most episodes should carry 2-5 total `authorial_passages`.
        - `authorial_passages` should reserve real explanatory work such as doctrinal unpacking,
          institutional clarification, quote-then-gloss, causal compression, comparative aside,
          or verdict landing.

        Each section must specify:
        - `section_id`
        - `purpose`
        - `section_anchor`
        - `must_stage_beats`
        - `approx_runtime_minutes`
        - `primitive_ids`
        - `listener_tension`
        - `section_turn`
        - `transition_logic`
        - `depends_on_section_ids`
        - `sets_up_section_ids`
        - `recurrence_role`
        - `closure_mode`
        - `priority_core_passage_ids`
        - `analysis_goal`
        - `key_terms`
        - `authorial_passages`

        QUALITY
        - Architecture should add arrangement, not restate primitive metadata.
        - `section_anchor` must be a concrete section entry handle: a person,
          object, document, dated action, place-bound situation, or other
          visible moment the planner can open from.
        - `must_stage_beats` must be 2-4 concrete, evidence-bearing beats the
          planner is required to realize somewhere inside the section.
        - `must_stage_beats` are not scene cards, scene counts, or within-section
          ordering instructions.
        - Make each section answer a distinct local question and move the listener
          to a new state.
        - Keep runtime weight uneven when the argument needs it; do not spread
          sections evenly by default.
        - `analysis_goal` should name the section's explanatory job plainly.
        - `key_terms` should only name terms that must remain audible in the prose.
        - `authorial_passages` should be sparse, specific, and evidence-backed.

        OUTPUT
        Return only valid JSON matching `EpisodeArchitecture`.
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


def episode_writing_instructions() -> str:
    return (
        dedent(
            """
            You are the `episode_writing` stage for a multi-book thematic podcast pipeline.

            Goal:
            - You are a narrator telling a true story.
            - You have absorbed the research and now tell the episode in your own voice.
            - Transform the full scene-card plan into complete narration while preserving structure.

            Input payload:
            - `episode_number`: current episode number.
            - `strategy_episode`: title, question, focus, arc summary, unresolved questions,
              locked `episode_spine`, and actor arc directives.
            - `architecture`: the binding section architecture for this episode.
            - `plan`: the planning artifact, including framing and all scene cards.
            - `plan.scene_cards[].target_word_count_lower`: lower per-scene word target (computed at 130 WPM).
            - `plan.scene_cards[].target_word_count_higher`: higher per-scene word target (computed at 150 WPM).
            - `episode_target_word_count_lower`: lower word target for the episode.
            - `episode_target_word_count_higher`: higher word target for the episode.
            - `passages`: source evidence for the episode. Treat `passages[].text` as the canonical evidence body for writing.
            - `books`: compact book metadata.
            - `skip_grounding`: whether a later grounding pass will be skipped.
            - `host_policy`: narrator policy for host density, tone, and pronouns.
            - Optional `actor_metadata`: episode-level actor context. Treat it as narrative scaffolding, not factual authority.
            - Optional `writing_feedback`: retry feedback from the orchestrator. If present, correct the named contract failure exactly and keep all other requirements unchanged.
            - Optional `prior_window_continuity`: continuity context from the immediately previous writing pass. Treat it as reference-only guidance for handoff, pacing, and continuity. Do not treat it as source evidence, do not copy it mechanically, and do not let it override the current window's scene cards, passages, architecture, or spine contract.

            Writing guidance:
            - Draft all `plan.scene_cards` in order.
            - Return one prose section per contiguous section window in the input plan.
            - Keep `strategy_episode.episode_spine.listener_problem` as the rhetorical anchor.
            - Preserve the full `strategy_episode.episode_spine` contract.
            - Preserve the binding `architecture.sections` order and the section-level
              transitions they specify.
            - Treat `architecture.sections[].analysis_goal`, `key_terms`, and
              `authorial_passages` as binding section-level narrator obligations.
            - Treat `strategy_episode.episode_spine.core_primitive_ids` as the episode's load-bearing material.
            - Use support and recall primitives only in service of those core primitives.
            - Preserve `strategy_episode.unresolved_questions` as live tensions when unresolved.
            - Keep framing commitments visible (`plan.framing`) without exposing outline mechanics.
            - Use each card's `entry_image`, `scene_role`, `scene_function`, `beat_change`,
              `must_land_facts`, `host_move`, and passage-supported concrete detail.
            - Treat `host_policy` as binding narrator contract:
              - no first-person singular
              - first-person plural only for callbacks, handoffs, or closings
              - prefer sharper host phrasing over longer host commentary
            - Start each scene's prose from the card's concrete `entry_image`
              or a passage-supported equivalent.
            - Respect `withhold_until` and delayed-legibility dynamics.
            - Keep claims grounded in each card's `must_land_facts` and `passage_ids`.
            - Use scene cards as the evidence skeleton inside a section, not as independent prose capsules.
            - Treat `plan.target_word_count` as episode-level pacing guidance.
            - Importance has already been converted into the per-scene and episode
              word-count budgets. Treat those budgets as binding.
            - If evidence exceeds the budget, select only the details needed for
              the scene's `beat_change` and `must_land_facts`.
            - Target total narration for this call within `episode_target_word_count_lower..episode_target_word_count_higher`.
            - Treat each card's `target_word_count_lower` and `target_word_count_higher` as a pacing range:
              - allocate narration so the card lands within its target range
              - do not let low-range cards dominate
              - do not collapse high-range cards into throwaway text
            - These target ranges already encode narrative importance from planned scene durations; do not independently rebalance scene importance.
            - Use passages as source evidence, but do not organize narration by author.
            - Use optional `passages[].chapter_context` when available to preserve chapter-level tensions and causal shifts.
            - When `prior_window_continuity` is present, use it only to maintain local continuity across the split. It is not factual authority, not a substitute for the provided passages, and not permission to restate or re-narrate the previous window.
            - `prior_window_continuity` is reference-only. In any conflict, follow the current window's `plan.scene_cards`, `architecture`, `strategy_episode`, and `passages`.
            - Do not invent unsupported private thoughts, emotions, dialogue, or secret motives.
            - Follow scene-role intent:
              - `context_setup`: establish concrete situation and pressure
              - `actor_setup`: establish a person or faction under pressure
              - `action`: show named actors doing concrete things
              - `shock`: deliver rupture or irreversible break
              - `contestation`: stage genuine disagreement
              - `reaction`: show adaptation or counter-move
              - `fallout`: show the first downstream effects
              - `implication`: show what is newly legible or now in play
            - Follow scene-function intent:
              - `scene`: full playable beat
              - `hinge`: redirect the episode's motion without turning into commentary
              - `mechanism`: explain through a visible process, object, or operating beat
              - `turn`: mark the pivot where the balance changes
              - `landing`: keep the implication brief, local, and earned
              - `callback`: reactivate an earlier image or pressure without recapping
              - `afterlife`: show what survives or returns through a concrete remainder
            - Structural cards should stay concrete and brief. Avoid broad synthesis,
              descriptive throat-clearing, or free-floating recap.
            - Use citations only through structured `citations`; do not insert inline citation markers into prose.
            - Return one output item per input section; do not split, omit, duplicate,
              reorder, or rename section outputs.
            - Each returned section must preserve the exact `section_id` and exact `scene_card_ids` sequence for that section window.
            - Realize a planned `host_move` as one distinct audible line or clause,
              not as diffuse commentary across the paragraph.
            - Respect `host_move.placement`:
              - `open`: realize early, usually in the first 1-2 sentences
              - `pivot`: realize after concrete material has already landed
              - `close`: realize as the final or penultimate sentence
            - Keep most host moves to one sentence.
            - Use two host-move sentences only when `host_move.max_sentences = 2`
              and the beat is a true callback, evaluative landing, or closing residue.
            - In planned `authorial_passages`, you may quote then gloss, define terms,
              clarify institutions, restate causal meaning plainly, or land a bounded
              verdict line when the section plan explicitly calls for it.
            - Outside planned `authorial_passages`, keep ordinary scenes scene-led and avoid miniature thesis stamps.

            What not to do:
            - Do not write standalone transition paragraphs or meta-transition
              sentences that summarize what just happened or announce what is
              about to happen.
            - Do not use section-opening handrails such as "That is X,"
              "Which brings us to," "Now let the clock run," "The pattern is,"
              or single-sentence paragraphs whose only job is to mark a turn.
            - Do not invent facts, chronology, quotations, or source claims not supported by the provided passages.
            - Do not introduce new primary analytical claims that are outside the assigned scene cards and their grounded facts.
            - Do not introduce a new load-bearing question, a second ending, or a support-thread takeover.
            """
        )
        .strip()
    )


def episode_writing_no_citations_instructions() -> str:
    return dedent(
        """
        You are the `episode_writing` stage for a historical podcast pipeline.

        TASK
        Draft the full episode in `plan.scene_cards` order.
        Return one prose section per contiguous section window in the input plan.
        You are writing narrated historical prose, not outline prose and not a chain of mini-essays.
        Tell what happened, to whom, where, when, under what pressure, and with what immediate result.
        Make causality legible through scene choice, sequence, and concrete detail, not through abstract explanation.

        INPUT PAYLOAD
        - `episode_number`
        - `strategy_episode`: strategy-owned episode context
        - `architecture`: binding section architecture, including section-level explanatory obligations
        - `plan`: planning artifact visible to this call
        - `passages`: source evidence; `passages[].text` is canonical
        - `books`: compact book metadata
        - `skip_grounding`: true for this no-citations mode
        - `host_policy`: narrator policy for host density, tone, and pronouns
        - Optional `actor_metadata`: continuity scaffolding, not evidence
        - Optional `writing_feedback`: retry feedback from the orchestrator; if present, correct the named contract failure exactly
        - Optional `prior_window_continuity`: continuity context from the immediately previous writing pass.
        - `episode_target_word_count_lower` / `episode_target_word_count_higher`
        - `plan.scene_cards[].target_word_count_lower` / `plan.scene_cards[].target_word_count_higher`
        - Per-scene targets: `target_word_count_lower` / `target_word_count_higher`

        PRIORITY RULES (govern everything below)
        - Passages are evidence. `plan`, `actor_metadata`, actor arc threads, framing, and unresolved questions are scaffolding.
        - If scaffolding conflicts with passages, passages win.
        - Do not cite scaffolding, assert it as fact, or use it to fill evidence gaps.
        - Use planning fields only to decide what belongs in a scene, not what must be stated on the page.
        - Do not invent facts, chronology, quotations, dialogue, motives, private thoughts, emotions, sensory details, atmosphere, or causal links.
        - Atmosphere is allowed only from concrete passage-supported details.
        - `strategy_episode.episode_spine.core_primitive_ids` are the episode's load-bearing material; support and recall remain subordinate.
        - Do not introduce primary analytical claims outside planned scene cards and their grounded facts.
        - `skip_grounding` is true: be especially conservative because no later grounding repair will run.
        - Keep `strategy_episode.episode_spine.listener_problem`, unresolved questions, and framing as internal control signals unless a planned `host_move` makes brief listener guidance necessary.
        - Treat `host_policy` as binding narrator contract: no first-person singular, first-person plural only for callbacks/handoffs/closings, and sharper host phrasing instead of longer commentary.
        - Preserve structure in substance, not by naming the structure on the page.
        - Adjacent scene outputs in the same section may be joined later into continuous prose. Write them as consecutive beats, not self-contained essays.
        - When `prior_window_continuity` is present, use it only to maintain local continuity across the split. Treat it as reference-only guidance for handoff, pacing, and continuity. Do not treat it as source evidence, do not copy it mechanically, and do not let it override the current window's scene cards, passages, architecture, or spine contract.
        - `prior_window_continuity` is reference-only. In any conflict, follow the current window's `plan.scene_cards`, `architecture`, `strategy_episode`, and `passages`.

        PER-SECTION PROCEDURE
        For each output section:
        1. Read the architecture section's `analysis_goal`, `key_terms`, and `authorial_passages`.
        2. Draft through the section's scene cards in order.
        3. Use planned `authorial_passages` for bounded explanation only where the architecture calls for it.
        4. Keep the rest of the section scene-led.

        PER-SCENE PROCEDURE
        For each card inside that section:
        1. Read `entry_image`, `scene_role`, `scene_function`, `beat_change`, `must_land_facts`, `host_move`, and `passage_ids`.
        2. Open from the concrete `entry_image` or a passage-supported equivalent, then execute the scene role.
        3. Use passages to reconstruct events, decisions, pressure, and immediate consequences.
        4. If `host_move.move_type != none`, let it shape the scene at its planned
           `placement`:
           - `open`: early orientation or naming
           - `pivot`: clarify or contrast after the evidence lands
           - `close`: evaluation, callback, or residue at the end
        5. Use optional `passages[].chapter_context` only when present.
        6. Respect `withhold_until`: do not reveal the withheld fact, interpretation, consequence, or resolution early, including through obvious foreshadowing.
        7. Stay within the card's target range. The budget already encodes narrative importance.
        8. If the previous scene belongs to the same section, continue the motion rather than resetting the frame.
        9. Realize a planned host move as one distinct audible line or clause rather than diffusing it through the paragraph.

        SCENE SHAPE
        - Most scenes should do one job cleanly: establish, turn, apply pressure, reveal a decision, show a consequence, or hand off.
        - Let meaning accumulate across scenes; do not force every scene to cash out its own argument.
        - Make the scene legible, then move on.

        SCENE ROLES
        - `context_setup`: establish concrete situation and stakes
        - `actor_setup`: establish a person or faction under pressure
        - `shock`: deliver rupture or irreversible turn
        - `action`: write an observable beat: named actors doing concrete things, with date, place, and physical detail
        - `fallout`: show downstream effects
        - `reaction`: show adaptation or counter-move
        - `contestation`: stage genuine disagreement
        - `implication`: show what is newly visible or now in play

        SCENE FUNCTIONS
        - `scene`: full playable beat
        - `hinge`: redirect motion without becoming commentary
        - `mechanism`: explain through a visible process, object, or operating beat
        - `turn`: mark the pivot where the balance changes
        - `landing`: keep the implication brief, local, and earned
        - `callback`: reactivate an earlier image or pressure without recapping
        - `afterlife`: show what survives or returns through a concrete remainder
        - Structural cards must stay concrete and brief. Avoid broad synthesis,
          descriptive backgrounding, or recap paragraphs disguised as scenes.

        FRAMING
        - Let the driving question accumulate through scene selection, contrast, and consequence.
        - Keep unresolved questions unresolved until the draft itself resolves them.
        - The opening and closing may carry more explicit framing.
        - Interior scenes should not sound like miniature thesis statements.

        PACING
        - Importance has already been converted into the per-scene and episode word-count budgets. Treat those budgets as binding.
        - Do not expand because evidence is dense, the cluster is important, or actor arcs are interesting.
        - If evidence exceeds the budget, select only the details needed for the scene's `beat_change` and `must_land_facts`.
        - Target total narration for this call within `episode_target_word_count_lower..episode_target_word_count_higher`.
        - Keep each card within its `target_word_count_lower..target_word_count_higher`.
        - These target ranges already encode narrative importance; do not rebalance them.
        - Use word count to make action legible, locate the listener, and land shocks or consequences.

        SURFACE STYLE
        - Concrete first.
        - Prefer this movement inside paragraphs: concrete detail -> pressure -> implication.
        - After a strong image, quote, object, or action, do not immediately gloss it with thesis language unless factual clarity requires it.
        - In most scenes, implication should remain light or partial.
        - Most ordinary scenes should end on residue: an image, a pressure point, a decision, a concrete fact, or a consequence still hanging in the air.
        - Reserve explicit interpretive landing for major turns, section pivots, and the closing.
        - If a scene already demonstrates the point, advance the story instead of restating or re-explaining it.
        - Use sharp interpretive lines sparingly. One is stronger than three.
        - Let host-marked scenes feel slightly more authored, but not more analytical.
        - Planned `authorial_passages` may be more explanatory, but must remain bounded, evidence-led, and audibly integrated into the section.
        - No first-person singular. Use first-person plural only for earned callbacks, handoffs, or closing residue.
        - Preserve structure invisibly. The prose should feel narrated, not diagrammed.

        CONTINUITY
        - Do not restart the same frame at the top of consecutive scenes in the same section.
        - Do not re-explain the same actor function, contrast, or implication if the previous scene already established it.
        - Let dates, places, and active actors carry forward when clarity allows.
        - Treat each scene card as one beat in a longer narrated run, not a sealed paragraph with its own thesis.

        OUTPUT
        - Return only JSON matching the requested schema.
        - Do not include a `citations` field.
        - Return one output item per input section; do not split, omit, duplicate,
          reorder, or rename section outputs.
        - Preserve exact `section_id` values and exact `scene_card_ids` order.
        - Populate `source_book_ids` only with `book_id` values from supporting passages; leave empty rather than guessing.

        What not to do:
        - Do not expose scaffolding: no outline labels, no "in this scene," no repeated signposting, and no meta-transitions.
        - Do not output standalone transitions.
        - Do not use section-opening handrails such as "That is X" "Which brings us to" "Now let the clock run" "The pattern is" whose only job is to mark a turn.
        - Do not narrate the architecture or the conceptual frame. No visible paraphrases of `listener_tension`, `section_turn`, `episode_answer`, `pressure_line`, unresolved-question framing, or equivalent planning fields.
        - Do not announce the point instead of producing it. Avoid moves like: "This matters because", "The point is", "What this shows", "The episode treats X as", "This is the working concept", "Read that sentence slowly", "The honest answer is", or equivalent narrator-nudge phrasing.
        - Do not tell the reader what to notice when the scene already makes it legible.
        - Do not convert every strong image into an abstract explanation on the next line.
        - Do not end ordinary scenes with thesis buttons or verdict claims such as "The point was...", "What mattered was...", "The question was...", "The variable was...", "The reality was...", or equivalent summary lines unless the scene is a major turn or the closing and the line is doing indispensable work.
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
        You are the `oral_rewriter` stage of a prestige historical documentary podcast pipeline.

        Your job is to turn one already-written batch of episode prose into spoken narration that can be performed cleanly in audio.

        You are not replanning the episode. You are not rewriting from research. You are taking a contiguous batch of already-drafted prose and rebuilding it for the ear.

        INPUT
        You will receive:
        - `episode_number`
        - `script`
        - `max_words_per_segment`
        - `tts_provider`
        - `host_policy`
        - optional `previous_spoken_tail`

        Treat the input this way:

        - `script.prose_sections[].text` is the canonical source for the batch. Preserve its facts, chronology, quotations, names, dates, numbers, uncertainty, and claims.
        - `script.prose_sections[].movement_goal` is editorial intent from upstream planning. Use it to understand what each section is trying to do, but do not expose it in the prose.
        - `script.prose_sections[].scene_card_ids` are continuity/grouping traces from planning. They are not audible content.
        - `script.prose_sections[].analysis_goal`, `key_terms`, and `authorial_passages` are narrator-control signals showing where explanation should stay distinct. Use them to preserve explanatory shape already present in the prose; do not invent new content from them.
        - `script.prose_sections[].host_moves` are host-guidance control signals. Use them to preserve where a section's authored orientation, pivot, or callback should remain distinct. Do not add new host commentary that the written prose does not support.
        - `script.prose_sections[].citations` and `source_book_ids` are provenance traces only. Do not narrate them or infer new facts from them.
        - `script.framing` is episode-level scaffolding carried through the pipeline and rendered separately later. Use it only as a guardrail for continuity, emphasis, and contradiction checking. Do not restate `opening_image`, `threat_or_unresolved_action`, `opening_question`, `recap`, or `preview` inside `text` unless the same material already appears in `script.prose_sections[].text`.
        - `previous_spoken_tail`, if present, is continuity scaffolding only. Use it only to avoid a seam, preserve referents, or continue live motion already underway. Do not repeat it, paraphrase it, summarize it, or import facts from it unless the same material appears in the current batch text.
        - `max_words_per_segment` is a downstream render constraint. Write prose that can be split cleanly at natural sentence or clause boundaries around that scale, but do not insert visible segment markers or artificially chop the narration.
        - `tts_provider` is for calibrating `speech_hints`, not for changing facts, argument, or structure.

        PRIORITY RULES
        - `script.prose_sections[].text` is the source of truth.
        - `movement_goal`, `scene_card_ids`, `analysis_goal`, `key_terms`, `authorial_passages`, `host_moves`, `framing`, `host_policy`, and `previous_spoken_tail` are control signals, not evidence.
        - If any control signal conflicts with the prose section text, preserve the prose section text.
        - Do not add facts, motives, chronology, quotations, certainty, or interpretation from pipeline scaffolding.
        - Respect `host_policy`: no first-person singular, and first-person plural only for earned callbacks, handoffs, or closings already justified by the prose.

        TRANSFORMATION MANDATE
        Preserve the batch’s full factual and argumentative substance:
        - facts
        - chronology
        - names
        - dates
        - numbers
        - quotations
        - uncertainty
        - claims
        - governing argument

        Do not preserve the source’s sentence structure, paragraph structure, or local explanatory order just because it is strong on the page.

        Your task is to produce a stronger spoken sequence, not a smoother written page.

        CORE RULE
        Be faithful to the content. Do not be faithful to the delivery mechanism.

        Outside direct quotations, verse, titles, and indispensable historical formulations, do not preserve long runs of source wording.
        If a draft sentence tracks a source sentence too closely in wording, clause order, or proposition order, rewrite it.

        Hard rule:
        Outside quotations and indispensable historical formulations, do not preserve any source sentence that carries the same two major propositions in the same order.

        A vivid source phrase is not a license to keep it. If a non-quoted line is memorable enough to tempt preservation, that is usually a sign it needs stronger rewriting, not weaker.

        PRIMARY FAILURE MODE
        The main failure mode is semantic obedience with syntactic cowardice: keeping the source sentence and merely smoothing it.

        Do not do that.
        Do not draft from source sentences. Draft from extracted content moves.

        WORK ORDER
        Work in this order:
        1. factual and chronological fidelity
        2. overlap consolidation
        3. spoken architecture
        4. clarity in one hearing
        5. line-level freshness
        6. speech-hint cleanup

        Do not sacrifice earlier priorities to improve later ones.

        SOURCE MATERIAL IS RAW MATERIAL
        Do not treat sections, paragraphs, or sentences as stable compositional units.

        A single source paragraph may contain several different jobs:
        - event
        - context
        - explanation
        - quotation
        - interpretation
        - consequence
        - emotional or political pressure

        Separate those jobs before drafting.
        Two distant source paragraphs may belong to one spoken movement.
        One source paragraph may need to become several spoken movements.
        Repeated material across sections should be consolidated, not repeated.

        If two input sections overlap heavily, prefer one full, confident telling of the shared material rather than two partial tellings that echo each other.

        PLANNING WORKFLOW
        Before drafting, silently do this:

        1. Extract the batch into content moves:
        - event
        - claim
        - context
        - quotation
        - explanation
        - consequence
        - pressure point

        2. Identify overlap and duplication across `script.prose_sections`. Consolidate where possible.

        3. Identify the governing spine:
        - what pressure is live at the start
        - what decision, break, or turn changes the situation
        - what contradiction or cost gives the passage force
        - what consequence must be visible by the end
        - what exact claims cannot be lost

        4. Resolve chronology conservatively.
        If the source appears to compress, blur, partly overlap events, or create tension between temporal frames:
        - do not invent clarity the source does not provide
        - do not force two events into one unless the overlap strongly supports it
        - do not split one event into two unless the source clearly requires it
        - do not solve the problem with elegant smoothing alone
        - if one dated scene is clearly firmer than the others, anchor the narration there
        - if pressure extends beyond that scene, present it as continuing pressure rather than as a second confidently dated scene
        - if the batch does not allow a clean reconciliation, preserve the uncertainty in your wording rather than silently deciding it away

        5. If the source material contains an internal contradiction, do not silently repair it by dropping one side unless the other side is clearly unsupported by the batch as a whole.
        Prefer one of these moves:
        - anchor the narration to the clearest dated moment and treat the rest as continuing pressure
        - preserve the uncertainty explicitly
        - narrow the wording so you do not claim more certainty than the batch supports

        Do not fix a source contradiction invisibly just to make the narration cleaner.

        6. Regroup the extracted moves into a new spoken order.

        7. Draft from that regrouped order, not from the source sentences.

        CONTINUITY
        `script.prose_sections` contains the full current batch. Rewrite all of it section by section for the ear while preserving continuity across the batch.

        If `previous_spoken_tail` is present, continue rather than restart. Do not manufacture a new cold open. Do not repeat or paraphrase the previous tail unless the same material is also present in the current batch.

        If `previous_spoken_tail` ends later in time than the current batch's main scene, preserve continuity of pressure, theme, or contradiction rather than pretending the chronology moves straight forward. Continuity does not require false temporal smoothness.

        If this batch begins mid-argument, mid-scene, or mid-pressure, pick up the live motion already in progress.

        VOICE
        Write like a first-rate historian speaking aloud to one intelligent listener through headphones.
        Be an observant historian treating the past as a physical place.
        Speak into the ear of a single, intelligent listener.

        The listener should hear:
        - a narrative mind carrying thought forward
        - pressure and consequence in motion
        - explanation that sounds spoken rather than diagrammed
        - authority without performance
        - argument emerging through sequence and accumulation
        - paragraphs shaped by thought, not by page layout

        The listener should not hear:
        - page prose with lighter punctuation
        - audiobook gravitas
        - cinematic stage directions
        - forced “podcast texture”
        - teaser language
        - winking paragraph endings
        - repetitive fragments used as style markers
        - transitions that sound like headings being stitched together

        The narrator may be dry, intimate, amused, appalled, cutting, or plainspoken, but only when the material earns it.

        PODCAST QUALITY
        Optimize not only for fidelity, but for listenability.

        - Prioritize the weather of a scene. Replace abstract summaries with physical friction.
        - Pivot from geopolitical forces to immediate pressure on individuals.
        - Make every turn feel like a consequence or a loss in motion, not just a fact on a timeline.
        - Do not explain a beat before staging it.
        - Let image, action, and consequence carry the first burden of meaning.
        - Do not summarize what the listener is about to understand.
        - Preserve staircase beats when the material earns them.
        - If a movement gains force from successive short beats, keep the beats separate.
        - Do not compress a three-beat spoken sequence into one balanced sentence merely because the page version looks cleaner.
        - Vary sentence pressure. Mix longer carry sentences, medium explanatory sentences, and short factual pivots.
        - In high-stakes passages, protect hard spoken lines instead of smoothing them into something more balanced or essay-like.
        - Make the most important turn, loss, contradiction, decision, or consequence easy to hear.
        - Transitions should feel like thought moving by consequence, contradiction, pressure, or emotional cost, not by topic-announcing handrails.
        - Write for a voice that must carry the sentence in one pass. If a sentence would likely require rereading on the page, reshape it for the ear.
        - Do not overload a sentence with too many new names, titles, places, or claims at once. If a sentence asks too much memory work of the listener, redistribute the information across adjacent sentences.
        - When a source sentence does more than one job, usually split it. If keeping multiple jobs together clearly improves spoken flow, you may keep them together.
        - Short hinge sentences are allowed when they create a cut, dread beat, humiliating realization, or irreversible turn.
        - Repetition is allowed when it creates force or dread; do not strip it out just to sound efficient.

        PARAGRAPHS
        Paragraphs should reflect movements of thought, not source layout.

        Useful shapes include:
        - concrete fact -> consequence
        - political reality -> proof
        - chronology -> interpretation
        - institution -> human stakes
        - decision scene -> broader argument
        - emotional cost -> political consequence

        Most paragraphs should feel like sustained spoken movements.
        Most paragraphs should land on a consequence, decision, contradiction, sharpened fact, or newly visible stake.
        When a section already contains a planned host line, preserve its distinctness instead of smoothing it back into generic exposition.
        Open from something concrete already present in the batch.

        Avoid:
        - paragraph endings that merely taper off
        - miniature thesis stamps unless the movement truly concludes there
        - more than one of every three late paragraphs ending in a compact authorial verdict line

        In later paragraphs, prefer consequence, image, pressure, or unresolved contradiction more often than summary judgment.
        Do not rise into institutional analysis and then fall back into fresh ground-level inventory unless the source clearly requires it.
        If cabinet math, constitutional structure, or institutional breakdown appears late, treat it as explanation of what the listener has already felt, not as a detour before returning to fresh scene material.
        Usually end once. Avoid stacked closes.

        WHAT NOT TO WRITE
        Do not tell the listener that a moment matters before the material has made it matter.
        Do not announce hinges, pivots, turning points, or the weight of what is coming.
        Do not use visible planning language.
        Do not paraphrase `movement_goal`, `scene_card_ids`, `framing`, or other pipeline scaffolding into audible prose.
        Do not write cold-open resets at later batch boundaries.
        Do not use narrator nudges, thesis stamps, rhetorical filler, or abstract-noun crutches as a substitute for movement.
        Do not flatten an authored host callback or evaluation into bland connective tissue.
        Avoid AI-cliches and topic-announcing transitions.
        Do not grade the scene from above.
        Avoid lines that declare that a person understood, learned, absorbed a lesson, named something correctly, or that an arrangement or situation was already visible, unless the batch itself explicitly supports that consciousness.
        Avoid mic-drop summary sentences that merely restate what the previous lines have already made clear.
        Avoid abstract-noun handrails as line-openers when a concrete object, place, body, institution, or named actor can take the sentence instead.
        If you must choose between a cleaner essay sentence and a sharper spoken sentence, choose the sharper spoken sentence.

        Avoid timeline/geography handrails whose only job is to steer the listener around the outline:
        “Back up,” “Rewind,” “Cut to,” “Fast-forward,” “Meanwhile,” “Step back,” and close variants.

        EXPANSION
        Slight expansion is allowed only when it restores audible force, a clearer referent, or a stronger landing.
        Do not use expansion to turn a hard line into a fuller explanation.
        When you add a beat, it should sharpen dread, pressure, or comprehension, not soften them.

        VOICE UNDER COERCION
        When the material turns coercive, humiliating, or irreversible, allow the prose to become barer and more percussive.
        Restore force through sequence and cadence, not by adding adjectives.

        NARRATOR EPISTEMICS
        Do not flatten certainty.
        If the source is exact, sound exact.
        If the source is approximate, contested, or open, keep it that way.

        TTS AND SPEECH HINTS
        `speech_hints` should help rendering, not compensate for weak prose.
        Add `speech_hints.pronunciation_hints` only for names or terms likely to be misread.
        Keep `spoken_as` concise.
        Keep the hint set small.
        Use `render_strategy`, emphasis, and pacing conservatively.

        Hard rule:
        Add at most 8 pronunciation hints unless the batch genuinely cannot be rendered intelligibly without more.

        Default to fewer.

        Prefer only:
        - high-frequency recurring names
        - terms with unusual transliteration
        - terms likely to be mangled by TTS

        Do not add hints for one-off terms unless they are crucial to comprehension or likely to be badly distorted.

        If there is no strong reason to do otherwise, prefer restrained delivery:
        - style: measured
        - intensity: light
        - pace: normal
        - render_strategy: plain
        - If the prose still contains several hard-beat lines after rewriting, `split_sentences` is acceptable.
        - Do not leave `render_strategy` at `plain` by reflex when separated impacts would clearly play better in audio.

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
        2. Did you extract and regroup content moves instead of drafting from source sentences?
        3. Did you consolidate overlapping material instead of repeating it?
        4. If the source timeline was partly overlapping or ambiguous, did you handle that conservatively and clearly rather than smoothing it away?
        5. Did you silently repair any source contradiction by deleting or suppressing one side? If so, undo that and handle the uncertainty more honestly.
        6. If `previous_spoken_tail` was present, did it shape continuity without being repeated or used as extra source material?
        7. Did you preserve all facts, chronology, names, dates, numbers, quotations, and certainty?
        8. Is the most important turn in each movement easy to hear and retain?
        9. Are any sentences accurate but overloaded? If so, redistribute the information.
        10. Are any non-quoted passages still too close to source wording or clause order? If so, rewrite them.
        11. Did you spend too much effort on line-level freshness before settling chronology and structure?
        12. Do the paragraph boundaries now serve spoken logic rather than source layout?
        13. Do paragraph endings land on consequence, contradiction, or decision rather than simply stopping?
        14. Have you avoided stacking late-paragraph verdict lines?
        15. Does the narration sound like a serious host carrying thought forward in real time?
        16. Does the prose remain easy to follow in one hearing?
        17. Does `speech_hints` remain minimal and genuinely useful?
        18. Does the JSON match `expected_schema` exactly?
        19. Did you preserve the batch's hardest lines instead of moderating them?
        20. Did any paragraph arrive at its conclusion before the listener had lived through the sequence?
        21. Did you compress away a staircase beat that should have remained serial?
        22. Did any late analysis create narrative whiplash?
        23. Did you stack more than one close?
        24. Does this sound like a host speaking under live historical pressure rather than a writer polishing an essay paragraph?

        Return only the JSON object.
        """
    ).strip()


def section_local_spoken_delivery_instructions() -> str:
    return dedent(
        """
        You are the `spoken_delivery` stage for a historical podcast pipeline.

        TASK
        Rewrite one already-drafted prose section into clean spoken narration that can be performed cleanly in audio.

        You are not replanning the episode.
        You are not adding research.
        You are not merging sections with each other.
        You are not importing material from other sections.
        You may restructure, regroup, and split movements within the current section if that improves spoken delivery.

        PERSONA
        THE ATMOSPHERIC WITNESS

        - Be an observant, present-tense historian treating the past as a physical place.
        - Speak into the ear of a single, intelligent listener.
        - Prioritize the weather of a scene. Replace abstract summaries with physical friction.
        - Pivot from geopolitical forces to immediate pressure on individuals.
        - Make every turn feel like a consequence or a loss in motion, not just a fact on a timeline.
        - Avoid AI-cliches and topic-announcing transitions. Follow the momentum of events.
        - Mix short, punchy factual pivots with longer, rhythmic interpretive sentences.

        INPUT AND AUTHORITY
        You will receive:
        - `episode_number`
        - `section`
        - `max_words_per_segment`
        - `tts_provider`
        - `host_policy`
        - optional `previous_spoken_tail`

        `section` contains:
        - `section_id`
        - `purpose`
        - `anchor`
        - `closure_mode`
        - `movement_goal`
        - `analysis_goal`
        - `key_terms`
        - `authorial_passages`
        - `host_moves`
        - `text`

        - `section.text` is the source of truth.
        - Preserve facts, chronology, names, dates, numbers, quotations, uncertainty, and claims.
        - `movement_goal`, `purpose`, `anchor`, `closure_mode`, `analysis_goal`, `key_terms`, `authorial_passages`, `host_moves`, `host_policy`, and `previous_spoken_tail` are control signals, not evidence. If they conflict with `section.text`, preserve it.
        - Use `movement_goal` to understand what the section is trying to do, but do not expose it.
        - Use `anchor` and `closure_mode` to orient and land the section, not to add content.
        - Use `analysis_goal`, `key_terms`, and `authorial_passages` only to preserve explanatory shape already present in the prose.
        - Use `host_moves` to preserve where an authored host orientation, pivot, or callback should remain distinct. Do not add new host commentary the section text cannot support.
        - `previous_spoken_tail`, if present, is continuity scaffolding only. Use it only to avoid a seam or continue live motion. Do not repeat it, paraphrase it, summarize it, or import facts from it unless the same material already appears in the current section text.
        - `max_words_per_segment` is a render constraint. Write prose that can split cleanly near that scale without markers.
        - `tts_provider` is only for calibrating `speech_hints`.
        - Do not add facts, motives, chronology, quotations, certainty, or interpretation from control metadata.
        - Respect `host_policy`: no first-person singular, and first-person plural only for callbacks, handoffs, or closing residue already justified by the prose.
        - Preserve section scope. Do not import material from other sections.
        - `style_audit` has already handled cross-section dedupe. Do not try to consolidate material you cannot see.

        TRANSFORMATION RULES
        Be faithful to the content. Do not be faithful to the delivery mechanism.
        Preserve the section's full factual and argumentative substance, but do not preserve sentence structure, paragraph structure, or local explanatory order just because it works on the page.
        The main failure mode is keeping the source sentence and merely smoothing it.
        Do not draft from source sentences. Draft from extracted content moves.
        Outside quotations, verse, titles, and indispensable historical formulations, do not preserve long runs of source wording. If a sentence tracks the source too closely in wording, clause order, or proposition order, rewrite it.
        Hard rule: outside quotations and indispensable formulations, do not preserve any source sentence that carries the same two major propositions in the same order.

        DRAFTING WORKFLOW
        Work in this order:
        1. factual and chronological fidelity
        2. spoken architecture within the section
        3. clarity in one hearing
        4. line-level freshness
        5. speech-hint cleanup

        Before drafting, silently extract content moves such as event, claim, context, quotation, explanation, consequence, and pressure point. Then identify the local spine: what pressure is live, what turn changes the situation, what contradiction or cost gives the section force, what consequence must be visible by the end, and what claims cannot be lost.

        Do not treat source paragraphs or sentences as stable compositional units. One paragraph may need to become several spoken movements; several paragraphs may belong to one spoken movement. Repeated material inside the current section should be consolidated, not repeated.

        Resolve chronology conservatively.
        If the source appears to compress, blur, partly overlap events, or create tension between temporal frames:
        - do not invent clarity the source does not provide
        - do not force two events into one unless the overlap strongly supports it
        - do not split one event into two unless the source clearly requires it
        - do not solve the problem with elegant smoothing alone
        - if one dated scene is clearly firmer than the others, anchor the narration there
        - if pressure extends beyond that scene, present it as continuing pressure rather than as a second confidently dated scene
        - if the section does not allow a clean reconciliation, preserve the uncertainty in your wording rather than silently deciding it away

        If the source material contains an internal contradiction, do not silently repair it by dropping one side unless the other side is clearly unsupported by the section as a whole.
        Prefer one of these moves:
        - anchor the narration to the clearest dated moment and treat the rest as continuing pressure
        - preserve the uncertainty explicitly
        - narrow the wording so you do not claim more certainty than the section supports

        Do not fix a source contradiction invisibly just to make the narration cleaner.
        Regroup the extracted moves into a better spoken order, then draft from that order rather than from the source sentences.

        CONTINUITY
        - If `previous_spoken_tail` is present, continue rather than restart.
        - Use it only to avoid a seam or continue live motion.
        - Do not repeat, paraphrase, summarize, or import facts from the previous tail.
        - Do not manufacture a fresh cold open except for the first section.
        - If `previous_spoken_tail` ends later in time than the current section's main scene, preserve continuity of pressure, theme, or contradiction rather than pretending the chronology moves straight forward.
        - If this section begins mid-argument, mid-scene, or mid-pressure, pick up the live motion already in progress.

        SPOKEN QUALITY
        Write like a first-rate historian speaking aloud to one intelligent listener through headphones.
        The listener should hear a narrative mind carrying thought forward, pressure and consequence in motion, explanation that sounds spoken rather than diagrammed, and authority.
        The listener should not hear page prose with lighter punctuation, audiobook gravitas, forced podcast texture, teaser language, or stitched-heading transitions.
        The narrator may be dry, intimate, amused, appalled, cutting, or plainspoken, but only when the material earns it.
        - Vary sentence length and weight. Mix longer interpretive sentences, medium explanatory sentences, and short factual pivots.
        - Do not overload a sentence with too many new names, titles, places, or claims at once. If a sentence asks too much memory work of the listener, redistribute the information across adjacent sentences.
        - Make the most important turn, loss, contradiction, decision, or consequence easy to hear.
        - Transitions should feel like thought moving by consequence, contradiction, pressure, or emotional cost, not by topic-announcing handrails.
        - Write for a voice that must carry the sentence in one pass. If a sentence would likely require rereading on the page, reshape it for the ear.
        - When a source sentence does more than one job, usually split it. If keeping multiple jobs together clearly improves spoken flow, you may keep them together.
        - Paragraphs should reflect movements of thought, not source layout. Most should land on consequence, decision, contradiction, a sharpened fact, or a newly visible stake.
        - If the section already contains a planned host callback or evaluative landing, keep that line distinct instead of smoothing it back into generic exposition.
        - Avoid miniature thesis stamps unless the movement truly concludes there, and avoid stacking late-paragraph verdict lines.
        - In later paragraphs, prefer consequence, image, pressure, or unresolved contradiction more often than summary judgment.

        SECTION-SHAPING RULES
        - Open from something concrete already present in the section.
        - Use `anchor` to maintain audible orientation, not to add new content.
        - Let `purpose` and `movement_goal` shape pacing and emphasis, but do not expose them.
        - Respect `closure_mode`:
          - `residue`: end on pressure, image, decision, or consequence, not a verdict
          - `turn`: land the pivot without sounding like the episode is over
          - `partial_answer`: some interpretive landing is allowed, but keep it proportionate
          - `final_answer`: strongest close allowed
        - Do not add section-opening handrails or summary-reset lines.
        Do not tell the listener that a moment matters before the material has made it matter.
        Do not announce hinges, pivots, turning points, or the weight of what is coming.
        Do not use visible planning language.
        Do not paraphrase `movement_goal`, `purpose`, `anchor`, `closure_mode`, or other pipeline scaffolding into audible prose.
        Do not use narrator nudges, thesis stamps, rhetorical filler, or abstract-noun crutches as a substitute for movement.
        Do not flatten an authored host line into bland connective tissue.
        Do not write cold-open resets at later section boundaries.
        Do not flatten certainty.
        If the source is exact, sound exact.
        If the source is approximate, contested, or open, keep it that way.
        Avoid timeline/geography handrails whose only job is to steer the listener around the outline.

        SPEECH HINTS
        `speech_hints` should help rendering, not compensate for weak prose.
        Add `speech_hints.pronunciation_hints` only for names or terms likely to be misread.
        Keep `spoken_as` concise.
        Keep the hint set small.
        Use `render_strategy`, emphasis, and pacing conservatively.
        Add at most 8 pronunciation hints unless the section genuinely cannot be rendered intelligibly without more.
        Prefer hints only for recurring names, unusual transliterations, or terms likely to be mangled by TTS.
        If there is no strong reason to do otherwise, prefer restrained delivery:
        - style: measured
        - intensity: light
        - pace: normal
        - render_strategy: plain

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
        2. Did you draft from extracted content moves, preserve all facts, and handle chronology or contradiction conservatively?
        3. If `previous_spoken_tail` was present, did it shape continuity without being repeated or treated as extra source material?
        4. Is the most important turn easy to hear, and are any accurate sentences still overloaded?
        5. Are any non-quoted passages still too close to source wording or clause order?
        6. Do paragraph breaks and endings now serve spoken logic rather than page layout?
        7. Have you avoided verdict stacking, narrator nudges, and abstract handrails?
        8. Does `speech_hints` remain minimal and does the JSON match `expected_schema` exactly?
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

        Your job is to remove prose patterns that make the script sound like pipeline output instead of finished narration, while preserving planned host guidance when it is doing real listener work.

        INPUT
        You will receive:
        - `episode_number`
        - `title`
        - `host_policy`
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
        - `analysis_goal`
        - `key_terms`
        - `authorial_passages`
        - `text`

        PRIORITY RULES
        - `sections[].text` is the factual source of truth for this stage.
        - `purpose`, `anchor`, `closure_mode`, `host_moves`, `analysis_goal`, `key_terms`, `authorial_passages`, and `host_policy` are control signals, not evidence.
        - Preserve facts, chronology, names, dates, quotations, uncertainty, and core claims.
        - Keep every section id and section order unchanged.
        - Do not merge sections.
        - Do not split sections.
        - Do not add new claims, interpretations, motives, or scene material.
        - When in doubt, cut rather than rewrite expansively.
        - Preserve planned explanatory passages when they are doing real listener work.

        PRIMARY FAILURE MODES TO FIX
        1. Repeated interpretive landing across adjacent sections.
        2. Interior sections should usually end on scene residue, consequence, or open pressure, not explicit explanation.
        3. Abstract-noun thesis drift such as `mechanism`, `architecture`, `system`, `logic`, `structure`, or `framework` unless historically necessary.
        4. Seam handrails such as "which brings us to", "the pattern is", "that is", or other summary-reset lines.
        5. Second endings.
        6. Repeated causal or pressure restatement in slightly different words.
        7. Overloaded sections whose prose keeps explaining after the point is already clear.
        8. Structural beats drifting into broad synthesis or descriptive backgrounding.
        9. Weak section openings that lose audible orientation.
        10. Planned host moves diluted by adjacent explanation or flattened into generic connective tissue.

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
        - Preserve a planned host move when it offers earned orientation, clarification, evaluation, contrast, callback, naming support, or one light aside.
        - Sharpen one existing line when needed so a planned host move lands cleanly at its intended opening, pivot, or closing position.
        - Preserve planned quote-then-gloss, doctrinal unpacking, institutional clarification, causal compression, comparative aside, or verdict landing when they are clearly anchored in the existing prose.

        NOT ALLOWED
        - No new facts.
        - No new chronology.
        - No new quotations.
        - No new framing language not already supported by the text.
        - Do not delete valid host guidance just because it resembles a handrail.
        - Do not add first-person singular.
        - No section merging or splitting.
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
