from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.agents.writing import WritingAgent
from podcast_agent.config import LLMConfig
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import _build_scene_primitive_briefs
from podcast_agent.prompts import (
    episode_writing_instructions,
    episode_writing_no_citations_instructions,
    primitive_function_tagging_instructions,
    primitive_substrate_extraction_instructions,
)
from podcast_agent.schemas.models import (
    EventPrimitive,
    NarrationHooks,
    PivotJustification,
    PrimitiveFunction,
    PrimitiveFunctionTaggingArtifact,
    PrimitiveFunctionTaggingOverlayArtifact,
    PrimitiveIrreversibility,
    PrimitiveSalience,
    PrimitiveSubstrate,
    SceneCardDraft,
    SynthesisPrimitivesArtifact,
)


def test_event_primitive_requires_pivot_payload_when_pivot_tag_present() -> None:
    with pytest.raises(ValidationError, match="pivot requires its paired justification payload"):
        EventPrimitive(
            substrate=PrimitiveSubstrate.EVENTS,
            title="A turn",
            core_passage_ids=["passage_01"],
            functions=["pivot"],
            event_type="crisis",
            what_happened="A crisis lands publicly.",
        )


def test_scene_card_dedupes_primitive_ids() -> None:
    card = SceneCardDraft.model_validate(
        {
            "section_id": "section_01",
            "title": "Opening beat",
            "scene_role": "context_setup",
            "scene_function": "scene",
            "beat_change": "The listener enters the pressure.",
            "must_land_facts": ["A concrete change lands."],
            "primitive_ids": ["prim_1", "prim_1", "prim_2", ""],
            "passage_ids": ["passage_01"],
            "estimated_duration_seconds": 120,
            "host_moves": {
                "open": [
                    {
                        "move_type": "orient",
                        "note": "Use the room to orient the listener.",
                        "surface_mode": "woven",
                        "address_mode": "implicit",
                    }
                ]
            },
        }
    )

    assert card.primitive_ids == ["prim_1", "prim_2"]


def test_primitive_extraction_prompt_is_rich_and_schema_safe() -> None:
    extraction = primitive_substrate_extraction_instructions()

    assert "SUBSTRATE ONTOLOGY" in extraction
    assert "WORKING METHOD" in extraction
    assert "FIELDS PRESENT ON EVERY PRIMITIVE" in extraction
    assert "SUBSTRATE REQUIRED FIELD MATRIX" in extraction
    assert "SCHEMA-BOUND FIELD RULES" in extraction
    assert "OUTPUT ASSEMBLY METHOD" in extraction
    assert "SUBSTRATE-SPECIFIC DECISION RULES" in extraction
    assert "AMBIGUITY HANDLING" in extraction
    assert "OUTPUT CONTRACT" in extraction
    assert "specificity_anchors" not in extraction
    assert "`events`: required `event_type`, required `what_happened`" in extraction
    assert "`events`: required `event_type`, required `what_happened`, optional `event_result`" not in extraction
    assert "COMPACT TRANSPORT KEYS" not in extraction
    assert "Do not treat `title` as a substitute for subtype-required fields." in extraction
    assert "Do not emit primitive `id`; local code will assign ids after extraction." in extraction
    assert "Do not emit per-item `substrate`" in extraction
    assert "The top-level output must be a substrate map" in extraction
    assert "events (87–120)" in extraction
    assert "readings (11–13)" in extraction
    assert "`project_id`, `events`, `acts`, `actor_portraits`, `mechanisms`, `conditions`, `artifacts`, `readings`, `quality_score`, `quality_notes`." in extraction
    assert "- `id`" not in extraction
    assert "- `substrate`" not in extraction
    assert "For every `events` primitive, are `event_type` and `what_happened` present and non-empty?" in extraction
    assert "In this stage, do not fill deferred substrate-detail fields that belong to later narrative enrichment." in extraction
    assert "Do not emit `event_result` in this stage; later function tagging may add downstream consequence wording." in extraction
    assert "Do not emit `immediate_result` in this stage." in extraction
    assert "defer `operating_pressure` to function tagging." in extraction
    assert "do not emit `operating_chain`, `inputs`, or `outputs` in this stage." in extraction
    assert "Do not emit `active_tension` in this stage." in extraction
    assert "`artifact_detail` is deferred to function tagging." in extraction
    assert "If `synthesis_feedback` is present, fix only the named primitive/field errors first" in extraction
    assert (
        "`acts.act_type`: `decision`, `refusal`, `delay`, `deferral`, `order`, `defection`, `other`"
        in extraction
    )
    assert "Never emit ad hoc labels such as `tactical_pivot`" in extraction
    assert "tactical_pivot" in extraction
    assert "tactical pivots" not in extraction
    assert "To keep extraction lean, aim to keep these fields at 20 words or fewer" in extraction
    assert "- `axis_ids`" not in extraction


def test_primitive_function_tagging_prompt_is_rich_and_substrate_specific() -> None:
    tagging = primitive_function_tagging_instructions("events")

    assert "primitive_function_tagging_events" in tagging
    assert "WORKING METHOD" in tagging
    assert "FUNCTION-JUSTIFICATION DISCIPLINE" in tagging
    assert "OUTPUT ASSEMBLY METHOD" in tagging
    assert "SALIENCE SCORING METHOD" in tagging
    assert "NARRATION HOOKS" in tagging
    assert "SUBSTRATE-SPECIFIC FUNCTION LOGIC" in tagging
    assert "BORDERLINE CASES AND TIE-BREAKS" in tagging
    assert "FAILURE MODES" in tagging
    assert "OUTPUT CONTRACT" in tagging
    assert "COMPACT TRANSPORT KEYS" not in tagging
    assert "Assign between 0 and 3 function tags per primitive" in tagging
    assert "Do not use substrate identity as a proxy for function." in tagging
    assert "A primitive is not automatically entitled to a tag just because of its substrate." in tagging
    assert "If choosing between `stake` and `cost`" in tagging
    assert "The contract is bidirectional" in tagging
    assert "Compare `functions` against `pivot`, `stake`, `texture`, `cost`, `complication`, `recurrence`, and `contest`" in tagging
    assert "fix only the named issue" in tagging
    assert "narration_hooks" in tagging
    assert "authorial_move" in tagging
    assert "`passage_list`" in tagging
    assert "`core_passage_ids`" in tagging
    assert "`support_passage_ids`" in tagging
    assert "`axis_ids`" not in tagging
    assert "some deferred substrate-detail fields may be absent by design" in tagging
    assert "Return exactly one enrichment overlay for every input primitive" in tagging
    assert "Return only valid JSON matching `PrimitiveFunctionTaggingOverlayArtifact`" in tagging
    assert "First resolve any deferred substrate-detail fields for this substrate" in tagging
    assert "DEFERRED SUBSTRATE DETAIL COMPLETION" in tagging
    assert "`events.event_result`" in tagging
    assert "`event_type -> etype`, `what_happened -> event`." not in tagging
    assert "`condition_summary -> cond`" not in tagging
    assert "`mechanisms.operating_chain`, `mechanisms.inputs`, and `mechanisms.outputs`" not in tagging
    assert "`artifacts.artifact_detail`" not in tagging
    assert "Build each overlay in this order: resolve deferred substrate-detail fields" in tagging
    assert (
        "Did you fill the deferred substrate-detail fields required for this substrate, and only for this substrate?"
        in tagging
    )


def test_primitive_function_tagging_prompt_for_conditions_is_family_clean() -> None:
    tagging = primitive_function_tagging_instructions("conditions")

    assert "primitive_function_tagging_conditions" in tagging
    assert "`condition_summary -> cond`." not in tagging
    assert "`conditions.active_tension`" in tagging
    assert "This `conditions` batch may add only deferred substrate-detail fields owned by `conditions`." in tagging
    assert "`event_type -> etype`" not in tagging
    assert "`acts.immediate_result`" not in tagging
    assert "`mechanisms.operating_chain`, `mechanisms.inputs`, and `mechanisms.outputs`" not in tagging
    assert "`artifacts.artifact_detail`" not in tagging


def test_primitive_function_tagging_prompt_for_readings_has_no_foreign_deferred_fields() -> None:
    tagging = primitive_function_tagging_instructions("readings")

    assert "primitive_function_tagging_readings" in tagging
    assert "`reading_summary -> read`." not in tagging
    assert "There are no deferred substrate-detail fields to fill for this substrate in this pass." in tagging
    assert "This `readings` batch may not add deferred substrate-detail fields owned by other substrates." in tagging
    assert "`events.event_result`" not in tagging
    assert "`conditions.active_tension`" not in tagging
    assert "`artifacts.artifact_detail`" not in tagging


def test_extraction_artifact_allows_deferred_mechanism_and_artifact_fields() -> None:
    artifact = SynthesisPrimitivesArtifact.model_validate(
        {
            "project_id": "proj",
            "primitives": [
                {
                    "id": "mech_1",
                    "substrate": "mechanisms",
                    "title": "A patronage circuit",
                    "core_passage_ids": ["passage_01"],
                    "mechanism_name": "patronage chain",
                },
                {
                    "id": "art_1",
                    "substrate": "artifacts",
                    "title": "A visible decree",
                    "core_passage_ids": ["passage_02"],
                    "artifact_type": "object",
                    "artifact_label": "A signed parchment on the table",
                },
            ],
        }
    )

    assert [primitive.id for primitive in artifact.primitives] == ["mech_1", "art_1"]


def test_tagged_artifact_still_requires_full_mechanism_and_artifact_fields() -> None:
    with pytest.raises(ValidationError, match="operating_chain"):
        PrimitiveFunctionTaggingArtifact.model_validate(
            {
                "project_id": "proj",
                "primitives": [
                    {
                        "id": "mech_1",
                        "substrate": "mechanisms",
                        "title": "A patronage circuit",
                        "core_passage_ids": ["passage_01"],
                        "mechanism_name": "patronage chain",
                        "functions": [],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="artifact_detail"):
        PrimitiveFunctionTaggingArtifact.model_validate(
            {
                "project_id": "proj",
                "primitives": [
                    {
                        "id": "art_1",
                        "substrate": "artifacts",
                        "title": "A visible decree",
                        "core_passage_ids": ["passage_02"],
                        "artifact_type": "object",
                        "artifact_label": "A signed parchment on the table",
                        "functions": [],
                    }
                ],
            }
        )


def test_overlay_artifact_forbids_extraction_owned_fields() -> None:
    with pytest.raises(ValidationError, match="summary"):
        PrimitiveFunctionTaggingOverlayArtifact.model_validate(
            {
                "project_id": "proj",
                "overlays_by_id": {
                    "evt_1": {
                        "summary": "A revised summary.",
                        "functions": ["pivot"],
                        "pivot": {
                            "what_changed": "The field reroutes.",
                            "irreversibility": "med",
                        },
                        "event_result": "The turn hardens the conflict.",
                    }
                },
            }
        )


def test_event_primitive_validates_without_removed_fields() -> None:
    primitive = EventPrimitive(
        substrate=PrimitiveSubstrate.EVENTS,
        title="A turn",
        core_passage_ids=["passage_01"],
        event_type="crisis",
        what_happened="A crisis lands publicly.",
    )

    assert primitive.core_passage_ids == ["passage_01"]


def test_writing_prompts_align_scene_primitive_brief_contract() -> None:
    prompt_with_citations = episode_writing_instructions()
    prompt_without_citations = episode_writing_no_citations_instructions()
    required_fragments = [
        "full, window-scoped scene-bound primitive objects",
        "Treat `plan.scene_cards[].primitive_ids` as the binding scene-level primitive contract.",
        "`scene_primitive_briefs`, when present, is window-scoped",
        "substrate-specific fields, function payloads and justifications, `salience`, and `narration_hooks`",
        "Use enriched fields to decide foreground, bounded explanation, detail triage, host move eligibility, and residue",
        "Do not treat `scene_primitive_briefs` as substitute evidence",
    ]

    for prompt in (prompt_with_citations, prompt_without_citations):
        for fragment in required_fragments:
            assert fragment in prompt


def test_llm_config_includes_new_primitive_grid_stage_entries() -> None:
    config = LLMConfig()

    assert "primitive_substrate_extraction" in config.agent_configs
    assert "primitive_function_tagging_events" in config.agent_configs
    assert "primitive_function_tagging_readings" in config.agent_configs
    assert config.resolve_max_retry_attempts("primitive_substrate_extraction") == 2
    assert config.resolve_concurrency_limit("primitive_function_tagging_events") == 8
    assert config.resolve_thinking_budget("primitive_function_tagging_events") == 20_000


def test_writing_agent_payload_includes_scene_primitive_briefs_only_when_provided() -> None:
    agent = WritingAgent(HeuristicLLMClient())
    payload = agent.build_payload(
        episode_number=1,
        strategy_episode={"episode_number": 1},
        architecture={"episode_number": 1, "sections": []},
        episode_plan={"episode_number": 1, "scene_cards": []},
        passages=[],
        book_metadata=[],
    )

    assert "scene_primitive_briefs" not in payload

    scene_primitive_briefs = {
        "scene_01": [
            {
                "id": "prim_1",
                "substrate": "events",
                "title": "A decisive turn",
                "core_passage_ids": ["passage_01"],
                "support_passage_ids": ["passage_02"],
                "timeframe": "1857",
                "geography": "Delhi",
                "actor_ids": ["actor_1"],
                "functions": ["pivot"],
                "salience": None,
                "pivot": {
                    "what_changed": "The crisis becomes public.",
                    "irreversibility": "high",
                },
                "stake": None,
                "texture": None,
                "cost": None,
                "complication": None,
                "recurrence": None,
                "contest": None,
                "narration_hooks": None,
                "event_type": "crisis",
                "what_happened": "A crisis lands publicly.",
                "event_result": "Public pressure rises.",
            }
        ]
    }
    payload_with_briefs = agent.build_payload(
        episode_number=1,
        strategy_episode={"episode_number": 1},
        architecture={"episode_number": 1, "sections": []},
        episode_plan={"episode_number": 1, "scene_cards": []},
        passages=[],
        book_metadata=[],
        scene_primitive_briefs=scene_primitive_briefs,
    )

    assert payload_with_briefs["scene_primitive_briefs"] == scene_primitive_briefs


def test_scene_primitive_briefs_are_window_scoped() -> None:
    primitive = EventPrimitive(
        id="prim_event",
        substrate=PrimitiveSubstrate.EVENTS,
        title="A turning event",
        core_passage_ids=["passage_01"],
        support_passage_ids=["passage_02"],
        timeframe="1857",
        geography="Delhi",
        actor_ids=["actor_1"],
        functions=[PrimitiveFunction.PIVOT],
        salience=PrimitiveSalience(
            score=0.9,
            justification="The scene's load-bearing rupture."
        ),
        pivot=PivotJustification(
            what_changed="A crisis becomes undeniable.",
            irreversibility=PrimitiveIrreversibility.HIGH,
        ),
        narration_hooks=NarrationHooks(
            concrete_detail="A sealed order breaks open in public.",
            host_lens="The break stops being private.",
            carry_forward="Everyone now has to react in public.",
        ),
        event_type="crisis",
        what_happened="A crisis lands.",
        event_result="The room understands the stakes at once.",
    )
    scene_in_window = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_01",
            "section_id": "section_01",
            "title": "Window scene",
            "scene_role": "action",
            "scene_function": "scene",
            "beat_change": "The scene turns.",
            "must_land_facts": ["A concrete change lands."],
            "primitive_ids": ["prim_event", "missing_primitive"],
            "passage_ids": ["passage_01"],
            "estimated_duration_seconds": 120,
            "host_moves": {
                "open": [
                    {
                        "move_type": "orient",
                        "note": "Use the room to orient the listener.",
                        "surface_mode": "woven",
                        "address_mode": "implicit",
                    }
                ]
            },
        }
    )
    scene_outside_window = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_02",
            "section_id": "section_01",
            "title": "Later scene",
            "scene_role": "reaction",
            "scene_function": "scene",
            "beat_change": "A later reaction lands.",
            "must_land_facts": ["A later concrete change lands."],
            "primitive_ids": ["prim_event"],
            "passage_ids": ["passage_01"],
            "estimated_duration_seconds": 120,
            "host_moves": {
                "open": [
                    {
                        "move_type": "orient",
                        "note": "Keep the listener inside the action.",
                        "surface_mode": "woven",
                        "address_mode": "implicit",
                    }
                ]
            },
        }
    )

    briefs = _build_scene_primitive_briefs(
        scene_cards=[scene_in_window],
        primitive_lookup={primitive.id: primitive},
    )

    assert set(briefs) == {"scene_01"}
    assert scene_outside_window.scene_id not in briefs
    assert [item["id"] for item in briefs["scene_01"]] == ["prim_event"]
    assert briefs["scene_01"][0] == primitive.model_dump(mode="json")
