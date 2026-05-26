from __future__ import annotations

from podcast_agent.pipeline.orchestrator import (
    _build_architecture_grounding_diagnostics,
    _build_human_grounding_warnings,
    _build_host_density_diagnostics,
    _build_narrative_strategy_actor_arc_diagnostics,
    _build_section_plan_realization,
    _build_section_sonic_realization_diagnostics,
    _build_structural_card_concreteness_warnings,
    _build_style_audit_sections_payload,
)
from _section_progression_helpers import make_section_progression
from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcThread,
    ActorMetadata,
    ActorProfile,
    ArchitectureSection,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeSpine,
    EpisodeScript,
    EventPrimitive,
    FramingBlock,
    NarrativeStrategy,
    ProseSection,
    SceneDiscoveryArtifact,
    SceneActor,
    SceneCard,
    SceneCardDraft,
    SectionSonicPlan,
    StrategyEpisode,
    SynthesisMap,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A file hits the table.",
        threat_or_unresolved_action="Nobody knows who signed it yet.",
        opening_question="What does this order set in motion?",
        handoff_scene_card_id="scene_1",
    )


def _section(section_id: str) -> ArchitectureSection:
    is_close = section_id == "section_close"
    return ArchitectureSection(
        section_id=section_id,
        purpose="setup" if not is_close else "closing",
        approx_runtime_minutes=12.0,
        primitive_ids=["core_1"],
        section_anchor="A folder opens.",
        must_stage_beats=["The order lands.", "The pressure becomes visible."],
        listener_tension="What changes now?",
        section_turn="The pressure becomes visible.",
        transition_logic="The paper trail becomes public.",
        section_progression=make_section_progression(
            "close" if is_close else "setup", label=section_id
        ),
    )


def _strategy_episode_with_actor_directives(
    actor_ids: list[str],
) -> StrategyEpisode:
    return StrategyEpisode.model_construct(
        episode_number=1,
        title="Episode 1",
        arc_summary="Arc",
        episode_spine=EpisodeSpine.model_construct(
            core_primitive_ids=["core_1"],
            support_primitive_roles={},
        ),
        actor_arc_directives=[
            ActorArcDirective(
                actor_id=actor_id,
                arc_threads=[
                    ActorArcThread(
                        thread_id=f"thread_{actor_id}",
                        arc_type="pressure",
                        label=f"{actor_id} thread",
                        premise="Carry the institutional cost.",
                    )
                ],
            )
            for actor_id in actor_ids
        ],
    )


def _scene(
    scene_id: str,
    *,
    section_id: str = "section_1",
    scene_role: str = "action",
    scene_function: str = "scene",
    duration: int = 120,
    actors: list[SceneActor] | None = None,
    entry_image: str = "A file opens.",
    observable_detail: str = "The stamp is still wet.",
    audible_detail: str = "Paper rasps against the desk.",
) -> SceneCard:
    return SceneCard(
        scene_id=scene_id,
        section_id=section_id,
        title=scene_id,
        scene_role=scene_role,
        scene_function=scene_function,
        beat_change="The pressure becomes harder to ignore.",
        must_land_facts=["The order is real."],
        entry_image=entry_image,
        observable_detail=observable_detail,
        audible_detail=audible_detail,
        actors=list(actors or []),
        passage_ids=["p1"],
        host_moves={
            "open": [
                {
                    "move_type": "orient",
                    "note": "Set the listener's footing before the pressure turns.",
                }
            ],
            "pivot": [],
            "close": [],
        },
        estimated_duration_seconds=duration,
    )


def test_scene_card_draft_maps_legacy_setup_role_to_two_axis_defaults() -> None:
    card = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_setup",
            "section_id": "section_1",
            "title": "The file arrives",
            "scene_role": "setup",
            "beat_change": "Pressure becomes concrete.",
            "passage_ids": ["p1"],
            "host_moves": {
                "open": [{"move_type": "orient", "note": "Set the footing."}],
                "pivot": [],
                "close": [],
            },
            "estimated_duration_seconds": 90,
        }
    )

    assert card.scene_role == "context_setup"
    assert card.scene_job == "build"


def test_scene_card_draft_maps_legacy_synthesis_role_to_implication_landing() -> None:
    card = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_synth",
            "section_id": "section_1",
            "title": "What survives the decree",
            "scene_role": "synthesis",
            "beat_change": "The implication can now be stated briefly.",
            "passage_ids": ["p1"],
            "host_moves": {
                "open": [{"move_type": "orient", "note": "Set the footing."}],
                "pivot": [],
                "close": [],
            },
            "estimated_duration_seconds": 75,
        }
    )

    assert card.scene_role == "implication"
    assert card.scene_job == "build"


def test_structural_card_concreteness_warnings_report_missing_image_and_detail() -> (
    None
):
    warnings = _build_structural_card_concreteness_warnings(
        scene_cards=[
            _scene(
                "scene_mech",
                scene_role="implication",
                scene_function="turn",
                entry_image="",
                observable_detail="",
                audible_detail="",
            )
        ]
    )

    assert any(
        warning.startswith("structural_card_missing_entry_image")
        for warning in warnings
    )
    assert any(
        warning.startswith("structural_card_missing_observable_detail")
        for warning in warnings
    )


def test_human_grounding_warning_fires_for_structurally_heavy_episode_without_grounding() -> (
    None
):
    diagnostics, warnings = _build_human_grounding_warnings(
        scene_cards=[
            _scene(
                f"scene_{idx}",
                scene_role="implication",
                scene_function="turn",
                actors=[],
            )
            for idx in range(6)
        ]
    )

    assert diagnostics["structurally_heavy"] is True
    assert diagnostics["grounding_scene_count"] == 0
    assert warnings == [
        "structurally_heavy_episode_missing_human_grounding: 6/6 structural cards and no grounding card"
    ]


def test_section_plan_realization_adds_section_load_warnings() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[_section("section_1")],
        architecture_notes=[],
    )
    scene_cards = [
        _scene(f"scene_{idx}", duration=300, section_id="section_1") for idx in range(5)
    ]

    section_reports, warnings = _build_section_plan_realization(
        episode=architecture,
        scene_cards=scene_cards,
        words_per_minute=145.0,
    )

    assert "section_scene_card_load_high: section_1 -> 5 cards" in warnings
    assert any(
        warning.startswith("section_projected_word_count_high: section_1")
        for warning in warnings
    )
    assert section_reports[0]["scene_card_count"] == 5
    assert section_reports[0]["structural_card_count"] == 0
    assert section_reports[0]["projected_word_count"] > 2200


def test_section_plan_realization_warns_when_required_section_sonic_plan_lacks_derivation() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            _section("section_1").model_copy(
                update={
                    "section_sonic_plan": SectionSonicPlan.model_validate(
                        {
                            "obligation": "required",
                            "opening_anchor": "A crowd goes quiet.",
                            "opening_pressure": "The silence marks organized refusal.",
                        }
                    )
                }
            )
        ],
        architecture_notes=[],
    )

    section_reports, warnings = _build_section_plan_realization(
        episode=architecture,
        scene_cards=[
            _scene(
                "scene_1",
                section_id="section_1",
                audible_detail="",
            )
        ],
        words_per_minute=145.0,
    )

    assert (
        "section_sonic_plan_required_missing_first_scene_derivation: section_1 -> scene_1"
        in warnings
    )
    assert (
        "section_sonic_plan_required_missing_first_scene_derivation: section_1 -> scene_1"
        in (
        section_reports[0]["warnings"]
    )
    )


def test_style_audit_payload_includes_section_load_metadata() -> None:
    plan = EpisodePlan(
        episode_number=1,
        framing=_framing(),
        scene_cards=[
            _scene(
                "scene_1",
                section_id="section_1",
                scene_role="implication",
                scene_function="turn",
            ),
            _scene(
                "scene_2",
                section_id="section_1",
                actors=[SceneActor(name="Worker", presence="primary")],
            ),
        ],
        target_word_count=1000,
    )
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1", "scene_2"],
                movement_goal="keep pressure moving",
                text="The section text.",
            )
        ],
    )
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            _section("section_1").model_copy(
                update={
                    "section_sonic_plan": SectionSonicPlan.model_validate(
                        {
                            "obligation": "preferred",
                            "opening_anchor": "The room goes quiet.",
                            "opening_pressure": "The silence shows everyone waiting for the verdict.",
                        }
                    )
                }
            )
        ],
        architecture_notes=[],
    )

    payload = _build_style_audit_sections_payload(
        script=script,
        architecture=architecture,
        plan=plan,
    )

    assert payload[0]["scene_card_count"] == 2
    assert payload[0]["structural_card_count"] == 1
    assert payload[0]["projected_word_count"] > 0
    assert payload[0]["term_explanations"] == []
    assert payload[0]["section_sonic_plan"]["obligation"] == "preferred"
    assert "host_presence_beats" not in payload[0]


def test_section_plan_realization_warns_on_verbatim_section_sonic_copy_and_unbound_later_beat() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            _section("section_1").model_copy(
                update={
                    "section_sonic_plan": SectionSonicPlan.model_validate(
                        {
                            "obligation": "required",
                            "opening_anchor": "A crowd goes quiet.",
                            "opening_pressure": "The silence marks organized refusal.",
                            "later_beats": [
                                {
                                    "moment": "the rifles answer",
                                    "cue": "rifle reports echo off the walls",
                                }
                            ],
                        }
                    )
                }
            )
        ],
        architecture_notes=[],
    )

    _, warnings = _build_section_plan_realization(
        episode=architecture,
        scene_cards=[
            _scene(
                "scene_1",
                section_id="section_1",
                audible_detail="A crowd goes quiet.",
            ),
            _scene(
                "scene_2",
                section_id="section_1",
                audible_detail="Boots scrape across the stones.",
            ),
        ],
    )

    assert "scene_audible_detail_verbatim_section_copy: section_1 -> scene_1" in warnings
    assert "section_sonic_plan_later_beat_unbound: section_1 -> the rifles answer" in warnings


def test_section_sonic_realization_diagnostics_flag_dry_opening_and_missed_later_beat() -> None:
    diagnostics = _build_section_sonic_realization_diagnostics(
        episode_number=1,
        stage="script",
        sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="setup",
                text="The file arrives and the clerk reads it aloud.",
                section_sonic_plan=SectionSonicPlan.model_validate(
                    {
                        "obligation": "required",
                        "opening_anchor": "The room goes quiet.",
                        "opening_pressure": "The silence shows everyone waiting for the verdict.",
                        "later_beats": [
                            {
                                "moment": "the stamp lands",
                                "cue": "a stamp cracks onto the paper",
                            }
                        ],
                    }
                ),
            )
        ],
    )

    assert diagnostics["warning_count"] >= 2
    assert "section_sonic_opening_not_realized_early: section_1" in diagnostics["warning_labels"]
    assert (
        "section_opening_dry_despite_required_section_sonic_plan: section_1"
        in diagnostics["warning_labels"]
    )
    assert (
        "section_sonic_later_beat_not_realized: section_1 -> the stamp lands"
        in diagnostics["warning_labels"]
    )


def test_host_density_diagnostics_flag_build_scene_overcoverage_and_verdict_density() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            ArchitectureSection(
                section_id="section_1",
                purpose="setup",
                section_progression=make_section_progression("setup", label="section_1"),
                approx_runtime_minutes=10.0,
                primitive_ids=["core_1"],
                section_anchor="A file opens.",
                must_stage_beats=["Beat one.", "Beat two."],
            ),
            ArchitectureSection(
                section_id="section_close",
                purpose="closing",
                section_progression=make_section_progression("close", label="section_close"),
                approx_runtime_minutes=2.0,
                primitive_ids=["core_1"],
                section_anchor="A final folder closes.",
                must_stage_beats=["Beat close one.", "Beat close two."],
            ),
        ],
        architecture_notes=[],
    )
    scene_cards = [
        SceneCard.model_validate(
            {
                **_scene(
                    "scene_dense",
                    section_id="section_1",
                    scene_role="action",
                    scene_function="scene",
                ).model_dump(mode="json"),
                "host_moves": {
                    "open": [{"move_type": "orient", "note": "Wet stamp"}],
                    "pivot": [{"move_type": "clarify", "note": "Court transfer"}],
                    "close": [{"move_type": "evaluate", "note": "Stored grievance"}],
                },
            }
        )
    ] + [
        SceneCard.model_validate(
            {
                **_scene(
                    f"scene_verdict_{idx}",
                    section_id="section_1",
                ).model_dump(mode="json"),
                "host_moves": {
                    "close": [
                        {
                            "move_type": "evaluate",
                            "note": f"Verdict {idx}",
                            "surface_mode": "distinct",
                        }
                    ]
                },
            }
        )
        for idx in range(1, 5)
    ]

    diagnostics, warnings = _build_host_density_diagnostics(
        scene_cards=scene_cards,
        architecture=architecture,
    )

    assert diagnostics["build_scene_ids_with_three_populated_phases"] == ["scene_dense"]
    assert diagnostics["explicit_verdict_scene_count"] == 5
    assert any(
        warning.startswith("build_scene_phase_cap_exceeded:")
        for warning in warnings
    )
    assert any(
        warning.startswith("explicit_verdict_scene_cap_exceeded: count=5")
        for warning in warnings
    )


def test_architecture_grounding_diagnostics_flag_overloaded_run_without_actor_arc_directive() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_2",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            ArchitectureSection(
                section_id="section_1",
                purpose="setup",
                section_progression=make_section_progression("setup", label="section_1"),
                approx_runtime_minutes=9.0,
                primitive_ids=["core_1"],
                section_anchor="Ledger one.",
                must_stage_beats=["Beat one.", "Beat two."],
                key_terms=["court", "waqf", "registry", "decree"],
                term_explanations=[
                    {"item_id": "term_1", "stage": "define"},
                    {"item_id": "term_2", "stage": "define"},
                ],
            ),
            ArchitectureSection(
                section_id="section_2",
                purpose="setup",
                section_progression=make_section_progression("setup", label="section_2"),
                approx_runtime_minutes=9.0,
                primitive_ids=["core_2"],
                section_anchor="Ledger two.",
                must_stage_beats=["Beat three.", "Beat four."],
                key_terms=["seminary", "deed", "court", "trustee"],
                term_explanations=[
                    {"item_id": "term_3", "stage": "define"},
                    {"item_id": "term_4", "stage": "payoff"},
                ],
            ),
        ],
        architecture_notes=[],
    )

    diagnostics = _build_architecture_grounding_diagnostics(
        strategy_episode=_strategy_episode_with_actor_directives([]),
        architecture=architecture,
    )

    assert diagnostics["warning_count"] == 1
    assert diagnostics["overloaded_runs"][0]["directive_actor_ids"] == []
    assert diagnostics["overloaded_runs"][0]["has_recurring_actor_arc_realization"] is False
    assert diagnostics["warnings"][0].startswith(
        "overloaded_run_missing_actor_arc_directive:"
    )


def test_architecture_grounding_diagnostics_flag_missing_directive_realization() -> None:
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_2",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            ArchitectureSection(
                section_id="section_1",
                purpose="setup",
                section_progression=make_section_progression("setup", label="section_1"),
                approx_runtime_minutes=9.0,
                primitive_ids=["core_1"],
                section_anchor="Ledger one.",
                must_stage_beats=["Beat one.", "Beat two."],
                key_terms=["court", "waqf", "registry", "decree"],
                term_explanations=[
                    {"item_id": "term_1", "stage": "define"},
                    {"item_id": "term_2", "stage": "define"},
                ],
                actor_explanations=[
                    {"actor_id": "reza_shah", "stage": "reminder", "role_label": "King"}
                ],
            ),
            ArchitectureSection(
                section_id="section_2",
                purpose="setup",
                section_progression=make_section_progression("setup", label="section_2"),
                approx_runtime_minutes=9.0,
                primitive_ids=["core_2"],
                section_anchor="Ledger two.",
                must_stage_beats=["Beat three.", "Beat four."],
                key_terms=["seminary", "deed", "court", "trustee"],
                term_explanations=[
                    {"item_id": "term_3", "stage": "define"},
                    {"item_id": "term_4", "stage": "payoff"},
                ],
            ),
        ],
        architecture_notes=[],
    )

    diagnostics = _build_architecture_grounding_diagnostics(
        strategy_episode=_strategy_episode_with_actor_directives(["young_khomeini"]),
        architecture=architecture,
    )

    assert diagnostics["warning_count"] == 1
    assert diagnostics["overloaded_runs"][0]["directive_actor_ids"] == ["young_khomeini"]
    assert diagnostics["overloaded_runs"][0]["realized_directive_actor_ids"] == []
    assert diagnostics["warnings"][0].startswith(
        "overloaded_run_missing_actor_arc_realization:"
    )


def test_narrative_strategy_actor_arc_diagnostics_warn_when_actor_evidence_has_no_directives() -> None:
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(actor_id="young_khomeini", display_name="Young Khomeini", actor_type="person"),
            ActorProfile(actor_id="reza_shah", display_name="Reza Shah", actor_type="person"),
        ],
    )
    strategy = NarrativeStrategy.model_construct(
        episodes=[_strategy_episode_with_actor_directives([])],
    )
    synthesis_map = SynthesisMap.model_construct(
        project_id="proj",
        primitives=[
            EventPrimitive(
                id="core_1",
                substrate="events",
                title="A seminary shock",
                core_passage_ids=["p1"],
                actor_ids=["young_khomeini", "reza_shah"],
                event_type="shock",
                what_happened="The seminary witnesses the rupture.",
            )
        ],
    )
    scene_discovery = SceneDiscoveryArtifact.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "c1",
                    "primitive_ids": ["core_1"],
                    "passage_ids": ["p1"],
                    "scene_sketch": "A seminarian watches the rupture.",
                    "scene_jobs": ["opening"],
                    "anchor_image": "A student freezes in the courtyard.",
                    "why_sceneable": "The human witness is obvious.",
                    "quote_anchor": "",
                    "actor_ids": ["young_khomeini"],
                }
            ],
        }
    )

    diagnostics = _build_narrative_strategy_actor_arc_diagnostics(
        strategy=strategy,
        synthesis_map=synthesis_map,
        scene_discovery=scene_discovery,
        actor_metadata=actor_metadata,
    )

    assert diagnostics["warning_count"] == 1
    assert diagnostics["episodes"][0]["clear_actor_evidence"] is True
    assert diagnostics["warnings"][0] == (
        "actor_arc_directives_missing_with_actor_evidence: episode_1"
    )


def test_narrative_strategy_actor_arc_diagnostics_warn_when_actor_rich_episode_has_only_one_directive() -> None:
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(actor_id="young_khomeini", display_name="Young Khomeini", actor_type="person"),
            ActorProfile(actor_id="reza_shah", display_name="Reza Shah", actor_type="person"),
        ],
    )
    strategy = NarrativeStrategy.model_construct(
        episodes=[_strategy_episode_with_actor_directives(["young_khomeini"])],
    )
    synthesis_map = SynthesisMap.model_construct(
        project_id="proj",
        primitives=[
            EventPrimitive(
                id="core_1",
                substrate="events",
                title="A seminary shock",
                core_passage_ids=["p1"],
                actor_ids=["young_khomeini", "reza_shah"],
                event_type="shock",
                what_happened="The seminary witnesses the rupture.",
            )
        ],
    )

    diagnostics = _build_narrative_strategy_actor_arc_diagnostics(
        strategy=strategy,
        synthesis_map=synthesis_map,
        scene_discovery=None,
        actor_metadata=actor_metadata,
    )

    assert diagnostics["warning_count"] == 1
    assert diagnostics["episodes"][0]["actor_rich_episode"] is True
    assert diagnostics["warnings"][0] == (
        "actor_arc_directives_thin_for_actor_rich_episode: episode_1"
    )
