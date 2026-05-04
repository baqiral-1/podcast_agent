from __future__ import annotations

from podcast_agent.pipeline.orchestrator import (
    _build_human_grounding_warnings,
    _build_section_plan_realization,
    _build_structural_card_concreteness_warnings,
    _build_style_audit_sections_payload,
)
from podcast_agent.schemas.models import (
    ArchitectureSection,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodeScript,
    FramingBlock,
    ProseSection,
    SceneActor,
    SceneCard,
    SceneCardDraft,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A file hits the table.",
        threat_or_unresolved_action="Nobody knows who signed it yet.",
        opening_question="What does this order set in motion?",
        handoff_scene_card_id="scene_1",
    )


def _section(section_id: str) -> ArchitectureSection:
    return ArchitectureSection(
        section_id=section_id,
        purpose="setup" if section_id != "section_close" else "closing",
        approx_runtime_minutes=12.0,
        primitive_ids=["core_1"],
        section_anchor="A folder opens.",
        must_stage_beats=["The order lands.", "The pressure becomes visible."],
        listener_tension="What changes now?",
        section_turn="The pressure becomes visible.",
        transition_logic="The paper trail becomes public.",
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
        actors=list(actors or []),
        passage_ids=["p1"],
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
            "estimated_duration_seconds": 90,
        }
    )

    assert card.scene_role == "context_setup"
    assert card.scene_function == "scene"


def test_scene_card_draft_maps_legacy_synthesis_role_to_implication_landing() -> None:
    card = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_synth",
            "section_id": "section_1",
            "title": "What survives the decree",
            "scene_role": "synthesis",
            "beat_change": "The implication can now be stated briefly.",
            "passage_ids": ["p1"],
            "estimated_duration_seconds": 75,
        }
    )

    assert card.scene_role == "implication"
    assert card.scene_function == "landing"


def test_structural_card_concreteness_warnings_report_missing_image_and_detail() -> None:
    warnings = _build_structural_card_concreteness_warnings(
        scene_cards=[
            _scene(
                "scene_mech",
                scene_role="implication",
                scene_function="mechanism",
                entry_image="",
                observable_detail="",
            )
        ]
    )

    assert any(warning.startswith("structural_card_missing_entry_image") for warning in warnings)
    assert any(
        warning.startswith("structural_card_missing_observable_detail")
        for warning in warnings
    )


def test_human_grounding_warning_fires_for_structurally_heavy_episode_without_grounding() -> None:
    diagnostics, warnings = _build_human_grounding_warnings(
        scene_cards=[
            _scene(
                f"scene_{idx}",
                scene_role="implication",
                scene_function="mechanism" if idx < 5 else "landing",
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
        _scene(f"scene_{idx}", duration=300, section_id="section_1")
        for idx in range(5)
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


def test_style_audit_payload_includes_section_load_metadata() -> None:
    plan = EpisodePlan(
        episode_number=1,
        framing=_framing(),
        scene_cards=[
            _scene(
                "scene_1",
                section_id="section_1",
                scene_role="implication",
                scene_function="mechanism",
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
        sections=[_section("section_1")],
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
