from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_episode_architecture_realization,
    _build_episode_architecture_core_passages,
    _build_section_plan_realization,
    _build_episode_planning_passage_refs,
    _flatten_synthesis_primitives,
    _split_sentences,
    _validate_architecture_transition,
)
from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcThread,
    ActorMetadata,
    ActorProfile,
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EpisodeSpine,
    EpochalTurnPrimitive,
    ExtractedPassage,
    NarrativeStrategy,
    PipelineConfig,
    SceneCard,
    StrategyEpisode,
    SupportPrimitiveRole,
    SynthesisPrimitive,
    SynthesisPrimitiveBase,
    SystemsOperatingLogicPrimitive,
    SynthesisMap,
    ThematicCorpus,
    ThematicProject,
    VerdictMode,
)


class _StubLLM:
    def set_run_logger(self, run_logger) -> None:  # pragma: no cover - interface stub
        self.run_logger = run_logger


class _StubTTSClient:
    def set_run_logger(self, run_logger) -> None:  # pragma: no cover - interface stub
        self.run_logger = run_logger


def _core_primitive_ids(prefix: str = "core") -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, 8)]


def _support_primitive_roles(
    prefix: str = "support",
    *,
    overrides: dict[str, SupportPrimitiveRole] | None = None,
) -> dict[str, SupportPrimitiveRole]:
    roles: dict[str, SupportPrimitiveRole] = {
        f"{prefix}_{idx}": SupportPrimitiveRole.MECHANISM
        for idx in range(1, 10)
    }
    if overrides:
        roles.update(overrides)
    return roles


def _primitive(
    primitive_id: str,
    core_passage_ids: list[str] | None = None,
    support_passage_ids: list[str] | None = None,
    *,
    family: str = "epochal_turns",
) -> SynthesisPrimitiveBase:
    if family == "epochal_turns":
        return EpochalTurnPrimitive(
            id=primitive_id,
            family="epochal_turns",
            title=primitive_id,
            summary=f"{primitive_id} summary",
            axis_ids=["axis_1"],
            core_passage_ids=list(core_passage_ids or []),
            support_passage_ids=list(support_passage_ids or []),
            before_state="The prior balance still holds.",
            after_state="The balance breaks.",
            change_driver="A decisive move forces the turn.",
            irreversibility_reason="The fallout cannot be unwound quickly.",
        )
    if family == "systems_and_operating_logics":
        return SystemsOperatingLogicPrimitive(
            id=primitive_id,
            family="systems_and_operating_logics",
            title=primitive_id,
            summary=f"{primitive_id} summary",
            axis_ids=["axis_1"],
            core_passage_ids=list(core_passage_ids or []),
            support_passage_ids=list(support_passage_ids or []),
            system_name=primitive_id,
            mechanism="Pressure moves through the same institutional channels.",
            mechanism_steps=[
                "Orders move through the institution.",
                "Local actors translate those orders into pressure.",
            ],
            inputs=["orders"],
            outputs=["compliance"],
            failure_mode="The system distorts under stress.",
        )
    if family == "afterlives":
        return SynthesisPrimitive(
            id=primitive_id,
            family="afterlives",
            title=primitive_id,
            summary=f"{primitive_id} summary",
            axis_ids=["axis_1"],
            core_passage_ids=list(core_passage_ids or []),
            support_passage_ids=list(support_passage_ids or []),
        )
    raise ValueError(f"Unsupported test primitive family: {family}")


def _full_text(label: str) -> str:
    return (
        f"Drivingalpha {label} appears first. "
        f"Focusbeta {label} appears second. "
        f"Neutral {label} appears third. "
        f"Neutral {label} appears fourth. "
        f"Neutral {label} appears fifth. "
        f"Neutral {label} appears sixth. "
        f"Neutral {label} appears seventh. "
        f"Neutral {label} appears eighth."
    )


def _actor_arc_directive(actor_id: str) -> ActorArcDirective:
    return ActorArcDirective(
        actor_id=actor_id,
        arc_threads=[
            ActorArcThread(
                thread_id=f"{actor_id}_thread",
                arc_type="pressure",
                label="Thread",
                premise="Premise",
            )
        ],
    )


def _actor_profile(actor_id: str) -> ActorProfile:
    return ActorProfile(
        actor_id=actor_id,
        display_name=actor_id.replace("_", " ").title(),
        actor_type="person",
    )


def _planning_response(
    orchestrator: PipelineOrchestrator,
    payload: dict,
) -> object:
    architecture = payload["architecture"]
    strategy_episode = payload["strategy_episode"]
    section_ids = [section["section_id"] for section in architecture["sections"]]
    return orchestrator.episode_planning_agent.response_model.model_validate(
        {
            "episode_number": architecture["episode_number"],
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": strategy_episode["episode_spine"]["listener_question"],
                "handoff_scene_card_id": "scene_1",
            },
            "scene_cards": [
                {
                    "scene_id": f"scene_{idx + 1}",
                    "section_id": section_id,
                    "title": f"Scene {idx + 1}",
                    "scene_role": "setup" if idx == 0 else "reaction",
                    "dominant_primitive_id": "core_1",
                    "spine_relation": "set_stakes" if idx == 0 else "spine_advance",
                    "state_effect": "The stakes become visible.",
                    "entry_image": "Image",
                    "local_question": "Question?",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "passage_ids": ["p_core"],
                    "primitive_ids": list(
                        architecture["sections"][idx]["primitive_ids"]
                    ),
                    "estimated_duration_seconds": 60,
                }
                for idx, section_id in enumerate(section_ids)
            ],
        }
    )


def _strategy_episode(
    *,
    title: str = "Episode One",
    listener_question: str = "Question?",
    thematic_focus: str = "",
    arc_summary: str = "Arc",
    unresolved_questions: list[str] | None = None,
    core_primitive_ids: list[str] | None = None,
    support_primitive_roles: dict[str, SupportPrimitiveRole] | None = None,
    recall_primitive_ids: list[str] | None = None,
    actor_arc_directives: list[ActorArcDirective] | None = None,
) -> StrategyEpisode:
    return StrategyEpisode(
        episode_number=1,
        title=title,
        thematic_focus=thematic_focus,
        arc_summary=arc_summary,
        unresolved_questions=list(unresolved_questions or []),
        actor_arc_directives=list(actor_arc_directives or []),
        episode_spine=EpisodeSpine(
            listener_question=listener_question,
            argument="Claim",
            core_primitive_ids=list(core_primitive_ids or _core_primitive_ids()),
            support_primitive_roles=dict(
                support_primitive_roles or _support_primitive_roles()
            ),
            recall_primitive_ids=list(recall_primitive_ids or []),
        ),
    )


def _episode_architecture(
    *,
    sections: list[ArchitectureSection],
    allowed_recurring_primitive_ids: list[str] | None = None,
) -> EpisodeArchitecture:
    normalized_sections = [
        section.model_copy(
            update={
                "section_anchor": section.section_anchor or f"Anchor for {section.section_id}",
                "must_stage_beats": list(section.must_stage_beats or [
                    f"{section.section_id} beat one",
                    f"{section.section_id} beat two",
                ]),
            }
        )
        for section in sections
    ]
    return EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id=normalized_sections[min(2, len(normalized_sections) - 1)].section_id,
        allowed_recurring_primitive_ids=list(allowed_recurring_primitive_ids or []),
        forbidden_redundancies=[],
        sections=normalized_sections,
        architecture_notes=[],
    )


def test_build_episode_planning_passage_refs_preserves_order_and_provenance() -> None:
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                _primitive(
                    "core_1",
                    core_passage_ids=["p1", "p2"],
                    support_passage_ids=["p3", "p2"],
                )
            ],
            "systems_and_operating_logics": [
                _primitive(
                    "support_1",
                    core_passage_ids=["p3"],
                    support_passage_ids=["p4"],
                    family="systems_and_operating_logics",
                )
            ],
            "afterlives": [
                _primitive("recall_1", support_passage_ids=["p5"], family="afterlives")
            ],
        },
    )

    refs = _build_episode_planning_passage_refs(
        primitive_ids_by_role={
            "core": ["core_1"],
            "support": ["support_1"],
            "recall": ["recall_1"],
        },
        primitive_lookup=_flatten_synthesis_primitives(synthesis_map),
    )

    assert refs == [
        {
            "passage_id": "p1",
            "episode_role": "core",
            "passage_kind": "core",
            "primitive_id": "core_1",
        },
        {
            "passage_id": "p2",
            "episode_role": "core",
            "passage_kind": "core",
            "primitive_id": "core_1",
        },
        {
            "passage_id": "p3",
            "episode_role": "core",
            "passage_kind": "support",
            "primitive_id": "core_1",
        },
        {
            "passage_id": "p3",
            "episode_role": "support",
            "passage_kind": "core",
            "primitive_id": "support_1",
        },
        {
            "passage_id": "p4",
            "episode_role": "support",
            "passage_kind": "support",
            "primitive_id": "support_1",
        },
        {
            "passage_id": "p5",
            "episode_role": "recall",
            "passage_kind": "support",
            "primitive_id": "recall_1",
        },
    ]


def test_build_episode_architecture_realization_reports_omitted_support_and_recall() -> None:
    strategy_episode = _strategy_episode(
        support_primitive_roles=_support_primitive_roles(),
        recall_primitive_ids=["recall_1"],
    )
    architecture = _episode_architecture(
        sections=[
                ArchitectureSection(
                    section_id="section_01",
                    purpose="opening",
                    approx_runtime_minutes=20.0,
                    primitive_ids=_core_primitive_ids(),
                    section_question="Q1?",
                section_resolution="R1",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=["section_02"],
                argument_role="frame",
                inference_mode="scene_first",
                recurrence_role="plant",
                pressure_type="mass_political",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_02",
                purpose="setup",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1"],
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["section_01"],
                sets_up_section_ids=["section_03"],
                argument_role="establish_mechanism",
                inference_mode="mechanism_first",
                recurrence_role="deepen",
                pressure_type="constitutional",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_03",
                purpose="turn",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1"],
                section_question="Q3?",
                section_resolution="R3",
                entry_state="E3",
                exit_state="X3",
                transition_logic="T3",
                depends_on_section_ids=["section_02"],
                sets_up_section_ids=["section_04"],
                argument_role="test_viability",
                inference_mode="contrast_first",
                recurrence_role="deepen",
                pressure_type="communal",
                resolution_type="reversal",
                closure_level="medium",
            ),
            ArchitectureSection(
                section_id="section_04",
                purpose="closing",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1"],
                section_question="Q4?",
                section_resolution="R4",
                entry_state="E4",
                exit_state="X4",
                transition_logic="T4",
                depends_on_section_ids=["section_03"],
                sets_up_section_ids=[],
                argument_role="close",
                inference_mode="aftermath_first",
                recurrence_role="payoff",
                pressure_type="moral",
                resolution_type="containment",
                closure_level="high",
            ),
        ],
    )

    realization = _build_episode_architecture_realization(
        strategy_episode=strategy_episode,
        architecture=architecture,
    )

    assert realization["section_primitive_ids"] == _core_primitive_ids()
    assert realization["missing_core_primitive_ids"] == []
    assert realization["missing_support_primitive_ids"] == [
        "support_1",
        "support_2",
        "support_3",
        "support_4",
        "support_5",
        "support_6",
        "support_7",
        "support_8",
        "support_9",
    ]
    assert realization["missing_recall_primitive_ids"] == ["recall_1"]
    assert realization["warning_count"] == 3
    assert (
        "architecture_section_count_below_target: 4 < 6 (target range 6-8)"
        in realization["warnings"]
    )


def test_build_episode_architecture_realization_skips_section_count_warning_inside_preferred_range() -> None:
    sections = []
    for idx in range(8):
        section_id = f"section_{idx + 1:02d}"
        sections.append(
            ArchitectureSection(
                section_id=section_id,
                purpose="opening" if idx == 0 else "closing" if idx == 7 else "setup",
                approx_runtime_minutes=15.0,
                primitive_ids=["core_1"],
                section_question=f"Q{idx + 1}?",
                section_resolution=f"R{idx + 1}",
                entry_state=f"E{idx + 1}",
                exit_state=f"X{idx + 1}",
                transition_logic=f"T{idx + 1}",
                depends_on_section_ids=[f"section_{idx:02d}"] if idx > 0 else [],
                sets_up_section_ids=[f"section_{idx + 2:02d}"] if idx < 7 else [],
                argument_role="frame" if idx == 0 else "close" if idx == 7 else "establish_mechanism",
                inference_mode="scene_first" if idx == 0 else "mechanism_first",
                recurrence_role="plant" if idx == 0 else "payoff" if idx == 7 else "deepen",
                pressure_type="mass_political",
                resolution_type="containment" if idx == 7 else "escalation",
                closure_level="high" if idx == 7 else "low",
            )
        )
    architecture = _episode_architecture(sections=sections)

    realization = _build_episode_architecture_realization(
        strategy_episode=_strategy_episode(),
        architecture=architecture,
    )

    assert not any(
        warning.startswith("architecture_section_count_")
        for warning in realization["warnings"]
    )


def test_build_episode_architecture_core_passages_uses_full_text_bm25_trim() -> None:
    passages = _build_episode_architecture_core_passages(
        driving_question="Drivingalpha",
        thematic_focus="Focusbeta",
        episode_spine=EpisodeSpine(
            listener_question="Drivingalpha",
            argument="Claim",
            core_primitive_ids=_core_primitive_ids(),
            support_primitive_roles=_support_primitive_roles(),
            recall_primitive_ids=[],
        ),
        primitive_lookup={"core_1": _primitive("core_1", core_passage_ids=["p_core"])},
        passage_lookup={
            "p_core": ExtractedPassage(
                passage_id="p_core",
                book_id="b1",
                chunk_ids=["c1"],
                text="Fallback text",
                trimmed_text="Irrelevant trimmed text.",
                full_text=_full_text("core"),
                chapter_ref="1",
                axis_id="axis_1",
            )
        },
    )

    assert passages == [
        {
            "passage_id": "p_core",
            "primitive_id": "core_1",
            "book_id": "b1",
            "chapter_ref": "1",
            "summary_text": "Drivingalpha core appears first.",
        }
    ]


def test_build_episode_architectures_uses_only_strategy_actor_directives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: _StubLLM(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: _StubTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    captured_payload: dict[str, object] = {}

    def fake_architecture_run(payload: dict) -> EpisodeArchitecture:
        captured_payload.update(payload)
        return _episode_architecture(
            sections=[
                    ArchitectureSection(
                        section_id=f"section_0{idx}",
                        purpose="opening" if idx == 1 else ("turn" if idx == 3 else ("closing" if idx == 4 else "setup")),
                        approx_runtime_minutes=22.5,
                        primitive_ids=_core_primitive_ids() if idx == 1 else ["core_1"],
                        section_question=f"Q{idx}?",
                    section_resolution=f"R{idx}",
                    entry_state=f"E{idx}",
                    exit_state=f"X{idx}",
                    transition_logic=f"T{idx}",
                    depends_on_section_ids=[] if idx == 1 else [f"section_0{idx-1}"],
                    sets_up_section_ids=[] if idx == 4 else [f"section_0{idx+1}"],
                    argument_role="frame" if idx == 1 else ("test_viability" if idx == 3 else ("close" if idx == 4 else "establish_mechanism")),
                    inference_mode="scene_first",
                    recurrence_role="none",
                    pressure_type="mass_political",
                    resolution_type="escalation" if idx < 4 else "containment",
                    closure_level="high" if idx == 4 else "low",
                )
                for idx in range(1, 5)
            ],
        )

    orchestrator.episode_architecture_agent.run = fake_architecture_run

    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        episode_count=1,
        config=PipelineConfig(),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="Because",
        series_arc="Arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                thematic_focus="Focus",
                arc_summary="Arc",
                actor_arc_directives=[_actor_arc_directive("actor_strategy")],
                episode_spine=EpisodeSpine(
                    listener_question="Question?",
                    argument="Claim",
                    core_primitive_ids=_core_primitive_ids(),
                    support_primitive_roles=_support_primitive_roles(),
                    recall_primitive_ids=[],
                ),
            )
        ],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                _primitive("core_1", core_passage_ids=["p_core"]).model_copy(
                    update={
                        "actor_ids": ["actor_primitive"],
                        "primary_actor_ids": ["actor_primitive"],
                        "affected_actor_ids": ["actor_primitive"],
                    }
                )
            ]
        },
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p_core",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Fallback text",
                    trimmed_text="Trimmed text",
                    full_text=_full_text("core"),
                    chapter_ref="1",
                    axis_id="axis_1",
                )
            ]
        },
        total_passages=1,
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            _actor_profile("actor_strategy"),
            _actor_profile("actor_primitive"),
        ],
    )

    architectures, _metrics = asyncio.run(
        orchestrator._build_episode_architectures(
            project,
            synthesis_map,
            strategy,
            corpus,
            tmp_path,
            actor_metadata=actor_metadata,
        )
    )

    assert [actor["actor_id"] for actor in captured_payload["actor_metadata"]["actors"]] == [
        "actor_strategy"
    ]
    assert architectures[0].episode_number == 1


def test_build_section_plan_realization_reports_section_drift_and_unused_priority_passages() -> None:
    episode = _episode_architecture(
        sections=[
            ArchitectureSection(
                section_id="section_01",
                purpose="opening",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1"],
                section_question="What breaks first?",
                section_resolution="The pressure becomes unmistakable.",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=["section_02"],
                argument_role="frame",
                inference_mode="scene_first",
                recurrence_role="plant",
                pressure_type="mass_political",
                resolution_type="escalation",
                closure_level="low",
                priority_core_passage_ids=["p_core"],
            ),
            ArchitectureSection(
                section_id="section_02",
                purpose="setup",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1", "support_1"],
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["section_01"],
                sets_up_section_ids=["section_03"],
                argument_role="establish_mechanism",
                inference_mode="mechanism_first",
                recurrence_role="deepen",
                pressure_type="constitutional",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_03",
                purpose="turn",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1"],
                section_question="Q3?",
                section_resolution="R3",
                entry_state="E3",
                exit_state="X3",
                transition_logic="T3",
                depends_on_section_ids=["section_02"],
                sets_up_section_ids=["section_04"],
                argument_role="test_viability",
                inference_mode="contrast_first",
                recurrence_role="deepen",
                pressure_type="communal",
                resolution_type="reversal",
                closure_level="medium",
            ),
            ArchitectureSection(
                section_id="section_04",
                purpose="closing",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1"],
                section_question="Q4?",
                section_resolution="R4",
                entry_state="E4",
                exit_state="X4",
                transition_logic="T4",
                depends_on_section_ids=["section_03"],
                sets_up_section_ids=[],
                argument_role="close",
                inference_mode="aftermath_first",
                recurrence_role="payoff",
                pressure_type="moral",
                resolution_type="containment",
                closure_level="high",
            ),
        ],
    )
    scene_cards = [
        SceneCard(
            scene_id="scene_1",
            section_id="section_01",
            title="Scene 1",
            scene_role="setup",
            dominant_primitive_id="support_1",
            spine_relation="set_stakes",
            state_effect="The crowd panics.",
            entry_image="Image",
            local_question="Who will hold the line?",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["support_1"],
            passage_ids=["p_other"],
            estimated_duration_seconds=60,
        ),
        SceneCard(
            scene_id="scene_2",
            section_id="section_02",
            title="Scene 2",
            scene_role="reaction",
            dominant_primitive_id="support_1",
            spine_relation="supply_mechanism",
            state_effect="R2 becomes legible.",
            entry_image="Image",
            local_question="Q2?",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["support_1"],
            passage_ids=["p_support"],
            estimated_duration_seconds=60,
        ),
        SceneCard(
            scene_id="scene_3",
            section_id="section_03",
            title="Scene 3",
            scene_role="reaction",
            dominant_primitive_id="core_1",
            spine_relation="turn",
            state_effect="R3 becomes clear.",
            entry_image="Image",
            local_question="Q3?",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["core_1"],
            passage_ids=["p_core"],
            estimated_duration_seconds=60,
        ),
        SceneCard(
            scene_id="scene_4",
            section_id="section_04",
            title="Scene 4",
            scene_role="consequence",
            dominant_primitive_id="core_1",
            spine_relation="show_consequence",
            state_effect="R4 settles in.",
            entry_image="Image",
            local_question="Q4?",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["core_1"],
            passage_ids=["p_core"],
            estimated_duration_seconds=60,
        ),
    ]

    section_reports, warnings = _build_section_plan_realization(
        episode=episode,
        scene_cards=scene_cards,
    )

    first_section = section_reports[0]
    assert first_section["declared_primitive_ids"] == ["core_1"]
    assert first_section["scene_primitive_ids"] == ["support_1"]
    assert first_section["out_of_section_primitive_ids"] == ["support_1"]
    assert first_section["used_priority_core_passage_ids"] == []
    assert first_section["unused_priority_core_passage_ids"] == ["p_core"]
    assert "scene_primitive_outside_section: section_01 -> support_1" in warnings
    assert "dominant_primitive_outside_section: section_01 -> support_1" in warnings
    assert "section_question_not_realized: section_01" in warnings
    assert "section_resolution_not_realized: section_01" in warnings
    assert "section_priority_core_passages_unused: section_01 -> p_core" in warnings


def test_build_section_plan_realization_reports_unrealized_must_stage_beats() -> None:
    episode = _episode_architecture(
        sections=[
            ArchitectureSection(
                section_id="section_01",
                purpose="opening",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1"],
                section_anchor="A cable lands on a desk.",
                must_stage_beats=[
                    "A cable lands on a desk.",
                    "The minister burns the decree in silence.",
                ],
                section_question="What breaks first?",
                section_resolution="The pressure becomes unmistakable.",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=["section_02"],
                argument_role="frame",
                inference_mode="scene_first",
                recurrence_role="plant",
                pressure_type="mass_political",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_02",
                purpose="closing",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1"],
                section_anchor="The square is empty by dawn.",
                must_stage_beats=[
                    "The square is empty by dawn.",
                    "The regime is now visibly alone.",
                ],
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["section_01"],
                sets_up_section_ids=[],
                argument_role="close",
                inference_mode="aftermath_first",
                recurrence_role="payoff",
                pressure_type="moral",
                resolution_type="containment",
                closure_level="high",
            ),
        ],
    )
    scene_cards = [
        SceneCard(
            scene_id="scene_1",
            section_id="section_01",
            title="A cable on the blotter",
            scene_role="setup",
            dominant_primitive_id="core_1",
            spine_relation="set_stakes",
            state_effect="The room now knows the order has arrived.",
            entry_image="A cable lands on a desk.",
            local_question="What breaks first?",
            observable_detail="The paper is still warm from the machine.",
            intended_move="Show the message arriving.",
            primitive_ids=["core_1"],
            passage_ids=["p_core"],
            estimated_duration_seconds=60,
        ),
        SceneCard(
            scene_id="scene_2",
            section_id="section_02",
            title="Dawn after the rout",
            scene_role="consequence",
            dominant_primitive_id="core_1",
            spine_relation="show_consequence",
            state_effect="R2 settles in.",
            entry_image="The square is empty by dawn.",
            local_question="Q2?",
            observable_detail="Paper banners stick to the pavement.",
            intended_move="Show the aftermath.",
            primitive_ids=["core_1"],
            passage_ids=["p_core"],
            estimated_duration_seconds=60,
        ),
    ]

    section_reports, warnings = _build_section_plan_realization(
        episode=episode,
        scene_cards=scene_cards,
    )

    assert section_reports[0]["unrealized_must_stage_beats"] == [
        "The minister burns the decree in silence."
    ]
    assert (
        "section_must_stage_beats_not_realized: section_01 -> 1 beats"
        in warnings
    )


def test_validate_architecture_transition_rejects_missing_section_seed_fields() -> None:
    strategy_episode = _strategy_episode()
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_02",
        sections=[
            ArchitectureSection(
                section_id="section_01",
                purpose="opening",
                approx_runtime_minutes=20.0,
                primitive_ids=_core_primitive_ids(),
                section_question="Q1?",
                section_resolution="R1",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=["section_02"],
                argument_role="frame",
                inference_mode="scene_first",
                pressure_type="mass_political",
                resolution_type="escalation",
            ),
            ArchitectureSection(
                section_id="section_02",
                purpose="closing",
                approx_runtime_minutes=20.0,
                primitive_ids=_core_primitive_ids(),
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["section_01"],
                sets_up_section_ids=[],
                argument_role="close",
                inference_mode="aftermath_first",
                pressure_type="moral",
                resolution_type="containment",
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="section_anchor"):
        _validate_architecture_transition(
            strategy_episode=strategy_episode,
            architecture=architecture,
        )

    anchored_architecture = architecture.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={"section_anchor": f"Anchor for {section.section_id}"}
                )
                for section in architecture.sections
            ]
        }
    )

    with pytest.raises(RuntimeError, match="must_stage_beats"):
        _validate_architecture_transition(
            strategy_episode=strategy_episode,
            architecture=anchored_architecture,
        )

def test_plan_series_trims_core_support_and_recall_by_passage_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: _StubLLM(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: _StubTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    captured_payload: dict[str, object] = {}

    def fake_planning_run(payload: dict) -> object:
        captured_payload.update(payload)
        return _planning_response(orchestrator, payload)

    orchestrator.episode_planning_agent.run = fake_planning_run

    project = ThematicProject(
        project_id="proj",
        theme="themeomega",
        sub_themes=["arcgamma"],
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        episode_count=1,
        config=PipelineConfig(),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="test",
        series_arc="test arc",
        episodes=[
            _strategy_episode(
                title="Episode One",
                listener_question="drivingalpha",
                thematic_focus="focusbeta",
                arc_summary="arcgamma",
                unresolved_questions=["unresolveddelta"],
                core_primitive_ids=_core_primitive_ids(),
                support_primitive_roles=_support_primitive_roles(
                    overrides={"support_omitted_1": SupportPrimitiveRole.TEXTURE}
                ),
                recall_primitive_ids=["recall_1"],
            )
        ],
    )
    architecture = _episode_architecture(
        allowed_recurring_primitive_ids=["core_1"],
        sections=[
            ArchitectureSection(
                section_id="section_01",
                purpose="opening",
                approx_runtime_minutes=20.0,
                primitive_ids=[*_core_primitive_ids(), "support_1"],
                section_question="Q1?",
                section_resolution="R1",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=["section_02"],
                argument_role="frame",
                inference_mode="scene_first",
                recurrence_role="plant",
                pressure_type="mass_political",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_02",
                purpose="setup",
                approx_runtime_minutes=20.0,
                primitive_ids=["core_1", "support_1"],
                section_question="Q2?",
                section_resolution="R2",
                entry_state="E2",
                exit_state="X2",
                transition_logic="T2",
                depends_on_section_ids=["section_01"],
                sets_up_section_ids=["section_03"],
                argument_role="establish_mechanism",
                inference_mode="mechanism_first",
                recurrence_role="deepen",
                pressure_type="constitutional",
                resolution_type="escalation",
                closure_level="low",
            ),
            ArchitectureSection(
                section_id="section_03",
                purpose="turn",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1"],
                section_question="Q3?",
                section_resolution="R3",
                entry_state="E3",
                exit_state="X3",
                transition_logic="T3",
                depends_on_section_ids=["section_02"],
                sets_up_section_ids=["section_04"],
                argument_role="test_viability",
                inference_mode="contrast_first",
                recurrence_role="deepen",
                pressure_type="communal",
                resolution_type="reversal",
                closure_level="medium",
            ),
            ArchitectureSection(
                section_id="section_04",
                purpose="closing",
                approx_runtime_minutes=25.0,
                primitive_ids=["core_1", "recall_1"],
                section_question="Q4?",
                section_resolution="R4",
                entry_state="E4",
                exit_state="X4",
                transition_logic="T4",
                depends_on_section_ids=["section_03"],
                sets_up_section_ids=[],
                argument_role="close",
                inference_mode="aftermath_first",
                recurrence_role="payoff",
                pressure_type="moral",
                resolution_type="containment",
                closure_level="high",
            ),
        ],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                _primitive(
                    "core_1",
                    core_passage_ids=["p_core"],
                    support_passage_ids=["p_core_support"],
                ),
            ],
            "systems_and_operating_logics": [
                    _primitive(
                        "support_1",
                        core_passage_ids=["p_support"],
                        support_passage_ids=["p_support_support"],
                        family="systems_and_operating_logics",
                    ),
                    _primitive(
                        "support_omitted_1",
                        core_passage_ids=["p_omitted"],
                        family="systems_and_operating_logics",
                    ),
                ],
                "afterlives": [
                    _primitive(
                        "recall_1",
                        support_passage_ids=["p_recall"],
                        family="afterlives",
                    ),
                ],
        },
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p_core",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="trimmed core",
                    full_text=_full_text("core"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p_support",
                    book_id="b1",
                    chunk_ids=["c2"],
                    text="trimmed support",
                    full_text=_full_text("support"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p_support_support",
                    book_id="b1",
                    chunk_ids=["c2b"],
                    text="trimmed support support",
                    full_text=_full_text("support support"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p_omitted",
                    book_id="b1",
                    chunk_ids=["c3"],
                    text="trimmed omitted",
                    full_text=_full_text("omitted"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p_core_support",
                    book_id="b1",
                    chunk_ids=["c3b"],
                    text="trimmed core support",
                    full_text=_full_text("core support"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p_recall",
                    book_id="b1",
                    chunk_ids=["c4"],
                    text="trimmed recall",
                    full_text=_full_text("recall"),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
            ]
        },
    )

    plans, _actor_metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            [architecture],
            corpus,
            tmp_path,
        )
    )

    assert [plan.episode_number for plan in plans] == [1]
    available_passages = {
        passage["passage_id"]: passage["text"]
        for passage in captured_payload["available_passages"]
    }
    assert captured_payload["strategy_episode"]["title"] == "Episode One"
    assert _split_sentences(available_passages["p_core"]) == [
        "Drivingalpha core appears first.",
        "Focusbeta core appears second.",
        "Neutral core appears third.",
        "Neutral core appears fourth.",
    ]
    assert _split_sentences(available_passages["p_core_support"]) == [
        "Drivingalpha core support appears first.",
        "Focusbeta core support appears second.",
    ]
    assert _split_sentences(available_passages["p_support"]) == [
        "Drivingalpha support appears first.",
        "Focusbeta support appears second.",
    ]
    assert _split_sentences(available_passages["p_support_support"]) == [
        "Drivingalpha support support appears first.",
        "Focusbeta support support appears second.",
    ]
    assert _split_sentences(available_passages["p_recall"]) == [
        "Drivingalpha recall appears first.",
        "Focusbeta recall appears second.",
    ]
    assert "p_omitted" not in available_passages
    realization_payload = json.loads(
        (tmp_path / "episode_plan_realization.json").read_text(encoding="utf-8")
    )
    episode_report = realization_payload["episodes"][0]
    assert "section_realization" in episode_report
    assert episode_report["section_realization"][1]["declared_primitive_ids"] == [
        "core_1",
        "support_1",
    ]


def test_plan_series_uses_primitive_aware_queries_for_reused_passages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: _StubLLM(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: _StubTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    captured_payload: dict[str, object] = {}

    def fake_planning_run(payload: dict) -> object:
        captured_payload.update(payload)
        return _planning_response(orchestrator, payload)

    orchestrator.episode_planning_agent.run = fake_planning_run

    project = ThematicProject(
        project_id="proj",
        theme="foundation",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        episode_count=1,
        config=PipelineConfig(),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="test",
        series_arc="test arc",
        episodes=[
            _strategy_episode(
                title="Episode One",
                listener_question="What matters here?",
                thematic_focus="Episodefocus should dominate the base query.",
                unresolved_questions=["Which evidence matters most?"],
                core_primitive_ids=_core_primitive_ids(),
                support_primitive_roles=_support_primitive_roles(),
            )
        ],
    )
    architecture = _episode_architecture(
        sections=[
                ArchitectureSection(
                    section_id="section_01",
                    purpose="opening",
                    approx_runtime_minutes=20.0,
                    primitive_ids=[*_core_primitive_ids(), "support_1"],
                    section_question="Q1?",
                section_resolution="R1",
                entry_state="E1",
                exit_state="X1",
                transition_logic="T1",
                depends_on_section_ids=[],
                sets_up_section_ids=[],
                argument_role="frame",
                inference_mode="scene_first",
                recurrence_role="plant",
                pressure_type="mass_political",
                resolution_type="escalation",
                closure_level="low",
            ),
                ArchitectureSection(
                    section_id="section_02",
                    purpose="setup",
                    approx_runtime_minutes=10.0,
                    primitive_ids=["core_1"],
                    section_question="Q2?",
                    section_resolution="R2",
                    entry_state="E2",
                    exit_state="X2",
                    transition_logic="T2",
                    depends_on_section_ids=["section_01"],
                    sets_up_section_ids=["section_03"],
                    argument_role="establish_mechanism",
                    inference_mode="scene_first",
                    recurrence_role="deepen",
                    pressure_type="constitutional",
                    resolution_type="escalation",
                    closure_level="low",
                ),
                ArchitectureSection(
                    section_id="section_03",
                    purpose="turn",
                    approx_runtime_minutes=10.0,
                    primitive_ids=["core_1"],
                    section_question="Q3?",
                    section_resolution="R3",
                    entry_state="E3",
                    exit_state="X3",
                    transition_logic="T3",
                    depends_on_section_ids=["section_02"],
                    sets_up_section_ids=["section_04"],
                    argument_role="test_viability",
                    inference_mode="scene_first",
                    recurrence_role="deepen",
                    pressure_type="communal",
                    resolution_type="reversal",
                    closure_level="medium",
                ),
                ArchitectureSection(
                    section_id="section_04",
                    purpose="closing",
                    approx_runtime_minutes=10.0,
                    primitive_ids=["core_1"],
                    section_question="Q4?",
                    section_resolution="R4",
                    entry_state="E4",
                    exit_state="X4",
                    transition_logic="T4",
                    depends_on_section_ids=["section_03"],
                    sets_up_section_ids=[],
                    argument_role="close",
                    inference_mode="scene_first",
                    recurrence_role="payoff",
                    pressure_type="moral",
                    resolution_type="containment",
                    closure_level="high",
                ),
            ]
        )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                EpochalTurnPrimitive(
                    id="core_1",
                    family="epochal_turns",
                    title="Founding rupture",
                    summary="Panipat becomes the decisive break with the old order.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p_reused"],
                    before_state="The old imperial balance still holds.",
                    after_state="The old order can no longer be restored.",
                    change_driver="A battlefield defeat rearranges the political field.",
                    irreversibility_reason="The institutions and alliances reorganize around the loss.",
                ),
            ],
            "systems_and_operating_logics": [
                SystemsOperatingLogicPrimitive(
                    id="support_1",
                    family="systems_and_operating_logics",
                    title="Dynastic survival logic",
                    summary="The founder pivots from exile toward preserving a threatened line.",
                    axis_ids=["axis_1"],
                    support_passage_ids=["p_reused"],
                    system_name="Dynastic survival",
                    mechanism="Political decisions flow through survival incentives.",
                    mechanism_steps=[
                        "Defeat redirects attention toward survival.",
                        "Survival incentives narrow the next political move.",
                    ],
                    inputs=["military defeat"],
                    outputs=["defensive consolidation"],
                    failure_mode="The system narrows future options as pressure rises.",
                ),
            ],
        },
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p_reused",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="trimmed reused",
                    full_text=(
                        "Episodefocus sentence frames the episode. "
                        "Panipat founding rupture secures legitimacy. "
                        "Dynastic survival preserves the line. "
                        "Neutral background keeps quiet context."
                    ),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
            ]
        },
    )

    plans, _actor_metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            [architecture],
            corpus,
            tmp_path,
        )
    )

    assert [plan.episode_number for plan in plans] == [1]
    available_passages = {
        passage["passage_id"]: passage["text"]
        for passage in captured_payload["available_passages"]
    }
    assert _split_sentences(available_passages["p_reused"]) == [
        "Episodefocus sentence frames the episode.",
        "Dynastic survival preserves the line.",
    ]
