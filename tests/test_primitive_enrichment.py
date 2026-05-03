from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import Any

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_primitive_enrichment_passages_by_id,
    _split_sentences,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    BaseSynthesisPrimitive,
    BookRecord,
    ExtractedPassage,
    NarrativeStrategy,
    PipelineConfig,
    StrategyEpisode,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def test_enrich_selected_primitives_merges_rich_fields_and_reuses_shared_core_passages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
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
        lambda settings: DummyTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    captured: dict[str, Any] = {}

    def fake_run(payload: dict[str, Any]) -> Any:
        captured["payload"] = payload
        return orchestrator.primitive_enrichment_agent.response_model.model_validate(
            {
                "project_id": "proj",
                "family": "epochal_turns",
                "enriched_primitives": [
                    {
                        "id": "et_1",
                        "family": "epochal_turns",
                        "before_state": "Before",
                        "after_state": "After",
                        "change_driver": "Driver",
                        "irreversibility_reason": "Reason",
                    }
                ],
            }
        )

    orchestrator.primitive_enrichment_agent.run = fake_run

    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description",
                theme_importance_score=1.0,
            )
        ],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="A long passage about a decisive turn that changes the balance and makes reversal impossible.",
                    full_text="A long passage about a decisive turn that changes the balance and makes reversal impossible.",
                    chapter_ref="ch. 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p2",
                    book_id="b1",
                    chunk_ids=["c2"],
                    text="A supporting passage that adds contextual pressure around the same decisive turn.",
                    full_text="A supporting passage that adds contextual pressure around the same decisive turn.",
                    chapter_ref="ch. 2",
                    axis_id="axis_1",
                )
            ]
        },
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )
    synthesis_primitives = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                BaseSynthesisPrimitive(
                    id="et_1",
                    family="epochal_turns",
                    title="Turn",
                    summary="A decisive turn.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    support_passage_ids=["p2"],
                    geography="Delhi",
                    actor_ids=["actor_1"],
                )
            ],
            "telling_details": [
                BaseSynthesisPrimitive(
                    id="td_1",
                    family="telling_details",
                    title="Detail",
                    summary="A local detail.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
            ]
            + [
                BaseSynthesisPrimitive(
                    id=f"core_{idx}",
                    family="telling_details",
                    title=f"Core {idx}",
                    summary="Extra selected primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(2, 7)
            ]
            + [
                BaseSynthesisPrimitive(
                    id=f"support_{idx}",
                    family="telling_details",
                    title=f"Support {idx}",
                    summary="Support primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(1, 8)
            ],
        },
    )
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc",
                episode_spine={
                    "listener_question": "Question?",
                    "argument": "Claim",
                    "core_primitive_ids": [
                        "et_1",
                        "core_2",
                        "core_3",
                        "core_4",
                        "core_5",
                        "core_6",
                        "td_1",
                    ],
                    "support_primitive_roles": {
                        f"support_{idx}": "mechanism"
                        for idx in range(1, 8)
                    },
                    "recall_primitive_ids": [],
                },
            )
        ],
    )

    synthesis_map = asyncio.run(
        orchestrator._enrich_selected_primitives(
            project,
            synthesis_primitives,
            strategy,
            corpus,
            tmp_path,
            actor_metadata,
        )
    )

    payload = captured["payload"]
    assert payload["family"] == "epochal_turns"
    assert payload["base_primitives"][0]["id"] == "et_1"
    assert set(payload["passages_by_id"]) == {"p1", "p2"}
    assert "primitive_ids_by_role" not in payload

    assert isinstance(synthesis_map, SynthesisMap)
    assert [item.id for item in synthesis_map.primitives_by_family["epochal_turns"]] == ["et_1"]
    assert "td_1" in [item.id for item in synthesis_map.primitives_by_family["telling_details"]]
    epochal = synthesis_map.primitives_by_family["epochal_turns"][0]
    assert epochal.before_state == "Before"
    assert epochal.after_state == "After"


def test_primitive_enrichment_passages_include_support_and_trim_with_core_precedence() -> None:
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )
    primitives = [
        BaseSynthesisPrimitive(
            id="et_1",
            family="epochal_turns",
            title="Turn",
            summary="A decisive turn.",
            axis_ids=["axis_1"],
            core_passage_ids=["p1"],
            support_passage_ids=["p2"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
        BaseSynthesisPrimitive(
            id="et_2",
            family="epochal_turns",
            title="Second turn",
            summary="A related turn.",
            axis_ids=["axis_1"],
            core_passage_ids=[],
            support_passage_ids=["p1"],
            geography="Delhi",
            actor_ids=["actor_1"],
        ),
    ]
    passage_lookup = {
        "p1": ExtractedPassage(
            passage_id="p1",
            book_id="b1",
            chunk_ids=["c1"],
            text="Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one.",
            full_text="Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one.",
            chapter_ref="ch. 1",
            axis_id="axis_1",
        ),
        "p2": ExtractedPassage(
            passage_id="p2",
            book_id="b1",
            chunk_ids=["c2"],
            text="Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one.",
            full_text="Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one. Turn Delhi actor one.",
            chapter_ref="ch. 2",
            axis_id="axis_1",
        ),
    }

    passages_by_id = _build_primitive_enrichment_passages_by_id(
        family="epochal_turns",
        primitives=primitives,
        passage_lookup=passage_lookup,
        actor_metadata=actor_metadata,
    )

    assert set(passages_by_id) == {"p1", "p2"}
    assert len(_split_sentences(passages_by_id["p1"]["text"])) == 2
    assert len(_split_sentences(passages_by_id["p2"]["text"])) == 1


def test_enrich_selected_primitives_runs_family_batches_concurrently_with_cap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
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
        lambda settings: DummyTTSClient(),
    )

    orchestrator = PipelineOrchestrator()
    active = 0
    max_active = 0
    seen_families: list[str] = []
    lock = threading.Lock()

    family_delta_payloads = {
        "epochal_turns": {
            "before_state": "Before",
            "after_state": "After",
            "change_driver": "Driver",
            "irreversibility_reason": "Reason",
        },
        "decisions_and_nondecisions": {
            "actor_ids": ["actor_1"],
            "decision_question": "Should the ruler strike now or hold back?",
            "decision_mode": "decision",
            "options_considered": ["Act now", "Wait"],
            "immediate_consequence": "The choice redirects the next move.",
        },
        "set_piece_scenes": {
            "actor_ids": ["actor_1"],
            "scene_anchor": "A public confrontation turns the room.",
            "scene_outcome": "The visible result hardens the next phase.",
            "location": "Delhi",
        },
        "human_costs": {
            "actor_ids": [],
            "affected_group": "Urban households",
            "cost_type": "displacement",
            "lived_consequence": "Families lose shelter and local protection.",
            "visibility": "Visible on the ground and easy to ignore from above.",
        },
        "character_engines": {
            "actor_id": "actor_1",
            "goal": "Hold a fragile position.",
            "fear": "Losing command and legitimacy.",
            "constraint": "Rivals and institutions narrow the path.",
            "stakes": "Status and survival both depend on the outcome.",
        },
        "coalitions_and_fault_lines": {
            "actor_ids": ["actor_1"],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "coalition": "A tactical alignment of convenience.",
            "shared_interest": "Each side needs the other temporarily.",
            "fault_line": "Their longer-term aims diverge sharply.",
            "stress_point": "Pressure rises once immediate danger recedes.",
        },
    }

    def fake_run(payload: dict[str, Any]) -> Any:
        nonlocal active, max_active
        family = str(payload["family"])
        with lock:
            active += 1
            max_active = max(max_active, active)
            seen_families.append(family)
        time.sleep(0.1)
        try:
            delta = {
                "id": payload["base_primitives"][0]["id"],
                "family": family,
                **family_delta_payloads[family],
            }
            return orchestrator.primitive_enrichment_agent.response_model_for_family(
                family
            ).model_validate(
                {
                    "project_id": "proj",
                    "family": family,
                    "enriched_primitives": [delta],
                }
            )
        finally:
            with lock:
                active -= 1

    orchestrator.primitive_enrichment_agent.run = fake_run

    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description",
                theme_importance_score=1.0,
            )
        ],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="A passage with enough detail to support all selected primitives.",
                    full_text="A passage with enough detail to support all selected primitives.",
                    chapter_ref="ch. 1",
                    axis_id="axis_1",
                )
            ]
        },
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )
    synthesis_primitives = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives_by_family={
            "epochal_turns": [
                BaseSynthesisPrimitive(
                    id="et_1",
                    family="epochal_turns",
                    title="Turn",
                    summary="A decisive turn.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "decisions_and_nondecisions": [
                BaseSynthesisPrimitive(
                    id="dn_1",
                    family="decisions_and_nondecisions",
                    title="Decision",
                    summary="A consequential choice.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "set_piece_scenes": [
                BaseSynthesisPrimitive(
                    id="sp_1",
                    family="set_piece_scenes",
                    title="Scene",
                    summary="A major confrontation.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "human_costs": [
                BaseSynthesisPrimitive(
                    id="hc_1",
                    family="human_costs",
                    title="Cost",
                    summary="The rupture lands on households.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
            ],
            "character_engines": [
                BaseSynthesisPrimitive(
                    id="ce_1",
                    family="character_engines",
                    title="Engine",
                    summary="One actor is under pressure.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "coalitions_and_fault_lines": [
                BaseSynthesisPrimitive(
                    id="cf_1",
                    family="coalitions_and_fault_lines",
                    title="Coalition",
                    summary="An alliance carries strain.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "telling_details": [
                BaseSynthesisPrimitive(
                    id="td_1",
                    family="telling_details",
                    title="Detail",
                    summary="A local detail.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
            ]
            + [
                BaseSynthesisPrimitive(
                    id=f"support_{idx}",
                    family="telling_details",
                    title=f"Support {idx}",
                    summary="Support primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(1, 8)
            ],
        },
    )
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc",
                episode_spine={
                    "listener_question": "Question?",
                    "argument": "Claim",
                    "core_primitive_ids": [
                        "et_1",
                        "dn_1",
                        "sp_1",
                        "hc_1",
                        "ce_1",
                        "cf_1",
                        "td_1",
                    ],
                    "support_primitive_roles": {
                        f"support_{idx}": "mechanism"
                        for idx in range(1, 8)
                    },
                    "recall_primitive_ids": [],
                },
            )
        ],
    )

    synthesis_map = asyncio.run(
        orchestrator._enrich_selected_primitives(
            project,
            synthesis_primitives,
            strategy,
            corpus,
            tmp_path,
            actor_metadata,
        )
    )

    assert isinstance(synthesis_map, SynthesisMap)
    assert sorted(seen_families) == sorted(family_delta_payloads)
    assert max_active == 5


def test_enrich_selected_primitives_preserves_new_structured_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
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
        lambda settings: DummyTTSClient(),
    )

    orchestrator = PipelineOrchestrator()

    def fake_run(payload: dict[str, Any]) -> Any:
        family = str(payload["family"])
        delta = {"id": payload["base_primitives"][0]["id"], "family": family}
        if family == "decisions_and_nondecisions":
            delta.update(
                {
                    "actor_ids": ["actor_1"],
                    "decision_question": "Should the ruler force a confrontation now?",
                    "decision_mode": "decision",
                    "options_considered": ["Force the move", "Delay the confrontation"],
                    "immediate_consequence": "The court reorients around the choice.",
                }
            )
        elif family == "systems_and_operating_logics":
            delta.update(
                {
                    "system_name": "Court patronage",
                    "mechanism": "Orders and incentives move through patronage channels.",
                    "mechanism_steps": [
                        "Orders move downward through loyal intermediaries.",
                        "Provincial actors translate orders into local pressure.",
                    ],
                    "inputs": ["orders"],
                    "outputs": ["compliance"],
                    "failure_mode": "The chain distorts when local incentives diverge.",
                }
            )
        else:
            raise AssertionError(f"Unexpected family {family}")
        return orchestrator.primitive_enrichment_agent.response_model_for_family(
            family
        ).model_validate(
            {
                "project_id": "proj",
                "family": family,
                "enriched_primitives": [delta],
            }
        )

    orchestrator.primitive_enrichment_agent.run = fake_run

    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description",
                theme_importance_score=1.0,
            )
        ],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="A passage with enough detail to support selected primitives.",
                    full_text="A passage with enough detail to support selected primitives.",
                    chapter_ref="ch. 1",
                    axis_id="axis_1",
                )
            ]
        },
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="actor_1",
                display_name="Actor One",
                actor_type="person",
            )
        ],
    )
    synthesis_primitives = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives_by_family={
            "decisions_and_nondecisions": [
                BaseSynthesisPrimitive(
                    id="dn_1",
                    family="decisions_and_nondecisions",
                    title="Decision",
                    summary="A consequential choice.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "systems_and_operating_logics": [
                BaseSynthesisPrimitive(
                    id="sys_1",
                    family="systems_and_operating_logics",
                    title="System",
                    summary="An institutional chain channels pressure.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
            "telling_details": [
                BaseSynthesisPrimitive(
                    id=f"core_{idx}",
                    family="telling_details",
                    title=f"Core {idx}",
                    summary="Extra selected primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(2, 7)
            ]
            + [
                BaseSynthesisPrimitive(
                    id=f"support_{idx}",
                    family="telling_details",
                    title=f"Support {idx}",
                    summary="Support primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(1, 8)
            ],
        },
    )
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc",
                episode_spine={
                    "listener_question": "Question?",
                    "argument": "Claim",
                    "core_primitive_ids": [
                        "dn_1",
                        "sys_1",
                        "core_2",
                        "core_3",
                        "core_4",
                        "core_5",
                    ],
                    "support_primitive_roles": {
                        f"support_{idx}": "mechanism"
                        for idx in range(1, 8)
                    },
                    "recall_primitive_ids": [],
                },
            )
        ],
    )

    synthesis_map = asyncio.run(
        orchestrator._enrich_selected_primitives(
            project,
            synthesis_primitives,
            strategy,
            corpus,
            tmp_path,
            actor_metadata,
        )
    )

    decision = synthesis_map.primitives_by_family["decisions_and_nondecisions"][0]
    assert decision.decision_question == "Should the ruler force a confrontation now?"

    system = synthesis_map.primitives_by_family["systems_and_operating_logics"][0]
    assert system.mechanism_steps == [
        "Orders move downward through loyal intermediaries.",
        "Provincial actors translate orders into local pressure.",
    ]


def test_enrich_selected_primitives_allows_null_character_engine_actor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
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
        lambda settings: DummyTTSClient(),
    )

    orchestrator = PipelineOrchestrator()

    def fake_run(payload: dict[str, Any]) -> Any:
        family = str(payload["family"])
        if family != "character_engines":
            raise AssertionError(f"Unexpected family {family}")
        return orchestrator.primitive_enrichment_agent.response_model_for_family(
            family
        ).model_validate(
            {
                "project_id": "proj",
                "family": family,
                "enriched_primitives": [
                    {
                        "id": "ce_1",
                        "family": family,
                        "actor_id": None,
                        "goal": "Hold a fragile position.",
                        "fear": "Public loss of legitimacy.",
                        "constraint": "The institutions around the actor are unstable.",
                        "stakes": "Failure would dissolve the faction's leverage.",
                    }
                ],
            }
        )

    orchestrator.primitive_enrichment_agent.run = fake_run

    project = ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="b1",
                title="Book",
                author="Author",
                source_path="book.txt",
                source_type="text",
            )
        ],
        config=PipelineConfig(),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[
            ThematicAxis(
                axis_id="axis_1",
                name="Axis",
                description="Axis description",
                theme_importance_score=1.0,
            )
        ],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="A passage with enough detail to support the pressure profile.",
                    full_text="A passage with enough detail to support the pressure profile.",
                    chapter_ref="ch. 1",
                    axis_id="axis_1",
                )
            ]
        },
    )
    actor_metadata = ActorMetadata(project_id="proj", actors=[])
    synthesis_primitives = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives_by_family={
            "character_engines": [
                BaseSynthesisPrimitive(
                    id="ce_1",
                    family="character_engines",
                    title="Pressure without one clean owner",
                    summary="The pressure is clear even if one canonical actor is not.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
            ],
            "telling_details": [
                BaseSynthesisPrimitive(
                    id=f"core_{idx}",
                    family="telling_details",
                    title=f"Core {idx}",
                    summary="Extra selected primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(2, 8)
            ]
            + [
                BaseSynthesisPrimitive(
                    id=f"support_{idx}",
                    family="telling_details",
                    title=f"Support {idx}",
                    summary="Support primitive.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                )
                for idx in range(1, 8)
            ],
        },
    )
    strategy = NarrativeStrategy(
        strategy_type="chronological",
        justification="Test",
        series_arc="Arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                arc_summary="Arc",
                episode_spine={
                    "listener_question": "Question?",
                    "argument": "Claim",
                    "core_primitive_ids": [
                        "ce_1",
                        "core_2",
                        "core_3",
                        "core_4",
                        "core_5",
                        "core_6",
                    ],
                    "support_primitive_roles": {
                        f"support_{idx}": "mechanism"
                        for idx in range(1, 8)
                    },
                    "recall_primitive_ids": [],
                },
            )
        ],
    )

    synthesis_map = asyncio.run(
        orchestrator._enrich_selected_primitives(
            project,
            synthesis_primitives,
            strategy,
            corpus,
            tmp_path,
            actor_metadata,
        )
    )

    character_engine = synthesis_map.primitives_by_family["character_engines"][0]
    assert character_engine.actor_id is None
