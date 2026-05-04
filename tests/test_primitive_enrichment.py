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
    _build_primitive_enrichment_evidence_by_primitive_id,
    _split_sentences,
)
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorRelationship,
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


def _hooks() -> dict[str, str]:
    return {
        "concrete_detail": "A concrete detail lands.",
        "host_lens": "The pressure is visible.",
        "carry_forward": "The residue lingers.",
    }


def _generic_narration_only_result(
    orchestrator: PipelineOrchestrator,
    payload: dict[str, Any],
) -> Any:
    family = str(payload["family"])
    return orchestrator.primitive_enrichment_agent.response_model_for_family(
        family
    ).model_validate(
        {
            "project_id": "proj",
            "family": family,
            "enriched_primitives": [
                {
                    "id": primitive["id"],
                    "family": family,
                    "narration_hooks": _hooks(),
                }
                for primitive in payload["base_primitives"]
            ],
        }
    )


def _long_phrase(prefix: str, count: int) -> str:
    return " ".join([prefix, *[f"word{i}" for i in range(count - 1)]])


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
    captured_payloads: list[dict[str, Any]] = []

    def fake_run(payload: dict[str, Any]) -> Any:
        captured_payloads.append(payload)
        if payload["family"] == "telling_details":
            return _generic_narration_only_result(orchestrator, payload)
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
                        "proof_of_change": "The break is undeniable.",
                        "why_no_return": "Reason",
                        "narration_hooks": _hooks(),
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
            ),
            ActorProfile(
                actor_id="actor_2",
                display_name="Actor Two",
                actor_type="person",
            ),
        ],
        relationships=[
            ActorRelationship(
                source_actor_id="actor_1",
                target_actor_id="actor_2",
                relationship_type="pressures",
                description="Actor One pressures Actor Two.",
                confidence="high",
            ),
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
                    actor_ids=["actor_1", "actor_2"],
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

    payload = next(item for item in captured_payloads if item["family"] == "epochal_turns")
    assert payload["family"] == "epochal_turns"
    assert payload["base_primitives"][0]["id"] == "et_1"
    assert set(payload["evidence_by_primitive_id"]) == {"et_1"}
    assert [
        item["passage_id"]
        for item in payload["evidence_by_primitive_id"]["et_1"]["core_passages"]
    ] == ["p1"]
    assert [
        item["passage_id"]
        for item in payload["evidence_by_primitive_id"]["et_1"]["support_passages"]
    ] == ["p2"]
    assert payload["actor_metadata"] == {
        "actors": [
            {
                "actor_id": "actor_1",
                "display_name": "Actor One",
                "aliases": [],
                "actor_type": "person",
                "one_line_role": "person",
            },
            {
                "actor_id": "actor_2",
                "display_name": "Actor Two",
                "aliases": [],
                "actor_type": "person",
                "one_line_role": "person",
            },
        ]
    }
    assert "primitive_ids_by_role" not in payload

    assert isinstance(synthesis_map, SynthesisMap)
    assert [item.id for item in synthesis_map.primitives_by_family["epochal_turns"]] == ["et_1"]
    assert "td_1" in [item.id for item in synthesis_map.primitives_by_family["telling_details"]]
    epochal = synthesis_map.primitives_by_family["epochal_turns"][0]
    assert epochal.before_state == "Before"
    assert epochal.after_state == "After"


def test_enrich_selected_primitives_accepts_overlong_set_piece_scene_fields_after_merge(
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
    scene_anchor = _long_phrase("anchor", 22)
    hinge_action = _long_phrase("hinge", 21)
    scene_outcome = _long_phrase("outcome", 23)
    location = _long_phrase("location", 18)
    concrete_detail = _long_phrase("detail", 26)

    def fake_run(payload: dict[str, Any]) -> Any:
        if payload["family"] == "telling_details":
            return _generic_narration_only_result(orchestrator, payload)
        return orchestrator.primitive_enrichment_agent.response_model_for_family(
            "set_piece_scenes"
        ).model_validate(
            {
                "project_id": "proj",
                "family": "set_piece_scenes",
                "enriched_primitives": [
                    {
                        "id": "sp_1",
                        "family": "set_piece_scenes",
                        "actor_ids": ["actor_1"],
                        "scene_anchor": scene_anchor,
                        "hinge_action": hinge_action,
                        "scene_outcome": scene_outcome,
                        "location": location,
                        "narration_hooks": {
                            "concrete_detail": concrete_detail,
                            "host_lens": _long_phrase("lens", 35),
                            "carry_forward": _long_phrase("carry", 37),
                        },
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
                    text="A passage with enough scene detail to support a long-form enriched scene.",
                    full_text="A passage with enough scene detail to support a long-form enriched scene.",
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
            "set_piece_scenes": [
                BaseSynthesisPrimitive(
                    id="sp_1",
                    family="set_piece_scenes",
                    title="Scene",
                    summary="A public confrontation shifts visibly.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=["actor_1"],
                )
            ],
                "telling_details": [
                    BaseSynthesisPrimitive(
                        id=f"td_{idx}",
                        family="telling_details",
                        title=f"Detail {idx}",
                    summary="Selected filler primitive.",
                    axis_ids=["axis_1"],
                        core_passage_ids=["p1"],
                        actor_ids=[],
                    )
                    for idx in range(1, 10)
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
                    "core_primitive_ids": ["sp_1", "td_1", "td_2", "td_3", "td_4"],
                    "support_primitive_roles": {
                        "td_5": "mechanism",
                        "td_6": "texture",
                        "td_7": "stakes",
                        "td_8": "consequence",
                        "td_9": "texture",
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

    enriched = synthesis_map.primitives_by_family["set_piece_scenes"][0]
    assert enriched.scene_anchor == scene_anchor
    assert enriched.hinge_action == hinge_action
    assert enriched.scene_outcome == scene_outcome
    assert enriched.location == location
    assert enriched.narration_hooks.concrete_detail == concrete_detail


def test_primitive_enrichment_evidence_is_primitive_scoped_with_core_support_and_trim() -> None:
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

    evidence_by_primitive_id = _build_primitive_enrichment_evidence_by_primitive_id(
        family="epochal_turns",
        primitives=primitives,
        passage_lookup=passage_lookup,
        actor_metadata=actor_metadata,
    )

    assert set(evidence_by_primitive_id) == {"et_1", "et_2"}

    et_1 = evidence_by_primitive_id["et_1"]
    assert [item["passage_id"] for item in et_1["core_passages"]] == ["p1"]
    assert [item["passage_id"] for item in et_1["support_passages"]] == ["p2"]
    assert len(_split_sentences(et_1["core_passages"][0]["text"])) == 2
    assert len(_split_sentences(et_1["support_passages"][0]["text"])) == 1

    et_2 = evidence_by_primitive_id["et_2"]
    assert [item["passage_id"] for item in et_2["core_passages"]] == []
    assert [item["passage_id"] for item in et_2["support_passages"]] == ["p1"]
    assert len(_split_sentences(et_2["support_passages"][0]["text"])) == 1


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
            "proof_of_change": "The break is undeniable.",
            "why_no_return": "Reason",
            "narration_hooks": _hooks(),
        },
        "decisions_and_nondecisions": {
            "actor_ids": ["actor_1"],
            "decision_trigger": "The pressure spikes now.",
            "decision_question": "Should the ruler strike now or hold back?",
            "decision_mode": "decision",
            "options_considered": ["Act now", "Wait"],
            "next_result": "The choice redirects the next move.",
            "narration_hooks": _hooks(),
        },
        "set_piece_scenes": {
            "actor_ids": ["actor_1"],
            "scene_anchor": "A public confrontation turns the room.",
            "hinge_action": "The first move breaks the standoff.",
            "scene_outcome": "The visible result hardens the next phase.",
            "location": "Delhi",
            "narration_hooks": _hooks(),
        },
        "human_costs": {
            "actor_ids": [],
            "affected_group": "Urban households",
            "cost_type": "displacement",
            "concrete_marker": "Families carry bundles into the road.",
            "lived_consequence": "Families lose shelter and local protection.",
            "who_saw_it": "Visible on the ground and easy to ignore from above.",
            "narration_hooks": _hooks(),
        },
        "character_engines": {
            "actor_id": "actor_1",
            "goal": "Hold a fragile position.",
            "pressure_box": "Rivals and institutions narrow the path.",
            "risk_if_it_breaks": "Status and survival both depend on the outcome.",
            "tell": "He keeps insisting the gamble will work.",
            "narration_hooks": _hooks(),
        },
        "coalitions_and_fault_lines": {
            "actor_ids": ["actor_1"],
            "alignment_type": "tactical",
            "coalition_phase": "holding",
            "alignment_shape": "A tactical alignment of convenience.",
            "alignment_basis": "Each side needs the other temporarily.",
            "fracture_trigger": "Pressure rises once immediate danger recedes.",
            "narration_hooks": _hooks(),
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
            if family == "telling_details":
                return _generic_narration_only_result(orchestrator, payload)
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
    assert sorted(seen_families) == sorted([*family_delta_payloads, "telling_details"])
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
                    "decision_trigger": "The next crisis arrives.",
                    "decision_question": "Should the ruler force a confrontation now?",
                    "decision_mode": "decision",
                    "options_considered": ["Force the move", "Delay the confrontation"],
                    "next_result": "The court reorients around the choice.",
                    "narration_hooks": _hooks(),
                }
            )
        elif family == "systems_and_operating_logics":
            delta.update(
                {
                    "system_name": "Court patronage",
                    "operating_chain": [
                        "Orders move downward through loyal intermediaries.",
                        "Provincial actors translate orders into local pressure.",
                    ],
                    "inputs": ["orders"],
                    "outputs": ["compliance"],
                    "where_it_shows_up": "Officials enforce it face to face.",
                    "failure_mode": "The chain distorts when local incentives diverge.",
                    "narration_hooks": _hooks(),
                }
            )
        elif family == "telling_details":
            return _generic_narration_only_result(orchestrator, payload)
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
    assert system.operating_chain == [
        "Orders move downward through loyal intermediaries.",
        "Provincial actors translate orders into local pressure.",
    ]


def test_enrich_selected_primitives_retries_family_when_selected_primitive_is_omitted(
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
    monkeypatch.setattr("podcast_agent.agents.primitive_enrichment.time.sleep", lambda _seconds: None)

    orchestrator = PipelineOrchestrator()
    call_counts: dict[str, int] = {}

    def fake_generate_json(**kwargs: Any) -> Any:
        payload = kwargs["payload"]
        family = str(payload["family"])
        call_counts[family] = call_counts.get(family, 0) + 1
        if family == "systems_and_operating_logics":
            if call_counts[family] == 1:
                assert "enrichment_feedback" not in payload
                enriched_primitives = [
                    {
                        "id": "sys_1",
                        "family": family,
                        "system_name": "Court patronage",
                        "operating_chain": [
                            "Orders move downward through loyal intermediaries.",
                            "Provincial actors translate orders into local pressure.",
                        ],
                        "inputs": ["orders"],
                        "outputs": ["compliance"],
                        "where_it_shows_up": "Officials enforce it face to face.",
                        "failure_mode": "The chain distorts when local incentives diverge.",
                        "narration_hooks": _hooks(),
                    }
                ]
            else:
                feedback = payload["enrichment_feedback"]
                assert feedback["issue"] == "missing_selected_primitives"
                assert feedback["missing_primitive_ids"] == ["sys_2"]
                enriched_primitives = [
                    {
                        "id": "sys_1",
                        "family": family,
                        "system_name": "Court patronage",
                        "operating_chain": [
                            "Orders move downward through loyal intermediaries.",
                            "Provincial actors translate orders into local pressure.",
                        ],
                        "inputs": ["orders"],
                        "outputs": ["compliance"],
                        "where_it_shows_up": "Officials enforce it face to face.",
                        "failure_mode": "The chain distorts when local incentives diverge.",
                        "narration_hooks": _hooks(),
                    },
                    {
                        "id": "sys_2",
                        "family": family,
                        "system_name": "Ritualized protest cadence",
                        "operating_chain": [
                            "Mourning days set the protest calendar.",
                            "Security violence seeds the next observance.",
                        ],
                        "inputs": ["mourning networks"],
                        "outputs": ["repeat mobilization"],
                        "where_it_shows_up": "Crowds reassemble on each fortieth day.",
                        "failure_mode": "The chain breaks if repression stops producing martyrs.",
                        "narration_hooks": _hooks(),
                    },
                ]
        elif family == "telling_details":
            return orchestrator.primitive_enrichment_agent.response_model_for_family(
                family
            ).model_validate(
                {
                    "project_id": "proj",
                    "family": family,
                    "enriched_primitives": [
                        {
                            "id": primitive["id"],
                            "family": family,
                            "narration_hooks": _hooks(),
                        }
                        for primitive in payload["base_primitives"]
                    ],
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
                "enriched_primitives": enriched_primitives,
            }
        )

    orchestrator.primitive_enrichment_agent.llm.generate_json = fake_generate_json

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
    actor_metadata = ActorMetadata(project_id="proj")
    synthesis_primitives = SynthesisPrimitivesArtifact(
        project_id="proj",
        primitives_by_family={
            "systems_and_operating_logics": [
                BaseSynthesisPrimitive(
                    id="sys_1",
                    family="systems_and_operating_logics",
                    title="System One",
                    summary="The first operating chain.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                ),
                BaseSynthesisPrimitive(
                    id="sys_2",
                    family="systems_and_operating_logics",
                    title="System Two",
                    summary="The second operating chain.",
                    axis_ids=["axis_1"],
                    core_passage_ids=["p1"],
                    actor_ids=[],
                ),
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
                        "sys_1",
                        "sys_2",
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

    assert call_counts["systems_and_operating_logics"] == 2
    assert [item.id for item in synthesis_map.primitives_by_family["systems_and_operating_logics"]] == [
        "sys_1",
        "sys_2",
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
        if family == "telling_details":
            return _generic_narration_only_result(orchestrator, payload)
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
                        "pressure_box": "The institutions around the actor are unstable.",
                        "risk_if_it_breaks": "Failure would dissolve the faction's leverage.",
                        "tell": "The actor keeps returning to the same justification.",
                        "narration_hooks": _hooks(),
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
