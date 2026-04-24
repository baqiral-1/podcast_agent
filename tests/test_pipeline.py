"""Focused tests for active orchestrator helpers in the redesigned pipeline."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _allocate_synthesis_passages_by_axis,
    _allocate_scene_durations,
    _build_scene_card_family_warnings,
    _compute_scene_word_count_targets,
    _build_scene_card_count_warnings,
    _build_scene_card_primitive_warnings,
    _build_passage_lookup,
    _estimate_duration_seconds_from_words,
    _flatten_synthesis_primitives,
    _resolve_synthesis_bm25_keep_fraction_by_passage,
    _round_allocations_to_total,
    _scene_duration_bounds,
    _scene_importance_weight,
    _split_sentences,
    _script_total_word_count,
    _trim_candidate_texts_by_bm25,
    _trim_candidate_texts_by_bm25_query_text,
)
from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcThread,
    ActorMetadata,
    ActorProfile,
    BookRecord,
    ClusterPathOccurrence,
    EpisodeCandidateCluster,
    EpisodeScript,
    EpisodePlan,
    ExtractedPassage,
    FramingBlock,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    ProseSection,
    SceneActor,
    SceneActorArcBinding,
    SceneCard,
    SpokenScript,
    SpokenSection,
    StrategyEpisode,
    SYNTHESIS_PRIMITIVE_FAMILIES,
    SynthesisConsolidationResult,
    SynthesisPrimitive,
    SynthesisPrimitivesArtifact,
    SynthesisMap,
    ThematicCorpus,
    ThematicAxis,
    ThematicProject,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="Image",
        threat_or_unresolved_action="Threat",
        opening_question="Question",
        handoff_scene_card_id="scene_1",
    )


def _strategy_episode(*occurrences: ClusterPathOccurrence) -> StrategyEpisode:
    return StrategyEpisode(
        episode_number=1,
        title="Episode",
        driving_question="Question?",
        arc_summary="Arc",
        cluster_path=list(occurrences),
    )


def _scene_card(
    scene_id: str,
    occurrence_id: str,
    *,
    scene_role: str = "action",
    actors: list[SceneActor] | None = None,
) -> SceneCard:
    return SceneCard(
        scene_id=scene_id,
        title=scene_id,
        scene_role=scene_role,
        dominant_cluster_occurrence_id=occurrence_id,
        entry_image="Image",
        local_question="Question",
        observable_detail="Detail",
        intended_move="Move",
        passage_ids=["p1"],
        actors=list(actors or []),
        estimated_duration_seconds=120,
    )


def _scene_actor(
    *,
    name: str = "Actor",
    presence: str = "secondary",
    binding_weight: str | None = None,
) -> SceneActor:
    bindings = []
    if binding_weight is not None:
        bindings.append(
            SceneActorArcBinding(
                thread_id="thread_1",
                scene_role="driver",
                scene_use="develop",
                weight=binding_weight,
            )
        )
    return SceneActor(name=name, presence=presence, arc_bindings=bindings)


def _load_middle_east_v2_episode_fixtures() -> list[tuple[StrategyEpisode, EpisodePlan]]:
    run_dir = Path("runs/middle_east_v2")
    strategy_payload = json.loads((run_dir / "narrative_strategy.json").read_text())
    plan_payload = json.loads((run_dir / "series_plan.json").read_text())
    strategy_by_episode = {
        episode["episode_number"]: StrategyEpisode.model_validate(episode)
        for episode in strategy_payload["episodes"]
    }
    return [
        (
            strategy_by_episode[episode["episode_number"]],
            EpisodePlan.model_validate(episode),
        )
        for episode in plan_payload["episodes"]
    ]


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def _primitive(primitive_id: str, title: str = "Title") -> SynthesisPrimitive:
    match = primitive_id.rsplit("_", 1)
    suffix = match[1] if len(match) == 2 and match[1].isdigit() else "1"
    return SynthesisPrimitive(
        id=primitive_id,
        title=title,
        summary="Summary",
        core_passage_ids=[f"p{suffix}"],
    )


def _actor_arc_directive(actor_id: str = "actor_primary") -> ActorArcDirective:
    return ActorArcDirective(
        actor_id=actor_id,
        arc_threads=[
            ActorArcThread(
                thread_id=f"{actor_id}_role_1",
                arc_type="role",
                label="episode role",
                premise="Central to this episode.",
                pressure="Pressure narrows choices.",
                movement="Track a fragile goal.",
                payoff="Use where pressure becomes consequence.",
            ),
            ActorArcThread(
                thread_id=f"{actor_id}_guardrail_1",
                arc_type="guardrail",
                label="repetition guardrail",
                premise="Do not repeat without change.",
                pressure="Repetition can flatten the actor.",
                movement="Vary the actor function by scene.",
                payoff="Continuity remains legible without repetition.",
            ),
        ],
    )


def _primitives_by_family(
    **overrides: list[SynthesisPrimitive],
) -> dict[str, list[SynthesisPrimitive]]:
    payload = {family: [] for family in SYNTHESIS_PRIMITIVE_FAMILIES}
    payload.update(overrides)
    return payload


def test_build_scene_card_count_warnings_for_under_target():
    warnings = _build_scene_card_count_warnings(
        scene_card_count=18,
        scene_card_target_min=25,
        scene_card_target_max=40,
    )
    assert len(warnings) == 1
    assert warnings[0].startswith("scene_card_count_below_target")


def test_build_scene_card_primitive_warnings_reports_density_and_unknown_ids():
    cards = [
        SceneCard(
            scene_id="scene_1",
            title="Scene 1",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p1"],
            estimated_duration_seconds=60,
        ),
        SceneCard(
            scene_id="scene_2",
            title="Scene 2",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=["tp_1", "tp_2", "tp_3"],
            passage_ids=["p2"],
            estimated_duration_seconds=90,
        ),
    ]
    warnings = _build_scene_card_primitive_warnings(
        scene_cards=cards,
        primitive_pool_ids={"tp_1", "tp_2"},
        primitive_min=1,
        primitive_max=2,
    )
    assert any(warning.startswith("scene_primitive_density_out_of_range") for warning in warnings)
    assert any(warning.startswith("scene_card_unknown_primitive_ids") for warning in warnings)


def test_script_total_word_count_counts_sections():
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="One two three",
            )
        ],
    )
    assert _script_total_word_count(script) == 3


def test_estimate_duration_seconds_from_words_handles_zero_rate():
    assert _estimate_duration_seconds_from_words(120, 120) == 60
    assert _estimate_duration_seconds_from_words(120, 0) == 0


def test_compute_scene_word_count_targets_uses_scene_durations():
    scenes = [
        SceneCard(
            scene_id="scene_1",
            title="Scene 1",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p1"],
            estimated_duration_seconds=30,
        ),
        SceneCard(
            scene_id="scene_2",
            title="Scene 2",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            primitive_ids=[],
            passage_ids=["p2"],
            estimated_duration_seconds=90,
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=1000, words_per_minute=120.0)
    assert targets == {"scene_1": 60, "scene_2": 180}


def test_compute_scene_word_count_targets_scales_with_words_per_minute():
    scenes = [
        SceneCard(
            scene_id="scene_anchor",
            title="Anchor",
            scene_role="setup",
            dominant_cluster_occurrence_id="occ_1",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            passage_ids=["p1"],
            estimated_duration_seconds=30,
        ),
        SceneCard(
            scene_id="scene_compressed",
            title="Compressed",
            scene_role="reaction",
            dominant_cluster_occurrence_id="occ_2",
            entry_image="Image",
            local_question="Question",
            observable_detail="Detail",
            intended_move="Move",
            passage_ids=["p2"],
            estimated_duration_seconds=90,
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=1200, words_per_minute=140.0)
    assert targets == {"scene_anchor": 70, "scene_compressed": 210}


def test_compute_scene_word_count_targets_requires_positive_scene_durations():
    valid_scene = SceneCard(
        scene_id="scene_1",
        title="Scene 1",
        scene_role="setup",
        dominant_cluster_occurrence_id="occ_1",
        entry_image="Image",
        local_question="Question",
        observable_detail="Detail",
        intended_move="Move",
        passage_ids=["p1"],
        estimated_duration_seconds=60,
    )
    scenes = [
        valid_scene,
        valid_scene.model_copy(
            update={
                "scene_id": "scene_2",
                "title": "Scene 2",
                "scene_role": "reaction",
                "dominant_cluster_occurrence_id": "occ_2",
                "passage_ids": ["p2"],
                "estimated_duration_seconds": 0,
            }
        ),
    ]

    with pytest.raises(ValueError, match="scene_2"):
        _compute_scene_word_count_targets(
            scenes,
            episode_target_word_count=1200,
            words_per_minute=120.0,
        )


def test_scene_importance_weight_three_factor():
    scene = _scene_card(
        "scene_1",
        "occ_1",
        scene_role="shock",
        actors=[_scene_actor(presence="primary", binding_weight="strong")],
    )

    assert _scene_importance_weight(scene) == pytest.approx(1.30 * 1.50 * 1.20)


def test_scene_duration_bounds_no_actor_by_role_bucket():
    episode = _strategy_episode(
        ClusterPathOccurrence(
            occurrence_id="occ_1",
            cluster_id="cluster_1",
            usage="primary",
            emphasis="anchor",
        )
    )

    assert _scene_duration_bounds(
        _scene_card("argument", "occ_1", scene_role="synthesis"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((40.5, 81.0))
    assert _scene_duration_bounds(
        _scene_card("action", "occ_1", scene_role="consequence"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((90.0, 180.0))
    assert _scene_duration_bounds(
        _scene_card("mid", "occ_1", scene_role="setup"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((67.5, 135.0))


def test_scene_duration_bounds_scales_with_avg_sec():
    episode = _strategy_episode(
        ClusterPathOccurrence(
            occurrence_id="occ_1",
            cluster_id="cluster_1",
            usage="primary",
            emphasis="anchor",
        )
    )
    scene = _scene_card(
        "scene_1",
        "occ_1",
        actors=[_scene_actor(presence="primary", binding_weight="strong")],
    )

    short_bounds = _scene_duration_bounds(scene, episode, avg_sec=100.0)
    long_bounds = _scene_duration_bounds(scene, episode, avg_sec=250.0)

    assert long_bounds[0] == pytest.approx(short_bounds[0] * 2.5)
    assert long_bounds[1] == pytest.approx(short_bounds[1] * 2.5)


def test_allocate_scene_durations_spillover_goes_only_to_anchor_and_major():
    episode = _strategy_episode(
        ClusterPathOccurrence(
            occurrence_id="occ_anchor",
            cluster_id="cluster_anchor",
            usage="primary",
            emphasis="anchor",
        ),
        ClusterPathOccurrence(
            occurrence_id="occ_major",
            cluster_id="cluster_major",
            usage="primary",
            emphasis="major",
            transition_note="Then shift.",
        ),
        ClusterPathOccurrence(
            occurrence_id="occ_supporting",
            cluster_id="cluster_supporting",
            usage="primary",
            emphasis="supporting",
            transition_note="Then shift.",
        ),
    )

    allocated = _allocate_scene_durations(
        [
            _scene_card("anchor_scene", "occ_anchor"),
            _scene_card("major_scene", "occ_major"),
            _scene_card("support_scene", "occ_supporting"),
        ],
        episode,
        target_duration_seconds=600,
    )
    seconds = {scene.scene_id: scene.estimated_duration_seconds for scene in allocated}

    assert sum(seconds.values()) == 600
    assert seconds["support_scene"] == 120
    assert seconds["anchor_scene"] > seconds["major_scene"] > seconds["support_scene"]
    assert seconds["anchor_scene"] + seconds["major_scene"] == 480


def test_allocate_scene_durations_middle_east_v2_ep1_rebalances_retarged_roles():
    episode, plan = _load_middle_east_v2_episode_fixtures()[0]
    role_remap = {
        "e1_s06": "synthesis",
        "e1_s21": "action",
        "e1_s26": "synthesis",
        "e1_s31": "action",
    }
    remapped_scene_cards = [
        scene.model_copy(update={"scene_role": role_remap.get(scene.scene_id, scene.scene_role)})
        for scene in plan.scene_cards
    ]
    allocated = _allocate_scene_durations(
        remapped_scene_cards,
        episode,
        target_duration_seconds=int(plan.target_duration_minutes * 60),
    )
    seconds = {scene.scene_id: scene.estimated_duration_seconds for scene in allocated}

    assert sum(seconds.values()) == 6000
    assert seconds["e1_s07"] == pytest.approx(90, abs=10)
    assert seconds["e1_s06"] == pytest.approx(90, abs=10)
    assert seconds["e1_s26"] == pytest.approx(105, abs=15)
    assert seconds["e1_s36"] == pytest.approx(65, abs=10)
    assert seconds["e1_s21"] == pytest.approx(215, abs=20)
    assert seconds["e1_s34"] == pytest.approx(175, abs=20)
    assert seconds["e1_s26"] < seconds["e1_s21"]
    assert seconds["e1_s36"] < seconds["e1_s34"]
    assert seconds["e1_s33"] > seconds["e1_s36"]


def test_allocate_scene_durations_totals_sum_to_target():
    for episode, plan in _load_middle_east_v2_episode_fixtures():
        target_duration_seconds = int(plan.target_duration_minutes * 60)
        allocated = _allocate_scene_durations(
            plan.scene_cards,
            episode,
            target_duration_seconds=target_duration_seconds,
        )

        assert len(allocated) == len(plan.scene_cards)
        assert sum(scene.estimated_duration_seconds for scene in allocated) == target_duration_seconds


def test_round_allocations_to_total_handles_large_positive_delta():
    rounded = _round_allocations_to_total([1.1, 2.1, 3.1], 25)

    assert sum(rounded) == 25
    assert rounded == [8, 8, 9]


def test_trim_candidate_texts_by_bm25_keeps_one_quarter_of_sentences():
    axis = ThematicAxis(
        axis_id="axis_1",
        name="Query",
        description="Axis description",
        theme_importance_score=0.7,
    )
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "One sentence. Two sentence. Three sentence. "
                "Four sentence. Five sentence. Six sentence."
            ),
        },
        {
            "passage_id": "p2",
            "book_id": "b1",
            "text": "Alpha sentence. Beta sentence.",
        },
    ]

    _trim_candidate_texts_by_bm25(axis, candidates, keep_fraction=0.25)

    assert len(_split_sentences(candidates[0]["text"])) == 2
    assert len(_split_sentences(candidates[1]["text"])) == 1


def test_resolve_synthesis_bm25_keep_fraction_by_passage_uses_relevance_tiers():
    passages = [
        ExtractedPassage(
            passage_id=f"p{idx}",
            book_id="b1",
            chunk_ids=[f"c{idx}"],
            text=f"Passage {idx}",
            axis_id="axis_1",
            relevance_score=1.0 - (0.01 * idx),
            quotability_score=0.5,
        )
        for idx in range(1, 11)
    ]

    keep_fraction_by_passage_id, tier_counts = _resolve_synthesis_bm25_keep_fraction_by_passage(passages)

    assert tier_counts == {
        "top_tier_passages": 1,
        "mid_tier_passages": 2,
        "tail_tier_passages": 7,
    }
    assert keep_fraction_by_passage_id["p1"] == 0.5
    assert keep_fraction_by_passage_id["p2"] == 0.4
    assert keep_fraction_by_passage_id["p3"] == 0.4
    assert keep_fraction_by_passage_id["p4"] == 0.3
    assert keep_fraction_by_passage_id["p5"] == 0.3
    assert keep_fraction_by_passage_id["p6"] == 0.3
    assert keep_fraction_by_passage_id["p7"] == 0.3
    assert keep_fraction_by_passage_id["p8"] == 0.3
    assert keep_fraction_by_passage_id["p9"] == 0.3
    assert keep_fraction_by_passage_id["p10"] == 0.3


def test_allocate_synthesis_passages_by_axis_uses_dynamic_floor_global_refill_and_exact_dedupe():
    axis_1 = ThematicAxis(axis_id="axis_1", name="Axis 1", description="A1", theme_importance_score=0.95)
    axis_2 = ThematicAxis(axis_id="axis_2", name="Axis 2", description="A2", theme_importance_score=0.75)
    axis_3 = ThematicAxis(axis_id="axis_3", name="Axis 3", description="A3", theme_importance_score=0.65)
    selected_by_axis, cap_report = _allocate_synthesis_passages_by_axis(
        axes=[axis_1, axis_2, axis_3],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(passage_id="p1", book_id="b1", chunk_ids=["c1"], text="P1", axis_id="axis_1", relevance_score=0.92, quotability_score=0.8),
                ExtractedPassage(passage_id="p2", book_id="b1", chunk_ids=["c2"], text="P2", axis_id="axis_1", relevance_score=0.55, quotability_score=0.5),
            ],
            "axis_2": [
                ExtractedPassage(passage_id="p3", book_id="b1", chunk_ids=["c1"], text="P3 dup", axis_id="axis_2", relevance_score=0.99, quotability_score=0.95),
                ExtractedPassage(passage_id="p4", book_id="b1", chunk_ids=["c4"], text="P4", axis_id="axis_2", relevance_score=0.78, quotability_score=0.7),
                ExtractedPassage(passage_id="p5", book_id="b1", chunk_ids=["c5"], text="P5", axis_id="axis_2", relevance_score=0.4, quotability_score=0.4),
            ],
            "axis_3": [
                ExtractedPassage(passage_id="p6", book_id="b1", chunk_ids=["c6"], text="P6", axis_id="axis_3", relevance_score=0.88, quotability_score=0.82),
                ExtractedPassage(passage_id="p7", book_id="b1", chunk_ids=["c7"], text="P7", axis_id="axis_3", relevance_score=0.87, quotability_score=0.83),
            ],
        },
        total_cap=5,
        importance_power=1.3,
        cross_pair_ids=set(),
        floor_budget_fraction=0.25,
        axis_floor_min=1,
        axis_floor_max=1,
        axis_ceiling_multiplier=1.2,
    )

    all_ids = [
        passage.passage_id
        for passages in selected_by_axis.values()
        for passage in passages
    ]
    assert all_ids == ["p1", "p2", "p4", "p6", "p7"]
    assert len(set(all_ids)) == len(all_ids)
    assert cap_report["axis_floor"] == 1
    assert cap_report["axis_ceiling"] == 2
    assert cap_report["per_axis_counts_before_refill"] == {
        "axis_1": 1,
        "axis_2": 1,
        "axis_3": 1,
    }
    assert cap_report["per_axis_counts_after_refill"] == {
        "axis_1": 2,
        "axis_2": 1,
        "axis_3": 2,
    }
    assert cap_report["duplicate_source_candidates_skipped"] == 1


def test_allocate_synthesis_passages_by_axis_raises_on_duplicate_passage_ids():
    axis_1 = ThematicAxis(axis_id="axis_1", name="Axis 1", description="A1", theme_importance_score=0.95)
    axis_2 = ThematicAxis(axis_id="axis_2", name="Axis 2", description="A2", theme_importance_score=0.75)

    with pytest.raises(ValueError, match="Duplicate synthesis passages detected"):
        _allocate_synthesis_passages_by_axis(
            axes=[axis_1, axis_2],
            passages_by_axis={
                "axis_1": [
                    ExtractedPassage(passage_id="dup", book_id="b1", chunk_ids=["c1"], text="P1", axis_id="axis_1", relevance_score=0.92, quotability_score=0.8),
                ],
                "axis_2": [
                    ExtractedPassage(passage_id="dup", book_id="b2", chunk_ids=["c2"], text="P2", axis_id="axis_2", relevance_score=0.91, quotability_score=0.8),
                ],
            },
            total_cap=2,
            importance_power=1.3,
            cross_pair_ids=set(),
            floor_budget_fraction=0.25,
            axis_floor_min=1,
            axis_floor_max=1,
            axis_ceiling_multiplier=1.2,
        )


def test_trim_candidate_texts_by_bm25_supports_per_passage_keep_fractions():
    axis = ThematicAxis(
        axis_id="axis_1",
        name="Query",
        description="Axis description",
        theme_importance_score=0.7,
    )
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "One sentence. Two sentence. Three sentence. "
                "Four sentence. Five sentence. Six sentence."
            ),
        },
        {
            "passage_id": "p2",
            "book_id": "b1",
            "text": (
                "Alpha sentence. Beta sentence. Gamma sentence. "
                "Delta sentence. Epsilon sentence. Zeta sentence."
            ),
        },
    ]

    _trim_candidate_texts_by_bm25(
        axis,
        candidates,
        keep_fraction=0.25,
        keep_fraction_by_passage_id={"p1": 0.5, "p2": 0.25},
    )

    assert len(_split_sentences(candidates[0]["text"])) == 3
    assert len(_split_sentences(candidates[1]["text"])) == 2


def test_trim_candidate_texts_by_bm25_query_text_uses_explicit_query():
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "Keepalpha sentence one. Keepalpha sentence two. "
                "Dropgamma sentence three. Dropgamma sentence four."
            ),
        }
    ]

    _trim_candidate_texts_by_bm25_query_text(
        "keepalpha",
        candidates,
        keep_fraction=0.5,
    )

    kept = _split_sentences(candidates[0]["text"])
    assert kept == [
        "Keepalpha sentence one.",
        "Keepalpha sentence two.",
    ]


def test_build_passage_lookup_and_flatten_primitives():
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Text",
                    axis_id="axis_1",
                )
            ]
        },
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        primitives_by_family=_primitives_by_family(
            epochal_turns=[_primitive("tp_1", "Turn")]
        ),
    )
    assert _build_passage_lookup(corpus)["p1"].book_id == "b1"
    assert _flatten_synthesis_primitives(synthesis_map)["tp_1"].title == "Turn"


def test_orchestrator_initializes_redesigned_agents(monkeypatch):
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

    assert orchestrator.synthesis_primitives_agent.schema_name == "synthesis_primitives"
    assert orchestrator.synthesis_consolidation_agent.schema_name == "synthesis_consolidation"
    assert not hasattr(orchestrator, "synthesis_mapping_agent")


def test_build_scene_card_family_warnings_reports_missing_mix():
    episode = StrategyEpisode(
        episode_number=1,
        title="Episode 1",
        driving_question="Q1",
        arc_summary="A1",
        cluster_path=[
            ClusterPathOccurrence(
                occurrence_id="occ_1",
                cluster_id="cluster_1",
                usage="primary",
                emphasis="anchor",
            ),
            ClusterPathOccurrence(
                occurrence_id="occ_2",
                cluster_id="cluster_2",
                usage="echo",
                emphasis="major",
                transition_note="Shift to the next packet.",
            ),
        ],
    )

    warnings = _build_scene_card_family_warnings(
        episode=episode,
        primitive_pool_ids={"et_1"},
        primitive_family_by_id={"et_1": "epochal_turns"},
    )

    assert "primitive_family_missing_scene_or_detail" in warnings[0] or any(
        warning.startswith("primitive_family_missing_scene_or_detail")
        for warning in warnings
    )
    assert any(
        warning.startswith("primitive_family_missing_human_grounding")
        for warning in warnings
    )
    assert any(
        warning.startswith("primitive_family_missing_system_or_context")
        for warning in warnings
    )
    assert any(
        warning.startswith("primitive_family_missing_recurrence")
        for warning in warnings
    )


def test_map_synthesis_caps_total_passages_and_keeps_cross_pair_priority(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_primitives_run(payload: dict):
        captured["payload"] = payload
        return SynthesisPrimitivesArtifact(project_id="proj")

    orchestrator.synthesis_primitives_agent.run = fake_primitives_run
    orchestrator.synthesis_consolidation_agent.run = lambda payload: SynthesisConsolidationResult(project_id="proj")

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        config=PipelineConfig(
            synthesis_total_passage_cap=4,
            synthesis_axis_pct=1.0,
            synthesis_axis_min=2,
            synthesis_axis_max=2,
            synthesis_floor_budget_fraction=0.25,
            synthesis_axis_floor_min=1,
            synthesis_axis_floor_max=1,
            synthesis_axis_ceiling_multiplier=1.5,
        ),
    )
    axis_1 = ThematicAxis(axis_id="axis_1", name="Axis 1", description="A1", theme_importance_score=0.95)
    axis_2 = ThematicAxis(axis_id="axis_2", name="Axis 2", description="A2", theme_importance_score=0.45)
    corpus = ThematicCorpus(
        project_id="proj",
        axes=[axis_1, axis_2],
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(passage_id="p1", book_id="b1", chunk_ids=["c1"], text="P1", axis_id="axis_1", relevance_score=0.9, quotability_score=0.8),
                ExtractedPassage(passage_id="p2", book_id="b1", chunk_ids=["c2"], text="P2", axis_id="axis_1", relevance_score=0.7, quotability_score=0.7),
                ExtractedPassage(passage_id="p3", book_id="b1", chunk_ids=["c3"], text="P3", axis_id="axis_1", relevance_score=0.6, quotability_score=0.6),
            ],
            "axis_2": [
                ExtractedPassage(passage_id="p4", book_id="b1", chunk_ids=["c4"], text="P4", axis_id="axis_2", relevance_score=0.95, quotability_score=0.9),
                ExtractedPassage(passage_id="p5", book_id="b1", chunk_ids=["c5"], text="P5", axis_id="axis_2", relevance_score=0.85, quotability_score=0.8),
                ExtractedPassage(passage_id="p6", book_id="b1", chunk_ids=["c6"], text="P6", axis_id="axis_2", relevance_score=0.1, quotability_score=0.1),
                ExtractedPassage(passage_id="p7", book_id="b1", chunk_ids=["c1"], text="P7 duplicate chunk", axis_id="axis_2", relevance_score=0.99, quotability_score=0.95),
            ],
        },
        cross_book_pairs=[
            PassagePair(
                passage_a_id="p1",
                passage_b_id="p6",
                relationship="contradicts",
                strength=0.9,
                axis_id="axis_2",
            )
        ],
    )

    asyncio.run(orchestrator._map_synthesis(project, corpus, tmp_path))

    payload = captured["payload"]
    passages = payload["passages_by_axis"]
    for axis_books in passages.values():
        for book_group in axis_books:
            assert set(book_group.keys()) == {"book_id", "passages"}
            for item in book_group["passages"]:
                assert set(item.keys()) == {"passage_id", "text"}
    all_ids = [
        item["passage_id"]
        for axis_books in passages.values()
        for book_group in axis_books
        for item in book_group["passages"]
    ]
    assert len(all_ids) == 4
    assert len(set(all_ids)) == 4
    assert "p1" in all_ids
    assert "p6" in all_ids
    assert "p7" not in all_ids
    assert payload["cross_book_pairs"] == [
        {
            "passage_a_id": "p1",
            "passage_b_id": "p6",
            "relationship": "contradicts",
            "strength": 0.9,
        }
    ]


def test_write_episode_passes_full_text_to_writing_agent(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_writing_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                    }
                ]
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "title": "Scene 1",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 300,
                }
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for writing.",
                    axis_id="axis_1",
                )
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    payload = captured["payload"]
    assert payload["passages"][0]["text"] == "Full text evidence for writing."
    assert payload["episode_target_word_count_lower"] == 550
    assert payload["episode_target_word_count_higher"] == 650


def test_write_episode_uses_single_writing_call_for_many_scene_cards(monkeypatch, tmp_path):
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
    captured_payloads: list[dict] = []

    def fake_writing_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": [scene["scene_id"] for scene in payload["plan"]["scene_cards"]],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                    }
                ]
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": f"scene_{idx}",
                    "title": f"Scene {idx}",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": f"occ_{idx}",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": [f"p{idx}"],
                    "estimated_duration_seconds": 300,
                }
                for idx in range(1, 27)
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id=f"p{idx}",
                    book_id="b1",
                    chunk_ids=[f"c{idx}"],
                    text=f"Trimmed text {idx}",
                    full_text=f"Full text evidence {idx}.",
                    axis_id="axis_1",
                )
                for idx in range(1, 27)
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert len(payload["passages"]) == 26
    assert payload["episode_target_word_count_lower"] == 14300
    assert payload["episode_target_word_count_higher"] == 16900
    assert "previous_sections" not in payload
    payload_scene_cards = payload["plan"]["scene_cards"]
    assert [scene["scene_id"] for scene in payload_scene_cards] == [
        f"scene_{idx}" for idx in range(1, 27)
    ]
    assert payload["plan"]["framing"]["handoff_scene_card_id"] == "scene_1"
    assert all("estimated_duration_seconds" not in scene for scene in payload_scene_cards)
    assert all(scene["target_word_count_lower"] == 550 for scene in payload_scene_cards)
    assert all(scene["target_word_count_higher"] == 650 for scene in payload_scene_cards)


def test_write_episode_payload_uses_unequal_scene_duration_targets(monkeypatch, tmp_path):
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
    captured: dict[str, dict] = {}

    def fake_writing_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": [scene["scene_id"] for scene in payload["plan"]["scene_cards"]],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                    }
                ]
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": {
                **_framing().model_dump(mode="json"),
                "handoff_scene_card_id": "scene_anchor",
            },
            "scene_cards": [
                {
                    "scene_id": "scene_anchor",
                    "title": "Anchor scene",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 6300,
                },
                {
                    "scene_id": "scene_context",
                    "title": "Context scene",
                    "scene_role": "action",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p2"],
                    "estimated_duration_seconds": 2100,
                },
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text 1",
                    full_text="Full text evidence 1.",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p2",
                    book_id="b1",
                    chunk_ids=["c2"],
                    text="Trimmed text 2",
                    full_text="Full text evidence 2.",
                    axis_id="axis_1",
                ),
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"

    asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    scene_targets = {
        scene["scene_id"]: (
            scene["target_word_count_lower"],
            scene["target_word_count_higher"],
        )
        for scene in captured["payload"]["plan"]["scene_cards"]
    }
    assert scene_targets == {
        "scene_anchor": (11550, 13650),
        "scene_context": (3850, 4550),
    }
    assert captured["payload"]["episode_target_word_count_lower"] == 15400
    assert captured["payload"]["episode_target_word_count_higher"] == 18200


def test_write_episode_uses_no_citation_agent_when_skip_grounding(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}
    logged_events: list[tuple[str, dict]] = []
    original_log = orchestrator.run_logger.log

    def capture_log(event_type: str, **payload):
        logged_events.append((event_type, payload))
        original_log(event_type, **payload)

    orchestrator.run_logger.log = capture_log  # type: ignore[method-assign]

    def fail_if_standard_agent_used(_payload: dict):
        raise AssertionError("standard writing agent should not be used when skip_grounding=True")

    def fake_no_citation_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.writing_agent_no_citations.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": " ".join(["word"] * 800),
                        "source_book_ids": ["b1"],
                    }
                ]
            }
        )

    orchestrator.writing_agent.run = fail_if_standard_agent_used
    orchestrator.writing_agent_no_citations.run = fake_no_citation_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        config=PipelineConfig(skip_grounding=True),
    )
    plan = EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "driving_question": "What changes?",
            "thematic_focus": "Focus",
            "arc_summary": "Arc",
            "unresolved_questions": [],
            "framing": _framing().model_dump(mode="json"),
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "title": "Scene 1",
                    "scene_role": "setup",
                    "dominant_cluster_occurrence_id": "occ_1",
                    "entry_image": "Image",
                    "local_question": "Question",
                    "observable_detail": "Detail",
                    "intended_move": "Move",
                    "primitive_ids": [],
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 300,
                }
            ],
            "target_duration_minutes": 140.0,
            "target_word_count": 16800,
        }
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for writing.",
                    axis_id="axis_1",
                )
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    script = asyncio.run(orchestrator._write_episode(plan, project, corpus, ep_dir, tmp_path))

    payload = captured["payload"]
    assert payload["skip_grounding"] is True
    assert "scene_word_count_targets" not in payload
    assert "previous_sections" not in payload
    assert payload["episode_target_word_count_lower"] == 550
    assert payload["episode_target_word_count_higher"] == 650
    scene = payload["plan"]["scene_cards"][0]
    assert scene["scene_id"] == "scene_1"
    assert "estimated_duration_seconds" not in scene
    assert scene["target_word_count_lower"] == 550
    assert scene["target_word_count_higher"] == 650
    assert script.prose_sections[0].citations == []
    budget_warnings = [
        payload
        for event_type, payload in logged_events
        if event_type == "episode_writing_budget_warning"
    ]
    section_count_warnings = [
        payload
        for event_type, payload in logged_events
        if event_type == "episode_writing_section_count_warning"
    ]
    assert budget_warnings
    assert budget_warnings[0]["actual_word_count"] == 800
    assert budget_warnings[0]["target_word_count_higher"] == 650
    assert section_count_warnings
    assert section_count_warnings[0]["section_count"] == 1
    assert section_count_warnings[0]["target_section_count_min"] == 8
    assert section_count_warnings[0]["target_section_count_max"] == 12


def test_validate_grounding_uses_full_text_lookup(monkeypatch, tmp_path):
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
    captured: dict[str, object] = {}

    def fake_grounding_run(payload: dict):
        captured["payload"] = payload
        return orchestrator.grounding_agent.response_model.model_validate({"episode_number": 1})

    orchestrator.grounding_agent.run = fake_grounding_run
    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="Draft.",
            )
        ],
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed text",
                    full_text="Full text evidence for grounding.",
                    axis_id="axis_1",
                )
            ]
        },
    )

    asyncio.run(orchestrator._validate_grounding(1, script, corpus, tmp_path, tmp_path))

    payload = captured["payload"]
    assert payload["cited_passages"]["p1"]["text"] == "Full text evidence for grounding."


def test_rewrite_for_speech_rewrites_each_section_individually(monkeypatch, tmp_path):
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
    payloads: list[dict] = []

    def fake_spoken_run(payload: dict):
        payloads.append(payload)
        section = payload["script"]["prose_sections"][0]
        return orchestrator.spoken_delivery_agent.response_model.model_validate(
            {
                "sections": [
                    {
                        "section_id": section["section_id"],
                        "text": f"spoken::{section['text']}",
                    }
                ]
            }
        )

    orchestrator.spoken_delivery_agent.run = fake_spoken_run

    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="First section",
            ),
            ProseSection(
                section_id="section_2",
                scene_card_ids=["scene_2"],
                movement_goal="discover",
                text="Second section",
            ),
        ],
    )
    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
    )
    ep_dir = tmp_path / "episodes" / "1"
    ep_dir.mkdir(parents=True, exist_ok=True)

    spoken = asyncio.run(
        orchestrator._rewrite_for_speech(1, script, project, ep_dir, tmp_path)
    )

    assert len(payloads) == 2
    assert payloads[0]["script"]["prose_sections"][0]["section_id"] == "section_1"
    assert payloads[1]["script"]["prose_sections"][0]["section_id"] == "section_2"
    assert len(payloads[0]["script"]["prose_sections"]) == 1
    assert len(payloads[1]["script"]["prose_sections"]) == 1
    assert [section.section_id for section in spoken.sections] == ["section_1", "section_2"]
    assert [section.text for section in spoken.sections] == [
        "spoken::First section",
        "spoken::Second section",
    ]


def test_rewrite_for_speech_raises_on_invalid_section_contract(monkeypatch, tmp_path):
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

    def fake_spoken_run(_payload: dict):
        return orchestrator.spoken_delivery_agent.response_model.model_validate(
            {"sections": [{"section_id": "wrong_id", "text": "spoken"}]}
        )

    orchestrator.spoken_delivery_agent.run = fake_spoken_run

    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="First section",
            )
        ],
    )
    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
    )
    ep_dir = tmp_path / "episodes" / "1"
    ep_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="section_id mismatch"):
        asyncio.run(orchestrator._rewrite_for_speech(1, script, project, ep_dir, tmp_path))


def test_render_episode_audio_writes_section_only_manifest(monkeypatch, tmp_path):
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
    logged_events: list[tuple[str, dict]] = []
    original_log = orchestrator.run_logger.log

    def capture_log(event_type: str, **payload):
        logged_events.append((event_type, payload))
        original_log(event_type, **payload)

    orchestrator.run_logger.log = capture_log  # type: ignore[method-assign]

    spoken = SpokenScript(
        episode_number=4,
        title="Episode 4",
        framing=_framing(),
        sections=[
            SpokenSection(section_id="sec_01", text="One."),
            SpokenSection(section_id="sec_02", text="Two."),
            SpokenSection(section_id="sec_03", text="Three."),
        ],
    )

    config = PipelineConfig(skip_audio=True)
    asyncio.run(
        orchestrator._render_episode_audio(
            episode_number=4,
            spoken=spoken,
            config=config,
            project_dir=tmp_path,
            semaphore=asyncio.Semaphore(1),
            skip_audio=True,
        )
    )

    warning_events = [payload for event_type, payload in logged_events if event_type == "spoken_transition_mismatch_warning"]
    assert not warning_events
    assert (tmp_path / "episodes" / "4" / "render_manifest.json").exists()


def test_plan_series_parallelizes_episode_planning_with_configured_limit(monkeypatch, tmp_path):
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
    in_flight = 0
    max_in_flight = 0
    payloads_by_episode: dict[int, dict] = {}
    lock = threading.Lock()

    def fake_planning_run(payload: dict):
        nonlocal in_flight, max_in_flight
        episode_number = int(payload["episode"]["episode_number"])
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            payloads_by_episode[episode_number] = payload
        try:
            time.sleep(0.05)
            primary_occurrence_id = payload["episode"]["cluster_path"][0]["occurrence_id"]
            return orchestrator.episode_planning_agent.response_model.model_validate(
                {
                    "episode_number": episode_number,
                    "title": payload["episode"]["title"],
                    "driving_question": payload["episode"]["driving_question"],
                    "thematic_focus": payload["episode"]["thematic_focus"],
                    "arc_summary": payload["episode"]["arc_summary"],
                    "unresolved_questions": payload["episode"]["unresolved_questions"],
                    "framing": {
                        "opening_image": "Image",
                        "threat_or_unresolved_action": "Threat",
                        "opening_question": "Question",
                        "handoff_scene_card_id": "scene_1",
                    },
                    "scene_cards": [
                        {
                            "scene_id": "scene_1",
                            "title": "Scene 1",
                            "scene_role": "setup",
                            "dominant_cluster_occurrence_id": primary_occurrence_id,
                            "passage_ids": [],
                            "primitive_ids": [],
                        }
                    ],
                    "target_duration_minutes": 140.0,
                }
            )
        finally:
            with lock:
                in_flight -= 1

    orchestrator.episode_planning_agent.run = fake_planning_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
        episode_count=3,
        config=PipelineConfig(episode_planning_concurrency=2),
    )
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="test",
        series_arc="test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                driving_question="Q1",
                arc_summary="A1",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_1",
                        cluster_id="ec_1",
                        usage="primary",
                    )
                ],
            ),
            StrategyEpisode(
                episode_number=2,
                title="Episode 2",
                driving_question="Q2",
                arc_summary="A2",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_2",
                        cluster_id="ec_2",
                        usage="primary",
                    )
                ],
            ),
            StrategyEpisode(
                episode_number=3,
                title="Episode 3",
                driving_question="Q3",
                arc_summary="A3",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_3",
                        cluster_id="ec_3",
                        usage="primary",
                    )
                ],
            ),
        ],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        episode_candidate_clusters=[
            EpisodeCandidateCluster(
                cluster_id="ec_1",
                title="C1",
                summary="S1",
                primary_member_id="tp_1",
                member_ids=["tp_1"],
                local_question="L1",
                local_payoff_shape="reveal",
            ),
            EpisodeCandidateCluster(
                cluster_id="ec_2",
                title="C2",
                summary="S2",
                primary_member_id="tp_2",
                member_ids=["tp_2"],
                local_question="L2",
                local_payoff_shape="reveal",
            ),
            EpisodeCandidateCluster(
                cluster_id="ec_3",
                title="C3",
                summary="S3",
                primary_member_id="tp_3",
                member_ids=["tp_3"],
                local_question="L3",
                local_payoff_shape="reveal",
            ),
        ],
        primitives_by_family=_primitives_by_family(
            epochal_turns=[
                _primitive("tp_1", "T1"),
                _primitive("tp_2", "T2"),
                _primitive("tp_3", "T3"),
            ]
        ),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed planning text 1",
                    full_text="Full planning text 1",
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p2",
                    book_id="b1",
                    chunk_ids=["c2"],
                    text="Trimmed planning text 2",
                    full_text="",
                    chapter_ref="Chapter 2",
                    axis_id="axis_1",
                ),
                ExtractedPassage(
                    passage_id="p3",
                    book_id="b1",
                    chunk_ids=["c3"],
                    text="Trimmed planning text 3",
                    full_text="Full planning text 3",
                    chapter_ref="Chapter 3",
                    axis_id="axis_1",
                ),
            ]
        },
    )

    plans, _actor_metrics = asyncio.run(
        orchestrator._plan_series(project, synthesis_map, strategy, corpus, tmp_path)
    )

    assert [plan.episode_number for plan in plans] == [1, 2, 3]
    assert all(
        sum(scene.estimated_duration_seconds for scene in plan.scene_cards)
        == int(round(plan.target_duration_minutes * 60.0))
        for plan in plans
    )
    assert 1 < max_in_flight <= 2
    assert sorted(payloads_by_episode) == [1, 2, 3]
    expected_passage_text_by_episode = {
        1: "Full planning text 1",
        2: "Trimmed planning text 2",
        3: "Full planning text 3",
    }
    for episode_number in [1, 2, 3]:
        synthesis_payload = payloads_by_episode[episode_number]["synthesis_map"]
        assert [
            cluster["cluster_id"] for cluster in synthesis_payload["episode_candidate_clusters"]
        ] == [f"ec_{episode_number}"]
        assert [item["id"] for item in synthesis_payload["primitives_by_family"]["epochal_turns"]] == [
            f"tp_{episode_number}"
        ]
        available_passages = payloads_by_episode[episode_number]["available_passages"]
        assert [passage["passage_id"] for passage in available_passages] == [f"p{episode_number}"]
        assert [passage["text"] for passage in available_passages] == [
            expected_passage_text_by_episode[episode_number]
        ]


def test_plan_series_uses_actor_arc_directives_for_episode_metadata(monkeypatch, tmp_path):
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
    captured_payload: dict[str, object] = {}

    def fake_planning_run(payload: dict):
        captured_payload.update(payload)
        primary_occurrence_id = payload["episode"]["cluster_path"][0]["occurrence_id"]
        return orchestrator.episode_planning_agent.response_model.model_validate(
            {
                "episode_number": payload["episode"]["episode_number"],
                "title": payload["episode"]["title"],
                "driving_question": payload["episode"]["driving_question"],
                "thematic_focus": payload["episode"]["thematic_focus"],
                "arc_summary": payload["episode"]["arc_summary"],
                "unresolved_questions": payload["episode"]["unresolved_questions"],
                "framing": {
                    "opening_image": "Image",
                    "threat_or_unresolved_action": "Threat",
                    "opening_question": "Question",
                    "handoff_scene_card_id": "scene_1",
                },
                "scene_cards": [
                    {
                        "scene_id": "scene_1",
                        "title": "Scene 1",
                        "scene_role": "setup",
                        "dominant_cluster_occurrence_id": primary_occurrence_id,
                        "passage_ids": ["p1"],
                        "primitive_ids": ["tp_1"],
                    }
                ],
                "target_duration_minutes": 140.0,
            }
        )

    orchestrator.episode_planning_agent.run = fake_planning_run
    strategy_actor = _actor_arc_directive("actor_primary")
    strategy = NarrativeStrategy(
        strategy_type="convergence",
        justification="test",
        series_arc="test arc",
        episodes=[
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                driving_question="Q1",
                arc_summary="A1",
                actor_arc_directives=[strategy_actor],
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_1",
                        cluster_id="ec_1",
                        usage="primary",
                    )
                ],
            )
        ],
    )
    primitive = _primitive("tp_1", "T1").model_copy(
        update={
            "actor_ids": ["actor_primitive"],
            "primary_actor_ids": ["actor_primitive"],
        }
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        episode_candidate_clusters=[
            EpisodeCandidateCluster(
                cluster_id="ec_1",
                title="C1",
                summary="S1",
                primary_member_id="tp_1",
                member_ids=["tp_1"],
                local_question="L1",
                local_payoff_shape="reveal",
                actor_ids=["actor_cluster"],
                primary_actor_id="actor_cluster",
            )
        ],
        primitives_by_family=_primitives_by_family(epochal_turns=[primitive]),
    )
    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
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
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="Trimmed planning text 1",
                    full_text="Full planning text 1",
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                )
            ]
        },
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(actor_id="actor_primary", display_name="Primary Actor", actor_type="person"),
            ActorProfile(actor_id="actor_cluster", display_name="Cluster Actor", actor_type="person"),
            ActorProfile(actor_id="actor_primitive", display_name="Primitive Actor", actor_type="person"),
        ],
    )

    plans, _actor_metrics = asyncio.run(
        orchestrator._plan_series(
            project,
            synthesis_map,
            strategy,
            corpus,
            tmp_path,
            actor_metadata=actor_metadata,
        )
    )

    assert [actor["actor_id"] for actor in captured_payload["actor_metadata"]["actors"]] == [
        "actor_primary"
    ]
    assert [actor.actor_id for actor in plans[0].actor_arc_directives] == ["actor_primary"]


def test_plan_series_trims_available_passages_for_planning_with_episode_context(
    monkeypatch, tmp_path
):
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
    captured_payload: dict[str, object] = {}

    def fake_planning_run(payload: dict):
        captured_payload.update(payload)
        primary_occurrence_id = payload["episode"]["cluster_path"][0]["occurrence_id"]
        return orchestrator.episode_planning_agent.response_model.model_validate(
            {
                "episode_number": payload["episode"]["episode_number"],
                "title": payload["episode"]["title"],
                "driving_question": payload["episode"]["driving_question"],
                "thematic_focus": payload["episode"]["thematic_focus"],
                "arc_summary": payload["episode"]["arc_summary"],
                "unresolved_questions": payload["episode"]["unresolved_questions"],
                "framing": {
                    "opening_image": "Image",
                    "threat_or_unresolved_action": "Threat",
                    "opening_question": "Question",
                    "handoff_scene_card_id": "scene_1",
                },
                "scene_cards": [
                    {
                        "scene_id": "scene_1",
                        "title": "Scene 1",
                        "scene_role": "setup",
                        "dominant_cluster_occurrence_id": primary_occurrence_id,
                        "passage_ids": [],
                        "primitive_ids": [],
                    }
                ],
                "target_duration_minutes": 130.0,
            }
        )

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
            StrategyEpisode(
                episode_number=1,
                title="Episode One",
                driving_question="drivingalpha",
                thematic_focus="focusbeta",
                arc_summary="arcgamma",
                unresolved_questions=["unresolveddelta"],
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_1",
                        cluster_id="ec_1",
                        usage="primary",
                    )
                ],
            )
        ],
    )
    synthesis_map = SynthesisMap(
        project_id="proj",
        episode_candidate_clusters=[
            EpisodeCandidateCluster(
                cluster_id="ec_1",
                title="Cluster 1",
                summary="Summary",
                primary_member_id="tp_1",
                member_ids=["tp_1"],
                local_question="Question",
                local_payoff_shape="reveal",
            )
        ],
        primitives_by_family=_primitives_by_family(
            epochal_turns=[_primitive("tp_1", "T1")]
        ),
    )
    corpus = ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id="p1",
                    book_id="b1",
                    chunk_ids=["c1"],
                    text="trimmed",
                    full_text=(
                        "Drivingalpha evidence appears first. "
                        "Focusbeta evidence appears second. "
                        "Arcgamma evidence appears third. "
                        "Themeomega evidence appears fourth."
                    ),
                    chapter_ref="Chapter 1",
                    axis_id="axis_1",
                )
            ]
        },
    )

    plans, _actor_metrics = asyncio.run(
        orchestrator._plan_series(project, synthesis_map, strategy, corpus, tmp_path)
    )

    assert [plan.episode_number for plan in plans] == [1]
    available_passages = captured_payload["available_passages"]
    assert isinstance(available_passages, list)
    assert len(available_passages) == 1
    kept_sentences = _split_sentences(available_passages[0]["text"])
    assert kept_sentences == [
        "Drivingalpha evidence appears first.",
        "Focusbeta evidence appears second.",
    ]
