"""Focused tests for active orchestrator helpers in the redesigned pipeline."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _allocate_synthesis_passages_by_axis,
    _allocate_scene_durations,
    _build_spoken_delivery_batches,
    _build_spoken_delivery_sections_payload,
    _build_spine_plan_diagnostics,
    _build_scene_card_family_warnings,
    _compute_scene_word_count_targets,
    _build_scene_card_count_warnings,
    _build_scene_card_primitive_warnings,
    _build_passage_lookup,
    _build_episode_architecture_realization,
    _estimate_duration_seconds_from_words,
    _flatten_synthesis_primitives,
    _normalize_writing_section_outputs,
    _occurrence_spillover_weight,
    _occurrence_weight,
    _prioritize_cross_book_pairs,
    _resolve_synthesis_bm25_keep_fraction_by_passage,
    _round_allocations_to_total,
    _scene_duration_bounds,
    _scene_importance_weight,
    _split_episode_writing_windows,
    _extract_previous_spoken_tail,
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
    ActorRelationship,
    ArchitectureSection,
    BookRecord,
    EpisodeScript,
    EpisodePlan,
    EpisodeSpine,
    EpisodeArchitecture,
    EventPrimitive,
    ExtractedPassage,
    FramingBlock,
    NarrativeStrategy,
    PassagePair,
    PipelineConfig,
    PodcastMode,
    ProseSection,
    SceneActor,
    SceneActorArcBinding,
    SceneCard,
    SceneJob,
    SpineRelation,
    SpokenScript,
    SpokenSection,
    StrategyEpisode,
    SupportPrimitiveRole,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicCorpus,
    ThematicAxis,
    ThematicProject,
    VerdictMode,
    resolve_pipeline_config_for_mode,
)


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="Image",
        threat_or_unresolved_action="Threat",
        opening_question="Question",
        handoff_scene_card_id="scene_1",
    )


def _episode_spine(
    *spine_pack_ids: str,
    support_pack_roles: dict[str, SupportPrimitiveRole] | None = None,
    allowed_recalls: list[str] | None = None,
) -> EpisodeSpine:
    provided_core_ids = list(spine_pack_ids) or ["pack_1"]
    core_primitive_ids = list(provided_core_ids)
    while len(core_primitive_ids) < 6:
        core_primitive_ids.append(f"core_{len(core_primitive_ids) + 1}")
    merged_support_roles = dict(support_pack_roles or {})
    next_support_idx = 1
    while len(merged_support_roles) < 7:
        support_id = f"support_{next_support_idx}"
        next_support_idx += 1
        if support_id in core_primitive_ids or support_id in merged_support_roles:
            continue
        merged_support_roles[support_id] = (
            SupportPrimitiveRole.MECHANISM if len(merged_support_roles) % 2 == 0 else SupportPrimitiveRole.TEXTURE
        )
    return EpisodeSpine(
        listener_question="Question?",
        argument="A working claim.",
        core_primitive_ids=core_primitive_ids,
        support_primitive_roles=merged_support_roles,
        recall_primitive_ids=list(allowed_recalls or []),
    )


def _strategy_episode(
    *spine_pack_ids: str,
    support_pack_roles: dict[str, SupportPrimitiveRole] | None = None,
    allowed_recalls: list[str] | None = None,
) -> StrategyEpisode:
    return StrategyEpisode(
        episode_number=1,
        title="Episode",
        arc_summary="Arc",
        episode_spine=_episode_spine(
            *spine_pack_ids,
            support_pack_roles=support_pack_roles,
            allowed_recalls=allowed_recalls,
        ),
    )


def _host_moves() -> dict[str, list[dict[str, str]]]:
    return {
        "open": [
            {
                "move_type": "orient",
                "note": "Set the listener's footing before the beat turns.",
            }
        ],
        "pivot": [],
        "close": [],
    }


def _scene_card(
    scene_id: str,
    pack_id: str,
    *,
    scene_role: str = "action",
    actors: list[SceneActor] | None = None,
    section_id: str = "section_1",
) -> SceneCard:
    return SceneCard(
        scene_id=scene_id,
        section_id=section_id,
        title=scene_id,
        scene_role=scene_role,
        dominant_pack_id=pack_id,
        spine_relation=SpineRelation.SPINE_ADVANCE,
        state_effect="The listener's understanding moves forward.",
        entry_image="Image",
        local_question="Question",
        observable_detail="Detail",
        audible_detail="The stamp snaps down.",
        intended_move="Move",
        passage_ids=["p1"],
        host_moves=_host_moves(),
        actors=list(actors or []),
        estimated_duration_seconds=120,
    )


def _episode_plan(
    scene_cards: list[SceneCard],
    *,
    episode_number: int = 1,
    target_word_count: int = 18900,
    handoff_scene_card_id: str | None = None,
    answer_scene_card_id: str | None = None,
    residue_scene_card_id: str | None = None,
) -> EpisodePlan:
    normalized_scene_cards = [scene.model_copy(deep=True) for scene in scene_cards]
    if not normalized_scene_cards:
        raise ValueError("scene_cards must not be empty")
    if answer_scene_card_id is None:
        answer_index = -2 if len(normalized_scene_cards) >= 2 else -1
        resolved_answer_scene_card_id = normalized_scene_cards[answer_index].scene_id
    else:
        resolved_answer_scene_card_id = answer_scene_card_id
    resolved_residue_scene_card_id = residue_scene_card_id
    if (
        answer_scene_card_id is None
        and len(normalized_scene_cards) >= 2
        and resolved_residue_scene_card_id is None
    ):
        resolved_residue_scene_card_id = normalized_scene_cards[-1].scene_id
    if (
        resolved_residue_scene_card_id == resolved_answer_scene_card_id
        and len(normalized_scene_cards) >= 2
    ):
        fallback_residue_scene = next(
            scene
            for scene in reversed(normalized_scene_cards)
            if scene.scene_id != resolved_answer_scene_card_id
        )
        resolved_residue_scene_card_id = fallback_residue_scene.scene_id
    normalized_scene_cards = [
        scene.model_copy(
            update={
                "scene_job": (
                    SceneJob.ANSWER
                    if scene.scene_id == resolved_answer_scene_card_id
                    else (
                        SceneJob.RESIDUE
                        if scene.scene_id == resolved_residue_scene_card_id
                        else scene.scene_job
                    )
                )
            }
        )
        for scene in normalized_scene_cards
    ]
    return EpisodePlan(
        episode_number=episode_number,
        framing=_framing().model_copy(
            update={
                "handoff_scene_card_id": (
                    handoff_scene_card_id or normalized_scene_cards[0].scene_id
                )
            }
        ),
        scene_cards=normalized_scene_cards,
        answer_scene_card_id=resolved_answer_scene_card_id,
        residue_scene_card_id=resolved_residue_scene_card_id,
        target_word_count=target_word_count,
    )


def _episode_architecture_for_scene_cards(scene_cards: list[SceneCard]) -> EpisodeArchitecture:
    ordered_section_ids: list[str] = []
    for scene in scene_cards:
        if scene.section_id not in ordered_section_ids:
            ordered_section_ids.append(scene.section_id)
    sections = []
    for idx, section_id in enumerate(ordered_section_ids, start=1):
        purpose = "closing" if idx == len(ordered_section_ids) else ("opening" if idx == 1 else "setup")
        sections.append(
            ArchitectureSection.model_validate(
                {
                    "section_id": section_id,
                    "purpose": purpose,
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["pack_1"],
                    "section_anchor": f"Anchor for {section_id}",
                    "must_stage_beats": [f"Beat A {section_id}", f"Beat B {section_id}"],
                    "listener_tension": f"Tension for {section_id}",
                    "section_turn": f"Turn for {section_id}",
                    "transition_logic": f"Transition for {section_id}",
                    "depends_on_section_ids": [ordered_section_ids[idx - 2]] if idx > 1 else [],
                    "sets_up_section_ids": [ordered_section_ids[idx]] if idx < len(ordered_section_ids) else [],
                    "recurrence_role": "none",
                    "priority_core_passage_ids": [],
                }
            )
        )
    return EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id=ordered_section_ids[min(len(ordered_section_ids), 2) - 1],
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=sections,
        architecture_notes=[],
    )


def _writing_prose_sections_from_payload(
    payload: dict,
    *,
    renamed_scene_ids_by_section: dict[str, list[str]] | None = None,
    text_prefix: str = "Draft text.",
) -> list[dict[str, object]]:
    renamed_scene_ids_by_section = renamed_scene_ids_by_section or {}
    sections: list[dict[str, object]] = []
    for section in payload["architecture"]["sections"]:
        section_id = section["section_id"]
        section_scene_ids = renamed_scene_ids_by_section.get(
            section_id,
            [
                scene["scene_id"]
                for scene in payload["plan"]["scene_cards"]
                if scene["section_id"] == section_id
            ],
        )
        sections.append(
            {
                "section_id": section_id,
                "scene_card_ids": section_scene_ids,
                "movement_goal": "discover",
                "text": " ".join(text_prefix for _ in section_scene_ids).strip(),
                "source_book_ids": [],
            }
        )
    return sections


def _scene_actor(
    *,
    name: str = "Actor",
    presence: str = "secondary",
    binding_weight: str | None = None,
) -> SceneActor:
    _ = binding_weight
    return SceneActor(name=name, presence=presence)


def _normalize_fixture_strategy_episode(payload: dict[str, object]) -> dict[str, object]:
    if "episode_spine" in payload:
        payload = dict(payload)
        payload.pop("driving_question", None)
        return payload
    cluster_path = list(payload.pop("cluster_path", []) or [])
    spine_pack_ids = [
        item["cluster_id"]
        for item in cluster_path
        if item.get("usage") == "primary"
    ]
    support_pack_roles: dict[str, str] = {}
    allowed_recalls: list[str] = []
    for item in cluster_path:
        cluster_id = str(item.get("cluster_id", ""))
        if not cluster_id:
            continue
        if item.get("usage") == "echo":
            allowed_recalls.append(cluster_id)
            continue
        if cluster_id in spine_pack_ids[:3]:
            continue
        support_pack_roles[cluster_id] = (
            "texture" if item.get("emphasis") == "compressed" else "mechanism"
        )
    listener_question = str(payload.get("driving_question", "What changes here?"))
    thematic_focus = str(payload.get("thematic_focus", "") or listener_question)
    arc_summary = str(payload.get("arc_summary", "") or thematic_focus)
    unresolved_questions = list(payload.get("unresolved_questions", []) or [])
    payload["episode_spine"] = {
        "listener_question": listener_question,
        "argument": thematic_focus,
        "core_primitive_ids": (spine_pack_ids[:3] or [cluster_path[0]["cluster_id"]])
        + [f"core_{idx}" for idx in range(4, 8)],
        "support_primitive_roles": {
            **support_pack_roles,
            **{
                f"support_{idx}": ("texture" if idx % 2 else "mechanism")
                for idx in range(len(support_pack_roles) + 1, 11)
            },
        },
        "recall_primitive_ids": allowed_recalls,
    }
    payload.pop("driving_question", None)
    return payload


def _normalize_fixture_plan_episode(
    payload: dict[str, object],
    strategy_episode: StrategyEpisode,
    occurrence_to_pack_id: dict[str, str],
) -> dict[str, object]:
    normalized = dict(payload)
    for stale_key in (
        "title",
        "driving_question",
        "thematic_focus",
        "arc_summary",
        "unresolved_questions",
        "actor_arc_directives",
        "target_duration_minutes",
        "episode_spine",
    ):
        normalized.pop(stale_key, None)
    normalized_scene_cards: list[dict[str, object]] = []
    for idx, card in enumerate(normalized.get("scene_cards", []), start=1):
        scene = dict(card)
        scene.setdefault("section_id", f"section_{idx:02d}")
        dominant_pack_id = scene.pop("dominant_cluster_occurrence_id", None)
        if dominant_pack_id:
            scene["dominant_pack_id"] = occurrence_to_pack_id.get(str(dominant_pack_id), str(dominant_pack_id))
        if "spine_relation" not in scene:
            role = str(scene.get("scene_role", "")).strip()
            scene["spine_relation"] = {
                "setup": "set_stakes",
                "shock": "turn",
                "action": "spine_advance",
                "consequence": "show_consequence",
                "reaction": "supply_mechanism",
                "contestation": "apply_counterpressure",
                "synthesis": "spine_advance",
            }.get(role, "spine_advance")
        if "state_effect" not in scene:
            scene["state_effect"] = (
                str(scene.get("intended_move", "")).strip()
                or str(scene.get("local_question", "")).strip()
                or "The listener's understanding shifts."
            )
        scene.setdefault("host_moves", _host_moves())
        scene.setdefault("beat_change", str(scene.get("state_effect", "")).strip())
        normalized_scene_cards.append(scene)
    normalized["scene_cards"] = normalized_scene_cards
    if len(normalized_scene_cards) >= 2:
        normalized["answer_scene_card_id"] = normalized_scene_cards[-2]["scene_id"]
        normalized["residue_scene_card_id"] = normalized_scene_cards[-1]["scene_id"]
        normalized_scene_cards[-2]["scene_job"] = "answer"
        normalized_scene_cards[-1]["scene_job"] = "residue"
    elif normalized_scene_cards:
        normalized["answer_scene_card_id"] = normalized_scene_cards[-1]["scene_id"]
        normalized_scene_cards[-1]["scene_job"] = "answer"
    return normalized


def _load_middle_east_v2_episode_fixtures() -> list[tuple[StrategyEpisode, EpisodePlan]]:
    run_dir = Path("runs/middle_east_v2")
    strategy_payload = json.loads((run_dir / "narrative_strategy.json").read_text())
    plan_payload = json.loads((run_dir / "series_plan.json").read_text())
    occurrence_map_by_episode = {
        episode["episode_number"]: {
            str(item.get("occurrence_id", "")): str(item.get("cluster_id", ""))
            for item in episode.get("cluster_path", [])
            if item.get("occurrence_id") and item.get("cluster_id")
        }
        for episode in strategy_payload["episodes"]
    }
    strategy_by_episode = {
        episode["episode_number"]: StrategyEpisode.model_validate(
            _normalize_fixture_strategy_episode(dict(episode))
        )
        for episode in strategy_payload["episodes"]
    }
    return [
        (
            strategy_by_episode[episode["episode_number"]],
            EpisodePlan.model_validate(
                _normalize_fixture_plan_episode(
                    dict(episode),
                    strategy_by_episode[episode["episode_number"]],
                    occurrence_map_by_episode.get(episode["episode_number"], {}),
                )
            ),
        )
        for episode in plan_payload["episodes"]
    ]


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def _primitive(primitive_id: str, title: str = "Title") -> EventPrimitive:
    match = primitive_id.rsplit("_", 1)
    suffix = match[1] if len(match) == 2 and match[1].isdigit() else "1"
    return EventPrimitive(
        id=primitive_id,
        substrate="events",
        title=title,
        core_passage_ids=[f"p{suffix}"],
        event_type="political rupture",
        what_happened=f"Event {suffix} changes the balance.",
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
                resolution="Use where pressure becomes consequence.",
            ),
            ActorArcThread(
                thread_id=f"{actor_id}_guardrail_1",
                arc_type="guardrail",
                label="repetition guardrail",
                premise="Do not repeat without change.",
                pressure="Repetition can flatten the actor.",
                movement="Vary the actor function by scene.",
                resolution="Continuity remains legible without repetition.",
            ),
        ],
    )



def test_build_scene_card_count_warnings_for_under_target():
    warnings = _build_scene_card_count_warnings(
        scene_card_count=18,
        scene_card_target_min=25,
        scene_card_target_max=40,
    )
    assert len(warnings) == 1
    assert warnings[0].startswith("scene_card_count_below_target")


def test_episode_architecture_realization_uses_resolved_config_targets():
    strategy_episode = _strategy_episode("et_1", "et_2", "et_3")
    sections = []
    for idx in range(6):
        section_id = f"section_{idx + 1:02d}"
        sections.append(
            ArchitectureSection(
                section_id=section_id,
                purpose="closing" if idx == 5 else "setup",
                approx_runtime_minutes=2.0 if idx == 5 else 8.0,
                primitive_ids=[
                    strategy_episode.episode_spine.assigned_primitive_ids[
                        min(idx, len(strategy_episode.episode_spine.assigned_primitive_ids) - 1)
                    ]
                ],
                section_anchor="Anchor",
                must_stage_beats=["Visible move", "Immediate consequence"],
            )
        )
    architecture = EpisodeArchitecture(
        episode_number=1,
        major_turn_section_id="section_03",
        sections=sections,
        architecture_notes=[],
    )

    realization = _build_episode_architecture_realization(
        strategy_episode=strategy_episode,
        architecture=architecture,
        pipeline_config=resolve_pipeline_config_for_mode(
            PipelineConfig(podcast_mode=PodcastMode.MINIFIED)
        ),
    )

    assert any(
        warning.startswith("architecture_section_count_below_target")
        for warning in realization["warnings"]
    )


def test_build_scene_card_primitive_warnings_reports_density_and_unknown_ids():
    cards = [
        _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
            update={"primitive_ids": [], "estimated_duration_seconds": 60}
        ),
        _scene_card("scene_2", "pack_2", scene_role="reaction").model_copy(
            update={
                "primitive_ids": ["tp_1", "tp_2", "tp_3"],
                "passage_ids": ["p2"],
                "estimated_duration_seconds": 90,
            }
        ),
    ]
    warnings = _build_scene_card_primitive_warnings(
        scene_cards=cards,
        primitive_pool_ids={"tp_1", "tp_2"},
        primitive_by_id={},
        primitive_min=1,
        primitive_max=2,
    )
    assert warnings == []

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
        _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
            update={"primitive_ids": [], "estimated_duration_seconds": 30}
        ),
        _scene_card("scene_2", "pack_2", scene_role="reaction").model_copy(
            update={"primitive_ids": [], "passage_ids": ["p2"], "estimated_duration_seconds": 90}
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=1000, words_per_minute=120.0)
    assert targets == {"scene_1": 60, "scene_2": 180}


def test_compute_scene_word_count_targets_scales_with_words_per_minute():
    scenes = [
        _scene_card("scene_anchor", "pack_1", scene_role="setup").model_copy(
            update={"estimated_duration_seconds": 30}
        ),
        _scene_card("scene_compressed", "pack_2", scene_role="reaction").model_copy(
            update={"passage_ids": ["p2"], "estimated_duration_seconds": 90}
        ),
    ]
    targets = _compute_scene_word_count_targets(scenes, episode_target_word_count=1200, words_per_minute=130.0)
    assert targets == {"scene_anchor": 65, "scene_compressed": 195}


def test_compute_scene_word_count_targets_requires_positive_scene_durations():
    valid_scene = _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
        update={"estimated_duration_seconds": 60}
    )
    scenes = [
        valid_scene,
        valid_scene.model_copy(
            update={
                "scene_id": "scene_2",
                "title": "Scene 2",
                "scene_role": "reaction",
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
        "pack_1",
        scene_role="shock",
        actors=[_scene_actor(presence="primary", binding_weight="strong")],
    )

    assert _scene_importance_weight(scene) == pytest.approx(1.30 * 0.85 * 1.20)


def test_occurrence_weight_uses_increased_spine_weight():
    episode = _strategy_episode(
        "pack_anchor",
        support_pack_roles={
            "pack_major": SupportPrimitiveRole.MECHANISM,
            "pack_supporting": SupportPrimitiveRole.TEXTURE,
        },
    )

    assert _occurrence_weight(episode, "pack_anchor") == pytest.approx(1.75)
    assert _occurrence_weight(episode, "pack_major") == pytest.approx(0.90)
    assert _occurrence_weight(episode, "pack_supporting") == pytest.approx(0.60)


def test_occurrence_spillover_weight_uses_increased_spine_weight():
    episode = _strategy_episode(
        "pack_anchor",
        support_pack_roles={
            "pack_major": SupportPrimitiveRole.MECHANISM,
            "pack_supporting": SupportPrimitiveRole.TEXTURE,
        },
    )

    assert _occurrence_spillover_weight(episode, "pack_anchor") == pytest.approx(1.15)
    assert _occurrence_spillover_weight(episode, "pack_major") == pytest.approx(0.25)
    assert _occurrence_spillover_weight(episode, "pack_supporting") == pytest.approx(0.00)


def test_scene_duration_bounds_no_actor_by_role_bucket():
    episode = _strategy_episode("pack_1")

    assert _scene_duration_bounds(
        _scene_card("argument", "pack_1", scene_role="synthesis"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((40.5, 81.0))
    assert _scene_duration_bounds(
        _scene_card("action", "pack_1", scene_role="consequence"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((40.5, 81.0))
    assert _scene_duration_bounds(
        _scene_card("mid", "pack_1", scene_role="setup"),
        episode,
        avg_sec=150.0,
    ) == pytest.approx((40.5, 81.0))


def test_scene_duration_bounds_scales_with_avg_sec():
    episode = _strategy_episode("pack_1")
    scene = _scene_card(
        "scene_1",
        "pack_1",
        actors=[_scene_actor(presence="primary", binding_weight="strong")],
    )

    short_bounds = _scene_duration_bounds(scene, episode, avg_sec=100.0)
    long_bounds = _scene_duration_bounds(scene, episode, avg_sec=250.0)

    assert long_bounds[0] == pytest.approx(short_bounds[0] * 2.5)
    assert long_bounds[1] == pytest.approx(short_bounds[1] * 2.5)


def test_allocate_scene_durations_spillover_goes_only_to_anchor_and_major():
    episode = _strategy_episode(
        "pack_anchor",
        support_pack_roles={
            "pack_major": SupportPrimitiveRole.MECHANISM,
            "pack_supporting": SupportPrimitiveRole.TEXTURE,
        },
    )

    allocated = _allocate_scene_durations(
        [
            _scene_card("anchor_scene", "pack_anchor"),
            _scene_card("major_scene", "pack_major"),
            _scene_card("support_scene", "pack_supporting"),
        ],
        episode,
        target_duration_seconds=600,
    )
    seconds = {scene.scene_id: scene.estimated_duration_seconds for scene in allocated}

    assert sum(seconds.values()) == 600
    assert seconds["support_scene"] == 200
    assert seconds["anchor_scene"] == 200
    assert seconds["major_scene"] == 200


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
        target_duration_seconds=sum(
            scene.estimated_duration_seconds for scene in plan.scene_cards
        ),
    )
    seconds = {scene.scene_id: scene.estimated_duration_seconds for scene in allocated}

    assert sum(seconds.values()) == 6000
    assert seconds["e1_s07"] == pytest.approx(148, abs=5)
    assert seconds["e1_s06"] == pytest.approx(148, abs=5)
    assert seconds["e1_s26"] == pytest.approx(148, abs=5)
    assert seconds["e1_s36"] == pytest.approx(148, abs=5)
    assert seconds["e1_s21"] == pytest.approx(148, abs=5)
    assert seconds["e1_s34"] == pytest.approx(198, abs=5)
    assert seconds["e1_s26"] == seconds["e1_s21"]
    assert seconds["e1_s36"] < seconds["e1_s34"]
    assert seconds["e1_s33"] == seconds["e1_s36"]


def test_allocate_scene_durations_totals_sum_to_target():
    for episode, plan in _load_middle_east_v2_episode_fixtures():
        target_duration_seconds = sum(
            scene.estimated_duration_seconds for scene in plan.scene_cards
        )
        allocated = _allocate_scene_durations(
            plan.scene_cards,
            episode,
            target_duration_seconds=target_duration_seconds,
        )

        assert len(allocated) == len(plan.scene_cards)
        assert sum(scene.estimated_duration_seconds for scene in allocated) == target_duration_seconds


def test_build_spine_plan_diagnostics_accepts_065_spine_share():
    episode = _strategy_episode(
        "pack_spine",
        support_pack_roles={"pack_support": SupportPrimitiveRole.MECHANISM},
    )
    scene_cards = [
        _scene_card(f"spine_{idx}", "pack_spine")
        for idx in range(12)
    ] + [
        _scene_card(
            f"support_{idx}",
            "pack_support",
            scene_role="synthesis",
        ).model_copy(update={"spine_relation": SpineRelation.SUPPLY_MECHANISM})
        for idx in range(7)
    ] + [
        _scene_card("spine_ending", "pack_spine", scene_role="consequence")
    ]
    scene_cards[-1] = scene_cards[-1].model_copy(
        update={
            "dominant_pack_id": "pack_spine",
            "spine_relation": SpineRelation.SHOW_CONSEQUENCE,
        }
    )
    plan = _episode_plan(
        scene_cards,
        handoff_scene_card_id="spine_0",
        target_word_count=1000,
    )

    diagnostics = _build_spine_plan_diagnostics(
        strategy_episode=episode,
        plan=plan,
    )

    assert diagnostics["core_scene_share"] == pytest.approx(1.0)
    assert diagnostics["core_word_share"] == pytest.approx(1.0)
    assert "spine_underweight" not in diagnostics["failure_labels"]


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

    assert len(_split_sentences(candidates[0]["text"])) == 1
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
    assert keep_fraction_by_passage_id["p1"] == 0.35
    assert keep_fraction_by_passage_id["p2"] == 0.25
    assert keep_fraction_by_passage_id["p3"] == 0.25
    assert keep_fraction_by_passage_id["p4"] == 0.15
    assert keep_fraction_by_passage_id["p5"] == 0.15
    assert keep_fraction_by_passage_id["p6"] == 0.15
    assert keep_fraction_by_passage_id["p7"] == 0.15
    assert keep_fraction_by_passage_id["p8"] == 0.15
    assert keep_fraction_by_passage_id["p9"] == 0.15
    assert keep_fraction_by_passage_id["p10"] == 0.15


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
    assert len(_split_sentences(candidates[1]["text"])) == 1


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


def test_trim_candidate_texts_by_bm25_query_text_supports_per_passage_queries():
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
        query_text_by_passage_id={"p1": "dropgamma"},
    )

    kept = _split_sentences(candidates[0]["text"])
    assert kept == [
        "Dropgamma sentence three.",
        "Dropgamma sentence four.",
    ]


def test_trim_candidate_texts_by_bm25_query_text_keeps_at_least_one_sentence_when_budget_is_tight():
    candidates = [
        {
            "passage_id": "p1",
            "book_id": "b1",
            "text": (
                "Keepalpha sentence one with extra words. "
                "Dropgamma short two."
            ),
        }
    ]

    _trim_candidate_texts_by_bm25_query_text(
        "keepalpha",
        candidates,
        keep_fraction=0.2,
    )

    kept = _split_sentences(candidates[0]["text"])
    assert kept == [
        "Keepalpha sentence one with extra words.",
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
        primitives=[_primitive("tp_1", "Turn")],
    )
    assert _build_passage_lookup(corpus)["p1"].book_id == "b1"
    assert _flatten_synthesis_primitives(synthesis_map)["tp_1"].title == "Turn"


def test_prioritize_cross_book_pairs_prefers_contradicts_and_filters_independent():
    prioritized = _prioritize_cross_book_pairs(
        [
            PassagePair(
                passage_a_id="p3",
                passage_b_id="p4",
                relationship="exemplifies",
                strength=0.99,
                axis_id="axis_1",
            ),
            PassagePair(
                passage_a_id="p5",
                passage_b_id="p6",
                relationship="contextualizes",
                strength=0.95,
                axis_id="axis_1",
            ),
            PassagePair(
                passage_a_id="p1",
                passage_b_id="p2",
                relationship="contradicts",
                strength=0.4,
                axis_id="axis_1",
            ),
            PassagePair(
                passage_a_id="p0",
                passage_b_id="p9",
                relationship="contradicts",
                strength=0.3,
                axis_id="axis_1",
            ),
            PassagePair(
                passage_a_id="p7",
                passage_b_id="p8",
                relationship="independent",
                strength=1.0,
                axis_id="axis_1",
            ),
        ]
    )

    assert [
        (pair.relationship, pair.passage_a_id, pair.passage_b_id)
        for pair in prioritized
    ] == [
        ("contradicts", "p1", "p2"),
        ("contradicts", "p0", "p9"),
        ("contextualizes", "p5", "p6"),
        ("exemplifies", "p3", "p4"),
    ]


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

    assert (
        orchestrator.synthesis_primitives_agent.schema_name
        == "primitive_substrate_extraction"
    )
    assert not hasattr(orchestrator, "synthesis_mapping_agent")


def test_build_scene_card_family_warnings_reports_missing_mix():
    episode = StrategyEpisode(
        episode_number=1,
        title="Episode 1",
        arc_summary="A1",
        episode_spine=_episode_spine(
            "pack_1",
            support_pack_roles={"pack_2": SupportPrimitiveRole.MECHANISM},
            allowed_recalls=["pack_3"],
        ),
    )

    warnings = _build_scene_card_family_warnings(
        strategy_episode=episode,
        primitive_pool_ids={"et_1"},
        primitive_by_id={"et_1": _primitive("et_1", "Turn")},
    )

    assert any(
        warning.startswith("primitive_pool_missing_human_grounding")
        for warning in warnings
    )
    assert any(
        warning.startswith("primitive_pool_missing_system_or_context")
        for warning in warnings
    )
    assert any(
        warning.startswith("primitive_pool_missing_recurrence")
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
                ExtractedPassage(passage_id="p1", book_id="b1", chunk_ids=["c1"], text="P1", full_text="P1 full text.", axis_id="axis_1", relevance_score=0.9, quotability_score=0.8),
                ExtractedPassage(passage_id="p2", book_id="b1", chunk_ids=["c2"], text="P2", full_text="", axis_id="axis_1", relevance_score=0.7, quotability_score=0.7),
                ExtractedPassage(passage_id="p3", book_id="b1", chunk_ids=["c3"], text="P3", full_text="P3 full text.", axis_id="axis_1", relevance_score=0.6, quotability_score=0.6),
            ],
            "axis_2": [
                ExtractedPassage(passage_id="p4", book_id="b1", chunk_ids=["c4"], text="P4", full_text="P4 full text.", axis_id="axis_2", relevance_score=0.95, quotability_score=0.9),
                ExtractedPassage(passage_id="p5", book_id="b1", chunk_ids=["c5"], text="P5", full_text="P5 full text.", axis_id="axis_2", relevance_score=0.85, quotability_score=0.8),
                ExtractedPassage(passage_id="p6", book_id="b1", chunk_ids=["c6"], text="P6", full_text="P6 full text.", axis_id="axis_2", relevance_score=0.1, quotability_score=0.1),
                ExtractedPassage(passage_id="p7", book_id="b1", chunk_ids=["c1"], text="P7 duplicate chunk", full_text="P7 duplicate full text.", axis_id="axis_2", relevance_score=0.99, quotability_score=0.95),
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
    passage_text_by_id = {
        item["passage_id"]: item["text"]
        for axis_books in passages.values()
        for book_group in axis_books
        for item in book_group["passages"]
    }
    assert passage_text_by_id["p1"] == "P1 full text."
    assert passage_text_by_id["p6"] == "P6 full text."
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
    plan = _episode_plan(
        [
            _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
                update={
                    "primitive_ids": ["prim_1"],
                    "estimated_duration_seconds": 300,
                    "scene_job": "answer",
                }
            )
        ]
    )
    strategy_episode = _strategy_episode("pack_1")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
    primitive = EventPrimitive(
        id="prim_1",
        substrate="events",
        title="A decisive rupture",
        core_passage_ids=["p1"],
        support_passage_ids=["p_support"],
        timeframe="2001",
        geography="Kabul",
        actor_ids=["actor_1"],
        event_type="attack",
        what_happened="A decisive rupture lands.",
        event_result="The stakes become public.",
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
    asyncio.run(
        orchestrator._write_episode(
            plan,
            strategy_episode,
            architecture,
            project,
            corpus,
            ep_dir,
            tmp_path,
            primitive_lookup={primitive.id: primitive},
        )
    )

    payload = captured["payload"]
    assert payload["passages"][0]["text"] == "Full text evidence for writing."
    assert payload["episode_target_word_count_lower"] == 686
    assert payload["episode_target_word_count_higher"] == 839
    assert payload["scene_primitive_briefs"] == {
        "scene_1": [primitive.model_dump(mode="json")]
    }


def test_write_episode_uses_three_writing_calls_for_full_mode(monkeypatch, tmp_path):
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
                "prose_sections": _writing_prose_sections_from_payload(payload)
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = _episode_plan(
        [
            _scene_card(
                f"scene_{idx}",
                f"pack_{idx}",
                scene_role="setup",
                section_id=f"section_{idx}",
            ).model_copy(update={"passage_ids": [f"p{idx}"], "estimated_duration_seconds": 300})
            for idx in range(1, 27)
        ]
    )
    strategy_episode = _strategy_episode("pack_1", "pack_2", "pack_3")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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
    asyncio.run(
        orchestrator._write_episode(
            plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
        )
    )

    assert len(captured_payloads) == 5
    assert sum(payload["episode_target_word_count_lower"] for payload in captured_payloads) == 17836
    assert sum(payload["episode_target_word_count_higher"] for payload in captured_payloads) == 21814
    assert sum(len(payload["passages"]) for payload in captured_payloads) == 26
    assert all("previous_sections" not in payload for payload in captured_payloads)
    assert "prior_window_continuity" not in captured_payloads[0]
    assert captured_payloads[1]["prior_window_continuity"]["completed_scene_count"] > 0
    assert captured_payloads[2]["prior_window_continuity"]["completed_scene_count"] > 0
    payload_scene_cards = [
        scene
        for payload in captured_payloads
        for scene in payload["plan"]["scene_cards"]
    ]
    assert [scene["scene_id"] for scene in payload_scene_cards] == [
        f"scene_{idx}" for idx in range(1, 27)
    ]
    assert captured_payloads[0]["plan"]["framing"]["handoff_scene_card_id"] == "scene_1"
    assert all("estimated_duration_seconds" not in scene for scene in payload_scene_cards)
    assert all(scene["target_word_count_lower"] == 686 for scene in payload_scene_cards)
    assert all(scene["target_word_count_higher"] == 839 for scene in payload_scene_cards)


def test_write_episode_uses_two_writing_calls_for_minified_mode(monkeypatch, tmp_path):
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
                "prose_sections": _writing_prose_sections_from_payload(payload)
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
        config=resolve_pipeline_config_for_mode(
            PipelineConfig(podcast_mode=PodcastMode.MINIFIED)
        ),
    )
    plan = _episode_plan(
        [
            _scene_card(
                f"scene_{idx}",
                f"pack_{((idx - 1) % 2) + 1}",
                scene_role="setup",
                section_id=f"section_{idx}",
            ).model_copy(update={"passage_ids": [f"p{idx}"], "estimated_duration_seconds": 300})
            for idx in range(1, 9)
        ],
        target_word_count=5400,
    )
    strategy_episode = _strategy_episode("pack_1", "pack_2")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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
                for idx in range(1, 9)
            ]
        },
    )
    ep_dir = tmp_path / "episodes" / "1"
    asyncio.run(
        orchestrator._write_episode(
            plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
        )
    )

    assert len(captured_payloads) == 2
    assert "prior_window_continuity" not in captured_payloads[0]
    assert captured_payloads[1]["prior_window_continuity"]["completed_scene_count"] > 0
    assert [
        scene["scene_id"]
        for payload in captured_payloads
        for scene in payload["plan"]["scene_cards"]
    ] == [f"scene_{idx}" for idx in range(1, 9)]


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
                "prose_sections": _writing_prose_sections_from_payload(payload)
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[BookRecord(book_id="b1", title="Book 1", author="Author", source_path="/tmp/book.txt", source_type="txt")],
    )
    plan = _episode_plan(
        [
            _scene_card(
                "scene_anchor",
                "pack_1",
                scene_role="setup",
                section_id="section_1",
            ).model_copy(update={"title": "Anchor scene", "passage_ids": ["p1"], "estimated_duration_seconds": 6300}),
            _scene_card(
                "scene_context",
                "pack_1",
                scene_role="action",
                section_id="section_2",
            ).model_copy(
                update={
                    "title": "Context scene",
                    "beat_change": "The argument advances.",
                    "passage_ids": ["p2"],
                    "estimated_duration_seconds": 2100,
                }
            ),
        ],
        handoff_scene_card_id="scene_anchor",
    )
    strategy_episode = _strategy_episode("pack_1")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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

    asyncio.run(
        orchestrator._write_episode(
            plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
        )
    )

    scene_targets = {
        scene["scene_id"]: (
            scene["target_word_count_lower"],
            scene["target_word_count_higher"],
        )
        for scene in captured["payload"]["plan"]["scene_cards"]
    }
    assert scene_targets == {
        "scene_anchor": (14411, 17614),
        "scene_context": (4803, 5872),
    }
    assert captured["payload"]["episode_target_word_count_lower"] == 19214
    assert captured["payload"]["episode_target_word_count_higher"] == 23486


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
    plan = _episode_plan(
        [
            _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
                update={"estimated_duration_seconds": 300, "scene_job": "answer"}
            )
        ]
    )
    strategy_episode = _strategy_episode("pack_1")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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
    script = asyncio.run(
        orchestrator._write_episode(
            plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
        )
    )

    payload = captured["payload"]
    assert payload["skip_grounding"] is True
    assert "scene_word_count_targets" not in payload
    assert "previous_sections" not in payload
    assert payload["episode_target_word_count_lower"] == 686
    assert payload["episode_target_word_count_higher"] == 839
    scene = payload["plan"]["scene_cards"][0]
    assert scene["scene_id"] == "scene_1"
    assert "estimated_duration_seconds" not in scene
    assert scene["target_word_count_lower"] == 686
    assert scene["target_word_count_higher"] == 839
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
    assert budget_warnings == []
    assert section_count_warnings == []


def test_normalize_writing_section_outputs_backfills_compact_section_metadata():
    scene_cards = [
        _scene_card("scene_1", "pack_1", section_id="section_1"),
        _scene_card("scene_2", "pack_2", section_id="section_2"),
    ]
    architecture = _episode_architecture_for_scene_cards(scene_cards)
    result = SimpleNamespace(
        prose_sections=[
            SimpleNamespace(text="Opening text."),
            SimpleNamespace(text="Closing text."),
        ]
    )

    normalized = _normalize_writing_section_outputs(
        result=result,
        architecture=architecture,
        scene_cards=scene_cards,
        episode_number=1,
        skip_grounding=True,
    )

    assert normalized[0]["section_id"] == "section_1"
    assert normalized[0]["scene_card_ids"] == ["scene_1"]
    assert normalized[0]["movement_goal"] == "opening"
    assert normalized[1]["section_id"] == "section_2"
    assert normalized[1]["scene_card_ids"] == ["scene_2"]
    assert normalized[1]["movement_goal"] == "closing"


def test_write_episode_retries_on_scene_id_contract_failure(monkeypatch, tmp_path):
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

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)

    orchestrator = PipelineOrchestrator()
    captured_payloads: list[dict[str, object]] = []

    def fake_writing_run(payload: dict):
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return orchestrator.writing_agent.response_model.model_validate(
                {
                    "prose_sections": [
                        {
                            "section_id": "section_1",
                            "scene_card_ids": ["scene_renamed"],
                            "movement_goal": "discover",
                            "text": "Draft text.",
                            "source_book_ids": [],
                        }
                    ]
                }
            )
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_1"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                        "source_book_ids": [],
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
    plan = _episode_plan(
        [
            _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
                update={"estimated_duration_seconds": 300, "scene_job": "answer"}
            )
        ]
    )
    strategy_episode = _strategy_episode("pack_1")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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

    script = asyncio.run(
        orchestrator._write_episode(
            plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
        )
    )

    assert len(captured_payloads) == 2
    assert "writing_feedback" not in captured_payloads[0]
    assert "writing_feedback" in captured_payloads[1]
    assert "scene_renamed" in str(captured_payloads[1]["writing_feedback"])
    assert script.prose_sections[0].scene_card_ids == ["scene_1"]
    assert (ep_dir / "episode_script.json").exists()


def test_write_episode_raises_after_retry_exhaustion_on_scene_id_contract_failure(
    monkeypatch, tmp_path,
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

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)

    orchestrator = PipelineOrchestrator()
    call_count = 0

    def fake_writing_run(_payload: dict):
        nonlocal call_count
        call_count += 1
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "prose_sections": [
                    {
                        "section_id": "section_1",
                        "scene_card_ids": ["scene_renamed"],
                        "movement_goal": "discover",
                        "text": "Draft text.",
                        "source_book_ids": [],
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
    plan = _episode_plan(
        [
            _scene_card("scene_1", "pack_1", scene_role="setup").model_copy(
                update={"estimated_duration_seconds": 300, "scene_job": "answer"}
            )
        ]
    )
    strategy_episode = _strategy_episode("pack_1")
    architecture = _episode_architecture_for_scene_cards(plan.scene_cards)
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

    with pytest.raises(
        RuntimeError,
        match="expected .*scene_1.*received .*scene_renamed",
    ):
        asyncio.run(
            orchestrator._write_episode(
                plan, strategy_episode, architecture, project, corpus, ep_dir, tmp_path
            )
        )

    assert call_count == orchestrator.writing_agent.max_retry_attempts
    assert not (ep_dir / "episode_script.json").exists()


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


def test_split_episode_writing_windows_uses_three_balanced_batches_when_available():
    scene_cards = [
        _scene_card(
            f"scene_{idx}",
            "pack_1",
            section_id=f"section_{idx}",
        )
        for idx in range(1, 10)
    ]
    plan = _episode_plan(scene_cards, target_word_count=9450)
    architecture = _episode_architecture_for_scene_cards(scene_cards)
    lower_targets = {scene.scene_id: 650 for scene in scene_cards}
    higher_targets = {scene.scene_id: 750 for scene in scene_cards}

    windows = _split_episode_writing_windows(
        plan=plan,
        architecture=architecture,
        scene_word_count_targets_lower=lower_targets,
        scene_word_count_targets_higher=higher_targets,
        max_windows=3,
    )

    assert [[scene.scene_id for scene in window] for window in windows] == [
        ["scene_1", "scene_2", "scene_3"],
        ["scene_4", "scene_5", "scene_6"],
        ["scene_7", "scene_8", "scene_9"],
    ]


def test_split_episode_writing_windows_falls_back_when_requested_windows_exceed_boundaries():
    scene_cards = [
        _scene_card(
            f"scene_{idx}",
            "pack_1",
            section_id=f"section_{idx}",
        )
        for idx in range(1, 3)
    ]
    plan = _episode_plan(scene_cards, target_word_count=1500)
    architecture = _episode_architecture_for_scene_cards(scene_cards)
    lower_targets = {scene.scene_id: 650 for scene in scene_cards}
    higher_targets = {scene.scene_id: 750 for scene in scene_cards}

    windows = _split_episode_writing_windows(
        plan=plan,
        architecture=architecture,
        scene_word_count_targets_lower=lower_targets,
        scene_word_count_targets_higher=higher_targets,
        max_windows=3,
    )

    assert [[scene.scene_id for scene in window] for window in windows] == [
        ["scene_1"],
        ["scene_2"],
    ]


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
        text = str(payload["section"]["text"])
        return orchestrator.spoken_delivery_agent.response_model.model_validate(
            {
                "text": f"spoken::{text}.",
                "speech_hints": {
                    "style": "measured",
                    "intensity": "light",
                    "pause_before_ms": 100,
                    "pause_after_ms": 200,
                    "pace": "slower",
                    "pronunciation_hints": [
                        {"text": "Panipat", "spoken_as": "PAH-nee-puht"},
                    ],
                    "emphasis_targets": ["spoken"],
                    "render_strategy": "plain",
                },
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
    plan = EpisodePlan.model_construct(
        episode_number=1,
        framing=_framing().model_copy(update={"handoff_scene_card_id": "scene_1"}),
        scene_cards=[
            _scene_card("scene_1", "pack_1", section_id="section_1"),
            _scene_card("scene_2", "pack_1", section_id="section_2"),
        ],
        target_word_count=120,
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
        orchestrator._rewrite_for_speech(
            1,
            script,
            project,
            ep_dir,
            tmp_path,
            plan=plan,
        )
    )

    assert len(payloads) == 2
    assert payloads[0]["section"]["section_id"] == "section_1"
    assert payloads[1]["section"]["section_id"] == "section_2"
    assert "batch_index" not in payloads[0]
    assert "batch_index" not in payloads[1]
    assert "batch_count" not in payloads[0]
    assert "batch_count" not in payloads[1]
    assert "previous_spoken_text" not in payloads[0]
    assert "previous_spoken_text" not in payloads[1]
    assert "previous_spoken_tail" not in payloads[0]
    assert payloads[1]["previous_spoken_tail"] == "spoken::First section."
    assert payloads[0]["section"]["scene_cues"][0]["audible_detail"] == "The stamp snaps down."
    assert "upcoming_batches_summary" not in payloads[0]
    assert "upcoming_batches_summary" not in payloads[1]
    assert [section.section_id for section in spoken.sections] == ["section_1", "section_2"]
    assert [section.text for section in spoken.sections] == [
        "spoken::First section.",
        "spoken::Second section.",
    ]
    assert spoken.sections[0].speech_hints.style == "measured"
    assert spoken.sections[0].speech_hints.pronunciation_hints[0].text == "Panipat"
    assert spoken.sections[0].sonic_cues[0].audible_detail == "The stamp snaps down."


def test_extract_previous_spoken_tail_prefers_last_four_complete_sentences():
    tail = _extract_previous_spoken_tail(
        "Alpha starts here. Beta continues with detail. Gamma ends the run. Delta widens the point. Epsilon closes it."
    )
    assert tail == (
        "Beta continues with detail. Gamma ends the run. Delta widens the point. Epsilon closes it."
    )


def test_build_spoken_delivery_batches_uses_four_batches_when_four_sections_are_available():
    sections = [
        ProseSection(
            section_id=f"section_{idx}",
            scene_card_ids=[f"scene_{idx}"],
            movement_goal=f"Goal {idx}",
            text=" ".join(["word"] * word_count),
        )
        for idx, word_count in enumerate([100, 200, 300, 400], start=1)
    ]

    batches = _build_spoken_delivery_batches(sections)

    assert [[section.section_id for section in batch] for batch in batches] == [
        ["section_1"],
        ["section_2"],
        ["section_3"],
        ["section_4"],
    ]


def test_spoken_delivery_payload_includes_scene_cues_from_plan():
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
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            ArchitectureSection(
                section_id="section_1",
                purpose="opening",
                approx_runtime_minutes=1.0,
                primitive_ids=["pack_1"],
                section_anchor="Anchor",
                must_stage_beats=["Beat one", "Beat two"],
            )
        ],
        architecture_notes=[],
    )
    plan = EpisodePlan.model_construct(
        episode_number=1,
        framing=_framing().model_copy(update={"handoff_scene_card_id": "scene_1"}),
        scene_cards=[_scene_card("scene_1", "pack_1")],
        target_word_count=120,
    )

    payload_sections = _build_spoken_delivery_sections_payload(
        script=script,
        architecture=architecture,
        plan=plan,
    )

    assert payload_sections[0]["scene_cues"] == [
        {
            "scene_id": "scene_1",
            "scene_job": "build",
            "entry_image": "Image",
            "observable_detail": "Detail",
            "audible_detail": "The stamp snaps down.",
        }
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
            {"speech_hints": {"style": "neutral"}}
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

    with pytest.raises(ValidationError, match="text"):
        asyncio.run(orchestrator._rewrite_for_speech(1, script, project, ep_dir, tmp_path))


def test_produce_episode_releases_write_slot_before_spoken_delivery(monkeypatch, tmp_path):
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
    event_log: list[tuple[str, int, float]] = []
    lock = threading.Lock()

    async def fake_write_episode(
        plan: EpisodePlan,
        strategy_episode: StrategyEpisode,
        architecture: EpisodeArchitecture,
        project: ThematicProject,
        corpus: ThematicCorpus,
        ep_dir: Path,
        project_dir: Path,
        actor_metadata: ActorMetadata | None = None,
        host_policy: dict[str, Any] | None = None,
        primitive_lookup: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> EpisodeScript:
        with lock:
            event_log.append(("write_start", plan.episode_number, time.monotonic()))
        await asyncio.sleep(0.03)
        with lock:
            event_log.append(("write_end", plan.episode_number, time.monotonic()))
        return EpisodeScript(
            episode_number=plan.episode_number,
            title=f"Episode {plan.episode_number}",
            framing=_framing(),
            prose_sections=[
                ProseSection(
                    section_id="section_1",
                    scene_card_ids=["scene_1"],
                    movement_goal="discover",
                    text="Draft section.",
                )
            ],
        )

    async def fake_rewrite_for_speech(
        episode_number: int,
        script: EpisodeScript,
        project: ThematicProject,
        ep_dir: Path,
        project_dir: Path,
        **_kwargs: Any,
    ) -> SpokenScript:
        with lock:
            event_log.append(("spoken_start", episode_number, time.monotonic()))
        await asyncio.sleep(0.20 if episode_number == 1 else 0.02)
        with lock:
            event_log.append(("spoken_end", episode_number, time.monotonic()))
        return SpokenScript(
            episode_number=episode_number,
            title=script.title,
            framing=script.framing,
            sections=[SpokenSection(section_id="section_1", text="spoken section")],
            tts_provider=project.config.tts_provider,
        )

    orchestrator._write_episode = fake_write_episode  # type: ignore[method-assign]
    orchestrator._rewrite_for_speech = fake_rewrite_for_speech  # type: ignore[method-assign]

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
        config=PipelineConfig(
            skip_grounding=True,
            episode_write_concurrency=2,
            spoken_delivery_concurrency=1,
        ),
    )
    corpus = ThematicCorpus(project_id="proj")
    actor_metadata = ActorMetadata(project_id="proj")
    strategy_episode = StrategyEpisode(
        episode_number=1,
        title="Episode",
        arc_summary="Arc summary",
        episode_spine=_episode_spine("pack_1"),
        actor_arc_directives=[],
    )
    architecture = EpisodeArchitecture.model_construct(
        episode_number=1,
        major_turn_section_id="section_1",
        sections=[],
    )
    plans = [
        _episode_plan(
            [_scene_card("scene_1", "pack_1").model_copy(update={"scene_job": "answer"})],
            episode_number=episode_number,
            target_word_count=120,
        )
        for episode_number in (1, 2, 3)
    ]

    async def _run_production() -> None:
        write_sem = asyncio.Semaphore(2)
        spoken_sem = asyncio.Semaphore(1)
        await asyncio.gather(
            *[
                orchestrator._produce_episode(
                    plan,
                    strategy_episode.model_copy(
                        update={"episode_number": plan.episode_number}
                    ),
                    architecture.model_copy(update={"episode_number": plan.episode_number}),
                    project,
                    corpus,
                    actor_metadata,
                    tmp_path,
                    host_policy={},
                    primitive_lookup={},
                    semaphore=write_sem,
                    spoken_semaphore=spoken_sem,
                )
                for plan in plans
            ]
        )

    asyncio.run(_run_production())

    write_start_ep3 = next(
        timestamp
        for event, episode_number, timestamp in event_log
        if event == "write_start" and episode_number == 3
    )
    spoken_end_ep1 = next(
        timestamp
        for event, episode_number, timestamp in event_log
        if event == "spoken_end" and episode_number == 1
    )
    assert write_start_ep3 < spoken_end_ep1


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
