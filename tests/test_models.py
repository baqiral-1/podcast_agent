"""Unit tests for the redesigned schema contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcThread,
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    ArchitectureSection,
    CandidateReading,
    ChapterAnalysis,
    EvidencePack,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodePlanDraft,
    EpisodeSpine,
    ContestedExplanationPrimitive,
    EpochalTurnPrimitive,
    FramingBlock,
    HostPresenceBeat,
    HostMoveCue,
    HostMovesByPhase,
    NarrativeStrategy,
    ProseSection,
    SceneActor,
    SceneActorArcBinding,
    SceneCard,
    SceneCardDraft,
    SeriesNarratorProfile,
    SpineRelation,
    SpeechHints,
    SpokenScript,
    SpokenSection,
    StrategyEpisode,
    SupportPackRole,
    SYNTHESIS_PRIMITIVE_FAMILIES,
    SynthesisConsolidationResult,
    SynthesisMap,
    SynthesisPrimitiveBase,
    SynthesisPrimitivesArtifact,
    ThematicProject,
    PipelineConfig,
    PodcastMode,
    VerdictMode,
    resolve_pipeline_config_for_mode,
    synthesis_primitive_target_ranges_for_mode,
    validate_episode_architecture_targets,
    validate_episode_spine_targets,
)
from podcast_agent.utils.actor_metadata import (
    ActorMatcher,
    clean_scene_actor_links,
    clean_synthesis_primitive_actor_links,
    compact_consolidation_actor_metadata,
    normalize_actor_name,
    sanitize_actor_metadata_payload,
)


def _epochal_turn(primitive_id: str = "et_1") -> EpochalTurnPrimitive:
    return EpochalTurnPrimitive(
        id=primitive_id,
        family="epochal_turns",
        title="Threshold breaks",
        summary="A decision changes the field.",
        axis_ids=["axis_1"],
        core_passage_ids=["p1"],
        support_passage_ids=["p2"],
        before_state="The prior balance still holds.",
        after_state="A new political order is now in effect.",
        change_driver="A decision forces the transition.",
        irreversibility_reason="The institutions and actors cannot easily return to the old arrangement.",
    )


def _contested_explanation(primitive_id: str = "cx_1") -> ContestedExplanationPrimitive:
    return ContestedExplanationPrimitive(
        id=primitive_id,
        family="contested_explanations",
        title="Competing readings",
        summary="The evidence supports multiple paths.",
        core_passage_ids=["p1"],
        candidate_readings=[
            CandidateReading(
                label="reading_a",
                summary="One reading.",
                support_passage_ids=["p1"],
            ),
            CandidateReading(
                label="reading_b",
                summary="Another reading.",
                support_passage_ids=["p2"],
            ),
        ],
    )


def _family_map(
    **overrides: list[SynthesisPrimitiveBase],
) -> dict[str, list[SynthesisPrimitiveBase]]:
    payload = {family: [] for family in SYNTHESIS_PRIMITIVE_FAMILIES}
    payload.update(overrides)
    return payload


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A convoy moves at dawn.",
        threat_or_unresolved_action="Nobody yet knows what the order will trigger.",
        opening_question="Why does this decision land so hard?",
        handoff_scene_card_id="scene_1",
        recap="Previously on the series.",
        preview="Next, the fallout spreads.",
    )


def _host_moves_payload() -> HostMovesByPhase:
    return HostMovesByPhase(
        open=[
            HostMoveCue(
                move_type="orient",
                note="Set the listener's footing before the scene turns.",
            )
        ]
    )


def _normal_scene(scene_id: str = "scene_1", pack_id: str = "pack_1") -> SceneCard:
    return SceneCard(
        scene_id=scene_id,
        section_id="section_1",
        title="The order arrives",
        scene_role="setup",
        dominant_pack_id=pack_id,
        spine_relation=SpineRelation.SET_STAKES,
        state_effect="The stakes become legible.",
        entry_image="A clerk opens the envelope.",
        local_question="What changes first?",
        observable_detail="Hands freeze over the paper.",
        intended_move="Move from abstract policy to lived consequence.",
        actors=[SceneActor(name="Clerk", presence="background")],
        primitive_ids=["et_1"],
        passage_ids=["p1", "p2"],
        host_moves=_host_moves_payload(),
        estimated_duration_seconds=600,
    )


def _normal_scene_draft(
    scene_id: str = "scene_1",
    pack_id: str = "pack_1",
) -> SceneCardDraft:
    return SceneCardDraft(
        scene_id=scene_id,
        section_id="section_1",
        title="The order arrives",
        scene_role="setup",
        dominant_pack_id=pack_id,
        spine_relation=SpineRelation.SET_STAKES,
        state_effect="The stakes become legible.",
        entry_image="A clerk opens the envelope.",
        local_question="What changes first?",
        observable_detail="Hands freeze over the paper.",
        intended_move="Move from abstract policy to lived consequence.",
        actors=[SceneActor(name="Clerk", presence="background")],
        primitive_ids=["et_1"],
        passage_ids=["p1", "p2"],
        host_moves=_host_moves_payload(),
    )


def _episode_spine(pack_id: str = "pack_1") -> EpisodeSpine:
    core_primitive_ids = [pack_id, *[f"core_{idx}" for idx in range(2, 8)]]
    support_primitive_roles = {
        f"support_{idx}": SupportPackRole.MECHANISM
        if idx % 2 == 0
        else SupportPackRole.TEXTURE
        for idx in range(1, 8)
    }
    return EpisodeSpine(
        listener_question="Why does this decision land so hard?",
        argument="This episode tests one controlling proposition.",
        core_primitive_ids=core_primitive_ids,
        support_primitive_roles=support_primitive_roles,
        recall_primitive_ids=[],
    )


class TestThematicProject:
    def test_recommended_episode_count_is_range_agnostic(self):
        project = ThematicProject(theme="War on terror", recommended_episode_count=7)
        assert project.recommended_episode_count == 7

        project = ThematicProject(theme="War on terror", recommended_episode_count=13)
        assert project.recommended_episode_count == 13

    def test_recommended_episode_count_rejects_zero(self):
        with pytest.raises(ValidationError):
            ThematicProject(theme="War on terror", recommended_episode_count=0)

    def test_pipeline_config_exposes_narrative_strategy_episode_count_defaults(self):
        config = PipelineConfig()
        assert config.podcast_mode == PodcastMode.FULL
        assert config.min_axes == 12
        assert config.max_axes == 20
        assert config.synthesis_axis_min == 12
        assert config.synthesis_axis_max == 20
        assert config.narrative_strategy_episode_count_min == 8
        assert config.narrative_strategy_episode_count_max == 12

    def test_pipeline_config_rejects_narrative_strategy_episode_count_bound_inversion(
        self,
    ):
        with pytest.raises(
            ValidationError, match="narrative_strategy_episode_count_max"
        ):
            PipelineConfig(
                narrative_strategy_episode_count_min=12,
                narrative_strategy_episode_count_max=8,
            )

    def test_pipeline_config_rejects_scene_card_bound_inversion(self):
        with pytest.raises(ValidationError, match="scene_card_target_max"):
            PipelineConfig(scene_card_target_min=40, scene_card_target_max=25)

    def test_pipeline_config_rejects_removed_scene_batch_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(scene_batch_min_cards=1)

    def test_pipeline_config_defaults_synthesis_cap_to_450(self):
        config = PipelineConfig()
        assert config.synthesis_total_passage_cap == 450
        assert config.synthesis_floor_budget_fraction == 0.0
        assert config.synthesis_axis_floor_min == 0
        assert config.synthesis_axis_floor_max == 0
        assert config.synthesis_axis_ceiling_multiplier == 1.68
        assert config.synthesis_trim_top_fraction == 0.10
        assert config.synthesis_trim_mid_fraction == 0.20
        assert config.synthesis_trim_next_fraction == 0.0
        assert config.synthesis_trim_top_keep_fraction == 0.375
        assert config.synthesis_trim_mid_keep_fraction == 0.275
        assert config.synthesis_trim_next_keep_fraction == 0.325
        assert config.synthesis_trim_tail_keep_fraction == 0.175
        assert config.passage_extraction_concurrency == 16
        assert config.episode_write_concurrency == 8
        assert config.spoken_delivery_concurrency == 8
        assert config.min_episode_minutes == 90.0
        assert config.max_episode_minutes == 105.0
        assert config.architecture_section_target_min == 9
        assert config.architecture_section_target_max == 12
        assert config.episode_spine_core_primitive_target_min == 5
        assert config.episode_spine_core_primitive_target_max == 7
        assert config.episode_spine_support_primitive_target_min == 5
        assert config.episode_spine_support_primitive_target_max == 7
        assert config.episode_spine_recall_primitive_target_max == 2
        assert config.scene_card_target_min == 32
        assert config.scene_card_target_max == 40

    def test_resolve_pipeline_config_for_mode_applies_minified_profile(self):
        config = resolve_pipeline_config_for_mode(
            PipelineConfig(podcast_mode=PodcastMode.MINIFIED)
        )
        assert config.podcast_mode == PodcastMode.MINIFIED
        assert config.min_axes == 4
        assert config.max_axes == 6
        assert config.pre_axis_total_budget == 500
        assert config.synthesis_axis_min == 4
        assert config.synthesis_axis_max == 6
        assert config.narrative_strategy_episode_count_min == 2
        assert config.narrative_strategy_episode_count_max == 4
        assert config.synthesis_total_passage_cap == 200
        assert config.episode_spine_core_primitive_target_min == 2
        assert config.episode_spine_core_primitive_target_max == 4
        assert config.episode_spine_support_primitive_target_min == 2
        assert config.episode_spine_support_primitive_target_max == 4
        assert config.episode_spine_recall_primitive_target_max == 1
        assert config.architecture_section_target_min == 6
        assert config.architecture_section_target_max == 8
        assert config.min_episode_minutes == 54.0
        assert config.max_episode_minutes == 63.0
        assert config.scene_card_target_min == 21
        assert config.scene_card_target_max == 26

    def test_synthesis_primitive_target_ranges_for_minified_mode_floor_by_thirds(self):
        target_ranges = synthesis_primitive_target_ranges_for_mode(PodcastMode.MINIFIED)
        assert target_ranges["epochal_turns"] == (10, 12)
        assert target_ranges["telling_details"] == (1, 2)
        assert target_ranges["perspective_windows"] == (0, 1)
        assert target_ranges["recurring_images_and_symbols"] == (0, 1)

    def test_thematic_project_rejects_more_than_thirty_sub_themes(self):
        with pytest.raises(ValidationError, match="at most 30 entries"):
            ThematicProject(
                theme="Theme",
                sub_themes=[f"s{i}" for i in range(31)],
            )

    def test_chapter_analysis_discards_legacy_fields(self):
        analysis = ChapterAnalysis.model_validate(
            {
                "themes_touched": ["theme"],
                "major_actors": ["actor"],
                "key_events_or_arguments": ["event"],
                "key_places": ["place"],
                "key_institutions": ["institution"],
                "timeframe": "2001",
                "major_tensions": ["tension"],
            }
        )
        assert analysis.model_dump(mode="json") == {
            "themes_touched": ["theme"],
            "major_actors": ["actor"],
            "key_events_or_arguments": ["event"],
        }

    def test_series_narrator_profile_defaults_to_phase_coverage_targets(self):
        profile = SeriesNarratorProfile()
        assert profile.target_full_phase_scene_coverage_min == 0.60
        assert profile.target_full_phase_scene_coverage_target == 0.75
        assert profile.analysis_mode == "hybrid"
        assert profile.analysis_density == "medium"
        assert profile.target_authorial_passages_per_episode == 16

    def test_host_presence_beat_normalizes_legacy_term_reminder(self):
        beat = HostPresenceBeat.model_validate(
            {
                "kind": "term_reminder",
                "placement": "pivot",
                "seed": "Briefly restate the term in spoken language.",
            }
        )

        assert beat.kind == "clarify"
        assert beat.scope == "scene"
        assert beat.address_mode == "implicit"

    def test_host_moves_require_at_least_one_phase_bucket(self):
        with pytest.raises(ValidationError, match="at least one phase bucket"):
            HostMovesByPhase()


class TestSynthesisModels:
    def test_actor_metadata_validates_relationship_references(self):
        metadata = ActorMetadata(
            project_id="proj",
            actors=[
                ActorProfile(
                    actor_id="jawaharlal_nehru",
                    display_name="Jawaharlal Nehru",
                    actor_type="person",
                    narrative_importance_score=0.9,
                ),
                ActorProfile(
                    actor_id="indian_national_congress",
                    display_name="Indian National Congress",
                    actor_type="party",
                    narrative_importance_score=0.8,
                ),
            ],
            relationships=[
                ActorRelationship(
                    source_actor_id="jawaharlal_nehru",
                    target_actor_id="indian_national_congress",
                    relationship_type="legitimizes",
                )
            ],
        )
        assert metadata.relationships[0].relationship_type == "legitimizes"

    def test_actor_metadata_rejects_unknown_relationship_actor(self):
        with pytest.raises(ValidationError, match="target_actor_id"):
            ActorMetadata(
                actors=[
                    ActorProfile(
                        actor_id="nehru",
                        display_name="Nehru",
                        actor_type="person",
                    )
                ],
                relationships=[
                    ActorRelationship(
                        source_actor_id="nehru",
                        target_actor_id="missing",
                        relationship_type="pressures",
                    )
                ],
            )

    def test_actor_name_matching_exact_and_fuzzy(self):
        metadata = ActorMetadata(
            actors=[
                ActorProfile(
                    actor_id="lord_mountbatten",
                    display_name="Lord Mountbatten",
                    aliases=["Mountbatten"],
                    actor_type="person",
                )
            ]
        )
        matcher = ActorMatcher(metadata)
        assert normalize_actor_name(" Lord Mountbatten! ") == "lord mountbatten"
        assert matcher.match("Mountbatten").actor_id == "lord_mountbatten"
        fuzzy = matcher.match("Lord Mountbatte")
        assert fuzzy is not None
        assert fuzzy.match_type == "fuzzy"

    def test_sanitize_actor_metadata_caps_and_normalizes_relationship_types(self):
        raw = {
            "actors": [
                {
                    "actor_id": f"actor_{idx}",
                    "display_name": f"Actor {idx}",
                    "actor_type": "state" if idx == 0 else "person",
                    "narrative_importance_score": 1.0 - (idx * 0.01),
                }
                for idx in range(45)
            ],
            "relationships": [
                {
                    "source_actor_id": "actor_0",
                    "target_actor_id": "actor_1",
                    "relationship_type": "supports",
                },
                {
                    "source_actor_id": "actor_0",
                    "target_actor_id": "actor_999",
                    "relationship_type": "pressures",
                },
            ],
        }
        metadata, metrics = sanitize_actor_metadata_payload(raw, project_id="proj")
        assert len(metadata.actors) == 40
        assert metadata.actors[0].actor_type == "other"
        assert metadata.relationships[0].relationship_type == "other"
        assert metrics["dropped_over_cap_actor_count"] == 5
        assert metrics["dropped_relationship_count"] == 1

    def test_sanitize_actor_metadata_recovers_theme_decomposition_legacy_shape(self):
        raw = {
            "actors": [
                {
                    "actor_id": "mahatma_gandhi",
                    "display_name": "Mohandas Karamchand Gandhi",
                    "actor_type": "person",
                    "aliases": ["Mahatma Gandhi"],
                    "description": "Central architect of mass nationalist politics.",
                    "motivations": "Preserve nonviolence and Hindu-Muslim unity.",
                    "uncertainty_notes": "Evidence emphasizes public pressure rather than private intent.",
                    "evidence_confidence": 1.0,
                    "narrative_importance_score": 1.0,
                    "relationships": [
                        {
                            "target_actor_id": "indian_national_congress",
                            "relationship_type": "legitimizes",
                            "description": "Provided moral authority.",
                        }
                    ],
                },
                {
                    "actor_id": "indian_national_congress",
                    "display_name": "Indian National Congress",
                    "actor_type": "party",
                    "description": "Primary nationalist party.",
                    "evidence_confidence": 0.6,
                    "narrative_importance_score": 0.9,
                    "relationships": [
                        {
                            "source_actor_id": "indian_national_congress",
                            "target_actor_id": "mahatma_gandhi",
                            "relationship_type": "enables",
                            "description": "Provided organizational infrastructure.",
                        }
                    ],
                },
                {
                    "actor_id": "british_raj_viceroyalty",
                    "display_name": "British Raj Viceroyalty",
                    "actor_type": "institution",
                    "description": "Imperial executive authority in India.",
                    "evidence_confidence": 0.25,
                    "narrative_importance_score": 0.8,
                },
            ],
        }

        metadata, metrics = sanitize_actor_metadata_payload(raw, project_id="proj")

        assert [actor.actor_id for actor in metadata.actors] == [
            "mahatma_gandhi",
            "indian_national_congress",
            "british_raj_viceroyalty",
        ]
        assert [actor.evidence_confidence for actor in metadata.actors] == [
            "high",
            "medium",
            "low",
        ]
        assert metadata.actors[0].goals_or_motivational_pressures == [
            "Preserve nonviolence and Hindu-Muslim unity."
        ]
        assert (
            metadata.actors[0].uncertainty_notes
            == "Evidence emphasizes public pressure rather than private intent."
        )
        assert len(metadata.relationships) == 2
        assert metadata.relationships[0].source_actor_id == "mahatma_gandhi"
        assert metadata.relationships[0].target_actor_id == "indian_national_congress"
        assert metrics["raw_actor_count"] == 3
        assert metrics["dropped_actor_count"] == 0
        assert metrics["evidence_confidence_normalized_count"] == 3
        assert metrics["motivations_converted_count"] == 1
        assert metrics["nested_relationship_flattened_count"] == 2

    def test_sanitize_actor_metadata_drops_invalid_narrative_functions_not_actor(self):
        raw = {
            "actors": [
                {
                    "actor_id": "louis_mountbatten",
                    "display_name": "Louis Mountbatten",
                    "actor_type": "person",
                    "narrative_functions": ["decision_maker", "imperial_showman"],
                    "narrative_importance_score": 0.9,
                }
            ]
        }

        metadata, metrics = sanitize_actor_metadata_payload(raw, project_id="proj")

        assert [actor.actor_id for actor in metadata.actors] == ["louis_mountbatten"]
        assert metadata.actors[0].narrative_functions == ["decision_maker"]
        assert metrics["dropped_actor_count"] == 0
        assert metrics["invalid_narrative_function_dropped_count"] == 1

    def test_sanitize_actor_metadata_keeps_actor_when_all_narrative_functions_invalid(
        self,
    ):
        raw = {
            "actors": [
                {
                    "actor_id": "louis_mountbatten",
                    "display_name": "Louis Mountbatten",
                    "actor_type": "person",
                    "narrative_functions": ["imperial_showman", "deadline_driver"],
                    "narrative_importance_score": 0.9,
                }
            ]
        }

        metadata, metrics = sanitize_actor_metadata_payload(raw, project_id="proj")

        assert [actor.actor_id for actor in metadata.actors] == ["louis_mountbatten"]
        assert metadata.actors[0].narrative_functions == []
        assert metrics["dropped_actor_count"] == 0
        assert metrics["invalid_narrative_function_dropped_count"] == 2

    def test_synthesis_primitive_actor_tags_move_to_ids_and_unresolved(self):
        metadata = ActorMetadata(
            actors=[
                ActorProfile(
                    actor_id="jawaharlal_nehru",
                    display_name="Jawaharlal Nehru",
                    aliases=["Nehru"],
                    actor_type="person",
                )
            ]
        )
        artifact = SynthesisPrimitivesArtifact(
            project_id="proj",
            primitives_by_family=_family_map(
                epochal_turns=[
                    _epochal_turn("et_1").model_copy(
                        update={"actor_tags": ["Nehru", "Unknown actor"]}
                    )
                ]
            ),
        )
        cleaned, metrics = clean_synthesis_primitive_actor_links(artifact, metadata)
        primitive = cleaned.primitives_by_family["epochal_turns"][0]
        assert primitive.actor_ids == ["jawaharlal_nehru"]
        assert primitive.actor_tags == []
        assert primitive.unresolved_actor_tags == ["Unknown actor"]
        assert metrics["exact_actor_tag_matches"] == 1
        assert metrics["unmatched_actor_tags"] == 1

    def test_compact_consolidation_actor_metadata_keeps_minimal_actor_and_relationship_fields(
        self,
    ):
        metadata = ActorMetadata(
            actors=[
                ActorProfile(
                    actor_id="jawaharlal_nehru",
                    display_name="Jawaharlal Nehru",
                    aliases=["Nehru"],
                    actor_type="person",
                    description="Prime minister.",
                    book_ids=["book_1"],
                    goals_or_motivational_pressures=["Hold the union together"],
                ),
                ActorProfile(
                    actor_id="vallabhbhai_patel",
                    display_name="Vallabhbhai Patel",
                    actor_type="person",
                    description="Home minister.",
                ),
            ],
            relationships=[
                ActorRelationship(
                    source_actor_id="jawaharlal_nehru",
                    target_actor_id="vallabhbhai_patel",
                    relationship_type="other",
                    description="They share a cabinet and a rivalry.",
                    confidence="high",
                )
            ],
            quality_notes=["extra context"],
        )

        payload = compact_consolidation_actor_metadata(metadata)

        assert payload == {
            "actors": [
                {
                    "actor_id": "jawaharlal_nehru",
                    "display_name": "Jawaharlal Nehru",
                },
                {
                    "actor_id": "vallabhbhai_patel",
                    "display_name": "Vallabhbhai Patel",
                },
            ],
            "relationships": [
                {
                    "source_actor_id": "jawaharlal_nehru",
                    "target_actor_id": "vallabhbhai_patel",
                    "relationship_type": "other",
                    "description": "They share a cabinet and a rivalry.",
                }
            ],
        }

    def test_importance_fields_default_and_roundtrip(self):
        primitive = _epochal_turn("et_1")
        pack = EvidencePack(
            pack_id="pack_1",
            title="Pack",
            local_summary="Summary",
            primitive_ids=["et_1"],
        )

        assert primitive.narrative_importance_score == 0.5
        assert pack.primitive_ids == ["et_1"]

        restored = EvidencePack.model_validate(
            {
                **pack.model_dump(mode="json"),
                "actor_ids": ["actor_1", "actor_1", ""],
            }
        )
        assert restored.actor_ids == ["actor_1"]

    def test_synthesis_map_rejects_unknown_pack_primitive_ids(self):
        with pytest.raises(ValidationError, match="unknown primitive_ids"):
            SynthesisMap(
                project_id="proj",
                primitives_by_family=_family_map(epochal_turns=[_epochal_turn("et_1")]),
                evidence_packs=[
                    EvidencePack(
                        pack_id="pack_1",
                        title="Pack",
                        local_summary="Summary",
                        primitive_ids=["et_1", "tp_999"],
                    )
                ],
            )

    def test_contested_explanations_with_fewer_than_two_candidate_readings_are_dropped(
        self,
    ):
        invalid_contested_explanation = ContestedExplanationPrimitive(
            id="cx_invalid",
            family="contested_explanations",
            title="Single reading",
            summary="The evidence supports only one explicit reading.",
            core_passage_ids=["p1"],
            candidate_readings=[
                CandidateReading(
                    label="reading_a",
                    summary="Only one reading present.",
                    support_passage_ids=["p1"],
                )
            ],
        )

        artifact = SynthesisPrimitivesArtifact(
            project_id="proj",
            primitives_by_family=_family_map(
                epochal_turns=[_epochal_turn("et_1")],
                contested_explanations=[
                    invalid_contested_explanation,
                    _contested_explanation("cx_valid"),
                ],
            ),
        )
        synthesis_map = SynthesisMap(
            project_id="proj",
            primitives_by_family=_family_map(
                epochal_turns=[_epochal_turn("et_1")],
                contested_explanations=[
                    invalid_contested_explanation,
                    _contested_explanation("cx_valid"),
                ],
            ),
        )

        assert [
            item.id for item in artifact.primitives_by_family["contested_explanations"]
        ] == ["cx_valid"]
        assert [
            item.id
            for item in synthesis_map.primitives_by_family["contested_explanations"]
        ] == ["cx_valid"]
        assert [item.id for item in artifact.primitives_by_family["epochal_turns"]] == [
            "et_1"
        ]

    def test_synthesis_map_rejects_pack_references_to_dropped_contested_explanations(
        self,
    ):
        with pytest.raises(ValidationError, match="unknown primitive_ids"):
            SynthesisMap(
                project_id="proj",
                primitives_by_family=_family_map(
                    epochal_turns=[_epochal_turn("et_1")],
                    contested_explanations=[
                        ContestedExplanationPrimitive(
                            id="cx_invalid",
                            family="contested_explanations",
                            title="Single reading",
                            summary="The evidence supports only one explicit reading.",
                            core_passage_ids=["p1"],
                            candidate_readings=[
                                CandidateReading(
                                    label="reading_a",
                                    summary="Only one reading present.",
                                    support_passage_ids=["p1"],
                                )
                            ],
                        )
                    ],
                ),
                evidence_packs=[
                    EvidencePack(
                        pack_id="pack_1",
                        title="Opening pack",
                        local_summary="A compact causal chain.",
                        primitive_ids=["et_1", "cx_invalid"],
                    )
                ],
            )

    def test_synthesis_map_roundtrip_preserves_evidence_pack_shape(self):
        synthesis_map = SynthesisMap(
            project_id="proj",
            primitives_by_family=_family_map(
                epochal_turns=[_epochal_turn("et_1")],
                contested_explanations=[_contested_explanation("cx_1")],
            ),
            evidence_packs=[
                EvidencePack(
                    pack_id="pack_1",
                    title="Opening pack",
                    local_summary="A compact causal chain.",
                    primitive_ids=["et_1", "cx_1"],
                )
            ],
            quality_score=0.7,
        )
        restored = SynthesisMap.model_validate(
            json.loads(synthesis_map.model_dump_json())
        )
        assert restored.evidence_packs[0].primitive_ids[0] == "et_1"
        assert (
            restored.primitives_by_family["contested_explanations"][0]
            .candidate_readings[1]
            .label
            == "reading_b"
        )

    def test_synthesis_consolidation_result_rejects_unknown_pack_primitive_ids(self):
        with pytest.raises(ValidationError, match="unknown primitive_ids"):
            SynthesisConsolidationResult(
                project_id="proj",
                primitive_ids_by_family={"epochal_turns": ["et_1"]},
                evidence_packs=[
                    EvidencePack(
                        pack_id="pack_1",
                        title="Pack",
                        local_summary="Summary",
                        primitive_ids=["et_1", "tp_999"],
                    )
                ],
            )

    def test_synthesis_consolidation_result_accepts_family_id_universe(self):
        result = SynthesisConsolidationResult(
            project_id="proj",
            primitive_ids_by_family={
                "epochal_turns": ["et_1"],
                "contested_explanations": ["cx_1"],
            },
            evidence_packs=[
                EvidencePack(
                    pack_id="pack_1",
                    title="Pack",
                    local_summary="Summary",
                    primitive_ids=["et_1", "cx_1"],
                )
            ],
        )
        assert result.evidence_packs[0].primitive_ids == ["et_1", "cx_1"]


class TestNarrativeStrategy:
    def test_actor_arc_directive_roundtrip_uses_arc_threads(self):
        directive = ActorArcDirective(
            actor_id="mahatma_gandhi",
            arc_threads=[
                ActorArcThread(
                    thread_id="gandhi_role_1",
                    arc_type="role",
                    label="episode role",
                    premise="His restraint frames the episode's character function.",
                    pressure="Escalating violence narrows his choices.",
                    movement="Restraint becomes harder to sustain across scenes.",
                    resolution="The listener sees restraint as a contested strategy.",
                ),
                ActorArcThread(
                    thread_id="gandhi_tracking_1",
                    arc_type="tracking",
                    label="listener tracking",
                    premise="Track how restraint becomes harder to sustain.",
                    pressure="Public discipline collides with escalating violence.",
                    movement="Each appearance should add pressure.",
                    resolution="The episode clarifies what restraint costs.",
                ),
                ActorArcThread(
                    thread_id="gandhi_tension_1",
                    arc_type="tension",
                    label="tension line",
                    premise="Public discipline collides with escalating violence.",
                    pressure="Followers and opponents both test discipline.",
                    movement="Pressure accumulates until restraint changes meaning.",
                    resolution="The tension remains visible at the episode close.",
                ),
                ActorArcThread(
                    thread_id="gandhi_progression_1",
                    arc_type="turn",
                    label="arc progression",
                    premise="The episode changes what restraint can plausibly mean.",
                    pressure="A decision changes the arc.",
                    movement="Move from principle to tested practice.",
                    resolution="Restraint lands as a consequential choice.",
                ),
                ActorArcThread(
                    thread_id="gandhi_scene_job_1",
                    arc_type="payoff",
                    label="scene job",
                    premise="Use where restraint meets consequence.",
                    pressure="The scene must do more than repeat the actor function.",
                    movement="Bind actor presence to scene consequence.",
                    resolution="The scene makes the actor arc legible.",
                ),
                ActorArcThread(
                    thread_id="gandhi_guardrail_1",
                    arc_type="guardrail",
                    label="repetition guardrail",
                    premise="Do not re-explain restraint unless the scene changes its meaning.",
                    pressure="Repeated appearances can flatten the actor.",
                    movement="Vary actor function across scenes.",
                    resolution="The actor remains continuous without becoming repetitive.",
                ),
            ],
        )

        restored = ActorArcDirective.model_validate(
            json.loads(directive.model_dump_json())
        )

        assert restored.actor_id == "mahatma_gandhi"
        assert restored.arc_threads[1].thread_id == "gandhi_tracking_1"
        assert restored.arc_threads[1].arc_type == "tracking"
        assert restored.arc_threads[1].label == "listener tracking"
        assert (
            restored.arc_threads[1].pressure
            == "Public discipline collides with escalating violence."
        )

    def test_actor_arc_directive_rejects_duplicate_thread_ids(self):
        with pytest.raises(ValidationError, match="thread ids must be unique"):
            ActorArcDirective(
                actor_id="mahatma_gandhi",
                arc_threads=[
                    ActorArcThread(
                        thread_id="gandhi_1",
                        arc_type="role",
                        label="role",
                        premise="Role",
                    ),
                    ActorArcThread(
                        thread_id="gandhi_1",
                        arc_type="payoff",
                        label="scene job",
                        premise="Scene job",
                    ),
                ],
            )

    def test_actor_arc_directive_allows_single_thread(self):
        directive = ActorArcDirective(
            actor_id="mahatma_gandhi",
            arc_threads=[
                ActorArcThread(
                    thread_id="gandhi_1",
                    arc_type="role",
                    label="role",
                    premise="Role",
                )
            ],
        )

        assert directive.arc_threads[0].thread_id == "gandhi_1"

    def test_actor_arc_thread_allows_noncanonical_arc_type(self):
        thread = ActorArcThread(
            thread_id="gandhi_ideologue_1",
            arc_type="ideologue",
            label="ideological frame",
            premise="Frame policy around ideological commitments.",
        )
        assert thread.arc_type == "ideologue"

    def test_actor_arc_thread_accepts_legacy_payoff_alias(self):
        thread = ActorArcThread.model_validate(
            {
                "thread_id": "gandhi_turn_1",
                "arc_type": "turn",
                "label": "turning point",
                "premise": "An event forces a revised stance.",
                "payoff": "Legacy field remains readable.",
            }
        )
        assert thread.resolution == "Legacy field remains readable."
        dumped = thread.model_dump(mode="json")
        assert "payoff" not in dumped
        assert dumped["resolution"] == "Legacy field remains readable."

    def test_actor_arc_thread_ignores_extra_fields(self):
        thread = ActorArcThread.model_validate(
            {
                "thread_id": "gandhi_turn_1",
                "arc_type": "turn",
                "label": "turning point",
                "premise": "An event forces a revised stance.",
                "permise": "",
            }
        )
        assert thread.premise == "An event forces a revised stance."
        assert "permise" not in thread.model_dump()

    def test_actor_arc_directive_rejects_empty_threads(self):
        with pytest.raises(ValidationError):
            ActorArcDirective(
                actor_id="mahatma_gandhi",
                arc_threads=[],
            )

    def test_strategy_episode_rejects_old_episode_actor_fields(self):
        with pytest.raises(ValidationError):
            StrategyEpisode.model_validate(
                {
                    "episode_number": 1,
                    "title": "Episode 1",
                    "arc_summary": "Arc",
                    "episode_spine": _episode_spine("pack_1").model_dump(mode="json"),
                    "actor_ids": ["mahatma_gandhi"],
                    "primary_actor_ids": ["mahatma_gandhi"],
                    "actor_arc_summary": "Old summary.",
                    "cluster_path": [],
                }
            )

    def test_episode_spine_rejects_support_overlap_with_spine(self):
        support_primitive_roles = {
            "pack_1": SupportPackRole.STAKES,
            **{
                f"support_{idx}": (
                    SupportPackRole.MECHANISM
                    if idx % 2 == 0
                    else SupportPackRole.TEXTURE
                )
                for idx in range(1, 8)
            },
        }
        with pytest.raises(
            ValidationError,
            match="support primitives cannot also appear in core_primitive_ids",
        ):
            EpisodeSpine(
                listener_question="What changed?",
                argument="A claim",
                core_primitive_ids=["pack_1", *[f"core_{idx}" for idx in range(2, 8)]],
                support_primitive_roles=support_primitive_roles,
            )

    def test_episode_spine_rejects_core_primitive_count_outside_shared_range(self):
        with pytest.raises(
            ValidationError, match="core_primitive_ids must contain 2-7"
        ):
            EpisodeSpine(
                listener_question="What changed?",
                argument="A claim",
                core_primitive_ids=["core_1", "core_2"],
                support_primitive_roles={
                    f"support_{idx}": SupportPackRole.MECHANISM for idx in range(1, 8)
                },
            )

    def test_episode_spine_rejects_support_primitive_count_outside_shared_range(self):
        with pytest.raises(
            ValidationError, match="support_primitive_roles must contain 2-7"
        ):
            EpisodeSpine(
                listener_question="What changed?",
                argument="A claim",
                core_primitive_ids=[f"core_{idx}" for idx in range(1, 8)],
                support_primitive_roles={
                    "support_1": SupportPackRole.MECHANISM,
                    "support_2": SupportPackRole.MECHANISM,
                },
            )

    def test_validate_episode_spine_targets_enforces_full_mode_counts(self):
        spine = EpisodeSpine(
            listener_question="What changed?",
            argument="A claim",
            core_primitive_ids=[f"core_{idx}" for idx in range(1, 5)],
            support_primitive_roles={
                f"support_{idx}": SupportPackRole.MECHANISM for idx in range(1, 5)
            },
        )

        with pytest.raises(ValueError, match="core_primitive_ids must contain 5-7"):
            validate_episode_spine_targets(
                spine,
                core_target_min=5,
                core_target_max=7,
                support_target_min=5,
                support_target_max=7,
                recall_target_max=2,
            )

    def test_validate_episode_spine_targets_accepts_minified_counts(self):
        spine = EpisodeSpine(
            listener_question="What changed?",
            argument="A claim",
            core_primitive_ids=["core_1", "core_2"],
            support_primitive_roles={
                "support_1": SupportPackRole.MECHANISM,
                "support_2": SupportPackRole.TEXTURE,
            },
            recall_primitive_ids=["recall_1"],
        )

        validate_episode_spine_targets(
            spine,
            core_target_min=2,
            core_target_max=4,
            support_target_min=2,
            support_target_max=4,
            recall_target_max=1,
        )

    def test_validate_episode_spine_targets_rejects_minified_recall_overflow(self):
        spine = EpisodeSpine(
            listener_question="What changed?",
            argument="A claim",
            core_primitive_ids=["core_1", "core_2"],
            support_primitive_roles={
                "support_1": SupportPackRole.MECHANISM,
                "support_2": SupportPackRole.TEXTURE,
            },
            recall_primitive_ids=["recall_1", "recall_2"],
        )

        with pytest.raises(
            ValueError, match="recall_primitive_ids must contain at most 1"
        ):
            validate_episode_spine_targets(
                spine,
                core_target_min=2,
                core_target_max=4,
                support_target_min=2,
                support_target_max=4,
                recall_target_max=1,
            )

    def test_episode_spine_rejects_removed_rhetorical_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EpisodeSpine.model_validate(
                {
                    "listener_question": "What changed?",
                    "argument": "A claim",
                    "spine_pack_ids": ["pack_1"],
                    "support_pack_roles": {},
                    "allowed_recalls": [],
                    "opening_state": "Removed field",
                }
            )

    def test_episode_spine_rejects_episode_number_field(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EpisodeSpine.model_validate(
                {
                    "episode_number": 1,
                    "listener_question": "What changed?",
                    "argument": "A claim",
                    "spine_pack_ids": ["pack_1"],
                    "support_pack_roles": {},
                    "allowed_recalls": [],
                }
            )

    def test_narrative_strategy_rejects_pack_home_collision(self):
        with pytest.raises(ValidationError, match="multiple primary home episodes"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Use converging local causal chains.",
                series_arc="Each episode carries one pack home.",
                episode_arc_outline=["Ep 1", "Ep 2"],
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode 1",
                        arc_summary="Arc 1",
                        episode_spine=_episode_spine("pack_1"),
                    ),
                    StrategyEpisode(
                        episode_number=2,
                        title="Episode 2",
                        arc_summary="Arc 2",
                        episode_spine=_episode_spine("pack_1"),
                    ),
                ],
            )

    def test_narrative_strategy_requires_at_least_one_spine_pack_per_episode(self):
        with pytest.raises(ValidationError, match="at least 1 item"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Use converging local causal chains.",
                series_arc="Each episode carries one spine pack.",
                episode_arc_outline=["Ep 1"],
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode 1",
                        arc_summary="Arc 1",
                        episode_spine={
                            **_episode_spine("pack_1").model_dump(mode="json"),
                            "core_primitive_ids": [],
                        },
                    ),
                ],
            )


class TestPlanningModels:
    def test_scene_actor_uses_arc_bindings(self):
        actor = SceneActor(
            name="Mahatma Gandhi",
            actor_id="mahatma_gandhi",
            presence="primary",
            arc_bindings=[
                SceneActorArcBinding(
                    thread_id="gandhi_tension_1",
                    scene_role="blocked",
                    scene_use="complicate",
                    weight="strong",
                )
            ],
        )

        assert actor.presence == "primary"
        assert actor.arc_bindings[0].thread_id == "gandhi_tension_1"
        assert actor.arc_bindings[0].scene_role == "blocked"
        assert actor.arc_bindings[0].scene_use == "complicate"

    def test_scene_actor_rejects_old_actor_aspect_fields(self):
        with pytest.raises(ValidationError):
            SceneActor.model_validate(
                {
                    "name": "Mahatma Gandhi",
                    "role_in_scene": "tests restraint",
                    "arc_thread_ids": ["gandhi_tension_1"],
                    "motivation_aspect_ids": ["gandhi_motivation_1"],
                    "stake_aspect_ids": ["gandhi_stake_1"],
                    "pressure_aspect_ids": ["gandhi_pressure_1"],
                    "turning_point_aspect_ids": ["gandhi_turn_1"],
                    "scene_actor_directives": ["Old directive."],
                    "scene_actor_work": ["Old directive."],
                }
            )

    def test_clean_scene_actor_links_filters_unknown_arc_bindings(self):
        actor_directive = ActorArcDirective(
            actor_id="mahatma_gandhi",
            arc_threads=[
                ActorArcThread(
                    thread_id="gandhi_tension_1",
                    arc_type="tension",
                    label="tension line",
                    premise="Public discipline collides with escalating violence.",
                ),
                ActorArcThread(
                    thread_id="gandhi_guardrail_1",
                    arc_type="guardrail",
                    label="guardrail",
                    premise="Do not repeat the actor function without change.",
                ),
            ],
        )
        scene = _normal_scene_draft()
        scene.actors = [
            SceneActor(
                name="Mahatma Gandhi",
                actor_id="mahatma_gandhi",
                arc_bindings=[
                    SceneActorArcBinding(
                        thread_id="gandhi_tension_1",
                        scene_role="blocked",
                        scene_use="complicate",
                    ),
                    SceneActorArcBinding(
                        thread_id="missing_thread",
                        scene_role="blocked",
                        scene_use="complicate",
                    ),
                ],
            )
        ]
        plan = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            episode_spine=_episode_spine("pack_1"),
            actor_arc_directives=[actor_directive],
            framing=_framing(),
            scene_cards=[scene],
        )
        metadata = ActorMetadata(
            actors=[
                ActorProfile(
                    actor_id="mahatma_gandhi",
                    display_name="Mahatma Gandhi",
                    actor_type="person",
                )
            ]
        )

        cleaned, metrics = clean_scene_actor_links(plan, metadata)

        cleaned_actor = cleaned.scene_cards[0].actors[0]
        assert [binding.thread_id for binding in cleaned_actor.arc_bindings] == [
            "gandhi_tension_1"
        ]
        assert metrics["unknown_actor_arc_thread_ids"] == 1

    def test_scene_card_allows_noncanonical_scene_role(self):
        card = SceneCard(
            scene_id="scene_reveal",
            title="A hidden mechanism surfaces",
            scene_role="reveal",
            dominant_pack_id="pack_1",
            spine_relation=SpineRelation.SPINE_ADVANCE,
            state_effect="The mechanism becomes visible.",
            entry_image="The ledger opens.",
            local_question="What finally becomes visible?",
            observable_detail="A missing column is now clear.",
            intended_move="Expose the mechanism.",
            passage_ids=["p1"],
            host_moves=_host_moves_payload(),
            estimated_duration_seconds=90,
        )
        assert card.scene_role == "reveal"
        assert "coverage_depth" not in card.model_dump()

    def test_scene_card_draft_drops_coverage_depth_on_load(self):
        card = SceneCardDraft.model_validate(
            {
                "scene_id": "scene_draft",
                "section_id": "section_1",
                "title": "A hidden mechanism surfaces",
                "scene_role": "reveal",
                "dominant_pack_id": "pack_1",
                "spine_relation": "spine_advance",
                "state_effect": "The mechanism becomes visible.",
                "entry_image": "The ledger opens.",
                "local_question": "What finally becomes visible?",
                "observable_detail": "A missing column is now clear.",
                "intended_move": "Expose the mechanism.",
                "passage_ids": ["p1"],
                "host_moves": _host_moves_payload().model_dump(mode="json"),
                "coverage_depth": "deep",
            }
        )
        assert "coverage_depth" not in card.model_dump()

    def test_scene_card_draft_aliases_process_to_action(self):
        card = SceneCardDraft.model_validate(
            {
                "scene_id": "scene_action",
                "section_id": "section_1",
                "title": "A hidden mechanism surfaces",
                "scene_role": "process",
                "dominant_pack_id": "pack_1",
                "spine_relation": "spine_advance",
                "state_effect": "The mechanism becomes visible.",
                "entry_image": "The ledger opens.",
                "local_question": "What finally becomes visible?",
                "observable_detail": "A missing column is now clear.",
                "intended_move": "Expose the mechanism.",
                "passage_ids": ["p1"],
                "host_moves": _host_moves_payload().model_dump(mode="json"),
            }
        )
        assert card.scene_role == "action"

    def test_scene_card_draft_fixes_perspective_shift_typo(self):
        card = SceneCardDraft.model_validate(
            {
                "scene_id": "scene_shift",
                "section_id": "section_1",
                "title": "A hidden mechanism surfaces",
                "scene_role": "perspective shift",
                "dominant_pack_id": "pack_1",
                "spine_relation": "spine_advance",
                "state_effect": "The mechanism becomes visible.",
                "entry_image": "The ledger opens.",
                "local_question": "What finally becomes visible?",
                "observable_detail": "A missing column is now clear.",
                "intended_move": "Expose the mechanism.",
                "passage_ids": ["p1"],
                "host_moves": _host_moves_payload().model_dump(mode="json"),
            }
        )
        assert card.scene_role == "perspective_shift"

    def test_scene_card_draft_does_not_require_duration(self):
        card = SceneCardDraft(
            scene_id="scene_draft",
            title="A hidden mechanism surfaces",
            scene_role="reveal",
            dominant_pack_id="pack_1",
            spine_relation=SpineRelation.SPINE_ADVANCE,
            state_effect="The mechanism becomes visible.",
            entry_image="The ledger opens.",
            local_question="What finally becomes visible?",
            observable_detail="A missing column is now clear.",
            intended_move="Expose the mechanism.",
            passage_ids=["p1"],
            host_moves=_host_moves_payload(),
        )
        assert "estimated_duration_seconds" not in card.model_dump()

    def test_scene_card_draft_discards_legacy_duration(self):
        card = SceneCardDraft.model_validate(
            {
                "scene_id": "scene_draft",
                "title": "A hidden mechanism surfaces",
                "scene_role": "reveal",
                "dominant_pack_id": "pack_1",
                "spine_relation": "spine_advance",
                "state_effect": "The mechanism becomes visible.",
                "entry_image": "The ledger opens.",
                "local_question": "What finally becomes visible?",
                "observable_detail": "A missing column is now clear.",
                "intended_move": "Expose the mechanism.",
                "passage_ids": ["p1"],
                "host_moves": _host_moves_payload().model_dump(mode="json"),
                "estimated_duration_seconds": 120,
            }
        )
        assert "estimated_duration_seconds" not in card.model_dump()

    def test_scene_card_requires_duration(self):
        with pytest.raises(ValidationError, match="estimated_duration_seconds"):
            SceneCard.model_validate(
                {
                    "scene_id": "scene_final",
                    "section_id": "section_1",
                    "title": "A hidden mechanism surfaces",
                    "scene_role": "reveal",
                    "dominant_pack_id": "pack_1",
                    "spine_relation": "spine_advance",
                    "state_effect": "The mechanism becomes visible.",
                    "entry_image": "The ledger opens.",
                    "local_question": "What finally becomes visible?",
                    "observable_detail": "A missing column is now clear.",
                    "intended_move": "Expose the mechanism.",
                    "passage_ids": ["p1"],
                    "host_moves": _host_moves_payload().model_dump(mode="json"),
                }
            )

    def test_scene_card_rejects_legacy_narrative_weight_field(self):
        with pytest.raises(ValidationError):
            SceneCard.model_validate(
                {
                    "scene_id": "scene_deep",
                    "section_id": "section_1",
                    "title": "A hidden mechanism surfaces",
                    "scene_role": "reveal",
                    "dominant_pack_id": "pack_1",
                    "spine_relation": "spine_advance",
                    "state_effect": "The mechanism becomes visible.",
                    "entry_image": "The ledger opens.",
                    "local_question": "What finally becomes visible?",
                    "observable_detail": "A missing column is now clear.",
                    "intended_move": "Expose the mechanism.",
                    "passage_ids": ["p1"],
                    "host_moves": _host_moves_payload().model_dump(mode="json"),
                    "estimated_duration_seconds": 120,
                    "narrative_weight": 2.5,
                }
            )

    def test_scene_card_rejects_blank_scene_role(self):
        with pytest.raises(ValidationError, match="scene_role must not be blank"):
            SceneCard(
                scene_id="scene_blank",
                section_id="section_1",
                title="Invalid role",
                scene_role="   ",
                dominant_pack_id="pack_1",
                spine_relation=SpineRelation.SPINE_ADVANCE,
                state_effect="The mechanism becomes visible.",
                entry_image="Image",
                local_question="Question",
                observable_detail="Detail",
                intended_move="Move",
                passage_ids=["p1"],
                host_moves=_host_moves_payload(),
                estimated_duration_seconds=60,
            )

    def test_normal_scene_card_requires_dominant_occurrence(self):
        with pytest.raises(ValidationError, match="scene cards require"):
            SceneCard(
                scene_id="scene_1",
                title="The order arrives",
                scene_role="setup",
                spine_relation=SpineRelation.SET_STAKES,
                state_effect="The stakes become visible.",
                entry_image="Envelope on desk.",
                local_question="What changes first?",
                observable_detail="Hands stop.",
                intended_move="Set up the stakes.",
                passage_ids=["p1"],
                host_moves=_host_moves_payload(),
                estimated_duration_seconds=60,
            )

    def test_scene_card_rejects_bridge_card_fields(self):
        with pytest.raises(ValidationError):
            SceneCard(
                scene_id="scene_bridge",
                section_id="section_1",
                title="Bridge",
                card_kind="bridge",
                scene_role="synthesis",
                dominant_pack_id="pack_1",
                spine_relation=SpineRelation.SPINE_ADVANCE,
                state_effect="The argument advances.",
                entry_image="The silence shifts.",
                local_question="How do we move forward?",
                observable_detail="A messenger leaves.",
                intended_move="Bridge occurrences.",
                passage_ids=["p1"],
                host_moves=_host_moves_payload(),
                estimated_duration_seconds=60,
            )

    def test_episode_plan_draft_validates_framing_handoff(self):
        with pytest.raises(ValidationError, match="handoff_scene_card_id"):
            EpisodePlanDraft(
                episode_number=1,
                title="Episode 1",
                driving_question="Why begin here?",
                episode_spine=_episode_spine("pack_1"),
                framing=FramingBlock(
                    opening_image="Image",
                    threat_or_unresolved_action="Threat",
                    opening_question="Question",
                    handoff_scene_card_id="missing_scene",
                ),
                scene_cards=[_normal_scene_draft()],
            )

    def test_episode_plan_draft_allows_non_scene_withhold_until_reference(self):
        scene = _normal_scene_draft()
        scene.withhold_until = "Full consequences are developed in Episode 5."
        draft = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            episode_spine=_episode_spine("pack_1"),
            framing=_framing(),
            scene_cards=[scene],
        )
        assert (
            draft.scene_cards[0].withhold_until
            == "Full consequences are developed in Episode 5."
        )

    def test_episode_plan_draft_defaults_target_duration_minutes_to_90(self):
        draft = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            episode_spine=_episode_spine("pack_1"),
            framing=_framing(),
            scene_cards=[_normal_scene_draft()],
        )
        assert draft.target_duration_minutes == 100.0

    def test_episode_plan_draft_rejects_noncontiguous_section_ids(self):
        scene_one = _normal_scene_draft(scene_id="scene_1")
        scene_one.section_id = "section_1"
        scene_two = _normal_scene_draft(scene_id="scene_2")
        scene_two.section_id = "section_2"
        scene_three = _normal_scene_draft(scene_id="scene_3")
        scene_three.section_id = "section_1"
        with pytest.raises(
            ValidationError, match="contiguous architecture section order"
        ):
            EpisodePlanDraft(
                episode_number=1,
                title="Episode 1",
                driving_question="Why begin here?",
                episode_spine=_episode_spine("pack_1"),
                framing=FramingBlock(
                    opening_image="Image",
                    threat_or_unresolved_action="Threat",
                    opening_question="Question",
                    handoff_scene_card_id="scene_3",
                ),
                scene_cards=[scene_one, scene_two, scene_three],
            )

    def test_episode_plan_roundtrip(self):
        draft = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            thematic_focus="Opening moves",
            arc_summary="The episode traces one local causal chain.",
            unresolved_questions=["What still remains unclear?"],
            episode_spine=_episode_spine("pack_1"),
            framing=_framing(),
            scene_cards=[_normal_scene()],
            target_duration_minutes=140.0,
            target_word_count=18900,
        )
        restored = EpisodePlan.model_validate(json.loads(draft.model_dump_json()))
        assert restored.framing.handoff_scene_card_id == "scene_1"
        assert restored.scene_cards[0].actors[0].name == "Clerk"


class TestSpeechAndStyleModels:
    def test_prose_section_accepts_non_enum_movement_goal(self):
        section = ProseSection.model_validate(
            {
                "section_id": "section_1",
                "scene_card_ids": ["scene_1"],
                "movement_goal": "setup",
                "text": "The first image lands before the thesis.",
            }
        )
        assert section.movement_goal == "setup"

    def test_prose_section_allows_empty_text(self):
        section = ProseSection.model_validate(
            {
                "section_id": "section_1",
                "scene_card_ids": ["scene_1"],
                "movement_goal": "setup",
                "text": "",
            }
        )
        assert section.text == ""

    def test_spoken_script_roundtrip_preserves_sections(self):
        spoken = SpokenScript(
            episode_number=1,
            title="Episode 1",
            framing=_framing(),
            sections=[
                SpokenSection(
                    section_id="section_1",
                    text="The convoy moves before dawn.",
                    speech_hints=SpeechHints(style="measured", pace="slower"),
                )
            ],
        )
        restored = SpokenScript.model_validate(json.loads(spoken.model_dump_json()))
        assert restored.sections[0].speech_hints.style == "measured"


class TestEpisodeArchitectureModels:
    def _build_architecture(self, section_count: int) -> EpisodeArchitecture:
        sections = []
        for idx in range(section_count):
            section_id = f"section_{idx + 1:02d}"
            sections.append(
                ArchitectureSection(
                    section_id=section_id,
                    purpose="opening"
                    if idx == 0
                    else "closing"
                    if idx == section_count - 1
                    else "setup",
                    approx_runtime_minutes=100.0 / section_count,
                    primitive_ids=["et_1"],
                    section_anchor=f"Anchor {idx + 1}",
                    must_stage_beats=[
                        f"Beat {idx + 1}A",
                        f"Beat {idx + 1}B",
                    ],
                    section_question=f"Question {idx + 1}?",
                    section_resolution=f"Resolution {idx + 1}",
                    entry_state=f"Entry {idx + 1}",
                    exit_state=f"Exit {idx + 1}",
                    transition_logic=f"Transition {idx + 1}",
                    depends_on_section_ids=[f"section_{idx:02d}"] if idx > 0 else [],
                    sets_up_section_ids=[f"section_{idx + 2:02d}"]
                    if idx < section_count - 1
                    else [],
                    argument_role="frame"
                    if idx == 0
                    else "close"
                    if idx == section_count - 1
                    else "establish_mechanism",
                    inference_mode="scene_first" if idx == 0 else "mechanism_first",
                    recurrence_role="plant"
                    if idx == 0
                    else "payoff"
                    if idx == section_count - 1
                    else "deepen",
                    pressure_type="mass_political",
                    resolution_type="containment"
                    if idx == section_count - 1
                    else "escalation",
                    closure_level="high" if idx == section_count - 1 else "low",
                )
            )
        return EpisodeArchitecture(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            thematic_focus="Opening moves",
            arc_summary="The episode traces one local causal chain.",
            unresolved_questions=[],
            episode_spine=_episode_spine("et_1"),
            actor_arc_directives=[],
            major_turn_section_id=sections[min(2, len(sections) - 1)].section_id,
            allowed_recurring_primitive_ids=[],
            forbidden_redundancies=[],
            sections=sections,
            architecture_notes=[],
        )

    def test_episode_architecture_accepts_nine_sections(self):
        architecture = self._build_architecture(9)
        assert len(architecture.sections) == 9

    def test_episode_architecture_accepts_ten_sections(self):
        architecture = self._build_architecture(10)
        assert len(architecture.sections) == 10

    def test_episode_architecture_rejects_removed_target_section_count_field(self):
        with pytest.raises(ValidationError, match="target_section_count"):
            EpisodeArchitecture.model_validate(
                {
                    **self._build_architecture(9).model_dump(mode="json"),
                    "target_section_count": 6,
                }
            )

    def test_episode_architecture_accepts_twelve_sections(self):
        architecture = self._build_architecture(12)
        assert len(architecture.sections) == 12

    def test_episode_architecture_accepts_six_sections(self):
        architecture = self._build_architecture(6)
        assert len(architecture.sections) == 6

    def test_episode_architecture_rejects_fewer_than_six_sections(self):
        with pytest.raises(ValidationError, match="at least 6 items"):
            self._build_architecture(5)

    def test_episode_architecture_rejects_more_than_twelve_sections(self):
        with pytest.raises(ValidationError, match="at most 12 items"):
            self._build_architecture(13)

    def test_validate_episode_architecture_targets_enforces_full_mode_counts(self):
        architecture = self._build_architecture(8)

        with pytest.raises(ValueError, match="sections must contain 9-12 items"):
            validate_episode_architecture_targets(
                architecture,
                section_target_min=9,
                section_target_max=12,
            )

    def test_validate_episode_architecture_targets_accepts_minified_counts(self):
        architecture = self._build_architecture(6)

        validate_episode_architecture_targets(
            architecture,
            section_target_min=6,
            section_target_max=8,
        )

    def test_architecture_section_migrates_legacy_anchor_field(self):
        section = ArchitectureSection.model_validate(
            {
                "section_id": "section_01",
                "purpose": "opening",
                "approx_runtime_minutes": 10.0,
                "primitive_ids": ["et_1"],
                "anchor": "Legacy anchor",
                "must_stage_beats": ["Visible move", "Immediate consequence"],
                "section_question": "Question?",
                "section_resolution": "Resolution",
                "entry_state": "Entry",
                "exit_state": "Exit",
                "transition_logic": "Transition",
                "argument_role": "frame",
                "inference_mode": "scene_first",
                "pressure_type": "mass_political",
                "resolution_type": "redefinition",
            }
        )

        payload = section.model_dump(mode="json")
        assert section.section_anchor == "Legacy anchor"
        assert payload["section_anchor"] == "Legacy anchor"
        assert "anchor" not in payload

    def test_architecture_section_rejects_single_must_stage_beat_when_provided(self):
        with pytest.raises(
            ValidationError, match="must_stage_beats must contain at least 2 items"
        ):
            ArchitectureSection.model_validate(
                {
                    "section_id": "section_01",
                    "purpose": "opening",
                    "approx_runtime_minutes": 10.0,
                    "primitive_ids": ["et_1"],
                    "section_anchor": "Anchor",
                    "must_stage_beats": ["Only one beat"],
                    "section_question": "Question?",
                    "section_resolution": "Resolution",
                    "entry_state": "Entry",
                    "exit_state": "Exit",
                    "transition_logic": "Transition",
                    "argument_role": "frame",
                    "inference_mode": "scene_first",
                    "pressure_type": "mass_political",
                    "resolution_type": "redefinition",
                }
            )
