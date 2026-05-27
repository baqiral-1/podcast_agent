"""Unit tests for the redesigned schema contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from _section_progression_helpers import make_section_progression
from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcThread,
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    AuthorialPassage,
    ArchitectureSection,
    ChapterAnalysis,
    EpisodeArchitecture,
    EpisodePlan,
    EpisodePlanDraft,
    EpisodeSpine,
    EpisodeTakeaway,
    EventPrimitive,
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
    SectionSonicObligation,
    SectionSonicPlan,
    SeriesNarratorProfile,
    SpineRelation,
    SpeechHints,
    SpokenScript,
    SpokenSection,
    StrategyEpisode,
    SupportPrimitiveRole,
    SynthesisMap,
    SynthesisPrimitivesArtifact,
    ThematicProject,
    PipelineConfig,
    PodcastMode,
    authorial_passage_target_for_mode,
    authorial_passage_target_range_for_mode,
    dense_section_authorial_passage_range_for_mode,
    primitive_substrate_target_ranges_for_mode,
    resolve_pipeline_config_for_mode,
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


def _event_primitive(primitive_id: str = "e1") -> EventPrimitive:
    return EventPrimitive(
        id=primitive_id,
        substrate="events",
        title="Threshold breaks",
        core_passage_ids=["p1"],
        support_passage_ids=["p2"],
        event_type="political rupture",
        what_happened="A new political order takes effect.",
    )


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
        beat_change="The stakes become legible.",
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
        beat_change="The stakes become legible.",
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
        f"support_{idx}": SupportPrimitiveRole.MECHANISM
        if idx % 2 == 0
        else SupportPrimitiveRole.TEXTURE
        for idx in range(1, 8)
    }
    return EpisodeSpine(
        listener_problem="Why does this decision land so hard?",
        episode_answer="This episode tests one controlling proposition.",
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
        assert config.narrative_strategy_episode_count_min == 10
        assert config.narrative_strategy_episode_count_max == 16

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

    def test_pipeline_config_defaults_synthesis_cap_to_750(self):
        config = PipelineConfig()
        assert config.synthesis_total_passage_cap == 750
        assert config.synthesis_floor_budget_fraction == 0.0
        assert config.synthesis_axis_floor_min == 0
        assert config.synthesis_axis_floor_max == 0
        assert config.synthesis_axis_ceiling_multiplier == 1.68
        assert config.synthesis_trim_top_fraction == 0.10
        assert config.synthesis_trim_mid_fraction == 0.20
        assert config.synthesis_trim_top_keep_fraction == 0.35
        assert config.synthesis_trim_mid_keep_fraction == 0.25
        assert config.synthesis_trim_tail_keep_fraction == 0.15
        assert config.passage_extraction_concurrency == 16
        assert config.episode_write_concurrency == 8
        assert config.episode_writing_batch_count == 5
        assert config.spoken_delivery_concurrency == 8
        assert config.min_episode_minutes == 90.0
        assert config.max_episode_minutes == 120.0
        assert config.architecture_section_target_min == 11
        assert config.architecture_section_target_max == 14
        assert config.episode_spine_core_primitive_target_min == 6
        assert config.episode_spine_core_primitive_target_max == 8
        assert config.episode_spine_support_primitive_target_min == 6
        assert config.episode_spine_support_primitive_target_max == 9
        assert config.episode_spine_recall_primitive_target_max == 2
        assert config.scene_card_target_min == 30
        assert config.scene_card_target_max == 38

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
        assert config.episode_writing_batch_count == 2
        assert config.episode_spine_core_primitive_target_min == 3
        assert config.episode_spine_core_primitive_target_max == 5
        assert config.episode_spine_support_primitive_target_min == 4
        assert config.episode_spine_support_primitive_target_max == 6
        assert config.episode_spine_recall_primitive_target_max == 1
        assert config.architecture_section_target_min == 7
        assert config.architecture_section_target_max == 9
        assert config.min_episode_minutes == 54.0
        assert config.max_episode_minutes == 63.0
        assert config.scene_card_target_min == 18
        assert config.scene_card_target_max == 21

    def test_resolve_pipeline_config_for_mode_applies_full_profile(self):
        config = resolve_pipeline_config_for_mode(
            PipelineConfig(podcast_mode=PodcastMode.FULL)
        )

        assert config.podcast_mode == PodcastMode.FULL
        assert config.min_axes == 12
        assert config.max_axes == 20
        assert config.synthesis_axis_min == 12
        assert config.synthesis_axis_max == 20
        assert config.narrative_strategy_episode_count_min == 10
        assert config.narrative_strategy_episode_count_max == 16
        assert config.synthesis_total_passage_cap == 750
        assert config.episode_writing_batch_count == 5
        assert config.architecture_section_target_min == 11
        assert config.architecture_section_target_max == 14
        assert config.min_episode_minutes == 90.0
        assert config.max_episode_minutes == 120.0
        assert config.scene_card_target_min == 30
        assert config.scene_card_target_max == 38

    def test_authorial_passage_target_ranges_for_modes(self):
        assert authorial_passage_target_range_for_mode(PodcastMode.FULL) == (18, 28)
        assert authorial_passage_target_range_for_mode(PodcastMode.MINIFIED) == (12, 16)
        assert dense_section_authorial_passage_range_for_mode(PodcastMode.FULL) == (
            3,
            8,
        )
        assert dense_section_authorial_passage_range_for_mode(
            PodcastMode.MINIFIED
        ) == (2, 5)
        assert authorial_passage_target_for_mode(PodcastMode.FULL) == 23
        assert authorial_passage_target_for_mode(PodcastMode.MINIFIED) == 14

    def test_authorial_passage_allows_six_sentence_budget(self):
        passage = AuthorialPassage.model_validate(
            {
                "passage_id": "p1",
                "mode": "comparative_aside",
                "claim": "A scene fact turns into a brief benchmark.",
                "source_primitive_ids": ["primitive_1"],
                "budget_sentences": 6,
            }
        )
        assert passage.budget_sentences == 6

    def test_primitive_substrate_target_ranges_for_full_mode_match_expanded_table(self):
        target_ranges = primitive_substrate_target_ranges_for_mode(PodcastMode.FULL)
        assert target_ranges == {
            "events": (87, 120),
            "acts": (45, 59),
            "utterances": (12, 17),
            "actor_portraits": (18, 24),
            "mechanisms": (37, 51),
            "conditions": (21, 27),
            "artifacts": (18, 24),
            "readings": (11, 13),
        }
        assert sum(lower for lower, _ in target_ranges.values()) == 249
        assert sum(upper for _, upper in target_ranges.values()) == 335

    def test_primitive_substrate_target_ranges_for_minified_mode_match_expanded_table(self):
        target_ranges = primitive_substrate_target_ranges_for_mode(PodcastMode.MINIFIED)
        assert target_ranges == {
            "events": (19, 27),
            "acts": (10, 13),
            "utterances": (2, 3),
            "actor_portraits": (6, 8),
            "mechanisms": (7, 12),
            "conditions": (4, 5),
            "artifacts": (3, 4),
            "readings": (2, 2),
        }
        assert sum(lower for lower, _ in target_ranges.values()) == 53
        assert sum(upper for _, upper in target_ranges.values()) == 74

    def test_thematic_project_accepts_forty_sub_themes(self):
        project = ThematicProject(
            theme="Theme",
            sub_themes=[f"s{i}" for i in range(40)],
        )
        assert project.sub_themes == [f"s{i}" for i in range(40)]

    def test_thematic_project_rejects_more_than_forty_sub_themes(self):
        with pytest.raises(ValidationError, match="at most 40 entries"):
            ThematicProject(
                theme="Theme",
                sub_themes=[f"s{i}" for i in range(41)],
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
        assert profile.spoken_style_contract == "anti_academic_oral"
        assert profile.analysis_mode == "hybrid"
        assert profile.analysis_density == "medium"
        assert profile.target_authorial_passages_per_episode == 23

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

    def test_host_presence_beat_accepts_i_address_mode(self):
        beat = HostPresenceBeat.model_validate(
            {
                "kind": "light_aside",
                "placement": "open",
                "seed": "I keep coming back to the ledger on the desk.",
                "address_mode": "i",
            }
        )

        assert beat.address_mode == "i"

    def test_host_move_cue_accepts_i_address_mode(self):
        cue = HostMoveCue(
            move_type="light_aside",
            note="Let the ledger invite one brief personal aside.",
            address_mode="i",
        )

        assert cue.address_mode == "i"

    def test_host_moves_allow_empty_phase_buckets(self):
        moves = HostMovesByPhase()

        assert moves.open == []
        assert moves.pivot == []
        assert moves.close == []

    def test_authorial_passage_accepts_contested_claim_metadata(self):
        passage = AuthorialPassage(
            passage_id="p1",
            mode="quote_then_gloss",
            claim="Some accounts remember the scene differently.",
            claim_certainty="contested_memory",
            source_primitive_ids=["primitive_1"],
            counter_source_passage_ids=["p2", "p3"],
        )

        assert passage.claim_certainty == "contested_memory"
        assert passage.counter_source_passage_ids == ["p2", "p3"]

    def test_episode_takeaway_requires_both_agency_parts(self):
        takeaway = EpisodeTakeaway(
            inherited_condition="The rival bodies had already been stripped.",
            proximate_contingency="A printed insult still had to summon the one left standing.",
        )
        assert takeaway.inherited_condition
        assert takeaway.proximate_contingency
        with pytest.raises(ValidationError):
            EpisodeTakeaway(inherited_condition="Only the structural half.")
        with pytest.raises(ValidationError):
            EpisodeTakeaway(proximate_contingency="Only the choice half.")


class TestSynthesisModels:
    def test_actor_metadata_validates_relationship_references(self):
        metadata = ActorMetadata(
            project_id="proj",
            actors=[
                ActorProfile(
                    actor_id="jawaharlal_nehru",
                    display_name="Jawaharlal Nehru",
                    actor_type="person",
                ),
                ActorProfile(
                    actor_id="indian_national_congress",
                    display_name="Indian National Congress",
                    actor_type="party",
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
                for idx in range(65)
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
        assert len(metadata.actors) == 60
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
            primitives=[
                _event_primitive("e1").model_copy(
                    update={"actor_ids": ["jawaharlal_nehru", "unknown_actor"]}
                )
            ],
        )
        cleaned, metrics = clean_synthesis_primitive_actor_links(artifact, metadata)
        primitive = cleaned.primitives[0]
        assert primitive.actor_ids == ["jawaharlal_nehru"]
        assert metrics["unknown_actor_ids"] == 1

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

    def test_narrative_importance_score_defaults_to_zero_without_salience(self):
        primitive = _event_primitive("e1")

        assert primitive.narrative_importance_score == 0.0


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
            "pack_1": SupportPrimitiveRole.STAKES,
            **{
                f"support_{idx}": (
                    SupportPrimitiveRole.MECHANISM
                    if idx % 2 == 0
                    else SupportPrimitiveRole.TEXTURE
                )
                for idx in range(1, 8)
            },
        }
        with pytest.raises(
            ValidationError,
            match="support primitives cannot also appear in core_primitive_ids",
        ):
            EpisodeSpine(
                listener_problem="What changed?",
                episode_answer="A claim",
                core_primitive_ids=["pack_1", *[f"core_{idx}" for idx in range(2, 8)]],
                support_primitive_roles=support_primitive_roles,
            )

    def test_episode_spine_rejects_support_primitive_count_outside_shared_range(self):
        with pytest.raises(
            ValidationError, match="support_primitive_roles must contain 2-13"
        ):
            EpisodeSpine(
                listener_problem="What changed?",
                episode_answer="A claim",
                core_primitive_ids=[f"core_{idx}" for idx in range(1, 8)],
                support_primitive_roles={
                    "support_1": SupportPrimitiveRole.MECHANISM,
                },
            )

    def test_episode_spine_accepts_shared_upper_bounds_for_full_mode(self):
        spine = EpisodeSpine(
            listener_problem="What changed?",
            episode_answer="A claim",
            core_primitive_ids=[f"core_{idx}" for idx in range(1, 12)],
            support_primitive_roles={
                f"support_{idx}": SupportPrimitiveRole.MECHANISM for idx in range(1, 14)
            },
            recall_primitive_ids=["recall_1", "recall_2", "recall_3"],
        )
        assert len(spine.core_primitive_ids) == 11
        assert len(spine.support_primitive_roles) == 13
        assert len(spine.recall_primitive_ids) == 3

    def test_validate_episode_spine_targets_enforces_full_mode_counts(self):
        spine = EpisodeSpine(
            listener_problem="What changed?",
            episode_answer="A claim",
            core_primitive_ids=[f"core_{idx}" for idx in range(1, 5)],
            support_primitive_roles={
                f"support_{idx}": SupportPrimitiveRole.MECHANISM for idx in range(1, 5)
            },
        )

        with pytest.raises(ValueError, match="core_primitive_ids must contain 8-11"):
            validate_episode_spine_targets(
                spine,
                core_target_min=8,
                core_target_max=11,
                support_target_min=9,
                support_target_max=13,
                recall_target_max=3,
            )

    def test_validate_episode_spine_targets_accepts_full_mode_support_range(self):
        spine = EpisodeSpine(
            listener_problem="What changed?",
            episode_answer="A claim",
            core_primitive_ids=[f"core_{idx}" for idx in range(1, 9)],
            support_primitive_roles={
                f"support_{idx}": SupportPrimitiveRole.MECHANISM for idx in range(1, 10)
            },
        )

        validate_episode_spine_targets(
            spine,
            core_target_min=8,
            core_target_max=11,
            support_target_min=9,
            support_target_max=13,
            recall_target_max=3,
        )

    def test_validate_episode_spine_targets_accepts_minified_counts(self):
        spine = EpisodeSpine(
            listener_problem="What changed?",
            episode_answer="A claim",
            core_primitive_ids=["core_1", "core_2", "core_3"],
            support_primitive_roles={
                "support_1": SupportPrimitiveRole.MECHANISM,
                "support_2": SupportPrimitiveRole.TEXTURE,
                "support_3": SupportPrimitiveRole.STAKES,
                "support_4": SupportPrimitiveRole.CONSEQUENCE,
            },
            recall_primitive_ids=["recall_1"],
        )

        validate_episode_spine_targets(
            spine,
            core_target_min=3,
            core_target_max=5,
            support_target_min=4,
            support_target_max=6,
            recall_target_max=1,
        )

    def test_validate_episode_spine_targets_rejects_minified_recall_overflow(self):
        spine = EpisodeSpine(
            listener_problem="What changed?",
            episode_answer="A claim",
            core_primitive_ids=["core_1", "core_2", "core_3"],
            support_primitive_roles={
                "support_1": SupportPrimitiveRole.MECHANISM,
                "support_2": SupportPrimitiveRole.TEXTURE,
                "support_3": SupportPrimitiveRole.STAKES,
                "support_4": SupportPrimitiveRole.CONSEQUENCE,
            },
            recall_primitive_ids=["recall_1", "recall_2"],
        )

        with pytest.raises(
            ValueError, match="recall_primitive_ids must contain at most 1"
        ):
            validate_episode_spine_targets(
                spine,
                core_target_min=3,
                core_target_max=5,
                support_target_min=4,
                support_target_max=6,
                recall_target_max=1,
            )

    def test_episode_spine_rejects_removed_rhetorical_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EpisodeSpine.model_validate(
                {
                    "listener_problem": "What changed?",
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
                    "listener_problem": "What changed?",
                    "argument": "A claim",
                    "spine_pack_ids": ["pack_1"],
                    "support_pack_roles": {},
                    "allowed_recalls": [],
                }
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
    def test_scene_actor_allows_intro_fields_without_explanation_stage(self):
        actor = SceneActor.model_validate(
            {
                "name": "Muhammad Ali Jinnah",
                "actor_id": "muhammad_ali_jinnah",
                "role_label": "Bombay barrister",
                "source_primitive_ids": ["p_1"],
                "source_passage_ids": ["passage_1"],
                "intro_facts": ["He leads the League."],
                "why_now": "He now enters the scene directly.",
            }
        )

        assert actor.explanation_stage is None
        assert actor.role_label == "Bombay barrister"
        assert actor.source_primitive_ids == ["p_1"]

    def test_scene_card_draft_suggests_scene_function_for_swapped_scene_role(self):
        with pytest.raises(
            ValidationError,
            match="scene_role='mechanism' looks like a scene_function value",
        ):
            SceneCardDraft.model_validate(
                {
                    "scene_id": "scene_1",
                    "section_id": "section_1",
                    "title": "The chain becomes visible",
                    "scene_role": "mechanism",
                    "scene_function": "scene",
                    "beat_change": "The mechanism is now exposed.",
                    "passage_ids": ["p1"],
                    "host_moves": {
                        "open": [{"move_type": "orient", "note": "Enter through the ledger."}]
                    },
                    "estimated_duration_seconds": 90,
                }
            )

    def test_scene_card_draft_suggests_scene_role_for_swapped_scene_function(self):
        with pytest.raises(
            ValidationError,
            match="scene_function='contestation' looks like a scene_role value",
        ):
            SceneCardDraft.model_validate(
                {
                    "scene_id": "scene_1",
                    "section_id": "section_1",
                    "title": "The argument turns public",
                    "scene_role": "action",
                    "scene_function": "contestation",
                    "beat_change": "The disagreement becomes visible.",
                    "passage_ids": ["p1"],
                    "host_moves": {
                        "open": [{"move_type": "orient", "note": "Enter through the council chamber."}]
                    },
                    "estimated_duration_seconds": 90,
                }
            )

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
                    "beat_change": "The mechanism becomes visible.",
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
                    "beat_change": "The mechanism becomes visible.",
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
                beat_change="The argument advances.",
                entry_image="The silence shifts.",
                local_question="How do we move forward?",
                observable_detail="A messenger leaves.",
                intended_move="Bridge occurrences.",
                passage_ids=["p1"],
                host_moves=_host_moves_payload(),
                estimated_duration_seconds=60,
            )


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
                    section_sonic_plan=SectionSonicPlan(
                        obligation=SectionSonicObligation.REQUIRED,
                        opening_anchor="Truck engines idle in the dark.",
                        opening_pressure="The convoy's force arrives before daylight does.",
                    ),
                    sonic_cues=[
                        {
                            "scene_id": "scene_1",
                            "scene_job": "build",
                            "entry_image": "The convoy moves before dawn.",
                            "observable_detail": "The headlamps sweep the wall.",
                            "audible_detail": "Truck engines idle in the dark.",
                        }
                    ],
                )
            ],
        )
        restored = SpokenScript.model_validate(json.loads(spoken.model_dump_json()))
        assert restored.sections[0].speech_hints.style == "measured"
        assert restored.sections[0].section_sonic_plan is not None
        assert restored.sections[0].section_sonic_plan.obligation == "required"
        assert restored.sections[0].sonic_cues[0].audible_detail == (
            "Truck engines idle in the dark."
        )

    def test_prose_section_accepts_section_sonic_plan(self):
        section = ProseSection.model_validate(
            {
                "section_id": "section_1",
                "scene_card_ids": ["scene_1"],
                "movement_goal": "setup",
                "text": "The engines sit there before anyone speaks.",
                "section_sonic_plan": {
                    "obligation": "required",
                    "opening_anchor": "Truck engines idle in the dark.",
                    "opening_pressure": "The convoy's force arrives before daylight does.",
                    "later_beats": [
                        {
                            "moment": "the order hits the room",
                            "cue": "a stamp cracks onto the paper",
                        }
                    ],
                },
            }
        )

        assert section.section_sonic_plan is not None
        assert section.section_sonic_plan.obligation == "required"
        assert section.section_sonic_plan.later_beats[0].cue == (
            "a stamp cracks onto the paper"
        )


class TestEpisodeArchitectureModels:
    def _build_architecture(self, section_count: int) -> EpisodeArchitecture:
        sections = []
        for idx in range(section_count):
            section_id = f"section_{idx + 1:02d}"
            if idx == section_count - 1:
                stage = "close"
            elif idx == section_count - 2:
                stage = "answer"
            elif idx == 0:
                stage = "setup"
            else:
                stage = "advance"
            sections.append(
                ArchitectureSection(
                    section_id=section_id,
                    purpose="opening"
                    if idx == 0
                    else "closing"
                    if idx == section_count - 1
                    else "setup",
                    approx_runtime_minutes=130.0 / section_count,
                    primitive_ids=["et_1"],
                    section_anchor=f"Anchor {idx + 1}",
                    must_stage_beats=[
                        f"Beat {idx + 1}A",
                        f"Beat {idx + 1}B",
                    ],
                    section_progression=make_section_progression(stage, label=section_id),
                )
            )
        return EpisodeArchitecture(
            episode_number=1,
            major_turn_section_id=sections[min(2, len(sections) - 1)].section_id,
            allowed_recurring_primitive_ids=[],
            forbidden_redundancies=[],
            sections=sections,
            architecture_notes=[],
        )

    @staticmethod
    def _ap(mode: str, placement: str = "mid") -> dict:
        return {
            "passage_id": "p1",
            "mode": mode,
            "placement": placement,
            "claim": "x",
            "source_primitive_ids": ["et_1"],
        }

    def _architecture_with_passage_in(
        self, *, stage_index_from_end: int, mode: str, placement: str = "mid"
    ) -> dict:
        """Return an architecture payload (dict) with one authorial passage injected
        into the section `stage_index_from_end` from the end (1 = close, 2 = answer)."""
        payload = self._build_architecture(9).model_dump(mode="json")
        target = payload["sections"][len(payload["sections"]) - stage_index_from_end]
        target["authorial_passages"] = [self._ap(mode, placement)]
        return payload

    def test_authorial_passage_rejects_verdict_landing_at_close(self):
        with pytest.raises(ValidationError, match="placement='close'"):
            AuthorialPassage.model_validate(self._ap("verdict_landing", "close"))

    def test_authorial_passage_rejects_causal_compression_at_close(self):
        with pytest.raises(ValidationError, match="placement='close'"):
            AuthorialPassage.model_validate(self._ap("causal_compression", "close"))

    def test_architecture_accepts_single_verdict_in_answer_section(self):
        payload = self._architecture_with_passage_in(
            stage_index_from_end=2, mode="verdict_landing"
        )
        architecture = EpisodeArchitecture.model_validate(payload)
        answer_section = architecture.sections[-2]
        assert answer_section.authorial_passages[0].mode == "verdict_landing"

    def test_architecture_rejects_verdict_landing_outside_answer_section(self):
        # section index 1 (from start) is an `advance` stage, not the answer section.
        payload = self._build_architecture(9).model_dump(mode="json")
        payload["sections"][1]["authorial_passages"] = [self._ap("verdict_landing")]
        with pytest.raises(ValidationError, match="answer-stage section"):
            EpisodeArchitecture.model_validate(payload)

    def test_architecture_rejects_verdict_landing_in_close_section(self):
        payload = self._architecture_with_passage_in(
            stage_index_from_end=1, mode="verdict_landing"
        )
        with pytest.raises(ValidationError, match="answer-stage section"):
            EpisodeArchitecture.model_validate(payload)

    def test_architecture_rejects_causal_compression_in_close_section(self):
        payload = self._architecture_with_passage_in(
            stage_index_from_end=1, mode="causal_compression"
        )
        with pytest.raises(ValidationError, match="close section may not carry"):
            EpisodeArchitecture.model_validate(payload)

    def test_episode_architecture_rejects_removed_target_section_count_field(self):
        with pytest.raises(ValidationError, match="target_section_count"):
            EpisodeArchitecture.model_validate(
                {
                    **self._build_architecture(9).model_dump(mode="json"),
                    "target_section_count": 6,
                }
            )

    def test_episode_architecture_rejects_fewer_than_six_sections(self):
        with pytest.raises(ValidationError, match="at least 6 items"):
            self._build_architecture(5)

    def test_episode_architecture_accepts_thirteen_sections(self):
        architecture = self._build_architecture(13)

        assert len(architecture.sections) == 13

    def test_architecture_section_drops_legacy_fields(self):
        section = ArchitectureSection.model_validate(
            {
                "section_id": "section_01",
                "purpose": "opening",
                "approx_runtime_minutes": 10.0,
                "primitive_ids": ["et_1"],
                "section_anchor": "Opening anchor",
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
                "section_progression": make_section_progression("setup", label="section_01"),
            }
        )

        payload = section.model_dump(mode="json")
        assert section.section_anchor == "Opening anchor"
        assert payload["section_anchor"] == "Opening anchor"
        assert "section_question" not in payload

    def test_architecture_section_open_mode_defaults_and_round_trips(self):
        base = {
            "section_id": "section_01",
            "purpose": "opening",
            "approx_runtime_minutes": 10.0,
            "primitive_ids": ["et_1"],
            "section_anchor": "Anchor",
            "must_stage_beats": ["Visible move", "Immediate consequence"],
            "section_progression": make_section_progression("setup", label="section_01"),
        }

        default_section = ArchitectureSection.model_validate(base)
        assert default_section.open_mode == "scene_anchor"
        assert default_section.model_dump(mode="json")["open_mode"] == "scene_anchor"

        question_section = ArchitectureSection.model_validate(
            {**base, "open_mode": "question_first"}
        )
        assert question_section.open_mode == "question_first"
        assert (
            question_section.model_dump(mode="json")["open_mode"] == "question_first"
        )

        with pytest.raises(ValidationError):
            ArchitectureSection.model_validate({**base, "open_mode": "montage"})

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

    def test_architecture_section_rejects_blank_section_sonic_plan_fields(self):
        with pytest.raises(ValidationError):
            ArchitectureSection.model_validate(
                {
                    "section_id": "section_01",
                    "purpose": "opening",
                    "approx_runtime_minutes": 10.0,
                    "primitive_ids": ["et_1"],
                    "section_anchor": "Anchor",
                    "must_stage_beats": ["Beat one", "Beat two"],
                    "section_sonic_plan": {
                        "obligation": "required",
                        "opening_anchor": "   ",
                        "opening_pressure": "   ",
                    },
                }
            )

    def test_architecture_section_rejects_more_than_two_section_sonic_later_beats(self):
        with pytest.raises(ValidationError, match="at most 2 items"):
            ArchitectureSection.model_validate(
                {
                    "section_id": "section_01",
                    "purpose": "opening",
                    "approx_runtime_minutes": 10.0,
                    "primitive_ids": ["et_1"],
                    "section_anchor": "Anchor",
                    "must_stage_beats": ["Beat one", "Beat two"],
                    "section_sonic_plan": {
                        "obligation": "preferred",
                        "opening_anchor": "A crowd goes quiet.",
                        "opening_pressure": "The silence marks organized refusal.",
                        "later_beats": [
                            {"moment": "m1", "cue": "c1"},
                            {"moment": "m2", "cue": "c2"},
                            {"moment": "m3", "cue": "c3"},
                        ],
                    },
                }
            )


def test_episode_planning_model_dump_preserves_answer_scene_and_drops_residue():
    scene_card = SceneCardDraft.model_validate(
        {
            "scene_id": "scene_1",
            "section_id": "section_1",
            "title": "The order arrives",
            "scene_role": "setup",
            "scene_job": "answer",
            "beat_change": "The stakes become legible.",
            "entry_image": "A clerk opens the envelope.",
            "observable_detail": "Hands freeze over the paper.",
            "intended_move": "Move from abstract policy to lived consequence.",
            "primitive_ids": ["et_1"],
            "passage_ids": ["p1", "p2"],
            "host_moves": _host_moves_payload().model_dump(mode="json"),
            "estimated_duration_seconds": 600,
        }
    )
    draft = EpisodePlanDraft(
        episode_number=1,
        framing=_framing(),
        scene_cards=[scene_card],
        answer_scene_card_id="scene_1",
    )
    canonical_payload = draft.model_dump(mode="json")
    assert "answer_scene_card_id" in canonical_payload
    assert "residue_scene_card_id" not in canonical_payload

    revalidated = EpisodePlanDraft.model_validate(canonical_payload)
    assert revalidated.answer_scene_card_id == "scene_1"
