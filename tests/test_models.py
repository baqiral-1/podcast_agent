"""Unit tests for the redesigned schema contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorArcRef,
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    CandidateReading,
    ClusterPathOccurrence,
    EpisodeCandidateCluster,
    EpisodePlan,
    EpisodePlanDraft,
    FramingBlock,
    NarrativeStrategy,
    ProseSection,
    ScriptTransition,
    SceneActor,
    SceneActorArcBinding,
    SceneCard,
    SpeechHints,
    SpokenScript,
    SpokenSection,
    SpokenTransition,
    StrategyEpisode,
    SYNTHESIS_PRIMITIVE_FAMILIES,
    SynthesisConsolidationResult,
    SynthesisMap,
    SynthesisPrimitive,
    SynthesisPrimitivesArtifact,
    ThematicProject,
    PipelineConfig,
)
from podcast_agent.utils.actor_metadata import (
    ActorMatcher,
    clean_scene_actor_links,
    clean_synthesis_primitive_actor_links,
    normalize_actor_name,
    sanitize_actor_metadata_payload,
)


def _turning_point(primitive_id: str = "tp_1") -> SynthesisPrimitive:
    return SynthesisPrimitive(
        id=primitive_id,
        title="Threshold breaks",
        summary="A decision changes the field.",
        axis_ids=["axis_1"],
        core_passage_ids=["p1"],
        support_passage_ids=["p2"],
    )


def _live_question(primitive_id: str = "lq_1") -> SynthesisPrimitive:
    return SynthesisPrimitive(
        id=primitive_id,
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


def _family_map(**overrides: list[SynthesisPrimitive]) -> dict[str, list[SynthesisPrimitive]]:
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


def _normal_scene(scene_id: str = "scene_1", occurrence_id: str = "occ_1") -> SceneCard:
    return SceneCard(
        scene_id=scene_id,
        title="The order arrives",
        scene_role="setup",
        dominant_cluster_occurrence_id=occurrence_id,
        entry_image="A clerk opens the envelope.",
        local_question="What changes first?",
        observable_detail="Hands freeze over the paper.",
        intended_move="Move from abstract policy to lived consequence.",
        actors=[SceneActor(name="Clerk", presence="background")],
        primitive_ids=["tp_1"],
        passage_ids=["p1", "p2"],
        estimated_duration_seconds=600,
    )


class TestThematicProject:
    def test_recommended_episode_count_accepts_new_bounds(self):
        project = ThematicProject(theme="War on terror", recommended_episode_count=6)
        assert project.recommended_episode_count == 6

    def test_recommended_episode_count_rejects_old_lower_bound(self):
        with pytest.raises(ValidationError):
            ThematicProject(theme="War on terror", recommended_episode_count=5)

    def test_pipeline_config_rejects_scene_card_bound_inversion(self):
        with pytest.raises(ValidationError, match="scene_card_target_max"):
            PipelineConfig(scene_card_target_min=40, scene_card_target_max=25)

    def test_pipeline_config_rejects_removed_scene_batch_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(scene_batch_min_cards=1)

    def test_pipeline_config_defaults_synthesis_cap_to_600(self):
        config = PipelineConfig()
        assert config.synthesis_total_passage_cap == 600
        assert config.min_episode_minutes == 85.0
        assert config.target_episode_minutes == 90.0
        assert config.scene_card_target_min == 30
        assert config.scene_card_target_max == 45


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
                turning_points=[
                    SynthesisPrimitive(
                        id="tp_1",
                        title="Decision",
                        summary="A decision lands.",
                        core_passage_ids=["p1"],
                        actor_tags=["Nehru", "Unknown actor"],
                    )
                ]
            ),
        )
        cleaned, metrics = clean_synthesis_primitive_actor_links(artifact, metadata)
        primitive = cleaned.primitives_by_family["turning_points"][0]
        assert primitive.actor_ids == ["jawaharlal_nehru"]
        assert primitive.actor_tags == []
        assert primitive.unresolved_actor_tags == ["Unknown actor"]
        assert metrics["exact_actor_tag_matches"] == 1
        assert metrics["unmatched_actor_tags"] == 1

    def test_episode_candidate_cluster_requires_primary_member_in_member_ids(self):
        with pytest.raises(ValidationError, match="primary_member_id"):
            EpisodeCandidateCluster(
                title="Cluster",
                summary="Summary",
                primary_member_id="tp_missing",
                member_ids=["tp_1"],
                local_question="What changes?",
                local_payoff_shape="reveal",
            )

    def test_importance_fields_default_and_roundtrip(self):
        primitive = _turning_point("tp_1")
        cluster = EpisodeCandidateCluster(
            cluster_id="cluster_1",
            title="Cluster",
            summary="Summary",
            primary_member_id="tp_1",
            member_ids=["tp_1"],
            local_question="What changes?",
            local_payoff_shape="reveal",
        )

        assert primitive.narrative_importance_score == 0.5
        assert cluster.narrative_importance_score == 0.5
        assert cluster.coverage_policy == "supporting"

        restored = EpisodeCandidateCluster.model_validate(
            {
                **cluster.model_dump(mode="json"),
                "narrative_importance_score": 0.85,
                "coverage_policy": "anchor",
            }
        )
        assert restored.narrative_importance_score == 0.85
        assert restored.coverage_policy == "anchor"

    def test_synthesis_map_rejects_unknown_cluster_member_ids(self):
        with pytest.raises(ValidationError, match="unknown member_ids"):
            SynthesisMap(
                project_id="proj",
                primitives_by_family=_family_map(turning_points=[_turning_point("tp_1")]),
                episode_candidate_clusters=[
                    EpisodeCandidateCluster(
                        cluster_id="cluster_1",
                        title="Cluster",
                        summary="Summary",
                        primary_member_id="tp_1",
                        member_ids=["tp_1", "tp_999"],
                        local_question="What changes?",
                        local_payoff_shape="reveal",
                    )
                ],
            )

    def test_live_questions_with_fewer_than_two_candidate_readings_are_dropped(self):
        invalid_live_question = SynthesisPrimitive(
            id="lq_invalid",
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
                turning_points=[_turning_point("tp_1")],
                live_questions=[invalid_live_question, _live_question("lq_valid")],
            ),
        )
        synthesis_map = SynthesisMap(
            project_id="proj",
            primitives_by_family=_family_map(
                turning_points=[_turning_point("tp_1")],
                live_questions=[invalid_live_question, _live_question("lq_valid")],
            ),
        )

        assert [item.id for item in artifact.primitives_by_family["live_questions"]] == [
            "lq_valid"
        ]
        assert [item.id for item in synthesis_map.primitives_by_family["live_questions"]] == [
            "lq_valid"
        ]
        assert [item.id for item in artifact.primitives_by_family["turning_points"]] == ["tp_1"]

    def test_synthesis_map_rejects_cluster_references_to_dropped_live_questions(self):
        with pytest.raises(ValidationError, match="unknown member_ids"):
            SynthesisMap(
                project_id="proj",
                primitives_by_family=_family_map(
                    turning_points=[_turning_point("tp_1")],
                    live_questions=[
                        SynthesisPrimitive(
                            id="lq_invalid",
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
                episode_candidate_clusters=[
                    EpisodeCandidateCluster(
                        cluster_id="cluster_1",
                        title="Opening cluster",
                        summary="A compact causal chain.",
                        primary_member_id="tp_1",
                        member_ids=["tp_1", "lq_invalid"],
                        local_question="Why does the order matter?",
                        local_payoff_shape="reveal",
                    )
                ],
            )

    def test_synthesis_map_roundtrip_preserves_cluster_first_shape(self):
        synthesis_map = SynthesisMap(
            project_id="proj",
            primitives_by_family=_family_map(
                turning_points=[_turning_point("tp_1")],
                live_questions=[_live_question("lq_1")],
            ),
            episode_candidate_clusters=[
                EpisodeCandidateCluster(
                    cluster_id="cluster_1",
                    title="Opening cluster",
                    summary="A compact causal chain.",
                    primary_member_id="tp_1",
                    member_ids=["tp_1", "lq_1"],
                    local_question="Why does the order matter?",
                    local_payoff_shape="reveal",
                )
            ],
            quality_score=0.7,
        )
        restored = SynthesisMap.model_validate(json.loads(synthesis_map.model_dump_json()))
        assert restored.episode_candidate_clusters[0].primary_member_id == "tp_1"
        assert (
            restored.primitives_by_family["live_questions"][0].candidate_readings[1].label
            == "reading_b"
        )

    def test_synthesis_consolidation_result_rejects_unknown_cluster_member_ids(self):
        with pytest.raises(ValidationError, match="unknown member_ids"):
            SynthesisConsolidationResult(
                project_id="proj",
                primitive_ids_by_family={"turning_points": ["tp_1"]},
                episode_candidate_clusters=[
                    EpisodeCandidateCluster(
                        cluster_id="cluster_1",
                        title="Cluster",
                        summary="Summary",
                        primary_member_id="tp_1",
                        member_ids=["tp_1", "tp_999"],
                        local_question="What changes?",
                        local_payoff_shape="reveal",
                    )
                ],
            )

    def test_synthesis_consolidation_result_accepts_family_id_universe(self):
        result = SynthesisConsolidationResult(
            project_id="proj",
            primitive_ids_by_family={
                "turning_points": ["tp_1"],
                "live_questions": ["lq_1"],
            },
            episode_candidate_clusters=[
                EpisodeCandidateCluster(
                    cluster_id="cluster_1",
                    title="Cluster",
                    summary="Summary",
                    primary_member_id="tp_1",
                    member_ids=["tp_1", "lq_1"],
                    local_question="What changes?",
                    local_payoff_shape="reveal",
                )
            ],
        )
        assert result.episode_candidate_clusters[0].member_ids == ["tp_1", "lq_1"]


class TestNarrativeStrategy:
    def test_actor_arc_directive_roundtrip_uses_arc_refs(self):
        directive = ActorArcDirective(
            actor_id="mahatma_gandhi",
            arc_refs=[
                ActorArcRef(
                    ref_id="gandhi_role_1",
                    arc_type="role",
                    label="episode role",
                    premise="His restraint frames the episode's character function.",
                    pressure="Escalating violence narrows his choices.",
                    movement="Restraint becomes harder to sustain across scenes.",
                    payoff="The listener sees restraint as a contested strategy.",
                ),
                ActorArcRef(
                    ref_id="gandhi_tracking_1",
                    arc_type="tracking",
                    label="listener tracking",
                    premise="Track how restraint becomes harder to sustain.",
                    pressure="Public discipline collides with escalating violence.",
                    movement="Each appearance should add pressure.",
                    payoff="The episode clarifies what restraint costs.",
                ),
                ActorArcRef(
                    ref_id="gandhi_tension_1",
                    arc_type="tension",
                    label="tension line",
                    premise="Public discipline collides with escalating violence.",
                    pressure="Followers and opponents both test discipline.",
                    movement="Pressure accumulates until restraint changes meaning.",
                    payoff="The tension remains visible at the episode close.",
                ),
                ActorArcRef(
                    ref_id="gandhi_progression_1",
                    arc_type="turn",
                    label="arc progression",
                    premise="The episode changes what restraint can plausibly mean.",
                    pressure="A decision changes the arc.",
                    movement="Move from principle to tested practice.",
                    payoff="Restraint lands as a consequential choice.",
                ),
                ActorArcRef(
                    ref_id="gandhi_scene_job_1",
                    arc_type="payoff",
                    label="scene job",
                    premise="Use where restraint meets consequence.",
                    pressure="The scene must do more than repeat the actor function.",
                    movement="Bind actor presence to scene consequence.",
                    payoff="The scene makes the actor arc legible.",
                ),
                ActorArcRef(
                    ref_id="gandhi_guardrail_1",
                    arc_type="guardrail",
                    label="repetition guardrail",
                    premise="Do not re-explain restraint unless the scene changes its meaning.",
                    pressure="Repeated appearances can flatten the actor.",
                    movement="Vary actor function across scenes.",
                    payoff="The actor remains continuous without becoming repetitive.",
                ),
            ],
        )

        restored = ActorArcDirective.model_validate(json.loads(directive.model_dump_json()))

        assert restored.actor_id == "mahatma_gandhi"
        assert restored.arc_refs[1].ref_id == "gandhi_tracking_1"
        assert restored.arc_refs[1].arc_type == "tracking"
        assert restored.arc_refs[1].label == "listener tracking"
        assert restored.arc_refs[1].pressure == "Public discipline collides with escalating violence."

    def test_actor_arc_directive_rejects_duplicate_ref_ids(self):
        with pytest.raises(ValidationError, match="ref ids must be unique"):
            ActorArcDirective(
                actor_id="mahatma_gandhi",
                arc_refs=[
                    ActorArcRef(
                        ref_id="gandhi_1",
                        arc_type="role",
                        label="role",
                        premise="Role",
                    ),
                    ActorArcRef(
                        ref_id="gandhi_1",
                        arc_type="payoff",
                        label="scene job",
                        premise="Scene job",
                    ),
                ],
            )

    def test_strategy_episode_rejects_old_episode_actor_fields(self):
        with pytest.raises(ValidationError):
            StrategyEpisode.model_validate(
                {
                    "episode_number": 1,
                    "title": "Episode 1",
                    "driving_question": "What changed?",
                    "arc_summary": "Arc",
                    "actor_ids": ["mahatma_gandhi"],
                    "primary_actor_ids": ["mahatma_gandhi"],
                    "actor_arc_summary": "Old summary.",
                    "cluster_path": [
                        {
                            "occurrence_id": "occ_1",
                            "cluster_id": "cluster_1",
                            "usage": "primary",
                        }
                    ],
                }
            )

    def test_strategy_episode_allows_echo_on_first_and_last_occurrence(self):
        episode = StrategyEpisode(
            episode_number=1,
            title="Episode 1",
            driving_question="What changed?",
            arc_summary="Arc",
            cluster_path=[
                ClusterPathOccurrence(
                    occurrence_id="occ_1",
                    cluster_id="cluster_1",
                    usage="echo",
                    transition_note="",
                )
            ],
        )
        assert episode.cluster_path[0].usage == "echo"
        assert episode.cluster_path[0].emphasis == "supporting"

    def test_cluster_path_occurrence_accepts_emphasis(self):
        occurrence = ClusterPathOccurrence(
            occurrence_id="occ_1",
            cluster_id="cluster_1",
            usage="primary",
            emphasis="anchor",
        )
        assert occurrence.emphasis == "anchor"

    def test_strategy_episode_requires_transition_notes_after_first_occurrence(self):
        with pytest.raises(ValidationError, match="transition_note"):
            StrategyEpisode(
                episode_number=1,
                title="Episode 1",
                driving_question="What changed?",
                arc_summary="Arc",
                cluster_path=[
                    ClusterPathOccurrence(
                        occurrence_id="occ_1",
                        cluster_id="cluster_1",
                        usage="primary",
                        transition_note="",
                    ),
                    ClusterPathOccurrence(
                        occurrence_id="occ_2",
                        cluster_id="cluster_1",
                        usage="primary",
                        transition_note="",
                    ),
                ],
            )

    def test_narrative_strategy_rejects_primary_cluster_in_multiple_home_episodes(self):
        with pytest.raises(ValidationError, match="multiple primary home episodes"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Use converging local causal chains.",
                series_arc="Each episode carries one cluster home.",
                episode_arc_outline=["Ep 1", "Ep 2"],
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode 1",
                        driving_question="Why begin here?",
                        arc_summary="Arc 1",
                        cluster_path=[
                            ClusterPathOccurrence(
                                occurrence_id="occ_1",
                                cluster_id="cluster_1",
                                usage="primary",
                                transition_note="",
                            )
                        ],
                    ),
                    StrategyEpisode(
                        episode_number=2,
                        title="Episode 2",
                        driving_question="Why return?",
                        arc_summary="Arc 2",
                        cluster_path=[
                            ClusterPathOccurrence(
                                occurrence_id="occ_2",
                                cluster_id="cluster_1",
                                usage="primary",
                                transition_note="",
                            )
                        ],
                    ),
                ],
            )

    def test_narrative_strategy_requires_at_least_one_primary_cluster_per_episode(self):
        with pytest.raises(ValidationError, match="must contain at least one primary cluster"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Use converging local causal chains.",
                series_arc="Each episode carries one cluster home.",
                episode_arc_outline=["Ep 1"],
                episodes=[
                    StrategyEpisode(
                        episode_number=1,
                        title="Episode 1",
                        driving_question="Why begin here?",
                        arc_summary="Arc 1",
                        cluster_path=[
                            ClusterPathOccurrence(
                                occurrence_id="occ_1",
                                cluster_id="cluster_1",
                                usage="echo",
                                transition_note="",
                            )
                        ],
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
                    ref_id="gandhi_tension_1",
                    scene_role="blocked",
                    scene_use="complicate",
                    weight="strong",
                )
            ],
        )

        assert actor.presence == "primary"
        assert actor.arc_bindings[0].ref_id == "gandhi_tension_1"
        assert actor.arc_bindings[0].scene_role == "blocked"
        assert actor.arc_bindings[0].scene_use == "complicate"

    def test_scene_actor_rejects_old_actor_aspect_fields(self):
        with pytest.raises(ValidationError):
            SceneActor.model_validate(
                {
                    "name": "Mahatma Gandhi",
                    "role_in_scene": "tests restraint",
                    "arc_ref_ids": ["gandhi_tension_1"],
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
            arc_refs=[
                ActorArcRef(
                    ref_id="gandhi_tension_1",
                    arc_type="tension",
                    label="tension line",
                    premise="Public discipline collides with escalating violence.",
                )
            ],
        )
        scene = _normal_scene()
        scene.actors = [
            SceneActor(
                name="Mahatma Gandhi",
                actor_id="mahatma_gandhi",
                arc_bindings=[
                    SceneActorArcBinding(
                        ref_id="gandhi_tension_1",
                        scene_role="blocked",
                        scene_use="complicate",
                    ),
                    SceneActorArcBinding(
                        ref_id="missing_ref",
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
        assert [binding.ref_id for binding in cleaned_actor.arc_bindings] == ["gandhi_tension_1"]
        assert metrics["unknown_actor_arc_ref_ids"] == 1

    def test_scene_card_allows_noncanonical_scene_role(self):
        card = SceneCard(
            scene_id="scene_reveal",
            title="A hidden mechanism surfaces",
            scene_role="reveal",
            dominant_cluster_occurrence_id="ep1_occ1",
            entry_image="The ledger opens.",
            local_question="What finally becomes visible?",
            observable_detail="A missing column is now clear.",
            intended_move="Expose the mechanism.",
            passage_ids=["p1"],
            estimated_duration_seconds=90,
        )
        assert card.scene_role == "reveal"
        assert card.coverage_depth == "standard"

    def test_scene_card_accepts_coverage_depth(self):
        card = SceneCard(
            scene_id="scene_deep",
            title="A hidden mechanism surfaces",
            scene_role="reveal",
            dominant_cluster_occurrence_id="ep1_occ1",
            entry_image="The ledger opens.",
            local_question="What finally becomes visible?",
            observable_detail="A missing column is now clear.",
            intended_move="Expose the mechanism.",
            passage_ids=["p1"],
            estimated_duration_seconds=120,
            coverage_depth="deep",
        )
        assert card.coverage_depth == "deep"

    def test_scene_card_rejects_legacy_narrative_weight_field(self):
        with pytest.raises(ValidationError):
            SceneCard.model_validate(
                {
                    "scene_id": "scene_deep",
                    "title": "A hidden mechanism surfaces",
                    "scene_role": "reveal",
                    "dominant_cluster_occurrence_id": "ep1_occ1",
                    "entry_image": "The ledger opens.",
                    "local_question": "What finally becomes visible?",
                    "observable_detail": "A missing column is now clear.",
                    "intended_move": "Expose the mechanism.",
                    "passage_ids": ["p1"],
                    "estimated_duration_seconds": 120,
                    "narrative_weight": 2.5,
                    "coverage_depth": "deep",
                }
            )

    def test_scene_card_rejects_blank_scene_role(self):
        with pytest.raises(ValidationError, match="scene_role must not be blank"):
            SceneCard(
                scene_id="scene_blank",
                title="Invalid role",
                scene_role="   ",
                dominant_cluster_occurrence_id="ep1_occ1",
                entry_image="Image",
                local_question="Question",
                observable_detail="Detail",
                intended_move="Move",
                passage_ids=["p1"],
                estimated_duration_seconds=60,
            )

    def test_normal_scene_card_requires_dominant_occurrence(self):
        with pytest.raises(ValidationError, match="normal scene cards require"):
            SceneCard(
                scene_id="scene_1",
                title="The order arrives",
                scene_role="setup",
                entry_image="Envelope on desk.",
                local_question="What changes first?",
                observable_detail="Hands stop.",
                intended_move="Set up the stakes.",
                passage_ids=["p1"],
                estimated_duration_seconds=60,
            )

    def test_bridge_scene_card_requires_bridge_references(self):
        with pytest.raises(ValidationError, match="bridge scene cards require"):
            SceneCard(
                scene_id="scene_bridge",
                title="Bridge",
                card_kind="bridge",
                scene_role="synthesis",
                entry_image="The silence shifts.",
                local_question="How do we move forward?",
                observable_detail="A messenger leaves.",
                intended_move="Bridge occurrences.",
                passage_ids=["p1"],
                estimated_duration_seconds=60,
            )

    def test_episode_plan_draft_validates_framing_handoff_and_bridge_limit(self):
        with pytest.raises(ValidationError, match="handoff_scene_card_id"):
            EpisodePlanDraft(
                episode_number=1,
                title="Episode 1",
                driving_question="Why begin here?",
                framing=FramingBlock(
                    opening_image="Image",
                    threat_or_unresolved_action="Threat",
                    opening_question="Question",
                    handoff_scene_card_id="missing_scene",
                ),
                scene_cards=[_normal_scene()],
            )

    def test_episode_plan_draft_allows_non_scene_withhold_until_reference(self):
        scene = _normal_scene()
        scene.withhold_until = "Full consequences are developed in Episode 5."
        draft = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            framing=_framing(),
            scene_cards=[scene],
        )
        assert draft.scene_cards[0].withhold_until == "Full consequences are developed in Episode 5."

    def test_episode_plan_draft_defaults_target_duration_minutes_to_90(self):
        draft = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            framing=_framing(),
            scene_cards=[_normal_scene()],
        )
        assert draft.target_duration_minutes == 90.0

    def test_episode_plan_roundtrip(self):
        draft = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            thematic_focus="Opening moves",
            arc_summary="The episode traces one local causal chain.",
            unresolved_questions=["What still remains unclear?"],
            framing=_framing(),
            scene_cards=[_normal_scene()],
            target_duration_minutes=140.0,
            target_word_count=16800,
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

    def test_script_transition_allows_empty_text(self):
        transition = ScriptTransition.model_validate(
            {
                "transition_id": "transition_1",
                "after_section_id": "section_1",
                "text": "",
            }
        )
        assert transition.text == ""

    def test_spoken_script_roundtrip_preserves_sections_and_transitions(self):
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
            transitions=[
                SpokenTransition(
                    transition_id="transition_1",
                    text="Then the order reaches the city.",
                )
            ],
        )
        restored = SpokenScript.model_validate(json.loads(spoken.model_dump_json()))
        assert restored.sections[0].speech_hints.style == "measured"
        assert restored.transitions[0].transition_id == "transition_1"
