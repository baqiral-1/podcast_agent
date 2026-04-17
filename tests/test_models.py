"""Unit tests for the redesigned schema contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
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
    ThematicProject,
    PipelineConfig,
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
        actors=[SceneActor(name="Clerk", role_in_scene="recipient")],
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

    def test_pipeline_config_defaults_synthesis_cap_to_800(self):
        config = PipelineConfig()
        assert config.synthesis_total_passage_cap == 800
        assert config.min_episode_minutes == 85.0
        assert config.target_episode_minutes == 110.0


class TestSynthesisModels:
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

    def test_live_question_requires_at_least_two_candidate_readings(self):
        with pytest.raises(ValidationError):
            SynthesisMap(
                project_id="proj",
                primitives_by_family=_family_map(
                    live_questions=[
                        SynthesisPrimitive(
                            id="lq_1",
                            title="Competing readings",
                            summary="The evidence supports multiple paths.",
                            core_passage_ids=["p1"],
                            candidate_readings=[
                                CandidateReading(
                                    label="reading_a",
                                    summary="Only one reading present.",
                                    support_passage_ids=["p1"],
                                )
                            ],
                        )
                    ]
                ),
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
        )
        assert card.scene_role == "reveal"

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

    def test_episode_plan_draft_defaults_target_duration_minutes_to_110(self):
        draft = EpisodePlanDraft(
            episode_number=1,
            title="Episode 1",
            driving_question="Why begin here?",
            framing=_framing(),
            scene_cards=[_normal_scene()],
        )
        assert draft.target_duration_minutes == 110.0

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
