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
    LiveQuestion,
    NarrativeStrategy,
    SceneActor,
    SceneCard,
    SpeechHints,
    SpokenScript,
    SpokenSection,
    SpokenTransition,
    StrategyEpisode,
    StyleAuditReport,
    StyleWarning,
    SynthesisMap,
    ThematicProject,
    TurningPoint,
)


def _turning_point(primitive_id: str = "tp_1") -> TurningPoint:
    return TurningPoint(
        id=primitive_id,
        title="Threshold breaks",
        summary="A decision changes the field.",
        axis_ids=["axis_1"],
        core_passage_ids=["p1"],
        support_passage_ids=["p2"],
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
                turning_points=[_turning_point("tp_1")],
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
            LiveQuestion(
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

    def test_synthesis_map_roundtrip_preserves_cluster_first_shape(self):
        synthesis_map = SynthesisMap(
            project_id="proj",
            turning_points=[_turning_point("tp_1")],
            live_questions=[
                LiveQuestion(
                    id="lq_1",
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
            ],
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
        assert restored.live_questions[0].candidate_readings[1].label == "reading_b"


class TestNarrativeStrategy:
    def test_strategy_episode_requires_primary_on_first_and_last_occurrence(self):
        with pytest.raises(ValidationError, match="first cluster_path occurrence must be primary"):
            StrategyEpisode(
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


class TestPlanningModels:
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

    def test_style_audit_report_counts_roundtrip(self):
        report = StyleAuditReport(
            episode_number=1,
            warnings=[
                StyleWarning(
                    warning_type="author_hand_language",
                    text_unit_id="section_1",
                    message="Avoid telling the listener what the author means.",
                )
            ],
            counts_by_type={"author_hand_language": 1},
        )
        restored = StyleAuditReport.model_validate(json.loads(report.model_dump_json()))
        assert restored.counts_by_type["author_hand_language"] == 1
