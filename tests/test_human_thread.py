"""Tests for the narrator persona (T1) and human thread (T3) contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorPrimitive,
    ActorProfile,
    ArchitectureSection,
    EventPrimitive,
    HumanThread,
    HumanThreadMember,
    NarratorPersona,
    PrimitiveSalience,
    PrimitiveSubstrate,
    SectionThreadRef,
    SeriesNarratorProfile,
    SynthesisPrimitivesArtifact,
    effective_narrator_allowed_moves,
)
from podcast_agent.pipeline.orchestrator import (
    _build_host_policy_payload,
    _build_human_thread_candidate_index,
)


def _anchor(**overrides) -> HumanThreadMember:
    base = dict(
        member_id="anchor",
        display_name="Anchor",
        role="anchor",
        grounding_primitive_ids=["p1"],
        grounding_passage_ids=["x1"],
    )
    base.update(overrides)
    return HumanThreadMember(**base)


# --------------------------------------------------------------------------- T1


def test_persona_aside_survives_move_filter():
    # The effective-moves filter drops anything not in the default tuple; the new
    # value must survive so an authored persona can use it.
    assert "persona_aside" in effective_narrator_allowed_moves(["persona_aside"])
    assert "persona_aside" in SeriesNarratorProfile().allowed_moves


def test_host_policy_gates_persona_aside_on_authored_persona():
    bare = _build_host_policy_payload(SeriesNarratorProfile())
    assert "persona_aside" not in bare["allowed_moves"]
    assert bare["persona"] is None
    assert bare["target_persona_asides_per_episode"] == 0
    assert "persona_aside" not in bare["pronoun_policy"]["first_person_singular_allowed_for"]

    profile = SeriesNarratorProfile(
        persona=NarratorPersona(temperament="dry", intellectual_obsessions=["why states fall"])
    )
    rich = _build_host_policy_payload(profile)
    assert "persona_aside" in rich["allowed_moves"]
    assert rich["persona"]["temperament"] == "dry"
    assert rich["target_persona_asides_per_episode"] == 3
    assert "persona_aside" in rich["pronoun_policy"]["first_person_singular_allowed_for"]


# --------------------------------------------------------------------------- T3


def test_member_requires_grounding_floor():
    with pytest.raises(ValidationError):
        HumanThreadMember(
            member_id="m", display_name="M", role="anchor", grounding_passage_ids=["x"]
        )
    with pytest.raises(ValidationError):
        HumanThreadMember(
            member_id="m", display_name="M", role="anchor", grounding_primitive_ids=["p"]
        )


def test_person_thread_must_be_singleton_with_one_anchor():
    thread = HumanThread(
        thread_id="t1",
        thread_key="qom_family",
        kind="person",
        label="one cleric",
        premise="a single life under pressure",
        members=[_anchor()],
        anchor_member_id="anchor",
    )
    assert thread.kind.value == "person"

    with pytest.raises(ValidationError):  # person kind cannot have two members
        HumanThread(
            thread_id="t2",
            thread_key="k",
            kind="person",
            label="x",
            premise="p",
            members=[_anchor(), _anchor(member_id="kin", role="kin")],
            anchor_member_id="anchor",
        )


def test_family_thread_requires_two_members_and_unique_anchor():
    with pytest.raises(ValidationError):  # family needs >= 2
        HumanThread(
            thread_id="t",
            thread_key="k",
            kind="family",
            label="x",
            premise="p",
            members=[_anchor()],
            anchor_member_id="anchor",
        )
    with pytest.raises(ValidationError):  # two anchors
        HumanThread(
            thread_id="t",
            thread_key="k",
            kind="family",
            label="x",
            premise="p",
            members=[_anchor(), _anchor(member_id="kin")],
            anchor_member_id="anchor",
        )
    thread = HumanThread(
        thread_id="t",
        thread_key="k",
        kind="family",
        label="x",
        premise="p",
        members=[_anchor(), _anchor(member_id="kin", role="kin", relation_to_anchor="his mother")],
        anchor_member_id="anchor",
    )
    assert len(thread.members) == 2


def test_section_thread_ref_evidence_gate_ladder():
    # carried needs both grounding lists + a carrying member
    SectionThreadRef(
        presence="carried",
        carrying_member_id="anchor",
        thread_movement="loses his post",
        binds_to_answer_via="the decree lands on him",
        grounding_primitive_ids=["p"],
        grounding_passage_ids=["x"],
    )
    with pytest.raises(ValidationError):
        SectionThreadRef(
            presence="carried", thread_movement="m", binds_to_answer_via="b"
        )
    # peripheral needs a passage
    SectionThreadRef(
        presence="peripheral",
        thread_movement="m",
        binds_to_answer_via="b",
        grounding_passage_ids=["x"],
    )
    with pytest.raises(ValidationError):
        SectionThreadRef(presence="peripheral", thread_movement="m", binds_to_answer_via="b")
    # absent must declare a non-none fallback
    with pytest.raises(ValidationError):
        SectionThreadRef(presence="absent", thread_movement="m", binds_to_answer_via="b")
    SectionThreadRef(
        presence="absent",
        fallback_mode="structural_only",
        thread_movement="the family is offstage this section",
        binds_to_answer_via="the mechanism advances without them",
    )


# ----------------------------------------------------- canonical thread_binding


def test_thread_binding_on_section_canonical():
    section = ArchitectureSection.model_validate(
        {
            "section_id": "s1",
            "purpose": "opening",
            "approx_runtime_minutes": 3.0,
            "primitive_ids": ["pr1"],
            "section_anchor": "a courtyard",
            "section_progression": {
                "stage": "setup",
                "becomes_obvious": "the throne can enter the shrine",
                "answer_contribution": "opens the question",
                "theme_link": "state vs sanctuary",
                "what_remains_live": "who answers for it",
            },
            "thread_binding": {
                "presence": "carried",
                "carrying_member_id": "anchor",
                "thread_movement": "is seized in the sanctuary",
                "binds_to_answer_via": "the violation is felt on his body",
                "grounding_primitive_ids": ["pr1"],
                "grounding_passage_ids": ["x1"],
            },
        }
    )
    assert section.thread_binding is not None
    assert section.thread_binding.grounding_passage_ids == ["x1"]


# --------------------------------------------------------------- optional fields


def test_candidate_index_ranks_coverage_over_fame():
    # A famous, series-wide major actor with thin coverage (1 passage) vs. a situated
    # label-only person with broad coverage (3 passages). Coverage should win.
    def _sal():
        return PrimitiveSalience(score=0.5, justification="ok")

    primitives = [
        EventPrimitive(
            id="e1",
            substrate=PrimitiveSubstrate.EVENTS,
            title="Royal decree",
            core_passage_ids=["pf1"],
            actor_ids=["the_shah"],
            salience=_sal(),
            event_type="turning_point",
            what_happened="The shah issues a decree.",
        ),
    ]
    for idx in range(3):
        primitives.append(
            ActorPrimitive(
                id=f"ap{idx}",
                substrate=PrimitiveSubstrate.ACTOR_PORTRAITS,
                title=f"The striker {idx}",
                core_passage_ids=[f"ps{idx}"],
                salience=_sal(),
                actor_label="a striker at Abadan",
                goal_or_project="Hold the picket line.",
                stakes_or_fears="Arrest and lost wages.",
            )
        )
    synthesis = SynthesisPrimitivesArtifact(
        project_id="proj", primitives=primitives, quality_score=0.6, quality_notes=[]
    )
    actor_metadata = ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(
                actor_id="the_shah",
                display_name="The Shah",
                actor_type="person",
                narrative_tier="major",
                series_scope="series_wide",
            )
        ],
    )
    ranked = _build_human_thread_candidate_index(
        synthesis, actor_metadata, top_n=12
    )
    assert ranked[0]["kind"] == "situated"
    assert ranked[0]["label"] == "a striker at Abadan"
    # the thin famous actor is present but ranked below the situated carrier
    shah = next(c for c in ranked if c["actor_id"] == "the_shah")
    assert ranked.index(shah) > 0


def test_section_validates_without_thread_binding():
    section = ArchitectureSection.model_validate(
        {
            "section_id": "s1",
            "purpose": "opening",
            "approx_runtime_minutes": 3.0,
            "primitive_ids": ["pr1"],
            "section_anchor": "a courtyard",
            "section_progression": {
                "stage": "setup",
                "becomes_obvious": "x",
                "answer_contribution": "y",
                "theme_link": "z",
                "what_remains_live": "w",
            },
        }
    )
    assert section.thread_binding is None
