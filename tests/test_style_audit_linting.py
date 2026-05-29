"""Tests for the pure-Python style_audit lint signals.

Change 2 replaced the regex tic detector with a semantic-detector callable
that the orchestrator wires through ``tic_families.detect_tic_hits`` with a
shared ``TextEmbedder``. These tests inject a fake semantic detector so
we exercise the lint-pass shape contract without depending on real
embeddings.
"""

from __future__ import annotations

import pytest

from podcast_agent.pipeline.style_audit_linting import (
    aggregate_section_must_land_facts,
    collect_surface_phrases,
    compute_fact_coverage_diagnostics,
    compute_style_audit_lint_flags,
)
from podcast_agent.pipeline.tic_families import TicHit


@pytest.fixture
def thesis_args() -> dict[str, str]:
    return {
        "spine_episode_answer": (
            "The throne built its own enemies through its own instruments."
        ),
        "spine_pressure_line": (
            "Every instrument the throne reaches for builds the coalition against it."
        ),
    }


def _section(section_id: str, text: str) -> dict[str, str]:
    return {"section_id": section_id, "text": text}


def _empty_detector(_text: str, _section_id: str) -> list[TicHit]:
    return []


class TestTicDetection:
    def test_semantic_detector_hits_are_aggregated_by_family(self, thesis_args):
        sections = [
            _section("s01", "Hold that boy in your peripheral vision."),
            _section("s02", "Plainly: the regime is broken."),
        ]

        def detector(text: str, section_id: str) -> list[TicHit]:
            if section_id == "s01":
                return [
                    TicHit(
                        family="hold_sit",
                        section_id="s01",
                        sentence="Hold that boy in your peripheral vision.",
                        char_start=0,
                        cosine=0.83,
                    )
                ]
            if section_id == "s02":
                return [
                    TicHit(
                        family="in_plain_x",
                        section_id="s02",
                        sentence="Plainly: the regime is broken.",
                        char_start=0,
                        cosine=0.81,
                    )
                ]
            return []

        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=detector
        )
        counts = flags["tic_counts"]
        assert counts["hold_sit"] == 1
        assert counts["in_plain_x"] == 1
        # Untouched families default to zero.
        assert counts["picture_imagine"] == 0
        # Per-section breakdown carries the same hits.
        assert flags["by_section"]["s01"]["tic_hits"][0]["family"] == "hold_sit"
        assert flags["by_section"]["s02"]["tic_hits"][0]["family"] == "in_plain_x"

    def test_tic_locations_record_section_id(self, thesis_args):
        sections = [
            _section("s01", "Hold that thought."),
            _section("s02", "Hold the line."),
        ]

        def detector(text: str, section_id: str) -> list[TicHit]:
            return [
                TicHit(
                    family="hold_sit",
                    section_id=section_id,
                    sentence=text.strip(),
                    char_start=0,
                    cosine=0.8,
                )
            ]

        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=detector
        )
        locations = flags["tic_locations"]["hold_sit"]
        assert len(locations) == 2
        assert {loc["section_id"] for loc in locations} == {"s01", "s02"}
        assert all("char_start" in loc for loc in locations)
        assert all("cosine" in loc for loc in locations)

    def test_series_carryover_warning_families_are_surfaced(self, thesis_args):
        sections = [_section("s01", "Hold that boy.")]

        def detector(_text: str, section_id: str) -> list[TicHit]:
            return [
                TicHit(
                    family="hold_sit",
                    section_id=section_id,
                    sentence="Hold that boy.",
                    char_start=0,
                    cosine=0.82,
                )
            ]

        flags = compute_style_audit_lint_flags(
            sections,
            **thesis_args,
            semantic_detector=detector,
            series_carryover_counts={"hold_sit": 2},
            series_carryover_threshold=3,
        )
        # 1 (episode) + 2 (series) = 3, hits the threshold.
        assert "hold_sit" in flags["series_carryover_warning_families"]
        assert flags["tic_counts_episode_plus_series"]["hold_sit"] == 3

    def test_collect_surface_phrases_returns_unique_sentences(self):
        lint_flags = {
            "tic_locations": {
                "hold_sit": [
                    {"sentence": "Hold that thought.", "section_id": "s01"},
                    {"sentence": "Hold that thought.", "section_id": "s02"},
                ],
                "in_plain_x": [
                    {"sentence": "Plainly: the regime is broken.", "section_id": "s02"},
                ],
            }
        }
        phrases = collect_surface_phrases(lint_flags)
        assert phrases == ["Hold that thought.", "Plainly: the regime is broken."]


class TestFrameSignals:
    def test_opening_overlap_high_when_thesis_pre_stated(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "The throne built its own enemies through its own instruments. "
                    "Every instrument the throne reaches for builds a coalition. "
                    "The chapter begins in a quiet bazaar."
                ),
            ),
        ]
        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=_empty_detector
        )
        assert flags["by_section"]["s01"]["opening_thesis_overlap"] >= 0.30

    def test_closing_overlap_high_when_thesis_restated(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "Many small details fill the opening. "
                    "Mid-section the host narrates events plainly. "
                    "The throne built its own enemies through its own instruments. "
                    "Every instrument the throne reaches for builds the coalition against it."
                ),
            ),
        ]
        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=_empty_detector
        )
        assert flags["by_section"]["s01"]["closing_thesis_overlap"] >= 0.30

    def test_clean_section_no_overlap(self, thesis_args):
        sections = [
            _section(
                "s01",
                "A pilgrim slides a cassette tape into a Volkswagen camper at dusk.",
            ),
        ]
        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=_empty_detector
        )
        assert flags["by_section"]["s01"]["opening_thesis_overlap"] < 0.20
        assert flags["by_section"]["s01"]["closing_thesis_overlap"] < 0.20

    def test_answer_stage_flagged(self, thesis_args):
        sections = [_section("s01", "An ordinary middle section.")]
        flags = compute_style_audit_lint_flags(
            sections,
            **thesis_args,
            section_progression_by_id={"s01": "answer"},
            semantic_detector=_empty_detector,
        )
        assert flags["by_section"]["s01"]["is_answer_stage"] is True


class TestAbstractNounFrames:
    def test_picks_up_abstract_nouns_in_open_or_close(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "A repressive apparatus opens this section visibly. "
                    "Many small concrete sentences follow. "
                    "The whole closes on an emergent structure of dependencies."
                ),
            ),
        ]
        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=_empty_detector
        )
        hits = flags["by_section"]["s01"]["abstract_noun_hits_in_frames"]
        assert "apparatus" in hits
        assert "structure" in hits

    def test_body_only_nouns_not_flagged(self, thesis_args):
        # Construct a long-enough section so the frame windows (first/last
        # ~220 chars) do not contain the body-only abstract noun.
        opening = (
            "A pilgrim slides a cassette tape into a Volkswagen camper. "
            "Pilgrims have crossed this border for as long as anyone can remember. "
            "The cassette in his hand is wrapped in plain paper, no markings, "
            "stamped only with a hand-written sura number on the back."
        )
        body = (
            " The mechanism of distribution carries doctrine across the border, "
            "from pilgrim to pilgrim, from shrine to shrine, with no central registry. "
        )
        closing = (
            "By the time the camper reaches the customs post, the cassette has "
            "already been copied, recopied, and tucked into other panels. "
            "The doors of the shrine close behind the lecture without ceremony."
        )
        sections = [_section("s01", opening + body + closing)]
        flags = compute_style_audit_lint_flags(
            sections, **thesis_args, semantic_detector=_empty_detector
        )
        hits = flags["by_section"]["s01"]["abstract_noun_hits_in_frames"]
        # 'mechanism' lives in the middle body, not the first/last 220 chars.
        assert "mechanism" not in hits


# ---------------------------------------------------------------------------
# Fact-coverage diagnostics (Change A+ — post-audit verification)
# ---------------------------------------------------------------------------


class _FakeMustLandFacts:
    def __init__(self, required, strongly_preferred=()):
        self.required = list(required)
        self.strongly_preferred = list(strongly_preferred)


class _FakeSceneCard:
    def __init__(self, section_id, required, strongly_preferred=()):
        self.section_id = section_id
        self.must_land_facts = _FakeMustLandFacts(required, strongly_preferred)


class _FakeCitation:
    def __init__(self, passage_id):
        self.passage_id = passage_id


class _FakeProseSection:
    def __init__(self, section_id, text, citations=()):
        self.section_id = section_id
        self.text = text
        self.citations = list(citations)


class _FakeAuditedScript:
    def __init__(self, episode_number, prose_sections):
        self.episode_number = episode_number
        self.prose_sections = list(prose_sections)


class TestAggregateMustLandFacts:
    def test_deduplicates_across_scene_cards(self):
        cards = [
            _FakeSceneCard("s1", ["The order is real.", "Tehran, March 1979"]),
            _FakeSceneCard("s1", ["The order is real.", "Khomeini speaks"]),
        ]
        required, strongly_preferred = aggregate_section_must_land_facts(cards)
        assert required == [
            "The order is real.",
            "Tehran, March 1979",
            "Khomeini speaks",
        ]
        assert strongly_preferred == []

    def test_preserves_strongly_preferred(self):
        cards = [
            _FakeSceneCard(
                "s1",
                required=["A required fact"],
                strongly_preferred=["A preferred fact"],
            ),
        ]
        required, preferred = aggregate_section_must_land_facts(cards)
        assert required == ["A required fact"]
        assert preferred == ["A preferred fact"]


class TestFactCoverageDiagnostics:
    def test_signs_off_when_every_required_fact_landed(self):
        prose = _FakeProseSection(
            "s1",
            text=(
                "Tehran, March 1979. Bazargan went to Qom with a draft ballot. "
                "The order is real."
            ),
            citations=[_FakeCitation("p1"), _FakeCitation("p2")],
        )
        audited = _FakeAuditedScript(1, [prose])
        scene_cards_by_section_id = {
            "s1": [_FakeSceneCard("s1", ["The order is real.", "Tehran, March 1979"])],
        }
        original_citations_by_section_id = {
            "s1": [_FakeCitation("p1"), _FakeCitation("p2")],
        }
        report = compute_fact_coverage_diagnostics(
            audited_script=audited,
            scene_cards_by_section_id=scene_cards_by_section_id,
            original_citations_by_section_id=original_citations_by_section_id,
        )
        assert report["episode_total_misses"] == 0
        assert report["episode_total_citation_misses"] == 0
        assert report["sections"][0]["missing_required"] == []
        assert report["sections"][0]["missing_citation_passage_ids"] == []

    def test_flags_paraphrased_required_fact_as_missing(self):
        # The audit paraphrased the required fact past the substring threshold.
        prose = _FakeProseSection(
            "s1",
            text="A directive arrives from the leader's chambers.",
            citations=[_FakeCitation("p1")],
        )
        audited = _FakeAuditedScript(1, [prose])
        scene_cards_by_section_id = {
            "s1": [_FakeSceneCard("s1", ["The order is real."])],
        }
        original_citations_by_section_id = {"s1": [_FakeCitation("p1")]}
        report = compute_fact_coverage_diagnostics(
            audited_script=audited,
            scene_cards_by_section_id=scene_cards_by_section_id,
            original_citations_by_section_id=original_citations_by_section_id,
        )
        assert report["episode_total_misses"] == 1
        assert report["sections"][0]["missing_required"] == ["The order is real."]
        # Citation survived; only the required fact was lost.
        assert report["episode_total_citation_misses"] == 0

    def test_flags_dropped_citation(self):
        prose = _FakeProseSection(
            "s1",
            text="Tehran, March 1979.",
            citations=[_FakeCitation("p1")],
        )
        audited = _FakeAuditedScript(1, [prose])
        scene_cards_by_section_id = {
            "s1": [_FakeSceneCard("s1", ["Tehran, March 1979"])],
        }
        # Originally had two citations; audited prose carries only one.
        original_citations_by_section_id = {
            "s1": [_FakeCitation("p1"), _FakeCitation("p2")],
        }
        report = compute_fact_coverage_diagnostics(
            audited_script=audited,
            scene_cards_by_section_id=scene_cards_by_section_id,
            original_citations_by_section_id=original_citations_by_section_id,
        )
        assert report["episode_total_citation_misses"] == 1
        assert report["sections"][0]["missing_citation_passage_ids"] == ["p2"]

    def test_normalizes_whitespace_and_case(self):
        # Audited prose differs only in case + whitespace.
        prose = _FakeProseSection(
            "s1",
            text="tehran,    March 1979",
            citations=[],
        )
        audited = _FakeAuditedScript(1, [prose])
        scene_cards_by_section_id = {
            "s1": [_FakeSceneCard("s1", ["TEHRAN, MARCH 1979"])],
        }
        original_citations_by_section_id = {"s1": []}
        report = compute_fact_coverage_diagnostics(
            audited_script=audited,
            scene_cards_by_section_id=scene_cards_by_section_id,
            original_citations_by_section_id=original_citations_by_section_id,
        )
        assert report["episode_total_misses"] == 0
