"""Tests for the pure-Python style_audit lint signals."""

from __future__ import annotations

import pytest

from podcast_agent.pipeline.style_audit_linting import (
    compute_style_audit_lint_flags,
)


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


class TestTicDetection:
    def test_counts_top_offenders_by_family(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "Hold that boy in your peripheral vision. "
                    "Here is the real pressure point. "
                    "Watch what happens next. "
                    "Look at what unfolds. "
                    "In plain terms the regime is broken. "
                    "I keep getting stuck on this one. "
                    "Picture an accounts ledger."
                ),
            ),
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        counts = flags["tic_counts"]
        assert counts["hold_that"] == 1
        assert counts["here_is_the"] == 1
        assert counts["watch_what"] == 1
        assert counts["look_at_what"] == 1
        assert counts["in_plain_terms"] == 1
        assert counts["i_keep_getting_stuck"] == 1
        assert counts["picture_imagine"] == 1
        assert counts["seam_handrails"] == 0

    def test_tic_locations_record_section_id(self, thesis_args):
        sections = [
            _section("s01", "Hold that thought."),
            _section("s02", "Hold the line."),
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        locations = flags["tic_locations"]["hold_that"]
        assert len(locations) == 2
        assert {loc["section_id"] for loc in locations} == {"s01", "s02"}
        assert all("char_start" in loc for loc in locations)


class TestFrameSignals:
    def test_opening_overlap_high_when_thesis_pre_stated(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "Every instrument the throne reaches for builds the "
                    "coalition against it. We open here."
                ),
            )
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        opening = flags["by_section"]["s01"]["opening_thesis_overlap"]
        assert opening >= 0.30, opening

    def test_closing_overlap_high_when_thesis_restated(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "A normal section about a different topic that does not "
                    "rehearse the spine until the very end. "
                    "Then: the throne built its own enemies through its own instruments."
                ),
            )
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        closing = flags["by_section"]["s01"]["closing_thesis_overlap"]
        assert closing >= 0.20, closing

    def test_clean_section_no_overlap(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "January 1963. A television announcement. The Shah reads "
                    "from a paper, the camera holds, and the day moves on."
                ),
            )
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        sig = flags["by_section"]["s01"]
        assert sig["opening_thesis_overlap"] < 0.20
        assert sig["closing_thesis_overlap"] < 0.20

    def test_answer_stage_flagged(self, thesis_args):
        flags = compute_style_audit_lint_flags(
            [_section("s_ans", "anything")],
            section_progression_by_id={"s_ans": "answer"},
            **thesis_args,
        )
        assert flags["by_section"]["s_ans"]["is_answer_stage"] is True


class TestAbstractNounFrames:
    def test_picks_up_abstract_nouns_in_open_or_close(self, thesis_args):
        sections = [
            _section(
                "s01",
                (
                    "The mechanism is now visible to the listener. "
                    "Three thousand more characters in the body, all concrete. "
                    "Then a closing about architecture and apparatus."
                ),
            )
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        hits = flags["by_section"]["s01"]["abstract_noun_hits_in_frames"]
        assert "mechanism" in hits
        # Closing window contains "architecture and apparatus"
        assert "architecture" in hits or "apparatus" in hits

    def test_body_only_nouns_not_flagged(self, thesis_args):
        body = "x " * 500
        sections = [
            _section(
                "s01",
                (
                    "A concrete image opens the section here. "
                    f"{body}"
                    "The mechanism explains why."
                    + " . " * 200
                    + "End on a concrete image."
                ),
            )
        ]
        flags = compute_style_audit_lint_flags(sections, **thesis_args)
        # The word "mechanism" sits in the body, not the opening/closing window.
        assert "mechanism" not in flags["by_section"]["s01"][
            "abstract_noun_hits_in_frames"
        ]
