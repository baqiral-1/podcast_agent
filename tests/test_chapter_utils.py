"""Tests for chapter utilities."""

from __future__ import annotations

import pytest

from podcast_agent.utils.chapter_utils import SectionInput, validate_chapter_range


def test_validate_chapter_range_matches_exact_titles() -> None:
    sections = [
        SectionInput(title="Prologue", body="..."),
        SectionInput(title="Chapter 1: Start", body="..."),
        SectionInput(title="Chapter 2: Middle", body="..."),
    ]
    start_index, end_index = validate_chapter_range(
        sections,
        start_chapter="Chapter 1: Start",
        end_chapter="Chapter 2: Middle",
    )
    assert (start_index, end_index) == (1, 2)


def test_validate_chapter_range_raises_on_missing_title() -> None:
    sections = [
        SectionInput(title="Prologue", body="..."),
        SectionInput(title="Chapter 1: Start", body="..."),
    ]
    with pytest.raises(ValueError, match="Unable to find start chapter"):
        validate_chapter_range(sections, start_chapter="Chapter 9: Missing")
