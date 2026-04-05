"""Unit tests for CLI argument parsing helpers."""

from __future__ import annotations

import pytest
import typer

from podcast_agent.cli.app import _parse_sub_themes


class TestParseSubThemes:
    def test_empty_is_allowed(self):
        assert _parse_sub_themes(None) == []
        assert _parse_sub_themes("") == []

    def test_trim_and_dedupe_preserves_order(self):
        result = _parse_sub_themes(" borders,displacement,borders, governance ")
        assert result == ["borders", "displacement", "governance"]

    def test_rejects_empty_entries(self):
        with pytest.raises(typer.BadParameter, match="non-empty"):
            _parse_sub_themes("valid, ,other")

    def test_rejects_more_than_eight(self):
        raw = "a1,a2,a3,a4,a5,a6,a7,a8,a9"
        with pytest.raises(typer.BadParameter, match="at most 8"):
            _parse_sub_themes(raw)
