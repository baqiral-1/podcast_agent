"""Tests for excerpt sourcing verification."""

from __future__ import annotations

from podcast_agent.pipeline.excerpt_verification import verify_excerpt
from podcast_agent.schemas.models import ExcerptRecord


def _record(verbatim: str, *, passage_ids: list[str] | None = None) -> ExcerptRecord:
    return ExcerptRecord(
        title="example",
        passage_ids=passage_ids or ["p1"],
        summary="example summary",
        verbatim_excerpt=verbatim,
    )


class TestVerifyExcerpt:
    def test_exact_clause_match_preserves_verbatim(self) -> None:
        excerpt = _record("They sold us. They sold our independence.")
        passages = {
            "p1": (
                "In the autumn of 1964 the cleric said: They sold us. "
                "They sold our independence. The room held its breath."
            )
        }
        result = verify_excerpt(excerpt, passages)
        assert result.verbatim_excerpt == excerpt.verbatim_excerpt
        assert result.verbatim_match_ratio == 1.0
        assert "p1" in result.verbatim_provenance

    def test_retranslation_drift_dropped(self) -> None:
        # Source has "They sold our independence to the foreign powers"; the
        # excerpt has lightly retranslated word order ("They have sold...").
        excerpt = _record("They have sold our independence to foreign powers")
        passages = {"p1": "He told the room: They sold our independence to the foreign powers."}
        result = verify_excerpt(excerpt, passages)
        assert result.verbatim_excerpt == ""
        assert result.verbatim_match_ratio < 0.30

    def test_fully_fabricated_dropped(self) -> None:
        excerpt = _record("This sentence is entirely invented and does not appear anywhere.")
        passages = {"p1": "Some unrelated passage about a different topic."}
        result = verify_excerpt(excerpt, passages)
        assert result.verbatim_excerpt == ""
        assert result.verbatim_match_ratio == 0.0
        assert result.verbatim_provenance == []

    def test_empty_verbatim_passes_through(self) -> None:
        excerpt = _record("")
        passages = {"p1": "Any text."}
        result = verify_excerpt(excerpt, passages)
        assert result.verbatim_excerpt == ""
        assert result.verbatim_match_ratio == 0.0
        assert result.verbatim_provenance == []

    def test_ellipsis_compression_tolerated(self) -> None:
        excerpt = _record(
            "They sold our independence to the foreign powers … and that moment was deliberate."
        )
        passages = {
            "p1": (
                "He said: They sold our independence to the foreign powers. "
                "Pause. Then: and that moment was deliberate. Pause again."
            )
        }
        result = verify_excerpt(excerpt, passages)
        # At least one clause matched contiguously after normalization.
        assert result.verbatim_excerpt == excerpt.verbatim_excerpt
        assert result.verbatim_match_ratio >= 0.30

    def test_curly_quotes_normalized(self) -> None:
        excerpt = _record("‘They sold our independence to the foreign powers’")
        passages = {
            "p1": "He told them: 'They sold our independence to the foreign powers' that day."
        }
        result = verify_excerpt(excerpt, passages)
        assert result.verbatim_excerpt == excerpt.verbatim_excerpt
        assert result.verbatim_match_ratio >= 0.30

    def test_provenance_includes_passage_with_match(self) -> None:
        excerpt = _record(
            "They sold our independence to the foreign powers",
            passage_ids=["p1", "p2"],
        )
        passages = {
            "p1": "An unrelated passage.",
            "p2": "He said: They sold our independence to the foreign powers.",
        }
        result = verify_excerpt(excerpt, passages)
        assert "p2" in result.verbatim_provenance
        assert "p1" not in result.verbatim_provenance
