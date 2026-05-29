"""Post-extraction verification for excerpt `verbatim_excerpt` fields.

Pure-Python, no LLM call. For each excerpt with non-empty ``verbatim_excerpt``,
fuzzy-match the verbatim text against the concatenated text of its source
``passage_ids``. Populate two new fields on the excerpt:

- ``verbatim_match_ratio`` (0.0–1.0): how strongly the verbatim is anchored in
  source passages. Clause-level substring match yields 1.0; otherwise the
  maximum 5-gram density across the excerpt's source passages.
- ``verbatim_provenance``: the subset of ``passage_ids`` that actually
  contributed verifiable text (≥ 0.20 5-gram density OR any clause match).

When ``verbatim_match_ratio < 0.30`` the verbatim text is dropped to "" — the
excerpt remains in the pool as a named-but-not-quoted record. A misquoted line
under quotation marks is worse than no quote.
"""

from __future__ import annotations

import re

from podcast_agent.schemas.models import ExcerptRecord

# 5-gram density below this drops verbatim_excerpt to empty.
_DROP_RATIO_THRESHOLD = 0.30
# Per-passage threshold for inclusion in verbatim_provenance.
_PROVENANCE_DENSITY_THRESHOLD = 0.20
# n-gram size for density scoring.
_NGRAM_N = 5
# Minimum clause length (in words) eligible for whole-clause substring match.
_MIN_CLAUSE_WORDS = 4


_CURLY_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def _normalize(text: str) -> str:
    """Lowercase, drop ellipses + punctuation, collapse whitespace.

    Designed so that "They sold us … They sold our independence." and the
    source passage's "They sold us. They sold our independence." normalize to
    the same string, so the substring/n-gram checks match the way a listener
    would hear them.
    """
    if not text:
        return ""
    s = text.translate(_CURLY_QUOTE_MAP).lower()
    s = re.sub(r"…", " ", s)  # ellipsis char
    s = re.sub(r"\.\.\.+", " ", s)  # textual ellipsis
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:…|\.\.\.+|[.!?;])\s*")


def _clauses(verbatim: str) -> list[str]:
    """Split verbatim on sentence/clause boundaries; keep clauses ≥ 4 words."""

    return [
        c.strip()
        for c in _CLAUSE_SPLIT_RE.split(verbatim)
        if len(c.strip().split()) >= _MIN_CLAUSE_WORDS
    ]


def _ngram_density(verbatim_norm: str, passage_norm: str, n: int = _NGRAM_N) -> float:
    """Fraction of ``n``-word windows from ``verbatim_norm`` appearing as a
    contiguous substring of ``passage_norm``."""

    words = verbatim_norm.split()
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    hits = sum(1 for gram in grams if gram in passage_norm)
    return hits / len(grams)


def verify_excerpt(
    excerpt: ExcerptRecord,
    passage_text_by_id: dict[str, str],
) -> ExcerptRecord:
    """Compute ``verbatim_match_ratio`` and ``verbatim_provenance``; drop
    ``verbatim_excerpt`` to empty when the match ratio is below threshold.

    Idempotent and pure: returns a new ``ExcerptRecord``, never mutates the
    input.
    """

    verbatim = (excerpt.verbatim_excerpt or "").strip()
    if not verbatim:
        return excerpt.model_copy(update={"verbatim_match_ratio": 0.0, "verbatim_provenance": []})

    verbatim_norm = _normalize(verbatim)
    clause_norms = [_normalize(c) for c in _clauses(verbatim)]
    clause_norms = [c for c in clause_norms if c]

    best_ratio = 0.0
    provenance: list[str] = []
    for passage_id in excerpt.passage_ids:
        passage_text = passage_text_by_id.get(passage_id, "")
        if not passage_text:
            continue
        passage_norm = _normalize(passage_text)
        if not passage_norm:
            continue
        clause_match = any(clause and clause in passage_norm for clause in clause_norms)
        density = _ngram_density(verbatim_norm, passage_norm)
        passage_ratio = 1.0 if clause_match else density
        if passage_ratio > best_ratio:
            best_ratio = passage_ratio
        if clause_match or density >= _PROVENANCE_DENSITY_THRESHOLD:
            provenance.append(passage_id)

    updates: dict[str, object] = {
        "verbatim_match_ratio": round(best_ratio, 4),
        "verbatim_provenance": provenance,
    }
    if best_ratio < _DROP_RATIO_THRESHOLD:
        # Misquoted line under quotation marks is worse than no quote: drop the
        # verbatim text but keep the excerpt record (it is still a named
        # source).
        updates["verbatim_excerpt"] = ""
    return excerpt.model_copy(update=updates)
