"""Pure-Python lint signals for the style_audit stage.

The style_audit LLM receives these flags so it can target three narrowly
defined failures with confidence:

1. **Verbal tics**: semantically-clustered narrator signature phrases. The
   regex-only detector from earlier versions missed near-variants like
   "In plain language" / "Plainly" / "Bluntly", which all express the same
   rhetorical move as "In plain terms". The new detector embeds each
   sentence and matches it against per-family centroids (see
   ``pipeline/tic_families.py``). When a family's per-episode count reaches
   2 or more, the audit rewrites all-but-first occurrences. When a family
   crosses the series-cumulative carryover threshold, the next episode's
   writing payload picks up the family's canonical phrases as a blocklist.

2. **Frame-level over-explanation**: section openings that pre-state the
   episode's argument and section closings (non-answer-stage) that restate
   it. Detected via content-word Jaccard overlap between the first / last
   ~200 chars of each section and the spine's ``episode_answer`` +
   ``pressure_line``. Kept as keyword-overlap because it targets specific
   propositional content, not paraphrase clusters.

3. **Abstract-noun thesis drift at section frames**: occurrences of
   ``mechanism / architecture / framework / system / logic / apparatus /
   structure / paralysis / rentier / residue / contingency`` in section
   open / close windows. Kept as a literal word-list.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from podcast_agent.pipeline.text_utils import split_sentences
from podcast_agent.pipeline.tic_families import TIC_FAMILY_SEEDS, TicHit


# ---------------------------------------------------------------------------
# Abstract analytic nouns flagged at frames
# ---------------------------------------------------------------------------

_ABSTRACT_FRAME_NOUNS: tuple[str, ...] = (
    "mechanism",
    "architecture",
    "framework",
    "system",
    "logic",
    "apparatus",
    "structure",
    "paralysis",
    "rentier",
    "residue",
    "contingency",
)

_ABSTRACT_NOUN_RE = re.compile(
    r"\b(?:" + "|".join(_ABSTRACT_FRAME_NOUNS) + r")\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Frame windows
# ---------------------------------------------------------------------------

# Chars taken from the start (or end) of a section's text to compute frame
# signals. Long enough to span one paragraph; short enough not to bleed into
# the section's body.
_FRAME_CHAR_WINDOW = 220


# ---------------------------------------------------------------------------
# Thesis-overlap stopwords
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "by", "with", "from", "into", "onto", "than", "then", "as",
        "is", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "doing", "have", "has", "had", "having", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "there",
        "here", "what", "when", "where", "why", "how", "who", "whom",
        "which", "whose", "will", "would", "can", "could", "should",
        "might", "may", "must", "shall", "not", "no", "nor", "if",
        "because", "while", "about", "against", "between", "through",
        "during", "before", "after", "above", "below", "up", "down",
        "over", "under", "again", "further", "once", "very", "much",
        "most", "many", "more", "less", "such", "same", "different",
        "one", "two", "three", "first", "last", "next", "now", "still",
        "yet", "already", "also", "just", "only", "even", "both", "each",
        "every", "some", "any", "all", "none",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _content_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens = (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if not union:
        return 0.0
    return round(intersection / union, 4)


def _opening_window(text: str) -> str:
    return text[:_FRAME_CHAR_WINDOW]


def _closing_window(text: str) -> str:
    return text[-_FRAME_CHAR_WINDOW:]


def _abstract_noun_hits(text: str) -> list[str]:
    if not text:
        return []
    return sorted({m.group(0).lower() for m in _ABSTRACT_NOUN_RE.finditer(text)})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_style_audit_lint_flags(
    prose_sections: list[Any],
    *,
    spine_episode_answer: str = "",
    spine_pressure_line: str = "",
    section_progression_by_id: dict[str, str] | None = None,
    semantic_detector: Callable[[str, str], list[TicHit]],
    series_carryover_counts: dict[str, int] | None = None,
    series_carryover_threshold: int = 3,
) -> dict[str, Any]:
    """Compute per-section + episode-level lint flags for the style_audit stage.

    ``prose_sections`` is the list of section payload dicts (or models) as
    built for the style_audit prompt — each item must carry ``section_id``
    and ``text``.

    ``semantic_detector`` is a callable ``(text, section_id) -> list[TicHit]``
    that runs the embedding-based tic detector for one section's prose. The
    caller wires this through ``tic_families.detect_tic_hits`` with a shared
    ``TextEmbedder``. Passing in the detector (rather than constructing it
    inside this function) keeps the lint pass test-friendly and avoids
    pulling embeddings into pure-state codepaths.

    ``series_carryover_counts`` is the running per-family count across prior
    episodes (from ``SeriesTicState.cumulative_family_counts``). Families
    whose ``episode_count + series_count`` meet ``series_carryover_threshold``
    are surfaced separately in
    ``series_carryover_warning_families`` so the style_audit knows to rewrite
    every instance regardless of episode-local count.

    Returns the dict::

        {
          "tic_counts": {family: count_in_episode, ...},
          "tic_locations": {family: [{section_id, sentence, char_start,
                                       cosine}, ...]},
          "tic_counts_episode_plus_series": {family: int},
          "series_carryover_warning_families": [str, ...],
          "by_section": {
              section_id: {
                  "opening_thesis_overlap": float,
                  "closing_thesis_overlap": float,
                  "abstract_noun_hits_in_frames": [str, ...],
                  "tic_hits": [{family, sentence, char_start, cosine}, ...],
                  "is_answer_stage": bool,
              },
          },
        }
    """
    progression = section_progression_by_id or {}
    thesis_tokens = _content_tokens(
        f"{spine_episode_answer or ''} {spine_pressure_line or ''}"
    )

    tic_counts: dict[str, int] = {family: 0 for family in TIC_FAMILY_SEEDS}
    tic_locations: dict[str, list[dict[str, Any]]] = {
        family: [] for family in TIC_FAMILY_SEEDS
    }
    by_section: dict[str, dict[str, Any]] = {}

    for section in prose_sections or []:
        if isinstance(section, dict):
            section_id = str(section.get("section_id", "") or "")
            text = str(section.get("text", "") or "")
        else:
            section_id = str(getattr(section, "section_id", "") or "")
            text = str(getattr(section, "text", "") or "")
        if not section_id:
            continue

        section_hits = semantic_detector(text, section_id)
        for hit in section_hits:
            tic_counts[hit.family] = tic_counts.get(hit.family, 0) + 1
            tic_locations.setdefault(hit.family, []).append(
                {
                    "section_id": hit.section_id,
                    "sentence": hit.sentence,
                    "char_start": hit.char_start,
                    "cosine": hit.cosine,
                }
            )

        opening = _opening_window(text)
        closing = _closing_window(text)
        is_answer = (progression.get(section_id, "") or "").lower() == "answer"

        by_section[section_id] = {
            "opening_thesis_overlap": _jaccard(
                _content_tokens(opening), thesis_tokens
            ),
            "closing_thesis_overlap": _jaccard(
                _content_tokens(closing), thesis_tokens
            ),
            "abstract_noun_hits_in_frames": sorted(
                set(_abstract_noun_hits(opening) + _abstract_noun_hits(closing))
            ),
            "tic_hits": [
                {
                    "family": hit.family,
                    "sentence": hit.sentence,
                    "char_start": hit.char_start,
                    "cosine": hit.cosine,
                }
                for hit in section_hits
            ],
            "is_answer_stage": is_answer,
        }

    series_counts = series_carryover_counts or {}
    cumulative_counts: dict[str, int] = {
        family: tic_counts.get(family, 0) + int(series_counts.get(family, 0))
        for family in TIC_FAMILY_SEEDS
    }
    carryover_warning = sorted(
        family
        for family, total in cumulative_counts.items()
        if total >= series_carryover_threshold
    )

    return {
        "tic_counts": tic_counts,
        "tic_locations": tic_locations,
        "tic_counts_episode_plus_series": cumulative_counts,
        "series_carryover_warning_families": carryover_warning,
        "by_section": by_section,
    }


def collect_surface_phrases(lint_flags: dict[str, Any]) -> list[str]:
    """Extract the surface sentences that hit the tic detector. Used by the
    series-state recorder so later episodes can blocklist them.
    """
    locations = lint_flags.get("tic_locations") or {}
    phrases: set[str] = set()
    for hits in locations.values():
        for hit in hits:
            sentence = str(hit.get("sentence") or "").strip()
            if sentence:
                phrases.add(sentence)
    return sorted(phrases)


def aggregate_section_must_land_facts(
    scene_cards: list[Any],
) -> tuple[list[str], list[str]]:
    """De-duplicate and aggregate ``must_land_facts`` across a section's
    scene cards. Returns ``(required, strongly_preferred)`` lists preserving
    first-seen order. Lifted from the inline aggregation that
    ``_build_style_audit_sections_payload`` used to do directly so the
    payload builder and the post-audit fact-coverage diagnostic share one
    source of truth.
    """
    required: list[str] = []
    strongly_preferred: list[str] = []
    seen_required: set[str] = set()
    seen_strongly_preferred: set[str] = set()
    for scene in scene_cards:
        for fact in scene.must_land_facts.required:
            key = fact.strip()
            if not key or key in seen_required:
                continue
            seen_required.add(key)
            required.append(fact)
        for fact in scene.must_land_facts.strongly_preferred:
            key = fact.strip()
            if not key or key in seen_strongly_preferred:
                continue
            seen_strongly_preferred.add(key)
            strongly_preferred.append(fact)
    return required, strongly_preferred


def _normalize_for_substring_check(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def compute_fact_coverage_diagnostics(
    *,
    audited_script: Any,
    scene_cards_by_section_id: dict[str, list[Any]],
    original_citations_by_section_id: dict[str, list[Any]],
) -> dict[str, Any]:
    """Pure-Python verification that the audited prose still carries every
    binding ``must_land_facts.required`` and every input ``Citation.passage_id``
    per section.

    The required-fact check is a whitespace + case normalized substring
    presence test (lenient on punctuation, strict on substance). The
    citation check compares ``passage_id`` sets between the original
    pre-audit prose and the post-audit prose. Returns a JSON-serializable
    diagnostic payload; never raises on a miss — the caller decides what
    to do with the report.
    """
    section_reports: list[dict[str, Any]] = []
    episode_required_total = 0
    episode_required_landed = 0
    episode_citation_total = 0
    episode_citation_surviving = 0
    for prose_section in audited_script.prose_sections:
        section_id = prose_section.section_id
        scene_cards = scene_cards_by_section_id.get(section_id, [])
        required, _strongly_preferred = aggregate_section_must_land_facts(
            scene_cards
        )
        normalized_text = _normalize_for_substring_check(prose_section.text)
        missing_required = [
            fact
            for fact in required
            if _normalize_for_substring_check(fact) not in normalized_text
        ]
        landed_required = len(required) - len(missing_required)
        episode_required_total += len(required)
        episode_required_landed += landed_required

        original_citations = original_citations_by_section_id.get(section_id, [])
        original_passage_ids = [c.passage_id for c in original_citations]
        audited_passage_ids = {c.passage_id for c in prose_section.citations}
        missing_citation_ids = [
            passage_id
            for passage_id in original_passage_ids
            if passage_id and passage_id not in audited_passage_ids
        ]
        surviving_citations = len(original_passage_ids) - len(missing_citation_ids)
        episode_citation_total += len(original_passage_ids)
        episode_citation_surviving += surviving_citations

        section_reports.append(
            {
                "section_id": section_id,
                "required_facts_total": len(required),
                "required_facts_landed": landed_required,
                "missing_required": missing_required,
                "citations_total": len(original_passage_ids),
                "citations_surviving": surviving_citations,
                "missing_citation_passage_ids": missing_citation_ids,
            }
        )

    return {
        "episode_number": audited_script.episode_number,
        "sections": section_reports,
        "episode_total_required": episode_required_total,
        "episode_total_landed": episode_required_landed,
        "episode_total_misses": episode_required_total - episode_required_landed,
        "episode_total_citations": episode_citation_total,
        "episode_total_citations_surviving": episode_citation_surviving,
        "episode_total_citation_misses": episode_citation_total
        - episode_citation_surviving,
    }


__all__ = [
    "compute_style_audit_lint_flags",
    "collect_surface_phrases",
    "aggregate_section_must_land_facts",
    "compute_fact_coverage_diagnostics",
]
