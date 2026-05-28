"""Pure-Python lint signals for the style_audit stage.

The style_audit LLM gets these flags as input so it can target three
narrowly-defined failures with confidence:

1. **Verbal tics**: repeated narrator signature phrases ("Hold that…",
   "Watch what…", "I keep getting stuck…", "Here is/here's the…",
   "Look at what…", "Notice the…", "In plain terms", "Picture/Imagine",
   …). When a family count >= 2 in an episode, all-but-first occurrences
   should be rewritten to vary the surface form.

2. **Frame-level over-explanation**: section openings that pre-state the
   episode's argument, and section closings (non-answer-stage) that
   restate it. Detected via content-word Jaccard overlap between the
   first / last ~200 chars of each section and the spine's
   ``episode_answer`` + ``pressure_line``.

3. **Abstract-noun thesis drift at section frames**: occurrences of
   ``mechanism / architecture / framework / system / logic / apparatus /
   structure / paralysis / rentier / residue / contingency`` in section
   open / close windows.

No LLM call. The detector runs before style_audit and the flags ride along
in the input payload.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Tic families
# ---------------------------------------------------------------------------

_TIC_FAMILIES: dict[str, re.Pattern[str]] = {
    "hold_that": re.compile(r"\bhold (?:that|it|the)\b", re.IGNORECASE),
    "watch_what": re.compile(
        r"\bwatch (?:what|this|how|the)\b", re.IGNORECASE
    ),
    "look_at_what": re.compile(r"\blook at (?:what|the)\b", re.IGNORECASE),
    "here_is_the": re.compile(
        r"\bhere(?:'s| is) (?:the|what|why)\b", re.IGNORECASE
    ),
    "i_keep_getting_stuck": re.compile(
        r"\bI keep (?:getting stuck|coming back|returning)\b", re.IGNORECASE
    ),
    "notice_the": re.compile(r"\bnotice (?:the|that|how)\b", re.IGNORECASE),
    "in_plain_terms": re.compile(
        r"\bin plain (?:terms|english|bazaari|words)\b", re.IGNORECASE
    ),
    "picture_imagine": re.compile(r"\b(?:Picture|Imagine)\b"),
    "what_i_keep": re.compile(
        r"\bwhat I (?:keep|want|mean|need|am saying)\b", re.IGNORECASE
    ),
    "now_look_at": re.compile(r"\bnow (?:look at|watch)\b", re.IGNORECASE),
    "the_thing_is": re.compile(
        r"\bthe thing (?:is|to|about)\b", re.IGNORECASE
    ),
    # Seam handrails — connective tissue that announces a transition rather
    # than executing it.
    "seam_handrails": re.compile(
        r"\b(?:which brings us to|the pattern is|that is to say)\b",
        re.IGNORECASE,
    ),
}


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
) -> dict[str, Any]:
    """Compute per-section + episode-level lint flags for the style_audit stage.

    ``prose_sections`` is the list of section payload dicts as built for the
    style_audit prompt — each item must carry ``section_id`` and ``text``.

    Returns a dict with the shape:
    ::

        {
          "tic_counts": {family: total_count_in_episode, ...},
          "tic_locations": {family: [{"section_id": ..., "match": "...",
                                      "char_start": int}, ...], ...},
          "by_section": {
              section_id: {
                  "opening_thesis_overlap": float,
                  "closing_thesis_overlap": float,
                  "abstract_noun_hits_in_frames": [str, ...],
                  "is_answer_stage": bool,
              }, ...
          },
        }
    """

    progression = section_progression_by_id or {}
    thesis_tokens = _content_tokens(
        f"{spine_episode_answer or ''} {spine_pressure_line or ''}"
    )

    tic_counts: dict[str, int] = {family: 0 for family in _TIC_FAMILIES}
    tic_locations: dict[str, list[dict[str, Any]]] = {family: [] for family in _TIC_FAMILIES}
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

        for family, pattern in _TIC_FAMILIES.items():
            for match in pattern.finditer(text):
                tic_counts[family] += 1
                tic_locations[family].append(
                    {
                        "section_id": section_id,
                        "match": match.group(0),
                        "char_start": match.start(),
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
            "is_answer_stage": is_answer,
        }

    return {
        "tic_counts": tic_counts,
        "tic_locations": tic_locations,
        "by_section": by_section,
    }
