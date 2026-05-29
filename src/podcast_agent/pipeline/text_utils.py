"""Small text utilities shared by the lint pass, the judge payload builder,
and the oral-rewriter validation pass.

The intent here is to keep a single sentence-splitter contract that
``style_audit_linting`` and ``tic_families`` can both consume — so
char-offset reporting stays consistent between the lint flags and downstream
remediation.
"""

from __future__ import annotations

import re


# Sentence-terminal punctuation. We split on .?! followed by whitespace; we
# treat newlines as forced breaks. Em-dashes and parens stay inside the
# sentence. This matches the existing orchestrator splitter behavior for the
# purposes the lint pass needs (locating "Hold this." type sentence-fragment
# tics).
_SENTENCE_END = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z\"'])")
_NEWLINE_BLOCK = re.compile(r"\n+")


def split_sentences(text: str) -> list[tuple[str, int]]:
    """Return ``(sentence, char_start_within_text)`` pairs for ``text``.

    Sentences are stripped of leading/trailing whitespace; the ``char_start``
    is the offset of the first non-whitespace character within the original
    ``text``. Empty sentences are dropped.
    """
    if not text:
        return []
    results: list[tuple[str, int]] = []
    cursor = 0
    for block in _NEWLINE_BLOCK.split(text):
        block_start = text.find(block, cursor)
        if block_start < 0:
            block_start = cursor
        # Split on sentence-terminal punctuation followed by capital starts.
        pieces = _SENTENCE_END.split(block)
        local_cursor = 0
        for piece in pieces:
            if not piece.strip():
                local_cursor += len(piece)
                continue
            lead_ws = len(piece) - len(piece.lstrip())
            sentence = piece.strip()
            start = block_start + local_cursor + lead_ws
            results.append((sentence, start))
            local_cursor += len(piece)
            # Account for the captured whitespace separator that the regex
            # consumed (single space approximation; sufficient for char-offset
            # reporting in style audit).
            local_cursor += 1
        cursor = block_start + len(block)
    return results


def count_em_dash_runs(text: str, *, minimum_run: int = 3) -> int:
    """Count paragraphs in ``text`` that contain ``minimum_run`` or more em
    dashes. The oral rewriter must break these up; the validator counts them
    before/after to detect when the rewrite hasn't done its job.
    """
    if not text:
        return 0
    paragraphs = re.split(r"\n\s*\n", text)
    return sum(1 for p in paragraphs if p.count("—") >= minimum_run)


_NUMERIC_RUN = re.compile(r"\b\d[\d,\.]*\b")


def extract_numeric_tokens(text: str) -> list[str]:
    """Return all numeric-looking tokens (commas / decimals preserved)."""
    if not text:
        return []
    return _NUMERIC_RUN.findall(text)


_NAMED_ENTITY_RUN = re.compile(r"\b(?:[A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*)+)\b")


def extract_named_entity_runs(text: str) -> list[str]:
    """Return all multi-word Title-Case runs — a heuristic for named entities
    that the oral rewriter must preserve.
    """
    if not text:
        return []
    return _NAMED_ENTITY_RUN.findall(text)


def paragraph_reorder_score(input_text: str, output_text: str) -> float:
    """Return a score in [0, 1] estimating how much the paragraph order
    changed. 0.0 = identical paragraph sequence; 1.0 = nothing in common.

    Implemented as ``1 - LCS(input_paragraphs, output_paragraphs) /
    max(len_input, len_output)`` using normalized paragraph strings.
    """
    inp = [p.strip() for p in re.split(r"\n\s*\n", input_text or "") if p.strip()]
    out = [p.strip() for p in re.split(r"\n\s*\n", output_text or "") if p.strip()]
    if not inp or not out:
        return 0.0
    # Compute LCS length on normalized (lowercased first 80 chars) keys; this
    # tolerates light paraphrase while still rewarding identical ordering.
    keys_inp = [p.lower()[:80] for p in inp]
    keys_out = [p.lower()[:80] for p in out]
    n, m = len(keys_inp), len(keys_out)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if keys_inp[i - 1] == keys_out[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    base = max(n, m)
    return round(1 - (lcs / base), 4)
