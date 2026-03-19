"""Lightweight metrics and fidelity checks for spoken-delivery rewrites."""

from __future__ import annotations

import re
from dataclasses import dataclass

from podcast_agent.schemas.models import RewriteMetrics


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_DISCOURSE_TOKENS = {
    "Additionally",
    "Also",
    "After",
    "As",
    "By",
    "Beyond",
    "Despite",
    "Even",
    "Finally",
    "First",
    "For",
    "From",
    "He",
    "His",
    "However",
    "In",
    "Instead",
    "Later",
    "Meanwhile",
    "Moreover",
    "Notably",
    "On",
    "One",
    "Second",
    "Special",
    "That",
    "The",
    "Their",
    "They",
    "Then",
    "Third",
    "This",
    "When",
    "Yet",
}
_MONTH_TOKENS = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}


@dataclass(frozen=True)
class FidelityCheckResult:
    """Practical heuristic fidelity check for spoken delivery."""

    passed: bool
    missing_names: list[str]
    missing_numbers: list[str]
    source_paragraph_count: int
    spoken_paragraph_count: int


def build_rewrite_metrics(source_text: str, spoken_text: str) -> RewriteMetrics:
    """Compute bounded rewrite metrics for one segment."""

    source_word_count = word_count(source_text)
    spoken_word_count = word_count(spoken_text)
    return RewriteMetrics(
        source_word_count=source_word_count,
        spoken_word_count=spoken_word_count,
        expansion_ratio=(spoken_word_count / source_word_count) if source_word_count else 0.0,
        source_sentence_count=sentence_count(source_text),
        spoken_sentence_count=sentence_count(spoken_text),
        source_average_sentence_length=average_sentence_length(source_text),
        spoken_average_sentence_length=average_sentence_length(spoken_text),
        source_paragraph_count=paragraph_count(source_text),
        spoken_paragraph_count=paragraph_count(spoken_text),
    )


def check_fidelity(source_text: str, spoken_text: str, *, check_paragraph_drift: bool = True) -> FidelityCheckResult:
    """Perform a practical, debuggable fidelity check."""

    source_names = extract_names(source_text)
    spoken_names = extract_names(spoken_text)
    source_numbers = extract_numbers(source_text)
    spoken_numbers = extract_numbers(spoken_text)
    missing_names = [name for name in source_names if name not in spoken_names]
    missing_numbers = [number for number in source_numbers if number not in spoken_numbers]
    source_paragraphs = paragraph_count(source_text)
    spoken_paragraphs = paragraph_count(spoken_text)
    passed = not missing_numbers and len(missing_names) <= max(1, len(source_names) // 5)
    if check_paragraph_drift:
        passed = passed and spoken_paragraphs == source_paragraphs
    return FidelityCheckResult(
        passed=passed,
        missing_names=missing_names,
        missing_numbers=missing_numbers,
        source_paragraph_count=source_paragraphs,
        spoken_paragraph_count=spoken_paragraphs,
    )


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    return len(sentences)


def average_sentence_length(text: str) -> float:
    sentences = sentence_count(text)
    return (word_count(text) / sentences) if sentences else 0.0


def paragraph_count(text: str) -> int:
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    return len(paragraphs) or 1


def extract_numbers(text: str) -> list[str]:
    return sorted(set(_NUMBER_RE.findall(text)))


def extract_names(text: str) -> list[str]:
    filtered: set[str] = set()
    for match in _NAME_RE.finditer(text):
        candidate = match.group(0).strip()
        tokens = candidate.split()
        if not tokens:
            continue
        if len(tokens) == 1:
            token = tokens[0]
            if token in _DISCOURSE_TOKENS or token in _MONTH_TOKENS:
                continue
        else:
            if tokens[0] in _DISCOURSE_TOKENS:
                continue
            if len(tokens) == 2 and tokens[1] in _MONTH_TOKENS:
                continue
        filtered.add(candidate)
    return sorted(filtered)
