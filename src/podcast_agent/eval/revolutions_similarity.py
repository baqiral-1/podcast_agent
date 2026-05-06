"""Benchmark-relative similarity scoring for episode scripts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

HOST_GUIDANCE_PATTERNS = (
    r"(?:^|[.!?]\s+)(?:now|okay)\b",
    r"\bremember\b",
    r"\bas we (?:saw|said|have seen|are about to see)\b",
    r"\bwhat mattered was\b",
    r"\bthe problem was\b",
    r"\bthe point is\b",
    r"\bin practice\b",
    r"\bto be clear\b",
)

TRANSLATION_PATTERNS = (
    r"\bin plain english\b",
    r"\bwhat this means is\b",
    r"\bwhich is to say\b",
    r"\bthat is to say\b",
    r"\bthe effect is\b",
    r"\bso basically\b",
    r"\bso mostly\b",
    r"\bin practice\b",
)

CALLBACK_PATTERNS = (
    r"\blast time\b",
    r"\bwe left\b",
    r"\bas we saw\b",
    r"\bas we've seen\b",
    r"\bback in \d{4}\b",
    r"\bthat same logic\b",
    r"\bcomes back\b",
    r"\breturns here\b",
)

EVAL_CONTRAST_PATTERNS = (
    r"\bnot\b[^.?!;:]{0,80}\bbut\b",
    r"\brather than\b",
    r"\binstead\b",
    r"\bwhich meant\b",
    r"\bthat left\b",
    r"\bthe trouble was\b",
    r"\bthe result was\b",
)

CONTRACTION_PATTERNS = (
    r"\b(?:i'm|we're|you're|it's|that's|there's|don't|doesn't|didn't|can't|won't|isn't|aren't|wasn't|weren't|they're|we've|you've|i've|we'll|you'll|that's)\b",
)

PRONOUN_PATTERNS = (r"\b(?:i|we|you)\b",)

META_ANNOUNCER_PATTERNS = (
    r"\bthis series\b",
    r"\bthis episode\b",
    r"\bthis hour\b",
    r"\btonight\b",
)

NOMINALIZATION_PATTERN = r"\b[a-z]+(?:tion|sion|ment|ance|ence|ity|ism|ness)\b"
LONG_WORD_PATTERN = r"\b[a-z]{10,}\b"

ASIDE_PATTERNS = (
    r"\bof course\b",
    r"\boh well\b",
    r"\bpretty much\b",
    r"\bkind of\b",
    r"\bfor like\b",
    r"\bi swear\b",
    r"\bthat tells you\b",
    r"\bwhich tells you something\b",
)

SCALES = {
    "host_guidance": 1.2,
    "translation": 1.0,
    "callback": 0.9,
    "eval_contrast": 1.1,
    "orality": 1.8,
    "meta_announcer": 0.35,
    "sentence_mean": 4.0,
    "sentence_p90": 8.0,
    "formality": 0.02,
    "aside": 0.6,
}

WEIGHTS = {
    "host_guidance": 18,
    "translation": 16,
    "callback": 14,
    "eval_contrast": 12,
    "orality": 10,
    "meta_announcer": 10,
    "sentence_rhythm": 8,
    "formality": 7,
    "aside": 5,
}


@dataclass(frozen=True)
class FeatureVector:
    word_count: int
    sentence_count: int
    mean_sentence_length: float
    p90_sentence_length: float
    host_guidance_density: float
    translation_density: float
    callback_density: float
    eval_contrast_density: float
    orality_raw: float
    meta_announcer_density: float
    formality_raw: float
    aside_density: float


@dataclass(frozen=True)
class EpisodeSimilarityScore:
    episode_number: int
    title: str
    word_count: int
    similarity_score: float
    feature_scores: dict[str, float]
    features: FeatureVector

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_number": self.episode_number,
            "title": self.title,
            "word_count": self.word_count,
            "similarity_score": self.similarity_score,
            "feature_scores": self.feature_scores,
            "features": asdict(self.features),
        }


@dataclass(frozen=True)
class RunSimilarityScore:
    run_dir: str
    episode_count: int
    mean_episode_score: float
    weighted_episode_score: float
    corpus_score: float
    benchmark_features: FeatureVector
    corpus_features: FeatureVector
    episodes: list[EpisodeSimilarityScore]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "episode_count": self.episode_count,
            "mean_episode_score": self.mean_episode_score,
            "weighted_episode_score": self.weighted_episode_score,
            "corpus_score": self.corpus_score,
            "benchmark_features": asdict(self.benchmark_features),
            "corpus_features": asdict(self.corpus_features),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def _round_score(value: float) -> float:
    return round(value, 2)


def _lower_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _sentence_lengths(text: str) -> list[int]:
    sentences = [
        segment.strip()
        for segment in SENTENCE_SPLIT_RE.split(text.strip())
        if segment.strip()
    ]
    if not sentences:
        return []
    return [len(_lower_tokens(sentence)) for sentence in sentences]


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = position - lower
    return float(lower_value + (upper_value - lower_value) * fraction)


def _count_pattern_hits(text: str, patterns: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(
        len(re.findall(pattern, lowered, flags=re.IGNORECASE)) for pattern in patterns
    )


def _per_thousand(count: int, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return count * 1000.0 / word_count


def extract_feature_vector(text: str) -> FeatureVector:
    tokens = _lower_tokens(text)
    word_count = len(tokens)
    sentence_lengths = _sentence_lengths(text)
    sentence_count = len(sentence_lengths)
    mean_sentence_length = (
        sum(sentence_lengths) / sentence_count if sentence_count else 0.0
    )
    p90_sentence_length = _percentile(sentence_lengths, 0.9)

    host_guidance_density = _per_thousand(
        _count_pattern_hits(text, HOST_GUIDANCE_PATTERNS), word_count
    )
    translation_density = _per_thousand(
        _count_pattern_hits(text, TRANSLATION_PATTERNS), word_count
    )
    callback_density = _per_thousand(
        _count_pattern_hits(text, CALLBACK_PATTERNS), word_count
    )
    eval_contrast_density = _per_thousand(
        _count_pattern_hits(text, EVAL_CONTRAST_PATTERNS), word_count
    )
    contraction_density = _per_thousand(
        _count_pattern_hits(text, CONTRACTION_PATTERNS), word_count
    )
    pronoun_density = _per_thousand(
        _count_pattern_hits(text, PRONOUN_PATTERNS), word_count
    )
    orality_raw = 0.6 * contraction_density + 0.4 * pronoun_density
    meta_announcer_density = _per_thousand(
        _count_pattern_hits(text, META_ANNOUNCER_PATTERNS), word_count
    )

    nominalization_rate = (
        _count_pattern_hits(text, (NOMINALIZATION_PATTERN,)) / word_count
        if word_count
        else 0.0
    )
    long_word_ratio = (
        _count_pattern_hits(text, (LONG_WORD_PATTERN,)) / word_count
        if word_count
        else 0.0
    )
    formality_raw = 0.6 * nominalization_rate + 0.4 * long_word_ratio
    aside_density = _per_thousand(_count_pattern_hits(text, ASIDE_PATTERNS), word_count)

    return FeatureVector(
        word_count=word_count,
        sentence_count=sentence_count,
        mean_sentence_length=mean_sentence_length,
        p90_sentence_length=p90_sentence_length,
        host_guidance_density=host_guidance_density,
        translation_density=translation_density,
        callback_density=callback_density,
        eval_contrast_density=eval_contrast_density,
        orality_raw=orality_raw,
        meta_announcer_density=meta_announcer_density,
        formality_raw=formality_raw,
        aside_density=aside_density,
    )


def closeness(value: float, benchmark_value: float, scale: float) -> float:
    return 100.0 * math.exp(-abs(value - benchmark_value) / scale)


def score_text_against_benchmark(
    text: str, benchmark: FeatureVector
) -> tuple[float, dict[str, float], FeatureVector]:
    features = extract_feature_vector(text)

    host_guidance_score = closeness(
        features.host_guidance_density,
        benchmark.host_guidance_density,
        SCALES["host_guidance"],
    )
    translation_score = closeness(
        features.translation_density,
        benchmark.translation_density,
        SCALES["translation"],
    )
    callback_score = closeness(
        features.callback_density,
        benchmark.callback_density,
        SCALES["callback"],
    )
    eval_contrast_score = closeness(
        features.eval_contrast_density,
        benchmark.eval_contrast_density,
        SCALES["eval_contrast"],
    )
    orality_score = closeness(
        features.orality_raw, benchmark.orality_raw, SCALES["orality"]
    )
    meta_announcer_score = closeness(
        features.meta_announcer_density,
        benchmark.meta_announcer_density,
        SCALES["meta_announcer"],
    )
    sentence_rhythm_score = 0.7 * closeness(
        features.mean_sentence_length,
        benchmark.mean_sentence_length,
        SCALES["sentence_mean"],
    ) + 0.3 * closeness(
        features.p90_sentence_length,
        benchmark.p90_sentence_length,
        SCALES["sentence_p90"],
    )
    formality_score = closeness(
        features.formality_raw, benchmark.formality_raw, SCALES["formality"]
    )
    aside_score = closeness(
        features.aside_density, benchmark.aside_density, SCALES["aside"]
    )

    feature_scores = {
        "host_guidance": _round_score(host_guidance_score),
        "translation": _round_score(translation_score),
        "callback": _round_score(callback_score),
        "eval_contrast": _round_score(eval_contrast_score),
        "orality": _round_score(orality_score),
        "meta_announcer": _round_score(meta_announcer_score),
        "sentence_rhythm": _round_score(sentence_rhythm_score),
        "formality": _round_score(formality_score),
        "aside": _round_score(aside_score),
    }

    weighted_sum = sum(WEIGHTS[key] * feature_scores[key] for key in WEIGHTS)
    final_score = _round_score(weighted_sum / 100.0)
    return final_score, feature_scores, features


def load_episode_body_text(
    script_path: Path, *, ignore_intro_section: bool = True
) -> tuple[int, str, str]:
    payload = json.loads(script_path.read_text(encoding="utf-8"))
    prose_sections = payload.get("prose_sections") or []
    body_sections = (
        prose_sections[1:]
        if ignore_intro_section and len(prose_sections) > 1
        else prose_sections
    )
    body_text = "\n\n".join(
        section.get("text", "").strip()
        for section in body_sections
        if section.get("text", "").strip()
    )
    episode_number = int(payload.get("episode_number") or script_path.parent.name)
    title = str(payload.get("title") or f"Episode {episode_number}")
    return episode_number, title, body_text


def iter_episode_script_paths(run_dir: Path) -> list[Path]:
    paths = sorted(
        run_dir.glob("episodes/*/episode_script.json"),
        key=lambda path: int(path.parent.name),
    )
    if not paths:
        raise FileNotFoundError(
            f"No episode_script.json artifacts found under {run_dir}"
        )
    return paths


def score_run_dir(
    run_dir: Path,
    benchmark_text: str,
    *,
    ignore_intro_section: bool = True,
) -> RunSimilarityScore:
    benchmark_features = extract_feature_vector(benchmark_text)
    episode_scores: list[EpisodeSimilarityScore] = []
    corpus_text_parts: list[str] = []
    weighted_score_total = 0.0
    weighted_word_total = 0

    for script_path in iter_episode_script_paths(run_dir):
        episode_number, title, body_text = load_episode_body_text(
            script_path,
            ignore_intro_section=ignore_intro_section,
        )
        episode_score, feature_scores, features = score_text_against_benchmark(
            body_text, benchmark_features
        )
        episode_scores.append(
            EpisodeSimilarityScore(
                episode_number=episode_number,
                title=title,
                word_count=features.word_count,
                similarity_score=episode_score,
                feature_scores=feature_scores,
                features=features,
            )
        )
        corpus_text_parts.append(body_text)
        weighted_score_total += episode_score * features.word_count
        weighted_word_total += features.word_count

    corpus_text = "\n\n".join(part for part in corpus_text_parts if part)
    corpus_score, _, corpus_features = score_text_against_benchmark(
        corpus_text, benchmark_features
    )
    mean_episode_score = _round_score(
        sum(episode.similarity_score for episode in episode_scores)
        / len(episode_scores)
    )
    weighted_episode_score = _round_score(
        weighted_score_total / weighted_word_total
        if weighted_word_total
        else mean_episode_score
    )

    return RunSimilarityScore(
        run_dir=str(run_dir),
        episode_count=len(episode_scores),
        mean_episode_score=mean_episode_score,
        weighted_episode_score=weighted_episode_score,
        corpus_score=corpus_score,
        benchmark_features=benchmark_features,
        corpus_features=corpus_features,
        episodes=episode_scores,
    )
