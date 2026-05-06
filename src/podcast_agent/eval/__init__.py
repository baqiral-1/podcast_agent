"""Evaluation helpers."""

from .host_presence_review import build_review_payload, extract_host_snippets, render_review_html
from .revolutions_similarity import (
    EpisodeSimilarityScore,
    FeatureVector,
    RunSimilarityScore,
    extract_feature_vector,
    load_episode_body_text,
    score_run_dir,
    score_text_against_benchmark,
)

__all__ = [
    "EpisodeSimilarityScore",
    "FeatureVector",
    "RunSimilarityScore",
    "build_review_payload",
    "extract_feature_vector",
    "extract_host_snippets",
    "load_episode_body_text",
    "render_review_html",
    "score_run_dir",
    "score_text_against_benchmark",
]
