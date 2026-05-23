"""Evaluation helpers."""

from .actor_explanation_audit import (
    build_actor_explanation_audit_payload,
    render_actor_explanation_audit_html,
)
from .audio_style_report import (
    DEFAULT_SCRIPT_ARTIFACTS,
    build_audio_style_payload,
    render_audio_style_html,
)
from .blind_script_comparison import (
    build_blind_comparison_payload,
    render_blind_comparison_html,
    write_blind_comparison_outputs,
)
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
    "DEFAULT_SCRIPT_ARTIFACTS",
    "build_actor_explanation_audit_payload",
    "build_audio_style_payload",
    "build_blind_comparison_payload",
    "build_review_payload",
    "extract_feature_vector",
    "extract_host_snippets",
    "load_episode_body_text",
    "render_actor_explanation_audit_html",
    "render_audio_style_html",
    "render_blind_comparison_html",
    "render_review_html",
    "score_run_dir",
    "score_text_against_benchmark",
    "write_blind_comparison_outputs",
]
