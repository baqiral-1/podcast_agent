"""Seed phrases and centroid computation for semantic tic detection.

The style_audit lint pass uses these families to flag narrator signature
phrases that have become house tics. The detector embeds each sentence of a
section and compares against the mean centroid of each family's seed phrases.

Families are picked from the v66 audit's surviving tic clusters. Seeds were
chosen to cover the canonical phrasing plus 2-3 paraphrases so the detector
catches "Plainly," / "Bluntly," / "Let me name plainly," as members of the
same ``in_plain_x`` family.

Threshold is a runtime config knob
(``PipelineConfig.style_tic_semantic_threshold``) defaulting to 0.78 for
``text-embedding-3-small@256d`` — the empirical sweet spot for "speaks the
same surface idea" on that model. Under the deterministic-embeddings
fallback, the threshold is essentially meaningless; the caller should
acknowledge degraded mode and combine with a small regex floor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from podcast_agent.pipeline.text_embeddings import TextEmbedder, get_text_embedder


TIC_FAMILY_SEEDS: dict[str, list[str]] = {
    "in_plain_x": [
        "In plain terms,",
        "In plain English,",
        "In plain language,",
        "Plainly,",
        "Bluntly,",
        "Let me name plainly,",
        "Said plainly:",
    ],
    "hold_sit": [
        "Hold that for a second.",
        "Hold this in your head.",
        "Sit with that.",
        "Sit with this a moment.",
        "Stay with this.",
        "Hold the line.",
    ],
    "listen_watch": [
        "Listen to what the sentence does.",
        "Listen to him.",
        "Hear this cleanly.",
        "Watch what he does first.",
        "Watch what the gambit does.",
        "Listen to what just landed.",
    ],
    "notice_the": [
        "Notice the shape.",
        "Notice what just happened.",
        "Notice the move.",
        "Notice the order.",
    ],
    "let_me": [
        "Let me name what we know.",
        "Let me place this.",
        "Let me put it this way.",
        "Let me say plainly.",
        "Let me give the answer.",
    ],
    "i_find_what_stops": [
        "What stops me about this scene is",
        "What I keep getting stuck on is",
        "I find this part hard to leave alone.",
        "I cannot walk past the phrase",
        "What I cannot let go of is",
    ],
    "picture_imagine": [
        "Picture this.",
        "Imagine for a second.",
        "Picture fifteen years of this.",
        "Imagine the room.",
    ],
    "the_thing_is": [
        "The thing is,",
        "The thing to hold in your head is",
        "Here's the thing.",
        "Here's what matters.",
    ],
    "seam_handrails": [
        "Which brings us to,",
        "The pattern is,",
        "That is to say,",
        "Put another way,",
    ],
    "now_look": [
        "Now look at,",
        "Now watch,",
        "Now consider,",
    ],
    "here_is_the": [
        "Here is the move.",
        "Here's the answer.",
        "Here is what changed.",
        "Here is the bargain.",
    ],
    "what_i_keep": [
        "What I keep coming back to is",
        "What I want to say is",
        "What I mean is",
    ],
}


@dataclass(frozen=True)
class TicHit:
    family: str
    section_id: str
    sentence: str
    char_start: int
    cosine: float


_CENTROID_CACHE: "dict[int, dict[str, np.ndarray]]" = {}
_CACHE_LOCK = threading.Lock()


def family_centroids(
    embedder: TextEmbedder | None = None,
) -> dict[str, np.ndarray]:
    """Return mean-embedding centroids per tic family. Cached per embedder dim.

    The cache key is the embedder's dimensionality (32-bit) so the OpenAI and
    deterministic-fallback paths each get their own centroids.
    """
    embedder = embedder or get_text_embedder()
    cache_key = embedder.dim
    with _CACHE_LOCK:
        cached = _CENTROID_CACHE.get(cache_key)
        if cached is not None:
            return cached
        centroids: dict[str, np.ndarray] = {}
        for family, seeds in TIC_FAMILY_SEEDS.items():
            vectors = embedder.embed(seeds)
            centroid = vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm:
                centroid = centroid / norm
            centroids[family] = centroid
        _CENTROID_CACHE[cache_key] = centroids
        return centroids


def detect_tic_hits(
    sentences: list[tuple[str, int]],
    *,
    section_id: str,
    embedder: TextEmbedder | None = None,
    threshold: float = 0.78,
) -> list[TicHit]:
    """Embed each sentence and compare against every family centroid.

    ``sentences`` is a list of ``(sentence_text, char_start_within_section)``
    pairs. Returns one ``TicHit`` per (sentence, family) pair whose cosine
    similarity meets or exceeds ``threshold``. A sentence may hit multiple
    families if both look like the same rhetorical move.
    """
    if not sentences:
        return []
    embedder = embedder or get_text_embedder()
    texts = [s for s, _ in sentences]
    sentence_vectors = embedder.embed(texts)
    centroids = family_centroids(embedder)
    if not centroids:
        return []
    family_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[f] for f in family_names])
    sims = TextEmbedder.cosine_matrix(sentence_vectors, centroid_matrix)
    hits: list[TicHit] = []
    for s_idx, (sentence, char_start) in enumerate(sentences):
        for f_idx, family in enumerate(family_names):
            score = float(sims[s_idx, f_idx])
            if score >= threshold:
                hits.append(
                    TicHit(
                        family=family,
                        section_id=section_id,
                        sentence=sentence,
                        char_start=char_start,
                        cosine=score,
                    )
                )
    return hits


def reset_centroid_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CENTROID_CACHE.clear()
