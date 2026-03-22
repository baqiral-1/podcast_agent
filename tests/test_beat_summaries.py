"""Tests for local beat summary extraction."""

from __future__ import annotations

from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import EpisodeBeat, EpisodePlan, EpisodeScript, EpisodeSegment, ScriptClaim


def _sentence(prefix: str, count: int) -> str:
    words = [f"{prefix}{index}" for index in range(count)]
    return " ".join(words) + "."


def test_extract_key_sentences_hits_target_word_count() -> None:
    text = " ".join([_sentence("alpha", 30), _sentence("bravo", 30), _sentence("charlie", 30)])
    summary = PipelineOrchestrator._extract_key_sentences(text, target_words=60)
    assert len(summary.split()) == 60


def test_build_beat_outline_uses_narration() -> None:
    beat = EpisodeBeat(
        beat_id="beat-1",
        title="Beat Title",
        chunk_ids=[],
    )
    plan = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode Title",
        chapter_ids=["chapter-1"],
        chunk_ids=[],
        themes=[],
        beats=[beat],
    )
    claim = ScriptClaim(
        claim_id="claim-1",
        text="Claim text.",
        evidence_chunk_ids=["chunk-1"],
    )
    segment = EpisodeSegment(
        segment_id="segment-1",
        beat_id="beat-1",
        heading="Heading",
        narration="NARRATIONONLY should appear in the summary.",
        claims=[claim],
        citations=["chunk-1"],
    )
    script = EpisodeScript(
        episode_id="episode-1",
        title="Episode Title",
        narrator="Narrator",
        segments=[segment],
    )
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    outline = PipelineOrchestrator._build_beat_outline(orchestrator, plan, script)
    assert "NARRATIONONLY" in outline
