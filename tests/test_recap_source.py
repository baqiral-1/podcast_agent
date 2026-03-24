"""Tests for recap source construction from prior episode beats."""

from __future__ import annotations

from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import (
    EpisodeBeat,
    EpisodePlan,
    EpisodeScript,
    EpisodeSegment,
    ScriptClaim,
)


def _make_segment(beat_id: str, narration: str) -> EpisodeSegment:
    return EpisodeSegment(
        segment_id=f"{beat_id}-seg-1",
        beat_id=beat_id,
        heading="Heading",
        narration=narration,
        claims=[ScriptClaim(claim_id=f"{beat_id}-claim", text="claim", evidence_chunk_ids=["c1"])],
        citations=[],
    )


def test_build_recap_source_from_previous_beats() -> None:
    plan = EpisodePlan(
        episode_id="episode-1",
        sequence=1,
        title="Episode 1",
        chapter_ids=["chapter-1"],
        beats=[
            EpisodeBeat(beat_id="beat-1", title="Beat One"),
            EpisodeBeat(beat_id="beat-2", title="Beat Two"),
        ],
    )
    script = EpisodeScript(
        episode_id="episode-1",
        title="Episode 1",
        narrator="Narrator",
        segments=[
            _make_segment(
                "beat-1",
                "This is a long single sentence about beat one that contains enough words to exceed the target length easily.",
            ),
            _make_segment(
                "beat-2",
                "This is another long single sentence for beat two that also contains enough words to exceed the target length easily.",
            ),
        ],
    )

    recap_source = PipelineOrchestrator._build_recap_source_from_previous(
        plan,
        script,
        target_words=24,
        max_words=900,
    )

    lines = [line.strip() for line in recap_source.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("1. ")
    assert lines[1].startswith("2. ")
    assert "Beat One" in lines[0]
    assert "Beat Two" in lines[1]
    assert len(recap_source.split()) <= 900
