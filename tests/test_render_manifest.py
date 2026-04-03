"""Unit tests for render manifest construction."""

from __future__ import annotations

from podcast_agent.pipeline.orchestrator import build_render_manifest
from podcast_agent.schemas.models import (
    EpisodeFraming,
    SpokenScript,
    SpokenSegment,
)


class TestBuildRenderManifest:
    def test_basic_manifest(self):
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[
                SpokenSegment(segment_id="s1", text="Hello listeners."),
                SpokenSegment(segment_id="s2", text="Today we explore."),
            ],
        )
        manifest = build_render_manifest(spoken, framing=None)
        assert manifest.episode_number == 1
        assert manifest.total_segments == 2
        assert manifest.estimated_duration_seconds > 0

    def test_with_framing(self):
        spoken = SpokenScript(
            episode_number=2, title="Ep 2",
            segments=[SpokenSegment(segment_id="s1", text="Main content.")],
        )
        framing = EpisodeFraming(
            episode_number=2,
            recap="Last time we covered X.",
            preview="Next time we explore Y.",
            cold_open="What if everything you knew was wrong?",
        )
        manifest = build_render_manifest(spoken, framing)
        assert manifest.total_segments == 4  # cold_open + recap + content + preview
        assert manifest.segments[0].text == "What if everything you knew was wrong?"
        assert manifest.segments[1].text == "Last time we covered X."
        assert manifest.segments[-1].text == "Next time we explore Y."

    def test_first_episode_no_recap(self):
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[SpokenSegment(segment_id="s1", text="Content.")],
        )
        framing = EpisodeFraming(
            episode_number=1, recap=None,
            preview="Next time...",
        )
        manifest = build_render_manifest(spoken, framing)
        # No recap, so: content + preview
        assert manifest.total_segments == 2

    def test_last_episode_no_preview(self):
        spoken = SpokenScript(
            episode_number=5, title="Ep 5",
            segments=[SpokenSegment(segment_id="s1", text="Final content.")],
        )
        framing = EpisodeFraming(
            episode_number=5, recap="Previously...",
            preview=None,
        )
        manifest = build_render_manifest(spoken, framing)
        # recap + content
        assert manifest.total_segments == 2

    def test_pause_insertion(self):
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[
                SpokenSegment(segment_id="s1", text="Segment one."),
                SpokenSegment(segment_id="s2", text="Segment two."),
            ],
        )
        manifest = build_render_manifest(spoken, framing=None)
        for seg in manifest.segments:
            assert seg.pause_before_ms >= 0
            assert seg.pause_after_ms >= 0

    def test_voice_and_speed(self):
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[SpokenSegment(segment_id="s1", text="Test.")],
        )
        manifest = build_render_manifest(
            spoken, framing=None, voice_id="nova", speed=1.2,
        )
        assert manifest.segments[0].voice_id == "nova"
        assert manifest.segments[0].speed == 1.2

    def test_duration_estimation(self):
        # 130 words at 130 wpm = 60 seconds
        text = " ".join(["word"] * 130)
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[SpokenSegment(segment_id="s1", text=text)],
        )
        manifest = build_render_manifest(spoken, framing=None, words_per_minute=130)
        assert manifest.estimated_duration_seconds == 60
