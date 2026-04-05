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

    def test_legacy_ssml_hints_override_default_pause_and_speed(self):
        spoken = SpokenScript(
            episode_number=1,
            title="Ep 1",
            segments=[
                SpokenSegment(
                    segment_id="s1",
                    text="Test.",
                    ssml_hints={
                        "pause_before_ms": 800,
                        "pause_after_ms": 1200,
                        "speech_rate": "slower",
                    },
                )
            ],
        )
        manifest = build_render_manifest(spoken, framing=None, speed=1.0)
        assert manifest.segments[0].pause_before_ms == 800
        assert manifest.segments[0].pause_after_ms == 1200
        assert manifest.segments[0].speed == 0.94

    def test_invalid_hint_values_fall_back_to_safe_defaults(self):
        spoken = SpokenScript.model_validate(
            {
                "episode_number": 1,
                "title": "Ep 1",
                "segments": [
                    {
                        "segment_id": "s1",
                        "text": "Test.",
                        "ssml_hints": {
                            "delivery_style": "chaotic",
                            "pause_before_ms": "bad",
                            "pause_after_ms": 9999,
                            "speech_rate": "warp",
                        },
                    }
                ],
            }
        )
        manifest = build_render_manifest(spoken, framing=None, speed=1.0)
        assert manifest.segments[0].pause_before_ms == 300
        assert manifest.segments[0].pause_after_ms == 2000
        assert manifest.segments[0].speed == 1.0

    def test_openai_manifest_adds_segment_instructions_and_phrase_isolation(self):
        spoken = SpokenScript.model_validate(
            {
                "episode_number": 1,
                "title": "Ep 1",
                "segments": [
                    {
                        "segment_id": "s1",
                        "text": "The plan failed. Mountbatten forced the pace. Everyone felt it.",
                        "speech_hints": {
                            "style": "measured",
                            "intensity": "medium",
                            "pace": "normal",
                            "pause_after_ms": 600,
                            "emphasis_targets": ["Mountbatten forced the pace"],
                            "render_strategy": "isolate_phrase",
                        },
                    }
                ],
                "tts_provider": "openai",
            }
        )
        manifest = build_render_manifest(
            spoken,
            framing=None,
            base_instructions="Narrate as a clear documentary host.",
        )
        assert manifest.total_segments == 3
        assert manifest.segments[1].text == "Mountbatten forced the pace"
        assert "Segment guidance:" in manifest.segments[1].instructions
        assert "Mountbatten forced the pace" in manifest.segments[1].instructions
        assert manifest.segments[-1].pause_after_ms == 600

    def test_non_openai_manifest_logs_hint_degradation_for_prompt_only_controls(self):
        spoken = SpokenScript.model_validate(
            {
                "episode_number": 1,
                "title": "Ep 1",
                "segments": [
                    {
                        "segment_id": "s1",
                        "text": "Jinnah spoke carefully.",
                        "speech_hints": {
                            "style": "measured",
                            "pronunciation_hints": [{"text": "Jinnah", "spoken_as": "JIN-nah"}],
                            "emphasis_targets": ["spoke carefully"],
                        },
                    }
                ],
                "tts_provider": "kokoro",
            }
        )
        manifest = build_render_manifest(spoken, framing=None)
        assert manifest.segments[0].instructions is None
        assert "segment_instructions_not_supported" in manifest.segments[0].hint_degradations
        assert "pronunciation_hints_not_supported" in manifest.segments[0].hint_degradations

    def test_duration_estimation(self):
        # 130 words at 130 wpm = 60 seconds
        text = " ".join(["word"] * 130)
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[SpokenSegment(segment_id="s1", text=text)],
        )
        manifest = build_render_manifest(spoken, framing=None, words_per_minute=130)
        assert manifest.estimated_duration_seconds == 60
