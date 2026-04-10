"""Unit tests for render manifest construction under the new spoken-script contract."""

from __future__ import annotations

from podcast_agent.pipeline.orchestrator import build_render_manifest
from podcast_agent.schemas.models import FramingBlock, SpeechHints, SpokenScript, SpokenSection, SpokenTransition


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="A convoy leaves before dawn.",
        threat_or_unresolved_action="Nobody knows whether the order can hold.",
        opening_question="What breaks first?",
        handoff_scene_card_id="scene_1",
        recap="Previously, the coalition assembled.",
        preview="Next, the consequences spread outward.",
    )


class TestBuildRenderManifest:
    def test_manifest_uses_framing_sections_transitions_and_preview(self):
        spoken = SpokenScript(
            episode_number=1,
            title="Episode 1",
            framing=_framing(),
            sections=[
                SpokenSection(
                    section_id="section_1",
                    text="The convoy moves through the dark.",
                    speech_hints=SpeechHints(pause_before_ms=350, pause_after_ms=450),
                )
            ],
            transitions=[
                SpokenTransition(
                    transition_id="transition_1",
                    text="Then the order reaches the city.",
                    speech_hints=SpeechHints(pause_before_ms=200, pause_after_ms=300),
                )
            ],
        )
        manifest = build_render_manifest(spoken)
        assert manifest.episode_number == 1
        assert manifest.total_segments == 7
        assert [segment.segment_id for segment in manifest.segments] == [
            "framing_recap",
            "framing_opening_image",
            "framing_threat",
            "framing_question",
            "framing_preview",
            "section_1",
            "transition_1",
        ]

    def test_voice_speed_and_pause_values_propagate(self):
        spoken = SpokenScript(
            episode_number=1,
            title="Episode 1",
            framing=FramingBlock(
                opening_image="Image",
                threat_or_unresolved_action="Threat",
                opening_question="Question",
                handoff_scene_card_id="scene_1",
            ),
            sections=[
                SpokenSection(
                    section_id="section_1",
                    text="Main narration.",
                    speech_hints=SpeechHints(pause_before_ms=500, pause_after_ms=600),
                )
            ],
        )
        manifest = build_render_manifest(spoken, voice_id="nova", speed=1.2)
        section = next(segment for segment in manifest.segments if segment.segment_id == "section_1")
        assert section.voice_id == "nova"
        assert section.speed == 1.2
        assert section.pause_before_ms == 500
        assert section.pause_after_ms == 600

    def test_duration_estimation_uses_word_count(self):
        text = " ".join(["word"] * 130)
        spoken = SpokenScript(
            episode_number=1,
            title="Episode 1",
            framing=FramingBlock(
                opening_image="Image",
                threat_or_unresolved_action="Threat",
                opening_question="Question",
                handoff_scene_card_id="scene_1",
            ),
            sections=[SpokenSection(section_id="section_1", text=text)],
        )
        manifest = build_render_manifest(spoken, words_per_minute=130)
        assert manifest.estimated_duration_seconds > 0
