"""Unit tests for render manifest construction under the new spoken-script contract."""

from __future__ import annotations

from podcast_agent.pipeline.orchestrator import build_render_manifest
from podcast_agent.schemas.models import (
    FramingBlock,
    SpeechHints,
    SpokenScript,
    SpokenSection,
    SpokenSegment,
)


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
    def test_manifest_renders_sections_with_framing(self):
        spoken = SpokenScript(
            episode_number=1,
            title="Episode 1",
            framing=_framing(),
            sections=[
                SpokenSection(
                    section_id="section_1",
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The convoy moves through the dark."
                        )
                    ],
                    speech_hints=SpeechHints(pause_before_ms=350, pause_after_ms=450),
                ),
                SpokenSection(
                    section_id="section_2",
                    segments=[
                        SpokenSegment(
                            segment_id="section_2_seg1", text="The first checkpoint falls silent."
                        )
                    ],
                    speech_hints=SpeechHints(pause_before_ms=300, pause_after_ms=400),
                ),
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
            "section_1_seg1",
            "section_2_seg1",
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
                    segments=[SpokenSegment(segment_id="section_1_seg1", text="Main narration.")],
                    speech_hints=SpeechHints(pause_before_ms=500, pause_after_ms=600),
                )
            ],
        )
        manifest = build_render_manifest(spoken, voice_id="nova", speed=1.2)
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
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
            sections=[
                SpokenSection(
                    section_id="section_1",
                    segments=[SpokenSegment(segment_id="section_1_seg1", text=text)],
                )
            ],
        )
        manifest = build_render_manifest(spoken, words_per_minute=130)
        assert manifest.estimated_duration_seconds > 0

    def test_builds_manifest_from_sections_only(self):
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
                    segments=[SpokenSegment(segment_id="section_1_seg1", text="One.")],
                ),
                SpokenSection(
                    section_id="section_2",
                    segments=[SpokenSegment(segment_id="section_2_seg1", text="Two.")],
                ),
            ],
        )
        manifest = build_render_manifest(spoken)
        assert [segment.segment_id for segment in manifest.segments] == [
            "framing_opening_image",
            "framing_threat",
            "framing_question",
            "section_1_seg1",
            "section_2_seg1",
        ]

    def test_emphasis_targets_are_retained_via_segment_instructions(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The fragile truce did not hold."
                        )
                    ],
                    speech_hints=SpeechHints(
                        intensity="light",
                        emphasis_targets=["fragile truce"],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(
            spoken,
            base_instructions="Keep narration steady.",
            tts_model_name="gpt-4o-mini-tts",
        )
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.instructions is not None
        assert "fragile truce" in section.instructions
        assert section.hint_degradations == []

    def test_long_segment_instructions_are_not_truncated(self):
        long_base = " ".join(["steady narration profile"] * 80)
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The fragile truce did not hold."
                        )
                    ],
                    speech_hints=SpeechHints(
                        intensity="light",
                        emphasis_targets=["fragile truce"],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(
            spoken,
            base_instructions=long_base,
            tts_model_name="gpt-4o-mini-tts",
        )
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.instructions is not None
        assert len(section.instructions) > 500
        assert not section.instructions.endswith("...")
        assert "steady narration profile" in section.instructions
        assert "fragile truce" in section.instructions

    def test_unsupported_provider_records_hint_degradations(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The fragile truce did not hold."
                        )
                    ],
                    speech_hints=SpeechHints(
                        style="measured",
                        intensity="medium",
                        emphasis_targets=["fragile truce"],
                        pronunciation_hints=[{"text": "truce", "spoken_as": "trooss"}],
                    ),
                ),
            ],
            tts_provider="kokoro",
        )
        manifest = build_render_manifest(spoken)
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.instructions is None
        assert "segment_instructions_not_supported" in section.hint_degradations
        assert "pronunciation_hints_not_supported" in section.hint_degradations
        assert "phrase_emphasis_requires_prompt_steering" in section.hint_degradations

    def test_tts_1_hd_records_prompt_steering_degradations(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The fragile truce did not hold."
                        )
                    ],
                    speech_hints=SpeechHints(
                        style="measured",
                        intensity="medium",
                        emphasis_targets=["fragile truce"],
                        pronunciation_hints=[{"text": "truce", "spoken_as": "trooss"}],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(spoken)
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.voice_id == "fable"
        assert section.instructions is None
        assert "segment_instructions_not_supported" in section.hint_degradations
        assert "pronunciation_hints_not_supported" in section.hint_degradations
        assert "phrase_emphasis_requires_prompt_steering" in section.hint_degradations

    def test_render_strategy_split_sentences_creates_multiple_segments(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="First sentence. Second sentence."
                        )
                    ],
                    speech_hints=SpeechHints(render_strategy="split_sentences"),
                ),
            ],
        )
        manifest = build_render_manifest(spoken, tts_model_name="gpt-4o-mini-tts")
        segment_ids = [segment.segment_id for segment in manifest.segments]
        assert "section_1_seg1_1" in segment_ids
        assert "section_1_seg1_2" in segment_ids

    def test_split_sentences_filters_pronunciation_hints_per_chunk(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1",
                            text="The battle hardened at Panipat. The embassy regrouped in Shiraz.",
                        )
                    ],
                    speech_hints=SpeechHints(
                        render_strategy="split_sentences",
                        pronunciation_hints=[
                            {"text": "Panipat", "spoken_as": "PAH-nee-puht"},
                            {"text": "Shiraz", "spoken_as": "shee-RAHZ"},
                            {"text": "Delhi", "spoken_as": "DEL-hee"},
                        ],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(spoken, tts_model_name="gpt-4o-mini-tts")
        first = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1_1"
        )
        second = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1_2"
        )
        assert first.instructions is not None
        assert "Panipat as PAH-nee-puht" in first.instructions
        assert "Shiraz as shee-RAHZ" not in first.instructions
        assert second.instructions is not None
        assert "Shiraz as shee-RAHZ" in second.instructions
        assert "Panipat as PAH-nee-puht" not in second.instructions
        assert "Delhi as DEL-hee" not in first.instructions
        assert "Delhi as DEL-hee" not in second.instructions

    def test_pronunciation_matching_tolerates_diacritics_and_punctuation(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1",
                            text="From Isfahan to al-Mansur, envoys carried the warning.",
                        )
                    ],
                    speech_hints=SpeechHints(
                        pronunciation_hints=[
                            {"text": "Isfahán", "spoken_as": "ISS-fuh-hahn"},
                            {"text": "al mansur", "spoken_as": "ahl man-SOOR"},
                        ],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(spoken, tts_model_name="gpt-4o-mini-tts")
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.instructions is not None
        assert "Isfahán as ISS-fuh-hahn" in section.instructions
        assert "al mansur as ahl man-SOOR" in section.instructions

    def test_pronunciation_clause_omitted_when_no_terms_match_chunk(self):
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
                    segments=[
                        SpokenSegment(
                            segment_id="section_1_seg1", text="The envoy crossed the river at dawn."
                        )
                    ],
                    speech_hints=SpeechHints(
                        pronunciation_hints=[{"text": "Shiraz", "spoken_as": "shee-RAHZ"}],
                    ),
                ),
            ],
        )
        manifest = build_render_manifest(
            spoken,
            base_instructions="Keep narration steady.",
            tts_model_name="gpt-4o-mini-tts",
        )
        section = next(
            segment for segment in manifest.segments if segment.segment_id == "section_1_seg1"
        )
        assert section.instructions is not None
        assert "Keep narration steady." in section.instructions
        assert "Use these pronunciations:" not in section.instructions
