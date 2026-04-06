"""Unit tests for data model serialization and validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from podcast_agent.schemas.models import (
    AudioManifest,
    AudioSegmentResult,
    BookRecord,
    ChapterAnalysis,
    ChapterInfo,
    ChunkingConfig,
    Citation,
    ClaimAssessment,
    CoverageStats,
    CrossBookClaimAssessment,
    CrossReference,
    EpisodeArcDetail,
    EpisodeAssignment,
    EpisodeBeat,
    EpisodeFraming,
    EpisodeMergedNarrativeRef,
    EpisodePlan,
    EpisodeScript,
    EpisodeSynthesisContext,
    EpisodeSynthesisTension,
    ExtractedPassage,
    FairnessFlag,
    GroundingReport,
    InsightType,
    NarrativeStrategy,
    NarrativeThread,
    PassagePair,
    PipelineConfig,
    ProjectStatus,
    RenderManifest,
    RenderSegment,
    RepairResult,
    ScriptSegment,
    SegmentDiff,
    SpokenScript,
    SpokenSegment,
    SynthesisInsight,
    SynthesisMap,
    SynthesisTag,
    TextChunk,
    ThematicAxis,
    ThematicCorpus,
    ThematicProject,
)


# ---------------------------------------------------------------------------
# Serialization roundtrip tests
# ---------------------------------------------------------------------------


class TestBookRecord:
    def test_roundtrip(self):
        book = BookRecord(
            book_id="b1", title="Test Book", author="Author A",
            source_path="/tmp/test.txt", source_type="txt", total_words=5000,
        )
        data = json.loads(book.model_dump_json())
        restored = BookRecord.model_validate(data)
        assert restored.book_id == "b1"
        assert restored.title == "Test Book"

    def test_with_chapters(self):
        ch = ChapterInfo(
            chapter_id="ch1", title="Chapter 1",
            start_index=0, end_index=1000, word_count=200,
            summary="First chapter.",
            analysis=ChapterAnalysis(
                themes_touched=["partition"],
                major_tensions=["deadline vs legitimacy"],
                causal_shifts=["announcement accelerates unrest"],
                narrative_hooks=["A decision made in private reshapes millions of lives."],
                retrieval_keywords=["partition", "deadline"],
            ),
        )
        book = BookRecord(
            book_id="b1", title="Test", author="A",
            source_path="/test.txt", source_type="txt",
            chapters=[ch], total_words=200,
        )
        data = json.loads(book.model_dump_json())
        restored = BookRecord.model_validate(data)
        assert len(restored.chapters) == 1
        assert restored.chapters[0].chapter_id == "ch1"
        assert restored.chapters[0].analysis is not None
        assert restored.chapters[0].analysis.themes_touched == ["partition"]

    def test_chapter_info_accepts_missing_analysis_for_backward_compatibility(self):
        chapter = ChapterInfo(
            chapter_id="ch1",
            title="Chapter 1",
            start_index=0,
            end_index=100,
            word_count=50,
            summary="Legacy summary.",
        )
        assert chapter.analysis is None


class TestChapterAnalysis:
    def test_key_events_or_arguments_allows_more_than_six_items(self):
        analysis = ChapterAnalysis(
            key_events_or_arguments=[f"event-{idx}" for idx in range(7)]
        )
        assert len(analysis.key_events_or_arguments) == 7


class TestTextChunk:
    def test_roundtrip(self):
        chunk = TextChunk(
            chunk_id="c1", book_id="b1", chapter_id="ch1",
            text="Some text here", word_count=3, position=0,
        )
        data = json.loads(chunk.model_dump_json())
        restored = TextChunk.model_validate(data)
        assert restored.chunk_id == "c1"

    def test_with_metadata(self):
        chunk = TextChunk(
            chunk_id="c1", book_id="b1", chapter_id="ch1",
            text="text", word_count=1, position=0,
            metadata={"author": "Author A", "title": "Book A"},
        )
        assert chunk.metadata["author"] == "Author A"


class TestThematicAxis:
    def test_roundtrip(self):
        axis = ThematicAxis(
            axis_id="ax1", name="Decision Making",
            description="How decisions are made under pressure.",
            guiding_questions=["What drives decisions?"],
            relevance_by_book={"b1": 0.8, "b2": 0.6},
            keywords=["decision", "pressure"],
        )
        data = json.loads(axis.model_dump_json())
        restored = ThematicAxis.model_validate(data)
        assert restored.relevance_by_book["b1"] == 0.8

    def test_multi_book_relevance(self):
        axis = ThematicAxis(
            axis_id="ax1", name="Test",
            description="Test axis",
            relevance_by_book={"b1": 0.9, "b2": 0.7, "b3": 0.1},
        )
        books_above_threshold = sum(
            1 for s in axis.relevance_by_book.values() if s >= 0.3
        )
        assert books_above_threshold == 2


class TestSynthesisInsight:
    def test_requires_min_two_passages(self):
        with pytest.raises(ValidationError):
            SynthesisInsight(
                insight_id="i1", insight_type=InsightType.SYNCHRONICITY,
                title="Test", description="Test",
                passage_ids=["p1"],  # Only 1, need at least 2
            )

    def test_valid_insight(self):
        insight = SynthesisInsight(
            insight_id="i1", insight_type=InsightType.PRODUCTIVE_FRICTION,
            title="Authors disagree", description="A vs B",
            passage_ids=["p1", "p2"],
            podcast_potential=0.9, treatment="debate",
        )
        assert insight.treatment == "debate"


class TestSynthesisMap:
    def test_quality_score_range(self):
        sm = SynthesisMap(project_id="proj1", quality_score=0.75)
        assert 0 <= sm.quality_score <= 1

    def test_roundtrip(self):
        sm = SynthesisMap(
            project_id="proj1",
            insights=[
                SynthesisInsight(
                    insight_type=InsightType.LATENT_PATTERN,
                    title="Test tension", description="Tension desc",
                    passage_ids=["p1", "p2"],
                )
            ],
            narrative_threads=[
                NarrativeThread(
                    title="Main thread",
                    description="The main narrative",
                    insight_ids=["i1"],
                    arc_type="convergence",
                )
            ],
            quality_score=0.65,
        )
        data = json.loads(sm.model_dump_json())
        restored = SynthesisMap.model_validate(data)
        assert len(restored.insights) == 1
        assert restored.quality_score == 0.65


class TestEpisodePlan:
    def test_roundtrip(self):
        beat = EpisodeBeat(
            beat_id="bt1", description="Compare perspectives",
            passage_ids=["p1", "p2"],
            primary_book_id="b1", supporting_book_ids=["b2"],
            synthesis_instruction="contrast",
        )
        plan = EpisodePlan(
            episode_number=1, title="Episode 1",
            driving_question="What decision matters most?",
            thematic_focus="Decision making",
            unresolved_questions=["What remains unresolved?"],
            payoff_shape="Complicate the question without closing it.",
            beats=[beat],
            synthesis_context=EpisodeSynthesisContext(
                merged_narratives=[
                    EpisodeMergedNarrativeRef(
                        merged_narrative_id="merged_narrative_001",
                        topic="Topic",
                        narrative="Narrative",
                        source_passage_ids=["p1"],
                    )
                ],
                unresolved_tensions=[
                    EpisodeSynthesisTension(
                        tension_id="tension_001",
                        question="What remains unresolved?",
                    )
                ],
                quality_score=0.8,
            ),
            book_balance={"b1": 0.6, "b2": 0.4},
        )
        data = json.loads(plan.model_dump_json())
        restored = EpisodePlan.model_validate(data)
        assert len(restored.beats) == 1
        assert restored.beats[0].synthesis_instruction == "contrast"
        assert restored.synthesis_context is not None
        assert restored.synthesis_context.merged_narratives[0].merged_narrative_id == "merged_narrative_001"

    def test_default_target_duration_minutes(self):
        plan = EpisodePlan(
            episode_number=1,
            title="Episode 1",
            driving_question="What is at stake?",
            unresolved_questions=["What remains open?"],
            payoff_shape="Leave the listener with a sharpened tension.",
        )
        assert plan.target_duration_minutes == 140.0


class TestEpisodeScript:
    def test_roundtrip(self):
        citation = Citation(
            text_span="Author argues X", passage_id="p1",
            book_id="b1", chunk_ids=["c1"], confidence=0.95,
        )
        segment = ScriptSegment(
            segment_id="s1", text="The narration text.",
            segment_type="body", beat_id="bt1",
            source_book_ids=["b1"], citations=[citation],
        )
        script = EpisodeScript(
            episode_number=1, title="Ep 1",
            segments=[segment], total_word_count=4,
            citations=[citation],
        )
        data = json.loads(script.model_dump_json())
        restored = EpisodeScript.model_validate(data)
        assert len(restored.segments) == 1
        assert restored.citations[0].confidence == 0.95


class TestGroundingReport:
    def test_status_values(self):
        report = GroundingReport(
            episode_number=1,
            claim_assessments=[
                ClaimAssessment(
                    claim_text="Claim 1",
                    cited_passage_id="p1",
                    status="SUPPORTED",
                ),
                ClaimAssessment(
                    claim_text="Claim 2",
                    cited_passage_id="p2",
                    status="FABRICATED",
                    explanation="No support in passage.",
                ),
            ],
            overall_status="NEEDS_REPAIR",
            grounding_score=0.5,
        )
        supported = [ca for ca in report.claim_assessments if ca.status == "SUPPORTED"]
        assert len(supported) == 1

    def test_cross_book_claims(self):
        report = GroundingReport(
            episode_number=1,
            cross_book_claims=[
                CrossBookClaimAssessment(
                    claim_text="Author A agrees with Author B",
                    book_ids=["b1", "b2"],
                    passage_ids=["p1", "p2"],
                    comparison_valid=False,
                    failure_reason="false_equivalence",
                ),
            ],
            overall_status="NEEDS_REPAIR",
            attribution_accuracy=0.5,
        )
        assert not report.cross_book_claims[0].comparison_valid


class TestNarrativeStrategy:
    def test_valid_strategies(self):
        for strategy_type in ["thesis_driven", "debate", "chronological", "convergence", "mosaic"]:
            strategy = NarrativeStrategy(
                strategy_type=strategy_type,
                justification="Test",
                series_arc="Test arc",
                episode_arc_details=[],
            )
            assert strategy.strategy_type == strategy_type

    def test_episode_assignments_roundtrip(self):
        strategy = NarrativeStrategy(
            strategy_type="convergence",
            justification="Test",
            series_arc="Arc",
            episode_arc_details=[
                EpisodeArcDetail(
                    episode_number=1,
                    arc_summary="Arc summary",
                    narrative_stakes="Stakes",
                    progression_beats=["Beat 1"],
                    unresolved_questions=["Question 1"],
                    episode_inquiries=[
                        {"axis_id": "ax1", "question": "Inquiry 1?"},
                        {"axis_id": "ax1", "question": "Inquiry 2?"},
                        {"axis_id": "ax1", "question": "Inquiry 3?"},
                        {"axis_id": "ax1", "question": "Inquiry 4?"},
                    ],
                    payoff_shape="Payoff shape",
                )
            ],
            episode_assignments=[
                EpisodeAssignment(
                    episode_number=1,
                    title="Episode 1",
                    driving_question="What is this episode trying to answer?",
                    thematic_focus="Focus",
                    axis_ids=["ax1"],
                    insight_ids=["in1"],
                    merged_narrative_ids=["merged_narrative_001"],
                    tension_ids=["tension_001"],
                    episode_strategy="Set context",
                )
            ],
        )
        data = json.loads(strategy.model_dump_json())
        restored = NarrativeStrategy.model_validate(data)
        assert restored.episode_assignments[0].axes[0].axis_id == "ax1"
        assert restored.episode_assignments[0].axis_ids == ["ax1"]
        assert restored.episode_assignments[0].merged_narrative_ids == ["merged_narrative_001"]

    def test_rejects_misaligned_episode_arc_details(self):
        with pytest.raises(ValidationError, match="episode_arc_details must align"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Test",
                series_arc="Arc",
                episode_arc_details=[
                    EpisodeArcDetail(
                        episode_number=2,
                        arc_summary="Arc summary",
                        narrative_stakes="Stakes",
                        progression_beats=["Beat 1"],
                        unresolved_questions=["Question 1"],
                        episode_inquiries=[
                            {"axis_id": "ax1", "question": "Inquiry 1?"},
                            {"axis_id": "ax1", "question": "Inquiry 2?"},
                            {"axis_id": "ax1", "question": "Inquiry 3?"},
                            {"axis_id": "ax1", "question": "Inquiry 4?"},
                        ],
                        payoff_shape="Payoff shape",
                    )
                ],
                episode_assignments=[
                    EpisodeAssignment(
                        episode_number=1,
                        title="Episode 1",
                        driving_question="What is this episode trying to answer?",
                    )
                ],
            )

    def test_rejects_outline_and_detail_length_mismatch(self):
        with pytest.raises(ValidationError, match="episode_arc_outline and episode_arc_details"):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Test",
                series_arc="Arc",
                episode_arc_outline=["Ep1", "Ep2"],
                episode_arc_details=[
                    EpisodeArcDetail(
                        episode_number=1,
                        arc_summary="Arc summary",
                        narrative_stakes="Stakes",
                        progression_beats=["Beat 1"],
                        unresolved_questions=["Question 1"],
                        episode_inquiries=[
                            {"axis_id": "ax1", "question": "Inquiry 1?"},
                            {"axis_id": "ax1", "question": "Inquiry 2?"},
                            {"axis_id": "ax1", "question": "Inquiry 3?"},
                            {"axis_id": "ax1", "question": "Inquiry 4?"},
                        ],
                        payoff_shape="Payoff shape",
                    )
                ],
            )

    def test_rejects_episode_inquiries_below_minimum(self):
        with pytest.raises(ValidationError):
            EpisodeArcDetail(
                episode_number=1,
                arc_summary="Arc summary",
                narrative_stakes="Stakes",
                progression_beats=["Beat 1"],
                unresolved_questions=["Question 1"],
                episode_inquiries=[
                    {"axis_id": "ax1", "question": "Inquiry 1?"},
                    {"axis_id": "ax1", "question": "Inquiry 2?"},
                    {"axis_id": "ax1", "question": "Inquiry 3?"},
                ],
                payoff_shape="Payoff shape",
            )

    def test_rejects_recommended_episode_count_above_eight(self):
        with pytest.raises(ValidationError):
            NarrativeStrategy(
                strategy_type="convergence",
                justification="Test",
                series_arc="Arc",
                recommended_episode_count=9,
                episode_arc_details=[],
            )


class TestSpokenScript:
    def test_roundtrip(self):
        spoken = SpokenScript(
            episode_number=1, title="Ep 1",
            segments=[
                SpokenSegment(segment_id="s1", text="Hello listeners.", max_words=250),
            ],
            tts_provider="openai",
        )
        data = json.loads(spoken.model_dump_json())
        restored = SpokenScript.model_validate(data)
        assert len(restored.segments) == 1
        assert restored.segments[0].speech_hints.style == "neutral"
        assert restored.segments[0].ssml_hints.delivery_style == "neutral"

    def test_spoken_segment_normalizes_invalid_hint_values(self):
        segment = SpokenSegment.model_validate(
            {
                "segment_id": "s1",
                "text": "Hello listeners.",
                "ssml_hints": {
                    "delivery_style": "wild",
                    "emphasis_level": "MAXIMUM",
                    "pause_before_ms": -20,
                    "pause_after_ms": "9999",
                    "speech_rate": "warp",
                },
            }
        )
        assert segment.speech_hints.style == "neutral"
        assert segment.speech_hints.intensity == "strong"
        assert segment.speech_hints.pause_before_ms == 0
        assert segment.speech_hints.pause_after_ms == 2000
        assert segment.speech_hints.pace == "normal"
        assert segment.ssml_hints.delivery_style == "neutral"

    def test_spoken_segment_rejects_extra_hint_keys(self):
        with pytest.raises(ValidationError):
            SpokenSegment.model_validate(
                {
                    "segment_id": "s1",
                    "text": "Hello listeners.",
                    "ssml_hints": {
                        "delivery_style": "neutral",
                        "unexpected": True,
                    },
                }
            )

    def test_spoken_segment_accepts_canonical_speech_hints(self):
        segment = SpokenSegment.model_validate(
            {
                "segment_id": "s1",
                "text": "Hello listeners.",
                "speech_hints": {
                    "style": "measured",
                    "intensity": "medium",
                    "pace": "fast",
                    "pronunciation_hints": {"text": "Nehru", "spoken_as": "NAY-roo"},
                    "emphasis_targets": "transfer of power",
                    "render_strategy": "isolate_phrase",
                },
            }
        )
        assert segment.speech_hints.style == "measured"
        assert segment.speech_hints.intensity == "medium"
        assert segment.speech_hints.pace == "faster"
        assert segment.speech_hints.pronunciation_hints[0].spoken_as == "NAY-roo"
        assert segment.speech_hints.emphasis_targets == ["transfer of power"]
        assert segment.speech_hints.render_strategy == "isolate_phrase"


class TestRenderManifest:
    def test_roundtrip(self):
        manifest = RenderManifest(
            episode_number=1,
            segments=[
                RenderSegment(
                    segment_id="rs1", text="Hello",
                    voice_id="ballad", speed=1.0,
                ),
            ],
            total_segments=1,
            estimated_duration_seconds=120,
        )
        data = json.loads(manifest.model_dump_json())
        restored = RenderManifest.model_validate(data)
        assert restored.total_segments == 1


class TestAudioManifest:
    def test_roundtrip(self):
        manifest = AudioManifest(
            episode_number=1,
            audio_segments=[
                AudioSegmentResult(
                    segment_id="as1", audio_path="/tmp/audio.mp3",
                    duration_seconds=60.0, success=True,
                ),
            ],
            total_duration_seconds=60.0,
        )
        data = json.loads(manifest.model_dump_json())
        restored = AudioManifest.model_validate(data)
        assert restored.audio_segments[0].success


class TestThematicProject:
    def test_roundtrip(self):
        project = ThematicProject(
            project_id="proj1", theme="AI and creativity",
            requested_episode_count=4,
            recommended_episode_count=8,
            episode_count=3, status=ProjectStatus.INGESTING,
        )
        data = json.loads(project.model_dump_json())
        restored = ThematicProject.model_validate(data)
        assert restored.status == ProjectStatus.INGESTING
        assert restored.requested_episode_count == 4
        assert restored.recommended_episode_count == 8

    def test_with_books(self):
        project = ThematicProject(
            project_id="proj1", theme="Test",
            books=[
                BookRecord(
                    book_id="b1", title="Book 1", author="Author A",
                    source_path="/test.txt", source_type="txt",
                ),
                BookRecord(
                    book_id="b2", title="Book 2", author="Author B",
                    source_path="/test2.txt", source_type="txt",
                ),
            ],
            episode_count=3,
        )
        assert len(project.books) == 2

    def test_sub_themes_trim_dedupe_preserve_order(self):
        project = ThematicProject(
            project_id="proj1",
            theme="Test",
            sub_themes=[" borders ", "displacement", "borders", " governance "],
        )
        assert project.sub_themes == ["borders", "displacement", "governance"]

    def test_sub_themes_rejects_empty_entries(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ThematicProject(
                project_id="proj1",
                theme="Test",
                sub_themes=["valid", "   "],
            )

    def test_sub_themes_max_fifteen(self):
        with pytest.raises(ValidationError, match="at most 15"):
            ThematicProject(
                project_id="proj1",
                theme="Test",
                sub_themes=[
                    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
                    "a9", "a10", "a11", "a12", "a13", "a14", "a15", "a16",
                ],
            )

    def test_rejects_recommended_episode_count_above_eight(self):
        with pytest.raises(ValidationError):
            ThematicProject(
                project_id="proj1",
                theme="Test",
                recommended_episode_count=9,
            )


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.max_axes == 30
        assert config.min_axes == 25
        assert config.passage_retrieval_percentage == 0.25
        assert config.passage_retrieval_min_per_book == 20
        assert config.passage_retrieval_max_per_book == 50
        assert config.axis_candidate_target_total == 250
        assert config.admission_floor_per_book == 2
        assert config.retrieval_relevance_power == 1.2
        assert config.retrieval_soft_threshold == 0.35
        assert config.chapter_penalty_weight == 0.05
        assert config.rerank_top_k == 30
        assert config.max_repair_attempts == 3
        assert config.episode_write_concurrency == 7
        assert config.passage_extraction_concurrency == 10
        assert config.target_episode_minutes == 140.0
        assert config.min_episode_minutes == 125.0
        assert config.duration_shortfall_policy == "warn"

    def test_custom_values(self):
        config = PipelineConfig(
            max_axes=10, synthesis_quality_threshold=0.7,
            episode_write_concurrency=4,
        )
        assert config.max_axes == 10
        assert config.synthesis_quality_threshold == 0.7

    def test_retrieval_budget_bounds_validation(self):
        with pytest.raises(ValidationError, match="passage_retrieval_max_per_book"):
            PipelineConfig(
                passage_retrieval_min_per_book=21,
                passage_retrieval_max_per_book=20,
            )

    def test_rejects_removed_retrieval_weighting_fields(self):
        with pytest.raises(ValidationError):
            PipelineConfig(retrieval_conf_weight=0.2)


class TestEnums:
    def test_synthesis_tag_values(self):
        assert SynthesisTag.AGREES_WITH == "agrees_with"
        assert SynthesisTag.CONTRADICTS == "contradicts"

    def test_insight_type_values(self):
        assert InsightType.SYNCHRONICITY == "synchronicity"
        assert InsightType.EPISTEMIC_DRIFT == "epistemic_drift"

    def test_project_status_values(self):
        assert ProjectStatus.INGESTING == "ingesting"
        assert ProjectStatus.COMPLETE == "complete"


class TestChunkingConfig:
    def test_defaults(self):
        config = ChunkingConfig()
        assert config.max_chunk_words == 400
        assert config.overlap_words == 50
        assert config.min_chunk_words == 80

    def test_custom_split_on(self):
        config = ChunkingConfig(split_on=["\n\n", ". ", "! "])
        assert len(config.split_on) == 3
