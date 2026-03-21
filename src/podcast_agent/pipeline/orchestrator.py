"""End-to-end orchestration for the book-to-podcast pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import wave

from pydantic import ValidationError

from podcast_agent.agents import (
    AnalysisAgent,
    EpisodePlanningAgent,
    GroundingValidationAgent,
    RepairAgent,
    SpokenDeliveryAgent,
    StructuringAgent,
    WritingAgent,
)
from podcast_agent.config import Settings
from podcast_agent.db.repository import ArtifactStore, InMemoryRepository, Repository
from podcast_agent.ingestion import read_source_text
from podcast_agent.llm import LLMClient, build_llm_client
from podcast_agent.retrieval import RetrievalService, embed_text
from podcast_agent.run_logging import RunLogger
from podcast_agent.schemas.models import (
    AudioManifest,
    AudioSegmentFile,
    BookIngestionResult,
    BookStructure,
    EpisodeOutput,
    EpisodePlan,
    EpisodeScript,
    GroundingReport,
    RenderManifest,
    RenderSegment,
    RepairResult,
    EpisodeSegment,
    SpokenDeliveryArcPlan,
    SpokenDeliveryEpisodeResult,
    SpokenEpisodeNarration,
    SourceType,
)
from podcast_agent.tts import TTSClient, build_tts_client


class WavMergeError(RuntimeError):
    """Raised when WAV segments cannot be merged cleanly."""

    def __init__(self, message: str, diagnostics: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class PipelineOrchestrator:
    """Coordinates all stages from ingestion through render manifest."""

    def __init__(
        self,
        settings: Settings | None = None,
        repository: Repository | None = None,
        llm: LLMClient | None = None,
        tts: TTSClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.repository = repository or InMemoryRepository()
        self.llm = llm or build_llm_client(self.settings)
        self.artifacts = ArtifactStore(self.settings.pipeline.artifact_root)
        self.run_logger = RunLogger(self.settings.pipeline.artifact_root)
        self.run_id: str | None = None
        self.current_book_id: str | None = None
        self._audio_semaphore = threading.BoundedSemaphore(self.settings.pipeline.audio_parallelism)
        if hasattr(self.llm, "set_run_logger"):
            self.llm.set_run_logger(self.run_logger)
        self.tts = tts or build_tts_client(self.settings)
        if hasattr(self.tts, "set_run_logger"):
            self.tts.set_run_logger(self.run_logger)
        self.retrieval = RetrievalService(self.repository)
        self.structuring_agent = StructuringAgent(
            llm=self.llm,
            max_chunk_words=self.settings.pipeline.max_chunk_words,
            chunk_overlap_words=self.settings.pipeline.chunk_overlap_words,
            max_structuring_chapter_words=self.settings.pipeline.max_structuring_chapter_words,
            max_structuring_llm_chapter_words=self.settings.pipeline.max_structuring_llm_chapter_words,
            structuring_parallelism=self.settings.pipeline.structuring_parallelism,
            structuring_window_words=self.settings.pipeline.structuring_window_words,
            structuring_window_overlap_words=self.settings.pipeline.structuring_window_overlap_words,
        )
        self.analysis_agent = AnalysisAgent(
            self.llm,
            max_payload_bytes=self.settings.pipeline.max_analysis_payload_bytes,
            max_payload_bytes_with_episode_count=self.settings.pipeline.max_analysis_payload_bytes_with_episode_count,
        )
        self.planning_agent = EpisodePlanningAgent(
            self.llm,
            minimum_source_words_per_episode=self.settings.pipeline.minimum_source_words_per_episode,
            min_episode_source_ratio=self.settings.pipeline.min_episode_source_ratio,
            spoken_words_per_minute=self.settings.pipeline.spoken_words_per_minute,
            max_episode_minutes=self.settings.pipeline.max_episode_minutes,
            max_payload_bytes=self.settings.pipeline.max_planning_payload_bytes,
            max_payload_bytes_with_episode_count=self.settings.pipeline.max_planning_payload_bytes_with_episode_count,
            section_beat_target_words=self.settings.pipeline.section_beat_target_words,
            beat_evidence_window_size=self.settings.pipeline.beat_evidence_window_size,
        )
        self.writing_agent = WritingAgent(
            self.llm,
            minimum_source_words_per_episode=self.settings.pipeline.minimum_source_words_per_episode,
            spoken_words_per_minute=self.settings.pipeline.spoken_words_per_minute,
            coverage_warning_min_ratio=self.settings.pipeline.coverage_warning_min_ratio,
            target_script_source_ratio=self.settings.pipeline.target_script_source_ratio,
            max_target_script_words=self.settings.pipeline.max_target_script_words,
            beat_parallelism=self.settings.pipeline.beat_parallelism,
            beat_write_retry_attempts=self.settings.pipeline.beat_write_retry_attempts,
            beat_write_timeout_seconds=self.settings.pipeline.beat_write_timeout_seconds,
        )
        self.validation_agent = GroundingValidationAgent(
            self.llm,
            grounding_parallelism=self.settings.pipeline.grounding_parallelism,
        )
        self.repair_agent = RepairAgent(self.llm)
        spoken_chunk_min_words = self.settings.spoken_delivery.chunk_min_words
        spoken_chunk_max_words = self.settings.spoken_delivery.chunk_max_words
        if self.settings.tts.provider.lower() == "kokoro":
            spoken_chunk_min_words = self.settings.tts.kokoro_chunk_min_words
            spoken_chunk_max_words = self.settings.tts.kokoro_chunk_max_words
        self.spoken_delivery_agent = SpokenDeliveryAgent(
            self.llm,
            timeout_seconds=self.settings.spoken_delivery.timeout_seconds,
            chunk_min_words=spoken_chunk_min_words,
            chunk_max_words=spoken_chunk_max_words,
        )

    def log_command(self, command_name: str, arguments: dict) -> None:
        """Record the CLI command or orchestration entrypoint."""

        self.run_logger.log("command", command_name=command_name, arguments=arguments)

    def ingest_book(
        self,
        source_path: str | Path,
        title: str | None = None,
        author: str = "Unknown",
    ) -> BookIngestionResult:
        """Read source text and persist the book record."""

        path = Path(source_path)
        raw_text = read_source_text(path)
        resolved_title = title or path.stem.replace("_", " ").title()
        book_id = _slugify(resolved_title)
        source_type = _detect_source_type(path)
        ingestion = BookIngestionResult(
            book_id=book_id,
            title=resolved_title,
            author=author,
            source_path=str(path),
            source_type=source_type,
            raw_text=raw_text,
        )
        self.current_book_id = book_id
        if self.run_id is None:
            self.run_id = _new_run_id(book_id)
            self.run_logger.bind_run(self.run_id)
            self.run_logger.log("run_started", run_id=self.run_id, book_id=book_id, title=resolved_title)
        self.run_logger.log(
            "stage_start",
            stage="ingest_book",
            source_path=str(path),
            title=resolved_title,
            author=author,
        )
        if source_type == SourceType.PDF:
            page_count = len(re.findall(r"^\[Page \d+\]$", raw_text, flags=re.MULTILINE))
            word_count = len(raw_text.split())
            warnings: list[str] = []
            if page_count == 0:
                warnings.append("missing_page_markers")
            if word_count < 50:
                warnings.append("low_extracted_word_count")
            self.run_logger.log(
                "pdf_ingest_diagnostics",
                source_path=str(path),
                page_count=page_count,
                extracted_word_count=word_count,
                warnings=warnings,
            )
        self.repository.save_book(ingestion)
        self._write_artifact(self._book_artifact_key(book_id), "ingestion", ingestion.model_dump(mode="json"))
        self.run_logger.log("stage_end", stage="ingest_book", book_id=book_id)
        return ingestion

    def index_book(
        self,
        ingestion: BookIngestionResult,
        *,
        start_chapter: str | None = None,
        end_chapter: str | None = None,
    ):
        """Structure the book and store chunks plus embeddings in the repository."""

        self.run_logger.log("stage_start", stage="index_book", book_id=ingestion.book_id)
        structure = self.structuring_agent.structure(
            ingestion,
            start_chapter=start_chapter,
            end_chapter=end_chapter,
        )
        if start_chapter is not None or end_chapter is not None:
            self.run_logger.log(
                "chapter_selection_applied",
                book_id=structure.book_id,
                start_chapter=start_chapter,
                end_chapter=end_chapter,
                chapter_count=len(structure.chapters),
                chunk_count=len(structure.chunks),
            )
        self.repository.save_structure(structure)
        embeddings = {
            chunk.chunk_id: embed_text(
                chunk.text,
                dimensions=self.settings.pipeline.embedding_dimensions,
            )
            for chunk in structure.chunks
        }
        self.repository.save_embeddings(structure.book_id, embeddings)
        self._write_artifact(
            self._book_artifact_key(structure.book_id),
            "structure",
            structure.model_dump(mode="json"),
        )
        self._write_artifact(self._book_artifact_key(structure.book_id), "embeddings", embeddings)
        self.run_logger.log(
            "stage_end",
            stage="index_book",
            book_id=structure.book_id,
            chunk_count=len(structure.chunks),
        )
        return structure

    def plan_episodes(self, structure, *, episode_count: int):
        """Analyze the book and produce the series plan."""

        self.run_logger.log("stage_start", stage="plan_episodes", book_id=structure.book_id)
        analysis = self.analysis_agent.analyze(structure, episode_count)
        plan = self.planning_agent.plan(structure, analysis, episode_count)
        self._write_artifact(
            self._book_artifact_key(structure.book_id),
            "analysis",
            analysis.model_dump(mode="json"),
        )
        self._write_artifact(
            self._book_artifact_key(structure.book_id),
            "series_plan",
            plan.model_dump(mode="json"),
        )
        self.run_logger.log(
            "stage_end",
            stage="plan_episodes",
            book_id=structure.book_id,
            episode_count=len(plan.episodes),
            requested_episode_count=episode_count,
            episode_parallelism=self.settings.pipeline.episode_parallelism,
        )
        return analysis, plan

    def write_episode(self, book_id: str, episode_plan: EpisodePlan) -> EpisodeScript:
        """Generate a single episode script from retrieved evidence."""

        self.run_logger.log("stage_start", stage="write_episode", book_id=book_id, episode_id=episode_plan.episode_id)
        retrieval_hits = self.retrieval.fetch_for_episode(book_id=book_id, chunk_ids=episode_plan.chunk_ids)
        self.run_logger.log(
            "episode_assignment_diagnostics",
            book_id=book_id,
            episode_id=episode_plan.episode_id,
            assigned_chunk_count=len(retrieval_hits),
            beat_count=len(episode_plan.beats),
            beat_chunk_counts=[len(beat.chunk_ids) for beat in episode_plan.beats],
        )
        script = self.writing_agent.write(episode_plan, retrieval_hits)
        self.run_logger.log("stage_end", stage="write_episode", book_id=book_id, episode_id=episode_plan.episode_id)
        return script

    def validate_episode(self, book_id: str, script: EpisodeScript) -> GroundingReport:
        """Validate one episode script against the cited chunks."""

        self.run_logger.log("stage_start", stage="validate_episode", book_id=book_id, episode_id=script.episode_id)
        cited_chunk_ids = []
        for segment in script.segments:
            cited_chunk_ids.extend(segment.citations)
        retrieval_hits = self.retrieval.fetch_for_episode(book_id=book_id, chunk_ids=sorted(set(cited_chunk_ids)))
        report = self.validation_agent.validate(script, retrieval_hits)
        self.run_logger.log(
            "stage_end",
            stage="validate_episode",
            book_id=book_id,
            episode_id=script.episode_id,
            overall_status=report.overall_status,
        )
        return report

    def spoken_delivery_episode(
        self,
        book_id: str,
        script: EpisodeScript,
    ) -> tuple[
        SpokenEpisodeNarration | None,
        SpokenDeliveryEpisodeResult | None,
        SpokenDeliveryArcPlan | None,
    ]:
        """Rewrite a validated factual script into spoken-form delivery."""

        if not self.settings.spoken_delivery.enabled:
            return None, None, None
        self.run_logger.log("stage_start", stage="spoken_delivery_episode", book_id=book_id, episode_id=script.episode_id)
        spoken_script, spoken_delivery, arc_plan = self.spoken_delivery_agent.rewrite_full_episode_two_call(script)
        self.run_logger.log(
            "stage_end",
            stage="spoken_delivery_episode",
            book_id=book_id,
            episode_id=script.episode_id,
            chunk_count=len(spoken_script.chunks),
        )
        return spoken_script, spoken_delivery, arc_plan

    def repair_episode(
        self,
        book_id: str,
        script: EpisodeScript,
        report: GroundingReport,
    ) -> RepairResult:
        """Repair an episode until it validates or hits the attempt cap."""

        current_script = script
        current_report = report
        repair_result = None
        for attempt in range(1, self.settings.pipeline.max_repair_attempts + 1):
            failed_segments = self.repair_agent.failed_segments(current_script, current_report)
            if not failed_segments:
                return RepairResult(
                    episode_id=current_script.episode_id,
                    attempt=attempt,
                    repaired_segment_ids=[],
                    script=current_script,
                    report=current_report,
                )
            self.run_logger.log(
                "stage_start",
                stage="repair_episode",
                book_id=book_id,
                episode_id=script.episode_id,
                attempt=attempt,
            )
            failed_assessments = self.repair_agent.failed_claim_assessments(current_script, current_report)
            self.run_logger.log(
                "repair_payload_diagnostics",
                book_id=book_id,
                episode_id=current_script.episode_id,
                attempt=attempt,
                failed_segment_count=len(failed_segments),
                failed_segment_ids=[segment.segment_id for segment in failed_segments],
                failed_beat_ids=[segment.beat_id for segment in failed_segments],
                failed_claim_count=len(failed_assessments),
            )
            segment_repair = self.repair_agent.repair(current_script, current_report, attempt)
            previous_script = current_script
            current_script = self._merge_repaired_segments(previous_script, segment_repair.repaired_segments)
            self._log_repair_segment_diffs(
                book_id=book_id,
                episode_id=current_script.episode_id,
                attempt=attempt,
                before_script=previous_script,
                repaired_segments=segment_repair.repaired_segments,
            )
            current_report = self.validate_episode(book_id, current_script)
            repair_result = RepairResult(
                episode_id=current_script.episode_id,
                attempt=attempt,
                repaired_segment_ids=segment_repair.repaired_segment_ids,
                script=current_script,
                report=current_report,
            )
            self._write_artifact(
                self._episode_artifact_key(book_id, current_script.episode_id),
                f"repair_attempt_{attempt}",
                {
                    "episode_id": current_script.episode_id,
                    "attempt": attempt,
                    "repaired_segment_ids": segment_repair.repaired_segment_ids,
                    "repaired_beat_ids": [segment.beat_id for segment in segment_repair.repaired_segments],
                    "repaired_segments": [segment.model_dump(mode="json") for segment in segment_repair.repaired_segments],
                    "report": current_report.model_dump(mode="json"),
                },
            )
            self.run_logger.log(
                "stage_end",
                stage="repair_episode",
                book_id=book_id,
                episode_id=script.episode_id,
                attempt=attempt,
                overall_status=current_report.overall_status,
            )
            if current_report.overall_status == "pass":
                return repair_result.model_copy(update={"repaired_segment_ids": repair_result.repaired_segment_ids})
        if repair_result is None:
            raise RuntimeError("Repair requested with no attempts configured.")
        return repair_result

    def _merge_repaired_segments(
        self,
        script: EpisodeScript,
        repaired_segments: list[EpisodeSegment],
    ) -> EpisodeScript:
        repaired_by_id = {segment.segment_id: segment for segment in repaired_segments}
        merged_segments = [repaired_by_id.get(segment.segment_id, segment) for segment in script.segments]
        return script.model_copy(update={"segments": merged_segments})

    def _log_repair_segment_diffs(
        self,
        *,
        book_id: str,
        episode_id: str,
        attempt: int,
        before_script: EpisodeScript,
        repaired_segments: list[EpisodeSegment],
    ) -> None:
        previous_by_id = {segment.segment_id: segment for segment in before_script.segments}
        repaired_segment_ids: list[str] = []
        repaired_beat_ids: list[str] = []
        for repaired_segment in repaired_segments:
            previous_segment = previous_by_id[repaired_segment.segment_id]
            repaired_segment_ids.append(repaired_segment.segment_id)
            repaired_beat_ids.append(repaired_segment.beat_id)
            self.run_logger.log(
                "repair_segment_diff",
                book_id=book_id,
                episode_id=episode_id,
                attempt=attempt,
                segment_id=repaired_segment.segment_id,
                beat_id=repaired_segment.beat_id,
                heading_changed=previous_segment.heading != repaired_segment.heading,
                old_narration_word_count=len(previous_segment.narration.split()),
                new_narration_word_count=len(repaired_segment.narration.split()),
                old_claim_count=len(previous_segment.claims),
                new_claim_count=len(repaired_segment.claims),
                old_citation_count=len(previous_segment.citations),
                new_citation_count=len(repaired_segment.citations),
                old_claim_ids=[claim.claim_id for claim in previous_segment.claims],
                new_claim_ids=[claim.claim_id for claim in repaired_segment.claims],
            )
        self.run_logger.log(
            "repair_merge_summary",
            book_id=book_id,
            episode_id=episode_id,
            attempt=attempt,
            repaired_segment_ids=repaired_segment_ids,
            repaired_beat_ids=repaired_beat_ids,
        )

    def render_manifest(
        self,
        script: EpisodeScript,
        report: GroundingReport,
        spoken_script: SpokenEpisodeNarration | None = None,
    ) -> RenderManifest:
        """Build the TTS-ready manifest from a validated script."""

        if report.overall_status != "pass":
            raise RuntimeError(
                f"Cannot render manifest for episode '{script.episode_id}' with failed grounding."
            )
        grounded_claim_ids = {
            assessment.claim_id
            for assessment in report.claim_assessments
            if assessment.status.value == "grounded"
        }
        segments = []
        if spoken_script is not None:
            claim_ids = sorted(grounded_claim_ids)
            for chunk in spoken_script.chunks:
                text = chunk.text
                ssml = f"<speak>{text}</speak>"
                segments.append(
                    RenderSegment(
                        segment_id=chunk.chunk_id,
                        speaker=script.narrator,
                        text=text,
                        ssml=ssml,
                        grounded_claim_ids=claim_ids,
                    )
                )
        else:
            for segment in script.segments:
                claims = [claim.claim_id for claim in segment.claims if claim.claim_id in grounded_claim_ids]
                if not claims:
                    continue
                text = segment.narration
                ssml = f"<speak>{text}</speak>"
                segments.append(
                    RenderSegment(
                        segment_id=segment.segment_id,
                        speaker=script.narrator,
                        text=text,
                        ssml=ssml,
                        grounded_claim_ids=claims,
                    )
                )
        manifest = RenderManifest(
            episode_id=script.episode_id,
            title=script.title,
            narrator=script.narrator,
            segments=segments,
        )
        if self.current_book_id is None:
            raise RuntimeError("Current book context is required to persist render artifacts.")
        self.run_logger.log(
            "stage_start",
            stage="render_manifest",
            book_id=self.current_book_id,
            episode_id=script.episode_id,
        )
        self.run_logger.log(
            "stage_end",
            stage="render_manifest",
            book_id=self.current_book_id,
            episode_id=script.episode_id,
        )
        return manifest

    def _load_spoken_script_artifact(
        self,
        book_id: str,
        episode_id: str,
        *,
        episode_dir: Path | None = None,
    ) -> SpokenEpisodeNarration:
        if episode_dir is None:
            episode_dir = self.settings.pipeline.artifact_root / self._episode_artifact_key(book_id, episode_id)
        spoken_script_path = episode_dir / "spoken_script.json"
        if not spoken_script_path.exists():
            raise RuntimeError(
                f"spoken_script.json not found at '{spoken_script_path}'. "
                "Run spoken-delivery or enable spoken delivery before synthesizing audio."
            )
        try:
            return SpokenEpisodeNarration.model_validate_json(
                spoken_script_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid spoken_script.json at '{spoken_script_path}': {exc}") from exc

    def _render_manifest_from_spoken_script(
        self,
        spoken_script: SpokenEpisodeNarration,
        report: GroundingReport | None,
    ) -> RenderManifest:
        grounded_claim_ids: list[str] = []
        if report is not None and report.overall_status == "pass":
            grounded_claim_ids = sorted(
                assessment.claim_id
                for assessment in report.claim_assessments
                if assessment.status.value == "grounded"
            )
        segments = []
        for chunk in spoken_script.chunks:
            text = chunk.text
            ssml = f"<speak>{text}</speak>"
            segments.append(
                RenderSegment(
                    segment_id=chunk.chunk_id,
                    speaker=spoken_script.narrator,
                    text=text,
                    ssml=ssml,
                    grounded_claim_ids=grounded_claim_ids,
                )
            )
        return RenderManifest(
            episode_id=spoken_script.episode_id,
            title=spoken_script.title,
            narrator=spoken_script.narrator,
            segments=segments,
        )

    def _load_report_from_episode_dir(self, episode_dir: Path) -> GroundingReport | None:
        episode_output_path = episode_dir / "episode_output.json"
        if not episode_output_path.exists():
            return None
        try:
            payload = json.loads(episode_output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        report_payload = payload.get("report")
        if report_payload is None:
            return None
        try:
            return GroundingReport.model_validate_json(json.dumps(report_payload))
        except ValidationError:
            return None

    def _prepare_manifest_for_audio(self, manifest: RenderManifest) -> RenderManifest:
        if self.settings.tts.provider.lower() != "kokoro":
            return manifest
        chunk_size_words = self.settings.tts.kokoro_chunk_max_words
        prepared_segments: list[RenderSegment] = []
        for segment in manifest.segments:
            chunks = self._split_text_into_chunks(segment.text, chunk_size_words)
            if len(chunks) == 1:
                prepared_segments.append(segment)
                continue
            for index, chunk_text in enumerate(chunks, start=1):
                prepared_segments.append(
                    RenderSegment(
                        segment_id=f"{segment.segment_id}-chunk-{index}",
                        speaker=segment.speaker,
                        text=chunk_text,
                        ssml=f"<speak>{chunk_text}</speak>",
                        grounded_claim_ids=segment.grounded_claim_ids,
                    )
                )
        return RenderManifest(
            episode_id=manifest.episode_id,
            title=manifest.title,
            narrator=manifest.narrator,
            segments=prepared_segments,
        )

    @staticmethod
    def _split_text_into_chunks(text: str, chunk_size_words: int) -> list[str]:
        words = text.split()
        if len(words) <= chunk_size_words:
            return [text]
        return [
            " ".join(words[index : index + chunk_size_words])
            for index in range(0, len(words), chunk_size_words)
        ]

    def synthesize_audio(self, manifest: RenderManifest) -> AudioManifest:
        """Synthesize audio files for each renderable segment."""

        if self.current_book_id is None:
            raise RuntimeError("Current book context is required to synthesize audio.")
        manifest = self._prepare_manifest_for_audio(manifest)
        self.run_logger.log(
            "stage_start",
            stage="synthesize_audio",
            book_id=self.current_book_id,
            episode_id=manifest.episode_id,
        )
        provider = self.settings.tts.provider.lower()
        if provider == "kokoro":
            effective_parallelism = min(
                len(manifest.segments),
                self.settings.pipeline.audio_parallelism,
                self.settings.tts.kokoro_parallelism,
            )
            if effective_parallelism <= 1:
                synthesized_segments = [
                    self._synthesize_audio_segment(index, manifest.episode_id, segment)
                    for index, segment in enumerate(manifest.segments)
                ]
            else:
                with ThreadPoolExecutor(max_workers=effective_parallelism) as executor:
                    futures = [
                        executor.submit(self._synthesize_audio_segment, index, manifest.episode_id, segment)
                        for index, segment in enumerate(manifest.segments)
                    ]
                    synthesized_segments = []
                    try:
                        for future in futures:
                            synthesized_segments.append(future.result())
                    except Exception:
                        for future in futures:
                            future.cancel()
                        raise
        elif len(manifest.segments) <= 1 or self.settings.pipeline.audio_parallelism == 1:
            synthesized_segments = [
                self._synthesize_audio_segment(index, manifest.episode_id, segment)
                for index, segment in enumerate(manifest.segments)
            ]
        else:
            max_workers = min(len(manifest.segments), self.settings.pipeline.audio_parallelism)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._synthesize_audio_segment, index, manifest.episode_id, segment)
                    for index, segment in enumerate(manifest.segments)
                ]
                synthesized_segments = []
                try:
                    for future in futures:
                        synthesized_segments.append(future.result())
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
        synthesized_segments.sort(key=lambda item: item[0])
        audio_segments = [audio_segment for _, _, audio_segment in synthesized_segments]
        if provider == "kokoro":
            wav_segments = [
                (audio_segment.segment_id, audio_bytes)
                for _, audio_bytes, audio_segment in synthesized_segments
            ]
            try:
                combined_wav = self._combine_wav_segments(wav_segments)
            except WavMergeError as exc:
                self.run_logger.log(
                    "audio_merge_failed",
                    book_id=self.current_book_id,
                    episode_id=manifest.episode_id,
                    provider=provider,
                    segment_count=len(wav_segments),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    diagnostics=exc.diagnostics,
                )
                raise
            if self.settings.tts.audio_format.lower() == "mp3":
                combined_audio = self._convert_wav_to_mp3_bytes(combined_wav)
            else:
                combined_audio = combined_wav
        else:
            combined_audio = b"".join(audio_bytes for _, audio_bytes, _ in synthesized_segments)
        extension = self.settings.tts.audio_format
        file_name = f"{manifest.episode_id}.{extension}"
        relative_dir = self._episode_artifact_key(self.current_book_id, manifest.episode_id)
        audio_path = self.artifacts.write_bytes(relative_dir, file_name, combined_audio)
        self.run_logger.log(
            "artifact_written",
            artifact_path=str(audio_path),
            artifact_name=file_name,
        )
        audio_manifest = AudioManifest(
            episode_id=manifest.episode_id,
            title=manifest.title,
            narrator=manifest.narrator,
            voice=self.settings.tts.voice,
            audio_path=str(audio_path),
            audio_format=self.settings.tts.audio_format,
            segments=audio_segments,
        )
        self.run_logger.log(
            "stage_end",
            stage="synthesize_audio",
            book_id=self.current_book_id,
            episode_id=manifest.episode_id,
            segment_count=len(audio_segments),
        )
        return audio_manifest

    def _combine_wav_segments(self, segments: list[tuple[str, bytes]]) -> bytes:
        diagnostics: list[dict[str, object]] = []
        expected_format: tuple[int, int, int, str, str] | None = None
        expected_params: wave._wave_params | None = None
        frame_chunks: list[bytes] = []

        if not segments:
            raise WavMergeError("No WAV segments provided for merge.", diagnostics)

        for segment_id, audio_bytes in segments:
            try:
                with wave.open(io.BytesIO(audio_bytes)) as wav_handle:
                    params = wav_handle.getparams()
                    frames = wav_handle.readframes(wav_handle.getnframes())
            except wave.Error as exc:
                diagnostics.append(
                    {
                        "segment_id": segment_id,
                        "byte_count": len(audio_bytes),
                        "error": str(exc),
                    }
                )
                raise WavMergeError(
                    f"Failed to read WAV segment '{segment_id}': {exc}",
                    diagnostics,
                ) from exc
            diagnostics.append(
                {
                    "segment_id": segment_id,
                    "nchannels": params.nchannels,
                    "sampwidth": params.sampwidth,
                    "framerate": params.framerate,
                    "nframes": params.nframes,
                    "comptype": params.comptype,
                    "compname": params.compname,
                    "byte_count": len(audio_bytes),
                }
            )
            format_key = (
                params.nchannels,
                params.sampwidth,
                params.framerate,
                params.comptype,
                params.compname,
            )
            if expected_format is None:
                expected_format = format_key
                expected_params = params
            elif format_key != expected_format:
                raise WavMergeError(
                    f"WAV segment format mismatch: expected {expected_format} got {format_key} "
                    f"for segment '{segment_id}'.",
                    diagnostics,
                )
            frame_chunks.append(frames)

        if expected_params is None:
            raise WavMergeError("No WAV segments available after parsing.", diagnostics)

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_out:
            wav_out.setnchannels(expected_params.nchannels)
            wav_out.setsampwidth(expected_params.sampwidth)
            wav_out.setframerate(expected_params.framerate)
            wav_out.setcomptype(expected_params.comptype, expected_params.compname)
            for frames in frame_chunks:
                wav_out.writeframes(frames)
        return output.getvalue()

    def _convert_wav_to_mp3_bytes(self, wav_bytes: bytes) -> bytes:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg is required to convert WAV to MP3 but was not found on PATH.")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "output.mp3"
            input_path.write_bytes(wav_bytes)
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    str(output_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {message}")
            if not output_path.exists():
                raise RuntimeError("ffmpeg did not produce an output file.")
            return output_path.read_bytes()

    def regenerate_audio_from_artifact(
        self,
        artifact_path: str | Path,
        *,
        book_id: str | None = None,
    ) -> EpisodeOutput:
        """Regenerate episode audio from a saved artifact, using spoken_script.json for synthesis."""

        path = Path(artifact_path)
        payload = self._load_manifest_source(path)
        resolved_book_id = book_id or self._infer_book_id_from_artifact_path(path)
        if not resolved_book_id:
            raise ValueError(
                "Unable to determine book ID from artifact path. Provide --book-id for manifest-only audio regeneration."
            )
        episode_dir = path.parent
        spoken_script = self._load_spoken_script_artifact(
            resolved_book_id,
            episode_dir.name,
            episode_dir=episode_dir,
        )
        report = self._load_report_from_episode_dir(episode_dir)
        manifest = self._render_manifest_from_spoken_script(spoken_script, report)
        manifest = self._prepare_manifest_for_audio(manifest)

        self.current_book_id = resolved_book_id
        self.run_id = _new_run_id(resolved_book_id)
        self.run_logger.bind_run(self.run_id)
        self.run_logger.log("run_started", run_id=self.run_id, book_id=resolved_book_id, title=manifest.title)

        episode_output: EpisodeOutput
        if "manifest" in payload:
            source_output = EpisodeOutput.model_validate_json(json.dumps(payload))
            audio_manifest = self.synthesize_audio(manifest)
            episode_output = source_output.model_copy(update={"audio_manifest": audio_manifest})
        else:
            audio_manifest = self.synthesize_audio(manifest)
            episode_output = EpisodeOutput(
                plan=_placeholder_episode_plan(resolved_book_id, manifest),
                script=_placeholder_episode_script(manifest),
                report=_placeholder_grounding_report(manifest),
                manifest=manifest,
                audio_manifest=audio_manifest,
            )
        self._write_artifact(
            self._episode_artifact_key(resolved_book_id, manifest.episode_id),
            "episode_output",
            episode_output.model_dump(mode="json"),
        )
        return episode_output

    def spoken_delivery_from_artifact(
        self,
        artifact_path: str | Path,
    ) -> dict:
        """Run spoken delivery from a saved factual script or episode output artifact."""

        path = Path(artifact_path)
        payload = self._load_manifest_source(path)
        if "script" in payload and "report" in payload:
            report = GroundingReport.model_validate_json(json.dumps(payload["report"]))
            if report.overall_status != "pass":
                for artifact_name in (
                    "spoken_script.json",
                    "spoken_delivery.json",
                    "spoken_delivery_arc_plan.json",
                ):
                    (path.parent / artifact_name).unlink(missing_ok=True)
                raise ValueError(
                    f"Cannot run spoken delivery from artifact '{artifact_path}' because grounding did not pass."
                )
        try:
            if "script" in payload:
                script = EpisodeScript.model_validate_json(json.dumps(payload["script"]))
            else:
                script = EpisodeScript.model_validate_json(json.dumps(payload))
        except ValidationError as exc:
            raise ValueError(f"Artifact '{artifact_path}' does not contain a valid factual script: {exc}") from exc
        spoken_script, spoken_delivery, arc_plan = self.spoken_delivery_agent.rewrite_full_episode_two_call(script)
        script_path = path.parent / "spoken_script.json"
        delivery_path = path.parent / "spoken_delivery.json"
        script_path.write_text(spoken_script.model_dump_json(indent=2), encoding="utf-8")
        delivery_path.write_text(spoken_delivery.model_dump_json(indent=2), encoding="utf-8")
        if arc_plan is not None:
            arc_plan_path = path.parent / "spoken_delivery_arc_plan.json"
            arc_plan_path.write_text(arc_plan.model_dump_json(indent=2), encoding="utf-8")
        return {
            "factual_script": script.model_dump(mode="json"),
            "spoken_script": spoken_script.model_dump(mode="json"),
            "spoken_delivery": spoken_delivery.model_dump(mode="json"),
        }

    def _synthesize_audio_segment(
        self,
        index: int,
        episode_id: str,
        segment: RenderSegment,
    ) -> tuple[int, bytes, AudioSegmentFile]:
        audio_bytes = self._synthesize_segment_with_retry(episode_id, segment)
        return (
            index,
            audio_bytes,
            AudioSegmentFile(
                segment_id=segment.segment_id,
                speaker=segment.speaker,
                text=segment.text,
                grounded_claim_ids=segment.grounded_claim_ids,
            ),
        )

    def _synthesize_segment_with_retry(self, episode_id: str, segment: RenderSegment) -> bytes:
        total_attempts = self.settings.pipeline.audio_retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            self.run_logger.log(
                "tts_segment_attempt",
                book_id=self.current_book_id,
                episode_id=episode_id,
                segment_id=segment.segment_id,
                attempt=attempt,
                total_attempts=total_attempts,
            )
            try:
                with self._audio_semaphore:
                    return self.tts.synthesize(
                        segment.text,
                        voice=self.settings.tts.voice,
                        audio_format=self.settings.tts.audio_format,
                        instructions=self.settings.tts.instructions,
                    )
            except Exception as exc:
                if attempt == total_attempts:
                    self.run_logger.log(
                        "tts_segment_failed",
                        book_id=self.current_book_id,
                        episode_id=episode_id,
                        segment_id=segment.segment_id,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise RuntimeError(
                        f"Audio synthesis failed for segment '{segment.segment_id}' in episode "
                        f"'{episode_id}' after {attempt} attempts."
                    ) from exc
                self.run_logger.log(
                    "tts_segment_retry",
                    book_id=self.current_book_id,
                    episode_id=episode_id,
                    segment_id=segment.segment_id,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        raise RuntimeError(f"Audio synthesis exhausted retries for segment '{segment.segment_id}'.")

    def run_pipeline(
        self,
        source_path: str | Path,
        title: str | None = None,
        author: str = "Unknown",
        start_chapter: str | None = None,
        end_chapter: str | None = None,
        episode_count: int | None = None,
        synthesize_audio: bool = False,
    ) -> dict:
        """Run the end-to-end pipeline for every episode in the series plan."""

        if episode_count is None:
            raise TypeError("episode_count is required for run_pipeline")
        ingestion = self.ingest_book(source_path=source_path, title=title, author=author)
        structure = self.index_book(
            ingestion,
            start_chapter=start_chapter,
            end_chapter=end_chapter,
        )
        analysis, plan = self.plan_episodes(structure, episode_count=episode_count)
        self.run_logger.log(
            "episode_parallelism_configured",
            book_id=structure.book_id,
            requested_episode_count=episode_count,
            episode_count=len(plan.episodes),
            episode_parallelism=self.settings.pipeline.episode_parallelism,
        )
        episode_outputs = self._process_episode_plans(
            structure.book_id,
            plan.episodes,
            synthesize_audio=synthesize_audio,
        )
        episodes = [episode_output.model_dump(mode="json") for episode_output in episode_outputs]
        return {
            "ingestion": ingestion.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
            "series_plan": plan.model_dump(mode="json"),
            "episodes": episodes,
        }

    def _process_episode_plans(
        self,
        book_id: str,
        episode_plans: list[EpisodePlan],
        *,
        synthesize_audio: bool,
    ) -> list[EpisodeOutput]:
        if self.settings.pipeline.episode_parallelism == 1 or len(episode_plans) <= 1:
            return [
                self._run_episode_plan(book_id, episode_plan, synthesize_audio=synthesize_audio)
                for episode_plan in episode_plans
            ]
        with ThreadPoolExecutor(max_workers=self.settings.pipeline.episode_parallelism) as executor:
            episode_outputs = list(
                executor.map(
                    lambda episode_plan: self._run_episode_plan(
                        book_id,
                        episode_plan,
                        synthesize_audio=synthesize_audio,
                    ),
                    episode_plans,
                )
            )
        episode_outputs.sort(key=lambda episode_output: episode_output.plan.sequence)
        return episode_outputs

    def _run_episode_plan(
        self,
        book_id: str,
        episode_plan: EpisodePlan,
        *,
        synthesize_audio: bool,
    ) -> EpisodeOutput:
        try:
            script = self.write_episode(book_id, episode_plan)
        except Exception as exc:
            self.run_logger.log(
                "stage_failed",
                stage="write_episode",
                book_id=book_id,
                episode_id=episode_plan.episode_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        self._write_citation_audit(book_id, episode_plan, script)
        report = self.validate_episode(book_id, script)
        self._write_citation_audit(book_id, episode_plan, script, report)
        repair_attempts: list[RepairResult] = []
        if report.overall_status != "pass":
            repair = self.repair_episode(book_id, script, report)
            script = repair.script
            report = repair.report
            repair_attempts.append(repair)
            self._write_citation_audit(book_id, episode_plan, script, report)
        self.run_logger.log(
            "repair_summary",
            book_id=book_id,
            episode_id=episode_plan.episode_id,
            repair_attempt_count=len(repair_attempts),
            final_status=report.overall_status,
        )
        self._write_artifact(
            self._episode_artifact_key(book_id, episode_plan.episode_id),
            "factual_script",
            script.model_dump(mode="json"),
        )
        spoken_script = None
        spoken_delivery = None
        arc_plan = None
        if report.overall_status == "pass":
            spoken_script, spoken_delivery, arc_plan = self.spoken_delivery_episode(book_id, script)
            if spoken_script is not None and spoken_delivery is not None:
                self._write_artifact(
                    self._episode_artifact_key(book_id, episode_plan.episode_id),
                    "spoken_script",
                    spoken_script.model_dump(mode="json"),
                )
                self._write_artifact(
                    self._episode_artifact_key(book_id, episode_plan.episode_id),
                    "spoken_delivery",
                    spoken_delivery.model_dump(mode="json"),
                )
                if arc_plan is not None:
                    self._write_artifact(
                        self._episode_artifact_key(book_id, episode_plan.episode_id),
                        "spoken_delivery_arc_plan",
                        arc_plan.model_dump(mode="json"),
                    )
        manifest = None
        audio_manifest = None
        if report.overall_status == "pass":
            manifest = self.render_manifest(script, report, spoken_script=spoken_script)
            if synthesize_audio:
                spoken_script_artifact = self._load_spoken_script_artifact(book_id, episode_plan.episode_id)
                manifest = self._render_manifest_from_spoken_script(spoken_script_artifact, report)
                manifest = self._prepare_manifest_for_audio(manifest)
                audio_manifest = self.synthesize_audio(manifest)
        episode_output = EpisodeOutput(
            plan=episode_plan,
            script=script,
            report=report,
            spoken_script=spoken_script,
            spoken_delivery=spoken_delivery,
            manifest=manifest,
            audio_manifest=audio_manifest,
            repair_attempts=repair_attempts,
        )
        self._write_artifact(
            self._episode_artifact_key(book_id, episode_plan.episode_id),
            "episode_output",
            episode_output.model_dump(mode="json"),
        )
        return episode_output

    def _write_citation_audit(
        self,
        book_id: str,
        episode_plan: EpisodePlan,
        script: EpisodeScript,
        report: GroundingReport | None = None,
    ) -> None:
        retrieval_hits = self.retrieval.fetch_for_episode(book_id=book_id, chunk_ids=episode_plan.chunk_ids)
        payload = {
            "episode_id": episode_plan.episode_id,
            "writing": self.writing_agent.build_citation_audit(episode_plan, script, retrieval_hits),
            "validation": (
                self.validation_agent.build_citation_audit(script, report)
                if report is not None
                else None
            ),
        }
        self._write_artifact(
            self._episode_artifact_key(book_id, episode_plan.episode_id),
            "citation_audit",
            payload,
        )

    def _write_artifact(self, run_key: str, name: str, payload: dict) -> None:
        artifact_path = self.artifacts.write_json(run_key, name, payload)
        self.run_logger.log("artifact_written", artifact_path=str(artifact_path), artifact_name=name)

    def _book_artifact_key(self, book_id: str) -> str:
        if self.run_id is None:
            raise RuntimeError("Run ID is not initialized. Ingest a book before writing artifacts.")
        return f"{self.run_id}/{book_id}"

    def _episode_artifact_key(self, book_id: str, episode_id: str) -> str:
        if self.run_id is None:
            raise RuntimeError("Run ID is not initialized. Ingest a book before writing artifacts.")
        return f"{self.run_id}/{book_id}/{episode_id}"

    def _load_manifest_source(self, artifact_path: Path) -> dict:
        try:
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Artifact path does not exist: {artifact_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Artifact file is not valid JSON: {artifact_path}") from exc

    def _extract_render_manifest(self, payload: dict, artifact_path: Path) -> RenderManifest:
        try:
            if "manifest" in payload:
                manifest_payload = payload.get("manifest")
                if manifest_payload is None:
                    raise ValueError(f"Artifact '{artifact_path}' is missing a usable 'manifest' payload.")
                return RenderManifest.model_validate_json(json.dumps(manifest_payload))
            return RenderManifest.model_validate_json(json.dumps(payload))
        except ValidationError as exc:
            raise ValueError(f"Artifact '{artifact_path}' does not contain a valid render manifest: {exc}") from exc

    def _infer_book_id_from_artifact_path(self, artifact_path: Path) -> str | None:
        parts = artifact_path.parts
        for index, part in enumerate(parts[:-2]):
            if part == self.settings.pipeline.artifact_root.name and index + 2 < len(parts):
                return parts[index + 2]
        if len(parts) >= 3 and parts[-2].startswith("episode-"):
            return parts[-3]
        return None


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "book"


def _detect_source_type(path: Path) -> SourceType:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return SourceType.PDF
    if suffix == ".md":
        return SourceType.MARKDOWN
    return SourceType.TEXT

def _new_run_id(book_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    return f"{book_id}-{timestamp}"


def _placeholder_episode_plan(book_id: str, manifest: RenderManifest) -> EpisodePlan:
    del book_id
    return EpisodePlan(
        episode_id=manifest.episode_id,
        sequence=1,
        title=manifest.title,
        synopsis="Audio regenerated directly from an existing render manifest.",
        chapter_ids=[],
        themes=[],
    )


def _placeholder_episode_script(manifest: RenderManifest) -> EpisodeScript:
    return EpisodeScript(
        episode_id=manifest.episode_id,
        title=manifest.title,
        narrator=manifest.narrator,
        segments=[],
    )


def _placeholder_grounding_report(manifest: RenderManifest) -> GroundingReport:
    return GroundingReport(
        episode_id=manifest.episode_id,
        overall_status="pass",
        claim_assessments=[],
    )
