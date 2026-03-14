"""End-to-end orchestration for the book-to-podcast pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from podcast_agent.agents import (
    AnalysisAgent,
    EpisodePlanningAgent,
    GroundingValidationAgent,
    RepairAgent,
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
    EpisodeOutput,
    EpisodePlan,
    EpisodeScript,
    GroundingReport,
    RenderManifest,
    RenderSegment,
    RepairResult,
    SourceType,
)
from podcast_agent.tts import TTSClient, build_tts_client


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
            structuring_window_words=self.settings.pipeline.structuring_window_words,
            structuring_window_overlap_words=self.settings.pipeline.structuring_window_overlap_words,
        )
        self.analysis_agent = AnalysisAgent(self.llm)
        self.planning_agent = EpisodePlanningAgent(
            self.llm,
            minimum_source_words_per_episode=self.settings.pipeline.minimum_source_words_per_episode,
        )
        self.writing_agent = WritingAgent(
            self.llm,
            minimum_source_words_per_episode=self.settings.pipeline.minimum_source_words_per_episode,
            spoken_words_per_minute=self.settings.pipeline.spoken_words_per_minute,
        )
        self.validation_agent = GroundingValidationAgent(self.llm)
        self.repair_agent = RepairAgent(self.llm)

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
        ingestion = BookIngestionResult(
            book_id=book_id,
            title=resolved_title,
            author=author,
            source_path=str(path),
            source_type=_detect_source_type(path),
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
        self.repository.save_book(ingestion)
        self._write_artifact(self._book_artifact_key(book_id), "ingestion", ingestion.model_dump(mode="json"))
        self.run_logger.log("stage_end", stage="ingest_book", book_id=book_id)
        return ingestion

    def index_book(self, ingestion: BookIngestionResult):
        """Structure the book and store chunks plus embeddings in the repository."""

        self.run_logger.log("stage_start", stage="index_book", book_id=ingestion.book_id)
        structure = self.structuring_agent.structure(ingestion)
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

    def plan_episodes(self, structure):
        """Analyze the book and produce the series plan."""

        self.run_logger.log("stage_start", stage="plan_episodes", book_id=structure.book_id)
        analysis = self.analysis_agent.analyze(structure)
        plan = self.planning_agent.plan(structure, analysis)
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
        )
        return analysis, plan

    def write_episode(self, book_id: str, episode_plan: EpisodePlan) -> EpisodeScript:
        """Generate a single episode script from retrieved evidence."""

        self.run_logger.log("stage_start", stage="write_episode", book_id=book_id, episode_id=episode_plan.episode_id)
        retrieval_hits = self.retrieval.fetch_for_episode(book_id=book_id, chunk_ids=episode_plan.chunk_ids)
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
            self.run_logger.log(
                "stage_start",
                stage="repair_episode",
                book_id=book_id,
                episode_id=script.episode_id,
                attempt=attempt,
            )
            repair_result = self.repair_agent.repair(current_script, current_report, attempt)
            current_script = repair_result.script
            current_report = self.validate_episode(book_id, current_script)
            repair_result = repair_result.model_copy(update={"report": current_report})
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

    def render_manifest(self, script: EpisodeScript, report: GroundingReport) -> RenderManifest:
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
        for segment in script.segments:
            claims = [claim.claim_id for claim in segment.claims if claim.claim_id in grounded_claim_ids]
            if not claims:
                continue
            ssml = f"<speak>{segment.narration}</speak>"
            segments.append(
                RenderSegment(
                    segment_id=segment.segment_id,
                    speaker=script.narrator,
                    text=segment.narration,
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

    def synthesize_audio(self, manifest: RenderManifest) -> AudioManifest:
        """Synthesize audio files for each renderable segment."""

        if self.current_book_id is None:
            raise RuntimeError("Current book context is required to synthesize audio.")
        self.run_logger.log(
            "stage_start",
            stage="synthesize_audio",
            book_id=self.current_book_id,
            episode_id=manifest.episode_id,
        )
        audio_segments = []
        combined_audio = bytearray()
        for index, segment in enumerate(manifest.segments, start=1):
            audio_bytes = self.tts.synthesize(segment.text)
            combined_audio.extend(audio_bytes)
            audio_segments.append(
                AudioSegmentFile(
                    segment_id=segment.segment_id,
                    speaker=segment.speaker,
                    text=segment.text,
                    grounded_claim_ids=segment.grounded_claim_ids,
                )
            )
        extension = self.settings.tts.audio_format
        file_name = f"{manifest.episode_id}.{extension}"
        relative_dir = self._episode_artifact_key(self.current_book_id, manifest.episode_id)
        audio_path = self.artifacts.write_bytes(relative_dir, file_name, bytes(combined_audio))
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

    def run_pipeline(
        self,
        source_path: str | Path,
        title: str | None = None,
        author: str = "Unknown",
        synthesize_audio: bool = False,
    ) -> dict:
        """Run the end-to-end pipeline for every episode in the series plan."""

        ingestion = self.ingest_book(source_path=source_path, title=title, author=author)
        structure = self.index_book(ingestion)
        analysis, plan = self.plan_episodes(structure)
        episodes = []
        for episode_plan in plan.episodes:
            script = self.write_episode(structure.book_id, episode_plan)
            report = self.validate_episode(structure.book_id, script)
            repair_attempts: list[RepairResult] = []
            if report.overall_status != "pass":
                repair = self.repair_episode(structure.book_id, script, report)
                script = repair.script
                report = repair.report
                repair_attempts.append(repair)
            manifest = None
            audio_manifest = None
            if report.overall_status == "pass":
                manifest = self.render_manifest(script, report)
                if synthesize_audio:
                    audio_manifest = self.synthesize_audio(manifest)
            episode_output = EpisodeOutput(
                plan=episode_plan,
                script=script,
                report=report,
                manifest=manifest,
                audio_manifest=audio_manifest,
                repair_attempts=repair_attempts,
            )
            self._write_artifact(
                self._episode_artifact_key(structure.book_id, episode_plan.episode_id),
                "episode_output",
                episode_output.model_dump(mode="json"),
            )
            episodes.append(
                episode_output.model_dump(mode="json")
            )
        return {
            "ingestion": ingestion.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
            "series_plan": plan.model_dump(mode="json"),
            "episodes": episodes,
        }

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
