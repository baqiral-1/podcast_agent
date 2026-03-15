"""Analysis agent for theme and episode-cluster extraction."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookAnalysis, BookStructure
from podcast_agent.utils.planning_payloads import build_structure_summary, payload_size_bytes


class AnalysisAgent(Agent):
    """Agent that extracts themes and candidate multi-chapter episode clusters."""

    schema_name = "book_analysis"
    instructions = (
        "Analyze the book and propose multi-chapter episode clusters. "
        "Treat the clusters as a coverage-preserving partition of the book for downstream planning: "
        "every chapter should be assigned to exactly one cluster, clusters should usually follow contiguous "
        "chapter spans, and the cluster chunk_ids should cover the full source material rather than only a few "
        "representative excerpts."
    )
    response_model = BookAnalysis

    def __init__(self, llm, max_payload_bytes: int = 500000) -> None:
        super().__init__(llm)
        self.max_payload_bytes = max_payload_bytes

    def build_payload(self, structure: BookStructure) -> dict:
        return {"structure": build_structure_summary(structure)}

    def analyze(self, structure: BookStructure) -> BookAnalysis:
        """Run book analysis."""

        payload = self.build_payload(structure)
        self._check_payload_size(payload, structure)
        analysis = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=self.instructions,
            payload=payload,
            response_model=self.response_model,
        )
        analysis = self._normalize_analysis(analysis, structure)
        violations = self._compliance_violations(analysis, structure)
        self._log_analysis_metrics(analysis, structure, violations, retried=False)
        if not violations:
            return analysis

        retry_instructions = (
            f"{self.instructions} The previous analysis violated these constraints: {'; '.join(violations)}. "
            "Correct the cluster partition so it covers the full book exactly once and preserves chapter order."
        )
        retry_analysis = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=retry_instructions,
            payload=payload,
            response_model=self.response_model,
        )
        retry_analysis = self._normalize_analysis(retry_analysis, structure)
        retry_violations = self._compliance_violations(retry_analysis, structure)
        self._log_analysis_metrics(retry_analysis, structure, retry_violations, retried=True)
        if retry_violations:
            raise RuntimeError(
                "Book analysis violated coverage constraints after retry: "
                + "; ".join(retry_violations)
            )
        return retry_analysis

    def _normalize_analysis(self, analysis: BookAnalysis, structure: BookStructure) -> BookAnalysis:
        chapter_to_chunks = {chapter.chapter_id: list(chapter.chunk_ids) for chapter in structure.chapters}
        chapter_to_themes = {
            chapter.chapter_id: _ordered_unique(
                theme
                for chunk in structure.chunks
                if chunk.chapter_id == chapter.chapter_id
                for theme in chunk.themes
            )
            for chapter in structure.chapters
        }
        normalized_clusters = []
        for cluster in analysis.episode_clusters:
            chapter_ids = list(cluster.chapter_ids)
            chunk_ids = list(cluster.chunk_ids) or [
                chunk_id
                for chapter_id in chapter_ids
                for chunk_id in chapter_to_chunks.get(chapter_id, [])
            ]
            themes = list(cluster.themes) or _ordered_unique(
                theme
                for chapter_id in chapter_ids
                for theme in chapter_to_themes.get(chapter_id, [])
            )[:4]
            normalized_clusters.append(
                {
                    **cluster.model_dump(mode="python"),
                    "chunk_ids": chunk_ids,
                    "themes": themes,
                }
            )
        return BookAnalysis.model_validate(
            {
                **analysis.model_dump(mode="python"),
                "episode_clusters": normalized_clusters,
            }
        )

    def _check_payload_size(self, payload: dict, structure: BookStructure) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        payload_bytes = payload_size_bytes(payload)
        if run_logger is not None:
            run_logger.log(
                "analysis_payload_diagnostics",
                chapter_count=len(structure.chapters),
                chunk_count=len(structure.chunks),
                payload_bytes=payload_bytes,
                max_payload_bytes=self.max_payload_bytes,
            )
        if payload_bytes > self.max_payload_bytes:
            raise RuntimeError(
                "Analysis payload exceeds the configured maximum size: "
                f"{payload_bytes} bytes > {self.max_payload_bytes} bytes."
            )

    def _compliance_violations(self, analysis: BookAnalysis, structure: BookStructure) -> list[str]:
        chapter_order = {chapter.chapter_id: chapter.chapter_number for chapter in structure.chapters}
        all_chapter_ids = set(chapter_order)
        all_chunk_ids = {chunk.chunk_id for chunk in structure.chunks}
        assigned_chapters = [
            chapter_id
            for cluster in analysis.episode_clusters
            for chapter_id in cluster.chapter_ids
        ]
        assigned_chunks = [
            chunk_id
            for cluster in analysis.episode_clusters
            for chunk_id in cluster.chunk_ids
        ]
        violations: list[str] = []

        missing_chapters = sorted(all_chapter_ids - set(assigned_chapters))
        if missing_chapters:
            violations.append(f"analysis omitted chapters: {', '.join(missing_chapters)}")
        duplicate_chapters = sorted({chapter_id for chapter_id in assigned_chapters if assigned_chapters.count(chapter_id) > 1})
        if duplicate_chapters:
            violations.append(f"analysis duplicated chapters: {', '.join(duplicate_chapters)}")

        missing_chunks = sorted(all_chunk_ids - set(assigned_chunks))
        if missing_chunks:
            violations.append(
                f"analysis assigned only {len(set(assigned_chunks))} of {len(all_chunk_ids)} chunks"
            )
        duplicate_chunks = sorted({chunk_id for chunk_id in assigned_chunks if assigned_chunks.count(chunk_id) > 1})
        if duplicate_chunks:
            violations.append(
                f"analysis duplicated chunks: {', '.join(duplicate_chunks[:5])}"
            )

        for cluster in analysis.episode_clusters:
            chapter_numbers = sorted(chapter_order[chapter_id] for chapter_id in cluster.chapter_ids if chapter_id in chapter_order)
            if chapter_numbers and chapter_numbers[-1] - chapter_numbers[0] + 1 != len(chapter_numbers):
                violations.append(f"{cluster.cluster_id} is not a contiguous chapter span")
            cluster_chapter_prefixes = {f"{chapter_id}-chunk-" for chapter_id in cluster.chapter_ids}
            invalid_chunk_ids = [
                chunk_id
                for chunk_id in cluster.chunk_ids
                if not any(chunk_id.startswith(prefix) for prefix in cluster_chapter_prefixes)
            ]
            if invalid_chunk_ids:
                violations.append(
                    f"{cluster.cluster_id} includes chunks outside its chapter span"
                )
        return violations

    def _log_analysis_metrics(
        self,
        analysis: BookAnalysis,
        structure: BookStructure,
        violations: list[str],
        retried: bool,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        all_chapter_ids = {chapter.chapter_id for chapter in structure.chapters}
        all_chunk_ids = {chunk.chunk_id for chunk in structure.chunks}
        assigned_chapters = {
            chapter_id
            for cluster in analysis.episode_clusters
            for chapter_id in cluster.chapter_ids
        }
        assigned_chunks = {
            chunk_id
            for cluster in analysis.episode_clusters
            for chunk_id in cluster.chunk_ids
        }
        run_logger.log(
            "analysis_diagnostics",
            retried=retried,
            cluster_count=len(analysis.episode_clusters),
            cluster_chapter_counts=[len(cluster.chapter_ids) for cluster in analysis.episode_clusters],
            covered_chapter_count=len(assigned_chapters),
            total_chapter_count=len(all_chapter_ids),
            covered_chunk_count=len(assigned_chunks),
            total_chunk_count=len(all_chunk_ids),
            missing_chapters=sorted(all_chapter_ids - assigned_chapters),
            missing_chunk_count=len(all_chunk_ids - assigned_chunks),
            violations=violations,
        )


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
