"""Episode planning agent."""

from __future__ import annotations

from collections import Counter
from json import JSONDecodeError

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import (
    BookAnalysis,
    BookStructure,
    EpisodePlan,
    PlannerSeriesPlan,
    SeriesPlan,
)
from podcast_agent.utils.planning_payloads import (
    build_analysis_summary,
    build_structure_summary,
    payload_size_bytes,
)


class EpisodePlanningAgent(Agent):
    """Agent that turns analysis into a hierarchical series plan."""

    schema_name = "series_plan"
    instructions = (
        "Create a hierarchical series and episode plan for a single-narrator podcast. "
        "Episodes should preserve the multi-chapter clusters from analysis and return exactly the requested episode count. "
        "Return a contiguous partition of the full chapter order, choose episode spans by chapter_ids only, and distribute "
        "source material as evenly as practical across the requested episodes. Chunk coverage will be expanded deterministically "
        "from those chapter spans."
    )
    response_model = PlannerSeriesPlan

    def __init__(
        self,
        llm,
        minimum_source_words_per_episode: int = 50000,
        min_episode_source_ratio: float = 0.3,
        spoken_words_per_minute: int = 130,
        max_episode_minutes: int = 58,
        max_payload_bytes: int = 500000,
        max_payload_bytes_with_episode_count: int | None = None,
        section_beat_target_words: int = 1200,
        beat_evidence_window_size: int = 8,
    ) -> None:
        super().__init__(llm)
        self.minimum_source_words_per_episode = minimum_source_words_per_episode
        self.min_episode_source_ratio = min_episode_source_ratio
        self.spoken_words_per_minute = spoken_words_per_minute
        self.max_episode_minutes = max_episode_minutes
        self.max_payload_bytes = max_payload_bytes
        self.max_payload_bytes_with_episode_count = max_payload_bytes_with_episode_count or max_payload_bytes
        self.section_beat_target_words = section_beat_target_words
        self.beat_evidence_window_size = beat_evidence_window_size

    def build_payload(self, structure: BookStructure, analysis: BookAnalysis, episode_count: int) -> dict:
        target_source_words_per_episode = self._target_source_words_per_episode(structure, episode_count)
        return {
            "structure": build_structure_summary(structure),
            "analysis": build_analysis_summary(analysis),
            "minimum_source_words_per_episode": self.minimum_source_words_per_episode,
            "min_episode_source_ratio": self.min_episode_source_ratio,
            "episode_count": episode_count,
            "target_source_words_per_episode": target_source_words_per_episode,
            "max_episode_minutes": self.max_episode_minutes,
            "max_episode_script_words": self._max_episode_script_words(),
        }

    def plan(self, structure: BookStructure, analysis: BookAnalysis, episode_count: int) -> SeriesPlan:
        """Run episode planning."""

        if episode_count > len(structure.chapters):
            raise RuntimeError(
                f"Requested {episode_count} episodes, but only {len(structure.chapters)} chapters are available."
            )
        payload = self.build_payload(structure, analysis, episode_count)
        self._check_payload_size(payload, structure, analysis, episode_count)
        compact_plan = self._generate_compact_plan(payload=payload, episode_count=episode_count)
        plan = self._normalize_plan(compact_plan, structure, episode_count)
        violations = self._compliance_violations(plan, structure, analysis, episode_count)
        self._log_planning_metrics(plan, structure, analysis, violations, retried=False, episode_count=episode_count)
        if not violations:
            return plan

        retry_instructions = (
            f"{self.instructions} The previous plan violated these constraints: "
            f"{'; '.join(violations)}. Reuse or merge the provided analysis episode clusters instead of "
            f"re-splitting into chapter-level episodes, and return exactly {episode_count} episodes. "
            "Return only episode metadata and chapter_ids; do not include chunk_ids or beats."
        )
        retry_compact_plan = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=retry_instructions,
            payload=payload,
            response_model=self.response_model,
        )
        retry_plan = self._normalize_plan(retry_compact_plan, structure, episode_count)
        retry_violations = self._compliance_violations(retry_plan, structure, analysis, episode_count)
        self._log_planning_metrics(
            retry_plan,
            structure,
            analysis,
            retry_violations,
            retried=True,
            episode_count=episode_count,
        )
        if retry_violations:
            raise RuntimeError(
                "Series plan violated episode-length constraints after retry: "
                + "; ".join(retry_violations)
            )
        return retry_plan

    def _generate_compact_plan(self, *, payload: dict, episode_count: int) -> PlannerSeriesPlan:
        run_logger = getattr(self.llm, "run_logger", None)
        retry_instructions = self.instructions
        last_error: Exception | None = None
        for retried in (False, True):
            try:
                return self.llm.generate_json(
                    schema_name=self.schema_name,
                    instructions=retry_instructions,
                    payload=payload,
                    response_model=self.response_model,
                )
            except (ValidationError, JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if run_logger is not None:
                    run_logger.log(
                        "planning_retry",
                        retried=retried,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                if retried:
                    break
                retry_instructions = self._build_retry_instructions(
                    payload=payload,
                    episode_count=episode_count,
                    last_error=exc,
                )
        raise RuntimeError(f"Series plan generation failed after retry: {last_error}") from last_error

    def _build_retry_instructions(
        self,
        *,
        payload: dict,
        episode_count: int,
        last_error: Exception,
    ) -> str:
        correction_instructions = [
            "The previous series plan was invalid or incomplete.",
            "Return the compact planner output only.",
            f"Set book_id to {payload['structure']['book_id']}.",
            f"Return exactly {episode_count} episodes.",
            "Use chapter_ids as the authoritative contiguous partition of the selected chapters.",
            "Return only episode metadata and chapter_ids; do not include chunk_ids or beats.",
            "Keep strategy_summary, title, synopsis, and themes concise.",
        ]
        if "truncated because it hit the completion token limit" in str(last_error):
            correction_instructions.append(
                "The previous response was truncated. Reduce output length substantially and avoid unnecessary detail."
            )
        correction_instructions.append(f"Previous schema/generation error: {last_error}.")
        return f"{self.instructions} {' '.join(correction_instructions)}"

    def _compliance_violations(
        self,
        plan: SeriesPlan,
        structure: BookStructure,
        analysis: BookAnalysis,
        episode_count: int,
    ) -> list[str]:
        chunk_word_counts = {
            chunk.chunk_id: len(chunk.text.split())
            for chunk in structure.chunks
        }
        violations: list[str] = []
        if len(plan.episodes) != episode_count:
            violations.append(
                f"planner returned {len(plan.episodes)} episodes instead of required {episode_count}"
            )
        cluster_spans_multi_chapter = any(len(cluster.chapter_ids) > 1 for cluster in analysis.episode_clusters)
        chapter_level_episodes = [
            episode.episode_id for episode in plan.episodes if len(episode.chapter_ids) <= 1
        ]
        if cluster_spans_multi_chapter and chapter_level_episodes and len(plan.episodes) >= len(structure.chapters):
            violations.append("planner expanded multi-chapter analysis into chapter-level episodes")
        total_words = sum(chunk_word_counts.values())
        target_source_words_per_episode = self._target_source_words_per_episode(structure, episode_count)
        min_episode_source_words = max(1, int(target_source_words_per_episode * self.min_episode_source_ratio))
        all_chapter_ids = {chapter.chapter_id for chapter in structure.chapters}
        assigned_chapter_ids = [chapter_id for episode in plan.episodes for chapter_id in episode.chapter_ids]
        missing_chapters = sorted(all_chapter_ids - set(assigned_chapter_ids))
        if missing_chapters:
            violations.append(f"planner omitted chapters: {', '.join(missing_chapters)}")
        duplicate_chapters = sorted({
            chapter_id for chapter_id in assigned_chapter_ids if assigned_chapter_ids.count(chapter_id) > 1
        })
        if duplicate_chapters:
            violations.append(f"planner duplicated chapters: {', '.join(duplicate_chapters)}")
        assigned_words = sum(
            chunk_word_counts.get(chunk_id, 0)
            for episode in plan.episodes
            for chunk_id in episode.chunk_ids
        )
        if assigned_words < total_words * 0.8:
            violations.append(
                f"planner assigned only {assigned_words} of {total_words} available words"
            )
        for episode in plan.episodes:
            episode_words = sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
            if episode_words < min_episode_source_words:
                violations.append(
                    f"{episode.episode_id} estimated at {episode_words} source words, too small for target {target_source_words_per_episode}"
                )
            estimated_script_words = self._estimate_script_words_from_source_words(episode_words)
            estimated_minutes = self._estimate_episode_minutes(estimated_script_words)
            if estimated_minutes > self.max_episode_minutes:
                violations.append(
                    f"{episode.episode_id} estimated at {estimated_minutes:.1f} spoken minutes, above max {self.max_episode_minutes}; increase episode_count"
                )
        return violations

    def _normalize_plan(self, plan: PlannerSeriesPlan | SeriesPlan, structure: BookStructure, episode_count: int) -> SeriesPlan:
        chapter_order = {chapter.chapter_id: chapter.chapter_number for chapter in structure.chapters}
        chapter_to_chunks = {chapter.chapter_id: list(chapter.chunk_ids) for chapter in structure.chapters}
        chapter_titles = {chapter.chapter_id: chapter.title for chapter in structure.chapters}
        chunk_themes = {
            chunk.chunk_id: chunk.themes
            for chunk in structure.chunks
        }
        chunk_word_counts = {
            chunk.chunk_id: len(chunk.text.split())
            for chunk in structure.chunks
        }
        chapter_word_counts = {
            chapter.chapter_id: sum(
                len(chunk.text.split())
                for chunk in structure.chunks
                if chunk.chapter_id == chapter.chapter_id
            )
            for chapter in structure.chapters
        }
        normalized_episode_data = [
            self._normalize_episode_data(episode, chapter_to_chunks, chapter_order, chunk_themes)
            for episode in sorted(plan.episodes, key=lambda episode: episode.sequence)
        ]
        normalized_episode_data = self._rebalance_episode_data(
            normalized_episode_data,
            chapter_word_counts,
            chapter_order,
            episode_count,
        )
        normalized_episodes = [
            EpisodePlan.model_validate(
                self._episode_data_to_payload(
                    episode_data,
                    chapter_to_chunks=chapter_to_chunks,
                    chapter_order=chapter_order,
                    chapter_titles=chapter_titles,
                    chunk_themes=chunk_themes,
                    chunk_word_counts=chunk_word_counts,
                )
            )
            for episode_data in normalized_episode_data
        ]
        return SeriesPlan.model_validate(
            {
                **plan.model_dump(mode="python"),
                "episodes": [episode.model_dump(mode="python") for episode in normalized_episodes],
            }
        )

    def _normalize_episode_data(
        self,
        episode: EpisodePlan,
        chapter_to_chunks: dict[str, list[str]],
        chapter_order: dict[str, int],
        chunk_themes: dict[str, list[str]],
    ) -> dict:
        chapter_ids = sorted(episode.chapter_ids, key=lambda chapter_id: chapter_order[chapter_id])
        chunk_ids = [
            chunk_id
            for chapter_id in chapter_ids
            for chunk_id in chapter_to_chunks.get(chapter_id, [])
        ]
        themes = self._themes_for_chunks(chunk_ids, chunk_themes)
        return {
            "episode_id": episode.episode_id,
            "sequence": episode.sequence,
            "title": episode.title,
            "synopsis": episode.synopsis,
            "chapter_ids": chapter_ids,
            "chunk_ids": chunk_ids,
            "themes": themes or episode.themes,
        }

    def _rebalance_episode_data(
        self,
        episodes: list[dict],
        chapter_word_counts: dict[str, int],
        chapter_order: dict[str, int],
        episode_count: int,
    ) -> list[dict]:
        chapter_ids = sorted(chapter_word_counts, key=lambda chapter_id: chapter_order[chapter_id])
        if episode_count < 1:
            raise RuntimeError("episode_count must be at least 1")
        if episode_count > len(chapter_ids):
            raise RuntimeError(
                f"Requested {episode_count} episodes, but only {len(chapter_ids)} chapters are available for contiguous partitioning."
            )
        target_words = self._target_source_words_for_chapter_counts(chapter_word_counts, chapter_ids, episode_count)
        partitioned_chapter_ids = self._partition_chapters(chapter_ids, chapter_word_counts, episode_count, target_words)
        sequence_to_episode = {
            episode["sequence"]: episode
            for episode in episodes
        }
        normalized: list[dict] = []
        for sequence, chapter_group in enumerate(partitioned_chapter_ids, start=1):
            template = sequence_to_episode.get(sequence) or sequence_to_episode.get(min(sequence_to_episode, default=1), {
                "episode_id": f"episode-{sequence}",
                "title": f"Episode {sequence}",
                "synopsis": "",
                "themes": [],
                "chunk_ids": [],
            })
            chunk_ids = [
                chunk_id
                for chapter_id in chapter_group
                for chunk_id in template.get("chunk_ids", [])
                if chunk_id.startswith(f"{chapter_id}-chunk-")
            ]
            normalized.append(
                {
                    "episode_id": f"episode-{sequence}",
                    "sequence": sequence,
                    "title": template["title"],
                    "synopsis": template.get("synopsis", ""),
                    "chapter_ids": chapter_group,
                    "chunk_ids": chunk_ids,
                    "themes": template.get("themes", []),
                }
            )
        return normalized

    def _partition_chapters(
        self,
        chapter_ids: list[str],
        chapter_word_counts: dict[str, int],
        episode_count: int,
        target_words: int,
    ) -> list[list[str]]:
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_words = 0
        remaining_groups = episode_count
        remaining_chapters = len(chapter_ids)
        for chapter_id in chapter_ids:
            current_group.append(chapter_id)
            current_words += chapter_word_counts[chapter_id]
            remaining_chapters -= 1
            groups_needed_after_current = remaining_groups - 1
            if groups_needed_after_current == 0:
                continue
            if remaining_chapters == groups_needed_after_current:
                groups.append(current_group)
                current_group = []
                current_words = 0
                remaining_groups -= 1
                continue
            if current_words >= target_words:
                groups.append(current_group)
                current_group = []
                current_words = 0
                remaining_groups -= 1
        if current_group:
            groups.append(current_group)
        return groups

    def _merge_episode_group(
        self,
        episodes: list[dict],
        *,
        episode_id: str,
        title: str,
    ) -> dict:
        chapter_ids = [chapter_id for episode in episodes for chapter_id in episode["chapter_ids"]]
        chunk_ids = [chunk_id for episode in episodes for chunk_id in episode["chunk_ids"]]
        themes = [theme for episode in episodes for theme in episode["themes"]]
        synopses = [episode["synopsis"] for episode in episodes if episode["synopsis"]]
        return {
            "episode_id": episode_id,
            "sequence": episodes[0]["sequence"],
            "title": title,
            "synopsis": " ".join(dict.fromkeys(synopses)),
            "chapter_ids": chapter_ids,
            "chunk_ids": chunk_ids,
            "themes": [theme for theme, _ in Counter(themes).most_common(4)],
        }

    def _episode_data_to_payload(
        self,
        episode: dict,
        *,
        chapter_to_chunks: dict[str, list[str]],
        chapter_order: dict[str, int],
        chapter_titles: dict[str, str],
        chunk_themes: dict[str, list[str]],
        chunk_word_counts: dict[str, int],
    ) -> dict:
        chapter_ids = sorted(episode["chapter_ids"], key=lambda chapter_id: chapter_order[chapter_id])
        chunk_ids = [
            chunk_id
            for chapter_id in chapter_ids
            for chunk_id in chapter_to_chunks.get(chapter_id, [])
        ]
        episode["chapter_ids"] = chapter_ids
        episode["chunk_ids"] = chunk_ids
        episode["themes"] = self._themes_for_chunks(chunk_ids, chunk_themes) or episode["themes"]
        episode["beats"] = self._build_beats(
            episode["episode_id"],
            chapter_ids,
            chapter_to_chunks,
            chapter_titles,
            chunk_word_counts,
        )
        return episode

    def _build_beats(
        self,
        episode_id: str,
        chapter_ids: list[str],
        chapter_to_chunks: dict[str, list[str]],
        chapter_titles: dict[str, str],
        chunk_word_counts: dict[str, int],
    ) -> list[dict]:
        beats: list[dict] = []
        beat_sequence = 1
        for chapter_id in chapter_ids:
            chapter_chunk_ids = list(chapter_to_chunks.get(chapter_id, []))
            chapter_title = chapter_titles.get(chapter_id, chapter_id)
            groups = self._group_chapter_chunks(chapter_chunk_ids, chunk_word_counts)
            for section_index, group_chunk_ids in enumerate(groups, start=1):
                section_label = chapter_title if len(groups) == 1 else f"{chapter_title} (Section {section_index})"
                beats.append(
                    {
                        "beat_id": f"{episode_id}-beat-{beat_sequence}",
                        "title": section_label,
                        "objective": f"Synthesize the grounded material from {section_label.lower()}.",
                        "chunk_ids": group_chunk_ids,
                        "claim_requirements": [],
                    }
                )
                beat_sequence += 1
        return beats

    def _group_chapter_chunks(
        self,
        chunk_ids: list[str],
        chunk_word_counts: dict[str, int],
    ) -> list[list[str]]:
        if not chunk_ids:
            return []
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_words = 0
        for chunk_id in chunk_ids:
            chunk_words = chunk_word_counts.get(chunk_id, 0)
            if (
                current_group
                and (
                    current_words >= self.section_beat_target_words
                    or len(current_group) >= self.beat_evidence_window_size
                )
            ):
                groups.append(current_group)
                current_group = []
                current_words = 0
            current_group.append(chunk_id)
            current_words += chunk_words
        if current_group:
            groups.append(current_group)
        return groups

    def _themes_for_chunks(
        self,
        chunk_ids: list[str],
        chunk_themes: dict[str, list[str]],
    ) -> list[str]:
        counts: Counter[str] = Counter()
        for chunk_id in chunk_ids:
            counts.update(chunk_themes.get(chunk_id, []))
        return [theme for theme, _ in counts.most_common(4)]

    def _chapter_word_total(self, chapter_ids: list[str], chapter_word_counts: dict[str, int]) -> int:
        return sum(chapter_word_counts[chapter_id] for chapter_id in chapter_ids)

    def _target_source_words_per_episode(self, structure: BookStructure, episode_count: int) -> int:
        chapter_word_counts = {
            chapter.chapter_id: sum(
                len(chunk.text.split())
                for chunk in structure.chunks
                if chunk.chapter_id == chapter.chapter_id
            )
            for chapter in structure.chapters
        }
        chapter_ids = [chapter.chapter_id for chapter in structure.chapters]
        return self._target_source_words_for_chapter_counts(chapter_word_counts, chapter_ids, episode_count)

    def _target_source_words_for_chapter_counts(
        self,
        chapter_word_counts: dict[str, int],
        chapter_ids: list[str],
        episode_count: int,
    ) -> int:
        total_words = sum(chapter_word_counts[chapter_id] for chapter_id in chapter_ids)
        return max(1, (total_words + episode_count - 1) // episode_count)

    def _max_episode_script_words(self) -> int:
        return self.max_episode_minutes * self.spoken_words_per_minute

    def _estimate_script_words_from_source_words(self, source_words: int) -> int:
        return max(1, source_words // 3)

    def _estimate_episode_minutes(self, script_words: int) -> float:
        return script_words / self.spoken_words_per_minute

    def _log_planning_metrics(
        self,
        plan: SeriesPlan,
        structure: BookStructure,
        analysis: BookAnalysis,
        violations: list[str],
        retried: bool,
        *,
        episode_count: int,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        chunk_word_counts = {
            chunk.chunk_id: len(chunk.text.split())
            for chunk in structure.chunks
        }
        run_logger.log(
            "planning_diagnostics",
            retried=retried,
            requested_episode_count=episode_count,
            minimum_source_words_per_episode=self.minimum_source_words_per_episode,
            min_episode_source_ratio=self.min_episode_source_ratio,
            target_source_words_per_episode=self._target_source_words_per_episode(structure, episode_count),
            max_episode_minutes=self.max_episode_minutes,
            max_episode_script_words=self._max_episode_script_words(),
            section_beat_target_words=self.section_beat_target_words,
            beat_evidence_window_size=self.beat_evidence_window_size,
            analysis_cluster_count=len(analysis.episode_clusters),
            analysis_cluster_chapter_counts=[len(cluster.chapter_ids) for cluster in analysis.episode_clusters],
            episode_count=len(plan.episodes),
            episode_chapter_spans=[episode.chapter_ids for episode in plan.episodes],
            episode_chapter_counts=[len(episode.chapter_ids) for episode in plan.episodes],
            episode_source_word_estimates=[
                sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
                for episode in plan.episodes
            ],
            episode_script_word_estimates=[
                self._estimate_script_words_from_source_words(
                    sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
                )
                for episode in plan.episodes
            ],
            episode_minute_estimates=[
                self._estimate_episode_minutes(
                    self._estimate_script_words_from_source_words(
                        sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
                    )
                )
                for episode in plan.episodes
            ],
            episode_beat_counts=[len(episode.beats) for episode in plan.episodes],
            beat_assigned_chunk_counts=[
                [len(beat.chunk_ids) for beat in episode.beats]
                for episode in plan.episodes
            ],
            oversized_episode_ids=[
                episode.episode_id
                for episode in plan.episodes
                if self._estimate_episode_minutes(
                    self._estimate_script_words_from_source_words(
                        sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
                    )
                ) > self.max_episode_minutes
            ],
            violations=violations,
        )

    def _check_payload_size(
        self,
        payload: dict,
        structure: BookStructure,
        analysis: BookAnalysis,
        episode_count: int,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        payload_bytes = payload_size_bytes(payload)
        max_payload_bytes = self.max_payload_bytes_with_episode_count
        if run_logger is not None:
            run_logger.log(
                "planning_payload_diagnostics",
                chapter_count=len(structure.chapters),
                chunk_count=len(structure.chunks),
                analysis_cluster_count=len(analysis.episode_clusters),
                requested_episode_count=episode_count,
                payload_bytes=payload_bytes,
                max_payload_bytes=max_payload_bytes,
            )
        if payload_bytes > max_payload_bytes:
            raise RuntimeError(
                "Planning payload exceeds the configured maximum size: "
                f"{payload_bytes} bytes > {max_payload_bytes} bytes."
            )
