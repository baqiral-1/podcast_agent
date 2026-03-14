"""Episode planning agent."""

from __future__ import annotations

from collections import Counter

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BookAnalysis, BookStructure, EpisodePlan, SeriesPlan


class EpisodePlanningAgent(Agent):
    """Agent that turns analysis into a hierarchical series plan."""

    schema_name = "series_plan"
    instructions = (
        "Create a hierarchical series and episode plan for a single-narrator podcast. "
        "Episodes should usually span multiple chapters, preserve the multi-chapter clusters from analysis, "
        "and assign enough contiguous source material to each episode to satisfy the requested minimum source-word budget "
        "when the book is large enough. Return a contiguous partition of the full chapter order, avoid sparse chunk "
        "selection, and keep all assigned chunks for each chosen chapter span. If the full book is shorter than the "
        "minimum source-word budget, return a single episode covering the entire book."
    )
    response_model = SeriesPlan

    def __init__(self, llm, minimum_source_words_per_episode: int = 50000) -> None:
        super().__init__(llm)
        self.minimum_source_words_per_episode = minimum_source_words_per_episode

    def build_payload(self, structure: BookStructure, analysis: BookAnalysis) -> dict:
        return {
            "structure": structure.model_dump(mode="python"),
            "analysis": analysis.model_dump(mode="python"),
            "minimum_source_words_per_episode": self.minimum_source_words_per_episode,
        }

    def plan(self, structure: BookStructure, analysis: BookAnalysis) -> SeriesPlan:
        """Run episode planning."""

        payload = self.build_payload(structure, analysis)
        plan = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=self.instructions,
            payload=payload,
            response_model=self.response_model,
        )
        plan = self._normalize_plan(plan, structure)
        violations = self._compliance_violations(plan, structure, analysis)
        self._log_planning_metrics(plan, structure, analysis, violations, retried=False)
        if not violations:
            return plan

        retry_instructions = (
            f"{self.instructions} The previous plan violated these constraints: "
            f"{'; '.join(violations)}. Reuse or merge the provided analysis episode clusters instead of "
            "re-splitting into chapter-level episodes."
        )
        retry_plan = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=retry_instructions,
            payload=payload,
            response_model=self.response_model,
        )
        retry_plan = self._normalize_plan(retry_plan, structure)
        retry_violations = self._compliance_violations(retry_plan, structure, analysis)
        self._log_planning_metrics(retry_plan, structure, analysis, retry_violations, retried=True)
        if retry_violations:
            raise RuntimeError(
                "Series plan violated episode-length constraints after retry: "
                + "; ".join(retry_violations)
            )
        return retry_plan

    def _compliance_violations(
        self,
        plan: SeriesPlan,
        structure: BookStructure,
        analysis: BookAnalysis,
    ) -> list[str]:
        chunk_word_counts = {
            chunk.chunk_id: len(chunk.text.split())
            for chunk in structure.chunks
        }
        violations: list[str] = []
        cluster_spans_multi_chapter = any(len(cluster.chapter_ids) > 1 for cluster in analysis.episode_clusters)
        chapter_level_episodes = [
            episode.episode_id for episode in plan.episodes if len(episode.chapter_ids) <= 1
        ]
        if cluster_spans_multi_chapter and chapter_level_episodes and len(plan.episodes) >= len(structure.chapters):
            violations.append("planner expanded multi-chapter analysis into chapter-level episodes")
        total_words = sum(chunk_word_counts.values())
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
        if total_words < self.minimum_source_words_per_episode:
            if len(plan.episodes) != 1:
                violations.append(
                    "planner should return exactly one episode when the full book is shorter than the minimum source-word budget"
                )
            return violations
        for episode in plan.episodes:
            episode_words = sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
            if episode_words < self.minimum_source_words_per_episode:
                violations.append(
                    f"{episode.episode_id} estimated at {episode_words} source words, below minimum {self.minimum_source_words_per_episode}"
                )
        return violations

    def _normalize_plan(self, plan: SeriesPlan, structure: BookStructure) -> SeriesPlan:
        chapter_order = {chapter.chapter_id: chapter.chapter_number for chapter in structure.chapters}
        chapter_to_chunks = {chapter.chapter_id: list(chapter.chunk_ids) for chapter in structure.chapters}
        chunk_themes = {
            chunk.chunk_id: chunk.themes
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
        normalized_episode_data = self._rebalance_episode_data(normalized_episode_data, chapter_word_counts)
        normalized_episodes = [
            EpisodePlan.model_validate(
                self._episode_data_to_payload(
                    episode_data,
                    chapter_to_chunks=chapter_to_chunks,
                    chapter_order=chapter_order,
                    chunk_themes=chunk_themes,
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
    ) -> list[dict]:
        if len(episodes) <= 1:
            return episodes
        total_words = sum(self._chapter_word_total(episode["chapter_ids"], chapter_word_counts) for episode in episodes)
        if total_words < self.minimum_source_words_per_episode:
            merged = self._merge_episode_group(episodes, episode_id=episodes[0]["episode_id"], title=episodes[0]["title"])
            merged["sequence"] = 1
            return [merged]
        normalized: list[dict] = []
        current_group: list[dict] = []
        current_words = 0
        for episode in episodes:
            current_group.append(episode)
            current_words += self._chapter_word_total(episode["chapter_ids"], chapter_word_counts)
            if current_words >= self.minimum_source_words_per_episode:
                normalized.append(
                    self._merge_episode_group(
                        current_group,
                        episode_id=current_group[0]["episode_id"],
                        title=current_group[0]["title"],
                    )
                )
                current_group = []
                current_words = 0
        if current_group:
            if normalized:
                normalized[-1] = self._merge_episode_group(
                    [normalized[-1], *current_group],
                    episode_id=normalized[-1]["episode_id"],
                    title=normalized[-1]["title"],
                )
            else:
                normalized.append(
                    self._merge_episode_group(
                        current_group,
                        episode_id=current_group[0]["episode_id"],
                        title=current_group[0]["title"],
                    )
                )
        for sequence, episode in enumerate(normalized, start=1):
            episode["sequence"] = sequence
        return normalized

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
        chunk_themes: dict[str, list[str]],
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
        episode["beats"] = self._build_beats(episode["episode_id"], chapter_ids, chapter_to_chunks)
        return episode

    def _build_beats(
        self,
        episode_id: str,
        chapter_ids: list[str],
        chapter_to_chunks: dict[str, list[str]],
    ) -> list[dict]:
        beats: list[dict] = []
        for sequence, chapter_id in enumerate(chapter_ids, start=1):
            chapter_number = chapter_id.rsplit("-", 1)[-1]
            chunk_ids = list(chapter_to_chunks.get(chapter_id, []))
            beats.append(
                {
                    "beat_id": f"{episode_id}-beat-{sequence}",
                    "title": f"Chapter {chapter_number}",
                    "objective": f"Synthesize the grounded material from chapter {chapter_number}.",
                    "chunk_ids": chunk_ids,
                    "claim_requirements": [],
                }
            )
        return beats

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

    def _log_planning_metrics(
        self,
        plan: SeriesPlan,
        structure: BookStructure,
        analysis: BookAnalysis,
        violations: list[str],
        retried: bool,
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
            minimum_source_words_per_episode=self.minimum_source_words_per_episode,
            analysis_cluster_count=len(analysis.episode_clusters),
            analysis_cluster_chapter_counts=[len(cluster.chapter_ids) for cluster in analysis.episode_clusters],
            episode_count=len(plan.episodes),
            episode_chapter_spans=[episode.chapter_ids for episode in plan.episodes],
            episode_chapter_counts=[len(episode.chapter_ids) for episode in plan.episodes],
            episode_source_word_estimates=[
                sum(chunk_word_counts.get(chunk_id, 0) for chunk_id in episode.chunk_ids)
                for episode in plan.episodes
            ],
            violations=violations,
        )
