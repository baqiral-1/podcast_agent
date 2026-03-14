"""Writing agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import BeatScript, EpisodeBeat, EpisodePlan, EpisodeScript, EpisodeSegment, RetrievalHit


class WritingAgent(Agent):
    """Agent that turns an episode plan into a cited script."""

    schema_name = "beat_script"
    instructions = (
        "Write a grounded single-narrator podcast script section for the provided beat with claim-level citations. "
        "Use only the provided beat and retrieval hits. "
        "Develop the beat into a substantial spoken section rather than summarizing it away, and keep the narration "
        "proportional to the assigned source volume."
    )
    response_model = BeatScript

    def __init__(
        self,
        llm,
        minimum_source_words_per_episode: int = 50000,
        spoken_words_per_minute: int = 130,
        beat_evidence_limit: int = 6,
        beat_parallelism: int = 4,
    ) -> None:
        super().__init__(llm)
        self.minimum_source_words_per_episode = minimum_source_words_per_episode
        self.spoken_words_per_minute = spoken_words_per_minute
        self.beat_evidence_limit = beat_evidence_limit
        self.beat_parallelism = beat_parallelism

    def build_payload(self, episode_plan: EpisodePlan, retrieval_hits: list[RetrievalHit]) -> dict:
        assigned_source_words = sum(len(hit.text.split()) for hit in retrieval_hits)
        return {
            "episode_plan": episode_plan.model_dump(mode="python"),
            "retrieval_hits": [hit.model_dump(mode="python") for hit in retrieval_hits],
            "minimum_source_words_per_episode": self.minimum_source_words_per_episode,
            "assigned_source_words": assigned_source_words,
            "target_script_words": self._target_script_words(assigned_source_words),
        }

    def write(self, episode_plan: EpisodePlan, retrieval_hits: list[RetrievalHit]) -> EpisodeScript:
        """Generate an episode script."""
        payload = self.build_payload(episode_plan, retrieval_hits)
        beat_payloads = self._build_beat_payloads(payload)
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                "writing_schedule",
                episode_id=episode_plan.episode_id,
                beat_count=len(beat_payloads),
                concurrency=self.beat_parallelism,
            )
        with ThreadPoolExecutor(max_workers=self.beat_parallelism) as executor:
            beat_scripts = list(executor.map(self._write_beat, beat_payloads))
        beat_order = {beat.beat_id: index for index, beat in enumerate(episode_plan.beats)}
        beat_scripts.sort(key=lambda beat_script: beat_order[beat_script.beat_id])
        merged_segments: list[EpisodeSegment] = []
        segment_sequence = 1
        for beat_script in beat_scripts:
            for segment in beat_script.segments:
                merged_segments.append(
                    segment.model_copy(
                        update={"segment_id": f"{episode_plan.episode_id}-segment-{segment_sequence}"}
                    )
                )
                segment_sequence += 1
        return EpisodeScript(
            episode_id=episode_plan.episode_id,
            title=episode_plan.title,
            narrator="Narrator",
            segments=merged_segments,
        )

    def _target_script_words(self, assigned_source_words: int) -> int:
        if assigned_source_words <= 300:
            return max(40, assigned_source_words // 2)
        if assigned_source_words <= 600:
            return max(120, assigned_source_words // 2)
        return max(300, min(6000, assigned_source_words // 3))

    def _compliance_violations(self, script: BeatScript, payload: dict) -> list[str]:
        violations: list[str] = []
        target_script_words = payload["target_script_words"]
        assigned_source_words = payload["assigned_source_words"]
        if not script.segments:
            violations.append(f"beat {payload['beat']['beat_id']} produced no segments")
        covered_beat_ids = {segment.beat_id for segment in script.segments}
        if payload["beat"]["beat_id"] not in covered_beat_ids:
            violations.append(f"script omitted beat: {payload['beat']['beat_id']}")
        script_word_count = sum(len(segment.narration.split()) for segment in script.segments)
        if script_word_count < target_script_words:
            violations.append(
                f"script contains only {script_word_count} words for {assigned_source_words} assigned source words; target is {target_script_words}"
            )
        retrieval_chunk_ids = {hit["chunk_id"] for hit in payload["retrieval_hits"]}
        cited_chunk_ids = {
            chunk_id
            for segment in script.segments
            for claim in segment.claims
            for chunk_id in claim.evidence_chunk_ids
        }
        if retrieval_chunk_ids and len(cited_chunk_ids) < max(1, int(len(retrieval_chunk_ids) * 0.6)):
            violations.append(
                f"script cites only {len(cited_chunk_ids)} of {len(retrieval_chunk_ids)} assigned chunks"
            )
        return violations

    def _log_writing_metrics(self, script: BeatScript, payload: dict, violations: list[str], retried: bool) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        run_logger.log(
            "writing_diagnostics",
            retried=retried,
            episode_id=payload["episode"]["episode_id"],
            beat_id=payload["beat"]["beat_id"],
            assigned_source_words=payload["assigned_source_words"],
            target_script_words=payload["target_script_words"],
            segment_count=len(script.segments),
            script_word_count=sum(len(segment.narration.split()) for segment in script.segments),
            beat_count=1,
            covered_beat_count=len({segment.beat_id for segment in script.segments}),
            cited_chunk_count=len({
                chunk_id
                for segment in script.segments
                for claim in segment.claims
                for chunk_id in claim.evidence_chunk_ids
            }),
            assigned_chunk_count=len(payload["retrieval_hits"]),
            violations=violations,
        )

    def _build_beat_payloads(self, payload: dict) -> list[dict]:
        episode = payload["episode_plan"]
        retrieval_hit_map = {
            hit["chunk_id"]: RetrievalHit.model_validate(hit) for hit in payload["retrieval_hits"]
        }
        beat_word_targets = self._allocate_beat_targets(episode["beats"], retrieval_hit_map, payload["target_script_words"])
        beat_payloads = []
        for index, beat in enumerate(episode["beats"]):
            beat_hits = self._select_retrieval_hits_for_beat(beat, retrieval_hit_map)
            beat_assigned_source_words = sum(len(hit.text.split()) for hit in beat_hits)
            previous_title = episode["beats"][index - 1]["title"] if index > 0 else ""
            beat_payloads.append(
                {
                    "episode": {
                        "episode_id": episode["episode_id"],
                        "title": episode["title"],
                        "themes": episode["themes"],
                    },
                    "beat": beat,
                    "continuity_hint": previous_title,
                    "retrieval_hits": [hit.model_dump(mode="python") for hit in beat_hits],
                    "assigned_source_words": beat_assigned_source_words,
                    "target_script_words": beat_word_targets[beat["beat_id"]],
                }
            )
        return beat_payloads

    def _select_retrieval_hits_for_beat(
        self,
        beat: dict | EpisodeBeat,
        retrieval_hit_map: dict[str, RetrievalHit],
    ) -> list[RetrievalHit]:
        beat_chunk_ids = beat["chunk_ids"] if isinstance(beat, dict) else beat.chunk_ids
        selected_hits = [retrieval_hit_map[chunk_id] for chunk_id in beat_chunk_ids if chunk_id in retrieval_hit_map]
        return selected_hits[: self.beat_evidence_limit]

    def _allocate_beat_targets(
        self,
        beats: list[dict],
        retrieval_hit_map: dict[str, RetrievalHit],
        episode_target_script_words: int,
    ) -> dict[str, int]:
        relaxed_episode_target = max(20, episode_target_script_words // 2)
        beat_source_words = {
            beat["beat_id"]: sum(
                len(retrieval_hit_map[chunk_id].text.split())
                for chunk_id in beat["chunk_ids"]
                if chunk_id in retrieval_hit_map
            )
            for beat in beats
        }
        total_source_words = sum(beat_source_words.values())
        if total_source_words <= 0:
            return {beat["beat_id"]: max(20, relaxed_episode_target // max(1, len(beats))) for beat in beats}
        allocations: dict[str, int] = {}
        assigned_total = 0
        for beat in beats:
            beat_id = beat["beat_id"]
            proportional = int((beat_source_words[beat_id] / total_source_words) * relaxed_episode_target)
            allocated = min(
                beat_source_words[beat_id],
                max(20, proportional),
            )
            allocations[beat_id] = allocated
            assigned_total += allocated
        remainder = relaxed_episode_target - assigned_total
        for beat in reversed(beats):
            if remainder == 0:
                break
            beat_id = beat["beat_id"]
            beat_limit = beat_source_words[beat_id]
            if remainder > 0 and allocations[beat_id] < beat_limit:
                allocations[beat_id] += 1
                remainder -= 1
            elif remainder < 0 and allocations[beat_id] > 20:
                allocations[beat_id] -= 1
                remainder += 1
        return allocations

    def _write_beat(self, payload: dict) -> BeatScript:
        run_logger = getattr(self.llm, "run_logger", None)
        beat_id = payload["beat"]["beat_id"]
        episode_id = payload["episode"]["episode_id"]
        if run_logger is not None:
            run_logger.log(
                "beat_write_started",
                episode_id=episode_id,
                beat_id=beat_id,
                selected_chunk_count=len(payload["retrieval_hits"]),
            )
        beat_script = self.run(payload)
        violations = self._compliance_violations(beat_script, payload)
        self._log_writing_metrics(beat_script, payload, violations, retried=False)
        if not violations:
            if run_logger is not None:
                run_logger.log("beat_write_completed", episode_id=episode_id, beat_id=beat_id, retried=False)
            return beat_script
        retry_instructions = (
            f"{self.instructions} The previous beat script violated these constraints: {'; '.join(violations)}. "
            "Rewrite only this beat so the narration is substantially longer and grounded in most of the provided retrieval hits."
        )
        retry_script = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=retry_instructions,
            payload=payload,
            response_model=self.response_model,
        )
        retry_violations = self._compliance_violations(retry_script, payload)
        self._log_writing_metrics(retry_script, payload, retry_violations, retried=True)
        if retry_violations:
            raise RuntimeError(
                f"Beat script violated coverage constraints after retry for {beat_id}: "
                + "; ".join(retry_violations)
            )
        if run_logger is not None:
            run_logger.log("beat_write_completed", episode_id=episode_id, beat_id=beat_id, retried=True)
        return retry_script
