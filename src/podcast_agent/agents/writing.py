"""Writing agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError

from podcast_agent.agents.base import Agent
from podcast_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from podcast_agent.schemas.models import (
    BeatScript,
    BeatScriptDraft,
    EpisodeBeat,
    EpisodePlan,
    EpisodeScript,
    EpisodeSegment,
    EpisodeSegmentDraft,
    RetrievalHit,
    ScriptClaim,
)


class WritingAgent(Agent):
    """Agent that turns an episode plan into a cited script."""

    schema_name = "beat_script"
    instructions = (
        "Write a grounded single-narrator podcast script section for the provided beat. "
        "Use only the provided beat and retrieval hits. Write spoken narration, not notes or analysis. "
        "Develop the beat fully enough to reflect the assigned source material, but stay concise and proportional to the assigned source volume. "
        "Coverage requirements: cover the assigned retrieval hits comprehensively. "
        "If the beat has 7 or fewer assigned chunk_ids, every assigned chunk_id must appear in at least one claim's evidence_chunk_ids. "
        "Do not focus only on early chunks if later chunks contain distinct material. "
        "Use 1 or 2 segments unless more are clearly necessary. "
        "Grounding rules:"
        "1. Do not turn suspicion, rumor, belief, allegation, or interpretation into fact."
        "2. Do not infer motive, causality, or significance unless the cited evidence states it explicitly."
        "Hard rules: "
        "1. Every segment must contain at least one claim. "
        "2. Every claim must include one or more evidence_chunk_ids taken only from the assigned beat retrieval hits. "
        "3. Do not invent, rename, or omit assigned chunk ids in claim evidence. "
        "4. Cover later assigned chunks when they contain distinct material."
    )
    response_model = BeatScriptDraft

    def __init__(
        self,
        llm,
        minimum_source_words_per_episode: int = 50000,
        spoken_words_per_minute: int = 130,
        coverage_warning_min_ratio: float | None = None,
        beat_parallelism: int = 6,
        beat_write_retry_attempts: int = 2,
        beat_write_timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(llm)
        self.minimum_source_words_per_episode = minimum_source_words_per_episode
        self.spoken_words_per_minute = spoken_words_per_minute
        self.coverage_warning_min_ratio = coverage_warning_min_ratio
        self.beat_parallelism = beat_parallelism
        self.beat_write_retry_attempts = beat_write_retry_attempts
        self.beat_write_timeout_seconds = beat_write_timeout_seconds

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
                beat_write_retry_attempts=self.beat_write_retry_attempts,
                beat_write_timeout_seconds=self.beat_write_timeout_seconds,
                assigned_chunk_count=len(retrieval_hits),
                beat_chunk_counts=[len(payload["retrieval_hits"]) for payload in beat_payloads],
            )
        future_to_payload = {}
        beat_script_map: dict[str, BeatScript] = {}
        executor = ThreadPoolExecutor(max_workers=self.beat_parallelism)
        try:
            future_to_payload = {
                executor.submit(self._write_beat, beat_payload): beat_payload for beat_payload in beat_payloads
            }
            for future in as_completed(future_to_payload):
                beat_payload = future_to_payload[future]
                beat_id = beat_payload["beat"]["beat_id"]
                try:
                    beat_script_map[beat_id] = future.result()
                except Exception as exc:
                    for pending in future_to_payload:
                        if pending is not future:
                            pending.cancel()
                    raise RuntimeError(f"Beat writing failed for {beat_id}: {exc}") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        beat_scripts = [beat_script_map[beat.beat_id] for beat in episode_plan.beats]
        beat_order = {beat.beat_id: index for index, beat in enumerate(episode_plan.beats)}
        beat_scripts.sort(key=lambda beat_script: beat_order[beat_script.beat_id])
        merged_segments: list[EpisodeSegment] = []
        segment_sequence = 1
        for beat_script in beat_scripts:
            for segment in beat_script.segments:
                merged_segments.append(
                    segment.model_copy(update={"segment_id": f"{episode_plan.episode_id}-segment-{segment_sequence}"})
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
        assigned_chunk_ids = set(self._assigned_chunk_ids(payload))
        if not script.segments:
            violations.append(f"beat {payload['beat']['beat_id']} produced no segments")
        for segment in script.segments:
            if not segment.claims:
                violations.append(f"segment {segment.segment_id} has no claims")
            claim_chunk_ids = {
                chunk_id
                for claim in segment.claims
                for chunk_id in claim.evidence_chunk_ids
            }
            if not segment.citations:
                violations.append(f"segment {segment.segment_id} has no citations")
            if segment.citations and claim_chunk_ids and set(segment.citations) != claim_chunk_ids:
                violations.append(
                    f"segment {segment.segment_id} citations do not match claim evidence ids"
                )
            for claim in segment.claims:
                if not claim.evidence_chunk_ids:
                    violations.append(f"claim {claim.claim_id} has no evidence chunk ids")
                invalid_chunk_ids = [
                    chunk_id for chunk_id in claim.evidence_chunk_ids if chunk_id not in assigned_chunk_ids
                ]
                if invalid_chunk_ids:
                    violations.append(
                        f"claim {claim.claim_id} cites invalid chunk ids: {', '.join(invalid_chunk_ids)}"
                    )
        script_word_count = sum(len(segment.narration.split()) for segment in script.segments)
        if script_word_count < target_script_words:
            violations.append(
                f"script contains only {script_word_count} words for {assigned_source_words} assigned source words; target is {target_script_words}"
            )
        retrieval_chunk_ids = assigned_chunk_ids
        claim_evidence_chunk_ids = set(self._claim_evidence_chunk_ids(script))
        if retrieval_chunk_ids and len(claim_evidence_chunk_ids) < max(1, int(len(retrieval_chunk_ids) * 0.6)):
            violations.append(
                f"script cites only {len(claim_evidence_chunk_ids)} of {len(retrieval_chunk_ids)} assigned chunks"
            )
        return violations

    def _log_writing_metrics(self, script: BeatScript, payload: dict, violations: list[str], retried: bool) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        claim_evidence_chunk_ids = self._claim_evidence_chunk_ids(script)
        segment_citation_chunk_ids = self._segment_citation_chunk_ids(script)
        cited_chunk_ids = sorted(set(claim_evidence_chunk_ids) | set(segment_citation_chunk_ids))
        assigned_chunk_ids = self._assigned_chunk_ids(payload)
        missing_assigned_chunk_ids = [
            chunk_id for chunk_id in assigned_chunk_ids if chunk_id not in cited_chunk_ids
        ]
        extra_cited_chunk_ids = [
            chunk_id for chunk_id in cited_chunk_ids if chunk_id not in set(assigned_chunk_ids)
        ]
        claims_with_zero_evidence = [
            claim.claim_id
            for segment in script.segments
            for claim in segment.claims
            if not claim.evidence_chunk_ids
        ]
        segments_with_zero_citations = [
            segment.segment_id
            for segment in script.segments
            if not segment.citations
        ]
        run_logger.log(
            "writing_diagnostics",
            retried=retried,
            episode_id=payload["episode"]["episode_id"],
            beat_id=payload["beat"]["beat_id"],
            assigned_source_words=payload["assigned_source_words"],
            target_script_words=payload["target_script_words"],
            segment_count=len(script.segments),
            claim_count=sum(len(segment.claims) for segment in script.segments),
            script_word_count=sum(len(segment.narration.split()) for segment in script.segments),
            beat_count=1,
            covered_beat_count=len({segment.beat_id for segment in script.segments}),
            cited_chunk_count=len(cited_chunk_ids),
            claim_cited_chunk_count=len(set(claim_evidence_chunk_ids)),
            segment_citation_chunk_count=len(set(segment_citation_chunk_ids)),
            assigned_chunk_count=len(payload["retrieval_hits"]),
            claim_to_evidence_coverage_ratio=(
                len(set(claim_evidence_chunk_ids)) / len(assigned_chunk_ids) if assigned_chunk_ids else 0.0
            ),
            segment_citation_coverage_ratio=(
                len(set(segment_citation_chunk_ids)) / len(assigned_chunk_ids) if assigned_chunk_ids else 0.0
            ),
            uncited_chunk_count=len(missing_assigned_chunk_ids),
            dropped_chunk_count=len(payload["beat"]["chunk_ids"]) - len(payload["retrieval_hits"]),
            assigned_chunk_ids=assigned_chunk_ids,
            claim_evidence_chunk_ids=claim_evidence_chunk_ids,
            segment_citation_chunk_ids=segment_citation_chunk_ids,
            missing_assigned_chunk_ids=missing_assigned_chunk_ids,
            extra_cited_chunk_ids=extra_cited_chunk_ids,
            claims_with_zero_evidence=claims_with_zero_evidence,
            segments_with_zero_citations=segments_with_zero_citations,
            citations_derived=True,
            violations=violations,
        )
        run_logger.log(
            "citation_audit",
            stage="writing",
            episode_id=payload["episode"]["episode_id"],
            beat_id=payload["beat"]["beat_id"],
            assigned_chunk_ids=assigned_chunk_ids,
            claim_evidence_chunk_ids=claim_evidence_chunk_ids,
            segment_citation_chunk_ids=segment_citation_chunk_ids,
            missing_assigned_chunk_ids=missing_assigned_chunk_ids,
            extra_cited_chunk_ids=extra_cited_chunk_ids,
            claims_with_zero_evidence=claims_with_zero_evidence,
            segments_with_zero_citations=segments_with_zero_citations,
            citations_derived=True,
        )
        if self.coverage_warning_min_ratio is None:
            return
        assigned_count = len(assigned_chunk_ids)
        if assigned_count == 0:
            return
        cited_count = len(set(claim_evidence_chunk_ids))
        cited_ratio = cited_count / assigned_count
        if cited_ratio < self.coverage_warning_min_ratio:
            run_logger.log(
                "writing_coverage_warning",
                episode_id=payload["episode"]["episode_id"],
                beat_id=payload["beat"]["beat_id"],
                cited_ratio=cited_ratio,
                minimum_ratio=self.coverage_warning_min_ratio,
                uncited_chunk_ids=missing_assigned_chunk_ids,
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
        return [retrieval_hit_map[chunk_id] for chunk_id in beat_chunk_ids if chunk_id in retrieval_hit_map]

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
                assigned_chunk_count=len(payload["beat"]["chunk_ids"]),
                total_attempts=self.beat_write_retry_attempts + 1,
            )
        last_error: Exception | None = None
        last_violations: list[str] = []
        retry_instructions = self.instructions
        previous_script: BeatScript | None = None
        total_attempts = self.beat_write_retry_attempts + 1
        attempt = 1
        for attempt in range(1, total_attempts + 1):
            retried = attempt > 1
            try:
                beat_script = self._generate_beat_script(
                    schema_name=self.schema_name,
                    instructions=retry_instructions,
                    payload=payload,
                    response_model=self.response_model,
                )
            except (ValidationError, JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if run_logger is not None:
                    run_logger.log(
                        "beat_write_retry",
                        episode_id=episode_id,
                        beat_id=beat_id,
                        retried=retried,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        attempt=attempt,
                        total_attempts=total_attempts,
                    )
                if attempt == total_attempts:
                    break
                retry_instructions = self._build_retry_instructions(
                    payload=payload,
                    violations=[],
                    missing_chunk_ids=[],
                    cited_without_claim_chunk_ids=[],
                    last_error=exc,
                    previous_script=previous_script,
                )
                continue
            violations = self._compliance_violations(beat_script, payload)
            self._log_writing_metrics(beat_script, payload, violations, retried=retried)
            if not violations:
                if run_logger is not None:
                    run_logger.log(
                        "beat_write_completed",
                        episode_id=episode_id,
                        beat_id=beat_id,
                        retried=retried,
                        attempt=attempt,
                        total_attempts=total_attempts,
                    )
                return beat_script
            previous_script = beat_script
            last_violations = violations
            if attempt == total_attempts:
                break
            retry_instructions = self._build_retry_instructions(
                payload=payload,
                violations=violations,
                missing_chunk_ids=self._uncited_chunk_ids(beat_script, payload),
                cited_without_claim_chunk_ids=self._cited_without_claim_evidence_chunk_ids(beat_script),
                last_error=None,
                previous_script=previous_script,
            )
            continue
        if run_logger is not None:
            self._log_terminal_beat_failure(
                run_logger=run_logger,
                episode_id=episode_id,
                beat_id=beat_id,
                attempt=attempt,
                last_error=last_error,
                last_violations=last_violations,
            )
        if last_error is not None and not last_violations:
            raise RuntimeError(
                f"Beat script generation failed after retry for {beat_id}: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"Beat script violated coverage constraints after retry for {beat_id}: "
            + "; ".join(last_violations)
        )

    def _build_retry_instructions(
        self,
        *,
        payload: dict,
        violations: list[str],
        missing_chunk_ids: list[str],
        cited_without_claim_chunk_ids: list[str],
        last_error: Exception | None,
        previous_script: BeatScript | None,
    ) -> str:
        assigned_chunk_ids = self._assigned_chunk_ids(payload)
        coverage_instructions = [
            "The previous beat script was invalid or violated constraints.",
            "Rewrite only this beat.",
            "Every segment must contain one or more claims.",
            "Every claim must include non-empty evidence_chunk_ids chosen from the assigned beat retrieval hits.",
            "Do not concentrate evidence in the first few chunks when later assigned chunks contain distinct material.",
            "Keep the response compact and JSON-only.",
            "Use at most two segments unless one segment would be clearly insufficient.",
            "Revise and expand the previous draft instead of restarting from a blank outline.",
            "Preserve valid narration from the previous draft while fixing missing claim evidence.",
            f"Keep the narration at or above {payload['target_script_words']} words.",
            "Keep narration concise and avoid repeating or quoting source text at length.",
            "When evidence is partial or ambiguous, prefer omission or weaker wording over a more vivid summary.",
        ]
        if len(assigned_chunk_ids) <= 7:
            coverage_instructions.append(
                "This beat has 7 or fewer assigned chunks, so every assigned chunk_id must appear in claim evidence_chunk_ids."
            )
        if violations:
            coverage_instructions.append(
                "Previous violations: " + "; ".join(violations) + "."
            )
        if missing_chunk_ids:
            coverage_instructions.append(
                "You must explicitly cover these missing chunk ids by attaching each one to at least one claim's evidence_chunk_ids: "
                + ", ".join(missing_chunk_ids)
                + "."
            )
            coverage_instructions.append(
                "A missing chunk is not covered if it appears only in narration without being attached to a claim."
            )
        if cited_without_claim_chunk_ids:
            coverage_instructions.append(
                "These chunk ids were cited in claims inconsistently across the previous draft. "
                "For each one, either add it to a claim's evidence_chunk_ids or remove it from the rewritten draft: "
                + ", ".join(cited_without_claim_chunk_ids)
                + "."
            )
        if last_error is not None:
            if "truncated because it hit the completion token limit" in str(last_error):
                coverage_instructions.append(
                    "The previous response was truncated. Reduce output length substantially and avoid enumerating unnecessary details."
                )
            coverage_instructions.append(
                f"Previous schema/generation error: {last_error}."
            )
        if previous_script is not None:
            previous_word_count = sum(len(segment.narration.split()) for segment in previous_script.segments)
            coverage_instructions.append(
                f"The previous draft had {previous_word_count} narration words. Do not return fewer than {payload['target_script_words']} words."
            )
        return f"{self.instructions} {' '.join(coverage_instructions)}"

    def _generate_beat_script(
        self,
        *,
        schema_name: str,
        instructions: str,
        payload: dict,
        response_model: type[BeatScript],
    ) -> BeatScript:
        llm = self.llm
        if isinstance(self.llm, OpenAICompatibleLLMClient):
            llm = OpenAICompatibleLLMClient(
                self.llm.config.model_copy(update={"timeout_seconds": self.beat_write_timeout_seconds}),
                transport=self.llm.transport,
            )
            if getattr(self.llm, "run_logger", None) is not None:
                llm.set_run_logger(self.llm.run_logger)
        beat_script_draft = llm.generate_json(
            schema_name=schema_name,
            instructions=instructions,
            payload=payload,
            response_model=response_model,
        )
        return self._materialize_beat_script(beat_script_draft, payload["beat"]["beat_id"])

    def _log_terminal_beat_failure(
        self,
        *,
        run_logger,
        episode_id: str,
        beat_id: str,
        attempt: int,
        last_error: Exception | None,
        last_violations: list[str],
    ) -> None:
        event_type = "beat_write_failed"
        error_type = type(last_error).__name__ if last_error is not None else "RuntimeError"
        error_message = (
            str(last_error)
            if last_error is not None
            else "Beat script violated coverage constraints after retry: " + "; ".join(last_violations)
        )
        if last_error is not None and "timed out" in str(last_error).lower():
            event_type = "beat_write_timeout"
        elif last_error is not None and "truncated because it hit the completion token limit" in str(last_error):
            event_type = "beat_write_truncated"
        run_logger.log(
            event_type,
            episode_id=episode_id,
            beat_id=beat_id,
            attempt=attempt,
            error_type=error_type,
            error_message=error_message,
        )

    def _uncited_chunk_ids(self, script: BeatScript, payload: dict) -> list[str]:
        cited_chunk_ids = set(self._claim_evidence_chunk_ids(script)) | set(
            self._segment_citation_chunk_ids(script)
        )
        return [
            hit["chunk_id"]
            for hit in payload["retrieval_hits"]
            if hit["chunk_id"] not in cited_chunk_ids
        ]

    def build_citation_audit(
        self,
        episode_plan: EpisodePlan,
        script: EpisodeScript,
        retrieval_hits: list[RetrievalHit],
    ) -> dict[str, Any]:
        retrieval_hit_map = {hit.chunk_id: hit for hit in retrieval_hits}
        beat_segments: dict[str, list[EpisodeSegment]] = {}
        for segment in script.segments:
            beat_segments.setdefault(segment.beat_id, []).append(segment)
        beat_audits = []
        for beat in episode_plan.beats:
            segments = beat_segments.get(beat.beat_id, [])
            claim_evidence_chunk_ids = sorted(
                {
                    chunk_id
                    for segment in segments
                    for claim in segment.claims
                    for chunk_id in claim.evidence_chunk_ids
                }
            )
            segment_citation_chunk_ids = sorted(
                {chunk_id for segment in segments for chunk_id in segment.citations}
            )
            assigned_chunk_ids = list(beat.chunk_ids)
            beat_audits.append(
                {
                    "beat_id": beat.beat_id,
                    "beat_title": beat.title,
                    "assigned_chunk_ids": assigned_chunk_ids,
                    "assigned_chunks": [
                        {
                            "chunk_id": chunk_id,
                            "chapter_title": retrieval_hit_map[chunk_id].chapter_title,
                            "excerpt": retrieval_hit_map[chunk_id].text[:240],
                        }
                        for chunk_id in assigned_chunk_ids
                        if chunk_id in retrieval_hit_map
                    ],
                    "claim_evidence_chunk_ids": claim_evidence_chunk_ids,
                    "segment_citation_chunk_ids": segment_citation_chunk_ids,
                    "missing_assigned_chunk_ids": [
                        chunk_id
                        for chunk_id in assigned_chunk_ids
                        if chunk_id not in set(claim_evidence_chunk_ids) | set(segment_citation_chunk_ids)
                    ],
                    "extra_cited_chunk_ids": [
                        chunk_id
                        for chunk_id in set(claim_evidence_chunk_ids) | set(segment_citation_chunk_ids)
                        if chunk_id not in set(assigned_chunk_ids)
                    ],
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "text": claim.text,
                            "evidence_chunk_ids": list(claim.evidence_chunk_ids),
                        }
                        for segment in segments
                        for claim in segment.claims
                    ],
                    "segments": [
                        {
                            "segment_id": segment.segment_id,
                            "heading": segment.heading,
                            "citations": list(segment.citations),
                            "narration_excerpt": segment.narration[:240],
                        }
                        for segment in segments
                    ],
                }
            )
        return {
            "episode_id": episode_plan.episode_id,
            "episode_title": episode_plan.title,
            "beat_audits": beat_audits,
        }

    def _assigned_chunk_ids(self, payload: dict) -> list[str]:
        return [hit["chunk_id"] for hit in payload["retrieval_hits"]]

    def _claim_evidence_chunk_ids(self, script: BeatScript) -> list[str]:
        return sorted(
            {
                chunk_id
                for segment in script.segments
                for claim in segment.claims
                for chunk_id in claim.evidence_chunk_ids
            }
        )

    def _segment_citation_chunk_ids(self, script: BeatScript) -> list[str]:
        return sorted(
            {
                chunk_id
                for segment in script.segments
                for chunk_id in segment.citations
            }
        )

    def _cited_without_claim_evidence_chunk_ids(self, script: BeatScript) -> list[str]:
        claim_chunk_ids = set(self._claim_evidence_chunk_ids(script))
        return [
            chunk_id
            for chunk_id in self._segment_citation_chunk_ids(script)
            if chunk_id not in claim_chunk_ids
        ]

    def _materialize_beat_script(self, draft: BeatScriptDraft, beat_id: str) -> BeatScript:
        claim_sequence = 1
        segments: list[EpisodeSegment] = []
        for segment_index, segment in enumerate(draft.segments, start=1):
            materialized_segment, claim_sequence = self._materialize_segment(
                segment,
                beat_id=beat_id,
                segment_index=segment_index,
                claim_sequence_start=claim_sequence,
            )
            segments.append(materialized_segment)
        return BeatScript(
            beat_id=beat_id,
            segments=segments,
        )

    def _materialize_segment(
        self,
        draft: EpisodeSegmentDraft,
        *,
        beat_id: str,
        segment_index: int,
        claim_sequence_start: int,
    ) -> tuple[EpisodeSegment, int]:
        claims: list[ScriptClaim] = []
        claim_sequence = claim_sequence_start
        for claim in draft.claims:
            claims.append(
                ScriptClaim(
                    claim_id=f"{beat_id}-claim-{claim_sequence}",
                    text=claim.text,
                    evidence_chunk_ids=list(claim.evidence_chunk_ids),
                )
            )
            claim_sequence += 1
        citations = sorted(
            {
                chunk_id
                for claim in claims
                for chunk_id in claim.evidence_chunk_ids
            }
        )
        return EpisodeSegment(
            segment_id=f"{beat_id}-segment-{segment_index}",
            beat_id=beat_id,
            heading=draft.heading,
            narration=draft.narration,
            claims=claims,
            citations=citations,
        ), claim_sequence
