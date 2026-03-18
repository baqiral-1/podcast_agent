"""Spoken-delivery agent for post-grounding narration cleanup."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from podcast_agent.agents.base import Agent
from podcast_agent.eval.rewrite_metrics import FidelityCheckResult, build_rewrite_metrics, check_fidelity
from podcast_agent.prompts.spoken_delivery import (
    build_spoken_delivery_instructions,
    build_spoken_delivery_retry_instructions,
)
from podcast_agent.schemas.models import (
    EpisodeScript,
    EpisodeSegment,
    RewriteMetrics,
    SpokenDeliveryAttemptResult,
    SpokenDeliveryResult,
    SpokenDeliverySegmentResult,
    SpokenEpisodeScript,
    SpokenSegment,
    SpokenSegmentRewriteDraft,
)


class SpokenDeliveryAgent(Agent):
    """Convert factual narration into clearer spoken-form delivery."""

    schema_name = "spoken_delivery_segment"
    instructions = ""
    response_model = SpokenSegmentRewriteDraft

    def __init__(
        self,
        llm,
        *,
        tone_preset: str = "educational_suspenseful",
        target_expansion_ratio: float = 1.1,
        max_expansion_ratio: float = 1.2,
        retry_enabled: bool = True,
        spoken_delivery_parallelism: int = 6,
    ) -> None:
        super().__init__(llm)
        self.tone_preset = tone_preset
        self.target_expansion_ratio = target_expansion_ratio
        self.max_expansion_ratio = max_expansion_ratio
        self.retry_enabled = retry_enabled
        self.spoken_delivery_parallelism = spoken_delivery_parallelism

    def build_payload(
        self,
        script: EpisodeScript,
        segment_index: int,
        previous_output: str | None = None,
    ) -> dict:
        segment = script.segments[segment_index]
        previous_segment = script.segments[segment_index - 1] if segment_index > 0 else None
        next_segment = script.segments[segment_index + 1] if segment_index + 1 < len(script.segments) else None
        payload = {
            "episode_id": script.episode_id,
            "episode_title": script.title,
            "current_segment": self._segment_context(segment),
            "previous_segment": self._segment_context(previous_segment),
            "next_segment": self._segment_context(next_segment),
        }
        if previous_output is not None:
            payload["previous_output"] = previous_output
        return payload

    def rewrite(
        self,
        script: EpisodeScript,
    ) -> tuple[SpokenEpisodeScript, SpokenDeliveryResult]:
        """Rewrite a factual script into spoken delivery while preserving structure."""

        if self.spoken_delivery_parallelism == 1 or len(script.segments) <= 1:
            return self._rewrite_serial(script)

        max_workers = min(len(script.segments), self.spoken_delivery_parallelism)
        rewritten: list[tuple[SpokenSegment, SpokenDeliverySegmentResult] | None] = [None] * len(script.segments)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._rewrite_segment, script, segment, segment_index): segment_index
                for segment_index, segment in enumerate(script.segments)
            }
            for future, segment_index in future_map.items():
                rewritten[segment_index] = future.result()

        spoken_segments: list[SpokenSegment] = []
        segment_results: list[SpokenDeliverySegmentResult] = []
        for result in rewritten:
            assert result is not None
            spoken_segment, segment_result = result
            spoken_segments.append(spoken_segment)
            segment_results.append(segment_result)
        return self._build_rewrite_result(script, spoken_segments, segment_results)

    def _rewrite_serial(
        self,
        script: EpisodeScript,
    ) -> tuple[SpokenEpisodeScript, SpokenDeliveryResult]:
        spoken_segments: list[SpokenSegment] = []
        segment_results: list[SpokenDeliverySegmentResult] = []
        for segment_index, segment in enumerate(script.segments):
            spoken_segment, segment_result = self._rewrite_segment(script, segment, segment_index)
            spoken_segments.append(spoken_segment)
            segment_results.append(segment_result)
        return self._build_rewrite_result(script, spoken_segments, segment_results)

    def _build_rewrite_result(
        self,
        script: EpisodeScript,
        spoken_segments: list[SpokenSegment],
        segment_results: list[SpokenDeliverySegmentResult],
    ) -> tuple[SpokenEpisodeScript, SpokenDeliveryResult]:
        spoken_script = SpokenEpisodeScript(
            episode_id=script.episode_id,
            title=script.title,
            narrator=script.narrator,
            segments=spoken_segments,
        )
        delivery_result = SpokenDeliveryResult(
            episode_id=script.episode_id,
            tone_preset=self.tone_preset,
            target_expansion_ratio=self.target_expansion_ratio,
            max_expansion_ratio=self.max_expansion_ratio,
            segments=segment_results,
        )
        return spoken_script, delivery_result

    def _rewrite_segment(
        self,
        script: EpisodeScript,
        segment: EpisodeSegment,
        segment_index: int,
    ) -> tuple[SpokenSegment, SpokenDeliverySegmentResult]:
        retry_applied = False
        fallback_used = False
        attempts: list[SpokenDeliveryAttemptResult] = []
        payload = self.build_payload(script, segment_index)
        draft = self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=build_spoken_delivery_instructions(
                tone_preset=self.tone_preset,
                target_expansion_ratio=self.target_expansion_ratio,
                max_expansion_ratio=self.max_expansion_ratio,
            ),
            payload=payload,
            response_model=self.response_model,
        )
        narration = draft.narration
        metrics = build_rewrite_metrics(segment.narration, narration)
        fidelity = check_fidelity(segment.narration, narration, check_paragraph_drift=False)
        attempts.append(self._build_attempt_result("initial", narration, metrics, fidelity))
        if metrics.expansion_ratio > self.max_expansion_ratio or not fidelity.passed:
            self._log_fidelity_failure(
                script,
                segment,
                attempt="initial",
                narration=narration,
                metrics=metrics,
                fidelity=fidelity,
            )

        if (metrics.expansion_ratio > self.max_expansion_ratio or not fidelity.passed) and self.retry_enabled:
            retry_applied = True
            retry_payload = self.build_payload(script, segment_index, previous_output=narration)
            retry_draft = self.llm.generate_json(
                schema_name=self.schema_name,
                instructions=build_spoken_delivery_retry_instructions(
                    tone_preset=self.tone_preset,
                    target_expansion_ratio=self.target_expansion_ratio,
                    max_expansion_ratio=self.max_expansion_ratio,
                ),
                payload=retry_payload,
                response_model=self.response_model,
            )
            narration = retry_draft.narration
            metrics = build_rewrite_metrics(segment.narration, narration)
            fidelity = check_fidelity(segment.narration, narration, check_paragraph_drift=False)
            attempts.append(self._build_attempt_result("retry", narration, metrics, fidelity))
            if metrics.expansion_ratio > self.max_expansion_ratio or not fidelity.passed:
                self._log_fidelity_failure(
                    script,
                    segment,
                    attempt="retry",
                    narration=narration,
                    metrics=metrics,
                    fidelity=fidelity,
                )

        if metrics.expansion_ratio > self.max_expansion_ratio or not fidelity.passed:
            fallback_used = True
            narration = segment.narration
            metrics = build_rewrite_metrics(segment.narration, narration)
            fidelity = check_fidelity(segment.narration, narration, check_paragraph_drift=False)
            attempts.append(self._build_attempt_result("fallback", narration, metrics, fidelity))

        spoken_segment = SpokenSegment(
            segment_id=segment.segment_id,
            beat_id=segment.beat_id,
            heading=segment.heading,
            narration=narration,
            source_word_count=metrics.source_word_count,
            spoken_word_count=metrics.spoken_word_count,
            expansion_ratio=metrics.expansion_ratio,
            retry_applied=retry_applied,
            fidelity_passed=fidelity.passed,
            fallback_used=fallback_used,
        )
        result = SpokenDeliverySegmentResult(
            segment_id=segment.segment_id,
            beat_id=segment.beat_id,
            metrics=metrics,
            retry_applied=retry_applied,
            fidelity_passed=fidelity.passed,
            fallback_used=fallback_used,
            missing_names=fidelity.missing_names,
            missing_numbers=fidelity.missing_numbers,
            attempts=attempts,
        )
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                "spoken_delivery_segment",
                episode_id=script.episode_id,
                segment_id=segment.segment_id,
                beat_id=segment.beat_id,
                retry_applied=retry_applied,
                fallback_used=fallback_used,
                expansion_ratio=metrics.expansion_ratio,
                source_word_count=metrics.source_word_count,
                spoken_word_count=metrics.spoken_word_count,
                missing_names=fidelity.missing_names,
                missing_numbers=fidelity.missing_numbers,
            )
        return spoken_segment, result

    def _build_attempt_result(
        self,
        attempt: str,
        narration: str,
        metrics: RewriteMetrics,
        fidelity: FidelityCheckResult,
    ) -> SpokenDeliveryAttemptResult:
        return SpokenDeliveryAttemptResult(
            attempt=attempt,
            narration=narration,
            metrics=metrics,
            fidelity_passed=fidelity.passed,
            missing_names=fidelity.missing_names,
            missing_numbers=fidelity.missing_numbers,
            source_paragraph_count=fidelity.source_paragraph_count,
            spoken_paragraph_count=fidelity.spoken_paragraph_count,
            failure_reasons=self._failure_reasons(metrics, fidelity),
        )

    def _failure_reasons(self, metrics: RewriteMetrics, fidelity: FidelityCheckResult) -> list[str]:
        reasons: list[str] = []
        if metrics.expansion_ratio > self.max_expansion_ratio:
            reasons.append("expansion_limit")
        if fidelity.missing_numbers:
            reasons.append("missing_numbers")
        if fidelity.missing_names:
            reasons.append("missing_names")
        return reasons

    def _log_fidelity_failure(
        self,
        script: EpisodeScript,
        segment: EpisodeSegment,
        *,
        attempt: str,
        narration: str,
        metrics: RewriteMetrics,
        fidelity: FidelityCheckResult,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        run_logger.log(
            "spoken_delivery_fidelity_failure",
            episode_id=script.episode_id,
            segment_id=segment.segment_id,
            beat_id=segment.beat_id,
            attempt=attempt,
            narration=narration,
            expansion_ratio=metrics.expansion_ratio,
            source_word_count=metrics.source_word_count,
            spoken_word_count=metrics.spoken_word_count,
            fidelity_passed=fidelity.passed,
            missing_names=fidelity.missing_names,
            missing_numbers=fidelity.missing_numbers,
            source_paragraph_count=fidelity.source_paragraph_count,
            spoken_paragraph_count=fidelity.spoken_paragraph_count,
            failure_reasons=self._failure_reasons(metrics, fidelity),
        )

    def _segment_context(self, segment: EpisodeSegment | None) -> dict | None:
        if segment is None:
            return None
        return {
            "segment_id": segment.segment_id,
            "beat_id": segment.beat_id,
            "heading": segment.heading,
            "narration": segment.narration,
        }
