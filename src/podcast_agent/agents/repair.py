"""Repair agent for weak or unsupported script segments."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import (
    ClaimAssessment,
    EpisodeScript,
    EpisodeSegment,
    EpisodeSegmentDraft,
    GroundingReport,
    RepairResult,
    ScriptClaim,
    SegmentRepairDraftResult,
    SegmentRepairResult,
)


class RepairAgent(Agent):
    """Agent that rewrites failing segments while preserving grounded claims."""

    schema_name = "episode_repair"
    instructions = "Repair only the failed script segments and keep the output JSON-valid."
    response_model = SegmentRepairDraftResult

    def build_payload(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> dict:
        failed_segments = self.failed_segments(script, report)
        failed_claim_ids = {
            claim.claim_id
            for segment in failed_segments
            for claim in segment.claims
        }
        return {
            "episode_id": script.episode_id,
            "title": script.title,
            "narrator": script.narrator,
            "failed_segments": [segment.model_dump(mode="python") for segment in failed_segments],
            "report": {
                "episode_id": report.episode_id,
                "overall_status": report.overall_status,
                "claim_assessments": [
                    assessment.model_dump(mode="python")
                    for assessment in self.failed_claim_assessments(script, report)
                    if assessment.claim_id in failed_claim_ids
                ],
            },
            "attempt": attempt,
        }

    def repair(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> SegmentRepairResult:
        """Repair failing script segments."""

        draft = self.run(self.build_payload(script, report, attempt))
        claim_sequence_map = self._claim_sequence_map(script)
        return SegmentRepairResult(
            episode_id=draft.episode_id,
            attempt=draft.attempt,
            repaired_segment_ids=list(draft.repaired_segment_ids),
            repaired_segments=[
                self._materialize_segment(
                    segment,
                    segment_id=repaired_segment_id,
                    beat_id=self._beat_id_for_failed_segment(script, repaired_segment_id),
                    claim_sequence_start=claim_sequence_map[repaired_segment_id],
                )
                for repaired_segment_id, segment in zip(draft.repaired_segment_ids, draft.repaired_segments, strict=True)
            ],
        )

    def failed_segments(self, script: EpisodeScript, report: GroundingReport) -> list[EpisodeSegment]:
        """Return only the segments whose claims failed grounding."""

        failed_segments: list[EpisodeSegment] = []
        for segment, assessments in self.segment_assessments(script, report):
            if any(assessment.status.value != "grounded" for assessment in assessments):
                failed_segments.append(segment)
        return failed_segments

    def failed_claim_assessments(
        self,
        script: EpisodeScript,
        report: GroundingReport,
    ) -> list[ClaimAssessment]:
        """Return claim assessments belonging to failed segments."""

        failed_assessments: list[ClaimAssessment] = []
        for _, assessments in self.segment_assessments(script, report):
            if any(assessment.status.value != "grounded" for assessment in assessments):
                failed_assessments.extend(assessments)
        return failed_assessments

    def segment_assessments(
        self,
        script: EpisodeScript,
        report: GroundingReport,
    ) -> list[tuple[EpisodeSegment, list[ClaimAssessment]]]:
        """Rebuild per-segment assessment groups from the flattened report ordering."""

        grouped: list[tuple[EpisodeSegment, list[ClaimAssessment]]] = []
        cursor = 0
        for segment in script.segments:
            claim_count = len(segment.claims)
            grouped.append((segment, report.claim_assessments[cursor : cursor + claim_count]))
            cursor += claim_count
        return grouped

    def _materialize_segment(
        self,
        draft: EpisodeSegmentDraft,
        *,
        segment_id: str,
        beat_id: str,
        claim_sequence_start: int,
    ) -> EpisodeSegment:
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
            segment_id=segment_id,
            beat_id=beat_id,
            heading=draft.heading,
            narration=draft.narration,
            claims=claims,
            citations=citations,
        )

    def _claim_sequence_map(self, script: EpisodeScript) -> dict[str, int]:
        claim_index: dict[str, int] = {}
        for segment in script.segments:
            if not segment.claims:
                continue
            claim_index[segment.segment_id] = int(segment.claims[0].claim_id.rsplit("-", 1)[-1])
        return claim_index

    def _beat_id_for_failed_segment(self, script: EpisodeScript, segment_id: str) -> str:
        for segment in script.segments:
            if segment.segment_id == segment_id:
                return segment.beat_id
        raise RuntimeError(f"Unknown repaired segment id: {segment_id}")
