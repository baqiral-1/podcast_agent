"""Repair agent for weak or unsupported script segments."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import (
    ClaimAssessment,
    EpisodeScript,
    EpisodeSegment,
    GroundingReport,
    RepairResult,
    SegmentRepairResult,
)


class RepairAgent(Agent):
    """Agent that rewrites failing segments while preserving grounded claims."""

    schema_name = "episode_repair"
    instructions = "Repair only the failed script segments and keep the output JSON-valid."
    response_model = SegmentRepairResult

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

        return self.run(self.build_payload(script, report, attempt))

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
