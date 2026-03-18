"""Repair agent for weak or unsupported script segments."""

from __future__ import annotations

from pydantic import ValidationError

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


class RepairMappingError(RuntimeError):
    """Raised when the repair output cannot be mapped onto failed segments safely."""


class RepairAgent(Agent):
    """Agent that rewrites failing segments while preserving grounded claims."""

    schema_name = "episode_repair"
    instructions = (
        "Repair only the failed script segments.\n"
        "Return only JSON matching the provided schema.\n"
        "\n"
        "Output JSON shape (exact keys; no extras):\n"
        "{\n"
        '  "episode_id": "episode-2",\n'
        '  "attempt": 1,\n'
        '  "repaired_segment_ids": ["episode-2-segment-48"],\n'
        '  "repaired_segments": [\n'
        "    {\n"
        '      "heading": "Conclusion",\n'
        '      "narration": "Rewritten narration...",\n'
        '      "claims": [\n'
        "        {\n"
        '          "text": "A grounded factual claim...",\n'
        '          "evidence_chunk_ids": ["chunk-id-from-input"]\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Hard rules:\n"
        "- repaired_segment_ids must be chosen from failed_segment_ids.\n"
        "- repaired_segments must have the same length and ordering as repaired_segment_ids.\n"
        "- Each repaired segment may contain only: heading, narration, claims.\n"
        "- Each claim may contain only: text, evidence_chunk_ids.\n"
        "- evidence_chunk_ids must be chosen from the input failed_segments claims; do not invent chunk ids.\n"
        "- Forbidden keys: segment_id, beat_id, claim_id, citations, or any additional keys.\n"
        "You may leave a segment unchanged if it is already grounded."
    )
    response_model = SegmentRepairDraftResult

    def build_payload(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> dict:
        failed_segments = self.failed_segments(script, report)
        failed_segment_ids = [segment.segment_id for segment in failed_segments]
        claim_text_by_id = {
            claim.claim_id: claim.text
            for segment in failed_segments
            for claim in segment.claims
        }
        failed_claim_ids = set(claim_text_by_id)
        return {
            "episode_id": script.episode_id,
            "title": script.title,
            "narrator": script.narrator,
            "failed_segment_ids": failed_segment_ids,
            # Keep segment payloads id-free to avoid LLM echoing forbidden keys.
            # The segment_id mapping is positional: failed_segment_ids[i] -> failed_segments[i].
            "failed_segments": [self._segment_payload(segment) for segment in failed_segments],
            "report": {
                "episode_id": report.episode_id,
                "overall_status": report.overall_status,
                "claim_assessments": [
                    {
                        "text": claim_text_by_id.get(assessment.claim_id, ""),
                        "status": assessment.status.value,
                        "reason": assessment.reason,
                        "evidence_chunk_ids": list(assessment.evidence_chunk_ids),
                    }
                    for assessment in self.failed_claim_assessments(script, report)
                    if assessment.claim_id in failed_claim_ids
                ],
            },
            "attempt": attempt,
        }

    def repair(self, script: EpisodeScript, report: GroundingReport, attempt: int) -> SegmentRepairResult:
        """Repair failing script segments."""

        payload = self.build_payload(script, report, attempt)
        try:
            draft = self.run(payload)
            self._validate_repair_mapping(draft, payload["failed_segment_ids"])
        except (ValidationError, RepairMappingError) as exc:
            # Common failure modes:
            # - LLM echoes forbidden keys (ids/citations) -> schema ValidationError
            # - LLM returns invalid/dangerous repaired_segment_ids -> mapping error
            # Retry once with the exact error included to nudge correction.
            retry_instructions = f"{self.instructions}\nPrevious schema/mapping error: {exc}."
            draft = self.llm.generate_json(
                schema_name=self.schema_name,
                instructions=retry_instructions,
                payload=payload,
                response_model=self.response_model,
            )
            self._validate_repair_mapping(draft, payload["failed_segment_ids"])
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

    def _validate_repair_mapping(self, draft: SegmentRepairDraftResult, failed_segment_ids: list[str]) -> None:
        repaired_segment_ids = list(draft.repaired_segment_ids)
        if len(repaired_segment_ids) != len(draft.repaired_segments):
            raise RepairMappingError("repaired_segment_ids length does not match repaired_segments length")
        allowed_ids = set(failed_segment_ids)
        unknown_ids = [segment_id for segment_id in repaired_segment_ids if segment_id not in allowed_ids]
        if unknown_ids:
            raise RepairMappingError(f"Unknown repaired_segment_ids: {unknown_ids}")
        seen: set[str] = set()
        duplicates: list[str] = []
        for segment_id in repaired_segment_ids:
            if segment_id in seen and segment_id not in duplicates:
                duplicates.append(segment_id)
            seen.add(segment_id)
        if duplicates:
            raise RepairMappingError(f"Duplicate repaired_segment_ids: {duplicates}")

    def _segment_payload(self, segment: EpisodeSegment) -> dict:
        return {
            "heading": segment.heading,
            "narration": segment.narration,
            "claims": [
                {
                    "text": claim.text,
                    "evidence_chunk_ids": list(claim.evidence_chunk_ids),
                }
                for claim in segment.claims
            ],
        }

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
