"""Grounding validation agent."""

from __future__ import annotations

from typing import Any

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import EpisodeScript, GroundingReport, RetrievalHit


class GroundingValidationAgent(Agent):
    """Agent that validates script claims against retrieved evidence."""

    schema_name = "grounding_report"
    instructions = "Validate each claim against the cited evidence and classify grounding strength."
    response_model = GroundingReport

    def build_payload(self, script: EpisodeScript, retrieval_hits: list[RetrievalHit]) -> dict:
        return {
            "script": script.model_dump(mode="python"),
            "retrieval_hits": [hit.model_dump(mode="python") for hit in retrieval_hits],
        }

    def validate(self, script: EpisodeScript, retrieval_hits: list[RetrievalHit]) -> GroundingReport:
        """Produce a claim-level grounding report."""

        report = self.run(self.build_payload(script, retrieval_hits))
        self._log_validation_metrics(script, retrieval_hits, report)
        return report

    def build_citation_audit(self, script: EpisodeScript, report: GroundingReport) -> dict[str, Any]:
        claim_map = {
            claim.claim_id: claim
            for segment in script.segments
            for claim in segment.claims
        }
        return {
            "episode_id": script.episode_id,
            "claim_assessments": [
                {
                    "claim_id": assessment.claim_id,
                    "status": assessment.status.value,
                    "reason": assessment.reason,
                    "script_evidence_chunk_ids": claim_map.get(assessment.claim_id).evidence_chunk_ids
                    if assessment.claim_id in claim_map
                    else [],
                    "validator_evidence_chunk_ids": list(assessment.evidence_chunk_ids),
                    "validator_ignored_script_evidence": [
                        chunk_id
                        for chunk_id in claim_map.get(assessment.claim_id).evidence_chunk_ids
                        if chunk_id not in set(assessment.evidence_chunk_ids)
                    ]
                    if assessment.claim_id in claim_map
                    else [],
                }
                for assessment in report.claim_assessments
            ],
        }

    def _log_validation_metrics(
        self,
        script: EpisodeScript,
        retrieval_hits: list[RetrievalHit],
        report: GroundingReport,
    ) -> None:
        run_logger = getattr(self.llm, "run_logger", None)
        if run_logger is None:
            return
        claim_map = {
            claim.claim_id: claim
            for segment in script.segments
            for claim in segment.claims
        }
        status_counts = {
            "grounded_claim_count": 0,
            "weak_claim_count": 0,
            "unsupported_claim_count": 0,
            "conflicting_claim_count": 0,
        }
        unsupported_claim_ids: list[str] = []
        conflicting_claim_ids: list[str] = []
        claims_with_ignored_script_evidence: list[str] = []
        validator_evidence_chunk_ids = sorted(
            {
                chunk_id
                for assessment in report.claim_assessments
                for chunk_id in assessment.evidence_chunk_ids
            }
        )
        for assessment in report.claim_assessments:
            status_counts[f"{assessment.status.value}_claim_count"] += 1
            if assessment.status.value == "unsupported":
                unsupported_claim_ids.append(assessment.claim_id)
            if assessment.status.value == "conflicting":
                conflicting_claim_ids.append(assessment.claim_id)
            script_evidence_chunk_ids = set(
                claim_map.get(assessment.claim_id).evidence_chunk_ids
                if assessment.claim_id in claim_map
                else []
            )
            if script_evidence_chunk_ids - set(assessment.evidence_chunk_ids):
                claims_with_ignored_script_evidence.append(assessment.claim_id)
        run_logger.log(
            "validation_diagnostics",
            episode_id=script.episode_id,
            overall_status=report.overall_status,
            claim_count=len(report.claim_assessments),
            retrieval_hit_count=len(retrieval_hits),
            cited_chunk_ids=sorted({hit.chunk_id for hit in retrieval_hits}),
            validated_evidence_chunk_ids=validator_evidence_chunk_ids,
            unsupported_claim_ids=unsupported_claim_ids,
            conflicting_claim_ids=conflicting_claim_ids,
            claims_with_ignored_script_evidence=claims_with_ignored_script_evidence,
            **status_counts,
        )
