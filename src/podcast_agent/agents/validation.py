"""Stage 11: Grounding validation agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import GroundingReport


class GroundingValidationAgent(Agent):
    """Verifies that every claim is supported by cited passages and cross-book comparisons are fair."""

    schema_name = "grounding_validation"
    response_model = GroundingReport
    instructions = (
        "You are a fact-checker for a multi-book podcast. Perform two validation passes:\n\n"
        "Pass 1 — Claim-level grounding:\n"
        "For each citation in the script, check that the cited passage actually supports the claim. "
        "Classify each as SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or FABRICATED.\n\n"
        "Pass 2 — Cross-book validation:\n"
        "For claims referencing multiple books, check:\n"
        "- Attribution accuracy: Is the claim attributed to the correct book?\n"
        "- Comparison fairness: When comparing authors, does the script represent both fairly? "
        "Check for straw-manning, oversimplification, false equivalence.\n"
        "- Balance: Does the episode give fair representation to each book?\n\n"
        "Additional checks for narrative-first scripts:\n"
        "- Silent source coverage: For segments without attribution, every factual claim must "
        "still be supported by a citation.\n"
        "- Attribution moment accuracy: For segments with explicit disagreement, verify the "
        "disagreement is genuine between the named sources, not just a difference in emphasis.\n\n"
        "Flag any instance where an author's position is oversimplified, misattributed, or "
        "unfairly contrasted.\n\n"
        "Compute:\n"
        "- grounding_score: fraction of claims that are SUPPORTED or PARTIALLY_SUPPORTED\n"
        "- attribution_accuracy: fraction of cross-book claims correctly attributed\n"
        "- overall_status: PASSED if both scores >= thresholds, NEEDS_REPAIR if above 0.6, "
        "FAILED otherwise\n\n"
        "Return a JSON GroundingReport."
    )

    def build_payload(
        self,
        episode_number: int,
        script: dict,
        passages: dict[str, dict],
    ) -> dict:
        return {
            "episode_number": episode_number,
            "script": script,
            "cited_passages": passages,
        }
