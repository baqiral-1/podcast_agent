"""LLM-as-judge stage that scores the full assembled episode against the
six-criterion v66 audit rubric and points style_audit at the weakest sections.

The judge runs once per episode, after writing and before style_audit. It
returns diagnosis only — no prose rewrites. Its output is persisted as
``episodes/N/quality_judgment.json`` AND threaded into the style_audit
payload, where it directs the per-section editorial work.

Input scope is the full assembled episode (one call per episode). Four of
the six criteria — narrative_quality, listener_engagement, podcast_fidelity,
frame_discipline — can only be scored with full-arc context, so batching by
section would degrade the rubric.
"""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import quality_judge_instructions
from podcast_agent.schemas.models import EpisodeQualityScore


class QualityJudgeAgent(Agent):
    """Scores one assembled episode against the six-criterion rubric."""

    schema_name = "quality_judge"
    response_model = EpisodeQualityScore
    instructions = quality_judge_instructions()

    def build_instructions(self, payload: dict) -> str:
        base = quality_judge_instructions()
        # When `prepare_retry_payload` (below) has stashed field-level
        # validation errors from the previous attempt, render them as an
        # explicit RETRY FEEDBACK block so the LLM sees the exact loc/msg
        # tuples it must fix on this attempt.
        retry_feedback = payload.get("retry_feedback")
        if retry_feedback:
            return base + "\n\nRETRY FEEDBACK\n" + str(retry_feedback)
        return base

    def prepare_retry_payload(self, payload: dict, exc: Exception) -> dict:
        """Inject Pydantic field-level errors from a retryable failure into
        the next-attempt payload under ``retry_feedback``.

        Surfaces up to 8 field-level errors with their exact (loc, msg)
        tuple. The previous response's `weakest_criterion` failures now
        appear in the next prompt as e.g.
        ``- section_scores.3.weakest_criterion: weakest_criterion must be
        tied for the lowest score; got prose_polish at 88, minimum is 78
        on ['listener_engagement']``.
        """
        from pydantic import ValidationError

        # The retry harness wraps the underlying ValidationError in a
        # RetryableGenerationError; the original lives on ``__cause__``.
        real_exc: Exception | None = None
        if isinstance(exc, ValidationError):
            real_exc = exc
        else:
            cause = getattr(exc, "__cause__", None)
            if isinstance(cause, ValidationError):
                real_exc = cause
        if real_exc is None:
            return payload
        lines: list[str] = []
        for err in real_exc.errors()[:8]:
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = str(err.get("msg", "validation error"))
            lines.append(f"- `{loc}`: {msg}")
        total = len(real_exc.errors())
        suffix = f"\n(+{total - 8} more)" if total > 8 else ""
        new_payload = dict(payload)
        new_payload["retry_feedback"] = (
            "Your previous attempt was rejected by the response schema. "
            "Fix exactly these fields on this attempt:\n"
            + "\n".join(lines)
            + suffix
        )
        return new_payload

    def build_payload(
        self,
        *,
        episode_number: int,
        title: str,
        framing: dict,
        prose_sections: list[dict],
        architecture_summary: list[dict],
        excerpt_staging: list[dict],
        rubric_thresholds: dict,
        style_audit_lint_flags: dict,
        spine_diagnostics: dict | None = None,
        host_moves_diagnostics: dict | None = None,
        series_state: dict | None = None,
        prior_episode_remediation_hints: list[dict] | None = None,
    ) -> dict:
        payload: dict = {
            "episode_number": episode_number,
            "title": title,
            "framing": framing,
            "prose_sections": prose_sections,
            "architecture_summary": architecture_summary,
            "excerpt_staging": excerpt_staging,
            "thresholds": rubric_thresholds,
            "lint_flags": style_audit_lint_flags,
        }
        if spine_diagnostics is not None:
            payload["spine_diagnostics"] = spine_diagnostics
        if host_moves_diagnostics is not None:
            payload["host_moves_diagnostics"] = host_moves_diagnostics
        if series_state is not None:
            payload["series_state"] = series_state
        if prior_episode_remediation_hints:
            payload["prior_episode_remediation_hints"] = list(
                prior_episode_remediation_hints
            )
        return payload

    def validate_result(
        self,
        result: EpisodeQualityScore,
        payload: dict,
    ) -> EpisodeQualityScore:
        # Echo the calling episode_number to keep judgments self-consistent
        # when the orchestrator persists them per-episode.
        expected = payload.get("episode_number")
        if isinstance(expected, int) and result.episode_number != expected:
            raise ValueError(
                f"quality_judge returned episode_number={result.episode_number} "
                f"for episode {expected}"
            )
        return result
