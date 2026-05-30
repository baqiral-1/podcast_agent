"""Shared test helpers for building valid section_progression payloads.

The schema refactor made ``section_progression`` a required field on
``ArchitectureSection`` and replaced the old ``closure_mode`` /
``answer_section_id`` / ``residue_section_id`` pointers with a stage-based
progression contract enforced by ``EpisodeArchitecture.validate_architecture``.

These helpers let test fixtures assign valid stages without duplicating the
nested ``section_progression`` payload everywhere.
"""

from __future__ import annotations

from typing import Any, Iterable

_VALID_STAGES = {"setup", "advance", "answer", "afterpressure", "close"}


def make_section_progression(
    stage: str,
    *,
    state_effects: dict[str, Any] | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Return a valid ``section_progression`` payload for ``stage``."""

    if stage not in _VALID_STAGES:
        raise ValueError(f"invalid progression stage: {stage}")
    suffix = f" ({label})" if label else ""
    is_close = stage == "close"
    return {
        "stage": stage,
        "becomes_obvious": (f"The {stage} step makes the next move in the answer visible{suffix}."),
        "answer_contribution": (
            f"This section advances the episode answer at the {stage} stage{suffix}."
        ),
        "what_remains_live": (
            "The episode exits without reopening the answer."
            if is_close
            else f"This section hands its live pressure forward{suffix}."
        ),
        "state_effects": dict(state_effects or {}),
    }


def stages_for_purposes(purposes: Iterable[str]) -> list[str]:
    """Derive a valid stage layout from a list of section ``purpose`` values.

    Rules enforced by the model validator:
      * exactly one ``answer`` stage (non-opening purpose),
      * exactly one ``close`` stage, on the final section, aligned with
        ``purpose == 'closing'``,
      * ``afterpressure`` only strictly between answer and close.

    Strategy: the final section gets ``close``; the section immediately before
    it gets ``answer``; the first section gets ``setup``; everything else gets
    ``advance``. Any ``closing`` purpose that is not the final section is left
    as ``advance`` (callers should not produce such layouts).
    """

    purpose_list = list(purposes)
    count = len(purpose_list)
    if count < 2:
        raise ValueError("need at least two sections for a valid stage layout")
    stages = ["advance"] * count
    stages[0] = "setup"
    stages[-1] = "close"
    stages[-2] = "answer"
    return stages


def attach_section_progressions(
    sections: list[dict[str, Any]],
    *,
    stages: list[str] | None = None,
    state_effects_by_index: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Inject a valid ``section_progression`` into each section dict in place.

    If ``stages`` is omitted it is derived from each section's ``purpose``.
    Returns the mutated list for convenience.
    """

    if stages is None:
        stages = stages_for_purposes(section.get("purpose", "") for section in sections)
    if len(stages) != len(sections):
        raise ValueError("stages length must match sections length")
    effects = state_effects_by_index or {}
    for index, (section, stage) in enumerate(zip(sections, stages)):
        section["section_progression"] = make_section_progression(
            stage,
            state_effects=effects.get(index),
            label=section.get("section_id", f"section_{index + 1}"),
        )
    return sections
