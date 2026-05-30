"""Shared retry-feedback helpers for the episode_architecture stage.

Both the inner LLM retry (via ``EpisodeArchitectureAgent.prepare_retry_payload``)
and the outer orchestrator retry (via ``_build_architecture_retry_feedback``)
populate the same ``architecture_feedback`` payload key. This module renders the
shared, shape-aware repair instructions for the two failure modes that
recurred across ``runs/iranian_revolution_v74`` (peripheral_touch fallback
emitted with empty grounding_passage_ids, and host-beat eligible_phases
hallucinating ``"mid"``).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


INNER_RETRY_INSTRUCTION_HEADER = (
    "The previous attempt produced an invalid episode architecture. "
    "On this retry, repair exactly the failures listed below; keep every "
    "other field unchanged; do NOT reintroduce these same violations.\n"
)


def format_validation_errors(exc: ValidationError) -> str:
    """Render the first 8 Pydantic field errors as a bulleted instruction body."""
    lines: list[str] = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = str(err.get("msg", "validation error"))
        ctx = err.get("ctx") or {}
        # ctx may carry a wrapped ValueError; render its message rather than repr.
        ctx_bits_parts: list[str] = []
        for k, v in ctx.items():
            if isinstance(v, Exception):
                ctx_bits_parts.append(f"{k}={v}")
            else:
                ctx_bits_parts.append(f"{k}={v}")
        ctx_bits = ", ".join(ctx_bits_parts)
        suffix = f" ({ctx_bits})" if ctx_bits else ""
        lines.append(f"- `{loc}`: {msg}{suffix}")
    total = len(exc.errors())
    if total > 8:
        lines.append(f"(+{total - 8} more)")
    return "\n".join(lines)


def _failing_section_ids_for_peripheral_touch(
    exc: ValidationError,
    sections_by_index: dict[int, str],
) -> list[str]:
    """Pick out the section_ids whose thread_binding failed peripheral_touch grounding."""
    ids: list[str] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        if len(loc) < 2 or loc[0] != "sections":
            continue
        msg = str(err.get("msg", ""))
        if "peripheral_touch fallback must ground" not in msg:
            continue
        idx = loc[1]
        if not isinstance(idx, int):
            continue
        sid = sections_by_index.get(idx)
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def shape_specific_appendix(
    exc: ValidationError,
    *,
    section_context: dict[str, dict[str, list[str]]] | None = None,
    sections_by_index: dict[int, str] | None = None,
) -> str:
    """Append shape-specific corrective guidance for the two recurring failure modes.

    ``section_context``, when supplied, maps section_id -> {priority_core_passage_ids: [...],
    support_passage_ids: [...]}. When provided, the peripheral_touch repair rule
    additionally lists the actual passage_ids the model could have picked per failing
    section.

    ``sections_by_index``, when supplied, maps the integer index used inside the
    Pydantic error ``loc`` to a section_id, so per-section detail can be rendered
    even when the loc only carries an index.
    """
    msgs = [str(err.get("msg", "")) for err in exc.errors()]
    bits: list[str] = []
    if any("peripheral_touch fallback must ground" in m for m in msgs):
        rule = (
            "peripheral_touch repair rule: each named section either "
            "(a) supplies at least one passage_id in `thread_binding.grounding_passage_ids` "
            "drawn from that section's `priority_core_passage_ids` or the episode's "
            "support passages, OR (b) switches `fallback_mode` to `structural_only` "
            "(which requires no grounding). Do NOT keep `peripheral_touch` with empty "
            "`grounding_passage_ids`."
        )
        if section_context and sections_by_index:
            failing_ids = _failing_section_ids_for_peripheral_touch(
                exc, sections_by_index=sections_by_index
            )
            per_section_lines: list[str] = []
            for sid in failing_ids:
                ctx = section_context.get(sid) or {}
                core = list(ctx.get("priority_core_passage_ids", []))
                support = list(ctx.get("support_passage_ids", []))
                available = core + [pid for pid in support if pid not in core]
                if available:
                    rendered = ", ".join(f'"{pid}"' for pid in available[:8])
                    if len(available) > 8:
                        rendered += f", +{len(available) - 8} more"
                    per_section_lines.append(
                        f"- section `{sid}`: available passage_ids = [{rendered}]"
                    )
                else:
                    per_section_lines.append(
                        f"- section `{sid}`: no section-local passage_ids available — "
                        "switch `fallback_mode` to `structural_only`."
                    )
            if per_section_lines:
                rule = rule + "\n" + "\n".join(per_section_lines)
        bits.append(rule)
    if any("'open', 'pivot' or 'close'" in m for m in msgs):
        bits.append(
            "eligible_phases repair rule: replace any non-conforming value with one of "
            "exactly `open`, `pivot`, `close`. The string `mid` (or `middle`) is not legal "
            "for `host_beat_designations.eligible_phases`; the legal triple is "
            "`{open, pivot, close}`. Note: `authorial_passages.placement` uses a different "
            "triple `{open, mid, close}` — do not transfer that vocabulary here."
        )
    return ("\n" + "\n".join(bits)) if bits else ""


def build_section_context_from_architecture_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, list[str]]], dict[int, str]]:
    """Build the (section_context, sections_by_index) pair from an architecture LLM payload.

    Used by the orchestrator when wrapping a failed outer attempt: ``payload`` is the
    raw payload returned by the model (``exc.data["raw_payload"]``). For each section
    in the failed payload, we extract its ``priority_core_passage_ids``. Support
    passages live at the episode level (not section level), so they are layered in
    by the caller if available.
    """
    section_context: dict[str, dict[str, list[str]]] = {}
    sections_by_index: dict[int, str] = {}
    if not isinstance(payload, dict):
        return section_context, sections_by_index
    sections = payload.get("sections")
    if not isinstance(sections, list):
        return section_context, sections_by_index
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        sid = section.get("section_id")
        if not isinstance(sid, str):
            continue
        sections_by_index[idx] = sid
        core_ids = section.get("priority_core_passage_ids") or []
        section_context[sid] = {
            "priority_core_passage_ids": [str(pid) for pid in core_ids if isinstance(pid, str)],
            "support_passage_ids": [],
        }
    return section_context, sections_by_index


def layer_in_episode_support_passages(
    section_context: dict[str, dict[str, list[str]]],
    support_passage_ids: list[str],
) -> None:
    """Layer episode-level support_passage_ids into every section's context, in place."""
    if not support_passage_ids:
        return
    for ctx in section_context.values():
        existing = set(ctx.get("priority_core_passage_ids", []))
        ctx["support_passage_ids"] = [pid for pid in support_passage_ids if pid not in existing]
