"""Prompt builders for spoken-delivery rewriting."""

from __future__ import annotations


TONE_SUFFIXES = {
    "educational": "Prioritize clarity, continuity, and calm explanatory narration.",
    "educational_suspenseful": (
        "Prioritize clarity first. Add only a light undercurrent of suspense through pacing and transitions, "
        "without becoming theatrical."
    ),
    "reflective_history": (
        "Prioritize clarity and historical reflection, while remaining concrete and grounded in the source material."
    ),
}


def build_spoken_delivery_instructions(
    *,
    tone_preset: str,
    target_expansion_ratio: float,
    max_expansion_ratio: float,
) -> str:
    """Build the primary spoken-delivery prompt."""

    tone_suffix = TONE_SUFFIXES.get(tone_preset, TONE_SUFFIXES["educational_suspenseful"])
    target_percent = int(round(target_expansion_ratio * 100))
    max_percent = int(round(max_expansion_ratio * 100))
    return (
        "Rewrite the following text for spoken podcast delivery.\n\n"
        "Your task is a STRUCTURAL TRANSFORMATION, not a creative rewrite.\n\n"
        "---\n\n"
        "GOALS:\n"
        "- Make the text easier to follow when spoken aloud\n"
        "- Preserve ALL original meaning, facts, and details\n"
        "- Maintain an educational, serious tone with light narrative tension\n\n"
        "---\n\n"
        "CONSTRAINTS:\n\n"
        "Content:\n"
        "- Do NOT add new facts, examples, or interpretations\n"
        "- Do NOT remove important details\n"
        "- Preserve >95% of the original information\n\n"
        "Structure:\n"
        "- Maintain the same paragraph structure and ordering\n"
        "- Break long sentences into 2-3 shorter sentences where needed\n"
        "- Combine overly short sentences into natural spoken flow\n\n"
        "Length:\n"
        f"- Output should target {target_percent}% to {max_percent}% of the original word count\n"
        f"- If longer than {max_percent}%, revise and shorten before returning\n\n"
        "Style:\n"
        "- Use natural spoken English\n"
        "- Keep sentences medium-length (avoid overly short, dramatic fragments)\n"
        "- Maintain logical flow and clarity\n"
        "- Add minimal connective phrases only where necessary\n\n"
        "Tone:\n"
        "- Educational first, narrative second\n"
        "- Slightly reflective but not philosophical\n"
        "- Avoid dramatic or theatrical language\n\n"
        "---\n\n"
        "AVOID:\n"
        "- rhetorical questions (unless already implied)\n"
        "- philosophical digressions\n"
        "- repetition or paraphrasing for emphasis\n"
        "- dramatic pauses or filler phrases\n"
        "- overly short sentences designed only for effect\n\n"
        "---\n\n"
        "MENTAL MODEL:\n"
        "This is like converting a dense paragraph from a book into something a narrator can read clearly, "
        "not rewriting it into a different style.\n\n"
        "---\n\n"
        "Previous and next segments are provided only as read-only context for smoother transitions and local pacing. "
        "Do not move facts across segment boundaries or import adjacent-segment-only facts into the current segment. "
        f"{tone_suffix} "
        "Return only the rewritten narration for the current segment."
    )


def build_spoken_delivery_retry_instructions(
    *,
    tone_preset: str,
    target_expansion_ratio: float,
    max_expansion_ratio: float,
) -> str:
    """Build the stricter retry prompt for spoken delivery."""

    tone_suffix = TONE_SUFFIXES.get(tone_preset, TONE_SUFFIXES["educational_suspenseful"])
    target_percent = int(round(target_expansion_ratio * 100))
    max_percent = int(round(max_expansion_ratio * 100))
    return (
        "Revise the previous spoken-delivery draft for the current segment.\n\n"
        "Your task is still a STRUCTURAL TRANSFORMATION, not a creative rewrite.\n\n"
        "Keep all original meaning, facts, and details from the current segment, but shorten and tighten the prose "
        "so it fits the required spoken-delivery bounds.\n\n"
        "Requirements:\n"
        f"- Keep the output at or below {max_percent}% of the original word count\n"
        f"- Aim for roughly {target_percent}% to {max_percent}% of the original word count\n"
        "- Remove redundant connective phrases and repetition\n"
        "- Keep medium-length spoken sentences\n"
        "- Preserve paragraph order and core informational content\n"
        "- Do not add new facts, interpretations, analogies, rhetorical questions, or theatrical phrasing\n\n"
        "Previous and next segments are read-only context only. "
        "Do not import adjacent-segment-only facts into the current segment. "
        f"{tone_suffix} "
        "Return only the rewritten narration for the current segment."
    )
