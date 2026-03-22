"""Prompt builder for episode framing (intro/outro)."""

from __future__ import annotations


FRAMING_PROMPT = """You are a podcast writer. Generate framing text for one episode.

Return JSON ONLY with keys: recap, current_summary, next_overview.

Rules:
- If has_previous is false, set recap to an empty string.
- If has_next is false, set next_overview to an empty string.
- Recap must be exactly {recap_words} words when present.
- Current summary must be exactly {current_words} words.
- Next overview must be between {next_min_words} and {next_max_words} words when present.
- The current summary should provide context and tone without revealing main plot points.
- The next overview must build suspense without saying you are withholding information.
- Do not mention "this episode" or "next episode" explicitly; write as natural narration.
- Avoid meta language about sources, chapters, authors, or evidence.

Inputs you can use:
- recap_source: summary material from the prior episode
- current_themes: themes for the current episode
- current_outline: ordered beat-title outline for the current episode
- next_themes: themes for the next episode
- next_outline: ordered beat-title outline for the next episode
"""


def build_episode_framing_instructions(
    *,
    recap_words: int,
    current_words: int,
    next_min_words: int,
    next_max_words: int,
) -> str:
    return FRAMING_PROMPT.format(
        recap_words=recap_words,
        current_words=current_words,
        next_min_words=next_min_words,
        next_max_words=next_max_words,
    )
