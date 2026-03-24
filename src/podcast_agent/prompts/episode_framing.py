"""Prompt builder for episode framing (intro/outro)."""

from __future__ import annotations


FRAMING_PROMPT = """You are a podcast writer. Generate framing text for one episode.

Return JSON ONLY with keys: recap, next_overview.

Rules:
- If has_previous is false, set recap to an empty string.
- If has_next is false, set next_overview to an empty string.
- Recap must be between {recap_min_words} and {recap_max_words} words when present.
- Next overview must be between {next_min_words} and {next_max_words} words when present.
- Recap must start with one of: {recap_openers}.
- Next overview must start with one of: {next_openers}.
- The recap should use 3-5 concrete events from recap_source.
- The next overview must be a teaser that avoids spoilers and excessive detail.
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
    recap_min_words: int,
    recap_max_words: int,
    next_min_words: int,
    next_max_words: int,
    recap_openers: list[str],
    next_openers: list[str],
) -> str:
    return FRAMING_PROMPT.format(
        recap_min_words=recap_min_words,
        recap_max_words=recap_max_words,
        next_min_words=next_min_words,
        next_max_words=next_max_words,
        recap_openers=", ".join(recap_openers),
        next_openers=", ".join(next_openers),
    )
