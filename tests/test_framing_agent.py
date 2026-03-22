"""Tests for episode framing generation."""

from __future__ import annotations

from podcast_agent.agents.framing import EpisodeFramingAgent
from podcast_agent.llm.heuristic import HeuristicLLMClient


def _word_count(text: str) -> int:
    return len(text.split())


def test_episode_framing_agent_enforces_word_counts() -> None:
    agent = EpisodeFramingAgent(
        HeuristicLLMClient(),
        recap_words=80,
        current_words=120,
        next_min_words=40,
        next_max_words=50,
        max_recap_source_words=200,
    )
    payload = agent.build_payload(
        episode_id="episode-2",
        episode_title="Episode 2",
        recap_source="recap source text",
        current_themes=["theme"],
        next_themes=["theme"],
        current_outline="Beat one; Beat two",
        next_outline="Next beat one",
        has_previous=True,
        has_next=True,
    )
    framing = agent.generate(payload)
    assert _word_count(framing.recap) == 80
    assert _word_count(framing.current_summary) == 120
    assert 40 <= _word_count(framing.next_overview) <= 50


def test_episode_framing_agent_handles_missing_neighbors() -> None:
    agent = EpisodeFramingAgent(
        HeuristicLLMClient(),
        recap_words=80,
        current_words=120,
        next_min_words=40,
        next_max_words=50,
    )
    payload = agent.build_payload(
        episode_id="episode-1",
        episode_title="Episode 1",
        recap_source="",
        current_themes=[],
        next_themes=None,
        current_outline="Beat one",
        next_outline=None,
        has_previous=False,
        has_next=False,
    )
    framing = agent.generate(payload)
    assert framing.recap == ""
    assert framing.next_overview == ""
    assert _word_count(framing.current_summary) == 120
