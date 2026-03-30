"""Tests for episode framing generation."""

from __future__ import annotations

from podcast_agent.agents.framing import EpisodeFramingAgent
from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.schemas.models import EpisodeFraming


class FramingSequenceLLM(LLMClient):
    """Stub LLM that returns framing responses in sequence."""

    def __init__(self, responses: list[dict[str, str]]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        del schema_name, instructions, payload, response_model
        self.calls += 1
        if not self._responses:
            raise AssertionError("No response configured")
        return EpisodeFraming.model_validate(self._responses.pop(0))


def test_episode_framing_agent_returns_structured_framing() -> None:
    agent = EpisodeFramingAgent(
        HeuristicLLMClient(),
        recap_min_words=80,
        recap_max_words=100,
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
    assert isinstance(framing.recap, str)
    assert isinstance(framing.next_overview, str)
    assert framing.recap
    assert framing.next_overview


def test_episode_framing_agent_handles_missing_neighbors() -> None:
    agent = EpisodeFramingAgent(
        HeuristicLLMClient(),
        recap_min_words=80,
        recap_max_words=100,
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


def test_episode_framing_agent_accepts_previously_failing_format_variants() -> None:
    llm = FramingSequenceLLM(
        responses=[
            {
                "recap": (
                    "Previously, we saw the fall of Edessa shake Christendom, prompting Bernard of Clairvaux "
                    "to preach the Second Crusade at Vézelay in 1146. Kings Louis and Conrad marched east, "
                    "only to suffer humiliating defeat at Damascus in 1148. From that failure, Nur al-Din rose "
                    "as champion of the Muslim revival, commissioning a magnificent minbar for the eventual "
                    "reconquest of Jerusalem. Meanwhile, the young Kurdish warrior Saladin seized power in Egypt, "
                    "and after Nur al-Din's death in 1174, he battled both Frankish forces and the legendary "
                    "leper king Baldwin the Fourth for supremacy."
                ),
                "next_overview": (
                    "Next time, the story moves to new crusading ventures, as Frederick II pursues an "
                    "unconventional path to Jerusalem and a rising Mamluk sultan named Baybars reshapes the "
                    "struggle for the Holy Land."
                ),
            }
        ]
    )
    agent = EpisodeFramingAgent(
        llm,
        recap_min_words=80,
        recap_max_words=100,
        next_min_words=20,
        next_max_words=30,
    )
    payload = agent.build_payload(
        episode_id="episode-4",
        episode_title="Episode 4",
        recap_source="recap source text",
        current_themes=["theme"],
        next_themes=["theme"],
        current_outline="Beat one; Beat two",
        next_outline="Next beat one",
        has_previous=True,
        has_next=True,
    )
    framing = agent.generate(payload)
    assert framing.recap.startswith("Previously, we saw the")
    assert framing.next_overview.startswith("Next time,")
    assert llm.calls == 1
