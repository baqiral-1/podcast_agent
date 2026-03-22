"""Tests for the full-episode spoken-delivery agent."""

from __future__ import annotations

from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.eval.rewrite_metrics import check_fidelity, extract_names
from podcast_agent.llm.base import LLMClient
from podcast_agent.schemas.models import EpisodeScript, EpisodeSegment, ScriptClaim


def _build_script() -> EpisodeScript:
    return EpisodeScript(
        episode_id="episode-1",
        title="Episode 1",
        narrator="Narrator",
        segments=[
            EpisodeSegment(
                segment_id="episode-1-segment-1",
                beat_id="beat-1",
                heading="Opening",
                narration="In 1857, Delhi became the center of the uprising. Bahadur Shah Zafar was drawn into events.",
                claims=[
                    ScriptClaim(
                        claim_id="beat-1-claim-1",
                        text="Delhi became the center of the uprising in 1857.",
                        evidence_chunk_ids=["chunk-1"],
                    )
                ],
                citations=["chunk-1"],
            ),
            EpisodeSegment(
                segment_id="episode-1-segment-2",
                beat_id="beat-1",
                heading="Pressure",
                narration=(
                    "British officers tried to regain control, and the city became crowded with rival forces, "
                    "administrators, and frightened civilians as the crisis deepened in July 1857."
                ),
                claims=[
                    ScriptClaim(
                        claim_id="beat-1-claim-2",
                        text="The crisis deepened in July 1857.",
                        evidence_chunk_ids=["chunk-2"],
                    )
                ],
                citations=["chunk-2"],
            ),
        ],
    )


class TwoCallSpokenDeliveryLLM(LLMClient):
    """Stub that returns a plan then narration for full-episode flow."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        self.calls.append((schema_name, payload))
        if schema_name == "spoken_delivery_plan":
            return response_model.model_validate(
                {
                    "theme": "Test theme",
                    "threads": [
                        {
                            "name": "Thread",
                            "introduced": "Act 1",
                            "developed": "Act 2",
                            "payoff": "Act 3",
                            "listener_feeling": "Relief",
                        }
                    ],
                    "opening": {
                        "scene": "Opening scene",
                        "why": "Hook",
                        "transition_strategy": "Pull back.",
                    },
                    "acts": [
                        {
                            "number": 1,
                            "title": "Act 1",
                            "source_material": "Opening",
                            "why_here": "Start here",
                            "driving_tension": "Why now",
                            "transition_to_next": "Next",
                        }
                    ],
                    "plants_and_payoffs": [],
                    "key_moments": [],
                }
            )
        if schema_name == "spoken_delivery_narration":
            narration = payload.get("source_script", "").strip()
            return response_model.model_validate({"narration": narration})
        raise AssertionError(f"Unexpected schema name: {schema_name}")


def test_spoken_delivery_full_episode_two_call_flow() -> None:
    llm = TwoCallSpokenDeliveryLLM()
    agent = SpokenDeliveryAgent(llm, chunk_min_words=1, chunk_max_words=50)

    spoken_script, spoken_delivery, arc_plan = agent.rewrite_full_episode_two_call(_build_script())

    assert arc_plan.theme == "Test theme"
    assert spoken_delivery.chunk_count >= 1
    assert spoken_script.narration != ""
    assert llm.calls[0][0] == "spoken_delivery_plan"
    assert llm.calls[1][0] == "spoken_delivery_narration"


def test_extract_names_filters_discourse_tokens_and_keeps_real_entities() -> None:
    text = (
        "On November 5th, Bahadur Shah Zafar remained in the Red Fort. "
        "Meanwhile, Hakim Ahsanullah Khan spoke with the British. "
        "Moreover, the archivist noted the change. "
        "Additionally, the crowd dispersed."
    )

    names = extract_names(text)

    assert "On" not in names
    assert "November" not in names
    assert "On November" not in names
    assert "Meanwhile" not in names
    assert "Moreover" not in names
    assert "Additionally" not in names
    assert "Bahadur Shah Zafar" in names
    assert "Red Fort" in names
    assert "Hakim Ahsanullah Khan" in names
    assert "British" in names


def test_check_fidelity_ignores_sentence_starter_rephrasing_when_entities_are_preserved() -> None:
    source = (
        "As the crisis deepened in 1857, Bahadur Shah Zafar remained in Delhi. "
        "Meanwhile, Hakim Ahsanullah Khan advised caution."
    )
    spoken = (
        "The crisis deepened in 1857, and Bahadur Shah Zafar remained in Delhi. "
        "Hakim Ahsanullah Khan advised caution."
    )

    fidelity = check_fidelity(source, spoken)

    assert fidelity.passed is True
    assert fidelity.missing_numbers == []
    assert fidelity.missing_names == []


def test_check_fidelity_can_ignore_paragraph_drift_when_disabled() -> None:
    source = "Bahadur Shah Zafar remained in Delhi."
    spoken = "Bahadur Shah Zafar remained in Delhi.\n\nHe was under watch."

    fidelity_default = check_fidelity(source, spoken)
    fidelity_relaxed = check_fidelity(source, spoken, check_paragraph_drift=False)

    assert fidelity_default.passed is False
    assert fidelity_relaxed.passed is True


def test_check_fidelity_does_not_fail_when_number_is_dropped() -> None:
    source = "In 1857, Bahadur Shah Zafar remained in Delhi."
    spoken = "Bahadur Shah Zafar remained in Delhi."

    fidelity = check_fidelity(source, spoken)

    assert fidelity.passed is True
    assert fidelity.missing_numbers == []
