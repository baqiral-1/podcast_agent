"""Tests for the spoken-delivery agent."""

from __future__ import annotations

import threading
import time

from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.eval.rewrite_metrics import check_fidelity, extract_names
from podcast_agent.llm.base import LLMClient
from podcast_agent.prompts.spoken_delivery import build_spoken_delivery_instructions
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
            EpisodeSegment(
                segment_id="episode-1-segment-3",
                beat_id="beat-2",
                heading="Aftermath",
                narration="The struggle left the city shattered, and survivors faced reprisals in the months that followed.",
                claims=[
                    ScriptClaim(
                        claim_id="beat-2-claim-1",
                        text="Survivors faced reprisals.",
                        evidence_chunk_ids=["chunk-3"],
                    )
                ],
                citations=["chunk-3"],
            ),
        ],
    )


class ContextCapturingSpokenDeliveryLLM(LLMClient):
    """Stub that records adjacent context and returns a bounded rewrite."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    def generate_json(self, schema_name, instructions, payload, response_model):
        assert schema_name == "spoken_delivery_segment"
        self.payloads.append(payload)
        current = payload["current_segment"]["narration"]
        return response_model.model_validate({"narration": f"At this point, {current}"})


class RetryingSpokenDeliveryLLM(LLMClient):
    """Stub that over-expands once, then returns a bounded retry."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_json(self, schema_name, instructions, payload, response_model):
        assert schema_name == "spoken_delivery_segment"
        self.calls += 1
        if self.calls == 1:
            return response_model.model_validate(
                {
                    "narration": (
                        payload["current_segment"]["narration"]
                        + " This elaborates the same facts again and again without adding useful clarity."
                    )
                    * 2
                }
            )
        return response_model.model_validate({"narration": payload["current_segment"]["narration"]})


class FailingFidelitySpokenDeliveryLLM(LLMClient):
    """Stub that drops the key number on every attempt, forcing fallback."""

    def generate_json(self, schema_name, instructions, payload, response_model):
        assert schema_name == "spoken_delivery_segment"
        text = payload["current_segment"]["narration"].replace("1857", "")
        return response_model.model_validate({"narration": text})


class ParallelContextCapturingSpokenDeliveryLLM(LLMClient):
    """Stub that proves rewrite order is preserved under concurrent execution."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []
        self._lock = threading.Lock()

    def generate_json(self, schema_name, instructions, payload, response_model):
        assert schema_name == "spoken_delivery_segment"
        with self._lock:
            self.payloads.append(payload)
        segment_id = payload["current_segment"]["segment_id"]
        if segment_id == "episode-1-segment-1":
            time.sleep(0.03)
        elif segment_id == "episode-1-segment-2":
            time.sleep(0.01)
        current = payload["current_segment"]["narration"]
        return response_model.model_validate({"narration": f"{current} [{segment_id}]"})


def test_spoken_delivery_uses_previous_and_next_context() -> None:
    llm = ContextCapturingSpokenDeliveryLLM()
    agent = SpokenDeliveryAgent(llm, spoken_delivery_parallelism=1)

    spoken_script, spoken_delivery = agent.rewrite(_build_script())

    assert len(llm.payloads) == 3
    assert llm.payloads[0]["previous_segment"] is None
    assert llm.payloads[0]["next_segment"]["segment_id"] == "episode-1-segment-2"
    assert llm.payloads[1]["previous_segment"]["segment_id"] == "episode-1-segment-1"
    assert llm.payloads[1]["next_segment"]["segment_id"] == "episode-1-segment-3"
    assert llm.payloads[2]["next_segment"] is None
    assert spoken_script.segments[1].segment_id == "episode-1-segment-2"
    assert spoken_script.segments[1].beat_id == "beat-1"
    assert spoken_delivery.segments[1].fidelity_passed is True


def test_spoken_delivery_retries_when_output_exceeds_max_expansion() -> None:
    llm = RetryingSpokenDeliveryLLM()
    agent = SpokenDeliveryAgent(llm, max_expansion_ratio=1.2, target_expansion_ratio=1.1)

    spoken_script, spoken_delivery = agent.rewrite(_build_script())

    assert llm.calls == 4
    assert spoken_delivery.segments[0].retry_applied is True
    assert spoken_script.segments[0].expansion_ratio <= 1.2


def test_spoken_delivery_falls_back_when_fidelity_checks_fail() -> None:
    llm = FailingFidelitySpokenDeliveryLLM()
    script = _build_script()
    agent = SpokenDeliveryAgent(llm)

    spoken_script, spoken_delivery = agent.rewrite(script)

    assert spoken_delivery.segments[0].fallback_used is True
    assert spoken_script.segments[0].narration == script.segments[0].narration
    assert spoken_delivery.segments[0].missing_numbers == []
    assert [attempt.attempt for attempt in spoken_delivery.segments[0].attempts] == [
        "initial",
        "retry",
        "fallback",
    ]
    assert spoken_delivery.segments[0].attempts[0].missing_numbers == ["1857"]
    assert spoken_delivery.segments[0].attempts[0].failure_reasons == ["missing_numbers"]
    assert spoken_delivery.segments[0].attempts[1].missing_numbers == ["1857"]
    assert spoken_delivery.segments[0].attempts[1].failure_reasons == ["missing_numbers"]
    assert spoken_delivery.segments[0].attempts[2].failure_reasons == []


def test_spoken_delivery_prompt_uses_structural_transformation_wording() -> None:
    instructions = build_spoken_delivery_instructions(
        tone_preset="educational_suspenseful",
        target_expansion_ratio=1.1,
        max_expansion_ratio=1.2,
    )

    assert "Rewrite the following text for spoken podcast delivery." in instructions
    assert "Your task is a STRUCTURAL TRANSFORMATION, not a creative rewrite." in instructions
    assert "Preserve ALL original meaning, facts, and details" in instructions
    assert "Maintain the same paragraph structure and ordering" in instructions
    assert "AVOID:" in instructions
    assert "MENTAL MODEL:" in instructions
    assert "Previous and next segments are provided only as read-only context" in instructions
    assert "Return only the rewritten narration for the current segment." in instructions
    assert "110% to 120%" in instructions
    assert "If longer than 120%" in instructions


def test_spoken_delivery_parallel_rewrite_preserves_segment_order() -> None:
    llm = ParallelContextCapturingSpokenDeliveryLLM()
    agent = SpokenDeliveryAgent(llm, spoken_delivery_parallelism=4)
    script = _build_script()

    spoken_script, spoken_delivery = agent.rewrite(script)

    assert [segment.segment_id for segment in spoken_script.segments] == [
        "episode-1-segment-1",
        "episode-1-segment-2",
        "episode-1-segment-3",
    ]
    assert [segment.narration for segment in spoken_script.segments] == [
        script.segments[0].narration + " [episode-1-segment-1]",
        script.segments[1].narration + " [episode-1-segment-2]",
        script.segments[2].narration + " [episode-1-segment-3]",
    ]
    assert [segment_result.segment_id for segment_result in spoken_delivery.segments] == [
        "episode-1-segment-1",
        "episode-1-segment-2",
        "episode-1-segment-3",
    ]
    assert len(llm.payloads) == 3


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


def test_check_fidelity_still_fails_when_number_is_dropped() -> None:
    source = "In 1857, Bahadur Shah Zafar remained in Delhi."
    spoken = "Bahadur Shah Zafar remained in Delhi."

    fidelity = check_fidelity(source, spoken)

    assert fidelity.passed is False
    assert fidelity.missing_numbers == ["1857"]


def test_spoken_delivery_successful_segment_records_attempt_diagnostics() -> None:
    llm = ContextCapturingSpokenDeliveryLLM()
    agent = SpokenDeliveryAgent(llm, spoken_delivery_parallelism=1)

    _, spoken_delivery = agent.rewrite(_build_script())

    attempts = spoken_delivery.segments[0].attempts
    assert [attempt.attempt for attempt in attempts] == ["initial"]
    assert attempts[0].fidelity_passed is True
    assert attempts[0].failure_reasons == []
