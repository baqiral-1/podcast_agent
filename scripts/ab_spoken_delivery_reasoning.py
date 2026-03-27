#!/usr/bin/env python3
"""Quick A/B timing for spoken delivery narration with small inputs."""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

from podcast_agent.config import LLMConfig
from podcast_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from podcast_agent.prompts.spoken_delivery import build_spoken_delivery_narration_instructions
from podcast_agent.schemas.models import SpokenDeliveryNarrationDraft


def _build_minimal_arc_plan() -> dict:
    return {
        "theme": "A small test of delivery pacing.",
        "threads": [
            {
                "name": "Test thread",
                "introduced": "A quick setup.",
                "developed": "A brief development.",
                "payoff": "A short payoff.",
                "listener_feeling": "Curious.",
            }
        ],
        "opening": {
            "scene": "A quiet moment before the story starts.",
            "why": "Keeps the sample short.",
            "transition_strategy": "Slide into the main narration.",
        },
        "acts": [
            {
                "number": 1,
                "title": "Act One",
                "source_material": "Two short paragraphs of source.",
                "why_here": "It is the entire sample.",
                "driving_tension": "How quickly can we deliver?",
                "transition_to_next": "Close the sample.",
            }
        ],
        "plants_and_payoffs": [
            {
                "fact": "A single small fact.",
                "plant_in_act": 1,
                "payoff_in_act": 1,
                "bridging_language": "That detail matters here.",
            }
        ],
        "key_moments": [
            {
                "moment": "A brief turn.",
                "why_it_matters": "It marks the pivot.",
                "how_to_land": "Keep it tight.",
                "in_act": 1,
            }
        ],
    }


def _build_source_script() -> str:
    return (
        "The courier arrives at dusk. A single envelope changes the plan.\n\n"
        "By morning, the team has to move fast, and the city feels smaller."
    )


def _run_for_effort(client: OpenAICompatibleLLMClient, effort: str, runs: int) -> list[float]:
    durations: list[float] = []
    instructions = build_spoken_delivery_narration_instructions()
    payload = {
        "episode_id": "episode-1",
        "episode_title": "AB Test Episode",
        "arc_plan": _build_minimal_arc_plan(),
        "source_script": _build_source_script(),
    }
    for _ in range(runs):
        start = time.perf_counter()
        client.generate_json(
            schema_name="spoken_delivery_narration",
            instructions=instructions,
            payload=payload,
            response_model=SpokenDeliveryNarrationDraft,
        )
        durations.append(time.perf_counter() - start)
    return durations


def _format_stats(effort: str, durations: Iterable[float]) -> str:
    values = list(durations)
    if not values:
        return f"{effort}: no runs"
    total = sum(values)
    avg = total / len(values)
    return f"{effort}: runs={len(values)} avg={avg:.2f}s min={min(values):.2f}s max={max(values):.2f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("LLM_MODEL_NAME", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--efforts",
        default="medium,high",
        help="Comma-separated reasoning efforts to test (e.g., low,medium,high).",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    for effort in efforts:
        config = LLMConfig(
            api_key=args.api_key,
            model_name=args.model,
            base_url=args.base_url,
            reasoning_effort=effort,
        )
        client = OpenAICompatibleLLMClient(config=config)
        durations = _run_for_effort(client, effort, args.runs)
        print(_format_stats(effort, durations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
