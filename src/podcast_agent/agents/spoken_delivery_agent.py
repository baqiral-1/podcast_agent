"""Stage 13: Spoken delivery rewrite agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import SpokenSegment


class SpokenDeliveryResponse(BaseModel):
    segments: list[SpokenSegment] = Field(default_factory=list)
    arc_plan: str | None = None


class SpokenDeliveryAgent(Agent):
    """Transforms factual scripts into natural spoken narration without changing facts."""

    schema_name = "spoken_delivery"
    response_model = SpokenDeliveryResponse
    instructions = (
        "You are a spoken-word adaptation specialist. Transform a written podcast script "
        "into natural spoken narration without changing any facts.\n\n"
        "What you MUST do:\n"
        "- Convert written prose into conversational spoken language\n"
        "- Break long sentences into shorter, punchier ones\n"
        "- Keep each segment's text at or below max_words_per_segment\n"
        "- Set each segment's max_words to no more than max_words_per_segment\n"
        "- Return speech_hints for every segment using only the allowed controls\n\n"
        "Allowed speech_hints fields:\n"
        "- style: one of neutral, measured, urgent, dramatic\n"
        "- intensity: one of none, light, medium, strong\n"
        "- pace: one of slower, normal, faster\n"
        "- pause_before_ms: integer from 0 to 2000\n"
        "- pause_after_ms: integer from 0 to 2000\n"
        "- pronunciation_hints: optional list of {text, spoken_as}\n"
        "- emphasis_targets: optional list of short exact phrases already present in the text\n"
        "- render_strategy: one of plain, isolate_phrase, split_sentences, slow_clause\n\n"
        "What you must NOT do:\n"
        "- Include any emphasis_targets phrase that is not an exact substring of that segment's text\n"
        "- Change any facts, claims, or citations\n"
        "- Add author names or attributions not present in the source\n"
        "- Output raw SSML, XML, or markup tags in the narration text\n"
        "- Pretend these hints are literal SSML or backend-specific tags\n"
        "- Output any extra keys inside speech_hints\n"
        "- Use intense controls unless the narration clearly needs them\n"
        "- Repeat any sentence or distinctive phrase verbatim more than twice.\n"
        "- If a point must recur, paraphrase it with materially different wording\n\n"
        "Default control style:\n"
        "- Prefer neutral style\n"
        "- Prefer none or light intensity\n"
        "- Prefer normal pace\n"
        "- Prefer plain render_strategy unless phrase isolation or sentence splitting clearly helps delivery\n"
        "- Use short pauses unless a stronger beat is clearly justified\n\n"
        "Return a JSON object with 'segments' (list of SpokenSegment) and optional 'arc_plan'."
    )

    def build_payload(
        self,
        episode_number: int,
        script_segments: list[dict],
        max_words_per_segment: int,
        tts_provider: str,
    ) -> dict:
        return {
            "episode_number": episode_number,
            "script_segments": script_segments,
            "max_words_per_segment": max_words_per_segment,
            "tts_provider": tts_provider,
        }
