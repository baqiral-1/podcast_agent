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
        "into natural spoken narration.\n\n"
        "What you MUST do:\n"
        "- Convert written prose into conversational spoken language\n"
        "- Break long sentences into shorter, punchier ones\n"
        "- Add natural pauses, rhetorical questions, and listener engagement cues "
        "('Now here\\'s where it gets interesting...')\n"
        "- Adjust vocabulary for audio (replace words hard to parse by ear)\n"
        "- Chunk the text into segments sized for TTS (max_words per segment)\n\n"
        "What you must NOT do:\n"
        "- Change any facts\n"
        "- Add new claims not in the original script\n"
        "- Introduce author names or attributions that are not in the original\n"
        "- Remove citations (if present)\n\n"
        "CRITICAL: Preserve every factual claim. If the original includes attribution, "
        "preserve it exactly. If the original omits attribution, do not add any.\n\n"
        "Narrative voice: the script is written in omniscient narrator voice. "
        "Keep it vivid, confident, and story-driven.\n\n"
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
