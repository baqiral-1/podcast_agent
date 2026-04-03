"""Stage 10 (secondary): Disagreement narrator for selective attribution moments."""

from __future__ import annotations

from pydantic import BaseModel

from podcast_agent.agents.base import Agent


class DisagreementNarrationResponse(BaseModel):
    text: str


class SourceWeavingAgent(Agent):
    """Surfaces genuine disagreements as narrative moments."""

    schema_name = "source_weaving"
    response_model = DisagreementNarrationResponse
    instructions = (
        "You are a storyteller who has reached a point where sources disagree. "
        "Your job is to make the listener feel the genuine uncertainty of history. "
        "Do not present a balanced academic comparison.\n\n"
        "Techniques you may use:\n"
        "- Pose it as a question\n"
        "- Frame it as a mystery or split in the story\n"
        "- Use it as a dramatic pivot or cliffhanger\n\n"
        "Requirements:\n"
        "- Keep it short and narrative\n"
        "- Mention at most two authors in a paragraph\n"
        "- Do not list author positions in sequence\n"
        "- Do not use academic hedging language\n\n"
        "Return a JSON object with text."
    )

    def build_payload(
        self,
        moment: dict,
        passages: list[dict],
        books: list[dict],
    ) -> dict:
        return {
            "moment": moment,
            "passages": passages,
            "books": books,
        }
