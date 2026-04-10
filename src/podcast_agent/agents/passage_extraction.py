"""Stage 6: Cross-book passage extraction agent (LLM reranking pass)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import passage_extraction_instructions
from podcast_agent.schemas.models import PassagePair, SynthesisTag


class PassageExtractionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    quotability_score: float = Field(ge=0.0, le=1.0)
    synthesis_tags: list[SynthesisTag] = Field(default_factory=list)


class PassageExtractionResponse(BaseModel):
    passages: list[PassageExtractionScore] = Field(default_factory=list)
    cross_book_pairs: list[PassagePair] = Field(default_factory=list)


class PassageExtractionAgent(Agent):
    """Reranks and tags passages with thematic relevance and cross-book relationships."""

    schema_name = "passage_extraction"
    response_model = PassageExtractionResponse
    instructions = passage_extraction_instructions()

    def build_payload(
        self,
        axis_id: str,
        axis_name: str,
        axis_description: str,
        candidate_passages: list[dict],
    ) -> dict:
        return {
            "axis_id": axis_id,
            "axis_name": axis_name,
            "axis_description": axis_description,
            "candidate_passages": candidate_passages,
        }
