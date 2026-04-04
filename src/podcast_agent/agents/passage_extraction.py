"""Stage 6: Cross-book passage extraction agent (LLM reranking pass)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from podcast_agent.agents.base import Agent
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
    instructions = (
        "You are a passage analyst for a cross-book podcast. For each input passage, score its "
        "thematic relevance, its suitability for spoken delivery, and its relationship to passages "
        "from other books.\n\n"
        "Return JSON with exactly two arrays: passages and cross_book_pairs.\n\n"
        "For passages, output exactly one object per input candidate passage, in the same order. "
        "The number of passages MUST equal the number of input candidates. Do not omit any. "
        "Do not duplicate any passage_id. Preserve the input order.\n"
        "Each passages object must have ONLY these keys:\n"
        "- passage_id: copy exactly from the input candidate\n"
        "- relevance_score (0.0-1.0): How directly does this passage address the axis?\n"
        "- quotability_score (0.0-1.0): How suitable for spoken delivery? Favor concrete examples, "
        "vivid anecdotes, clear arguments. Penalize dense academic prose, heavy jargon, data tables.\n"
        "- synthesis_tags: list of tags from: contradicts, exemplifies, contextualizes, independent.\n\n"
        "Do NOT include book_id, chunk_ids, text, chapter_ref, axis_id, author, or title in passages.\n\n"
        "For cross_book_pairs, each object must have ONLY these keys:\n"
        "- passage_a_id\n"
        "- passage_b_id\n"
        "- relationship (one of: contradicts, exemplifies, contextualizes, independent)\n"
        "- strength (0.0-1.0)\n"
        "- axis_id\n\n"
        "Every cross_book_pairs item MUST connect passages from different books. Never pair two passages "
        "from the same book. If you are not certain both passages are from different books, omit the pair.\n\n"
        "Return at most 5 cross_book_pairs. If more are possible, choose the strongest, most salient "
        "relationships for this axis.\n\n"
        "If there are no cross_book_pairs, return an empty array. Do not omit required keys."
    )

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
