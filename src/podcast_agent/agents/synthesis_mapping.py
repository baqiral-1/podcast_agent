"""Stage 7: Synthesis mapping agent — the highest-value component."""

from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.base import Agent
from podcast_agent.schemas.models import (
    MergedNarrative,
    NarrativeThread,
    SynthesisInsight,
    SynthesisMap,
)


class SynthesisMappingResponse(BaseModel):
    insights: list[SynthesisInsight] = Field(default_factory=list)
    narrative_threads: list[NarrativeThread] = Field(default_factory=list)
    book_relationship_matrix: dict[str, dict[str, str]] = Field(default_factory=dict)
    unresolved_tensions: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    merged_narratives: list[MergedNarrative] = Field(default_factory=list)


class SynthesisMappingAgent(Agent):
    """Discovers intellectual relationships across books to produce genuine synthesis."""

    schema_name = "synthesis_mapping"
    response_model = SynthesisMappingResponse
    instructions = (
        "You are an intellectual synthesizer creating a multi-book podcast. Examine passages "
        "from N books organized by thematic axes. Your job is to discover:\n\n"
        "1. AGREEMENT — same conclusion, possibly different reasoning or evidence\n"
        "2. DISAGREEMENT — opposing conclusions. What's the root cause? Different evidence, "
        "values, or frameworks?\n"
        "3. EXTENSION — one author takes another's idea further\n"
        "4. TENSION — productive friction that neither author fully resolves\n"
        "5. SURPRISING_CONNECTION — seemingly unrelated ideas that connect unexpectedly\n"
        "6. EVOLUTION — books as an intellectual progression\n\n"
        "For each insight, recommend a narrative treatment:\n"
        "- 'debate': present both sides as a structured argument\n"
        "- 'build': show how ideas stack on top of each other\n"
        "- 'contrast': highlight differences without declaring a winner\n"
        "- 'resolve': show how the tension can be reconciled\n"
        "- 'leave_open': present the tension as an open question for the listener\n\n"
        "After identifying individual insights, group them into 3-7 NarrativeThreads. "
        "Each thread is a sequence of insights that tells a coherent story across episodes.\n"
        "Thread arc types: convergence, divergence, evolution, dialectic, deepening.\n\n"
        "Compute a quality_score:\n"
        "- 0.3 × (fraction of insights involving 2+ books)\n"
        "- 0.3 × (diversity of insight types / 6)\n"
        "- 0.2 × (average podcast_potential across insights)\n"
        "- 0.2 × (fraction of insights assigned to threads)\n\n"
        "Also produce:\n"
        "- book_relationship_matrix: pairwise book-to-book relationship summaries\n"
        "- unresolved_tensions: questions the books collectively leave open\n"
        "- merged_narratives: topic-level narrative summaries written in a single "
        "omniscient voice. Do not mention any author names in these narratives. "
        "List points_of_consensus and points_of_disagreement separately.\n\n"
        "Merged narrative requirements:\n"
        "- Write 2-4 paragraphs per topic\n"
        "- Each merged narrative must be 250 words or fewer total\n"
        "- Use source_passage_ids for all passages used\n"
        "- Do not organize the narrative around disagreements\n"
        "- Disagreements should be interruptions to the story, not its structure\n\n"
        "Return a JSON object with insights, narrative_threads, book_relationship_matrix, "
        "unresolved_tensions, merged_narratives, and quality_score."
    )

    def build_payload(
        self,
        project_id: str,
        axes_summary: list[dict],
        passages_by_axis: dict[str, list[dict]],
        cross_book_pairs: list[dict],
        book_metadata: list[dict],
    ) -> dict:
        return {
            "project_id": project_id,
            "axes": axes_summary,
            "passages_by_axis": passages_by_axis,
            "cross_book_pairs": cross_book_pairs,
            "books": book_metadata,
        }
