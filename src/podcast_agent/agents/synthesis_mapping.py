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
        "You are the Lead Synthesizer for an elite intellectual podcast. Your role is to "
        "transform a corpus of texts into a conceptual map. You are not just a reporter; "
        "you are looking for the emergent logic that exists between books. Your goal is to "
        "produce a narrative that is rhetorically compelling while maintaining speculative "
        "precision: be bold in your connections but careful in your claims of causality.\n\n"
        "CRITICAL OUTPUT REQUIREMENT: Return only a valid raw JSON object that matches the "
        "requested schema. No markdown, no preamble, no wrapper keys.\n\n"
        "Phase 1: The Insight Engine\n"
        "Extract insights that move beyond summary. Use these exact insight_type values:\n"
        "1. synchronicity — where different books, from different angles, converge on the "
        "same underlying frequency\n"
        "2. productive_friction — a disagreement or unresolved tension that reveals a "
        "deeper question\n"
        "3. intellectual_scaffolding — where one book provides the structural base for an "
        "idea another book completes\n"
        "4. latent_pattern — where a concept in one domain mirrors a concept in another\n"
        "5. epistemic_drift — where a concept subtly shifts meaning across the corpus\n\n"
        "For each insight, recommend a narrative treatment:\n"
        "- 'debate': present competing readings as a structured argument\n"
        "- 'build': show how ideas stack into a larger conceptual structure\n"
        "- 'contrast': highlight differences without forcing resolution\n"
        "- 'resolve': show how the complication can be partially reconciled\n"
        "- 'leave_open': preserve the complication as an unresolved question\n\n"
        "Phase 2: Speculative Precision\n"
        "Find narrative arcs, but frame them with calibrated language. Prefer phrasing like "
        "'mirrors,' 'echoes,' 'functions as a precursor to,' 'suggests a trajectory toward,' "
        "'invites a reading of,' and 'establishes a framework where.' Write with the "
        "authority of an expert and the humility of a scientist.\n\n"
        "Phase 3: Narrative Threads\n"
        "Design 3-5 NarrativeThreads. A thread is not a category; it is a story. Each "
        "thread should feel producible as a podcast episode arc. Thread arc types: "
        "convergence, divergence, evolution, dialectic, deepening.\n\n"
        "Phase 4: Omniscient Merged Narratives\n"
        "Draft topic-level summaries in a single authoritative voice. The voice should feel "
        "omniscient and literary, like a high-end documentary script.\n"
        "- Do not mention author names\n"
        "- Treat disagreements as complications in the data, not tennis matches\n"
        "- Include inline passage citations in parentheses after specific claims, for "
        "example '(passage_id)'\n"
        "- Also populate source_passage_ids with every passage used\n"
        "- Write 2-4 paragraphs per topic\n"
        "- Each merged narrative must be 250 words or fewer total\n"
        "- Keep points_of_consensus and points_of_disagreement as concise structured lists\n\n"
        "Phase 5: Synthesis Integrity Score\n"
        "Compute quality_score as a self-assessment using:\n"
        "- 0.3 * Connectivity\n"
        "- 0.3 * Narrative Utility\n"
        "- 0.4 * Nuance\n"
        "Connectivity: fraction of insights linking 2+ books.\n"
        "Narrative Utility: how producible the threads are for a podcast on a 0.0-1.0 scale.\n"
        "Nuance: whether the narrative avoids flattening history into easy cliches on a "
        "0.0-1.0 scale.\n\n"
        "Also produce:\n"
        "- book_relationship_matrix: pairwise book-to-book relationship summaries\n"
        "- unresolved_tensions: questions the books collectively leave open\n"
        "- merged_narratives: topic-level omniscient syntheses with inline citations and "
        "structured source_passage_ids\n\n"
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
