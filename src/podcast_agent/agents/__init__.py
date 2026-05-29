"""Agent exports for the multi-book thematic podcast pipeline."""

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.excerpt_extraction import ExcerptExtractionAgent
from podcast_agent.agents.narrative_strategy_enrichment import (
    NarrativeStrategyEnrichmentAgent,
)
from podcast_agent.agents.narrative_strategy_skeleton import (
    NarrativeStrategySkeletonAgent,
)
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_function_tagging import (
    PrimitiveFunctionTaggingAgent,
)
from podcast_agent.agents.quality_judge import QualityJudgeAgent
from podcast_agent.agents.scene_discovery import SceneDiscoveryAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.style_audit import StyleAuditAgent
from podcast_agent.agents.synthesis_primitives import (
    PrimitiveSubstrateExtractionAgent,
    SynthesisPrimitivesAgent,
)
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations

__all__ = [
    "BookSummaryAgent",
    "ChapterSummaryAgent",
    "EpisodeArchitectureAgent",
    "EpisodePlanningAgent",
    "ExcerptExtractionAgent",
    "NarrativeStrategyEnrichmentAgent",
    "NarrativeStrategySkeletonAgent",
    "PassageExtractionAgent",
    "PrimitiveFunctionTaggingAgent",
    "QualityJudgeAgent",
    "SceneDiscoveryAgent",
    "PrimitiveSubstrateExtractionAgent",
    "SpokenDeliveryAgent",
    "StyleAuditAgent",
    "SynthesisPrimitivesAgent",
    "ThemeDecompositionAgent",
    "WritingAgent",
    "WritingAgentNoCitations",
]
