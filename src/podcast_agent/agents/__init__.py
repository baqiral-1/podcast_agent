"""Agent exports for the multi-book thematic podcast pipeline."""

from podcast_agent.agents.book_summary import BookSummaryAgent
from podcast_agent.agents.chapter_summary import ChapterSummaryAgent
from podcast_agent.agents.episode_architecture import EpisodeArchitectureAgent
from podcast_agent.agents.narrative_strategy import NarrativeStrategyAgent
from podcast_agent.agents.passage_extraction import PassageExtractionAgent
from podcast_agent.agents.planning import EpisodePlanningAgent
from podcast_agent.agents.primitive_function_tagging import (
    PrimitiveFunctionTaggingAgent,
)
from podcast_agent.agents.repair import RepairAgent
from podcast_agent.agents.scene_discovery import SceneDiscoveryAgent
from podcast_agent.agents.spoken_delivery_agent import SpokenDeliveryAgent
from podcast_agent.agents.style_audit import StyleAuditAgent
from podcast_agent.agents.synthesis_primitives import (
    PrimitiveSubstrateExtractionAgent,
    SynthesisPrimitivesAgent,
)
from podcast_agent.agents.theme_decomposition import ThemeDecompositionAgent
from podcast_agent.agents.validation import GroundingValidationAgent
from podcast_agent.agents.writing import WritingAgent, WritingAgentNoCitations

__all__ = [
    "BookSummaryAgent",
    "ChapterSummaryAgent",
    "EpisodeArchitectureAgent",
    "EpisodePlanningAgent",
    "GroundingValidationAgent",
    "NarrativeStrategyAgent",
    "PassageExtractionAgent",
    "PrimitiveFunctionTaggingAgent",
    "RepairAgent",
    "SceneDiscoveryAgent",
    "PrimitiveSubstrateExtractionAgent",
    "SpokenDeliveryAgent",
    "StyleAuditAgent",
    "SynthesisPrimitivesAgent",
    "ThemeDecompositionAgent",
    "WritingAgent",
    "WritingAgentNoCitations",
]
