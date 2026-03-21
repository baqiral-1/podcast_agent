"""Podcast agent package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["PipelineOrchestrator"]

if TYPE_CHECKING:
    from podcast_agent.pipeline.orchestrator import PipelineOrchestrator


def __getattr__(name: str):
    if name == "PipelineOrchestrator":
        from podcast_agent.pipeline.orchestrator import PipelineOrchestrator

        return PipelineOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
