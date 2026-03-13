"""Base class for LLM-backed agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient


class Agent(ABC):
    """Shared base class for JSON-producing agents."""

    schema_name: str
    instructions: str
    response_model: type[BaseModel]

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, payload: dict) -> BaseModel:
        """Execute the agent and validate the returned JSON."""

        return self.llm.generate_json(
            schema_name=self.schema_name,
            instructions=self.instructions,
            payload=payload,
            response_model=self.response_model,
        )

    @abstractmethod
    def build_payload(self, *args, **kwargs) -> dict:
        """Construct the payload sent to the LLM."""
