from __future__ import annotations

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient
from podcast_agent.llm.openai_compatible import RoutingLLMClient


class DummyResponse(BaseModel):
    provider: str


class FakeClient(LLMClient):
    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def generate_json(self, schema_name: str, instructions: str, payload: dict, response_model: type[BaseModel]):
        return response_model.model_validate({"provider": self.provider})


def test_routing_llm_client_selects_provider_by_schema() -> None:
    router = RoutingLLMClient(
        default_provider="openai-compatible",
        clients={
            "openai-compatible": FakeClient("openai-compatible"),
            "anthropic": FakeClient("anthropic"),
        },
        provider_overrides={"book_analysis": "anthropic"},
    )

    result = router.generate_json(
        schema_name="book_analysis",
        instructions="",
        payload={},
        response_model=DummyResponse,
    )
    assert result.provider == "anthropic"

    fallback = router.generate_json(
        schema_name="series_plan",
        instructions="",
        payload={},
        response_model=DummyResponse,
    )
    assert fallback.provider == "openai-compatible"
