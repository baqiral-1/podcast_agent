from __future__ import annotations

from pydantic import BaseModel, Field

from podcast_agent.agents.chapter_summary import ChapterSummaryResponse
from podcast_agent.langchain.llm import _apply_schema_caps


def test_apply_schema_caps_truncates_chapter_summary_analysis_lists() -> None:
    payload = {
        "summary": "Summary text.",
        "analysis": {
            "themes_touched": [f"theme-{idx}" for idx in range(10)],
            "major_actors": [f"actor-{idx}" for idx in range(11)],
            "key_places": [f"place-{idx}" for idx in range(12)],
            "key_institutions": [f"institution-{idx}" for idx in range(9)],
            "major_tensions": [f"tension-{idx}" for idx in range(7)],
            "causal_shifts": [f"shift-{idx}" for idx in range(7)],
            "narrative_hooks": [f"hook-{idx}" for idx in range(6)],
            "retrieval_keywords": [f"keyword-{idx}" for idx in range(13)],
        },
    }

    capped, truncations = _apply_schema_caps(
        payload, ChapterSummaryResponse, "chapter_summary"
    )

    analysis = capped["analysis"]
    assert len(analysis["themes_touched"]) == 8
    assert len(analysis["major_actors"]) == 8
    assert len(analysis["key_places"]) == 8
    assert len(analysis["key_institutions"]) == 8
    assert len(analysis["major_tensions"]) == 6
    assert len(analysis["causal_shifts"]) == 6
    assert len(analysis["narrative_hooks"]) == 5
    assert len(analysis["retrieval_keywords"]) == 12
    assert any(t["path"] == "analysis.major_actors" for t in truncations)


def test_apply_schema_caps_is_noop_for_non_chapter_summary() -> None:
    class DummyResponse(BaseModel):
        tags: list[str] = Field(default_factory=list, max_length=2)

    payload = {"tags": ["a", "b", "c"]}
    capped, truncations = _apply_schema_caps(payload, DummyResponse, "episode_planning")

    assert capped == payload
    assert truncations == []
