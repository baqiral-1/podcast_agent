from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.schemas.models import (
    ActorMetadata,
    ActorProfile,
    BookRecord,
    EpisodeArchitecture,
    EpisodePlan,
    ExtractedPassage,
    SceneActor,
    StrategyEpisode,
    ThematicCorpus,
    ThematicProject,
)


class DummyTTSClient:
    def set_run_logger(self, _logger) -> None:
        return None


def _core_primitive_ids() -> list[str]:
    return [f"primitive_{idx}" for idx in range(1, 8)]


def _support_primitive_roles() -> dict[str, str]:
    return {f"support_{idx}": "mechanism" for idx in range(1, 10)}


def _strategy_episode() -> StrategyEpisode:
    return StrategyEpisode.model_validate(
        {
            "episode_number": 1,
            "title": "Episode 1",
            "arc_summary": "The episode tracks a split point cleanly.",
            "unresolved_questions": ["What follows from the hinge?"],
            "episode_spine": {
                "listener_question": "How does the hinge turn?",
                "argument": "The hinge redirects the episode.",
                "core_primitive_ids": _core_primitive_ids(),
                "support_primitive_roles": _support_primitive_roles(),
                "recall_primitive_ids": [],
            },
            "actor_arc_directives": [],
        }
    )


def _episode_architecture() -> EpisodeArchitecture:
    return EpisodeArchitecture.model_validate(
        {
            "episode_number": 1,
            "major_turn_section_id": "section_3",
            "allowed_recurring_primitive_ids": [],
            "forbidden_redundancies": [],
            "sections": [
                {
                    "section_id": "section_1",
                    "purpose": "opening",
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "section_question": "Q1",
                    "section_resolution": "R1",
                    "entry_state": "S1",
                    "exit_state": "S2",
                    "transition_logic": "T1",
                    "depends_on_section_ids": [],
                    "sets_up_section_ids": ["section_2"],
                    "argument_role": "frame",
                    "inference_mode": "scene_first",
                    "recurrence_role": "none",
                    "pressure_type": "constitutional",
                    "resolution_type": "redefinition",
                    "closure_level": "low",
                },
                {
                    "section_id": "section_2",
                    "purpose": "setup",
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "section_question": "Q2",
                    "section_resolution": "R2",
                    "entry_state": "S2",
                    "exit_state": "S3",
                    "transition_logic": "T2",
                    "depends_on_section_ids": ["section_1"],
                    "sets_up_section_ids": ["section_3"],
                    "argument_role": "establish_mechanism",
                    "inference_mode": "mechanism_first",
                    "recurrence_role": "none",
                    "pressure_type": "constitutional",
                    "resolution_type": "escalation",
                    "closure_level": "low",
                },
                {
                    "section_id": "section_3",
                    "purpose": "turn",
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "section_question": "Q3",
                    "section_resolution": "R3",
                    "entry_state": "S3",
                    "exit_state": "S4",
                    "transition_logic": "T3",
                    "depends_on_section_ids": ["section_2"],
                    "sets_up_section_ids": ["section_4"],
                    "argument_role": "test_viability",
                    "inference_mode": "contrast_first",
                    "recurrence_role": "none",
                    "pressure_type": "constitutional",
                    "resolution_type": "reversal",
                    "closure_level": "medium",
                },
                {
                    "section_id": "section_4",
                    "purpose": "closing",
                    "approx_runtime_minutes": 1.0,
                    "primitive_ids": ["primitive_1"],
                    "section_question": "Q4",
                    "section_resolution": "R4",
                    "entry_state": "S4",
                    "exit_state": "S5",
                    "transition_logic": "T4",
                    "depends_on_section_ids": ["section_3"],
                    "sets_up_section_ids": [],
                    "argument_role": "close",
                    "inference_mode": "aftermath_first",
                    "recurrence_role": "none",
                    "pressure_type": "constitutional",
                    "resolution_type": "containment",
                    "closure_level": "high",
                },
            ],
            "architecture_notes": [],
        }
    )


def _episode_plan() -> EpisodePlan:
    return EpisodePlan.model_validate(
        {
            "episode_number": 1,
            "framing": {
                "opening_image": "Image",
                "threat_or_unresolved_action": "Threat",
                "opening_question": "Question",
                "handoff_scene_card_id": "scene_1",
            },
            "scene_cards": [
                {
                    "scene_id": "scene_1",
                    "section_id": "section_1",
                    "title": "Scene 1",
                    "scene_role": "setup",
                    "dominant_primitive_id": "primitive_1",
                    "spine_relation": "set_stakes",
                    "state_effect": "The setup lands.",
                    "entry_image": "Image 1",
                    "local_question": "What has started?",
                    "observable_detail": "Detail 1",
                    "intended_move": "Move 1",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["p1"],
                    "actors": [{"name": "Actor 1", "actor_id": "actor_1", "presence": "primary"}],
                    "estimated_duration_seconds": 60,
                },
                {
                    "scene_id": "scene_2",
                    "section_id": "section_2",
                    "title": "Scene 2",
                    "scene_role": "action",
                    "dominant_primitive_id": "primitive_1",
                    "spine_relation": "spine_advance",
                    "state_effect": "The hinge becomes visible.",
                    "entry_image": "Image 2",
                    "local_question": "Who now gets to decide?",
                    "observable_detail": "Detail 2",
                    "withhold_until": "scene_3",
                    "what_becomes_legible_later": "The constitutional shift makes the next window possible.",
                    "intended_move": "Move 2",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["p2"],
                    "actors": [{"name": "Actor 2", "actor_id": "actor_2", "presence": "primary"}],
                    "estimated_duration_seconds": 60,
                },
                {
                    "scene_id": "scene_3",
                    "section_id": "section_3",
                    "title": "Scene 3",
                    "scene_role": "turn",
                    "dominant_primitive_id": "primitive_1",
                    "spine_relation": "turn",
                    "state_effect": "The consequence starts to cash out.",
                    "entry_image": "Image 3",
                    "local_question": "What does the hinge now enable?",
                    "observable_detail": "Detail 3",
                    "intended_move": "Move 3",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["p3"],
                    "actors": [{"name": "Actor 3", "actor_id": "actor_3", "presence": "primary"}],
                    "estimated_duration_seconds": 60,
                },
                {
                    "scene_id": "scene_4",
                    "section_id": "section_4",
                    "title": "Scene 4",
                    "scene_role": "closing",
                    "dominant_primitive_id": "primitive_1",
                    "spine_relation": "show_consequence",
                    "state_effect": "The consequence lands.",
                    "entry_image": "Image 4",
                    "local_question": "What remains after the hinge?",
                    "observable_detail": "Detail 4",
                    "intended_move": "Move 4",
                    "primitive_ids": ["primitive_1"],
                    "passage_ids": ["p4"],
                    "actors": [{"name": "Actor 4", "actor_id": "actor_4", "presence": "primary"}],
                    "estimated_duration_seconds": 60,
                },
            ],
            "target_word_count": 400,
        }
    )


def _corpus() -> ThematicCorpus:
    return ThematicCorpus(
        project_id="proj",
        passages_by_axis={
            "axis_1": [
                ExtractedPassage(
                    passage_id=f"p{idx}",
                    book_id="book_1",
                    chunk_ids=[f"c{idx}"],
                    text=f"Trimmed text {idx}",
                    full_text=f"Full text evidence {idx}.",
                    chapter_ref=f"Chapter {idx}",
                    axis_id="axis_1",
                )
                for idx in range(1, 5)
            ]
        },
    )


def _actor_metadata() -> ActorMetadata:
    return ActorMetadata(
        project_id="proj",
        actors=[
            ActorProfile(actor_id=f"actor_{idx}", display_name=f"Actor {idx}", actor_type="person")
            for idx in range(1, 5)
        ],
    )


def _project() -> ThematicProject:
    return ThematicProject(
        project_id="proj",
        theme="Theme",
        books=[
            BookRecord(
                book_id="book_1",
                title="Book",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
    )


def _build_orchestrator(monkeypatch: pytest.MonkeyPatch) -> PipelineOrchestrator:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.build_llm_client", lambda settings: heuristic)
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.PGVectorRetrieval",
        lambda settings, run_logger=None: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.RetrievalService",
        lambda settings, vector_store: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_tts_client",
        lambda settings: DummyTTSClient(),
    )
    return PipelineOrchestrator()


def test_write_episode_splits_large_episode_into_two_sequential_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(monkeypatch)
    captured_payloads: list[dict] = []

    def fake_writing_run(payload: dict):
        captured_payloads.append(payload)
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "scene_prose": [
                    {
                        "scene_card_id": scene["scene_id"],
                        "movement_goal": "discover",
                        "text": f"Draft for {scene['scene_id']}.",
                        "source_book_ids": ["book_1"],
                    }
                    for scene in payload["plan"]["scene_cards"]
                ]
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    plan = _episode_plan()
    strategy_episode = _strategy_episode()
    architecture = _episode_architecture()
    corpus = _corpus()
    actor_metadata = _actor_metadata()
    project = _project()
    ep_dir = tmp_path / "episodes" / "1"

    script = asyncio.run(
        orchestrator._write_episode(
            plan,
            strategy_episode,
            architecture,
            project,
            corpus,
            ep_dir,
            tmp_path,
            actor_metadata,
        )
    )

    assert len(captured_payloads) == 2
    first_payload, second_payload = captured_payloads
    assert [scene["scene_id"] for scene in first_payload["plan"]["scene_cards"]] == [
        "scene_1",
        "scene_2",
    ]
    assert [scene["scene_id"] for scene in second_payload["plan"]["scene_cards"]] == [
        "scene_3",
        "scene_4",
    ]
    assert [passage["passage_id"] for passage in first_payload["passages"]] == ["p1", "p2"]
    assert [passage["passage_id"] for passage in second_payload["passages"]] == ["p3", "p4"]
    assert [section["section_id"] for section in first_payload["architecture"]["sections"]] == [
        "section_1",
        "section_2",
    ]
    assert [section["section_id"] for section in second_payload["architecture"]["sections"]] == [
        "section_3",
        "section_4",
    ]
    assert "state_effect" not in first_payload["plan"]["scene_cards"][0]
    assert "local_question" not in first_payload["plan"]["scene_cards"][0]
    assert "what_becomes_legible_later" not in second_payload["plan"]["scene_cards"][0]
    assert "prior_window_continuity" not in first_payload
    continuity = second_payload["prior_window_continuity"]
    assert continuity["completed_scene_count"] == 2
    assert continuity["completed_scene_ids"] == ["scene_1", "scene_2"]
    assert set(continuity["last_completed_scene"]) == {
        "scene_id",
        "section_id",
        "scene_role",
        "spine_relation",
        "withhold_until",
        "intended_move",
    }
    assert continuity["last_completed_scene"]["scene_id"] == "scene_2"
    assert continuity["live_unresolved_questions"] == ["What follows from the hinge?"]
    assert continuity["carry_forward_threads"][0].startswith(
        "Keep this tension live without restating it explicitly:"
    )
    assert continuity["tail_excerpt"] == "Draft for scene_1. Draft for scene_2."
    assert [actor["actor_id"] for actor in second_payload["actor_metadata"]["actors"]] == [
        "actor_3",
        "actor_4",
    ]
    assert (tmp_path / "stage_artifacts" / "write_episode_1_part_1" / "input.json").exists()
    assert (tmp_path / "stage_artifacts" / "write_episode_1_part_2" / "input.json").exists()
    assert [scene_id for section in script.prose_sections for scene_id in section.scene_card_ids] == [
        "scene_1",
        "scene_2",
        "scene_3",
        "scene_4",
    ]


def test_write_episode_retries_failed_second_part_from_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("podcast_agent.pipeline.orchestrator.asyncio.sleep", fake_sleep)
    orchestrator = _build_orchestrator(monkeypatch)
    captured_payloads: list[dict] = []

    def fake_writing_run(payload: dict):
        captured_payloads.append(payload)
        scene_ids = [scene["scene_id"] for scene in payload["plan"]["scene_cards"]]
        if len(captured_payloads) == 2:
            scene_ids = ["scene_renamed", "scene_4"]
        return orchestrator.writing_agent.response_model.model_validate(
            {
                "scene_prose": [
                    {
                        "scene_card_id": scene_id,
                        "movement_goal": "discover",
                        "text": f"Draft for {scene_id}.",
                        "source_book_ids": ["book_1"],
                    }
                    for scene_id in scene_ids
                ]
            }
        )

    orchestrator.writing_agent.run = fake_writing_run

    plan = _episode_plan()
    strategy_episode = _strategy_episode()
    architecture = _episode_architecture()
    corpus = _corpus()
    actor_metadata = _actor_metadata()
    project = _project()
    ep_dir = tmp_path / "episodes" / "1"

    script = asyncio.run(
        orchestrator._write_episode(
            plan,
            strategy_episode,
            architecture,
            project,
            corpus,
            ep_dir,
            tmp_path,
            actor_metadata,
        )
    )

    assert len(captured_payloads) == 4
    assert "prior_window_continuity" not in captured_payloads[0]
    assert "prior_window_continuity" in captured_payloads[1]
    assert "prior_window_continuity" not in captured_payloads[2]
    assert "prior_window_continuity" in captured_payloads[3]
    assert "writing_feedback" not in captured_payloads[0]
    assert "writing_feedback" not in captured_payloads[1]
    assert "writing_feedback" not in captured_payloads[2]
    assert "writing_feedback" in captured_payloads[3]
    assert "scene_renamed" in str(captured_payloads[3]["writing_feedback"])
    assert [scene_id for section in script.prose_sections for scene_id in section.scene_card_ids] == [
        "scene_1",
        "scene_2",
        "scene_3",
        "scene_4",
    ]
