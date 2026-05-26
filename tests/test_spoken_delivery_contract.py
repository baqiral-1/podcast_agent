from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from _section_progression_helpers import make_section_progression
from podcast_agent.llm.heuristic import HeuristicLLMClient
from podcast_agent.pipeline.orchestrator import (
    PipelineOrchestrator,
    _build_spoken_delivery_batches,
    _extract_previous_spoken_tail,
)
from podcast_agent.schemas.models import (
    ArchitectureSection,
    BookRecord,
    EpisodeArchitecture,
    EpisodeScript,
    FramingBlock,
    NarrativeState,
    ProseSection,
    ThematicProject,
)


class DummyTTSClient:
    def set_run_logger(self, run_logger) -> None:  # pragma: no cover - stub
        self.run_logger = run_logger


def _framing() -> FramingBlock:
    return FramingBlock(
        opening_image="Image",
        threat_or_unresolved_action="Threat",
        opening_question="Question",
        handoff_scene_card_id="scene_1",
    )


def _architecture() -> EpisodeArchitecture:
    payload = {
        "episode_number": 1,
        "major_turn_section_id": "section_1",
        "sections": [
            {
                "section_id": "section_1",
                "purpose": "opening",
                "anchor": "Anchor 1",
                "approx_runtime_minutes": 1.0,
                "primitive_ids": ["primitive_1"],
                "section_question": "Q1",
                "section_resolution": "R1",
                "entry_state": "S1",
                "exit_state": "S2",
                "transition_logic": "T1",
                "argument_role": "frame",
                "inference_mode": "scene_first",
                "pressure_type": "mass_political",
                "resolution_type": "escalation",
                "section_progression": make_section_progression("setup", label="section_1"),
            },
            {
                "section_id": "section_2",
                "purpose": "closing",
                "anchor": "Anchor 2",
                "approx_runtime_minutes": 1.0,
                "primitive_ids": ["primitive_1"],
                "section_question": "Q2",
                "section_resolution": "R2",
                "entry_state": "S2",
                "exit_state": "S3",
                "transition_logic": "T2",
                "depends_on_section_ids": ["section_1"],
                "argument_role": "close",
                "inference_mode": "aftermath_first",
                "pressure_type": "mass_political",
                "resolution_type": "containment",
                "section_progression": make_section_progression("close", label="section_2"),
            },
        ],
    }
    return EpisodeArchitecture.model_construct(
        episode_number=payload["episode_number"],
        major_turn_section_id=payload["major_turn_section_id"],
        allowed_recurring_primitive_ids=[],
        forbidden_redundancies=[],
        sections=[
            ArchitectureSection.model_validate(section)
            for section in payload["sections"]
        ],
        architecture_notes=[],
    )


def _narrative_state(*, episode_number: int, carry_label: str) -> NarrativeState:
    return NarrativeState.model_validate(
        {
            "project_id": "proj",
            "next_episode_number": episode_number,
            "listener": {
                "carry_forward_memory": [
                    {
                        "item_id": f"carry_{episode_number}",
                        "label": carry_label,
                        "kind": "takeaway",
                        "source_episode_number": max(episode_number - 1, 0),
                        "priority": "high",
                        "desired_surface": "recap" if episode_number == 1 else "closing",
                        "recommended_action": "remind"
                        if episode_number == 1
                        else "keep_live",
                    }
                ],
                "last_episode_takeaway": carry_label,
            },
            "host": {},
        }
    )


def test_rewrite_for_speech_passes_continuity_tail_for_later_batches(
    monkeypatch, tmp_path: Path
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: heuristic,
    )
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

    orchestrator = PipelineOrchestrator()
    payloads: list[dict] = []

    def fake_spoken_run(payload: dict):
        payloads.append(payload)
        text = (
            "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
            if len(payloads) == 1
            else "Later batch."
        )
        return orchestrator.spoken_delivery_agent.response_model.model_validate(
            {
                "text": f"spoken::{text}",
                "speech_hints": {
                    "style": "measured",
                    "intensity": "light",
                    "pause_before_ms": 100,
                    "pause_after_ms": 200,
                    "pace": "slower",
                    "pronunciation_hints": [],
                    "emphasis_targets": ["spoken"],
                    "render_strategy": "plain",
                },
            }
        )

    orchestrator.spoken_delivery_agent.run = fake_spoken_run

    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="First section",
            ),
            ProseSection(
                section_id="section_2",
                scene_card_ids=["scene_2"],
                movement_goal="discover",
                text="Second section",
            ),
        ],
    )
    project = ThematicProject(
        project_id="proj",
        theme="War on terror",
        books=[
            BookRecord(
                book_id="b1",
                title="Book 1",
                author="Author",
                source_path="/tmp/book.txt",
                source_type="txt",
            )
        ],
    )
    ep_dir = tmp_path / "episodes" / "1"
    ep_dir.mkdir(parents=True, exist_ok=True)

    spoken = asyncio.run(
        orchestrator._rewrite_for_speech(
            1,
            script,
            project,
            ep_dir,
            tmp_path,
            architecture=_architecture(),
            narrative_state_pre=_narrative_state(
                episode_number=1, carry_label="Earlier pressure still matters."
            ),
            narrative_state_post=_narrative_state(
                episode_number=2, carry_label="Unfinished pressure still matters."
            ),
            continuity_contract_pre={
                "recap_items": [
                    {"item_id": "carry_1", "label": "Earlier pressure still matters."}
                ]
            },
            continuity_contract_post={
                "must_leave_live": [
                    {
                        "item_id": "carry_2",
                        "label": "Unfinished pressure still matters.",
                    }
                ]
            },
        )
    )

    assert len(payloads) == 2
    assert payloads[0]["section"]["section_id"] == "section_1"
    assert payloads[1]["section"]["section_id"] == "section_2"
    assert payloads[0]["section"]["anchor"] == "Anchor 1"
    assert payloads[1]["section"]["section_progression"]["stage"] == "close"
    assert payloads[0]["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
    assert payloads[1]["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
    assert payloads[0]["section"]["term_explanations"] == []
    assert "host_presence_beats" not in payloads[0]["section"]
    assert "batch_index" not in payloads[0]
    assert "batch_index" not in payloads[1]
    assert "batch_count" not in payloads[0]
    assert "batch_count" not in payloads[1]
    assert "previous_spoken_text" not in payloads[0]
    assert "previous_spoken_text" not in payloads[1]
    assert "previous_spoken_tail" not in payloads[0]
    assert payloads[1]["previous_spoken_tail"] == (
        "Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    )
    assert "upcoming_batches_summary" not in payloads[0]
    assert "upcoming_batches_summary" not in payloads[1]
    assert [section.section_id for section in spoken.sections] == [
        "section_1",
        "section_2",
    ]
    assert [section.text for section in spoken.sections] == [
        "spoken::First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
        "spoken::Later batch.",
    ]
    spoken_diagnostics = (ep_dir / "continuity_spoken_diagnostics.json").read_text()
    assert "carry_1" in spoken_diagnostics


def test_style_audit_episode_uses_section_anchor_payload(
    monkeypatch, tmp_path: Path
) -> None:
    heuristic = HeuristicLLMClient()
    monkeypatch.setattr(
        "podcast_agent.pipeline.orchestrator.build_llm_client",
        lambda settings: heuristic,
    )
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

    orchestrator = PipelineOrchestrator()
    payloads: list[dict] = []

    def fake_style_audit_run(payload: dict):
        payloads.append(payload)
        return orchestrator.style_audit_agent.response_model.model_validate(
            {
                "episode_number": payload["episode_number"],
                "sections": [
                    {
                        "section_id": "section_1",
                        "edited_text": "Edited first section.",
                        "edit_notes": ["Trimmed a repeated landing."],
                    },
                    {
                        "section_id": "section_2",
                        "edited_text": "Edited second section.",
                        "edit_notes": ["Tightened the close."],
                    },
                ],
                "episode_warnings": [],
            }
        )

    orchestrator.style_audit_agent.run = fake_style_audit_run

    script = EpisodeScript(
        episode_number=1,
        title="Episode 1",
        framing=_framing(),
        prose_sections=[
            ProseSection(
                section_id="section_1",
                scene_card_ids=["scene_1"],
                movement_goal="discover",
                text="First section",
            ),
            ProseSection(
                section_id="section_2",
                scene_card_ids=["scene_2"],
                movement_goal="discover",
                text="Second section",
            ),
        ],
    )
    ep_dir = tmp_path / "episodes" / "1"
    ep_dir.mkdir(parents=True, exist_ok=True)

    audited_script = asyncio.run(
        orchestrator._style_audit_episode(
            1,
            script,
            _architecture(),
            ep_dir,
            tmp_path,
            continuity_contract_pre={
                "recap_items": [
                    {"item_id": "carry_1", "label": "Earlier pressure still matters."}
                ]
            },
            continuity_contract_post={
                "must_leave_live": [
                    {
                        "item_id": "carry_2",
                        "label": "Unfinished pressure still matters.",
                    }
                ]
            },
        )
    )

    assert len(payloads) == 1
    assert [section["section_id"] for section in payloads[0]["sections"]] == [
        "section_1",
        "section_2",
    ]
    assert [section["anchor"] for section in payloads[0]["sections"]] == [
        "Anchor 1",
        "Anchor 2",
    ]
    assert payloads[0]["continuity_contract_pre"]["recap_items"][0]["item_id"] == "carry_1"
    assert payloads[0]["continuity_contract_post"]["must_leave_live"][0]["item_id"] == "carry_2"
    assert payloads[0]["sections"][0]["term_explanations"] == []
    assert "host_presence_beats" not in payloads[0]["sections"][0]
    assert [section.text for section in audited_script.prose_sections] == [
        "Edited first section.",
        "Edited second section.",
    ]
    script_diagnostics = (ep_dir / "continuity_script_diagnostics.json").read_text()
    assert "carry_1" in script_diagnostics


def test_extract_previous_spoken_tail_uses_complete_sentence_tail() -> None:
    text = (
        "First sentence. Second sentence has six words total. "
        "Third one closes here. Fourth sentence lands cleanly. Fifth sentence stays out."
    )
    assert _extract_previous_spoken_tail(text) == (
        "Second sentence has six words total. Third one closes here. "
        "Fourth sentence lands cleanly. Fifth sentence stays out."
    )


def test_extract_previous_spoken_tail_omits_incomplete_or_oversized_tails() -> None:
    assert _extract_previous_spoken_tail("No sentence boundary here") is None
    long_sentence = " ".join(["word"] * 91) + "."
    assert _extract_previous_spoken_tail(long_sentence) is None


def test_build_spoken_delivery_batches_uses_four_batches_by_default() -> None:
    sections = [
        ProseSection(
            section_id=f"section_{idx}",
            scene_card_ids=[f"scene_{idx}"],
            movement_goal=f"Goal {idx}",
            text=" ".join(["word"] * word_count),
        )
        for idx, word_count in enumerate([100, 200, 300, 400], start=1)
    ]

    batches = _build_spoken_delivery_batches(sections)

    assert [[section.section_id for section in batch] for batch in batches] == [
        ["section_1"],
        ["section_2"],
        ["section_3"],
        ["section_4"],
    ]
