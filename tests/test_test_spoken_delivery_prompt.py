from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from podcast_agent.schemas.models import SpokenScript


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "test_spoken_delivery_prompt.py"


spec = importlib.util.spec_from_file_location(
    "test_spoken_delivery_prompt",
    SCRIPT_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load script module: {SCRIPT_PATH}")

script_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script_module
spec.loader.exec_module(script_module)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_spoken_delivery_prompt.py"])
    args = script_module._parse_args()
    assert args.run_dir == Path("runs/independence_v13")
    assert args.episode == 1
    assert args.spoken_max_retry_attempts == 1
    assert args.prompt_file is None


def test_default_output_path_uses_prompt_lab_folder():
    output = script_module._default_output_path(Path("runs/example"), 3)
    assert output == Path("runs/example/prompt_lab/episode_3_spoken_script.json")


def test_load_episode_input_prefers_existing_spoken_tts_provider(tmp_path: Path):
    run_dir = tmp_path / "sample_run"
    episode_dir = run_dir / "episodes" / "1"
    episode_dir.mkdir(parents=True)

    (run_dir / "thematic_project.json").write_text(
        json.dumps(
            {
                "config": {
                    "spoken_chunk_max_words": 180,
                    "tts_provider": "kokoro",
                }
            }
        ),
        encoding="utf-8",
    )

    (episode_dir / "episode_script.json").write_text(
        json.dumps(
            {
                "episode_number": 1,
                "title": "Sample",
                "framing": {
                    "opening_image": "Opening image",
                    "threat_or_unresolved_action": "Threat",
                    "opening_question": "Question",
                    "handoff_scene_card_id": "sc_01",
                    "recap": None,
                    "preview": None,
                },
                "prose_sections": [
                    {
                        "section_id": "sc_01",
                        "movement_goal": "Open",
                        "text": "Original prose section.",
                        "scene_card_ids": ["sc_01"],
                        "source_book_ids": [],
                        "citations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    spoken = SpokenScript(
        episode_number=1,
        title="Sample",
        framing={
            "opening_image": "Opening image",
            "threat_or_unresolved_action": "Threat",
            "opening_question": "Question",
            "handoff_scene_card_id": "sc_01",
            "recap": None,
            "preview": None,
        },
        sections=[
            {
                "section_id": "sc_01",
                "segments": [
                    {
                        "segment_id": "sc_01_seg1",
                        "text": "Spoken text",
                        "speaker_role": "primary",
                        "tonal_register": "neutral",
                    }
                ],
                "tonal_register": "neutral",
            }
        ],
        tts_provider="openai-compatible",
    )
    (episode_dir / "spoken_script.json").write_text(
        spoken.model_dump_json(indent=2),
        encoding="utf-8",
    )

    episode_input = script_module._load_episode_input(run_dir, 1)

    assert episode_input.pipeline_config.spoken_chunk_max_words == 180
    assert episode_input.pipeline_config.tts_provider == "kokoro"
    assert episode_input.tts_provider == "openai-compatible"
    assert episode_input.existing_spoken_script is not None
