from __future__ import annotations

from podcast_agent.config import PipelineConfig


def test_pipeline_config_beat_defaults() -> None:
    config = PipelineConfig()

    assert config.target_script_source_ratio == 0.25
    assert config.max_target_script_words == 20000
    assert config.section_beat_target_words == 3000
    assert config.beat_evidence_window_size == 20
