"""Tests for package-level exports."""

from __future__ import annotations

import os
import subprocess


def test_podcast_agent_root_package_lazy_loads_pipeline_orchestrator() -> None:
    code = """
import sys
import podcast_agent

print('after_import', 'podcast_agent.pipeline.orchestrator' in sys.modules)
_ = podcast_agent.PipelineOrchestrator
print('after_attr', 'podcast_agent.pipeline.orchestrator' in sys.modules)
"""

    result = subprocess.run(
        ["python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.splitlines() == [
        "after_import False",
        "after_attr True",
    ]
