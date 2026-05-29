from __future__ import annotations

import argparse
import asyncio

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.pipeline.resume import resume_from_scene_discovery_stage


async def _resume_from_scene_discovery(project_id: str) -> None:
    try:
        await resume_from_scene_discovery_stage(
            project_id,
            stage_label="scene discovery",
            settings_factory=Settings,
            orchestrator_cls=PipelineOrchestrator,
        )
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("Resume failed from scene discovery onward:"):
            raise
        raise RuntimeError(f"Resume failed from scene discovery onward: {message}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an existing run from scene discovery onward using existing "
            "theme, corpus, substrate, tagged primitive, and actor artifacts "
            "with strict integrity checks."
        )
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Run directory name under the configured artifact root.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_resume_from_scene_discovery(args.project_id))


if __name__ == "__main__":
    main()
