from __future__ import annotations

import argparse
import asyncio

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.pipeline.resume import resume_single_episode_from_writing_stage


async def _resume_single_episode_from_writing(
    project_id: str,
    episode_number: int,
) -> None:
    await resume_single_episode_from_writing_stage(
        project_id,
        episode_number=episode_number,
        settings_factory=Settings,
        orchestrator_cls=PipelineOrchestrator,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an existing run from writing onward for one episode using "
            "persisted retained primitives, planning artifacts, and actor metadata."
        )
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Run directory name under the configured artifact root.",
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        required=True,
        help="Episode number to resume from writing onward.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        _resume_single_episode_from_writing(args.project_id, args.episode_number)
    )


if __name__ == "__main__":
    main()
