from __future__ import annotations

import argparse
import asyncio

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator
from podcast_agent.pipeline.resume import resume_from_narrative_strategy_stage


async def _resume_from_narrative_strategy(project_id: str) -> None:
    await resume_from_narrative_strategy_stage(
        project_id,
        stage_label="narrative strategy",
        settings_factory=Settings,
        orchestrator_cls=PipelineOrchestrator,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an existing run from narrative strategy onward using existing "
            "tagged primitives, scene discovery, and actor artifacts with strict "
            "integrity checks."
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
    asyncio.run(_resume_from_narrative_strategy(args.project_id))


if __name__ == "__main__":
    main()
