from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _save_json
from podcast_agent.schemas.models import (
    EpisodePlan,
    ProjectStatus,
    SynthesisMap,
    ThematicCorpus,
    ThematicProject,
    NarrativeStrategy,
)


PROJECT_ID = "war_on_terror_19"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def main() -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(settings)
    project_dir = settings.pipeline.artifact_root / PROJECT_ID

    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    project = ThematicProject.model_validate(_load_json(project_dir / "thematic_project.json"))
    corpus = ThematicCorpus.model_validate(_load_json(project_dir / "thematic_corpus.json"))
    synthesis_map = SynthesisMap.model_validate(_load_json(project_dir / "synthesis_map.json"))
    strategy = NarrativeStrategy.model_validate(_load_json(project_dir / "narrative_strategy.json"))

    forced_config = project.config.model_copy(
        update={
            "skip_grounding": True,
            "skip_audio": True,
        }
    )
    project = project.model_copy(update={"config": forced_config, "status": ProjectStatus.PLANNING})
    _save_json(project_dir / "thematic_project.json", project)

    episode_plans_payload = _load_json(project_dir / "series_plan.json")
    if isinstance(episode_plans_payload, dict) and "episodes" in episode_plans_payload:
        episode_plans = [EpisodePlan.model_validate(item) for item in episode_plans_payload["episodes"]]
    else:
        episode_plans = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
        )

    project = project.model_copy(update={"status": ProjectStatus.PRODUCING})
    _save_json(project_dir / "thematic_project.json", project)

    sem = asyncio.Semaphore(project.config.episode_write_concurrency)
    ep_tasks = [
        orchestrator._produce_episode(plan, project, corpus, project_dir, sem)
        for plan in episode_plans
    ]
    ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)
    for result in ep_results:
        if isinstance(result, Exception):
            raise result

    project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
    _save_json(project_dir / "thematic_project.json", project)


if __name__ == "__main__":
    asyncio.run(main())
