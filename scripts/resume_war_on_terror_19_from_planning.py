from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_agent.config import Settings
from podcast_agent.pipeline.orchestrator import PipelineOrchestrator, _save_json
from podcast_agent.schemas.models import (
    EpisodePlan,
    EpisodeFraming,
    EpisodeAssignment,
    ProjectStatus,
    SpokenScript,
    SynthesisMap,
    ThematicCorpus,
    ThematicProject,
    NarrativeStrategy,
)


PROJECT_ID = "war_on_terror_19"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_episode_mapping_from_log(
    run_log: Path,
) -> tuple[dict[str, int], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    request_to_episode: dict[str, int] = {}
    request_to_assignment: dict[str, dict[str, Any]] = {}
    latest_response_by_episode: dict[int, dict[str, Any]] = {}

    for line in run_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "llm_request" and payload.get("schema_name") == "episode_planning":
            request_uuid = payload.get("request_uuid")
            if not request_uuid:
                continue
            user_text = payload.get("user_text", "{}")
            request_payload = json.loads(user_text).get("payload", {})
            assignment = request_payload.get("episode_assignment", {})
            episode_number = assignment.get("episode_number")
            if isinstance(episode_number, int):
                request_to_episode[request_uuid] = episode_number
                request_to_assignment[request_uuid] = assignment
            continue
        if event_type == "llm_response" and payload.get("schema_name") == "episode_planning":
            request_uuid = payload.get("request_uuid")
            if not request_uuid or request_uuid not in request_to_episode:
                continue
            episode_number = request_to_episode[request_uuid]
            latest_response_by_episode[episode_number] = {
                "response": payload.get("response", {}),
                "assignment": request_to_assignment.get(request_uuid, {}),
            }

    return request_to_episode, request_to_assignment, latest_response_by_episode


def _hydrate_recovered_episode(
    response_payload: dict[str, Any],
    assignment_payload: dict[str, Any],
    spoken_wpm: float,
) -> EpisodePlan:
    assignment = EpisodeAssignment.model_validate(assignment_payload)
    partial = dict(response_payload)
    target_minutes = float(partial.get("target_duration_minutes", 0.0))
    unresolved_questions = partial.get("unresolved_questions") or [
        "What unresolved historical contingency most shaped the outcome in this episode?"
    ]
    payoff_shape = partial.get("payoff_shape") or (
        "Deliver a provisional answer to the driving question while naming the key uncertainty."
    )
    return EpisodePlan.model_validate({
        **partial,
        "episode_number": assignment.episode_number,
        "title": partial.get("title") or assignment.title,
        "driving_question": assignment.driving_question,
        "thematic_focus": partial.get("thematic_focus") or assignment.thematic_focus,
        "axis_ids": assignment.axis_ids,
        "insight_ids": assignment.insight_ids,
        "unresolved_questions": unresolved_questions,
        "payoff_shape": payoff_shape,
        "episode_strategy": partial.get("episode_strategy") or assignment.episode_strategy,
        "target_word_count": int(round(target_minutes * spoken_wpm)),
    })


async def main() -> None:
    settings = Settings()
    orchestrator = PipelineOrchestrator(settings)
    project_dir = settings.pipeline.artifact_root / PROJECT_ID
    run_log = project_dir / "run.log"

    if not project_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {project_dir}")

    project = ThematicProject.model_validate(_load_json(project_dir / "thematic_project.json"))
    corpus = ThematicCorpus.model_validate(_load_json(project_dir / "thematic_corpus.json"))
    synthesis_map = SynthesisMap.model_validate(_load_json(project_dir / "synthesis_map.json"))
    strategy = NarrativeStrategy.model_validate(_load_json(project_dir / "narrative_strategy.json"))

    _, _, recovered_responses = _build_episode_mapping_from_log(run_log)
    recovered_episodes = set(recovered_responses.keys())
    expected_episodes = set(range(1, project.episode_count + 1))
    to_replan = expected_episodes - recovered_episodes
    missing_recovered = sorted(expected_episodes - recovered_episodes)
    if missing_recovered:
        raise RuntimeError(
            f"Cannot resume: missing recovered plans for episodes {missing_recovered}"
        )

    forced_config = project.config.model_copy(
        update={
            "skip_grounding": True,
            "skip_audio": True,
        }
    )
    project = project.model_copy(update={"config": forced_config})

    original_run = orchestrator.episode_planning_agent.run

    def run_with_recovery(payload: dict[str, Any]) -> EpisodePlan:
        assignment = payload.get("episode_assignment", {})
        episode_number = assignment.get("episode_number")
        if isinstance(episode_number, int) and episode_number in recovered_responses:
            recovered = recovered_responses[episode_number]
            return _hydrate_recovered_episode(
                recovered["response"],
                recovered["assignment"],
                float(settings.pipeline.spoken_words_per_minute),
            )
        return original_run(payload)

    orchestrator.episode_planning_agent.run = run_with_recovery
    orchestrator._bind_run_logger(project_dir)
    orchestrator.run_logger.log(
        "pipeline_resume_start",
        project_id=PROJECT_ID,
        resume_from="episode_planning",
        recovered_episode_count=len(recovered_episodes),
        replanning_episode_numbers=sorted(to_replan),
        skip_grounding=True,
        skip_audio=True,
    )

    if to_replan:
        project = project.model_copy(update={"status": ProjectStatus.PLANNING})
        _save_json(project_dir / "thematic_project.json", project)

        episode_plans = await orchestrator._plan_series(
            project=project,
            synthesis_map=synthesis_map,
            strategy=strategy,
            corpus=corpus,
            project_dir=project_dir,
        )
    else:
        episode_plans = [
            _hydrate_recovered_episode(
                recovered_responses[episode_number]["response"],
                recovered_responses[episode_number]["assignment"],
                float(settings.pipeline.spoken_words_per_minute),
            )
            for episode_number in sorted(expected_episodes)
        ]
        _save_json(project_dir / "series_plan.json", episode_plans)
        orchestrator.run_logger.log(
            "episode_planning_recovered",
            project_id=PROJECT_ID,
            episode_numbers=[plan.episode_number for plan in episode_plans],
            recovered_only=True,
        )

    project = project.model_copy(update={"status": ProjectStatus.PRODUCING})
    _save_json(project_dir / "thematic_project.json", project)

    sem = asyncio.Semaphore(project.config.episode_write_concurrency)
    ep_tasks = [
        orchestrator._produce_episode(plan, project, corpus, project_dir, sem)
        for plan in episode_plans
    ]
    ep_results = await asyncio.gather(*ep_tasks, return_exceptions=True)

    spoken_scripts: list[tuple[int, SpokenScript]] = []
    for result in ep_results:
        if isinstance(result, Exception):
            raise result
        spoken_scripts.append(result)
    spoken_scripts.sort(key=lambda item: item[0])

    orchestrator._write_passage_utilization(
        project=project,
        corpus=corpus,
        episode_plans=episode_plans,
        project_dir=project_dir,
        episode_numbers=[episode_number for episode_number, _ in spoken_scripts],
    )

    framings: dict[int, EpisodeFraming] = {}
    for i, (ep_num, spoken) in enumerate(spoken_scripts):
        prev_summary = spoken_scripts[i - 1][1].arc_plan if i > 0 else None
        next_summary = None
        if i < len(spoken_scripts) - 1:
            next_idx = spoken_scripts[i + 1][0] - 1
            if next_idx < len(episode_plans):
                next_summary = episode_plans[next_idx].thematic_focus
        framing = await orchestrator._frame_episode(
            ep_num,
            len(spoken_scripts),
            spoken,
            prev_summary,
            next_summary,
            project,
            project_dir,
        )
        framings[ep_num] = framing

    audio_sem = asyncio.Semaphore(project.config.tts_concurrency)
    audio_tasks = [
        orchestrator._render_episode_audio(
            ep_num,
            spoken,
            project.config,
            framings.get(ep_num),
            project_dir,
            audio_sem,
            skip_audio=True,
        )
        for ep_num, spoken in spoken_scripts
    ]
    await asyncio.gather(*audio_tasks, return_exceptions=True)

    project = project.model_copy(update={"status": ProjectStatus.COMPLETE})
    _save_json(project_dir / "thematic_project.json", project)
    orchestrator.run_logger.log("pipeline_complete", project_id=PROJECT_ID, resumed=True)

    print(f"Resumed pipeline complete: {PROJECT_ID}")
    print(f"Replanned episodes: {sorted(to_replan)}")


if __name__ == "__main__":
    asyncio.run(main())
