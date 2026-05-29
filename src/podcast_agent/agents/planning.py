"""Stage 9: episode planning agent."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.prompts import episode_planning_instructions
from podcast_agent.schemas.models import (
    EpisodePlanDraft,
    SceneJob,
    scene_job_budget_for_mode,
)


class EpisodePlanningAgent(Agent):
    """Expands one episode cluster path into framing plus scene cards."""

    schema_name = "episode_planning"
    response_model = EpisodePlanDraft
    note_guidance = (
        "Every `note` must contain both:\n"
        "- one concrete target\n"
        "- one scene-rooted permission that can be translated into natural host phrasing\n"
        "- Write notes anchor-first.\n"
        "- Prefer noun-phrase or very short phrase targets grounded in the card's own concrete material.\n"
        "- Good targets look like: `press conference split`, `hidden ministries`, `empty villages`, `safe door`, `crowd outside the gate`.\n"
        '- Bad targets are sentence-shaped planning notes such as "tell the listener what to watch for" or "state the through-line." If a target reads like a sentence or instruction to yourself, compress it again.\n'
        "- Do not use writer-room control phrases, abstract management verbs, or episode-management phrasing.\n"
        "- Move-type-specific note rules:\n"
        "  - `orient` must identify where, when, or what visible setup is foregrounded first.\n"
        "  - `clarify` must identify the exact term, institution, relationship, or confusion being clarified.\n"
        "  - `contrast` must identify both sides of the contrast.\n"
        "  - `evaluate` must identify the concrete consequence, irony, verdict, or pressure being landed.\n"
        "  - `callback` must identify the earlier image, phrase, promise, actor, or event returning here.\n"
        "  - `naming_note` must identify the exact word, title, office, slogan, or phrase the listener should retain.\n"
        "- Phase-specific note rules:\n"
        "  - `open` should foreground entry, not conclusion.\n"
        "  - `pivot` should say what becomes newly legible after the material lands.\n"
        "  - `close` should leave after-pressure, verdict, or return, not re-explain setup.\n"
        "- Weak vs. strong note examples:\n"
        '  - Weak: `target = state the through-line`; note = "State the through-line."\n'
        '  - Strong: `target = hidden ministries`; note = "Use Bazargan\'s office and the hidden ministries to show that the Council is already governing behind the cabinet."'
    )
    instructions = f"{episode_planning_instructions()}\n\n{note_guidance}"

    def build_instructions(self, payload: dict) -> str:
        project = payload.get("project")
        if not isinstance(project, dict):
            instructions = self.instructions
        else:
            base_instructions = episode_planning_instructions(
                scene_card_target_min=int(project.get("scene_card_target_min", 36)),
                scene_card_target_max=int(project.get("scene_card_target_max", 42)),
            )
            instructions = f"{base_instructions}\n\n{self.note_guidance}"
        if payload:
            instructions = (
                f"{instructions}\n\n"
                "Legacy note: if you see `scene_function` in older planning notes, "
                "treat it as deprecated shorthand and map it onto `scene_role` plus `scene_job`; "
                "do not emit a `scene_function` field."
            )
        return instructions

    def validate_result(self, result: EpisodePlanDraft, payload: dict) -> EpisodePlanDraft:
        if not result.answer_scene_card_id:
            raise ValueError("episode plan must include answer_scene_card_id")
        close_scene_count = sum(
            1 for scene in result.scene_cards if scene.scene_job == SceneJob.CLOSE
        )
        if close_scene_count != 1:
            raise ValueError("episode plan must include exactly one close scene")
        return result

    def build_payload(
        self,
        strategy_episode: dict,
        architecture: dict,
        synthesis_map: dict,
        project_metadata: dict,
        scene_job_budget: dict | None,
        available_passages: list[dict],
        host_policy: dict | None = None,
        narrative_state_pre: dict | None = None,
        continuity_contract_pre: dict | None = None,
        actor_metadata: dict | None = None,
        planning_feedback: dict | None = None,
        field_semantics: dict | None = None,
        excerpts: list[dict] | None = None,
        section_scene_targets: dict[str, dict[str, int | bool]] | None = None,
    ) -> dict:
        payload = {
            "strategy_episode": strategy_episode,
            "architecture": architecture,
            "synthesis_map": synthesis_map,
            "project": project_metadata,
            "available_passages": available_passages,
        }
        if excerpts:
            payload["excerpts"] = excerpts
        if scene_job_budget is None:
            raw_mode = project_metadata.get("podcast_mode", "full")
            scene_job_budget = scene_job_budget_for_mode(raw_mode)
        payload["scene_job_budget"] = scene_job_budget
        if section_scene_targets is not None:
            payload["section_scene_targets"] = section_scene_targets
        if host_policy is not None:
            payload["host_policy"] = host_policy
        if narrative_state_pre is not None:
            payload["narrative_state_pre"] = narrative_state_pre
        if continuity_contract_pre is not None:
            payload["continuity_contract_pre"] = continuity_contract_pre
        if actor_metadata is not None:
            payload["actor_metadata"] = actor_metadata
        if planning_feedback is not None:
            payload["planning_feedback"] = planning_feedback
        if field_semantics is not None:
            payload["field_semantics"] = field_semantics
        return payload
