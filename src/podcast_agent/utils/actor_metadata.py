"""Actor metadata normalization, matching, and prompt-subset helpers."""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from podcast_agent.schemas.models import (
    ActorArcDirective,
    ActorMention,
    ActorMetadata,
    ActorProfile,
    ActorRelationship,
    ActorSeriesScope,
    ActorTier,
    BaseSynthesisPrimitive,
    ExcerptArtifact,
    EpisodePlan,
    EpisodePlanDraft,
    NarrativeStrategy,
    SceneActor,
    SceneCard,
    SceneCardDraft,
    StrategyEpisode,
    SynthesisPrimitiveBase,
    SynthesisPrimitivesArtifact,
    ThematicAxis,
)

_ACTOR_TIER_RANK = {
    ActorTier.MAJOR: 0,
    ActorTier.SUPPORTING: 1,
    ActorTier.MINOR: 2,
}
_ACTOR_SCOPE_RANK = {
    ActorSeriesScope.SERIES_WIDE: 0,
    ActorSeriesScope.RECURRING: 1,
    ActorSeriesScope.LOCAL: 2,
}

_VALID_RELATIONSHIP_TYPES = {
    "enables",
    "blocks",
    "pressures",
    "protects",
    "legitimizes",
    "delegitimizes",
    "replaces",
    "absorbs",
    "betrays",
    "other",
}
_VALID_ACTOR_TYPES = {
    "person",
    "institution",
    "faction",
    "military",
    "party",
    "movement",
    "other",
}
_VALID_EVIDENCE_CONFIDENCE = {"high", "medium", "low"}
_VALID_NARRATIVE_FUNCTIONS = {
    "decision_maker",
    "broker",
    "victim",
    "witness",
    "ideologue",
    "commander",
    "administrator",
    "opposition",
    "beneficiary",
    "constraint",
    "catalyst",
    "symbol",
    "other",
}
_ACTOR_PROFILE_FIELDS = set(ActorProfile.model_fields)


@dataclass(frozen=True)
class ActorMatch:
    actor_id: str
    match_type: str
    score: float


def normalize_actor_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.strip().strip(string.punctuation + " \t\r\n")
    normalized = " ".join(normalized.lower().split())
    return normalized


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for r_idx, r_char in enumerate(right, start=1):
        current = [r_idx]
        for l_idx, l_char in enumerate(left, start=1):
            insert_cost = current[l_idx - 1] + 1
            delete_cost = previous[l_idx] + 1
            replace_cost = previous[l_idx - 1] + (0 if l_char == r_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


class ActorMatcher:
    def __init__(self, actor_metadata: ActorMetadata) -> None:
        self._exact: dict[str, str] = {}
        self._display_names: dict[str, str] = {}
        for actor in actor_metadata.actors:
            names = [actor.display_name, *actor.aliases]
            for name in names:
                key = normalize_actor_name(name)
                if not key:
                    continue
                if key not in self._exact:
                    self._exact[key] = actor.actor_id
            display_key = normalize_actor_name(actor.display_name)
            if display_key:
                self._display_names[display_key] = actor.actor_id

    def match(self, value: str) -> ActorMatch | None:
        key = normalize_actor_name(value)
        if not key:
            return None
        actor_id = self._display_names.get(key)
        if actor_id is not None:
            return ActorMatch(actor_id=actor_id, match_type="exact_display_name", score=1.0)
        actor_id = self._exact.get(key)
        if actor_id is not None:
            return ActorMatch(actor_id=actor_id, match_type="exact_alias", score=1.0)
        return self._fuzzy_match(key)

    def _fuzzy_match(self, key: str) -> ActorMatch | None:
        if len(key) < 8:
            return None
        if " " not in key and len(key) < 12:
            return None
        candidates: list[tuple[float, int, str]] = []
        for candidate, actor_id in self._exact.items():
            if abs(len(candidate) - len(key)) > 2:
                continue
            distance = _levenshtein_distance(key, candidate)
            max_len = max(len(key), len(candidate))
            score = 1.0 - (distance / max_len)
            if distance <= 2 and score >= 0.92:
                candidates.append((score, distance, actor_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][0] == best[0]:
            return None
        return ActorMatch(actor_id=best[2], match_type="fuzzy", score=best[0])


def sanitize_actor_metadata_payload(
    payload: dict[str, Any] | None,
    *,
    project_id: str,
    max_actors: int = 60,
) -> tuple[ActorMetadata, dict[str, Any]]:
    raw = payload or {}
    metrics: dict[str, Any] = {
        "raw_actor_count": len(raw.get("actors", []) or []),
        "dropped_actor_count": 0,
        "dropped_duplicate_actor_count": 0,
        "dropped_over_cap_actor_count": 0,
        "actor_type_normalized_count": 0,
        "evidence_confidence_normalized_count": 0,
        "motivations_converted_count": 0,
        "unknown_actor_field_dropped_count": 0,
        "invalid_narrative_function_dropped_count": 0,
        "nested_relationship_flattened_count": 0,
        "dropped_relationship_count": 0,
        "relationship_type_normalized_count": 0,
    }
    indexed_actors: list[tuple[int, ActorProfile]] = []
    seen_actor_ids: set[str] = set()
    raw_top_level_relationships = raw.get("relationships", []) or []
    raw_relationships: list[dict[str, Any]] = []
    for raw_relationship in raw_top_level_relationships:
        if isinstance(raw_relationship, dict):
            raw_relationships.append(raw_relationship)
        else:
            metrics["dropped_relationship_count"] += 1
    for index, raw_actor in enumerate(raw.get("actors", []) or []):
        if not isinstance(raw_actor, dict):
            metrics["dropped_actor_count"] += 1
            continue
        actor_payload = dict(raw_actor)
        actor_type = str(actor_payload.get("actor_type", "other") or "other")
        if actor_type not in _VALID_ACTOR_TYPES:
            actor_payload["actor_type"] = "other"
            metrics["actor_type_normalized_count"] += 1
        confidence = actor_payload.get("evidence_confidence")
        normalized_confidence = _normalize_evidence_confidence(confidence)
        if normalized_confidence is not None and normalized_confidence != confidence:
            actor_payload["evidence_confidence"] = normalized_confidence
            metrics["evidence_confidence_normalized_count"] += 1
        if (
            "goals_or_motivational_pressures" not in actor_payload
            and "motivations" in actor_payload
        ):
            actor_payload["goals_or_motivational_pressures"] = _coerce_string_list(
                actor_payload.get("motivations")
            )
            metrics["motivations_converted_count"] += 1
        if "narrative_functions" in actor_payload:
            sanitized_functions, dropped_function_count = _sanitize_narrative_functions(
                actor_payload.get("narrative_functions")
            )
            actor_payload["narrative_functions"] = sanitized_functions
            metrics["invalid_narrative_function_dropped_count"] += dropped_function_count
        actor_id = actor_payload.get("actor_id")
        for nested_relationship in actor_payload.get("relationships", []) or []:
            if not isinstance(nested_relationship, dict) or not isinstance(actor_id, str):
                continue
            relationship_payload = dict(nested_relationship)
            relationship_payload.setdefault("source_actor_id", actor_id)
            raw_relationships.append(relationship_payload)
            metrics["nested_relationship_flattened_count"] += 1
        actor_payload, dropped_fields = _strip_unknown_actor_fields(actor_payload)
        metrics["unknown_actor_field_dropped_count"] += dropped_fields
        try:
            actor = ActorProfile.model_validate(actor_payload)
        except ValidationError:
            metrics["dropped_actor_count"] += 1
            continue
        if actor.actor_id in seen_actor_ids:
            metrics["dropped_duplicate_actor_count"] += 1
            continue
        seen_actor_ids.add(actor.actor_id)
        indexed_actors.append((index, actor))

    indexed_actors.sort(
        key=lambda item: (
            _ACTOR_TIER_RANK[item[1].narrative_tier],
            _ACTOR_SCOPE_RANK[item[1].series_scope],
            item[0],
        )
    )
    metrics["dropped_over_cap_actor_count"] = max(0, len(indexed_actors) - max_actors)
    actors = [actor for _, actor in indexed_actors[:max_actors]]
    actor_ids = {actor.actor_id for actor in actors}

    relationships: list[ActorRelationship] = []
    for raw_relationship in raw_relationships:
        relationship_payload = dict(raw_relationship)
        if relationship_payload.get("source_actor_id") not in actor_ids:
            metrics["dropped_relationship_count"] += 1
            continue
        if relationship_payload.get("target_actor_id") not in actor_ids:
            metrics["dropped_relationship_count"] += 1
            continue
        if relationship_payload.get("relationship_type") not in _VALID_RELATIONSHIP_TYPES:
            relationship_payload["relationship_type"] = "other"
            metrics["relationship_type_normalized_count"] += 1
        try:
            relationships.append(ActorRelationship.model_validate(relationship_payload))
        except ValidationError:
            metrics["dropped_relationship_count"] += 1

    unresolved_mentions: list[ActorMention] = []
    for raw_mention in raw.get("unresolved_mentions", []) or []:
        try:
            unresolved_mentions.append(ActorMention.model_validate(raw_mention))
        except ValidationError:
            continue

    metadata = ActorMetadata(
        project_id=project_id,
        actors=actors,
        relationships=relationships,
        unresolved_mentions=unresolved_mentions,
        quality_notes=[
            str(note)
            for note in raw.get("quality_notes", []) or []
            if str(note).strip()
        ],
    )
    return metadata, metrics


def _normalize_evidence_confidence(value: Any) -> str | None:
    if value in _VALID_EVIDENCE_CONFIDENCE:
        return str(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if value >= 0.75:
            return "high"
        if value >= 0.4:
            return "medium"
        return "low"
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _sanitize_narrative_functions(value: Any) -> tuple[list[str], int]:
    if value is None:
        return [], 0
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return [], 1
    filtered: list[str] = []
    dropped = 0
    for item in values:
        normalized = str(item).strip()
        if not normalized or normalized not in _VALID_NARRATIVE_FUNCTIONS:
            dropped += 1
            continue
        filtered.append(normalized)
    return filtered, dropped


def _strip_unknown_actor_fields(actor_payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    filtered = {
        key: value
        for key, value in actor_payload.items()
        if key in _ACTOR_PROFILE_FIELDS
    }
    return filtered, len(actor_payload) - len(filtered)


def clean_axis_actor_ids(
    axes: list[ThematicAxis],
    actor_metadata: ActorMetadata,
) -> tuple[list[ThematicAxis], dict[str, Any]]:
    valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
    dropped = 0
    cleaned: list[ThematicAxis] = []
    for axis in axes:
        actor_ids: list[str] = []
        seen: set[str] = set()
        for actor_id in axis.actor_ids:
            if actor_id not in valid_actor_ids:
                dropped += 1
                continue
            if actor_id in seen:
                continue
            seen.add(actor_id)
            actor_ids.append(actor_id)
        cleaned.append(axis.model_copy(update={"actor_ids": actor_ids}))
    return cleaned, {"dropped_axis_actor_ref_count": dropped}


def clean_synthesis_primitive_actor_links(
    artifact: SynthesisPrimitivesArtifact,
    actor_metadata: ActorMetadata,
) -> tuple[SynthesisPrimitivesArtifact, dict[str, Any]]:
    valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
    metrics = {"unknown_actor_ids": 0}
    cleaned_primitives: list[BaseSynthesisPrimitive] = []
    for primitive in artifact.primitives:
        actor_ids = _filter_known_ids(primitive.actor_ids, valid_actor_ids, metrics)
        cleaned_primitives.append(primitive.model_copy(update={"actor_ids": actor_ids}))
    return artifact.model_copy(update={"primitives": cleaned_primitives}), metrics


def clean_excerpt_actor_links(
    artifact: ExcerptArtifact,
    actor_metadata: ActorMetadata,
) -> tuple[ExcerptArtifact, dict[str, Any]]:
    valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
    metrics = {"unknown_actor_ids": 0}
    cleaned_excerpts = []
    for excerpt in artifact.excerpts:
        actor_ids = _filter_known_ids(excerpt.actor_ids, valid_actor_ids, metrics)
        cleaned_excerpts.append(excerpt.model_copy(update={"actor_ids": actor_ids}))
    return artifact.model_copy(update={"excerpts": cleaned_excerpts}), metrics


def clean_strategy_actor_links(
    strategy_episodes: list[StrategyEpisode],
    actor_metadata: ActorMetadata,
) -> tuple[list[StrategyEpisode], dict[str, Any]]:
    valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
    metrics = {"unknown_actor_ids": 0, "unknown_thread_actor_ids": 0}
    cleaned: list[StrategyEpisode] = []
    for episode in strategy_episodes:
        actor_arc_directives = []
        seen_actor_ids: set[str] = set()
        for actor in episode.actor_arc_directives:
            if actor.actor_id not in valid_actor_ids:
                metrics["unknown_actor_ids"] += 1
                continue
            if actor.actor_id in seen_actor_ids:
                continue
            seen_actor_ids.add(actor.actor_id)
            actor_arc_directives.append(actor)
        update: dict[str, Any] = {"actor_arc_directives": actor_arc_directives}
        thread = episode.human_thread
        if thread is not None:
            directive_actor_ids = {d.actor_id for d in actor_arc_directives}
            members = []
            for member in thread.members:
                member_update: dict[str, Any] = {}
                if member.actor_id is not None and member.actor_id not in valid_actor_ids:
                    metrics["unknown_thread_actor_ids"] += 1
                    member_update["actor_id"] = None
                if (
                    member.arc_actor_id is not None
                    and member.arc_actor_id not in directive_actor_ids
                ):
                    metrics["unknown_thread_actor_ids"] += 1
                    member_update["arc_actor_id"] = None
                members.append(
                    member.model_copy(update=member_update)
                    if member_update
                    else member
                )
            update["human_thread"] = thread.model_copy(update={"members": members})
        cleaned.append(episode.model_copy(update=update))
    return cleaned, metrics


def _fallback_actor_plain_gloss(actor: ActorProfile) -> str:
    text = str(actor.description or "").strip()
    if not text:
        return actor.actor_type.replace("_", " ")
    for delimiter in (".", ";", ":"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0].strip()
            break
    words = text.split()
    if len(words) > 18:
        text = " ".join(words[:18]).rstrip(",")
    return text


def clean_narrative_strategy_actor_links(
    strategy: NarrativeStrategy,
    actor_metadata: ActorMetadata,
) -> tuple[NarrativeStrategy, dict[str, Any]]:
    actor_by_id = {actor.actor_id: actor for actor in actor_metadata.actors}
    valid_actor_ids = set(actor_by_id)
    metrics = {
        "unknown_actor_ids": 0,
        "unknown_actor_registry_refs": 0,
        "fallback_actor_gloss_count": 0,
    }

    cleaned_registry = []
    seen_registry_actor_ids: set[str] = set()
    for item in strategy.series_actor_explanation_registry:
        if item.actor_id not in valid_actor_ids:
            metrics["unknown_actor_ids"] += 1
            continue
        if item.actor_id in seen_registry_actor_ids:
            continue
        seen_registry_actor_ids.add(item.actor_id)
        preferred_plain_gloss = str(item.preferred_plain_gloss or "").strip()
        if not preferred_plain_gloss:
            preferred_plain_gloss = _fallback_actor_plain_gloss(
                actor_by_id[item.actor_id]
            )
            metrics["fallback_actor_gloss_count"] += 1
        cleaned_registry.append(
            item.model_copy(update={"preferred_plain_gloss": preferred_plain_gloss})
        )

    valid_registry_actor_ids = {item.actor_id for item in cleaned_registry}
    cleaned_episodes, episode_metrics = clean_strategy_actor_links(
        strategy.episodes,
        actor_metadata,
    )
    metrics["unknown_actor_ids"] += int(episode_metrics.get("unknown_actor_ids", 0))

    updated_episodes: list[StrategyEpisode] = []
    for episode in cleaned_episodes:
        introduce_actor_ids = _filter_known_ids(
            episode.authorial_contract.introduce_actor_ids,
            valid_registry_actor_ids,
            metrics={"unknown_actor_ids": 0},
        )
        remind_actor_ids = _filter_known_ids(
            episode.authorial_contract.remind_actor_ids,
            valid_registry_actor_ids,
            metrics={"unknown_actor_ids": 0},
        )
        dropped_registry_refs = (
            len(episode.authorial_contract.introduce_actor_ids) - len(introduce_actor_ids)
        ) + (
            len(episode.authorial_contract.remind_actor_ids) - len(remind_actor_ids)
        )
        if dropped_registry_refs:
            metrics["unknown_actor_registry_refs"] += dropped_registry_refs
        updated_episodes.append(
            episode.model_copy(
                update={
                    "authorial_contract": episode.authorial_contract.model_copy(
                        update={
                            "introduce_actor_ids": introduce_actor_ids,
                            "remind_actor_ids": remind_actor_ids,
                        }
                    )
                }
            )
        )

    return (
        strategy.model_copy(
            update={
                "series_actor_explanation_registry": cleaned_registry,
                "episodes": updated_episodes,
            }
        ),
        metrics,
    )


def clean_scene_actor_links(
    plan: EpisodePlanDraft | EpisodePlan,
    actor_metadata: ActorMetadata,
    actor_arc_directives: list[ActorArcDirective],
) -> tuple[EpisodePlanDraft | EpisodePlan, dict[str, Any]]:
    matcher = ActorMatcher(actor_metadata)
    valid_actor_ids = {actor.actor_id for actor in actor_metadata.actors}
    metrics = {
        "exact_scene_actor_matches": 0,
        "fuzzy_scene_actor_matches": 0,
        "unmatched_scene_actor_names": 0,
        "unknown_actor_ids": 0,
        "unknown_actor_arc_thread_ids": 0,
    }
    cleaned_scenes: list[SceneCardDraft | SceneCard] = []
    for scene in plan.scene_cards:
        cleaned_actors: list[SceneActor] = []
        for actor in scene.actors:
            actor_id = actor.actor_id
            if actor_id and actor_id not in valid_actor_ids:
                actor_id = None
                metrics["unknown_actor_ids"] += 1
            if actor_id is None:
                match = matcher.match(actor.name)
                if match is None:
                    metrics["unmatched_scene_actor_names"] += 1
                else:
                    actor_id = match.actor_id
                    if match.match_type == "fuzzy":
                        metrics["fuzzy_scene_actor_matches"] += 1
                    else:
                        metrics["exact_scene_actor_matches"] += 1
            cleaned_actors.append(actor.model_copy(update={"actor_id": actor_id}))
        cleaned_scenes.append(scene.model_copy(update={"actors": cleaned_actors}))

    return plan.model_copy(
        update={
            "scene_cards": cleaned_scenes,
        }
    ), metrics


def _episode_actor_arc_thread_ids(
    actor_arc_directives: list[ActorArcDirective],
) -> dict[str, set[str]]:
    thread_ids_by_actor: dict[str, set[str]] = {}
    for actor in actor_arc_directives:
        thread_ids_by_actor[actor.actor_id] = {thread.thread_id for thread in actor.arc_threads}
    return thread_ids_by_actor


def compact_actor_metadata(actor_metadata: ActorMetadata) -> dict[str, Any]:
    return {
        "actors": [
            {
                "actor_id": actor.actor_id,
                "display_name": actor.display_name,
                "aliases": actor.aliases,
                "actor_type": actor.actor_type,
                "description": actor.description,
                "book_ids": actor.book_ids,
                "narrative_functions": actor.narrative_functions,
                "goals_or_motivational_pressures": actor.goals_or_motivational_pressures,
                "constraints": actor.constraints,
                "stakes": actor.stakes,
                "transformations": actor.transformations,
                "uncertainty_notes": actor.uncertainty_notes,
                "evidence_confidence": actor.evidence_confidence,
                "narrative_tier": actor.narrative_tier.value,
                "series_scope": actor.series_scope.value,
                "relevant_episode_numbers": actor.relevant_episode_numbers,
            }
            for actor in actor_metadata.actors
        ],
        "relationships": [
            relationship.model_dump(mode="json")
            for relationship in actor_metadata.relationships
        ],
        "quality_notes": actor_metadata.quality_notes,
    }


def compact_consolidation_actor_metadata(actor_metadata: ActorMetadata) -> dict[str, Any]:
    return {
        "actors": [
            {
                "actor_id": actor.actor_id,
                "display_name": actor.display_name,
            }
            for actor in actor_metadata.actors
        ],
        "relationships": [
            {
                "source_actor_id": relationship.source_actor_id,
                "target_actor_id": relationship.target_actor_id,
                "relationship_type": relationship.relationship_type,
                "description": relationship.description,
            }
            for relationship in actor_metadata.relationships
        ],
    }


def compact_actor_registry(actor_metadata: ActorMetadata) -> dict[str, Any]:
    return {
        "actors": [
            {
                "actor_id": actor.actor_id,
                "display_name": actor.display_name,
                "aliases": actor.aliases,
                "actor_type": actor.actor_type,
            }
            for actor in actor_metadata.actors
        ],
    }


def select_actor_metadata_subset(
    actor_metadata: ActorMetadata,
    actor_ids: set[str],
) -> ActorMetadata:
    selected_ids = {actor_id for actor_id in actor_ids if actor_id}
    actors = [actor for actor in actor_metadata.actors if actor.actor_id in selected_ids]
    actor_id_set = {actor.actor_id for actor in actors}
    relationships = [
        relationship
        for relationship in actor_metadata.relationships
        if relationship.source_actor_id in actor_id_set
        and relationship.target_actor_id in actor_id_set
    ]
    return actor_metadata.model_copy(
        update={
            "actors": actors,
            "relationships": relationships,
            "unresolved_mentions": [],
        }
    )


def collect_actor_ids_for_primitives(primitives: list[SynthesisPrimitiveBase]) -> set[str]:
    actor_ids: set[str] = set()
    for primitive in primitives:
        actor_ids.update(primitive.actor_ids)
    return actor_ids


def _filter_known_ids(
    actor_ids: list[str],
    valid_actor_ids: set[str],
    metrics: dict[str, Any],
) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for actor_id in actor_ids:
        if actor_id not in valid_actor_ids:
            metrics["unknown_actor_ids"] = int(metrics.get("unknown_actor_ids", 0)) + 1
            continue
        if actor_id in seen:
            continue
        seen.add(actor_id)
        filtered.append(actor_id)
    return filtered
