"""Heuristic LLM implementation used for local development and tests."""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

from pydantic import BaseModel

from podcast_agent.llm.base import LLMClient, PromptPayload
from podcast_agent.schemas.models import GroundingStatus


class HeuristicLLMClient(LLMClient):
    """Deterministic stand-in that behaves like a JSON-producing model."""

    def __init__(self) -> None:
        super().__init__()

    def generate_json(
        self,
        schema_name: str,
        instructions: str,
        payload: PromptPayload,
        response_model: type[BaseModel],
    ) -> BaseModel:
        if self.run_logger is not None:
            self.run_logger.log(
                "llm_request",
                client="heuristic",
                schema_name=schema_name,
                instructions=instructions,
                payload=payload,
            )
        generator = getattr(self, f"_generate_{schema_name}", None)
        if generator is None:
            raise ValueError(f"No heuristic generator available for schema '{schema_name}'.")
        response = generator(payload)
        if self.run_logger is not None:
            self.run_logger.log(
                "llm_response",
                client="heuristic",
                schema_name=schema_name,
                response=response,
            )
        return response_model.model_validate(response)

    def _generate_book_structure(self, payload: PromptPayload) -> dict[str, Any]:
        return payload["draft"]

    def _generate_structured_chapter(self, payload: PromptPayload) -> dict[str, Any]:
        draft = payload["draft"]
        return {
            "chapter_number": draft["chapter_number"],
            "title": draft["title"],
            "summary": draft["summary"],
            "chunks": [
                {
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                    "themes": chunk.get("themes", []),
                }
                for chunk in draft["chunks"]
            ],
        }

    def _generate_book_analysis(self, payload: PromptPayload) -> dict[str, Any]:
        structure = payload["structure"]
        chapters = structure["chapters"]
        episode_count = payload["episode_count"]
        all_terms: list[str] = []
        for chapter in chapters:
            all_terms.extend(chapter.get("themes", []))
            for section in chapter.get("sections", []):
                all_terms.extend(section.get("themes", []))
        top_themes = [item for item, _ in Counter(all_terms).most_common(6)]
        continuity_arcs = []
        for index in range(max(0, len(chapters) - 1)):
            chapter_ids = [chapters[index]["chapter_id"], chapters[index + 1]["chapter_id"]]
            continuity_arcs.append(
                {
                    "arc_id": f"arc-{index + 1}",
                    "label": f"Arc {index + 1}",
                    "description": f"Connects {chapters[index]['title']} to {chapters[index + 1]['title']}.",
                    "chapter_ids": chapter_ids,
                }
            )
        if episode_count > len(chapters):
            raise ValueError(
                f"Requested {episode_count} episodes, but only {len(chapters)} chapters are available."
            )
        chapter_groups = _partition_evenly(chapters, episode_count)
        clusters = []
        for sequence, chosen_chapters in enumerate(chapter_groups, start=1):
            chapter_ids = [chapter["chapter_id"] for chapter in chosen_chapters]
            cluster_themes = []
            for chapter in chosen_chapters:
                cluster_themes.extend(chapter.get("themes", []))
            clusters.append(
                {
                    "cluster_id": f"cluster-{sequence}",
                    "label": f"Episode Cluster {sequence}",
                    "rationale": "Grouped around adjacent chapters and shared themes.",
                    "chapter_ids": chapter_ids,
                    "themes": [item for item, _ in Counter(cluster_themes).most_common(4)],
                }
            )
        notable_claims = []
        for chapter in chapters:
            for section in chapter.get("sections", [])[:2]:
                sentence = section.get("excerpt", "").split(".")[0].strip()
                if sentence:
                    notable_claims.append(sentence)
                if len(notable_claims) >= 8:
                    break
            if len(notable_claims) >= 8:
                break
        return {
            "book_id": structure["book_id"],
            "themes": top_themes or ["summary", "narrative"],
            "continuity_arcs": continuity_arcs,
            "notable_claims": notable_claims,
            "episode_clusters": clusters,
        }

    def _generate_series_plan(self, payload: PromptPayload) -> dict[str, Any]:
        analysis = payload["analysis"]
        structure = payload["structure"]
        chapter_map = {chapter["chapter_id"]: chapter for chapter in structure["chapters"]}
        merged_clusters = list(analysis["episode_clusters"])
        episodes = []
        for sequence, cluster in enumerate(merged_clusters, start=1):
            chapter_titles = [chapter_map[chapter_id]["title"] for chapter_id in cluster["chapter_ids"]]
            if len(chapter_titles) > 2:
                episode_title = f"{chapter_titles[0]} / {chapter_titles[-1]}"
            else:
                episode_title = " / ".join(chapter_titles)
            episodes.append(
                {
                    "episode_id": f"episode-{sequence}",
                    "sequence": sequence,
                    "title": episode_title,
                    "synopsis": " ".join(cluster["rationale"]),
                    "chapter_ids": cluster["chapter_ids"],
                    "themes": [item for item, _ in Counter(cluster["themes"]).most_common(4)]
                    or analysis["themes"][:3],
                }
            )
        return {
            "book_id": structure["book_id"],
            "format": "single_narrator",
            "strategy_summary": "Episodes follow thematic continuity and usually span multiple chapters.",
            "episodes": episodes,
        }

    def _generate_episode_script(self, payload: PromptPayload) -> dict[str, Any]:
        episode = payload["episode_plan"]
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in payload["retrieval_hits"]}
        segments = []
        for sequence, beat in enumerate(episode["beats"], start=1):
            beat_chunks = [chunks_by_id[chunk_id] for chunk_id in beat["chunk_ids"] if chunk_id in chunks_by_id]
            citations = [chunk["chunk_id"] for chunk in beat_chunks]
            claims = []
            narration_parts = []
            for claim_sequence, chunk in enumerate(beat_chunks, start=1):
                claim_text = chunk["text"].split(".")[0].strip()
                if claim_text:
                    claims.append(
                        {
                            "claim_id": f"{beat['beat_id']}-claim-{claim_sequence}",
                            "text": claim_text,
                            "evidence_chunk_ids": [chunk["chunk_id"]],
                        }
                    )
                narration_parts.append(chunk["text"].strip())
            narration = " ".join(narration_parts) or beat["objective"]
            segments.append(
                {
                    "segment_id": f"{episode['episode_id']}-segment-{sequence}",
                    "beat_id": beat["beat_id"],
                    "heading": beat["title"],
                    "narration": narration,
                    "claims": claims,
                    "citations": citations,
                }
            )
        return {
            "episode_id": episode["episode_id"],
            "title": episode["title"],
            "narrator": "Narrator",
            "segments": segments,
        }

    def _generate_beat_script(self, payload: PromptPayload) -> dict[str, Any]:
        beat = payload["beat"]
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in payload["retrieval_hits"]}
        beat_chunks = [chunks_by_id[chunk_id] for chunk_id in beat["chunk_ids"] if chunk_id in chunks_by_id]
        claims = []
        narration_parts = []
        for claim_sequence, chunk in enumerate(beat_chunks, start=1):
            claim_text = chunk["text"].split(".")[0].strip()
            if claim_text:
                claims.append(
                    {
                        "text": claim_text,
                        "evidence_chunk_ids": [chunk["chunk_id"]],
                    }
                )
            narration_parts.append(chunk["text"].strip())
        return {
            "segments": [
                {
                    "heading": beat["title"],
                    "narration": " ".join(narration_parts) or beat["objective"],
                    "claims": claims,
                }
            ],
        }

    def _generate_grounding_report(self, payload: PromptPayload) -> dict[str, Any]:
        script = payload["script"]
        chunk_lookup = {chunk["chunk_id"]: chunk["text"].lower() for chunk in payload["retrieval_hits"]}
        assessments = []
        overall_status = "pass"
        for segment in script["segments"]:
            for claim in segment["claims"]:
                evidence_text = " ".join(chunk_lookup.get(chunk_id, "") for chunk_id in claim["evidence_chunk_ids"])
                claim_tokens = {token.strip(" ,.;:!?").lower() for token in claim["text"].split() if len(token) > 4}
                overlap = sum(1 for token in claim_tokens if token and token in evidence_text)
                if claim["evidence_chunk_ids"] and overlap >= max(1, len(claim_tokens) // 3):
                    status = GroundingStatus.GROUNDED
                    reason = "Evidence chunks contain the core claim terms."
                elif claim["evidence_chunk_ids"]:
                    status = GroundingStatus.WEAK
                    reason = "Claim cites evidence, but lexical overlap is limited."
                    overall_status = "fail"
                else:
                    status = GroundingStatus.UNSUPPORTED
                    reason = "No evidence chunks were attached to the claim."
                    overall_status = "fail"
                assessments.append(
                    {
                        "claim_id": claim["claim_id"],
                        "status": status,
                        "reason": reason,
                        "evidence_chunk_ids": claim["evidence_chunk_ids"],
                    }
                )
        return {
            "episode_id": script["episode_id"],
            "overall_status": overall_status,
            "claim_assessments": assessments,
        }

    def _generate_episode_repair(self, payload: PromptPayload) -> dict[str, Any]:
        failed_segments = payload["failed_segments"]
        failed_segment_ids = payload.get("failed_segment_ids")
        if not failed_segment_ids:
            # Backwards compatibility with older payload shape.
            failed_segment_ids = [segment.get("segment_id") for segment in failed_segments if segment.get("segment_id")]

        repaired_segment_ids: list[str] = []
        repaired_segments: list[dict[str, Any]] = []
        for segment_id, segment in zip(failed_segment_ids, failed_segments, strict=False):
            if segment_id:
                repaired_segment_ids.append(segment_id)
            segment = dict(segment)
            segment["narration"] = " ".join(claim["text"] for claim in segment.get("claims", [])) or segment.get("narration", "")
            segment.pop("segment_id", None)
            segment.pop("beat_id", None)
            segment.pop("citations", None)
            segment["claims"] = [
                {
                    "text": claim["text"],
                    "evidence_chunk_ids": claim["evidence_chunk_ids"],
                }
                for claim in segment.get("claims", [])
            ]
            repaired_segments.append(segment)
        return {
            "episode_id": payload["episode_id"],
            "attempt": payload["attempt"],
            "repaired_segment_ids": repaired_segment_ids,
            "repaired_segments": repaired_segments,
        }

    def _generate_spoken_delivery_plan(self, payload: PromptPayload) -> dict[str, Any]:
        source_script = payload.get("source_script", "")
        opening_phrase = source_script.split("\n", 1)[0][:120] if source_script else "Opening scene"
        return {
            "theme": "A high-level dramatic arc from the provided material.",
            "threads": [
                {
                    "name": "Primary thread",
                    "introduced": "Act 1",
                    "developed": "Acts 2-4",
                    "payoff": "Final act",
                    "listener_feeling": "Tension turning to release",
                }
            ],
            "opening": {
                "scene": opening_phrase or "Opening scene",
                "why": "It immediately signals the central stakes.",
                "transition_strategy": "To understand how this moment arrived, we need to go back.",
            },
            "acts": [
                {
                    "number": 1,
                    "title": "Act 1",
                    "source_material": "Opening setup",
                    "why_here": "Establish stakes early.",
                    "driving_tension": "How did events reach this point?",
                    "transition_to_next": "Which leads into the deeper context.",
                }
            ],
            "plants_and_payoffs": [],
            "key_moments": [],
        }

    def _generate_spoken_delivery_narration(self, payload: PromptPayload) -> dict[str, Any]:
        source_script = payload.get("source_script", "")
        narration = " ".join(source_script.split())
        return {"narration": narration}

    def _generate_episode_framing(self, payload: PromptPayload) -> dict[str, Any]:
        def build_words(prefix: str, count: int) -> str:
            if count <= 0:
                return ""
            words = [prefix] + [f"{prefix}{index}" for index in range(1, count)]
            return " ".join(words)

        has_previous = bool(payload.get("has_previous"))
        has_next = bool(payload.get("has_next"))
        recap_words = int(payload.get("recap_words", 80))
        current_words = int(payload.get("current_words", 120))
        next_min_words = int(payload.get("next_min_words", 40))
        recap = build_words("Recap", recap_words) if has_previous else ""
        current_summary = build_words("Current", current_words)
        next_overview = build_words("Next", next_min_words) if has_next else ""
        return {
            "recap": recap,
            "current_summary": current_summary,
            "next_overview": next_overview,
        }


def _group(values: list[str], group_size: int) -> list[list[str]]:
    return [values[index : index + group_size] for index in range(0, len(values), group_size)]


def _partition_evenly(values: list[dict[str, Any]], group_count: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    start = 0
    for remaining_groups in range(group_count, 0, -1):
        remaining_items = len(values) - start
        size = max(1, (remaining_items + remaining_groups - 1) // remaining_groups)
        groups.append(values[start : start + size])
        start += size
    return groups
