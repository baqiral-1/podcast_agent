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
        chunks = structure["chunks"]
        chapter_theme_map: dict[str, list[str]] = {}
        all_terms: list[str] = []
        for chunk in chunks:
            chapter_theme_map.setdefault(chunk["chapter_id"], []).extend(chunk.get("themes", []))
            all_terms.extend(chunk.get("themes", []))
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
        clusters = []
        cluster_size = 2 if len(chapters) > 2 else 1
        for sequence, start in enumerate(range(0, len(chapters), cluster_size), start=1):
            chosen_chapters = chapters[start : start + cluster_size]
            if not chosen_chapters:
                continue
            chapter_ids = [chapter["chapter_id"] for chapter in chosen_chapters]
            chunk_ids = list(
                itertools.chain.from_iterable(chapter["chunk_ids"] for chapter in chosen_chapters)
            )
            cluster_themes = []
            for chapter_id in chapter_ids:
                cluster_themes.extend(chapter_theme_map.get(chapter_id, []))
            clusters.append(
                {
                    "cluster_id": f"cluster-{sequence}",
                    "label": f"Episode Cluster {sequence}",
                    "rationale": "Grouped around adjacent chapters and shared themes.",
                    "chapter_ids": chapter_ids,
                    "chunk_ids": chunk_ids,
                    "themes": [item for item, _ in Counter(cluster_themes).most_common(4)],
                }
            )
        notable_claims = []
        for chunk in chunks[:8]:
            source_text = chunk.get("text") or chunk.get("excerpt", "")
            sentence = source_text.split(".")[0].strip()
            if sentence:
                notable_claims.append(sentence)
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
        minimum_source_words_per_episode = payload.get("minimum_source_words_per_episode", 50000)
        chapter_map = {chapter["chapter_id"]: chapter for chapter in structure["chapters"]}
        chunk_map = {chunk["chunk_id"]: chunk for chunk in structure["chunks"]}
        pending_clusters = list(analysis["episode_clusters"])
        merged_clusters = []
        current_cluster: dict[str, Any] | None = None

        for cluster in pending_clusters:
            cluster_word_count = sum(
                chunk_map[chunk_id].get("word_count", len(chunk_map[chunk_id].get("text", "").split()))
                for chunk_id in cluster["chunk_ids"]
                if chunk_id in chunk_map
            )
            if current_cluster is None:
                current_cluster = {
                    "cluster_id": f"cluster-{len(merged_clusters) + 1}",
                    "chapter_ids": list(cluster["chapter_ids"]),
                    "chunk_ids": list(cluster["chunk_ids"]),
                    "themes": list(cluster["themes"]),
                    "rationale": [cluster["rationale"]],
                    "word_count": cluster_word_count,
                }
                continue
            if current_cluster["word_count"] < minimum_source_words_per_episode:
                current_cluster["chapter_ids"].extend(cluster["chapter_ids"])
                current_cluster["chunk_ids"].extend(cluster["chunk_ids"])
                current_cluster["themes"].extend(cluster["themes"])
                current_cluster["rationale"].append(cluster["rationale"])
                current_cluster["word_count"] += cluster_word_count
            else:
                merged_clusters.append(current_cluster)
                current_cluster = {
                    "cluster_id": f"cluster-{len(merged_clusters) + 1}",
                    "chapter_ids": list(cluster["chapter_ids"]),
                    "chunk_ids": list(cluster["chunk_ids"]),
                    "themes": list(cluster["themes"]),
                    "rationale": [cluster["rationale"]],
                    "word_count": cluster_word_count,
                }
        if current_cluster is not None:
            if merged_clusters and current_cluster["word_count"] < minimum_source_words_per_episode:
                merged_clusters[-1]["chapter_ids"].extend(current_cluster["chapter_ids"])
                merged_clusters[-1]["chunk_ids"].extend(current_cluster["chunk_ids"])
                merged_clusters[-1]["themes"].extend(current_cluster["themes"])
                merged_clusters[-1]["rationale"].extend(current_cluster["rationale"])
                merged_clusters[-1]["word_count"] += current_cluster["word_count"]
            else:
                merged_clusters.append(current_cluster)

        episodes = []
        for sequence, cluster in enumerate(merged_clusters, start=1):
            beats = []
            grouped_chunk_ids = cluster["chunk_ids"]
            for beat_sequence, chunk_id_group in enumerate(_group(grouped_chunk_ids, 2), start=1):
                beats.append(
                    {
                        "beat_id": f"{cluster['cluster_id']}-beat-{beat_sequence}",
                        "title": f"Beat {beat_sequence}",
                        "objective": "Advance the episode through grounded synthesis.",
                        "chunk_ids": chunk_id_group,
                        "claim_requirements": [
                            f"Explain the significance of {chunk_id_group[0]}."
                        ],
                    }
                )
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
                    "chunk_ids": cluster["chunk_ids"],
                    "themes": [item for item, _ in Counter(cluster["themes"]).most_common(4)]
                    or analysis["themes"][:3],
                    "beats": beats,
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
        return {
            "beat_id": beat["beat_id"],
            "segments": [
                {
                    "segment_id": f"{beat['beat_id']}-segment-1",
                    "beat_id": beat["beat_id"],
                    "heading": beat["title"],
                    "narration": " ".join(narration_parts) or beat["objective"],
                    "claims": claims,
                    "citations": citations,
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
        script = payload["script"]
        report = payload["report"]
        weak_claims = {
            assessment["claim_id"]
            for assessment in report["claim_assessments"]
            if assessment["status"] != "grounded"
        }
        repaired_segment_ids = []
        for segment in script["segments"]:
            segment_claim_ids = {claim["claim_id"] for claim in segment["claims"]}
            if segment_claim_ids & weak_claims:
                repaired_segment_ids.append(segment["segment_id"])
                segment["narration"] = " ".join(claim["text"] for claim in segment["claims"])
        return {
            "episode_id": script["episode_id"],
            "attempt": payload["attempt"],
            "repaired_segment_ids": repaired_segment_ids,
            "script": script,
            "report": report,
        }


def _group(values: list[str], group_size: int) -> list[list[str]]:
    return [values[index : index + group_size] for index in range(0, len(values), group_size)]
