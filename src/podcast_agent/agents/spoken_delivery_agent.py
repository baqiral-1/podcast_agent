"""Spoken-delivery agent for full-episode narration rewrites."""

from __future__ import annotations

from podcast_agent.agents.base import Agent
from podcast_agent.eval.rewrite_metrics import build_rewrite_metrics, check_fidelity
from podcast_agent.llm.anthropic import AnthropicLLMClient
from podcast_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from podcast_agent.prompts.spoken_delivery import (
    build_spoken_delivery_arc_plan_instructions,
    build_spoken_delivery_narration_instructions,
)
from podcast_agent.schemas.models import (
    EpisodeScript,
    SpokenDeliveryArcPlan,
    SpokenDeliveryEpisodeResult,
    SpokenDeliveryNarrationDraft,
    SpokenEpisodeNarration,
    SpokenNarrationChunk,
)


class SpokenDeliveryAgent(Agent):
    """Convert a full factual script into spoken-form narration."""

    plan_schema_name = "spoken_delivery_plan"
    narration_schema_name = "spoken_delivery_narration"

    def __init__(
        self,
        llm,
        *,
        timeout_seconds: float = 1200.0,
        chunk_min_words: int = 700,
        chunk_max_words: int = 900,
    ) -> None:
        super().__init__(llm)
        self.timeout_seconds = timeout_seconds
        self.chunk_min_words = chunk_min_words
        self.chunk_max_words = chunk_max_words

    def rewrite_full_episode_two_call(
        self,
        script: EpisodeScript,
    ) -> tuple[SpokenEpisodeNarration, SpokenDeliveryEpisodeResult, SpokenDeliveryArcPlan]:
        """Rewrite an entire episode via a two-call arc-plan + narration flow."""

        source_script = self._build_source_script(script)
        source_narration = self._build_source_narration(script)

        plan_client = self._client_for_schema(self.plan_schema_name)
        plan = plan_client.generate_json(
            schema_name=self.plan_schema_name,
            instructions=build_spoken_delivery_arc_plan_instructions(),
            payload={
                "episode_id": script.episode_id,
                "episode_title": script.title,
                "source_script": source_script,
            },
            response_model=SpokenDeliveryArcPlan,
        )

        narration_client = self._client_for_schema(self.narration_schema_name)
        narration_draft = narration_client.generate_json(
            schema_name=self.narration_schema_name,
            instructions=build_spoken_delivery_narration_instructions(),
            payload={
                "episode_id": script.episode_id,
                "episode_title": script.title,
                "arc_plan": plan.model_dump(mode="json"),
                "source_script": source_script,
            },
            response_model=SpokenDeliveryNarrationDraft,
        )

        narration = narration_draft.narration.strip()
        metrics = build_rewrite_metrics(source_narration, narration)
        fidelity = check_fidelity(source_narration, narration, check_paragraph_drift=False)
        chunks = self._chunk_narration(script.episode_id, narration)

        spoken_script = SpokenEpisodeNarration(
            episode_id=script.episode_id,
            title=script.title,
            narrator=script.narrator,
            narration=narration,
            chunks=chunks,
        )
        spoken_delivery = SpokenDeliveryEpisodeResult(
            episode_id=script.episode_id,
            mode="full",
            metrics=metrics,
            fidelity_passed=fidelity.passed,
            missing_names=fidelity.missing_names,
            missing_numbers=fidelity.missing_numbers,
            chunk_count=len(chunks),
        )
        return spoken_script, spoken_delivery, plan

    def build_payload(self, script: EpisodeScript) -> dict:
        """Build the default payload for full-episode planning."""

        return {
            "episode_id": script.episode_id,
            "episode_title": script.title,
            "source_script": self._build_source_script(script),
        }

    def _client_for_schema(self, schema_name: str):
        llm = self.llm
        if hasattr(llm, "client_for_schema"):
            llm = llm.client_for_schema(schema_name)
        if isinstance(llm, (OpenAICompatibleLLMClient, AnthropicLLMClient)):
            run_logger = getattr(llm, "run_logger", None)
            llm = type(llm)(
                llm.config.model_copy(update={"timeout_seconds": self.timeout_seconds}),
                transport=llm.transport,
            )
            if run_logger is not None:
                llm.set_run_logger(run_logger)
        return llm

    def _build_source_script(self, script: EpisodeScript) -> str:
        segments = []
        for segment in script.segments:
            heading = segment.heading.strip()
            narration = segment.narration.strip()
            if heading:
                segments.append(f"{heading}\n{narration}")
            else:
                segments.append(narration)
        return "\n\n".join(segments).strip()

    def _build_source_narration(self, script: EpisodeScript) -> str:
        return "\n\n".join(segment.narration.strip() for segment in script.segments if segment.narration).strip()

    def _chunk_narration(self, episode_id: str, narration: str) -> list[SpokenNarrationChunk]:
        paragraphs = [para.strip() for para in narration.split("\n\n") if para.strip()]
        chunks: list[SpokenNarrationChunk] = []
        current: list[str] = []
        current_words = 0
        chunk_index = 1
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            if current and current_words + paragraph_words > self.chunk_max_words and current_words >= self.chunk_min_words:
                chunk_text = "\n\n".join(current).strip()
                chunks.append(
                    SpokenNarrationChunk(
                        chunk_id=f"{episode_id}-spoken-{chunk_index}",
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                    )
                )
                chunk_index += 1
                current = [paragraph]
                current_words = paragraph_words
            else:
                current.append(paragraph)
                current_words += paragraph_words
        if current:
            chunk_text = "\n\n".join(current).strip()
            chunks.append(
                SpokenNarrationChunk(
                    chunk_id=f"{episode_id}-spoken-{chunk_index}",
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                )
            )
        return chunks

    def chunk_narration(self, episode_id: str, narration: str) -> list[SpokenNarrationChunk]:
        return self._chunk_narration(episode_id, narration)
