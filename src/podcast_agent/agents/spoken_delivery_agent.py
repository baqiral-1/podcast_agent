"""Spoken-delivery agent for full-episode narration rewrites."""

from __future__ import annotations

import socket
from typing import Any, TypeVar

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

ResponseModelT = TypeVar("ResponseModelT")


class SpokenDeliveryAgent(Agent):
    """Convert a full factual script into spoken-form narration."""

    plan_schema_name = "spoken_delivery_plan"
    narration_schema_name = "spoken_delivery_narration"

    def __init__(
        self,
        llm,
        *,
        timeout_seconds: float = 10800.0,
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
        plan = self._generate_with_retry(
            client=plan_client,
            schema_name=self.plan_schema_name,
            episode_id=script.episode_id,
            instructions=build_spoken_delivery_arc_plan_instructions(),
            payload={
                "episode_id": script.episode_id,
                "episode_title": script.title,
                "source_script": source_script,
            },
            response_model=SpokenDeliveryArcPlan,
        )

        narration_client = self._client_for_schema(self.narration_schema_name)
        narration_draft = self._generate_with_retry(
            client=narration_client,
            schema_name=self.narration_schema_name,
            episode_id=script.episode_id,
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

    def _generate_with_retry(
        self,
        *,
        client: Any,
        schema_name: str,
        episode_id: str,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        total_attempts = 3
        run_logger = getattr(client, "run_logger", None) or getattr(self.llm, "run_logger", None)
        attempt = 1
        for attempt in range(1, total_attempts + 1):
            try:
                return client.generate_json(
                    schema_name=schema_name,
                    instructions=instructions,
                    payload=payload,
                    response_model=response_model,
                )
            except Exception as exc:
                if not self._is_retryable_failure(exc):
                    raise
                is_timeout = self._is_timeout_failure(exc)
                if run_logger is not None:
                    event_name = "spoken_delivery_failed" if attempt == total_attempts else "spoken_delivery_retry"
                    run_logger.log(
                        event_name,
                        stage="spoken_delivery_episode",
                        schema_name=schema_name,
                        episode_id=episode_id,
                        attempt=attempt,
                        total_attempts=total_attempts,
                        is_timeout=is_timeout,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                if attempt == total_attempts:
                    raise
        raise RuntimeError(
            f"Spoken delivery exhausted retries for schema '{schema_name}' and episode '{episode_id}'."
        )

    @staticmethod
    def _is_retryable_failure(exc: Exception) -> bool:
        return isinstance(exc, (RuntimeError, TimeoutError, socket.timeout))

    @classmethod
    def _is_timeout_failure(cls, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        for error in cls._exception_chain(exc):
            message = str(error).lower()
            if "timed out" in message or "timeout" in message:
                return True
        return False

    @staticmethod
    def _exception_chain(exc: BaseException) -> list[BaseException]:
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return chain

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
