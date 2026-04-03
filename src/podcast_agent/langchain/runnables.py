"""Helpers for LangChain runnable execution, retries, and context propagation."""

from __future__ import annotations

from contextvars import Context, copy_context
import socket
from typing import Any, Callable, Sequence, TypeVar

from langchain_core.runnables import Runnable, RunnableLambda

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class TransientLLMError(RuntimeError):
    """Represents a retryable transient failure (timeouts, network blips)."""


class RetryableGenerationError(RuntimeError):
    """Represents a retryable schema/validation/generation failure."""

    def __init__(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data or {}


class ComplianceViolationError(RuntimeError):
    """Represents a retryable semantic violation in model output."""

    def __init__(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data or {}


def is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return True
    return False


_CONTEXT_KEY = "__lc_context"
_INPUT_KEY = "__lc_input"


def context_runnable(fn: Callable[[dict[str, Any]], OutputT]) -> RunnableLambda:
    """Wrap a callable so its execution preserves caller contextvars."""

    def run_with_context(payload: dict[str, Any]) -> OutputT:
        context = payload[_CONTEXT_KEY]
        input_payload = {key: value for key, value in payload.items() if key != _CONTEXT_KEY}
        return context.run(fn, input_payload)

    return RunnableLambda(run_with_context)


def attach_context(items: Sequence[InputT]) -> list[dict[str, Any]]:
    wrapped: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            payload: dict[str, Any] = dict(item)
        else:
            payload = {_INPUT_KEY: item}
        payload[_CONTEXT_KEY] = copy_context()
        wrapped.append(payload)
    return wrapped


def apply_retry(
    runnable: Runnable[InputT, OutputT],
    *,
    max_attempts: int,
    retry_on: tuple[type[BaseException], ...],
    backoff_min: float = 1.0,
    backoff_max: float = 8.0,
    backoff_jitter: float = 1.0,
) -> Runnable[InputT, OutputT]:
    return runnable.with_retry(
        retry_if_exception_type=retry_on,
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=True,
        exponential_jitter_params={
            "initial": backoff_min,
            "max": backoff_max,
            "jitter": backoff_jitter,
        },
    )


def apply_fallbacks(
    runnable: Runnable[InputT, OutputT],
    fallbacks: Sequence[Runnable[InputT, OutputT]],
    *,
    exceptions: tuple[type[BaseException], ...],
    exception_key: str | None = None,
) -> Runnable[InputT, OutputT]:
    return runnable.with_fallbacks(
        fallbacks,
        exceptions_to_handle=exceptions,
        exception_key=exception_key,
    )


def batch_or_invoke(
    runnable: Runnable[dict[str, Any], OutputT],
    items: Sequence[InputT],
    *,
    max_concurrency: int,
) -> list[OutputT | Exception]:
    if not items:
        return []
    context_items = attach_context(items)
    if max_concurrency <= 1 or len(context_items) <= 1:
        results: list[OutputT | Exception] = []
        for item in context_items:
            try:
                results.append(runnable.invoke(item))
            except Exception as exc:
                results.append(exc)
        return results
    return runnable.batch(
        context_items,
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )
