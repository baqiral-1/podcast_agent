"""Helpers for LangChain runnable execution, retries, and context propagation."""

from __future__ import annotations

from contextvars import copy_context
import json
import re
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


_TRANSIENT_API_STATUS_MESSAGE_SNIPPETS = (
    "overloaded_error",
    "overloaded",
    "internal server error",
    "service unavailable",
    "gateway timeout",
    "temporarily unavailable",
)


def _contains_transient_api_status_snippet(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        haystack = value.lower()
    else:
        try:
            haystack = json.dumps(value, default=str, sort_keys=True).lower()
        except Exception:
            haystack = str(value).lower()
    return any(snippet in haystack for snippet in _TRANSIENT_API_STATUS_MESSAGE_SNIPPETS)


def _iter_exception_chain(exc: Exception) -> Sequence[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    ordered: list[BaseException] = []

    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(current)

        nested = getattr(current, "exceptions", None)
        if isinstance(nested, tuple):
            stack.extend(
                child for child in reversed(nested) if isinstance(child, BaseException)
            )
        for attr_name in ("__cause__", "__context__"):
            linked = getattr(current, attr_name, None)
            if isinstance(linked, BaseException):
                stack.append(linked)

    return ordered


def is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return True
    return False


def is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    error_name = type(exc).__name__.lower()
    if "connection" in error_name:
        return True
    if "remoteprotocolerror" in error_name:
        return True
    message = str(exc).lower()
    if "connection error" in message:
        return True
    if "connection reset" in message:
        return True
    if "connection aborted" in message:
        return True
    if "connection refused" in message:
        return True
    if "broken pipe" in message:
        return True
    if "peer closed connection" in message:
        return True
    if "incomplete chunked read" in message:
        return True
    if "server disconnected without sending a response" in message:
        return True
    return False


def _extract_http_status_code(exc: Exception) -> int | None:
    for attr_name in ("status_code", "status", "http_status"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

    message = str(exc)
    for pattern in (
        r"\bstatus(?:_code)?\s*[:=]\s*(\d{3})\b",
        r"\bhttp\s*(\d{3})\b",
        r"\bcode\s*[:=]\s*(\d{3})\b",
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def is_api_status_transient_error(exc: Exception) -> bool:
    error_name = type(exc).__name__.lower()
    status_code = _extract_http_status_code(exc)
    if isinstance(status_code, int):
        if status_code == 429 or status_code >= 500:
            return True
        return False

    if "ratelimit" in error_name:
        return True

    if _contains_transient_api_status_snippet(getattr(exc, "type", None)):
        return True
    if _contains_transient_api_status_snippet(getattr(exc, "body", None)):
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        if _contains_transient_api_status_snippet(getattr(response, "text", None)):
            return True
        if _contains_transient_api_status_snippet(getattr(response, "content", None)):
            return True

    message = str(exc)
    if _contains_transient_api_status_snippet(message):
        return True
    if "internalservererror" in error_name:
        return True
    if "apistatuserror" not in error_name:
        return False
    return False


def is_transient_error(exc: Exception) -> bool:
    return any(
        is_timeout_error(candidate)
        or is_connection_error(candidate)
        or is_api_status_transient_error(candidate)
        for candidate in _iter_exception_chain(exc)
        if isinstance(candidate, Exception)
    )


def is_json_parse_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if message.startswith("LLM response was not valid JSON."):
            return True
        if message.startswith("LLM response JSON must be an object."):
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
