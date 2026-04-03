"""Shared JSON parsing helpers for LLM outputs."""

from __future__ import annotations

import json
from typing import Any


def normalize_json_content(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("LLM response was not valid JSON.") from None
        cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response JSON must be an object.")
    return json.dumps(parsed)


def unwrap_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "payload" in payload and isinstance(payload["payload"], dict):
        wrapped_payload = payload["payload"]
        if "draft" in wrapped_payload and isinstance(wrapped_payload["draft"], dict):
            return wrapped_payload["draft"]
        return wrapped_payload
    if "result" in payload and isinstance(payload["result"], dict):
        return payload["result"]
    return payload

