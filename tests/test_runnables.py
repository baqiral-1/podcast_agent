from __future__ import annotations

import json

from podcast_agent.langchain.runnables import (
    is_connection_error,
    is_json_parse_error,
    is_transient_error,
)


class APIConnectionError(Exception):
    pass


class RemoteProtocolError(Exception):
    pass


def test_connection_error_detects_builtin_connection_error() -> None:
    assert is_connection_error(ConnectionError("socket disconnected"))


def test_connection_error_detects_provider_error_name() -> None:
    assert is_connection_error(APIConnectionError("Connection error."))


def test_connection_error_detects_remote_protocol_error_type() -> None:
    assert is_connection_error(RemoteProtocolError("peer closed connection unexpectedly"))


def test_connection_error_detects_incomplete_chunked_read_message() -> None:
    assert is_connection_error(
        RuntimeError("peer closed connection without sending complete message body (incomplete chunked read)")
    )


def test_connection_error_detects_server_disconnected_message() -> None:
    assert is_connection_error(RuntimeError("Server disconnected without sending a response."))


def test_transient_error_false_for_non_transient_value_error() -> None:
    assert not is_transient_error(ValueError("schema validation failed"))


def test_json_parse_error_detects_jsondecodeerror() -> None:
    exc = json.JSONDecodeError("Expecting ',' delimiter", '{"a":1 "b":2}', 7)
    assert is_json_parse_error(exc)


def test_json_parse_error_detects_json_runtime_errors() -> None:
    assert is_json_parse_error(RuntimeError("LLM response was not valid JSON."))
    assert is_json_parse_error(RuntimeError("LLM response JSON must be an object."))


def test_json_parse_error_false_for_other_runtime_error() -> None:
    assert not is_json_parse_error(RuntimeError("unrelated runtime failure"))
