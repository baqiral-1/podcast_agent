"""Unit tests for Kokoro TTS client integration."""

from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from podcast_agent.config import Settings
from podcast_agent.tts import KokoroTTSClient, build_tts_client


class _FakeStdin:
    def __init__(self, *, fail_on_write: bool = False) -> None:
        self.fail_on_write = fail_on_write
        self.writes: list[str] = []

    def write(self, value: str) -> int:
        if self.fail_on_write:
            raise BrokenPipeError("broken pipe")
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(
        self,
        name: str,
        *,
        fail_on_write: bool = False,
        returncode: int | None = None,
    ) -> None:
        self.name = name
        self.stdin = _FakeStdin(fail_on_write=fail_on_write)
        self.stdout = object()
        self._returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9


def _payload_from_last_write(process: _FakeProcess) -> dict[str, object]:
    return json.loads(process.stdin.writes[-1])


def test_build_tts_client_uses_kokoro_and_default_voice() -> None:
    settings = Settings()
    settings = settings.model_copy(
        update={"tts": settings.tts.model_copy(update={"provider": "kokoro"})}
    )
    client = build_tts_client(settings)
    assert isinstance(client, KokoroTTSClient)
    assert client.config.voice == "af_heart"


def test_kokoro_tts_client_reuses_worker_and_defaults_voice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "kokoro"
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    client = KokoroTTSClient(
        Settings().tts.model_copy(update={"voice": "ballad"}),
        binary_path=binary_path,
        hf_home=tmp_path / "hf_cache",
    )

    process = _FakeProcess("worker-1")
    popen_calls: list[list[str]] = []

    def fake_popen(args, **kwargs):
        del kwargs
        popen_calls.append(args)
        return process

    def fake_response_line(_stdout, proc):
        payload = _payload_from_last_write(proc)
        return json.dumps(
            {
                "id": payload["id"],
                "ok": True,
                "audio_b64": base64.b64encode(str(payload["text"]).encode("utf-8")).decode("ascii"),
            }
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(client, "_read_worker_response_line", fake_response_line)

    first = client.synthesize("hello")
    second = client.synthesize("again")

    assert first == b"hello"
    assert second == b"again"
    assert len(popen_calls) == 1
    requests = [json.loads(entry) for entry in process.stdin.writes]
    assert requests == [
        {"id": 1, "text": "hello", "voice": "af_heart", "speed": 1.0},
        {"id": 2, "text": "again", "voice": "af_heart", "speed": 1.0},
    ]


def test_kokoro_tts_client_uses_two_workers_and_passes_thread_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "kokoro"
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    client = KokoroTTSClient(
        Settings().tts.model_copy(
            update={"voice": "af_heart", "kokoro_parallelism": 2, "kokoro_worker_threads": 3}
        ),
        binary_path=binary_path,
        hf_home=tmp_path / "hf_cache",
    )

    processes = [_FakeProcess("worker-1"), _FakeProcess("worker-2")]
    process_iter = iter(processes)
    popen_envs: list[dict[str, str]] = []

    def fake_popen(args, **kwargs):
        del args
        popen_envs.append(kwargs["env"])
        return next(process_iter)

    def fake_response_line(_stdout, proc):
        payload = _payload_from_last_write(proc)
        time.sleep(0.05)
        return json.dumps(
            {
                "id": payload["id"],
                "ok": True,
                "audio_b64": base64.b64encode(str(payload["text"]).encode("utf-8")).decode("ascii"),
            }
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(client, "_read_worker_response_line", fake_response_line)

    results: dict[str, bytes] = {}
    start_event = threading.Event()

    def synthesize(text: str) -> None:
        start_event.wait()
        results[text] = client.synthesize(text)

    threads = [
        threading.Thread(target=synthesize, args=("alpha",)),
        threading.Thread(target=synthesize, args=("beta",)),
    ]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join()

    assert results == {"alpha": b"alpha", "beta": b"beta"}
    assert len(popen_envs) == 2
    assert {process.name for process in processes if process.stdin.writes} == {"worker-1", "worker-2"}
    assert all(env["KOKORO_WORKER_THREADS"] == "3" for env in popen_envs)


def test_kokoro_tts_client_surfaces_worker_errors_and_replaces_failed_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "kokoro"
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    client = KokoroTTSClient(
        Settings().tts.model_copy(update={"voice": "af_heart"}),
        binary_path=binary_path,
        hf_home=tmp_path / "hf_cache",
    )

    first_process = _FakeProcess("worker-1")
    replacement_process = _FakeProcess("worker-2")
    processes = iter([first_process, replacement_process])

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: next(processes))

    def fake_response_line(_stdout, proc):
        payload = _payload_from_last_write(proc)
        if proc is first_process:
            return json.dumps({"id": payload["id"], "ok": False, "error": "kokoro worker failed: boom"})
        return json.dumps(
            {
                "id": payload["id"],
                "ok": True,
                "audio_b64": base64.b64encode(b"ok").decode("ascii"),
            }
        )

    monkeypatch.setattr(client, "_read_worker_response_line", fake_response_line)

    with pytest.raises(RuntimeError, match="boom"):
        client.synthesize("hello")

    audio = client.synthesize("retry")

    assert audio == b"ok"
    assert first_process.terminated is True
    assert replacement_process.terminated is False


def test_kokoro_tts_client_restarts_worker_after_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "kokoro"
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    client = KokoroTTSClient(
        Settings().tts.model_copy(update={"voice": "af_heart"}),
        binary_path=binary_path,
        hf_home=tmp_path / "hf_cache",
    )

    failing_process = _FakeProcess("worker-1", fail_on_write=True)
    healthy_process = _FakeProcess("worker-2")
    processes = iter([failing_process, healthy_process])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: next(processes))

    def fake_response_line(_stdout, proc):
        payload = _payload_from_last_write(proc)
        return json.dumps(
            {
                "id": payload["id"],
                "ok": True,
                "audio_b64": base64.b64encode(b"ok").decode("ascii"),
            }
        )

    monkeypatch.setattr(client, "_read_worker_response_line", fake_response_line)

    with pytest.raises(RuntimeError, match="transport failed"):
        client.synthesize("first")

    audio = client.synthesize("second")

    assert audio == b"ok"
    assert failing_process.terminated is True
    assert healthy_process.stdin.writes == [
        json.dumps({"id": 2, "text": "second", "voice": "af_heart", "speed": 1}) + "\n"
    ]


def test_kokoro_tts_client_requires_binary(tmp_path: Path) -> None:
    client = KokoroTTSClient(
        Settings().tts.model_copy(update={"voice": "af_heart"}),
        binary_path=tmp_path / "missing-kokoro",
        hf_home=tmp_path / "hf_cache",
    )

    with pytest.raises(RuntimeError, match="kokoro binary not found"):
        client.synthesize("hello")
