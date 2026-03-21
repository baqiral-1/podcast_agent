"""Kokoro text-to-speech client backed by a persistent worker pool."""

from __future__ import annotations

import atexit
import base64
import json
import os
import queue
import select
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from podcast_agent.config import TTSConfig
from podcast_agent.tts.base import TTSClient


@dataclass(slots=True)
class _KokoroWorkerHandle:
    process: subprocess.Popen[str] | None = None
    request_id: int = 0


class KokoroTTSClient(TTSClient):
    """Client for the local kokoro-tts environment."""

    def __init__(
        self,
        config: TTSConfig,
        *,
        binary_path: Path | None = None,
        hf_home: Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.binary_path = binary_path or Path("/tmp/tts-kokoro/bin/kokoro")
        self.hf_home = hf_home or Path("/tmp/hf_cache_kokoro")
        self.default_voice = "af_heart"
        self._available_workers: queue.Queue[_KokoroWorkerHandle] = queue.Queue()
        self._pool_lock = threading.Lock()
        self._workers: list[_KokoroWorkerHandle] = []
        atexit.register(self.close)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        del audio_format, instructions
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if not self.binary_path.exists():
            raise RuntimeError(
                f"kokoro binary not found at '{self.binary_path}'. "
                "Install kokoro into /tmp/tts-kokoro and retry."
            )

        voice_value = voice or self.config.voice
        if not voice_value or voice_value == "ballad":
            voice_value = self.default_voice

        if self.run_logger is not None:
            self.run_logger.log(
                "tts_request",
                client="kokoro",
                voice=voice_value,
                audio_format="wav",
                speed=self.config.speed,
                text=text,
            )

        worker = self._checkout_worker()
        try:
            process = self._ensure_worker_process(worker)
            worker.request_id += 1
            request_id = worker.request_id
            payload = {
                "id": request_id,
                "text": text,
                "voice": voice_value,
                "speed": self.config.speed,
            }
            try:
                self._write_worker_request(process, payload)
                response = self._read_worker_response(process, request_id)
            except Exception:
                self._reset_worker(worker)
                raise
            if not response.get("ok"):
                self._reset_worker(worker)
                error_message = str(response.get("error") or "kokoro worker synthesis failed")
                raise RuntimeError(error_message)
            audio_payload = response.get("audio_b64")
            if not isinstance(audio_payload, str):
                self._reset_worker(worker)
                raise RuntimeError("kokoro worker returned an invalid audio payload.")
            try:
                audio_bytes = base64.b64decode(audio_payload)
            except ValueError as exc:
                self._reset_worker(worker)
                raise RuntimeError("kokoro worker returned malformed audio data.") from exc
        finally:
            self._release_worker(worker)

        if self.run_logger is not None:
            self.run_logger.log(
                "tts_response",
                client="kokoro",
                voice=voice_value,
                audio_format="wav",
                speed=self.config.speed,
                byte_count=len(audio_bytes),
            )
        return audio_bytes

    def close(self) -> None:
        """Terminate all worker processes."""

        with self._pool_lock:
            workers = list(self._workers)
            self._workers.clear()
            while not self._available_workers.empty():
                try:
                    self._available_workers.get_nowait()
                except queue.Empty:
                    break
        for worker in workers:
            self._reset_worker(worker)

    def _checkout_worker(self) -> _KokoroWorkerHandle:
        try:
            return self._available_workers.get_nowait()
        except queue.Empty:
            with self._pool_lock:
                if len(self._workers) < self.config.kokoro_parallelism:
                    worker = _KokoroWorkerHandle()
                    self._workers.append(worker)
                    return worker
            return self._available_workers.get()

    def _release_worker(self, worker: _KokoroWorkerHandle) -> None:
        with self._pool_lock:
            if worker not in self._workers:
                return
        self._available_workers.put(worker)

    def _ensure_worker_process(self, worker: _KokoroWorkerHandle) -> subprocess.Popen[str]:
        process = worker.process
        if process is not None and process.poll() is None:
            return process
        self._reset_worker(worker)

        worker_python = self.binary_path.parent / "python"
        if not worker_python.exists():
            raise RuntimeError(
                f"kokoro python executable not found at '{worker_python}'. "
                "Install kokoro into /tmp/tts-kokoro and retry."
            )

        self.hf_home.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(self.hf_home))
        env["KOKORO_WORKER_THREADS"] = str(self.config.kokoro_worker_threads)

        repo_src = Path(__file__).resolve().parents[2]
        pythonpath_entries = [str(repo_src)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        process = subprocess.Popen(
            [
                str(worker_python),
                "-u",
                "-m",
                "podcast_agent.tts.kokoro_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(repo_src.parent),
            env=env,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            self._terminate_process(process)
            raise RuntimeError("kokoro worker did not expose stdin/stdout pipes.")
        worker.process = process
        return process

    def _write_worker_request(
        self,
        process: subprocess.Popen[str],
        payload: dict[str, object],
    ) -> None:
        stdin = process.stdin
        if stdin is None:
            raise RuntimeError("kokoro worker stdin is unavailable.")
        try:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("kokoro worker transport failed while sending a request.") from exc

    def _read_worker_response(
        self,
        process: subprocess.Popen[str],
        request_id: int,
    ) -> dict[str, object]:
        line = self._read_worker_response_line(process.stdout, process)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("kokoro worker returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("kokoro worker returned an invalid response.")
        if payload.get("id") != request_id:
            raise RuntimeError("kokoro worker returned a mismatched response id.")
        return payload

    def _read_worker_response_line(
        self,
        stdout: TextIO | None,
        process: subprocess.Popen[str],
    ) -> str:
        if stdout is None:
            raise RuntimeError("kokoro worker stdout is unavailable.")
        ready, _, _ = select.select([stdout], [], [], self.config.timeout_seconds)
        if not ready:
            raise RuntimeError(
                f"kokoro worker timed out after {self.config.timeout_seconds} seconds."
            )
        line = stdout.readline()
        if line:
            return line
        exit_code = process.poll()
        if exit_code is None:
            raise RuntimeError("kokoro worker closed stdout before returning a response.")
        raise RuntimeError(f"kokoro worker exited with code {exit_code}.")

    def _reset_worker(self, worker: _KokoroWorkerHandle) -> None:
        process = worker.process
        worker.process = None
        if process is not None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
