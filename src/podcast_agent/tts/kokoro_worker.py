"""Persistent Kokoro worker process used by the main application."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from kokoro import KPipeline
from kokoro.model import KModel

PROTOCOL_STDOUT = sys.stdout
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


class KokoroWorker:
    """Serve Kokoro synthesis requests over stdin/stdout."""

    def __init__(self) -> None:
        thread_count = max(1, int(os.environ.get("KOKORO_WORKER_THREADS", "4")))
        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(1)
        self._snapshot_dir = self._resolve_snapshot_dir()
        self._config_path = self._snapshot_dir / "config.json"
        self._model_path = self._snapshot_dir / "kokoro-v1_0.pth"
        self._pipelines: dict[str, KPipeline] = {}
        self._models: dict[str, KModel] = {}

    def run(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                payload = json.loads(line)
                request_id = payload["id"]
                text = payload["text"]
                voice = payload["voice"]
                speed = payload["speed"]
                audio_bytes = self._synthesize(text=text, voice=voice, speed=speed)
                self._write_response(
                    {
                        "id": request_id,
                        "ok": True,
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                    }
                )
            except Exception as exc:
                self._write_response(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": f"kokoro worker failed: {exc}",
                    }
                )

    def _synthesize(self, *, text: str, voice: str, speed: float) -> bytes:
        buffer = io.BytesIO()
        voice_path = self._voice_path(voice)
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            with contextlib.redirect_stdout(sys.stderr):
                for result in self._pipeline_for_voice(voice)(
                    text,
                    voice=str(voice_path),
                    speed=speed,
                    split_pattern=r"\n+",
                ):
                    if result.audio is None:
                        continue
                    audio_bytes = (result.audio.numpy() * 32767).astype(np.int16).tobytes()
                    wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

    def _pipeline_for_voice(self, voice: str) -> KPipeline:
        lang_code = voice[0].lower()
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            with contextlib.redirect_stdout(sys.stderr):
                model = self._models.get(lang_code)
                if model is None:
                    model = KModel(
                        repo_id=KOKORO_REPO_ID,
                        config=str(self._config_path),
                        model=str(self._model_path),
                    )
                    model = model.to("cpu").eval()
                    self._models[lang_code] = model
                pipeline = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO_ID, model=model)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def _voice_path(self, voice: str) -> Path:
        if voice.endswith(".pt"):
            return Path(voice)
        voice_path = self._snapshot_dir / "voices" / f"{voice}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Local Kokoro voice not found: {voice_path}")
        return voice_path

    def _resolve_snapshot_dir(self) -> Path:
        hf_home = Path(os.environ.get("HF_HOME", "/tmp/hf_cache_kokoro"))
        candidates = [
            hf_home / "hub" / "models--hexgrad--Kokoro-82M",
            hf_home / "models--hexgrad--Kokoro-82M",
        ]
        for root in candidates:
            refs_main = root / "refs" / "main"
            if not refs_main.exists():
                continue
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = root / "snapshots" / revision
            if (snapshot_dir / "config.json").exists() and (snapshot_dir / "kokoro-v1_0.pth").exists():
                return snapshot_dir
        raise FileNotFoundError(
            f"Unable to locate a local Kokoro snapshot under '{hf_home}'."
        )

    @staticmethod
    def _write_response(payload: dict[str, object]) -> None:
        PROTOCOL_STDOUT.write(json.dumps(payload) + "\n")
        PROTOCOL_STDOUT.flush()


def main() -> None:
    KokoroWorker().run()


if __name__ == "__main__":
    main()
