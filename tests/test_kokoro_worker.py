"""Unit tests for Kokoro worker local model loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("kokoro")

from podcast_agent.tts.kokoro_worker import KokoroWorker


class TestKokoroWorker:
    def test_resolve_snapshot_dir_uses_mlx_layout(self, tmp_path: Path, monkeypatch):
        hf_home = tmp_path / "hf"
        root = hf_home / "hub" / "models--mlx-community--Kokoro-82M-bf16"
        revision = "rev-123"
        snapshot = root / "snapshots" / revision

        (root / "refs").mkdir(parents=True)
        (root / "refs" / "main").write_text(revision, encoding="utf-8")
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "kokoro-v1_0.safetensors").write_bytes(b"not-used")
        (snapshot / "voices").mkdir()

        monkeypatch.setenv("HF_HOME", str(hf_home))
        worker = object.__new__(KokoroWorker)

        resolved = worker._resolve_snapshot_dir()

        assert resolved == snapshot

    def test_resolve_snapshot_dir_requires_required_files(self, tmp_path: Path, monkeypatch):
        hf_home = tmp_path / "hf"
        root = hf_home / "hub" / "models--mlx-community--Kokoro-82M-bf16"
        revision = "rev-123"
        snapshot = root / "snapshots" / revision

        (root / "refs").mkdir(parents=True)
        (root / "refs" / "main").write_text(revision, encoding="utf-8")
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "voices").mkdir()

        monkeypatch.setenv("HF_HOME", str(hf_home))
        worker = object.__new__(KokoroWorker)

        with pytest.raises(FileNotFoundError, match="MLX Kokoro snapshot"):
            worker._resolve_snapshot_dir()

    def test_ensure_model_weights_uses_existing_pth(self, tmp_path: Path, monkeypatch):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        existing_pth = snapshot / "kokoro-v1_0.pth"
        existing_pth.write_bytes(b"ready")

        worker = object.__new__(KokoroWorker)

        def _fail_convert(_: Path, __: Path) -> None:
            raise AssertionError("conversion should not run when pth exists")

        monkeypatch.setattr(worker, "_convert_safetensors_to_pth", _fail_convert)

        resolved = worker._ensure_model_weights(snapshot)

        assert resolved == existing_pth

    def test_convert_safetensors_to_pth_groups_weights(self, tmp_path: Path):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        safetensors_path = snapshot / "kokoro-v1_0.safetensors"
        pth_path = snapshot / "kokoro-v1_0.pth"

        save_file(
            {
                "bert.embeddings.word_embeddings.weight": torch.randn(2, 2),
                "bert_encoder.weight": torch.randn(2, 2),
                "predictor.duration_proj.weight": torch.randn(2, 2),
                "text_encoder.blocks.0.weight": torch.randn(2, 2),
                "decoder.layers.0.weight": torch.randn(2, 2),
            },
            str(safetensors_path),
        )

        worker = object.__new__(KokoroWorker)
        worker._convert_safetensors_to_pth(safetensors_path, pth_path)

        grouped = torch.load(pth_path, map_location="cpu", weights_only=True)

        assert set(grouped) == {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
        assert "embeddings.word_embeddings.weight" in grouped["bert"]
        assert "weight" in grouped["bert_encoder"]
        assert "duration_proj.weight" in grouped["predictor"]
        assert "blocks.0.weight" in grouped["text_encoder"]
        assert "layers.0.weight" in grouped["decoder"]

    def test_voice_path_converts_local_safetensors_voice(self, tmp_path: Path):
        snapshot = tmp_path / "snapshot"
        voice_dir = snapshot / "voices"
        voice_dir.mkdir(parents=True)
        safetensors_voice = voice_dir / "af_heart.safetensors"
        pt_voice = voice_dir / "af_heart.pt"
        save_file({"voice": torch.randn(2, 2)}, str(safetensors_voice))

        worker = object.__new__(KokoroWorker)
        worker._snapshot_dir = snapshot

        resolved = worker._voice_path("af_heart")

        assert resolved == pt_voice
        assert pt_voice.exists()
        loaded = torch.load(pt_voice, map_location="cpu", weights_only=True)
        assert isinstance(loaded, torch.Tensor)
        assert loaded.shape == (2, 2)
