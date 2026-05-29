"""Unit tests for the local VoxCPM TTS helper script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_voxcpm_tts.py"
SPEC = importlib.util.spec_from_file_location("local_voxcpm_tts", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
local_voxcpm_tts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = local_voxcpm_tts
SPEC.loader.exec_module(local_voxcpm_tts)


def test_parse_args_accepts_minimal_design_invocation(tmp_path: Path) -> None:
    args = local_voxcpm_tts._parse_args(
        [
            "--text",
            "hello",
            "--output",
            str(tmp_path / "out.wav"),
        ]
    )

    assert args.text == "hello"
    assert args.output == tmp_path / "out.wav"
    assert args.model_id == "openbmb/VoxCPM2"
    assert args.load_denoiser is False
    assert args.cfg_value == 2.0
    assert args.inference_timesteps == 10


def test_parse_args_rejects_both_text_sources(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("hello", encoding="utf-8")

    with pytest.raises(SystemExit):
        local_voxcpm_tts._parse_args(
            [
                "--text",
                "hello",
                "--text-file",
                str(text_file),
                "--output",
                str(tmp_path / "out.wav"),
            ]
        )


def test_load_input_text_reads_and_trims_file(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("\n  First line.\n\nSecond line.  \n", encoding="utf-8")

    assert local_voxcpm_tts._load_input_text(None, text_file) == "First line. Second line."


def test_load_prompt_text_reads_and_trims_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("\n Reference transcript. \n", encoding="utf-8")

    assert local_voxcpm_tts._load_prompt_text(None, prompt_file) == "Reference transcript."


def test_validate_generation_options_requires_prompt_pair(tmp_path: Path) -> None:
    args = SimpleNamespace(
        prompt_audio=tmp_path / "prompt.wav",
        control=None,
        denoise=False,
        load_denoiser=False,
        silence_ms=350,
        cfg_value=2.0,
        inference_timesteps=10,
        max_len=4096,
    )

    with pytest.raises(ValueError, match="--prompt-audio requires"):
        local_voxcpm_tts._validate_generation_options(args, None)


def test_validate_generation_options_rejects_control_with_prompt_text() -> None:
    args = SimpleNamespace(
        prompt_audio=Path("prompt.wav"),
        control="warm voice",
        denoise=False,
        load_denoiser=False,
        silence_ms=350,
        cfg_value=2.0,
        inference_timesteps=10,
        max_len=4096,
    )

    with pytest.raises(ValueError, match="--control"):
        local_voxcpm_tts._validate_generation_options(args, "Reference transcript.")


def test_validate_generation_options_requires_loaded_denoiser() -> None:
    args = SimpleNamespace(
        prompt_audio=None,
        control=None,
        denoise=True,
        load_denoiser=False,
        silence_ms=350,
        cfg_value=2.0,
        inference_timesteps=10,
        max_len=4096,
    )

    with pytest.raises(ValueError, match="--denoise"):
        local_voxcpm_tts._validate_generation_options(args, None)


def test_build_generation_text_adds_control_parenthetical() -> None:
    assert (
        local_voxcpm_tts._build_generation_text("Hello.", " warm female voice ")
        == "(warm female voice)Hello."
    )


def test_split_text_uses_sentence_boundaries_for_long_text() -> None:
    text = (
        "This is the first sentence with enough detail. "
        "This is the second sentence with enough detail. "
        "This is the third sentence with enough detail."
    )

    chunks = local_voxcpm_tts._split_text_into_chunks(text, 100)

    assert len(chunks) == 2
    assert chunks[0].endswith("detail.")
    assert chunks[1] == "This is the third sentence with enough detail."


def test_configure_cpu_threads_sets_env_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    calls: list[tuple[str, int]] = []
    torch_module = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("threads", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
    )

    local_voxcpm_tts._configure_cpu_threads(2, torch_module)

    assert calls == [("threads", 2), ("interop", 2)]
    assert {
        name: local_voxcpm_tts.os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    } == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    }


def test_dry_run_skips_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = local_voxcpm_tts._parse_args(
        [
            "--text",
            "A short sentence.",
            "--output",
            str(tmp_path / "out.wav"),
            "--dry-run",
        ]
    )
    monkeypatch.setattr(
        local_voxcpm_tts,
        "_load_model",
        lambda args: pytest.fail("_load_model should not be called"),
    )

    local_voxcpm_tts.synthesize(args)

    assert "Dry run: 1 chunks" in capsys.readouterr().out


def test_synthesize_passes_voxcpm_generation_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference")
    calls: dict[str, object] = {}

    class FakeModel:
        tts_model = SimpleNamespace(sample_rate=48000)

        def generate(self, **kwargs: object) -> list[float]:
            calls["generate"] = kwargs
            return [0.0, 0.1]

    fake_sf = SimpleNamespace(
        write=lambda path, audio, sample_rate: calls.update({"write": (path, audio, sample_rate)})
    )
    args = local_voxcpm_tts._parse_args(
        [
            "--text",
            "Hello.",
            "--control",
            "warm voice",
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(tmp_path / "out.wav"),
            "--cfg-value",
            "2.5",
            "--inference-timesteps",
            "12",
            "--max-len",
            "1234",
        ]
    )

    monkeypatch.setattr(local_voxcpm_tts, "_load_model", lambda args: FakeModel())
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    local_voxcpm_tts.synthesize(args)

    assert calls["generate"] == {
        "text": "(warm voice)Hello.",
        "prompt_wav_path": None,
        "prompt_text": None,
        "reference_wav_path": str(reference_audio),
        "cfg_value": 2.5,
        "inference_timesteps": 12,
        "max_len": 1234,
        "normalize": False,
        "denoise": False,
    }
    assert calls["write"] == (str(tmp_path / "out.wav"), [0.0, 0.1], 48000)


def test_convert_wav_to_mp3_requires_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(local_voxcpm_tts.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        local_voxcpm_tts._convert_wav_to_mp3(
            tmp_path / "input.wav",
            tmp_path / "output.mp3",
        )
