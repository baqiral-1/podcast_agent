"""Unit tests for the local Chatterbox TTS helper script."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_chatterbox_tts.py"
SPEC = importlib.util.spec_from_file_location("local_chatterbox_tts", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
local_chatterbox_tts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = local_chatterbox_tts
SPEC.loader.exec_module(local_chatterbox_tts)


def test_parse_args_rejects_both_text_sources(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    voice_file = tmp_path / "voice.wav"
    text_file.write_text("hello", encoding="utf-8")
    voice_file.write_bytes(b"voice")

    with pytest.raises(SystemExit):
        local_chatterbox_tts._parse_args(
            [
                "--text",
                "hello",
                "--text-file",
                str(text_file),
                "--voice-reference",
                str(voice_file),
                "--output",
                str(tmp_path / "out.wav"),
            ]
        )


def test_parse_args_rejects_missing_text_source(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")

    with pytest.raises(SystemExit):
        local_chatterbox_tts._parse_args(
            [
                "--voice-reference",
                str(voice_file),
                "--output",
                str(tmp_path / "out.wav"),
            ]
        )


def test_parse_args_accepts_dry_run_and_max_chunks(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")

    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "hello",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--dry-run",
            "--max-chunks",
            "3",
        ]
    )

    assert args.dry_run is True
    assert args.max_chunks == 3
    assert args.cpu_threads == 2


def test_parse_args_accepts_chunk_resume_options(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")
    chunk_dir = tmp_path / "chunks"

    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "hello",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--chunk-output-dir",
            str(chunk_dir),
            "--only-chunk",
            "2",
            "--merge-existing-chunks",
        ]
    )

    assert args.chunk_output_dir == chunk_dir
    assert args.only_chunk == 2
    assert args.merge_existing_chunks is True


def test_parse_args_accepts_isolated_chunks(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")

    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "hello",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--isolate-chunks",
        ]
    )

    assert args.isolate_chunks is True


def test_load_input_text_reads_and_trims_file(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("\n  First line.\n\nSecond line.  \n", encoding="utf-8")

    assert local_chatterbox_tts._load_input_text(None, text_file) == "First line. Second line."


def test_load_input_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        local_chatterbox_tts._load_input_text("   \n", None)


def test_split_text_preserves_short_text_as_one_chunk() -> None:
    chunks = local_chatterbox_tts._split_text_into_chunks("A short sentence.", 900)

    assert chunks == ["A short sentence."]


def test_split_text_uses_sentence_boundaries_for_long_text() -> None:
    text = (
        "This is the first sentence with enough detail. "
        "This is the second sentence with enough detail. "
        "This is the third sentence with enough detail."
    )

    chunks = local_chatterbox_tts._split_text_into_chunks(text, 100)

    assert len(chunks) == 2
    assert chunks[0].endswith("detail.")
    assert chunks[1] == "This is the third sentence with enough detail."


def test_validate_chunk_plan_rejects_too_many_chunks() -> None:
    with pytest.raises(ValueError, match="exceeding --max-chunks 1"):
        local_chatterbox_tts._validate_chunk_plan(["first", "second"], 1)


def test_validate_chunk_plan_allows_disabled_guard() -> None:
    local_chatterbox_tts._validate_chunk_plan(["first", "second"], 0)


def test_limit_chunks_returns_prefix() -> None:
    assert local_chatterbox_tts._limit_chunks(["first", "second"], 1) == ["first"]


def test_limit_chunks_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="--limit-chunks"):
        local_chatterbox_tts._limit_chunks(["first"], 0)


def test_validate_only_chunk_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        local_chatterbox_tts._validate_only_chunk(3, 2)


def test_resolve_auto_device_prefers_cuda() -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )

    assert local_chatterbox_tts._resolve_device("auto", torch_module) == "cuda"


def test_resolve_auto_device_falls_back_to_cpu() -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    assert local_chatterbox_tts._resolve_device("auto", torch_module) == "cpu"


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

    local_chatterbox_tts._configure_cpu_threads(2, torch_module)

    assert calls == [("threads", 2), ("interop", 2)]
    assert {
        name: local_chatterbox_tts.os.environ[name]
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


def test_configure_cpu_threads_rejects_negative() -> None:
    with pytest.raises(ValueError, match="--cpu-threads"):
        local_chatterbox_tts._configure_cpu_threads(-1, SimpleNamespace())


def test_convert_wav_to_mp3_requires_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_chatterbox_tts.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        local_chatterbox_tts._convert_wav_to_mp3(
            tmp_path / "input.wav",
            tmp_path / "output.mp3",
        )


def test_dry_run_skips_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")
    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "A short sentence.",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--dry-run",
        ]
    )
    monkeypatch.setattr(
        local_chatterbox_tts,
        "_load_model",
        lambda **kwargs: pytest.fail("_load_model should not be called"),
    )

    local_chatterbox_tts.synthesize(args)

    assert "Dry run: 1 chunks" in capsys.readouterr().out


def test_write_ffmpeg_concat_file_quotes_paths(tmp_path: Path) -> None:
    concat_file = tmp_path / "concat.txt"

    local_chatterbox_tts._write_ffmpeg_concat_file(
        [tmp_path / "plain.wav", tmp_path / "has'quote.wav"],
        concat_file,
    )

    assert concat_file.read_text(encoding="utf-8").splitlines() == [
        f"file '{tmp_path}/plain.wav'",
        f"file '{tmp_path}/has'\\''quote.wav'",
    ]


def test_build_audio_sequence_inserts_silence(tmp_path: Path) -> None:
    chunks = [tmp_path / "chunk-0001.wav", tmp_path / "chunk-0002.wav"]
    silence = tmp_path / "silence.wav"

    assert local_chatterbox_tts._build_audio_sequence(
        chunk_paths=chunks,
        silence_path=silence,
    ) == [chunks[0], silence, chunks[1]]


def test_merge_existing_chunks_uses_persisted_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk-0001.wav").write_bytes(b"chunk1")
    (chunk_dir / "chunk-0002.wav").write_bytes(b"chunk2")
    (chunk_dir / "silence-350ms.wav").write_bytes(b"silence")
    output = tmp_path / "out.mp3"
    captured: dict[str, object] = {}

    def fake_concat(paths: list[Path], output_path: Path, output_suffix: str) -> None:
        captured["paths"] = paths
        captured["output_path"] = output_path
        captured["output_suffix"] = output_suffix

    monkeypatch.setattr(local_chatterbox_tts, "_concat_audio_files", fake_concat)
    args = SimpleNamespace(
        chunk_output_dir=chunk_dir,
        silence_ms=350,
        output=output,
    )

    local_chatterbox_tts._merge_existing_chunks(
        text_chunks=["first", "second"],
        args=args,
        output_suffix=".mp3",
    )

    assert captured == {
        "paths": [
            chunk_dir / "chunk-0001.wav",
            chunk_dir / "silence-350ms.wav",
            chunk_dir / "chunk-0002.wav",
        ],
        "output_path": output,
        "output_suffix": ".mp3",
    }


def test_isolated_chunk_command_renders_one_chunk(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")
    chunk_dir = tmp_path / "chunks"
    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "hello",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / "out.mp3"),
            "--chunk-output-dir",
            str(chunk_dir),
            "--limit-chunks",
            "3",
            "--isolate-chunks",
            "--dry-run",
        ]
    )

    command = local_chatterbox_tts._isolated_chunk_command(args, 2)

    assert command[:2] == [
        local_chatterbox_tts.sys.executable,
        str(SCRIPT_PATH),
    ]
    assert "--only-chunk" in command
    assert command[command.index("--only-chunk") + 1] == "2"
    assert "--chunk-output-dir" in command
    assert command[command.index("--chunk-output-dir") + 1] == str(chunk_dir)
    assert "--limit-chunks" in command
    assert "--isolate-chunks" not in command
    assert "--merge-existing-chunks" not in command
    assert "--dry-run" not in command


def test_isolate_chunks_uses_default_chunk_dir_without_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")
    output = tmp_path / "out.mp3"
    captured: dict[str, object] = {}
    args = local_chatterbox_tts._parse_args(
        [
            "--text",
            "A short sentence.",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(output),
            "--isolate-chunks",
        ]
    )

    def fake_run_isolated_chunks(
        *,
        text_chunks: list[str],
        args: argparse.Namespace,
        output_suffix: str,
    ) -> None:
        captured["text_chunks"] = text_chunks
        captured["chunk_output_dir"] = args.chunk_output_dir
        captured["output_suffix"] = output_suffix

    monkeypatch.setattr(
        local_chatterbox_tts,
        "_load_model",
        lambda **kwargs: pytest.fail("_load_model should not be called"),
    )
    monkeypatch.setattr(
        local_chatterbox_tts,
        "_run_isolated_chunks",
        fake_run_isolated_chunks,
    )

    local_chatterbox_tts.synthesize(args)

    assert captured == {
        "text_chunks": ["A short sentence."],
        "chunk_output_dir": tmp_path / "out_chunks",
        "output_suffix": ".mp3",
    }


def test_isolate_chunks_rejects_child_or_merge_modes(tmp_path: Path) -> None:
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"voice")

    for extra_arg in ("--only-chunk", "--merge-existing-chunks"):
        args_list = [
            "--text",
            "A short sentence.",
            "--voice-reference",
            str(voice_file),
            "--output",
            str(tmp_path / f"{extra_arg.removeprefix('--')}.mp3"),
            "--chunk-output-dir",
            str(tmp_path / "chunks"),
            "--isolate-chunks",
            extra_arg,
        ]
        if extra_arg == "--only-chunk":
            args_list.append("1")
        args = local_chatterbox_tts._parse_args(args_list)

        with pytest.raises(ValueError, match="--isolate-chunks"):
            local_chatterbox_tts.synthesize(args)
