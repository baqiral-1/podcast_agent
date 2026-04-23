"""Local high-quality Chatterbox TTS helper.

Install Chatterbox in a separate local environment to avoid adding heavy ML
dependencies to the main podcast-agent package:

    python3.11 -m venv /tmp/tts-chatterbox
    /tmp/tts-chatterbox/bin/pip install chatterbox-tts
    /tmp/tts-chatterbox/bin/python scripts/local_chatterbox_tts.py \
        --text-file input.txt \
        --voice-reference voice.wav \
        --output output.wav

For fully offline runs after model files are available, pass --local-model-dir.
Use --dry-run to inspect chunking before loading the model. Multi-chunk outputs
are merged with ffmpeg, which must be available on PATH. CPU runs default to a
small PyTorch/OpenMP thread cap; pass --cpu-threads 0 to disable it.
Use --chunk-output-dir to persist chunk WAVs for resumable runs; combine it
with --isolate-chunks to render each chunk in a fresh child process.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_MAX_CHUNKS = 120


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize local speech with Resemble AI Chatterbox.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text file to synthesize.",
    )
    parser.add_argument(
        "--voice-reference",
        type=Path,
        required=True,
        help="Reference voice clip, ideally 5-20 seconds of clean speech.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path ending in .wav or .mp3.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Inference device. Defaults to auto.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="PyTorch/OpenMP CPU thread cap. Pass 0 to disable the cap.",
    )
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        help="Local Chatterbox model directory for offline model loading.",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=900,
        help="Maximum characters per generated chunk.",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=350,
        help="Silence inserted between chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help="Maximum chunks to synthesize. Pass 0 to disable the guard.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        help="Only synthesize the first N chunks. Useful for smoke tests.",
    )
    parser.add_argument(
        "--chunk-output-dir",
        type=Path,
        help="Directory for reusable generated chunk WAV files.",
    )
    parser.add_argument(
        "--only-chunk",
        type=int,
        help="Only synthesize one 1-based chunk index. Requires --chunk-output-dir.",
    )
    parser.add_argument(
        "--isolate-chunks",
        action="store_true",
        help="Render each chunk in a fresh subprocess, then merge existing chunks.",
    )
    parser.add_argument(
        "--merge-existing-chunks",
        action="store_true",
        help="Merge existing chunk WAV files without loading Chatterbox.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunking details and exit before loading Chatterbox.",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration. Chatterbox default is 0.5.",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight. Lower values can improve pacing for fast voices.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Minimum probability sampling threshold.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty.",
    )
    return parser.parse_args(argv)


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _load_input_text(text: str | None, text_file: Path | None) -> str:
    if text is not None and text_file is not None:
        raise ValueError("Pass exactly one of --text or --text-file.")
    if text is None and text_file is None:
        raise ValueError("Pass exactly one of --text or --text-file.")
    raw_text = text if text is not None else text_file.read_text(encoding="utf-8")
    normalized = _normalize_text(raw_text)
    if not normalized:
        raise ValueError("Input text is empty.")
    return normalized


def _split_long_unit(unit: str, max_chars: int) -> list[str]:
    words = unit.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        projected_len = current_len + len(word) + (1 if current else 0)
        if current and projected_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue
        current.append(word)
        current_len = projected_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    if max_chars < 100:
        raise ValueError("--chunk-max-chars must be at least 100.")

    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    units: list[str] = []
    for sentence in _SENTENCE_BOUNDARY_RE.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_split_long_unit(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        candidate = f"{current} {unit}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        current = unit
    if current:
        chunks.append(current)
    return chunks


def _resolve_device(device: str, torch_module: ModuleType | None = None) -> str:
    if device != "auto":
        return device

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required. Install chatterbox-tts in a local Python environment."
            ) from exc
        torch = imported_torch

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def _configure_cpu_threads(
    cpu_threads: int,
    torch_module: ModuleType | None = None,
) -> None:
    if cpu_threads < 0:
        raise ValueError("--cpu-threads must be non-negative.")
    if cpu_threads == 0:
        return

    thread_count = str(cpu_threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, thread_count)

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError:
            return
        torch = imported_torch

    set_num_threads = getattr(torch, "set_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(cpu_threads)

    set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
    if callable(set_num_interop_threads):
        try:
            set_num_interop_threads(cpu_threads)
        except RuntimeError:
            # PyTorch only allows this before inter-op parallel work starts.
            pass


def _ensure_readable_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} not found: {path}")


def _ensure_output_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("Output path must end in .wav or .mp3.")
    return suffix


def _validate_silence_ms(silence_ms: int) -> None:
    if silence_ms < 0:
        raise ValueError("--silence-ms must be non-negative.")


def _validate_chunk_plan(chunks: list[str], max_chunks: int) -> None:
    if max_chunks < 0:
        raise ValueError("--max-chunks must be non-negative.")
    if max_chunks and len(chunks) > max_chunks:
        raise ValueError(
            f"Input split into {len(chunks)} chunks, exceeding --max-chunks "
            f"{max_chunks}. Use shorter input, increase --chunk-max-chars, or pass "
            "--max-chunks 0 to disable this guard."
        )


def _limit_chunks(chunks: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return chunks
    if limit < 1:
        raise ValueError("--limit-chunks must be at least 1.")
    return chunks[:limit]


def _validate_only_chunk(only_chunk: int | None, chunk_count: int) -> None:
    if only_chunk is None:
        return
    if only_chunk < 1:
        raise ValueError("--only-chunk must be at least 1.")
    if only_chunk > chunk_count:
        raise ValueError(
            f"--only-chunk {only_chunk} exceeds the {chunk_count} planned chunks."
        )


def _print_chunk_plan(chunks: list[str]) -> None:
    total_chars = sum(len(chunk) for chunk in chunks)
    longest_chunk = max((len(chunk) for chunk in chunks), default=0)
    print(
        "Dry run: "
        f"{len(chunks)} chunks, {total_chars} normalized chars, "
        f"longest chunk {longest_chunk} chars."
    )


def _default_chunk_output_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_chunks"


def _require_ffmpeg(action: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(f"ffmpeg is required to {action}.")
    return ffmpeg_path


def _run_ffmpeg(command: list[str], error_prefix: str) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(error_prefix + (f": {error_text}" if error_text else "."))


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg_path = _require_ffmpeg("write MP3 output")
    _run_ffmpeg(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        "ffmpeg failed to write MP3 output",
    )


def _quote_concat_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "'\\''")


def _write_ffmpeg_concat_file(paths: list[Path], concat_file: Path) -> None:
    lines = [f"file '{_quote_concat_path(path)}'" for path in paths]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_audio_files(paths: list[Path], output_path: Path, output_suffix: str) -> None:
    if not paths:
        raise ValueError("No audio chunks were generated.")

    ffmpeg_path = _require_ffmpeg("merge generated audio chunks")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        concat_file = Path(handle.name)
    try:
        _write_ffmpeg_concat_file(paths, concat_file)
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vn",
        ]
        if output_suffix == ".mp3":
            command.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
        else:
            command.extend(["-codec:a", "pcm_s16le"])
        command.append(str(output_path))
        _run_ffmpeg(command, "ffmpeg failed to merge generated audio chunks")
    finally:
        concat_file.unlink(missing_ok=True)


def _chunk_wav_path(chunk_output_dir: Path, index: int) -> Path:
    return chunk_output_dir / f"chunk-{index:04d}.wav"


def _silence_wav_path(chunk_output_dir: Path, silence_ms: int) -> Path:
    return chunk_output_dir / f"silence-{silence_ms}ms.wav"


def _existing_chunk_paths(chunk_output_dir: Path, chunk_count: int) -> list[Path]:
    return [_chunk_wav_path(chunk_output_dir, index) for index in range(1, chunk_count + 1)]


def _build_audio_sequence(
    *,
    chunk_paths: list[Path],
    silence_path: Path | None,
) -> list[Path]:
    audio_paths: list[Path] = []
    for index, chunk_path in enumerate(chunk_paths, start=1):
        if index > 1 and silence_path is not None:
            audio_paths.append(silence_path)
        audio_paths.append(chunk_path)
    return audio_paths


class _NoOpPerthWatermarker:
    """Fallback for local platforms where resemble-perth has no native watermarker."""

    def apply_watermark(self, wav: object, *, sample_rate: int) -> object:
        del sample_rate
        return wav


def _patch_missing_perth_watermarker() -> None:
    try:
        import perth
    except ImportError:
        return
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = _NoOpPerthWatermarker


def _load_model(*, device: str, local_model_dir: Path | None) -> object:
    _patch_missing_perth_watermarker()
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as exc:
        raise RuntimeError(
            "chatterbox-tts is not installed. Install it in a local environment, "
            "for example: /tmp/tts-chatterbox/bin/pip install chatterbox-tts"
        ) from exc

    if local_model_dir is not None:
        if not local_model_dir.exists() or not local_model_dir.is_dir():
            raise ValueError(f"Local model directory not found: {local_model_dir}")
        return ChatterboxTTS.from_local(local_model_dir, device)

    return ChatterboxTTS.from_pretrained(device=device)


def _audio_for_save(audio: Any) -> Any:
    detach = getattr(audio, "detach", None)
    if callable(detach):
        audio = detach()
    cpu = getattr(audio, "cpu", None)
    if callable(cpu):
        audio = cpu()
    return audio


def _make_silence_like(audio: Any, sample_rate: int, silence_ms: int) -> Any:
    import torch

    channels = int(audio.shape[0])
    silence_samples = int(sample_rate * silence_ms / 1000)
    return torch.zeros((channels, silence_samples), dtype=audio.dtype)


def _generate_audio_files(
    *,
    text_chunks: list[str],
    model: object,
    torchaudio: ModuleType,
    args: argparse.Namespace,
    chunk_output_dir: Path,
) -> list[Path]:
    silence_path: Path | None = None

    for index, chunk in enumerate(text_chunks, start=1):
        chunk_path = _chunk_wav_path(chunk_output_dir, index)
        if chunk_path.exists():
            print(f"Skipping existing chunk {index}/{len(text_chunks)}: {chunk_path}")
            continue

        print(f"Synthesizing chunk {index}/{len(text_chunks)} ({len(chunk)} chars)")
        generated_audio = model.generate(
            chunk,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
        audio_for_save = _audio_for_save(generated_audio)

        torchaudio.save(str(chunk_path), audio_for_save, model.sr)

        if (
            silence_path is None
            and args.silence_ms > 0
            and len(text_chunks) > 1
        ):
            silence_path = _silence_wav_path(chunk_output_dir, args.silence_ms)
            if silence_path.exists():
                del generated_audio
                del audio_for_save
                continue
            torchaudio.save(
                str(silence_path),
                _make_silence_like(audio_for_save, model.sr, args.silence_ms),
                model.sr,
            )

        del generated_audio
        del audio_for_save

    chunk_paths = _existing_chunk_paths(chunk_output_dir, len(text_chunks))
    missing_chunks = [path for path in chunk_paths if not path.exists()]
    if missing_chunks:
        missing_text = ", ".join(str(path) for path in missing_chunks)
        raise RuntimeError(f"Missing generated chunk files: {missing_text}")

    if args.silence_ms > 0 and len(text_chunks) > 1:
        silence_path = _silence_wav_path(chunk_output_dir, args.silence_ms)
        if not silence_path.exists():
            raise RuntimeError(f"Missing generated silence file: {silence_path}")

    return _build_audio_sequence(chunk_paths=chunk_paths, silence_path=silence_path)


def _synthesize_only_chunk(
    *,
    text_chunks: list[str],
    model: object,
    torchaudio: ModuleType,
    args: argparse.Namespace,
    chunk_output_dir: Path,
) -> None:
    assert args.only_chunk is not None
    index = args.only_chunk
    chunk = text_chunks[index - 1]
    chunk_path = _chunk_wav_path(chunk_output_dir, index)
    if chunk_path.exists():
        print(f"Skipping existing chunk {index}/{len(text_chunks)}: {chunk_path}")
        return

    print(f"Synthesizing chunk {index}/{len(text_chunks)} ({len(chunk)} chars)")
    generated_audio = model.generate(
        chunk,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    audio_for_save = _audio_for_save(generated_audio)
    torchaudio.save(str(chunk_path), audio_for_save, model.sr)

    if args.silence_ms > 0 and len(text_chunks) > 1:
        silence_path = _silence_wav_path(chunk_output_dir, args.silence_ms)
        if not silence_path.exists():
            torchaudio.save(
                str(silence_path),
                _make_silence_like(audio_for_save, model.sr, args.silence_ms),
                model.sr,
            )

    del generated_audio
    del audio_for_save


def _merge_existing_chunks(
    *,
    text_chunks: list[str],
    args: argparse.Namespace,
    output_suffix: str,
) -> None:
    if args.chunk_output_dir is None:
        raise ValueError("--merge-existing-chunks requires --chunk-output-dir.")
    chunk_paths = _existing_chunk_paths(args.chunk_output_dir, len(text_chunks))
    missing_chunks = [path for path in chunk_paths if not path.exists()]
    if missing_chunks:
        missing_text = ", ".join(str(path) for path in missing_chunks)
        raise RuntimeError(f"Missing generated chunk files: {missing_text}")

    silence_path = None
    if args.silence_ms > 0 and len(text_chunks) > 1:
        silence_path = _silence_wav_path(args.chunk_output_dir, args.silence_ms)
        if not silence_path.exists():
            raise RuntimeError(f"Missing generated silence file: {silence_path}")

    audio_paths = _build_audio_sequence(chunk_paths=chunk_paths, silence_path=silence_path)
    if len(audio_paths) == 1 and output_suffix == ".wav":
        shutil.copy2(audio_paths[0], args.output)
        return
    if len(audio_paths) == 1:
        _convert_wav_to_mp3(audio_paths[0], args.output)
        return
    _concat_audio_files(audio_paths, args.output, output_suffix)


def _add_path_arg(command: list[str], name: str, path: Path | None) -> None:
    if path is not None:
        command.extend([name, str(path)])


def _add_value_arg(
    command: list[str],
    name: str,
    value: str | int | float | None,
) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _isolated_chunk_command(args: argparse.Namespace, chunk_index: int) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    if args.text is not None:
        command.extend(["--text", args.text])
    else:
        command.extend(["--text-file", str(args.text_file)])

    command.extend(
        [
            "--voice-reference",
            str(args.voice_reference),
            "--output",
            str(args.output),
            "--device",
            args.device,
            "--cpu-threads",
            str(args.cpu_threads),
            "--chunk-max-chars",
            str(args.chunk_max_chars),
            "--silence-ms",
            str(args.silence_ms),
            "--max-chunks",
            str(args.max_chunks),
            "--chunk-output-dir",
            str(args.chunk_output_dir),
            "--only-chunk",
            str(chunk_index),
            "--exaggeration",
            str(args.exaggeration),
            "--cfg-weight",
            str(args.cfg_weight),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--min-p",
            str(args.min_p),
            "--repetition-penalty",
            str(args.repetition_penalty),
        ]
    )
    _add_path_arg(command, "--local-model-dir", args.local_model_dir)
    _add_value_arg(command, "--limit-chunks", args.limit_chunks)
    return command


def _run_isolated_chunks(
    *,
    text_chunks: list[str],
    args: argparse.Namespace,
    output_suffix: str,
) -> None:
    assert args.chunk_output_dir is not None
    args.chunk_output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, len(text_chunks) + 1):
        chunk_path = _chunk_wav_path(args.chunk_output_dir, index)
        if chunk_path.exists():
            print(f"Skipping existing chunk {index}/{len(text_chunks)}: {chunk_path}")
            continue

        print(f"Rendering isolated chunk {index}/{len(text_chunks)}")
        result = subprocess.run(
            _isolated_chunk_command(args, index),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Isolated chunk {index}/{len(text_chunks)} failed "
                f"with exit code {result.returncode}."
            )

    _merge_existing_chunks(
        text_chunks=text_chunks,
        args=args,
        output_suffix=output_suffix,
    )


def synthesize(args: argparse.Namespace) -> None:
    _ensure_readable_file(args.voice_reference, "--voice-reference")
    output_suffix = _ensure_output_extension(args.output)
    _validate_silence_ms(args.silence_ms)
    text = _load_input_text(args.text, args.text_file)
    text_chunks = _split_text_into_chunks(text, args.chunk_max_chars)
    if not text_chunks:
        raise ValueError("Input text is empty.")
    _validate_chunk_plan(text_chunks, args.max_chunks)
    text_chunks = _limit_chunks(text_chunks, args.limit_chunks)
    _validate_only_chunk(args.only_chunk, len(text_chunks))

    if args.isolate_chunks and args.only_chunk is not None:
        raise ValueError("--isolate-chunks cannot be combined with --only-chunk.")
    if args.isolate_chunks and args.merge_existing_chunks:
        raise ValueError(
            "--isolate-chunks cannot be combined with --merge-existing-chunks."
        )
    if args.isolate_chunks and args.chunk_output_dir is None:
        args.chunk_output_dir = _default_chunk_output_dir(args.output)
    if args.only_chunk is not None and args.chunk_output_dir is None:
        raise ValueError("--only-chunk requires --chunk-output-dir.")
    if args.merge_existing_chunks and args.chunk_output_dir is None:
        raise ValueError("--merge-existing-chunks requires --chunk-output-dir.")

    if args.dry_run:
        _print_chunk_plan(text_chunks)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_existing_chunks:
        _merge_existing_chunks(
            text_chunks=text_chunks,
            args=args,
            output_suffix=output_suffix,
        )
        return

    if args.isolate_chunks:
        _run_isolated_chunks(
            text_chunks=text_chunks,
            args=args,
            output_suffix=output_suffix,
        )
        return

    _configure_cpu_threads(args.cpu_threads)
    device = _resolve_device(args.device)

    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio is required. Install chatterbox-tts in a local Python environment."
        ) from exc

    model = _load_model(device=device, local_model_dir=args.local_model_dir)
    model.prepare_conditionals(
        str(args.voice_reference),
        exaggeration=args.exaggeration,
    )

    if args.chunk_output_dir is not None:
        args.chunk_output_dir.mkdir(parents=True, exist_ok=True)
        if args.only_chunk is not None:
            _synthesize_only_chunk(
                text_chunks=text_chunks,
                model=model,
                torchaudio=torchaudio,
                args=args,
                chunk_output_dir=args.chunk_output_dir,
            )
            return

        audio_paths = _generate_audio_files(
            text_chunks=text_chunks,
            model=model,
            torchaudio=torchaudio,
            args=args,
            chunk_output_dir=args.chunk_output_dir,
        )
        if len(audio_paths) == 1 and output_suffix == ".wav":
            shutil.copy2(audio_paths[0], args.output)
            return
        if len(audio_paths) == 1:
            _convert_wav_to_mp3(audio_paths[0], args.output)
            return
        _concat_audio_files(audio_paths, args.output, output_suffix)
        return

    with tempfile.TemporaryDirectory(prefix="local-chatterbox-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        audio_paths = _generate_audio_files(
            text_chunks=text_chunks,
            model=model,
            torchaudio=torchaudio,
            args=args,
            chunk_output_dir=temp_dir,
        )
        if len(audio_paths) == 1 and output_suffix == ".wav":
            shutil.move(str(audio_paths[0]), str(args.output))
            return
        if len(audio_paths) == 1:
            _convert_wav_to_mp3(audio_paths[0], args.output)
            return
        _concat_audio_files(audio_paths, args.output, output_suffix)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        synthesize(args)
    except Exception as exc:
        print(f"local-chatterbox-tts failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
