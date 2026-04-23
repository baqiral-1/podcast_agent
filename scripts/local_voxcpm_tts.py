"""Local VoxCPM TTS helper.

Install VoxCPM in a separate local environment to avoid adding heavy ML
dependencies to the main podcast-agent package:

    python3.11 -m venv /tmp/tts-voxcpm
    /tmp/tts-voxcpm/bin/pip install voxcpm
    /tmp/tts-voxcpm/bin/python scripts/local_voxcpm_tts.py \
        --text "VoxCPM2 can synthesize realistic multilingual speech." \
        --output output.wav

Use --control for VoxCPM2 voice design, --reference-audio for controllable
voice cloning, or --prompt-audio with --prompt-text/--prompt-text-file for
ultimate cloning. Use --dry-run to inspect chunking before loading the model.
MP3 output requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import inspect
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
_DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
_DEFAULT_ZIPENHANCER_MODEL_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"
_DEFAULT_MAX_CHUNKS = 120


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize local speech with OpenBMB VoxCPM.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text file to synthesize.",
    )
    prompt_text_group = parser.add_mutually_exclusive_group()
    prompt_text_group.add_argument(
        "--prompt-text",
        help="Transcript for --prompt-audio.",
    )
    prompt_text_group.add_argument(
        "--prompt-text-file",
        type=Path,
        help="UTF-8 transcript file for --prompt-audio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path ending in .wav or .mp3.",
    )
    parser.add_argument(
        "--control",
        help="VoxCPM2 voice/style control text, inserted as a leading parenthetical.",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Reference audio for VoxCPM2 controllable voice cloning.",
    )
    parser.add_argument(
        "--prompt-audio",
        type=Path,
        help="Prompt audio for continuation/ultimate cloning.",
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL_ID,
        help=f"Hugging Face repo id or local model path. Defaults to {_DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached model files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N. Defaults to auto.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="PyTorch/OpenMP CPU thread cap. Pass 0 to disable the cap.",
    )
    parser.add_argument(
        "--load-denoiser",
        action="store_true",
        help="Load VoxCPM's optional ZipEnhancer denoiser model.",
    )
    parser.add_argument(
        "--zipenhancer-model-id",
        default=_DEFAULT_ZIPENHANCER_MODEL_ID,
        help="ZipEnhancer model id or local path used when --load-denoiser is set.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Denoise prompt/reference audio before synthesis. Requires --load-denoiser.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable VoxCPM model optimization during loading.",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance value. VoxCPM2 default is 2.0.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Diffusion inference timesteps. VoxCPM2 default is 10.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=4096,
        help="Maximum VoxCPM generation token length.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable VoxCPM text normalization.",
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
        "--dry-run",
        action="store_true",
        help="Print chunking details and exit before loading VoxCPM.",
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


def _load_prompt_text(prompt_text: str | None, prompt_text_file: Path | None) -> str | None:
    if prompt_text is not None and prompt_text_file is not None:
        raise ValueError("Pass only one of --prompt-text or --prompt-text-file.")
    if prompt_text_file is not None:
        loaded = prompt_text_file.read_text(encoding="utf-8")
    else:
        loaded = prompt_text
    if loaded is None:
        return None
    normalized = _normalize_text(loaded)
    if not normalized:
        raise ValueError("Prompt text is empty.")
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


def _ensure_readable_file(path: Path | None, label: str) -> None:
    if path is None:
        return
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} not found: {path}")


def _ensure_output_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("Output path must end in .wav or .mp3.")
    return suffix


def _validate_generation_options(args: argparse.Namespace, prompt_text: str | None) -> None:
    if args.prompt_audio is None and prompt_text is not None:
        raise ValueError("--prompt-text/--prompt-text-file requires --prompt-audio.")
    if args.prompt_audio is not None and prompt_text is None:
        raise ValueError("--prompt-audio requires --prompt-text or --prompt-text-file.")
    if args.control and prompt_text is not None:
        raise ValueError("--control cannot be combined with prompt audio cloning.")
    if args.denoise and not args.load_denoiser:
        raise ValueError("--denoise requires --load-denoiser.")
    if args.silence_ms < 0:
        raise ValueError("--silence-ms must be non-negative.")
    if args.cfg_value <= 0:
        raise ValueError("--cfg-value must be positive.")
    if args.inference_timesteps < 1:
        raise ValueError("--inference-timesteps must be at least 1.")
    if args.max_len < 1:
        raise ValueError("--max-len must be at least 1.")


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


def _print_chunk_plan(chunks: list[str]) -> None:
    total_chars = sum(len(chunk) for chunk in chunks)
    longest_chunk = max((len(chunk) for chunk in chunks), default=0)
    print(
        "Dry run: "
        f"{len(chunks)} chunks, {total_chars} normalized chars, "
        f"longest chunk {longest_chunk} chars."
    )


def _build_generation_text(text: str, control: str | None) -> str:
    normalized_control = _normalize_text(control or "")
    if not normalized_control:
        return text
    return f"({normalized_control}){text}"


def _load_model(args: argparse.Namespace) -> object:
    try:
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise RuntimeError(
            "voxcpm is not installed. Install it in a local environment, "
            "for example: /tmp/tts-voxcpm/bin/pip install voxcpm"
        ) from exc

    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    model_kwargs: dict[str, object] = {
        "load_denoiser": args.load_denoiser,
        "zipenhancer_model_id": args.zipenhancer_model_id,
        "cache_dir": cache_dir,
        "local_files_only": args.local_files_only,
        "optimize": not args.no_optimize,
    }
    if _callable_accepts_keyword(VoxCPM.from_pretrained, "device"):
        model_kwargs["device"] = args.device
    elif args.device != "auto":
        print(
            "Installed VoxCPM does not expose a device option; ignoring --device.",
            file=sys.stderr,
        )
    return VoxCPM.from_pretrained(args.model_id, **model_kwargs)


def _callable_accepts_keyword(function: object, keyword: str) -> bool:
    signature = inspect.signature(function)
    return keyword in signature.parameters


def _generate_chunk(
    *,
    model: object,
    text: str,
    args: argparse.Namespace,
    prompt_text: str | None,
) -> Any:
    return model.generate(
        text=text,
        prompt_wav_path=str(args.prompt_audio) if args.prompt_audio is not None else None,
        prompt_text=prompt_text,
        reference_wav_path=(
            str(args.reference_audio) if args.reference_audio is not None else None
        ),
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=args.denoise
        and (args.prompt_audio is not None or args.reference_audio is not None),
    )


def _sample_rate(model: object) -> int:
    tts_model = getattr(model, "tts_model", None)
    sample_rate = getattr(tts_model, "sample_rate", None)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RuntimeError("VoxCPM model did not expose a valid sample rate.")
    return sample_rate


def _combine_audio_chunks(
    chunks: list[Any],
    *,
    sample_rate: int,
    silence_ms: int,
) -> Any:
    if not chunks:
        raise ValueError("No audio chunks were generated.")
    if len(chunks) == 1:
        return chunks[0]

    import numpy as np

    arrays: list[Any] = []
    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=chunks[0].dtype) if silence_samples else None
    for index, chunk in enumerate(chunks):
        if index > 0 and silence is not None:
            arrays.append(silence)
        arrays.append(chunk)
    return np.concatenate(arrays)


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


def _write_audio_file(
    *,
    audio: Any,
    sample_rate: int,
    output_path: Path,
    output_suffix: str,
    soundfile_module: ModuleType,
) -> None:
    if output_suffix == ".wav":
        soundfile_module.write(str(output_path), audio, sample_rate)
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_wav_path = Path(handle.name)
    try:
        soundfile_module.write(str(temp_wav_path), audio, sample_rate)
        _convert_wav_to_mp3(temp_wav_path, output_path)
    finally:
        temp_wav_path.unlink(missing_ok=True)


def synthesize(args: argparse.Namespace) -> None:
    output_suffix = _ensure_output_extension(args.output)
    _ensure_readable_file(args.reference_audio, "--reference-audio")
    _ensure_readable_file(args.prompt_audio, "--prompt-audio")
    _ensure_readable_file(args.prompt_text_file, "--prompt-text-file")

    text = _load_input_text(args.text, args.text_file)
    prompt_text = _load_prompt_text(args.prompt_text, args.prompt_text_file)
    _validate_generation_options(args, prompt_text)

    final_text = _build_generation_text(text, args.control)
    text_chunks = _split_text_into_chunks(final_text, args.chunk_max_chars)
    if not text_chunks:
        raise ValueError("Input text is empty.")
    _validate_chunk_plan(text_chunks, args.max_chunks)
    text_chunks = _limit_chunks(text_chunks, args.limit_chunks)

    if args.dry_run:
        _print_chunk_plan(text_chunks)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _configure_cpu_threads(args.cpu_threads)

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required. Install voxcpm in a local Python environment."
        ) from exc

    model = _load_model(args)
    sample_rate = _sample_rate(model)
    audio_chunks = []
    for index, chunk in enumerate(text_chunks, start=1):
        print(f"Synthesizing chunk {index}/{len(text_chunks)} ({len(chunk)} chars)")
        audio_chunks.append(
            _generate_chunk(
                model=model,
                text=chunk,
                args=args,
                prompt_text=prompt_text,
            )
        )

    output_audio = _combine_audio_chunks(
        audio_chunks,
        sample_rate=sample_rate,
        silence_ms=args.silence_ms,
    )
    _write_audio_file(
        audio=output_audio,
        sample_rate=sample_rate,
        output_path=args.output,
        output_suffix=output_suffix,
        soundfile_module=sf,
    )
    print(f"saved: {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        synthesize(args)
    except Exception as exc:
        print(f"local-voxcpm-tts failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
