#!/usr/bin/env python3
"""Synthesize one render segment with each OpenAI TTS HD voice."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from podcast_agent.config import TTSConfig
from podcast_agent.tts.openai_compatible import OpenAICompatibleTTSClient

OPENAI_TTS_HD_VOICES = (
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
)


class VoiceSampleError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize a single render_manifest segment with each supported "
            "OpenAI tts-1-hd voice and write a listening page."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="runs/war_on_terror_v33",
        help="Run directory containing episodes/<n>/render_manifest.json.",
    )
    parser.add_argument("--episode", type=int, default=1, help="Episode number to sample.")
    parser.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Zero-based index among non-empty render segments. Ignored when --segment-id is set.",
    )
    parser.add_argument("--segment-id", help="Specific render segment id to synthesize.")
    parser.add_argument("--model", default="tts-1-hd", help="OpenAI speech model.")
    parser.add_argument("--speed", type=float, help="Override speed. Defaults to the segment speed.")
    parser.add_argument("--format", default="mp3", help="Audio format to request.")
    parser.add_argument(
        "--voices",
        nargs="+",
        default=list(OPENAI_TTS_HD_VOICES),
        help="Voices to synthesize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults under <run-dir>/voice_samples/.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "sample"


def _select_segment(manifest: dict[str, Any], segment_id: str | None, segment_index: int) -> dict[str, Any]:
    segments = [segment for segment in manifest.get("segments", []) if (segment.get("text") or "").strip()]
    if not segments:
        raise VoiceSampleError("render_manifest.json has no non-empty segments.")
    if segment_id:
        for segment in segments:
            if segment.get("segment_id") == segment_id:
                return segment
        raise VoiceSampleError(f"Segment id not found in render manifest: {segment_id}")
    if segment_index < 0 or segment_index >= len(segments):
        raise VoiceSampleError(
            f"Segment index {segment_index} is out of range for {len(segments)} non-empty segments."
        )
    return segments[segment_index]


def _render_index(
    *,
    output_dir: Path,
    run_dir: Path,
    episode: int,
    model: str,
    speed: float,
    audio_format: str,
    segment: dict[str, Any],
    samples: list[dict[str, str]],
) -> str:
    text = segment.get("text") or ""
    segment_id = segment.get("segment_id", "unknown")
    sample_rows = "\n".join(
        """
        <section class="sample">
          <h2>{voice}</h2>
          <audio controls preload="metadata" src="{file_name}"></audio>
          <p><a href="{file_name}">{file_name}</a></p>
        </section>
        """.format(
            voice=html.escape(sample["voice"]),
            file_name=html.escape(sample["file_name"]),
        )
        for sample in samples
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAI TTS Voice Samples - {html.escape(str(segment_id))}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f3ea;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .meta, .source {{
      margin: 0 0 24px;
      line-height: 1.5;
    }}
    .source {{
      padding: 16px;
      border: 1px solid #d8cfbd;
      background: #fffaf0;
      white-space: pre-wrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .sample {{
      padding: 16px;
      border: 1px solid #d8cfbd;
      background: #fffdf7;
    }}
    .sample h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      text-transform: capitalize;
    }}
    audio {{
      width: 100%;
    }}
  </style>
</head>
<body>
  <main>
    <h1>OpenAI TTS Voice Samples</h1>
    <p class="meta">
      Run: {html.escape(str(run_dir))}<br>
      Episode: {episode}<br>
      Segment: {html.escape(str(segment_id))}<br>
      Model: {html.escape(model)}<br>
      Speed: {speed}<br>
      Format: {html.escape(audio_format)}<br>
      Output: {html.escape(str(output_dir))}
    </p>
    <h2>Source Text</h2>
    <p class="source">{html.escape(text)}</p>
    <div class="grid">
      {sample_rows}
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "episodes" / str(args.episode) / "render_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"render manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    segment = _select_segment(manifest, args.segment_id, args.segment_index)
    segment_id = str(segment.get("segment_id", f"segment-{args.segment_index}"))
    speed = args.speed if args.speed is not None else float(segment.get("speed", 1.0))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            run_dir
            / "voice_samples"
            / f"episode-{args.episode}-{_slug(segment_id)}-{_slug(args.model)}-speed-{speed:g}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TTSConfig(
        provider="openai-compatible",
        model_name=args.model,
        voice="fable",
        audio_format=args.format,
        speed=speed,
    )
    client = OpenAICompatibleTTSClient(config)

    samples: list[dict[str, str]] = []
    text = str(segment.get("text", ""))
    for voice in args.voices:
        file_name = f"{_slug(voice)}.{args.format}"
        output_path = output_dir / file_name
        audio_bytes = client.synthesize(
            text,
            voice=voice,
            audio_format=args.format,
            instructions=None,
            speed=speed,
        )
        output_path.write_bytes(audio_bytes)
        samples.append({"voice": voice, "file_name": file_name})
        print(f"wrote {voice}: {output_path}")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "episode": args.episode,
        "segment_id": segment_id,
        "model": args.model,
        "speed": speed,
        "format": args.format,
        "voices": list(args.voices),
        "text": text,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(
        _render_index(
            output_dir=output_dir,
            run_dir=run_dir,
            episode=args.episode,
            model=args.model,
            speed=speed,
            audio_format=args.format,
            segment=segment,
            samples=samples,
        )
    )
    print(f"index: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
