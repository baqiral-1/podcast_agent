# Podcast Agent

`podcast-agent` is a modular book-to-podcast pipeline that converts a source book into grounded podcast episode plans, episode scripts, TTS-ready render manifests, and optional synthesized audio.

## Overview

- Library-first Python package with a Typer CLI.
- LLM-backed pipeline with strict JSON contracts between stages.
- Optional PostgreSQL-backed retrieval for chunk storage and embeddings.
- Spoken-delivery rewrite pass for natural narration without changing facts.
- Optional audio synthesis from validated render manifests.
- Deterministic orchestration with validated artifacts at every stage.

For a full architecture guide, stage flow, and artifact details, see `docs/index.html`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com
export LLM_MODEL_NAME=gpt-4o-mini
export DATABASE_URL=postgresql://postgres:secret@localhost:5432/podcast_agent
```

Notes:
- `OPENAI_BASE_URL` can point at any OpenAI-compatible server.
- Use `--tts-provider kokoro` on audio commands to switch from OpenAI-compatible TTS.

## CLI Quickstart

```bash
podcast-agent ingest-book ./examples/book.txt --title "Example Book" --author "Author"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2 --database-url "$DATABASE_URL"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2 --with-audio
podcast-agent run-batch ./examples/batch_manifest.sample.json
podcast-agent render-audio-from-manifest ./.podcast_agent/runs/example-book-20260315T1106/example-book/episode-1/episode_output.json
```

## Outputs

Each run writes artifacts under `.podcast_agent/runs/<book-title>-<timestamp-to-minute>/`, including a `run.log` with stage transitions, prompt metadata for successful LLM requests (and full prompts only on LLM errors), plus responses.

Each episode directory contains a canonical `episode_output.json` artifact plus companion script/manifest files and optional final audio.

## Development

```bash
pytest
```
