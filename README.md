# Podcast Agent

`podcast-agent` is a modular book-to-podcast pipeline that converts a source book into grounded podcast episode plans, episode scripts, TTS-ready render manifests, and optional synthesized audio output. The project is library-first with a Typer CLI, PostgreSQL-backed retrieval, strict JSON contracts between stages, and specialized sub-agents for structure, analysis, planning, writing, validation, and repair.

## Overview

The current implementation is designed around a few fixed principles:

- Core middle-stage agents use a shared `LLMClient` interface.
- The default runtime model path is a real OpenAI-compatible JSON client.
- A heuristic adapter remains available as an explicit offline and test fallback.
- Every inter-stage artifact is schema-validated with strict Pydantic models before the next stage runs.
- Chunk storage happens during `index-book`, immediately after `BookStructure` is created.
- Grounding validation happens before render-manifest generation, and repair is a separate targeted step.
- Audio synthesis is optional and runs only after a validated render manifest exists.
- Episodes are sized by assigned source-book volume, with a default minimum of `50,000` source words per episode when the book is large enough.
- Books shorter than that source-word floor collapse to one episode that covers the full book.
- Writing validates that the generated script is proportional to the assigned source material before rendering or TTS.
- Writing now runs one LLM call per beat, selecting only the top 6 beat-local chunks per call and merging the results into one final episode script.
- Structuring is incremental and runs up to 3 chapter-level calls in parallel, with smaller fallbacks only when a chapter is too large.

## Architecture

The architecture guide now lives in [docs/index.html](docs/index.html), including the stage-by-stage execution flow, persisted artifacts, contract rules, and project layout.

At a high level, the pipeline alternates between deterministic orchestration steps and LLM-backed sub-agents:

1. `ingest-book`
2. `structuring_agent`
3. `index-book`
4. `analysis_agent`
5. `episode_planning_agent`
6. `writing_agent`
7. `grounding_validation_agent`
8. `repair_agent`
9. `render-manifest`
10. `synthesize-audio`

## Project Layout

```text
src/podcast_agent/
  agents/        # Agent classes and stage-specific logic
  cli/           # Typer commands
  db/            # Repository and artifact persistence
  llm/           # OpenAI-compatible and heuristic LLM adapters
  pipeline/      # Orchestration across all stages
  retrieval/     # Embeddings and retrieval helpers
  schemas/       # Strict stage contracts
  tts/           # OpenAI-compatible speech synthesis clients
sql/             # Directly runnable PostgreSQL setup
tests/           # Unit and integration tests
```

The repository also includes long-form sample inputs under `examples/`:

- `examples/river_of_hours.txt`
- `examples/houses_of_the_self.txt`
- `examples/stars_beneath_the_skin.txt`

## Database Setup

Bootstrap PostgreSQL with the included SQL file:

```bash
psql "$DATABASE_URL" -f sql/001_init.sql
```

The schema enables `pgvector` and creates tables for:

- `books`: ingested source metadata and raw source text
- `chapters`: canonical chapter records linked to each book
- `chunks`: chunked source content with offsets, ordering, and theme hints
- `chunk_embeddings`: `pgvector` embeddings for stored chunks
- `episode_plans`: persisted episode planning artifacts
- `episode_scripts`: generated script artifacts
- `grounding_reports`: claim-level validation outputs
- `repair_attempts`: persisted repair results per episode and attempt number

It also creates the following supporting indexes:

- `idx_chapters_book_id`
- `idx_chunks_book_id`
- `idx_chunk_embeddings_book_id`

Current retrieval is `pgvector`-only. A follow-up enhancement should add PostgreSQL full-text ranking or BM25-style lexical scoring over `chunks.text` for true hybrid retrieval.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Model Configuration

The default runtime client expects an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com
```

`OPENAI_BASE_URL` can point at any OpenAI-compatible server. The default model name is `gpt-4o-mini`; override it through `Settings.llm.model_name` in code if needed.
The default request timeout for live model calls is 300 seconds.

TTS uses the same OpenAI-compatible endpoint family by default:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com
```

The default speech model is `gpt-4o-mini-tts`, the default voice is `alloy`, and audio is written as `mp3` unless `Settings.tts` is overridden in code. Audio synthesis now runs up to `4` concurrent TTS requests by default via `Settings.pipeline.audio_parallelism`, with `Settings.pipeline.audio_retry_attempts` controlling per-segment retries. Final episode audio is still assembled by concatenating synthesized segment bytes in manifest order, so the current output path should be treated as `mp3`-oriented.

For offline tests or local deterministic runs, inject the heuristic adapter explicitly instead of relying on the default runtime client.

The database DSN can be supplied in either of two ways:

```bash
export DATABASE_URL=postgresql://postgres:secret@localhost:5432/podcast_agent
```

or per command:

```bash
podcast-agent index-book ./examples/book.txt --database-url "$DATABASE_URL"
```

## CLI Usage

The CLI currently supports stage-wise execution and end-to-end orchestration:

Input files may be plain text, markdown, or PDF on the existing commands.

```bash
podcast-agent ingest-book ./examples/book.txt --title "Example Book" --author "Author"
podcast-agent index-book ./examples/book.txt --title "Example Book" --author "Author" --database-url "$DATABASE_URL"
podcast-agent plan-episodes ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2 --database-url "$DATABASE_URL"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2 --database-url "$DATABASE_URL"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --end-chapter "Chapter 3: Turning Point" --episode-count 1
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --start-chapter "Chapter 3: Turning Point" --end-chapter "Chapter 5: Resolution" --episode-count 1
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2 --with-audio
podcast-agent render-audio ./examples/book.txt --title "Example Book" --author "Author" --episode-count 2
```

Example with one of the included sample books:

```bash
podcast-agent run-pipeline ./examples/river_of_hours.txt --title "River of Hours" --author "Sample Author" --episode-count 1
```

Artifacts are written to a per-run subdirectory under `.podcast_agent/runs/<book-title>-<timestamp-to-minute>/`, with nested book and episode folders inside that run. Each run root also includes `run.log`, which records stage transitions, full prompts, responses, TTS requests, and command metadata for that run.

When `--start-chapter` and/or `--end-chapter` is provided, the pipeline selects an inclusive range of detected section titles. `--start-chapter` runs from the matched title to the end of the book, `--end-chapter` runs from the beginning through the matched title, and using both processes the inclusive range between them. Matching is case-insensitive and requires the full detected title.

Episode planning now requires an explicit episode count. With the default settings:

- requested episode count is treated as exact
- planning fails clearly if the request would create an episode longer than the default `240` spoken-minute cap
- planning still preserves contiguous chapter coverage and deterministic beat construction inside each episode

## Outputs and Contracts

Important stage artifacts include:

- `BookIngestionResult`
- `BookStructure`
- `BookAnalysis`
- `SeriesPlan`
- `EpisodeScript`
- `GroundingReport`
- `RepairResult`
- `RenderManifest`
- `AudioManifest`
- `EpisodeOutput`

The contracts are intentionally strict:

- stage outputs are JSON-serializable
- extra fields are rejected
- malformed agent output should fail validation early
- downstream stages should consume validated artifacts rather than free-form text

Each episode directory now contains:

- `episode_output.json`: the single canonical JSON artifact with plan, script, validation, render, repair, and audio metadata
- `<episode-id>.mp3`: one final episode audio file when audio synthesis is enabled

## Development

Run tests with:

```bash
pytest
```

The test suite covers schema validation, chunk persistence and embeddings, multi-chapter episode planning behavior, the OpenAI-compatible client path, and optional audio synthesis artifacts.
