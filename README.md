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
- Episodes target at least 30 minutes of spoken narration when the source material is sufficient.
- Episodes are only merged away when they cannot be kept above a 10-minute standalone minimum after rebalancing.
- Structuring is incremental and runs up to 3 chapter-level calls in parallel, with smaller fallbacks only when a chapter is too large.

## Architecture

The pipeline is organized as typed stage artifacts and deterministic orchestration around agent boundaries.

![Agent execution flow](docs/agent-flow.svg)

1. `ingest-book`
   Reads a source file, normalizes source metadata, and creates a canonical `BookIngestionResult`.
   This is a deterministic pipeline step, not an LLM-driven agent.

2. `structuring_agent`
   - Produces `BookStructure` with chapters, chunks, stable IDs, and source offsets.
   - Uses deterministic preprocessing for chapter splitting and chunking, then routes through the shared LLM interface chapter by chapter.
   - Feeds `index-book`, where chunks are stored in PostgreSQL and embeddings are generated.

3. `index-book`
   Stores chunk rows in PostgreSQL immediately after structuring.
   Computes embeddings for those chunks and stores them in `pgvector`.
   This is also a deterministic orchestration step rather than an agent.

4. `analysis_agent`
   - Identifies themes, continuity arcs, notable claims, and candidate episode groupings from the structured book.
   - Produces coverage-preserving multi-chapter clusters that collectively account for the full book.
   - Retries once and fails fast if the analysis omits chapters, under-assigns chunks, or produces invalid spans.

5. `episode_planning_agent`
   - Converts `BookAnalysis` into a hierarchical `SeriesPlan` with episodes, beats, and claim requirements.
   - Defines episodes independently from chapter boundaries so they can span multiple chapters.
   - Strongly targets at least 30 minutes of spoken runtime per episode, then deterministically rebalances adjacent chapter boundaries before dropping any episode.
   - Keeps short-but-viable episodes when they remain above a 10-minute standalone minimum and retries once if the live plan ignores multi-chapter or coverage constraints.

6. `writing_agent`
   - Generates the single-narrator script for each episode from the plan plus retrieved source chunks.
   - Builds segment-level narration, citations, and claim records for each segment.
   - Emits an `EpisodeScript` artifact through the LLM interface using retrieval-backed evidence and enough source text to satisfy the runtime target when available.

7. `grounding_validation_agent`
   - Validates each claim-bearing script segment against cited source chunks before rendering.
   - Classifies claims as grounded, weak, unsupported, or conflicting and emits a `GroundingReport`.
   - Acts as the LLM-backed validation gate between writing and render-manifest generation.

8. `repair_agent`
   - Rewrites only weak or unsupported parts of the script instead of regenerating the full episode.
   - Uses the prior script and `GroundingReport` to target failures precisely.
   - Emits a `RepairResult` and is followed by revalidation before final manifest generation.

9. `render-manifest`
   Emits a TTS-ready manifest with grounded segments and SSML-ready text.
   This is deterministic in the current implementation and is the gate into audio rendering.

10. `synthesize-audio`
   - Uses the configured TTS client to synthesize validated render segments and combines them into one episode audio file.
   - Persists audio metadata inside the canonical episode output plus one final audio file on disk.
   - Runs only when the CLI requests audio output or `run_pipeline(..., synthesize_audio=True)` is used.

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

The default speech model is `gpt-4o-mini-tts`, the default voice is `alloy`, and audio is written as `mp3` unless `Settings.tts` is overridden in code.

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

```bash
podcast-agent ingest-book ./examples/book.txt --title "Example Book" --author "Author"
podcast-agent index-book ./examples/book.txt --title "Example Book" --author "Author" --database-url "$DATABASE_URL"
podcast-agent plan-episodes ./examples/book.txt --title "Example Book" --author "Author" --database-url "$DATABASE_URL"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --database-url "$DATABASE_URL"
podcast-agent run-pipeline ./examples/book.txt --title "Example Book" --author "Author" --with-audio
podcast-agent render-audio ./examples/book.txt --title "Example Book" --author "Author"
```

Example with one of the included sample books:

```bash
podcast-agent run-pipeline ./examples/river_of_hours.txt --title "River of Hours" --author "Sample Author"
```

Artifacts are written to a per-run subdirectory under `.podcast_agent/runs/<book-title>-<timestamp-to-minute>/`, with nested book and episode folders inside that run. Each run root also includes `run.log`, which records stage transitions, full prompts, responses, TTS requests, and command metadata for that run.

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
