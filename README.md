# Podcast Agent

`podcast-agent` is a multi-book thematic podcast pipeline. It takes N books that share a broad theme, finds intellectual connections between them, and produces a podcast series that synthesizes ideas across all the books — not summarizing each book in turn, but weaving them together.

## Overview

- Library-first Python package with a Typer CLI.
- Four-phase pipeline: Ingest & Index, Thematic Intelligence, Episode Production, Audio Rendering.
- 12 LLM-backed agents with per-agent model selection, temperature, retry, and concurrency.
- Cross-book synthesis: finds agreements, disagreements, extensions, tensions, and surprising connections.
- Grounding validation with citation-level fact-checking and fairness flags.
- Spoken-delivery rewrite for natural narration without changing facts.
- PostgreSQL/PGVector-backed retrieval with per-book metadata filtering.
- Structured JSON artifacts and run logging at every stage for debugging.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

```bash
# LLM (required — at least one provider)
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# Optional overrides
export LLM_PROVIDER=anthropic           # or openai-compatible, heuristic
export LLM_MODEL_NAME=claude-opus-4-6   # global default (agents override per-stage)

# PostgreSQL for vector retrieval (optional — pipeline works without it)
export DATABASE_URL=postgresql://postgres:secret@localhost:5432/podcast_agent

# Embeddings
export EMBEDDINGS_PROVIDER=openai
export EMBEDDINGS_MODEL_NAME=text-embedding-3-small

# Cache
export LANGCHAIN_CACHE_BACKEND=sqlite   # sqlite, memory, redis, or none
```

Each agent uses a default Claude model chosen for its task complexity:

| Agent | Model | Concurrency | Retries |
|-------|-------|-------------|---------|
| Structuring | Haiku 4.5 | 15 | 5 |
| Theme Decomposition | Opus 4.6 | 6 | 3 |
| Passage Extraction | Haiku 4.5 | 15 | 5 |
| Synthesis Mapping | Opus 4.6 | 3 | 3 |
| Narrative Strategy | Sonnet 4.6 | 6 | 3 |
| Series Planning | Sonnet 4.6 | 6 | 3 |
| Episode Writing | Opus 4.6 | 3 | 3 |
| Source Weaving | Sonnet 4.6 | 6 | 3 |
| Grounding Validation | Sonnet 4.6 | 6 | 3 |
| Repair | Sonnet 4.6 | 6 | 3 |
| Spoken Delivery | Sonnet 4.6 | 6 | 3 |
| Episode Framing | Haiku 4.5 | 15 | 5 |

Override any agent's model, temperature, retry count, or concurrency limit via `LLMConfig.agent_configs` in code or environment variables.

## CLI

### `podcast-agent run`

Run the full multi-book thematic podcast pipeline.

```bash
podcast-agent run book1.pdf book2.txt book3.md \
  --theme "the psychology of decision-making" \
  --sub-themes "risk,uncertainty,group dynamics" \
  --episodes 4 \
  --titles "Thinking Fast,Nudge,Predictably Irrational" \
  --authors "Kahneman,Thaler & Sunstein,Ariely"
```

| Argument/Option | Short | Default | Description |
|---|---|---|---|
| `SOURCES` | | required | Paths to book files (PDF, TXT, MD) |
| `--theme` | `-t` | required | Theme to explore across books |
| `--episodes` | `-n` | inferred | Override episode count (otherwise inferred from narrative strategy) |
| `--elaboration` | | | Optional longer theme description |
| `--sub-themes` | | | Optional comma-separated sub-themes (max 15, trimmed and deduped) |
| `--titles` | | | Comma-separated book titles |
| `--authors` | | | Comma-separated author names |
| `--output-dir` | `-o` | `runs/` | Custom output directory |
| `--skip-grounding` | | `False` | Skip grounding validation and repair |
| `--skip-spoken-delivery` | | `False` | Skip spoken delivery rewrite |
| `--skip-audio` | | `False` | Skip audio synthesis (still writes render manifest) |

### `podcast-agent status`

Check the status of a pipeline run.

```bash
podcast-agent status <project-id>
```

| Argument/Option | Short | Default | Description |
|---|---|---|---|
| `PROJECT_ID` | | required | Project ID to check |
| `--output-dir` | `-o` | | Output directory |

### `podcast-agent synthesize-audio`

Synthesize audio from existing `render_manifest.json` artifacts in a completed run directory.

```bash
podcast-agent synthesize-audio ./runs/<project-id>
```

This command:
- reads `episodes/<N>/render_manifest.json`
- synthesizes per-segment audio into `episodes/<N>/audio/`
- writes a merged `episodes/<N>/episode.mp3`
- updates `episodes/<N>/audio_manifest.json`

`ffmpeg` must be installed and available on `PATH` for merged MP3 output.

## Pipeline Phases

### Phase 1: Ingest & Index (parallel per book)
1. **Read source** — PDF or plain text
2. **Structure chapters** — LLM identifies chapter boundaries and chapter summaries
3. **Chunk text** — Overlapping chunks at paragraph/sentence boundaries
4. **Embed & store** — Index chunks in PGVector with book/project metadata

### Phase 2: Thematic Intelligence (sequential)
5. **Decompose theme** — Break theme into 5-7 strong thematic axes using chapter summaries plus synthesized per-book summaries
6. **Extract passages** — Vector retrieval + LLM reranking per axis per book
7. **Map synthesis** — Discover cross-book insights (agreements, disagreements, tensions, extensions)
8. **Choose narrative strategy** — Select series structure (thesis-driven, debate, convergence, etc.)
9. **Assign episodes** — Narrative strategy assigns axes and insights per episode
10. **Plan episodes** — Per-episode structure planning with beats and passage assignments

### Phase 3: Episode Production (parallel per episode)
11. **Write episode** — Script with citations and cross-book transitions
12. **Validate grounding** — Fact-check claims against cited passages (skippable)
13. **Repair loop** — Fix grounding failures up to N attempts (skippable)
14. **Spoken delivery** — Rewrite for natural speech without changing facts (skippable)
15. **Frame episode** — Recaps, previews, and cold opens (sequential)

### Phase 4: Audio Rendering (parallel per episode)
16. **Build render manifest** — TTS-ready segment specification
17. **Synthesize audio** — TTS with retry and concurrency control

Chapter summaries are currently consumed only for theme decomposition, including a
theme-conditioned per-book summary synthesized from those chapter summaries.
Downstream stages rely on retrieved passages and synthesis artifacts rather than
chapter summaries.

Passage extraction now budgets retrieval candidates per book with a hybrid rule:
`min(max_cap, max(min_floor, round(book_chunk_count * percentage)))`.
The default knobs are `passage_retrieval_percentage=0.25`,
`passage_retrieval_min_per_book=20`, and `passage_retrieval_max_per_book=60`.
`rerank_top_k` still applies after passage-extraction scoring; it does not control
how many passages are sent into the passage-extraction prompt.

## Outputs

Each run writes artifacts under `runs/<project-id>/`:

```
runs/<project-id>/
  thematic_project.json        # Root project state
  thematic_axes.json           # Decomposed theme axes
  thematic_corpus.json         # Extracted and reranked passages
  retrieval_metrics.json       # Per-axis/per-book retrieval accuracy
  synthesis_map.json           # Cross-book insights and narrative threads
  narrative_strategy.json      # Chosen series structure
  series_plan.json             # Episode plans with beats
  run.log                      # Structured JSON event log
  stage_artifacts/             # Per-stage input/output snapshots
  books/<book-id>/
    raw_text.txt
    book_record.json
  episodes/<N>/
    episode_script.json
    grounding_report.json
    repair_attempt_*.json
    spoken_script.json
    episode_framing.json
    render_manifest.json
    audio_manifest.json         # only when audio synthesis runs
    episode.mp3                 # merged episode audio when synthesis runs
    audio/                      # only when audio synthesis runs
```

The `run.log` contains structured JSON events for every stage start/end, LLM request/response (with token counts and timing), retries, errors, retrieval metrics, and skip decisions.

## Development

```bash
pytest                    # Run all tests (141 tests)
pytest tests/ -x --tb=short  # Stop on first failure
```

## Requirements

- Python >= 3.11
- PostgreSQL with pgvector extension (optional — for vector retrieval)
- At least one LLM API key (Anthropic or OpenAI-compatible)
