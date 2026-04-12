# Podcast Agent

`podcast-agent` is a multi-book thematic podcast pipeline. It takes N books that share a broad theme, finds intellectual connections between them, and produces a podcast series that synthesizes ideas across all the books — not summarizing each book in turn, but weaving them together.

## Overview

- Library-first Python package with a Typer CLI.
- Four-phase pipeline: Ingest & Index, Thematic Intelligence, Episode Production, Audio Rendering.
- 13 LLM-backed agents with per-agent model selection, temperature, retry, and concurrency.
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
| Synthesis Primitives | Opus 4.6 | 3 | 3 |
| Synthesis Consolidation | Opus 4.6 | 4 | 3 |
| Narrative Strategy | Sonnet 4.6 | 6 | 3 |
| Series Planning | Sonnet 4.6 | 6 | 3 |
| Episode Writing | Opus 4.6 | 6 | 3 |
| Grounding Validation | Sonnet 4.6 | 6 | 3 |
| Repair | Sonnet 4.6 | 6 | 3 |
| Spoken Delivery | Sonnet 4.6 | 6 | 3 |
| Style Audit | Sonnet 4.6 | 8 | 3 |

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
| `--passage-extraction-concurrency` | | `8` | Max concurrent passage extraction axis jobs |
| `--skip-grounding` | | `False` | Skip grounding validation and repair |
| `--skip-spoken-delivery` | | `False` | Skip spoken delivery rewrite |
| `--skip-audio` | | `False` | Skip audio synthesis (still writes render manifest) |
| `--tts-provider` | | settings default | TTS provider for audio synthesis (`openai-compatible`, `kokoro`; `openai` alias accepted) |

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
podcast-agent synthesize-audio ./runs/<project-id> --tts-provider kokoro
```

This command:
- reads `episodes/<N>/render_manifest.json`
- synthesizes per-segment audio into `episodes/<N>/audio/`
- writes a merged `episodes/<N>/episode.mp3`
- updates `episodes/<N>/audio_manifest.json`

`--tts-provider` is required and accepts `openai-compatible` or `kokoro` (`openai` is accepted as an alias of `openai-compatible`).

`ffmpeg` must be installed and available on `PATH` for merged MP3 output.

## Pipeline Phases

### Phase 1: Ingest & Index (parallel per book)
1. **Read source** — PDF or plain text
2. **Structure chapters** — LLM identifies chapter boundaries and chapter summaries
3. **Chunk text** — Overlapping chunks at paragraph/sentence boundaries
4. **Embed & store** — Index chunks in PGVector with book/project metadata

### Phase 2: Thematic Intelligence (sequential)
5. **Decompose theme** — Break theme into 10-15 strong thematic axes using chapter summaries plus synthesized per-book summaries
6. **Extract passages** — Vector retrieval + LLM reranking per axis per book
7. **Synthesis primitives** — Extract grounded turning points, consequences, mechanisms, and live questions
8. **Synthesis consolidation** — Consolidate primitives into cluster-first synthesis artifacts
9. **Choose narrative strategy** — Select series structure and assign discovery-ordered cluster paths
10. **Plan episodes** — Per-episode framing and scene-card planning from cluster paths

### Phase 3: Episode Production (parallel per episode)
11. **Write episode** — Single-batch section-based script drafting with citations
12. **Validate grounding** — Fact-check claims against cited passages (skippable)
13. **Repair loop** — Fix grounding failures up to N attempts (skippable)
14. **Spoken delivery** — Whole-episode spoken cleanup without structural reordering (skippable)
15. **Style audit** — Warnings-only post-delivery audit

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
`passage_retrieval_min_per_book=10`, and `passage_retrieval_max_per_book=25`.
The global pre-axis candidate budget defaults to
`pre_axis_total_budget=1200` with `pre_axis_floor=30`, allocated by normalized
axis theme-importance scores.
Passage extraction no longer applies a post-rerank per-axis trim; it retains all
scored passages for later stage-specific selection.
For synthesis, the active pipeline now restores a hard
`synthesis_total_passage_cap=720` across the whole run, allocated by
axis theme-importance with round-robin fill of remaining slots.

## Outputs

Each run writes artifacts under `runs/<project-id>/`:

```
runs/<project-id>/
  thematic_project.json        # Root project state
  thematic_axes.json           # Decomposed theme axes
  thematic_corpus.json         # Extracted and reranked passages
  retrieval_metrics.json       # Per-axis/per-book retrieval accuracy
  synthesis_primitives.json    # Grounded synthesis primitives
  synthesis_map.json           # Consolidated cluster-first synthesis artifact
  narrative_strategy.json      # Chosen series structure and episode cluster paths
  series_plan.json             # Episode plans with framing and scene cards
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

## Knowledge Wiki

This repo includes a code-focused impact-mapping wiki under `knowledge_wiki/`.

- Entry point: `knowledge_wiki/index.md`
- Rules and operation contract: `knowledge_wiki/AGENTS.md`
- Operation history: `knowledge_wiki/log.md`

Use these operation phrases with your coding agent when updating wiki state:

- `ingest repo` (refresh impacted pages from current codebase state)
- `query wiki` (answer questions using wiki citations)
- `lint wiki` (repair links/orphans/stale references inside `knowledge_wiki/`)

## Requirements

- Python >= 3.11
- PostgreSQL with pgvector extension (optional — for vector retrieval)
- At least one LLM API key (Anthropic or OpenAI-compatible)
