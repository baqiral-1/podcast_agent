# AGENTS.md

## Python Guidelines

- Follow PEP 8 for formatting, whitespace, imports, and naming.
- Use `snake_case` for functions, methods, variables, and module names.
- Use `PascalCase` for classes and Pydantic models.
- Add type hints to public functions, methods, and class attributes where practical.
- Keep modules focused; prefer small, composable classes over large mixed-responsibility files.
- Write concise docstrings for public modules, classes, and functions. Follow PEP 257 style.
- Prefer explicit data contracts with Pydantic models over untyped dictionaries at stage boundaries.
- Keep agent outputs JSON-serializable and schema-validated before passing them to the next stage.
- Do not bypass grounding validation before render-manifest generation.
- Prefer deterministic helpers for parsing, chunking, and repository logic; reserve LLM inference for judgment-heavy work.
- Update `README.md` whenever agent structure, agent responsibilities, default model behavior, or stage boundaries change.
- Update `README.md` whenever audio/TTS stages, output artifacts, CLI commands, or runtime configuration change.
- Keep README agent descriptions concise: prefer 2-3 bullets per agent unless extra detail is required to prevent confusion.

## Testing Expectations

- Add or update tests for new schemas, pipeline steps, and repository behavior.
- Cover malformed agent output and schema validation failures when changing agent contracts.
- Validate episode grouping behavior when changing planning heuristics.
- Keep integration fixtures small and readable.

## Commit Messages

- Write a short description line summarizing the changes. This should be short yet descriptive enough to explain the intent behind the commit and the changes.
- Use the following commit structure:
  Description
  Intent:
    - 2-3 bullet points summarizing intent
    - ...
  
  Changes:
    - 3-5 bullet points summarizing changes.
- Infer the intent from the actual diff; do not use generic summaries that are not supported by the changes.
- Keep bullets specific to concrete behavioural changes.
- Prefer describing why the change exists and what it changes over listing filenames.
