# AGENTS.md

## Purpose

This repository contains a production-grade Python service.
Agents working in this repo must prioritize correctness, safety, maintainability, testability, and minimal diffs.

## Core working rules

- Make the smallest correct change that fully solves the task.
- Do not rewrite large sections of code unless necessary.
- Preserve existing architecture, naming conventions, and public interfaces unless the task explicitly asks for a change.
- Prefer boring, well-understood solutions over clever ones.
- Optimize for readability and operability in production.
- When requirements are ambiguous, infer the most conservative production-safe behavior from surrounding code and tests.
- Do not introduce breaking changes unless explicitly requested.
- Never fabricate APIs, modules, environment variables, or config fields. Verify from the codebase first.
- Never claim code was run, tested, or validated unless it actually was.

## Repository assumptions

Unless the repo clearly does otherwise, assume the following:

- Python version: 3.11+
- Dependency management: prefer the project’s existing tool (`poetry`, `uv`, `pip-tools`, `requirements.txt`, or `pdm`)
- Test framework: `pytest`
- Linting/formatting: `ruff` and/or `black`
- Type checking: `mypy` or `pyright`
- Packaging/config: `pyproject.toml`
- Service style: modular application code under `src/` or package directory, tests under `tests/`

Always inspect the repo and follow what is already configured.

## Engineering standards

### Code quality

- Write explicit, readable Python.
- Favor pure functions and narrow interfaces where practical.
- Prefer composition over unnecessary inheritance.
- Avoid hidden side effects.
- Keep functions focused and reasonably short.
- Add comments only when they explain *why*, not *what*.
- Use dataclasses, typed dicts, enums, and pydantic models only when they fit existing project patterns.

### Typing

- Add or preserve type hints for all new or modified Python code.
- Do not weaken types to `Any` unless there is no practical alternative.
- Prefer precise types over overly broad ones.
- Keep type checker noise low.

### Errors and logging

- Fail loudly for programmer errors; handle operational errors gracefully.
- Do not swallow exceptions silently.
- Preserve existing exception hierarchy and error-handling style.
- Add actionable error messages.
- Use structured logging if the codebase supports it.
- Never log secrets, tokens, API keys, credentials, or sensitive personal data.

### Dependencies

- Avoid adding new dependencies unless necessary.
- Prefer the standard library or already-installed project dependencies.
- If a new dependency is required, choose one that is mature, minimal, and aligned with the repo’s existing practices.
- Document why a new dependency is needed in the final summary.

## Project structure expectations

When adding new code:

- Put production code in the existing application/package structure.
- Put tests in `tests/` mirroring the source layout where possible.
- Keep scripts out of the main package unless they are part of the product.
- Keep one-off/debug code out of committed changes.
- Keep configuration centralized and consistent with existing patterns.

## Testing rules

- Add or update tests for every behavioral change.
- Prefer unit tests unless integration tests are clearly more appropriate.
- Test the public behavior, not implementation details.
- Cover happy path, edge cases, and relevant failure modes.
- Do not delete failing tests unless the task explicitly requires removing obsolete coverage.
- If tests are flaky or broken for unrelated reasons, mention that clearly instead of masking the issue.

## Validation workflow

Before finishing, do as many of these as are available in the repo:

1. Format changed files.
2. Run lint checks on changed files or the smallest meaningful scope.
3. Run type checks on the smallest meaningful scope.
4. Run the most relevant tests first, then broader tests if practical.

Preferred command discovery order:

1. Commands documented in `README.md`, `CONTRIBUTING.md`, or developer docs
2. Scripts defined in `pyproject.toml`, `Makefile`, `justfile`, or CI config
3. Existing patterns from recent repo changes

If you cannot run validation, say exactly why.

## Change-size discipline

- Keep diffs minimal.
- Avoid opportunistic refactors unless they are required for correctness.
- If a refactor is necessary, isolate it and explain why.
- Preserve backward compatibility where possible.
- Avoid renaming files, functions, classes, or modules unless needed.

## Security and production safety

- Treat this repository as production-sensitive.
- Never commit secrets or sample credentials.
- Never hardcode tokens, passwords, private URLs, or internal keys.
- Sanitize inputs at boundaries when relevant.
- Preserve authn/authz behavior unless the task explicitly changes it.
- Be careful with migrations, deletes, concurrency, retries, caching, and data loss risks.
- For database changes, prefer reversible and backward-compatible migrations unless explicitly instructed otherwise.
- For background jobs, queues, and async workflows, preserve idempotency guarantees.

## API and schema changes

When touching APIs, models, events, or schemas:

- Preserve backward compatibility unless explicitly told otherwise.
- Update validators, serializers, and tests together.
- Call out contract changes explicitly.
- If changing request/response shapes, search for downstream usage and update impacted tests/docs.

## Database changes

If the repo includes database models or migrations:

- Prefer additive changes first.
- Avoid destructive schema changes unless explicitly requested.
- Include indexes only when justified.
- Consider rollback and deploy safety.
- Update ORM models, migrations, and tests consistently.

## Performance guidance

- Do not optimize prematurely.
- Preserve or improve asymptotic behavior in hot paths.
- Be mindful of N+1 queries, repeated I/O, excessive allocations, and blocking calls.
- For performance-sensitive code, explain tradeoffs briefly in the final summary.

## Documentation updates

Update docs when relevant, especially for:

- New commands
- New environment variables
- Changed API behavior
- New migrations
- Operational behavior changes
- Developer workflow changes

## What to inspect first

Before editing, inspect in this order when relevant:

1. `README.md`
2. `CONTRIBUTING.md`
3. `pyproject.toml`
4. `Makefile` / `justfile`
5. CI configuration
6. Existing tests near the affected code
7. Similar nearby modules for local patterns

## Decision heuristics

When several valid implementations exist, prefer the one that:

1. Matches existing repo conventions
2. Minimizes risk
3. Minimizes diff size
4. Is easiest to test
5. Is easiest for another engineer to understand at 2 AM

## Final response format

When you complete a task, provide:

### Summary
- What changed
- Why it changed

### Validation
- What you ran
- What passed
- What could not be run

### Risks / follow-ups
- Any edge cases, migration concerns, or operational risks
- Any recommended next steps

## Anti-patterns to avoid

- Massive refactors unrelated to the task
- Fake completion claims
- Ignoring failing tests
- Adding duplicate abstractions
- Introducing silent fallbacks for real errors
- Overusing `Any`, global state, or hidden mutation
- Catch-all exception handling without re-raising or logging
- Adding dependencies for trivial problems
- Editing unrelated files “while here”

## Agent-specific instructions for Python

### When writing new modules

- Include module-level imports grouped and sorted consistently with repo style.
- Prefer absolute imports if the repo uses them; otherwise follow local style.
- Keep side effects out of import time unless the codebase explicitly relies on them.

### When writing tests

- Use `pytest` style unless the repo uses something else.
- Prefer fixtures for reusable setup.
- Avoid brittle sleeps, network dependence, or real external services unless the repo already uses them in integration tests.
- Mock at stable boundaries, not deep internals.

### When touching async code

- Preserve async/sync API contracts.
- Do not block the event loop with synchronous I/O unless already accepted in that code path.
- Be careful with cancellation, retries, timeouts, and resource cleanup.

### When touching CLI code

- Preserve backward-compatible flags and exit codes unless explicitly asked otherwise.
- Update help text and docs for CLI changes.

## If the task is underspecified

Do not stop early.
Inspect surrounding code, tests, and docs, then make the safest reasonable implementation.
Document assumptions in the final response.

## Commit Messages

### Required Format

- Write a short description line that summarizes the full change set.
- Use this structure:
  Description
  Intent:
    - 2 bullet points summarizing intent (3 maximum only if necessary). Keep each bullet under 20 words.
    - ...

  Changes:
    - 3 bullet points summarizing changes (5 maximum only if necessary). Keep each bullet under 25 words.

### Content Rules

- Infer the intent from the actual diff; do not use generic summaries that are not supported by the changes.
- Prefer describing why the change exists and what it changes over listing filenames.
- Keep bullets specific to concrete behavioural changes.
- Ignore changes to the README or any files under docs unless they're the only changes in the commit.
- There should be no newlines between bullets.

### Ordering Rules

- In `Changes`, list functional or user-visible behaviour changes first.
- Place supporting updates such as logging, tests, README/docs, or guidance changes after the functional bullets.

### Amend Rules

- When amending a commit, rewrite both the description and the message so they cover the full amended diff across all files in the commit. Do not anchor it in the existing commit message at all.
- Do not reuse the previous message unchanged if the amended diff adds or changes the commit scope.
