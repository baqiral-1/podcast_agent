# Webpage Rules For Static Podcast Pages

This file is the canonical human-readable guide for the generated homepage scenes,
detail pages, and collection metadata.

## Theme Rules

- Collection themes are defined as exactly three short phrases.
- Each phrase is 1-2 words.
- Themes should be geography-specific when practical.
- Canonical values live in `docs/webpage_rules.json`.

## Homepage Rules

- `docs/index.html` uses one full-screen vertical `.scene` per collection.
- Each scene includes a `.scene-grid` with six raster images.
- Each scene metadata line includes episode count, book count, and runtime.
- Series with unavailable audio should still link to their detail page and say
  `Audio coming soon`.
- Keep the existing dark background, low-opacity image grid, reveal animation,
  and accent-colored link treatment.

## Detail Page Rules

- Detail pages use the split-panel waveform player layout.
- Episode data lives in the page-local `EPISODES` array.
- Playable pages must use GitHub Release URLs for audio assets.
- Pages without audio must use empty `audio` values and disable playback controls.
- Accent colors, episode counts, runtimes, and themes must match
  `docs/webpage_rules.json` and `docs/manifest.json`.

## Authoring Workflow

When adding a new series or changing collection styling:

1. Update `docs/index.html`.
2. Update or create `docs/<slug>.html`.
3. Update `docs/manifest.json`.
4. Update `docs/webpage_rules.json`.
5. Run:

```bash
python3 scripts/validate_webpage_rules.py
```

6. Verify no violations before publishing.
