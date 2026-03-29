# Webpage Rules For Collection Tiles

This file is the canonical human-readable guide for homepage tiles, carousel metadata, and cover SVG styling.

## Theme Rules

- Carousel themes are defined per collection as exactly three short phrases.
- Each phrase is 1-2 words.
- Phrases must be geography-specific (country/region context).
- Phrases are joined in carousel with ` / `.
- Canonical values live in `docs/webpage_rules.json`.

## Carousel Rules

- Keep one metadata row under the carousel image:
- Theme triplet on the left.
- Separator dot.
- Episode metadata on the right (`N episodes · Xh Ym`).
- Theme and metadata text size is `0.912rem` (20% smaller than `1.14rem`).
- Theme color uses `--brand-gold`.
- Metadata color is white.
- Description is always on the next line and uses the same font treatment as podcast-page descriptions.

## Layout Rules

- Desktop/tablet page horizontal padding is `2.1875rem`.
- Mobile horizontal padding remains `1.1rem`.
- Max content width is reduced to keep cards visibly smaller on wide screens.

## SVG Rules

- Top-right `AGENTIC PODCASTS` + bars block is scaled by 20%.
- Keep top/right padding fixed by anchoring transform at the original top-right logo reference.
- Canonical transform: `translate(836,72) translate(74,0) scale(1.2) translate(-74,0)`.
- Two-line title covers use:
- First line `y='690'` with `font-size='72'`.
- Second line `y='772'` with `font-size='72'`.
- Author baseline `y='842'` (author size may vary only for long names).

## Authoring Workflow

When adding a new tile or changing cover styling:

1. Update `docs/index.html`.
2. Update corresponding `docs/<slug>.html`.
3. Update canonical rules in `docs/webpage_rules.json` if theme values change.
4. Run:

```bash
python scripts/validate_webpage_rules.py
```

5. Verify no violations before publishing.
