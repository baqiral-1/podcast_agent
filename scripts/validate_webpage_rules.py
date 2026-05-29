#!/usr/bin/env python3
"""Validate homepage and detail-page rules for static docs pages."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INDEX_PATH = DOCS_DIR / "index.html"
RULES_PATH = DOCS_DIR / "webpage_rules.json"


def _word_count(value: str) -> int:
    return len([part for part in value.strip().split() if part])


def _count_word(value: int) -> str:
    words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
    }
    return words.get(value, str(value))


def _extract_scenes(index_html: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'<section class="scene" data-scene>.*?'
        r'<p class="scene-meta reveal d2">([^<]+)</p>.*?'
        r'<a href="./([^"]+)\.html"',
        re.S,
    )
    return [(meta, slug) for meta, slug in pattern.findall(index_html)]


def main() -> int:
    rules = json.loads(RULES_PATH.read_text())
    index_html = INDEX_PATH.read_text()
    errors: list[str] = []

    _delimiter = rules["theme"]["delimiter"]
    exact_count = int(rules["theme"]["exact_count"])
    max_words = int(rules["theme"]["max_words_per_theme"])
    collections: dict[str, dict[str, object]] = rules["collections"]

    scenes = _extract_scenes(index_html)
    if len(scenes) != len(collections):
        errors.append(
            f"homepage scene count mismatch: found {len(scenes)}, expected {len(collections)}"
        )

    seen_slugs: set[str] = set()
    for meta_text, slug in scenes:
        seen_slugs.add(slug)
        if slug not in collections:
            errors.append(f"unexpected homepage slug: {slug}")
            continue

        collection = collections[slug]
        themes = [str(part).strip() for part in collection["themes"]]
        if len(themes) != exact_count:
            errors.append(
                f"theme count mismatch for {slug}: found {len(themes)}, expected {exact_count}"
            )
        for item in themes:
            if _word_count(item) > max_words:
                errors.append(
                    f"theme word limit exceeded for {slug}: '{item}' has more than {max_words} words"
                )

        expected_count = int(collection["episode_count"])
        if (
            f"{expected_count} episodes" not in meta_text
            and f"{_count_word(expected_count)} episodes" not in meta_text
        ):
            errors.append(f"metadata missing episode count for {slug}: '{meta_text}'")
        runtime = str(collection["runtime"])
        if runtime not in meta_text:
            errors.append(f"metadata missing runtime for {slug}: '{meta_text}'")

    missing_slugs = sorted(set(collections) - seen_slugs)
    for slug in missing_slugs:
        errors.append(f"missing homepage scene for slug: {slug}")

    required_snippets = [
        'class="scene-grid"',
        'class="scene-meta reveal d2"',
        'class="scene-link reveal d3"',
        "IntersectionObserver",
    ]
    for snippet in required_snippets:
        if snippet not in index_html:
            errors.append(f"index missing required snippet: {snippet}")

    for slug, collection in collections.items():
        path = DOCS_DIR / f"{slug}.html"
        if not path.exists():
            errors.append(f"missing detail page for slug: {slug}")
            continue
        content = path.read_text()
        for snippet in (
            'class="split-container"',
            'id="waveformCanvas"',
            "const EPISODES =",
            "function loadEpisode",
        ):
            if snippet not in content:
                errors.append(f"{path.name}: missing detail-page snippet {snippet}")
        accent = str(collection["accent_color"])
        if f"--accent: {accent};" not in content:
            errors.append(f"{path.name}: accent mismatch, expected {accent}")
        expected_count = int(collection["episode_count"])
        match = re.search(r"const EPISODES = \[(.*?)\n\];", content, re.S)
        if not match:
            errors.append(f"{path.name}: could not parse EPISODES array")
        elif len(re.findall(r'"num":', match.group(1))) != expected_count:
            errors.append(f"{path.name}: episode count mismatch in EPISODES")

        for image_path in re.findall(r'"image": "([^"]+)"', content):
            if not (DOCS_DIR / image_path).exists():
                errors.append(f"{path.name}: missing image asset {image_path}")

    if errors:
        print("Webpage rules validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("Webpage rules validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
