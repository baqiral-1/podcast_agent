#!/usr/bin/env python3
"""Validate homepage and cover rules for static docs pages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INDEX_PATH = DOCS_DIR / "index.html"
RULES_PATH = DOCS_DIR / "webpage_rules.json"


def _word_count(value: str) -> int:
    return len([part for part in value.strip().split() if part])


def _extract_slides(index_html: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r'<div class="hero-carousel-slide(?: active)?"\s+data-theme="([^"]+)"\s+data-meta="([^"]+)">\s*<a href="./([^"]+)\.html"',
        re.S,
    )
    return [(theme, meta, slug) for theme, meta, slug in pattern.findall(index_html)]


def main() -> int:
    rules = json.loads(RULES_PATH.read_text())
    index_html = INDEX_PATH.read_text()
    errors: list[str] = []

    delimiter = rules["theme"]["delimiter"]
    exact_count = int(rules["theme"]["exact_count"])
    max_words = int(rules["theme"]["max_words_per_theme"])
    expected_themes: dict[str, list[str]] = rules["carousel_themes_by_slug"]

    slides = _extract_slides(index_html)
    if len(slides) != len(expected_themes):
        errors.append(
            f"carousel slide count mismatch: found {len(slides)}, expected {len(expected_themes)}"
        )

    seen_slugs: set[str] = set()
    for theme_text, meta_text, slug in slides:
        seen_slugs.add(slug)
        if slug not in expected_themes:
            errors.append(f"unexpected carousel slug: {slug}")
            continue

        expected_text = delimiter.join(expected_themes[slug])
        if theme_text != expected_text:
            errors.append(
                f"theme mismatch for {slug}: found '{theme_text}', expected '{expected_text}'"
            )

        themes = [part.strip() for part in theme_text.split(delimiter)]
        if len(themes) != exact_count:
            errors.append(
                f"theme count mismatch for {slug}: found {len(themes)}, expected {exact_count}"
            )
        for item in themes:
            if _word_count(item) > max_words:
                errors.append(
                    f"theme word limit exceeded for {slug}: '{item}' has more than {max_words} words"
                )

        if "episodes" not in meta_text:
            errors.append(f"metadata missing episode count for {slug}: '{meta_text}'")

    missing_slugs = sorted(set(expected_themes) - seen_slugs)
    for slug in missing_slugs:
        errors.append(f"missing carousel slide for slug: {slug}")

    required_snippets = [
        "class=\"carousel-caption-meta-row\"",
        "class=\"carousel-caption-theme\"",
        "class=\"carousel-caption-meta\"",
        "class=\"carousel-caption-summary\"",
        "font-size: 0.912rem;",
    ]
    for snippet in required_snippets:
        if snippet not in index_html:
            errors.append(f"index missing required snippet: {snippet}")

    old_transform = "transform='translate(836,72)' opacity='0.72'"
    new_transform = rules["svg"]["brand_transform"]
    for html_path in DOCS_DIR.glob("*.html"):
        content = html_path.read_text()
        if old_transform in content:
            errors.append(f"{html_path.name}: found unscaled logo transform")
        if new_transform not in content:
            errors.append(f"{html_path.name}: missing scaled logo transform")
        if "y='788'" in content:
            errors.append(f"{html_path.name}: found legacy two-line y='788'")
        if "y='848'" in content:
            errors.append(f"{html_path.name}: found legacy author baseline y='848'")
        if re.search(r"y='772'[^>]*font-size='86'", content):
            errors.append(f"{html_path.name}: found legacy two-line font-size='86'")

    for slug in rules["two_line_title_slugs"]:
        path = DOCS_DIR / f"{slug}.html"
        if not path.exists():
            errors.append(f"missing detail page for two-line title slug: {slug}")
            continue
        content = path.read_text()
        for snippet in ("y='690'", "y='772'", "font-size='72'", "y='842'"):
            if snippet not in content:
                errors.append(f"{path.name}: missing two-line style snippet {snippet}")

    if errors:
        print("Webpage rules validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("Webpage rules validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
