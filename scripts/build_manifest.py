#!/usr/bin/env python3
"""
Generate docs/manifest.json from the static HTML in docs/.

Usage:
  python scripts/build_manifest.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
SITE_BASE = "https://baqiral-1.github.io/podcast_agent"


def _clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return html.unescape(value.strip())


def _grab(part: str, cls: str) -> str:
    match = re.search(rf"<[^>]*class=\"{cls}\"[^>]*>(.*?)</", part, re.S)
    return _clean_text(match.group(1)) if match else ""


def _grab_page(page: str, cls: str, tag: str) -> str:
    match = re.search(rf"<[^>]*class=\"{cls}\"[^>]*>(.*?)</{tag}", page, re.S)
    return _clean_text(match.group(1)) if match else ""


def parse_index() -> list[dict]:
    index_path = DOCS_DIR / "index.html"
    parts = index_path.read_text().split('<article class="collection-card"')
    collections = []
    for part in parts[1:]:
        style_match = re.search(r"style=\"([^\"]+)\"", part)
        style = style_match.group(1) if style_match else ""
        accent_match = re.search(r"--accent:\s*([^;]+)", style)
        accent = accent_match.group(1).strip() if accent_match else None

        href_match = re.search(r"href=\"([^\"]+)\"", part)
        href = href_match.group(1) if href_match else ""
        slug = href.replace("./", "").replace(".html", "")

        tagline_match = re.search(r"collection-topline[\s\S]*?<span>(.*?)</span>", part)
        tagline = _clean_text(tagline_match.group(1)) if tagline_match else ""

        count_match = re.search(r"<div class=\"meta-row\">\s*<span>(\d+) episodes</span>", part)
        episode_count = int(count_match.group(1)) if count_match else None

        collections.append(
            {
                "id": slug,
                "title": _grab(part, "collection-title"),
                "author": _grab(part, "collection-author"),
                "tagline": tagline,
                "runtime": _grab(part, "runtime-pill"),
                "summary": _grab(part, "collection-summary"),
                "accentColor": accent,
                "episodeCount": episode_count,
                "detailUrl": f"{SITE_BASE}/{slug}.html",
                "episodes": [],
            }
        )
    return collections


def parse_collection_page(collection: dict) -> None:
    slug = collection["id"]
    page_path = DOCS_DIR / f"{slug}.html"
    if not page_path.exists():
        return
    page = page_path.read_text()

    show_title = _grab_page(page, "show-title", "h1")
    show_desc = _grab_page(page, "show-description", "p")
    if show_title:
        collection["title"] = show_title
    if show_desc:
        collection["summary"] = show_desc

    episodes = []
    for ep_part in page.split('<article class="episode-card">')[1:]:
        number_match = re.search(
            r"<p class=\"episode-meta\">\s*Episode\s*(\d+)\s*</p>",
            ep_part,
        )
        title_match = re.search(r"<h2 class=\"episode-title\">(.*?)</h2>", ep_part, re.S)
        preview_match = re.search(
            r"<p class=\"episode-description-preview\">(.*?)</p>",
            ep_part,
            re.S,
        )
        desc_match = re.search(
            r"<span class=\"episode-description-more\"[^>]*>(.*?)</span>",
            ep_part,
            re.S,
        )
        audio_match = re.search(r"<source src=\"([^\"]+)\"", ep_part)

        episodes.append(
            {
                "number": int(number_match.group(1)) if number_match else None,
                "title": _clean_text(title_match.group(1)) if title_match else "",
                "summary": _clean_text(preview_match.group(1)) if preview_match else "",
                "description": _clean_text(desc_match.group(1)) if desc_match else "",
                "audioUrl": audio_match.group(1) if audio_match else "",
            }
        )

    collection["episodes"] = episodes


def build_manifest() -> dict:
    collections = parse_index()
    for collection in collections:
        parse_collection_page(collection)
    return {
        "version": 1,
        "generatedFrom": "docs/*.html",
        "collections": collections,
    }


def main() -> None:
    manifest = build_manifest()
    output_path = DOCS_DIR / "manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
