from __future__ import annotations

import re
from pathlib import Path


BOOK_PATH = Path("sample_books/temp_clean/directorate_s_steve_coll.cleaned.txt")


def _split_chapters(text: str) -> list[str]:
    parts = re.split(r"^Chapter\s+\d+\s*$", text, flags=re.M)
    return [part.strip() for part in parts if part.strip()]


def _render_chapters(chapters: list[str]) -> str:
    return "\n\n".join(
        f"Chapter {index}\n\n{chapter.strip()}" for index, chapter in enumerate(chapters, start=1)
    ).strip() + "\n"


def _shift_embedded_heading(
    chapters: list[str],
    source_index: int,
    target_index: int,
    marker: str,
    replacement_prefix: str,
) -> None:
    source = chapters[source_index]
    if marker not in source:
        if chapters[target_index].startswith(replacement_prefix):
            return
        raise ValueError(f"marker not found in Chapter {source_index + 1}: {marker!r}")
    before, after = source.split(marker, 1)
    chapters[source_index] = before.strip()
    moved = f"{replacement_prefix}{after}".strip()
    chapters[target_index] = f"{moved}\n\n{chapters[target_index].strip()}".strip()


def _move_tail_to_next_chapter(
    chapters: list[str],
    source_index: int,
    target_index: int,
    marker: str,
) -> None:
    source = chapters[source_index]
    if marker not in source:
        if chapters[target_index].startswith(marker):
            return
        raise ValueError(f"tail marker not found in Chapter {source_index + 1}: {marker!r}")
    before, after = source.split(marker, 1)
    chapters[source_index] = before.strip()
    chapters[target_index] = f"{marker}{after.strip()}\n\n{chapters[target_index].strip()}".strip()


def main() -> None:
    text = BOOK_PATH.read_text()
    chapters = _split_chapters(text)
    if len(chapters) == 34:
        BOOK_PATH.write_text(_render_chapters(chapters))
        print(BOOK_PATH.name)
        return
    if len(chapters) != 35:
        raise ValueError(f"expected 35 or 34 chapters, found {len(chapters)}")

    if len(chapters[23].split()) < 1000:
        _move_tail_to_next_chapter(
            chapters,
            source_index=22,
            target_index=23,
            marker="Tayeb Agha was a relatively young man,",
        )

    _shift_embedded_heading(
        chapters,
        source_index=23,
        target_index=24,
        marker="FTWENTY -FIVE Kayani 2. aisal Shahzad,",
        replacement_prefix="Faisal Shahzad,",
    )
    _shift_embedded_heading(
        chapters,
        source_index=25,
        target_index=26,
        marker="OTWENTY -SEVEN Kayani 3. n Monday, September 13, 2010,",
        replacement_prefix="On Monday, September 13, 2010,",
    )

    replacements = {
        "Tayeb Agha was a relatively young man,believed": "Tayeb Agha was a relatively young man, believed",
        'Kayani\'s"2.0" white paper': 'Kayani\'s "2.0" white paper',
        'With that "double tap"land mine strike': 'With that "double tap" land mine strike',
        '"Kayani 2.."': '"Kayani 2.0"',
        "Kayani 2. paper": "Kayani 2.0 paper",
        "t sixty-nine,\n\nAt sixty-nine,": "At sixty-nine,",
        '"It is representative of the ideal."31 Division': '"It is representative of the ideal."',
        "RRahmatullah Nabil": "Rahmatullah Nabil",
    }

    chapters[25] = re.sub(r"\s+t sixty-nine,\s*$", "", chapters[25]).strip()

    cleaned: list[str] = []
    for chapter in chapters:
        for old, new in replacements.items():
            chapter = chapter.replace(old, new)
        cleaned.append(chapter.strip())

    # The Tayeb Agha / Conflict Resolution Cell material reads as a direct
    # continuation of Chapter 23, not a standalone chapter.
    cleaned[22] = f"{cleaned[22].strip()}\n\n{cleaned[23].strip()}".strip()
    del cleaned[23]

    BOOK_PATH.write_text(_render_chapters(cleaned))
    print(BOOK_PATH.name)


if __name__ == "__main__":
    main()
