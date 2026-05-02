"""Tests for one-off Mughal book cleanup helpers."""

from __future__ import annotations

from podcast_agent.utils.mughal_cleanup import clean_mughal_book_text, clean_mughal_ocr_pages, clean_mughal_pages


def test_clean_mughal_book_text_drops_preface_and_keeps_intro() -> None:
    raw_text = """
CONTENTS
Preface .... ix
Acknowledgements .... xi
Introduction .... 1
1. First Chapter .... 9

PREFACE
This preface should be removed even though it contains enough words to look
like real prose and would otherwise survive a naive chapter extraction pass.

ACKNOWLEDGEMENTS
These thanks should also disappear.

ABBREVIATIONS
OUP Oxford University Press

Introduction
Opening Perspective

This introduction should stay because it is narrative prose with enough words
to count as real body text for the cleaned output.

1. First Chapter
The Chapter Title

This is the first real chapter body and it should remain in the final text.

NOTES
1. This should not survive.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.startswith("Chapter 1:\nThis introduction should stay")
    assert "Chapter 2:\nThis is the first real chapter body" in cleaned
    assert "PREFACE" not in cleaned
    assert "ACKNOWLEDGEMENTS" not in cleaned
    assert "ABBREVIATIONS" not in cleaned
    assert "Opening Perspective" not in cleaned
    assert "The Chapter Title" not in cleaned
    assert "NOTES" not in cleaned


def test_clean_mughal_book_text_removes_subheadings_and_scan_attributions() -> None:
    raw_text = """
Digitized by Google
Original from Harvard University

CHAPTER ONE
An Elaborate Chapter Title

This is the first paragraph of the chapter and it should survive cleanup as
ordinary narrative prose with enough words to count as body text.

The Inner Court

This second paragraph should stay, but the short subheading above it should
not appear in the final output.

Figure 3. A courtyard image caption
Google

CHAPTER TWO
Another Chapter Title

This is the next chapter body, and it should remain after cleanup.

Index
Akbar, 21
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.startswith("Chapter 1:\nThis is the first paragraph")
    assert "Chapter 2:\nThis is the next chapter body" in cleaned
    assert "An Elaborate Chapter Title" not in cleaned
    assert "The Inner Court" not in cleaned
    assert "Figure 3." not in cleaned
    assert "Digitized by Google" not in cleaned
    assert "Original from Harvard University" not in cleaned
    assert "Index" not in cleaned


def test_clean_mughal_ocr_pages_uses_running_headers_for_chapters() -> None:
    pages = [
        """
PREFACE

This front matter should be skipped entirely.
""",
        """
TWILIGHT OF THE MUGHULS

This is the opening page of the first chapter and it has enough
words to count as real narrative prose for the first chapter body.
It also adds a second sentence with enough extra detail to clear
the pre-chapter front-matter threshold used for OCR scans.
""",
        """
THE KINGDOM OF DELHI

This page reveals the chapter running header and should anchor
the opening chapter that already started on the prior page.
""",
        """
THE KINGDOM OF DELHI

This page keeps the same running header so the cleaner can
confirm the first chapter title is genuine.
""",
        """
TWILIGHT OF THE MUGHULS

More prose from the first chapter should be retained here.
""",
        """
THE MUGHULS AND THE BRITISH

This page begins the second chapter and should therefore start a
new cleaned section in the final output.
""",
        """
THE MUGHULS AND THE BRITISH

This page continues under the same running header so the cleaner
can confirm it is a real chapter boundary rather than a stray title.
""",
        """
TWILIGHT OF THE MUGHULS

This page continues the second chapter with ordinary narrative text.
""",
        """
INDEX
Akbar, 21
""",
    ]

    cleaned = clean_mughal_ocr_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis page reveals the chapter running header")
    assert "This page reveals the chapter running header" in cleaned
    assert "Chapter 2:\nThis page begins the second chapter" in cleaned
    assert "PREFACE" not in cleaned
    assert "INDEX" not in cleaned


def test_clean_mughal_pages_skips_front_matter_until_first_explicit_chapter() -> None:
    pages = [
        """
CAMBRIDGE STUDIES IN INDIAN HISTORY

Series copy that should never appear in the cleaned output.
""",
        """
PREFACE

This preface should be dropped even though it contains prose.
""",
        """
INTRODUCTION

This introduction should also be skipped in the page-aware second pass.
""",
        """
CHAPTER 1
The First Real Chapter

This is the first chapter body and it should become Chapter 1
in the final rendered output without the source chapter title.
""",
        """
2
The first chapter continues with more narrative prose here.
""",
        """
CHAPTER 2
The Second Real Chapter

This is the second chapter body and it should become Chapter 2.
""",
    ]

    cleaned = clean_mughal_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis is the first chapter body")
    assert "Chapter 2:\nThis is the second chapter body" in cleaned
    assert "CAMBRIDGE STUDIES" not in cleaned
    assert "PREFACE" not in cleaned
    assert "INTRODUCTION" not in cleaned
    assert "The First Real Chapter" not in cleaned


def test_clean_mughal_pages_uses_toc_titles_and_stops_before_conclusion() -> None:
    pages = [
        """
CONTENTS
1 The First Campaign 1
2 The Second Campaign 3
""",
        """
PREFACE

Discarded front matter.
""",
        """
1
The First Campaign

This is the opening chapter body and it should survive.
""",
        """
2
More prose from the first chapter continues here.
""",
        """
3
The Second Campaign

This is the second chapter body and the TOC title should anchor it.
""",
        """
4
More prose from the second chapter continues here.
""",
        """
CONCLUSION

This should not appear in the final output.
""",
    ]

    cleaned = clean_mughal_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis is the opening chapter body")
    assert "Chapter 2:\nThis is the second chapter body" in cleaned
    assert "The Second Campaign" not in cleaned
    assert "CONCLUSION" not in cleaned


def test_clean_mughal_pages_uses_roman_toc_entries_and_page_number_then_chapter_heading() -> None:
    pages = [
        """
CONTENTS
Chapter I. Bahadur Shah 1
Chapter II. Jahandar Shah 5
Chapter III. Farrukh-Siyar 9
""",
        """
1
CHAPTER I
BAHADUR SHAH

Sec. I. Death of Alamgir

This is the first real chapter body and it should become Chapter 1.
""",
        """
2
More prose from the first chapter continues here.
""",
        """
5
CHAPTER II
JAHANDAR SHAH

This is the second chapter body and it should become Chapter 2.
""",
    ]

    cleaned = clean_mughal_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis is the first real chapter body")
    assert "Chapter 2:\nThis is the second chapter body" in cleaned
    assert "Sec. I. Death of Alamgir" not in cleaned
    assert "BAHADUR SHAH" not in cleaned


def test_clean_mughal_pages_does_not_misread_title_page_as_chapter() -> None:
    pages = [
        """
DELHI BETWEEN TWO EMPIRES

Society, Government and Urban Growth

NARAYANI GUPTA
""",
        """
CONTENTS
1. The British Peace and the British Terror
2. Portrait of the City
""",
        """
I
THE BRITISH PEACE AND
THE BRITISH TERROR

This is the first chapter body and it should be Chapter 1.
""",
        """
2
PORTRAIT OF
THE CITY

This is the second chapter body and it should be Chapter 2.
""",
    ]

    cleaned = clean_mughal_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis is the first chapter body")
    assert "Chapter 2:\nThis is the second chapter body" in cleaned
    assert "NARAYANI GUPTA" not in cleaned


def test_clean_mughal_pages_drops_publisher_and_note_paragraphs() -> None:
    pages = [
        """
CHAPTER 1
First Essay

Oxford University Press is a department of the University of Oxford.

This is the real opening paragraph and it should remain in the output.

1. IOR, BRP, P/71/26, 18 June 1790.
""",
        """
CHAPTER 2
Second Essay

This second chapter should also survive while the citation paragraph drops.

2. IOL, MS Eur. D. 75, vol. 2, book 4, fol. 84.
""",
    ]

    cleaned = clean_mughal_pages(pages)

    assert cleaned.startswith("Chapter 1:\nThis is the real opening paragraph")
    assert "Oxford University Press" not in cleaned
    assert "IOR, BRP" not in cleaned
    assert "IOL, MS Eur." not in cleaned


def test_clean_mughal_book_text_strips_leading_opening_artifact() -> None:
    raw_text = """
CHAPTER ONE
Title

[ 1700 the Mughal emperor began his reign with a clear program.

This next paragraph should remain.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.startswith("Chapter 1:\nIn 1700 the Mughal emperor began his reign")
    assert "[ 1700" not in cleaned


def test_clean_mughal_book_text_drops_spurious_short_opening_section() -> None:
    raw_text = """
CHAPTER ONE
Title

Burhanpur in the Dakhin on 30th Rajab 1053 (14th Oct., 1643). His mother, and the mother of the eldest son, Muhammad Sultan, was Nawab Bai, daughter of Rajah Raju.

Muhammad Azam the third son was born of Dilras Banu Begam, daughter of Shah Nawaz Khan Safawi.

CHAPTER TWO
Another Title

This is the first real narrative chapter and it should become Chapter 1.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.startswith("Chapter 1:\nThis is the first real narrative chapter")
    assert "Burhanpur in the Dakhin" not in cleaned


def test_clean_mughal_book_text_drops_trailing_reference_section() -> None:
    raw_text = """
CHAPTER ONE
Title

This is the real chapter body and it should survive in the final output.

CHAPTER TWO
Reference Tail

Vol. lvii, 1888.

Printed Books (Persian and Urdu)

Masir-i-Alamgiri, by Mbhd. Sagi, Mustaid Khan, composed 1122 H., (Bib. Ind.) 8vo., Calcutta, 1871.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.strip() == "Chapter 1:\nThis is the real chapter body and it should survive in the final output."


def test_clean_mughal_book_text_drops_note_heavy_tail_block() -> None:
    raw_text = """
CHAPTER ONE
Title

This is the real narrative body and it should remain.

The records of the church were lost during the Mutiny, and no picture of this monument has yet been discovered. We have, however, a description of the tomb by Fanny Parks, who visited Delhi and saw the tomb in 1838. In a compartment in front of the church is a Persian inscription.

Freed from human cares and woes; Like us his heart like ours his frame.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.strip() == "Chapter 1:\nThis is the real narrative body and it should remain."


def test_clean_mughal_book_text_truncates_embedded_notes_block() -> None:
    raw_text = """
CHAPTER ONE
Title

This is the final narrative paragraph and it should survive as the chapter body. NOTES 1. Home Poll. 12A/1914. 2. Ibid., p. 17.
"""

    cleaned = clean_mughal_book_text(raw_text)

    assert cleaned.strip() == (
        "Chapter 1:\nThis is the final narrative paragraph and it should survive as the chapter body."
    )
