"""Tests for downloaded book cleaning helpers."""

from __future__ import annotations

from pathlib import Path

from podcast_agent.utils.book_cleaning import clean_book_text, derive_output_filename, list_recent_book_files


def test_list_recent_book_files_excludes_non_books(tmp_path: Path) -> None:
    older = tmp_path / "older.pdf"
    older.write_text("x", encoding="utf-8")
    newer = tmp_path / "newer.epub"
    newer.write_text("x", encoding="utf-8")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("x", encoding="utf-8")

    older.touch()
    newer.touch()
    ignored.touch()

    selected = list_recent_book_files(tmp_path, limit=2)
    assert selected == [newer, older]


def test_derive_output_filename_strips_download_prefixes() -> None:
    path = Path("dokumen.pub_the-rise-and-fall-of-osama-bin-laden-9781982170547.epub")
    assert derive_output_filename(path) == "the_rise_and_fall_of_osama_bin_laden.cleaned.txt"


def test_clean_book_text_trims_noise_and_renumbers_chapters() -> None:
    raw_text = """
Contents
Chapter 1 Something .... 3

dokumen.pub

Introduction
How the story starts

This is the opening paragraph.
It continues across a wrapped line.

Figure 1.1 A photo caption
17

Chapter 5
The Real Chapter Title

This is chapter text.
It also wraps across lines.

Chapter 6
Another Title

More text lives here.

Index
Abbas, 22
"""

    cleaned = clean_book_text(raw_text)

    assert cleaned.startswith("Chapter 1\n\nThis is the opening paragraph. It continues across a wrapped line.")
    assert "Chapter 2\n\nThis is chapter text. It also wraps across lines." in cleaned
    assert "Chapter 3\n\nMore text lives here." in cleaned
    assert "Contents" not in cleaned
    assert "Index" not in cleaned
    assert "Figure 1.1" not in cleaned
    assert "dokumen.pub" not in cleaned
    assert "\n17\n" not in cleaned
    assert "The Real Chapter Title" not in cleaned


def test_clean_book_text_removes_repeated_headers_and_back_matter() -> None:
    raw_text = """
The Book Title
Chapter 1
Title Line

The Book Title
This paragraph belongs to chapter one.

The Book Title
This is another paragraph.

Notes
1. source note
"""

    cleaned = clean_book_text(raw_text)

    assert "The Book Title" not in cleaned
    assert "Notes" not in cleaned
    assert cleaned.count("Chapter 1") == 1
    assert "This paragraph belongs to chapter one." in cleaned


def test_clean_book_text_detects_inline_numbered_headings() -> None:
    raw_text = """
Introduction

This introduction has enough words to count as actual body text for the
opening section and not a table of contents entry.

1. First Chapter Title

This is the first chapter body with enough text to be treated as narrative.

2. Second Chapter Title

This is the second chapter body with enough text to be treated as narrative.
"""

    cleaned = clean_book_text(raw_text)

    assert "Chapter 1" in cleaned
    assert "Chapter 2" in cleaned
    assert "Chapter 3" in cleaned
    assert "First Chapter Title" not in cleaned


def test_clean_book_text_skips_toc_headings_until_real_intro_body() -> None:
    raw_text = """
CONTENTS
INTRODUCTION Question Marks
ONE At First Sight, 1897
TWO Into the Valley, 1921

INTRODUCTION
Question Marks

This is the actual opening paragraph with enough words to be treated as prose.
It continues with another sentence that makes the introduction clearly narrative.

ONE
At First Sight, 1897

This is the first chapter body with enough words to be treated as actual narrative text.
"""

    cleaned = clean_book_text(raw_text)

    assert cleaned.startswith("Chapter 1\n\nThis is the actual opening paragraph")
    assert "Question Marks" not in cleaned
    assert "At First Sight, 1897" not in cleaned
    assert "CONTENTS" not in cleaned


def test_clean_book_text_ignores_structural_wrappers_and_character_lists() -> None:
    raw_text = """
Table of Contents
CAST OF CHARACTERS
PART I: AFGHANISTAN BECOMES A COUNTRY
Chapter 1 - Founding Father
Chapter 2 - The Amir

Introduction

This introduction opens with enough words to count as real prose for the narrative.
It adds a second sentence so the cleaner can distinguish it from a contents block.

CAST OF CHARACTERS
Ahmad Shah Durrani

Chapter 1 - Founding Father

This is the first chapter body and it should survive cleanup as ordinary narrative prose.
"""

    cleaned = clean_book_text(raw_text)

    assert "CAST OF CHARACTERS" not in cleaned
    assert "PART I" not in cleaned
    assert cleaned.startswith("Chapter 1\n\nThis introduction opens with enough words")
    assert "Chapter 2\n\nThis is the first chapter body" in cleaned


def test_clean_book_text_skips_epub_style_toc_and_back_matter_noise() -> None:
    raw_text = """
Table of Contents

INTRODUCTION

PART ONE - 9/11 AND WAR

CHAPTER ONE - A Man with a Mission The Unending Conflict in Afghanistan

CHAPTER TWO - The U.S. Will Act Like a Wounded Bear Pakistan's Long Search for Its Soul

Acknowledgements

NOTES

INDEX

OceanofPDF.com

Introduction

Imperial Overreach and Nation Building

This introduction opens with enough words to count as real narrative prose for
the first surviving section of the cleaned text.
It continues with another sentence so the cleaner can distinguish it from the TOC.

PART ONE - 9/11 AND WAR

CHAPTER ONE

A Man with a Mission The Unending Conflict in Afghanistan

This is the first chapter body and it should survive as ordinary narrative prose.
It adds a second sentence so the section is unambiguously body text.

CHAPTER TWO

The Search for a Settlement

This is the second chapter body and it should also survive cleanup.

NOTES
1. source note
"""

    cleaned = clean_book_text(raw_text)

    assert cleaned.startswith("Chapter 1\n\nThis introduction opens with enough words")
    assert "Chapter 2\n\nThis is the first chapter body" in cleaned
    assert "Chapter 3\n\nThis is the second chapter body" in cleaned
    assert "Table of Contents" not in cleaned
    assert "PART ONE - 9/11 AND WAR" not in cleaned
    assert "OceanofPDF.com" not in cleaned
    assert "NOTES" not in cleaned
    assert "INDEX" not in cleaned
    assert "A Man with a Mission" not in cleaned
    assert "Imperial Overreach and Nation Building" not in cleaned
