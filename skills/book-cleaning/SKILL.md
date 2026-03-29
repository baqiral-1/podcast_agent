```
Clean this extracted book text for pipeline ingestion.

Goals
- Keep only the main narrative content.
- Remove all front/back matter, page noise, and structural artifacts.

Rules
1) Start point:
   - Remove everything before the first real content heading.
   - Treat “Introduction” or “Prologue” as valid start points.
   - If neither exists, start at Chapter 1 (e.g., “Chapter One”, “CHAPTER 1”, or “1”).

2) End point:
   - Remove everything from the first back‑matter heading onward.
   - Back‑matter includes: Notes, Bibliography, Glossary, Index, Acknowledgments, About the Author/Book, Image credits.

3) Headers/footers and repeated noise:
   - Remove repeated lines that look like page headers/footers, watermarks, library tags, or email/site stamps.
   - Remove repeated standalone book-title lines if they appear between pages.

4) Page/footnote markers:
   - Remove lines that are only numbers (page markers/footnote residue).

5) Heading normalization:
   - If a heading is split across multiple lines, merge into one line with single spaces.
   - Examples: PROLOGUE + title line; Chapter label + chapter title; numeric chapter + numeric year/title.

6) Preserve narrative text:
   - Do not delete regular paragraphs or dialogue.
   - Keep paragraph breaks; collapse excessive blank lines to at most two.

Output
- Write a new cleaned file (do not overwrite the original).
- Use a suffix like `.cleaned.v2.txt`.
```
