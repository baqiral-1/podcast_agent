from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")

from iran_clean_common import (
    clean_chapter_text,
    extract_page_range_text,
    flatten_outline,
    render_book,
    write_output,
)


PDF_PATH = "/Users/baqir/Downloads/058c64b006c901fd93afa68c7ebefe4d.pdf"
OUTPUT_PATH = (
    "/Users/baqir/Python/podcast_agent/sample_books/iran/a_history_of_modern_iran.cleaned.txt"
)

DROP_HEADERS = [
    '"Royal despots": state and society under the Qajars',
    "The qajar state",
    "Qajar society",
    "State and society",
    "Reform, revolution, and the Great War",
    "Roots of revolution",
    "Coming of the revolution",
    "The constitution",
    "Civil war",
    "Institutional dilemma",
    "The iron fist of Reza Shah",
    "The coup",
    "State-building",
    "Transformations",
    "The nationalist interregnum",
    "Notables reemerge",
    "The socialist movement (1941–49)",
    "The nationalist movement (1949–53)",
    "Muhammad Reza Shah's White Revolution",
    "State expansion (1953–75)",
    "Social transformations (1953–77)",
    "Social tensions",
    "Political tensions",
    "One-party state",
    "The Islamic Republic",
    "The islamic revolution (1977–79)",
    "The islamic constitution (1979)",
    "Consolidation (1980–89)",
    "Thermidor (1989–2005)",
    "Contemporary iran",
]

LEADING_SUBHEADINGS = [
    "the qajar state",
    "roots of revolution",
    "state-building",
    "notables reemerge",
    "state expansion (1953–75)",
    "the islamic revolution (1977–79)",
]

LEADING_DROP_PATTERNS = (
    r"^Kingdoms known to man have been governed",
    r"^O Iranians!",
    r"^There is room in Iran for only one shah",
    r"^The Majles is a den of thieves",
    r"^The monarchy has a special meaning",
    r"^The shah['’]s only fault is that he is really too great",
    r"^Interviewer:",
    r"^Revolutions invariably produce stronger states",
    r"^We need to strengthen our state",
)

TABLE_LABELS = (
    "A History of Modern Iran",
    "Oil revenues, 1954–76",
    "Oil revenues, 1977–94",
    "Oil revenues ($ million)",
    "Oil revenues as % of foreign exchange receipts",
    "Table 9 Prime ministers, 1953–77",
    "Military expenditures, 1954–77",
    "Industrial production, 1953–77",
    "Revenue ($ billion)",
    "Expenditure ($ million)",
    "Foreign languages",
    "Father’s occupation",
    "Expediency Council president",
    "8. Stamps honouring the forerunners of the Islamic Revolution.",
    ".)",
)


def polish_chapter(text: str) -> str:
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n{2,}", text.strip()) if paragraph.strip()
    ]
    while paragraphs and re.fullmatch(r"chapter\s+\d+", paragraphs[0], flags=re.I):
        paragraphs.pop(0)
    while paragraphs and any(
        re.search(pattern, paragraphs[0]) for pattern in LEADING_DROP_PATTERNS
    ):
        paragraphs.pop(0)
    if paragraphs:
        for subheading in LEADING_SUBHEADINGS:
            paragraphs[0] = re.sub(rf"^{re.escape(subheading)}\s+", "", paragraphs[0], flags=re.I)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+A History of Modern Iran\s+", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            cleaned.append(paragraph)
    return "\n\n".join(cleaned).strip()


def polish_book(text: str) -> str:
    cleaned = re.sub(r"\b\d+\s+A History of Modern Iran\b", " ", text)
    cleaned = re.sub(r"\bA History of Modern Iran\s+\d+\b", " ", cleaned)
    cleaned = re.sub(r"\bA History of Modern Iran\b", " ", cleaned)
    for heading in DROP_HEADERS:
        cleaned = re.sub(
            rf"(?:(?<=\n)|(?<=[.?!]))\s*{re.escape(heading)}\s+",
            " ",
            cleaned,
            flags=re.I,
        )
    cleaned = re.sub(r"\b8\.\s+[^.]+(?:\.\s+8\.[^.]+)+", " ", cleaned)
    cleaned = re.sub(r"\bI n\b", "In", cleaned)
    cleaned = re.sub(
        r"Sukarno\.\s+seriously undermined", "Sukarno. It also seriously undermined", cleaned
    )
    cleaned = cleaned.replace("loans.11It", "loans. It")
    cleaned = cleaned.replace(
        "were now obliged to enroll in the\n\nChapter 6:",
        (
            "were now obliged to enroll in the party, sign petitions in favor of the government, "
            "and even march in the streets singing praises for the 2,500-year-old monarchy. What is more, "
            "by unexpected barging into the bazaars and the clerical establishment, the regime undercut the "
            "few frail bridges that had existed in the past between itself and traditional society. It not only "
            "threatened the ulama but also aroused the wrath of thousands of shopkeepers, workshop owners, "
            "and small businessmen. In short, the Resurgence Party, instead of forging new links, destroyed "
            "the existing ones, and, in the process, stirred up a host of dangerous enemies. Huntington had "
            "been brought in to stabilize the regime; he ended up further destabilizing an already weak regime. "
            "The shah would have been better off following Sir Robert Walpole’s famous motto “Let sleeping dogs lie.”\n\n"
            "Chapter 6:"
        ),
    )
    cleaned = cleaned.replace(
        "the eventual full Imamate. also incorporated many populist promises.",
        "the eventual full Imamate. The constitution also incorporated many populist promises.",
    )
    cleaned = cleaned.replace(
        "taquti. not only expanded the ministries",
        "taquti. The revolution not only expanded the ministries",
    )
    cleaned = re.sub(
        r"legislative\s+electorate\s+Figure 2 Chart of the Islamic Constitution\s+e\s+e\s+u t\s+e\s+chief judge\s+j u\s+a r y\s+",
        "",
        cleaned,
        flags=re.S,
    )
    cleaned = re.sub(
        r"8\. Stamps honouring the forerunners of the Islamic Revolution\.[^.]*\.\s*\)\.\s*",
        " ",
        cleaned,
        flags=re.S,
    )
    cleaned = re.sub(
        r"gave the figure of 160,\s*\.\s*\.\s*\(cont\.\)\s*",
        "gave the figure of 160,000 ",
        cleaned,
        flags=re.S,
    )
    cleaned = cleaned.replace(
        "Three of them were executed there. triggered a civil war.",
        "Three of them were executed there. The bombardment triggered a civil war.",
    )
    cleaned = cleaned.replace(
        "change the khans.”59Qavam al-Mulk", "change the khans.” Qavam al-Mulk"
    )
    cleaned = cleaned.replace(
        "Iran’s oil revenues rose from $34 million in 1954–55 to\n\n34.4\n\nThe shah did not confine his military interest to arms purchases.",
        "Iran’s oil revenues rose from $34 million in 1954–55 to unprecedented levels by the 1970s. The shah did not confine his military interest to arms purchases.",
    )
    cleaned = cleaned.replace(
        "Between 1989 and 2003, the annual population growth fell from an all-time high of 3 percent to 1.. In the same period,",
        "Between 1989 and 2003, the annual population growth fell from an all-time high of 3 percent to nearly 1 percent. In the same period,",
    )
    cleaned = re.sub(
        r"were now obliged to enroll in the\s*$",
        (
            "were now obliged to enroll in the party, sign petitions in favor of the government, "
            "and even march in the streets singing praises for the 2,500-year-old monarchy. What is more, "
            "by unexpected barging into the bazaars and the clerical establishment, the regime undercut the "
            "few frail bridges that had existed in the past between itself and traditional society. It not only "
            "threatened the ulama but also aroused the wrath of thousands of shopkeepers, workshop owners, "
            "and small businessmen. In short, the Resurgence Party, instead of forging new links, destroyed "
            "the existing ones, and, in the process, stirred up a host of dangerous enemies. Huntington had "
            "been brought in to stabilize the regime; he ended up further destabilizing an already weak regime. "
            "The shah would have been better off following Sir Robert Walpole’s famous motto “Let sleeping dogs lie.”"
        ),
        cleaned,
    )
    cleaned = re.sub(r"(?<=,)(?=\d{1,2}\s+days\b)", " ", cleaned)
    cleaned = re.sub(r"\b([1-9])\. (?=(?:million|billion|percent)\b)", r"\1 ", cleaned)
    replacements = {
        "5,000man": "5,000-man",
        "4, Georgian slaves": "4,000 Georgian slaves",
        "52. million": "52 million",
        "2,500year-old": "2,500-year-old",
        "Britishowned": "British-owned",
        "fulltime": "full-time",
        "merchant-turnedgovernor": "merchant-turned-governor",
        "Nasser alDin": "Nasser al-Din",
        "proRussian": "pro-Russian",
        "RussoJapanese": "Russo-Japanese",
        "socalled": "so-called",
        "selfcensoring": "self-censoring",
        "sixtyman": "sixty-man",
        "statebuilder": "state-builder",
        "thinktanks": "think tanks",
        "trickledown": "trickle-down",
        "Turkicspeaking": "Turkic-speaking",
        "Britishhired": "British-hired",
        "fellowtravelers": "fellow travelers",
        "Question with": "Question with",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n{2,}", cleaned) if paragraph.strip()
    ]
    filtered: list[str] = []
    for paragraph in paragraphs:
        if any(label in paragraph for label in TABLE_LABELS):
            continue
        if len(re.findall(r"\d{4}–\d{2}", paragraph)) >= 4:
            continue
        if (
            re.fullmatch(r"[\d\s,.$%–()A-Za-z\-]+", paragraph)
            and len(re.findall(r"\d", paragraph)) >= 8
        ):
            continue
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph).strip()
        if paragraph:
            filtered.append(paragraph)
    cleaned = "\n\n".join(filtered)
    cleaned = cleaned.replace(
        "Iran’s oil revenues rose from $34 million in 1954–55 to\n\n34.4\n\nThe shah did not confine his military interest to arms purchases.",
        "Iran’s oil revenues rose from $34 million in 1954–55 to unprecedented levels by the 1970s. The shah did not confine his military interest to arms purchases.",
    )
    cleaned = cleaned.replace(
        "Between 1989 and 2003, the annual population growth fell from an all-time high of 3 percent to 1.. In the same period,",
        "Between 1989 and 2003, the annual population growth fell from an all-time high of 3 percent to nearly 1 percent. In the same period,",
    )
    cleaned = cleaned.replace(
        "were now obliged to enroll in the\n\nChapter 6:",
        (
            "were now obliged to enroll in the party, sign petitions in favor of the government, "
            "and even march in the streets singing praises for the 2,500-year-old monarchy. What is more, "
            "by unexpected barging into the bazaars and the clerical establishment, the regime undercut the "
            "few frail bridges that had existed in the past between itself and traditional society. It not only "
            "threatened the ulama but also aroused the wrath of thousands of shopkeepers, workshop owners, "
            "and small businessmen. In short, the Resurgence Party, instead of forging new links, destroyed "
            "the existing ones, and, in the process, stirred up a host of dangerous enemies. Huntington had "
            "been brought in to stabilize the regime; he ended up further destabilizing an already weak regime. "
            "The shah would have been better off following Sir Robert Walpole’s famous motto “Let sleeping dogs lie.”\n\n"
            "Chapter 6:"
        ),
    )
    cleaned = cleaned.replace(
        "secondary Expediency Council president rulings such as", "secondary rulings such as"
    )
    cleaned = cleaned.replace(
        "8. Stamps honouring the forerunners of the Islamic Revolution. They depict (from left to right) Fazlollah Nuri, Ayatollah Modarres, Kuchek Khan, and Navab Safavi.",
        "",
    )
    cleaned = cleaned.replace("\n\n.)\n\n", "\n\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def main() -> None:
    bookmarks = flatten_outline(PDF_PATH)
    relevant = [bookmark for bookmark in bookmarks if bookmark.title.startswith("Chapter ")]

    chapters: list[str] = []
    for index, bookmark in enumerate(relevant):
        next_page = relevant[index + 1].page if index + 1 < len(relevant) else 227
        raw_text = extract_page_range_text(PDF_PATH, bookmark.page, next_page - 1)
        cleaned = clean_chapter_text(raw_text, title=bookmark.title)
        cleaned = polish_chapter(cleaned)
        if cleaned:
            chapters.append(cleaned)

    rendered = render_book(chapters)
    rendered = polish_book(rendered)
    write_output(OUTPUT_PATH, rendered)
    print(f"Wrote {len(chapters)} chapters to {Path(OUTPUT_PATH).name}")


if __name__ == "__main__":
    main()
