from __future__ import annotations

import random
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


PDF_PATH = Path("/Users/baqir/Downloads/_OceanofPDF.com_America_and_Iran_-_John_Ghazvinian.pdf")
OUTPUT_PATH = Path("/Users/baqir/Python/podcast_agent/output/america_and_iran.cleaned.chapters.txt")
SAMPLES_PATH = Path("/Users/baqir/Python/podcast_agent/output/america_and_iran.cleaned.samples.txt")
SAMPLE_COUNT = 20
SAMPLE_WORDS = 200
SAMPLE_SEED = 20260501

CHAPTER_TITLES = [
    "East of Eden",
    "Tashrifat",
    "The Amateurs",
    "The Professionals",
    "The Man from Manila",
    "War and Peace",
    "“The Sordid Side”",
    "The Warrior-King",
    "Hello Johnny",
    "Tehran Spring",
    "“One Penny More”",
    "The Liberty Bell and the Wool Pajamas",
    "1953",
    "“Yes” and “Yes, Sir”",
    "You Say You Want a Revolution?",
    "This Turbulent Priest",
    "The Final Emperor",
    "The Unthinkable",
    "1979",
    "Dulce et Decorum Est",
    "Goodwill Hunting",
    "The First Hopey-Changey Moment",
    "That September Day",
    "The Moral Cold War",
    "Atoms for Peace?",
    "Designed to Fail",
]

FOOTNOTE_RE = re.compile(r"^[*†‡]\s+")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
MULTISPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(?:1[89]|20)\d{2}\b")
CAPTION_START_RE = re.compile(
    r"^(?:"
    r"Prime Minister|Captured U\.S\. officials|The first Iranian-Americans|"
    r"Departure|Arrival|Operation Ajax|Coup day|Black Friday|Sea of humanity|"
    r"Man of the hour|The Quiet American|Hostage crisis|Down to the wire|"
    r"Free at last|Coronation of an emperor|The blank check|Arms and the shah|"
    r"The court of the Peacock Throne|The boom before the bust|The smiling face of the revolution|"
    r"The new leadership|Man of the people|Running out of time|All smiles now|"
    r"It’s a deal|The art of no|The phone call|Lost in the midst)"
)
LOWER_EXCEPTIONS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "from",
    "by",
    "as",
    "into",
    "over",
    "under",
    "after",
    "before",
}
INLINE_NOTE_MARK_RE = re.compile(r"(?<=\S)[*†‡](?=(?:\s|$|[—–-]|[.,;:!?)]))")
LEFTOVER_CAPTION_PREFIXES = (
    "“Xerxes the Great did die, and so must you and I”:",
    "Mar Yohannan, Bishop of Urmia:",
    "Treasurer-general of Persia: W. Morgan Shuster.",
    "Muzaffar al-Din Shah (1896–1907).",
    "“Neither snow nor rain nor heat nor gloom of night stays these couriers from the swift completion of their appointed rounds”:",
    "The Great Persian Famine (1917–19):",
    "Helen Keller with Arthur Upham Pope.",
    "Samuel Jordan, legendary American missionary in Iran.",
    "First American Protestant church established in Iran (1853).",
    "Hossein Qoli Khan Nuri, “Hajji Washington.”",
    "Ghaffar Jalal, Iranian minister (ambassador) to the United States (1933–36), and his British wife, Agnes.",
    "“These people have got to be taught at whatever cost to them that they cannot get on without us.",
    "National Bank of Iran (c. 1930) and Federal Reserve Bank of the United States (1935–37).",
    "Allied invasion of Iran, September 1941:",
    "Tehran Conference, December 1943.",
    "The king and the president.",
    "U.S. ambassador John Wiley (1948–51).",
    "Mohammad Mosaddeq, prime minister (1951–53).",
    "Mosaddeq in exile.",
    "“Champion of Asia’s people”:",
    "Last of the reformists:",
    "Early concerns:",
    "The shah receives an honorary doctorate from President Gaylord Harnwell, University of Pennsylvania, 1962.",
    "Camelot:",
    "Horror in Beirut:",
    "Arms for hostages:",
    "Halabja:",
    "Death of the emam:",
    "“A vow made on marble steps”:",
    "“Axis of Evil.”",
    "Iranian foreign minister Mohammad Javad Zarif and his U.S. counterpart John Kerry arriving in Geneva on March 30, 2015, to discuss negotiations about the future of Iran’s nuclear program.",
    "Protesters demanding limitations in royal powers and the introduction of a representative assembly, 1906.",
    "Soviet tankmen of the Sixth Armored Division drive through the streets of Tabriz on a T-26 tank, August 1941.",
    "Directors of the newly renamed Anglo-Iranian Oil Company announce to workers that the British company has been taken into national ownership.",
    "“We like Ike too—Welcome to Iran!” The Eisenhower motorcade travels down a route lined with flowers and Persian carpets, March 1959.",
    "The shah lands his helicopter in the University of Pennsylvania’s football stadium, to receive an honorary doctorate, 1962.",
    "Revolutionaries cheer the return of Khomeini. Millions poured onto the streets to welcome him back to Iran—perhaps the largest spontaneous gathering in human history.",
    "Robert Dyson of the University of Pennsylvania Museum with a golden bowl discovered at Hasanlu, 1958.",
)


def extract_text(pdf_path: Path) -> str:
    with TemporaryDirectory(prefix="america-iran-clean-") as tmpdir:
        out_path = Path(tmpdir) / "raw.txt"
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-nopgbrk", str(pdf_path), str(out_path)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError((result.stderr or result.stdout).strip() or "pdftotext failed")
        text = out_path.read_text(encoding="utf-8", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def stripped_lines(text: str) -> list[str]:
    return [line.rstrip("\n") for line in text.split("\n")]


def next_nonblank(lines: list[str], start: int) -> int:
    idx = start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def find_actual_chapter_starts(lines: list[str]) -> list[tuple[int, int, int]]:
    starts: list[tuple[int, int, int]] = []
    cursor = 0
    for number, title in enumerate(CHAPTER_TITLES, start=1):
        number_text = str(number)
        found = False
        while cursor < len(lines):
            if lines[cursor].strip() != number_text:
                cursor += 1
                continue
            title_idx = next_nonblank(lines, cursor + 1)
            if title_idx >= len(lines) or lines[title_idx].strip() != title:
                cursor += 1
                continue
            body_idx = next_nonblank(lines, title_idx + 1)
            if body_idx >= len(lines):
                raise RuntimeError(f"Chapter {number} has no body start")
            starts.append((number, cursor, body_idx))
            cursor = body_idx
            found = True
            break
        if not found:
            raise RuntimeError(f"Could not find chapter {number}: {title}")
    return starts


def find_epilogue_start(lines: list[str], start: int) -> int:
    for idx in range(start, len(lines)):
        if lines[idx].strip() == "Epilogue":
            return idx
    raise RuntimeError("Could not find epilogue start")


def iter_paragraph_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def strip_note_runs(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if FOOTNOTE_RE.match(stripped):
            idx += 1
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(lines[idx])
        idx += 1
    return cleaned


def title_case_ratio(words: list[str]) -> float:
    alpha_words = []
    for word in words:
        cleaned = word.strip("“”\"'().,;:!?-")
        if not cleaned or not any(ch.isalpha() for ch in cleaned):
            continue
        alpha_words.append(cleaned)
    if not alpha_words:
        return 0.0
    titleish = 0
    for word in alpha_words:
        lower = word.lower()
        if lower in LOWER_EXCEPTIONS:
            continue
        if word[:1].isupper():
            titleish += 1
    return titleish / len(alpha_words)


def is_caption_block(block: list[str], text: str) -> bool:
    words = text.split()
    word_count = len(words)
    if not words:
        return False
    lower = text.lower()
    if text.startswith(LEFTOVER_CAPTION_PREFIXES):
        return True
    if word_count > 60:
        return False
    if "left to right" in lower:
        return True
    if CAPTION_START_RE.match(text):
        return True
    if ":" in text[:90]:
        return True
    if len(block) <= 2 and word_count <= 18 and title_case_ratio(words) >= 0.34:
        return True
    return False


def is_noise_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if FOOTNOTE_RE.match(stripped):
        return True
    if PAGE_NUMBER_RE.fullmatch(stripped):
        return True
    if stripped in {
        "PART I",
        "PART II",
        "PART III",
        "PART IV",
        "Spring",
        "Summer",
        "Autumn",
        "Winter",
    }:
        return True
    if stripped.startswith("CHAPTER ") and len(stripped.split()) <= 3:
        return True
    return False


def collapse_block(block: list[str]) -> str:
    pieces: list[str] = []
    for raw_line in block:
        line = raw_line.strip()
        if not line:
            continue
        if pieces and pieces[-1].endswith("-"):
            pieces[-1] = pieces[-1][:-1] + line
        else:
            pieces.append(line)
    text = " ".join(pieces)
    text = MULTISPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([*†‡])\s+", " ", text)
    text = INLINE_NOTE_MARK_RE.sub("", text)
    return text


def ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in {'"', "'", "”", "’", ")", "]", "*", "†", "‡"}:
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def should_merge_paragraphs(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith((",", ";", ":", "—", "–", "-", "“", '"', "(")):
        return True
    if not ends_sentence(previous):
        return True
    first_char = current[0]
    if first_char.islower():
        return True
    if first_char in {",", ";", ":", ")", "]", "”", '"', "—", "–", "-"}:
        return True
    return False


def merge_broken_paragraphs(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for paragraph in paragraphs:
        if merged and should_merge_paragraphs(merged[-1], paragraph):
            merged[-1] = f"{merged[-1]} {paragraph}".strip()
        else:
            merged.append(paragraph)
    return merged


def clean_chapter_lines(lines: list[str]) -> list[str]:
    lines = strip_note_runs(lines)
    paragraphs: list[str] = []
    for block in iter_paragraph_blocks(lines):
        text = collapse_block(block)
        if is_caption_block(block, text):
            continue
        if is_noise_block(text):
            continue
        paragraphs.append(text)
    paragraphs = merge_broken_paragraphs(paragraphs)
    return [
        p
        for p in paragraphs
        if not p.startswith(LEFTOVER_CAPTION_PREFIXES)
        and not FOOTNOTE_RE.match(p)
        and p not in {"*", "†", "‡"}
    ]


def build_output(chapters: list[tuple[int, list[str]]]) -> str:
    parts: list[str] = []
    for number, paragraphs in chapters:
        chapter_text = "\n\n".join(paragraphs).strip()
        parts.append(f"Chapter {number}:\n\n{chapter_text}")
    return "\n\n".join(parts).strip() + "\n"


def tokenize_for_samples(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def sample_segments(chapters: list[tuple[int, list[str]]]) -> str:
    rng = random.Random(SAMPLE_SEED)
    records: list[tuple[int, int, str]] = []
    for number, paragraphs in chapters:
        chapter_text = " ".join(paragraphs)
        words = tokenize_for_samples(chapter_text)
        if len(words) < SAMPLE_WORDS:
            raise RuntimeError(f"Chapter {number} is too short for {SAMPLE_WORDS}-word sampling")
        max_start = len(words) - SAMPLE_WORDS
        for _ in range(max(1, min(2, SAMPLE_COUNT - len(records)))):
            start = rng.randint(0, max_start)
            snippet = " ".join(words[start : start + SAMPLE_WORDS])
            records.append((number, start, snippet))
    while len(records) < SAMPLE_COUNT:
        number, paragraphs = rng.choice(chapters)
        words = tokenize_for_samples(" ".join(paragraphs))
        start = rng.randint(0, len(words) - SAMPLE_WORDS)
        snippet = " ".join(words[start : start + SAMPLE_WORDS])
        records.append((number, start, snippet))
    rendered: list[str] = []
    for idx, (chapter_number, start, snippet) in enumerate(records[:SAMPLE_COUNT], start=1):
        rendered.append(f"Sample {idx} | Chapter {chapter_number} | Word {start + 1}\n{snippet}\n")
    return "\n".join(rendered).strip() + "\n"


def main() -> None:
    text = extract_text(PDF_PATH)
    lines = stripped_lines(text)
    starts = find_actual_chapter_starts(lines)
    epilogue_start = find_epilogue_start(lines, starts[-1][2])
    chapters: list[tuple[int, list[str]]] = []
    for idx, (number, _start_idx, body_idx) in enumerate(starts):
        end_idx = starts[idx + 1][1] if idx + 1 < len(starts) else epilogue_start
        paragraphs = clean_chapter_lines(lines[body_idx:end_idx])
        chapters.append((number, paragraphs))
    OUTPUT_PATH.write_text(build_output(chapters), encoding="utf-8")
    SAMPLES_PATH.write_text(sample_segments(chapters), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
