"""Rubric-based audio style scoring and standalone HTML reporting."""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAPITALIZED_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

DEFAULT_SCRIPT_ARTIFACTS = ("episode_script.json",)
DEFAULT_SAMPLE_WORD_TARGETS = (1200, 800, 800)

DEFAULT_TARGET_SNIPPET = """
One of the most fascinating, interesting cultures on the planet and what Japanese expert are
Tagget Murphy calls unquestionably the most distinctive of all modern industrial societies
culture wise. And it is a fascinating example of how we human beings can be molded and
stretched and trimmed and shaved like a bonsai tree by the cultural influences into many
different diverse versions of ourselves. Anyone who's a fan of National Geographic, for
example, sees the many different colorful cultures and subcultures all over the world right.
The extreme differences of humankind and how wonderfully pliable we are to the cultural
influences that, you know, determine how we like to dress, what we think is attractive, little
marks of status, all kinds of things. Now, magnify that by all the errors humankind's ever
been around, right, so not just the diversity of exotic, different places all over the world,
but all of those places all throughout history. And you can see that we've, well, probably
tried just about everything as a species, wouldn't you think?
And and we've seen all kinds of different extreme versions of what we can become. I mean, take
the physical version. I'm a fan of the people of the Eurasian step and those tribes, as many
of you know, some of those tribes from time to time used to like to practice something known
as head binding. It's sort of the counterpoint to foot binding in China, which was popular in
some circles at some times to keep the feet small and to have them develop a certain way.
Well, same thing with heads.
You see this in other parts of the world, too, and sometimes cradles. For example, certain
Native American tribes had cradles that would shape the head a certain way to. The point is,
is that when the skull is pliable, when the person is young, you can shape that head sort of
like a bonsai tree, if you will. And the Romans used to complain that the Huns looked
monstrous to them.
And I remember growing up thinking, wow, these Romans are really sensitive to seeing some
people who maybe have slightly Asian features thinking that that's monstrous. Well, now we
have skulls that have come out of the ground. Some of these tombs, some of these Kirkenes
where they buried some of these Huns and some of these Huns have skulls that looked like
they're taken right out of an alien movie, big elongated skulls that go way far to the back.
You wonder what they must have looked like with hair and skin.
The point is, is if you were a Roman looking at them, you might have thought that they looked
monstrous. However, if you were a young Komla Hunwick noblewoman, you might have thought that
guy looked hot. So it's all a question, right, of of the cultural carrots and sticks and
whatnot. And sometimes it's not things that are so visible on the outside. I mean, I think we
all recognize that the vast majority of our makeup in terms of our attitudes and the way we
think in the way we organize reality is probably going to be culturally determined.
Now, I am a fan of extreme situations in history and the country that produced guys like Hiroo
Onoda were extreme and the cultural carrots and sticks were designed to take many of the, you
know, civic attitudes that most societies consider to be positives. And turn them up to such a
high level on the dial of intensity that they took a bunch of things that might be considered
good in another context and made them dangerous. And how can your culture impact things like
Dudi?
And honor. And patriotism and a willingness to sacrifice and lay down one's life for the
greater good. I mean, these are all things that most societies, you know, would figure out all
sorts of cultural ways to encourage. I mean, how about just recognizing it? I mean, take, for
example, a medal ceremony that might happen in the United States or an award, a citation,
somebody was extra brave or extra heroic or did something out of the ordinary.
And maybe you get a medal and when you get the medal, the person giving the award will say
something like, you know, for heroism or bravery or sacrifice above and beyond the call of
duty, above and beyond the call of duty.
That right there is a recognition, right, that there are limits to what normal people can be
asked to do in normal times. Right. And if you exceed those expectations, go above and beyond
the call of duty, you get a medal. But what are the cultural carrots and sticks in a given
society have evolved in such a way? So that there's almost nothing that is above and beyond
the call of duty. Where the culture expects, encourages and only approves of a level of
sacrifice on the part of every man, woman and child in that society that most other cultures
reserve for their most elite military units.
Welcome to the world that famed Japanese hold out hero Onoda was raised in. When Onoda got
back to Japan in 1974, a ghost written book of his remembrances was created, it's called No
Surrender My 30 Year War, and in it he describes the Japan he was raised in. He says, quote.
At that time, if a soldier who had been taken prisoner later managed to return to Japan, he
was subject to a court martial and a possible death penalty, even if the penalty was not
carried out.
He was so thoroughly ostracized by others that he might as well have been dead. Soldiers were
supposed to give their lives for the cause, not groveled an enemy. Prison camps General Hideki
Tojo's instructions for the military said explicitly. He's now quoting the instructions,
quote. He who would not disgrace himself must be strong, he must remember always the honor of
his family and his community, and he must strive fervently to live up to their trust in him,
do not live in shame as a prisoner, die and leave no ignominious crime behind you.
And to quote. When he leaves to go off to war, one of the things he takes with him is a family
dagger that his mother says if your capture, use this to kill yourself with. And I tried to
think of any other mother sending her son off to war in any other major power in the Second
World War with that kind of advice.
It's interesting, isn't it? A very distinctive, different culture. Now, some might have a
counterargument to this point that the reason these Japanese holdouts were continuing to hold
out was for duty, honor, country, the emperor, that sort of thing, by pointing out that a
bunch of these people said they didn't know the war was over. So maybe it's more of a sense of
being isolated and not getting the news. But the counterargument to that has to do with the
fact that these people were often told explicitly in many different ways that the war was over
and they chose not to believe it and they chose not to believe it because it clashed with
their, you know, mental indoctrination into all this.
I mean, for example, Hyrule Noda, he had family members flown in at high expense to the
Japanese government because he were not it was still killing people in the Philippines after
the war. But he's not just sitting up there in the mountains by himself, not bothering
anybody. He killed 20 to 30 Filipinos.
So that makes you want to tell this person in no uncertain terms and spend a little money to
make it happen. Hey, you're killing people in the war is over. But Ainata didn't believe it.
They left him newspapers. Here's how he puts it in the book written with him.
""".strip()

DEFAULT_TSS_V2_SNIPPET = """
Hello and welcome to Revolutions. Episode 3.44 The War Feeds Itself. Since the Termidorians came to power, we've seen them beat down an insurrection from the left, then an insurrection from the right, and then just last week, another insurrection from the left. We've seen them disband the National Guard in Paris, set up a new police legion, only to disband that new police legion a few months later. So you may be asking yourself through all of this, if both wings of the political spectrum are trying to overthrow the new regime, how on earth are they still in power? The answer lies in the institution that will increasingly serve as the backbone of the directory, the army. The regular army played a crucial role in overawing the insurrectionaries of Germinal and Prairial, and then of course under Bonaparte's direction, regular army troops subdued the royalists of Vendemiaire. And then last week, when the conspiracy of equals was exposed, Lazare Carnot transferred police power directly to the army. So today, we're going to talk about the army, how it was being transformed into an institution with its own permanent interests, and how it fared in the campaigns of 1795 and 1796. Campaigns that helped transform it into an institution with its own permanent interests, interests that for the moment aligned with the directory, but we all know how this plays out in the end.

Okay, so just to recap a little bit. Before the revolution, the French army was trending in a bad direction. Instead of modernizing through the 1780s, the French nobility had decided to turn the military back into an iron-clad stronghold of aristocratic privilege. The common soldiers were subjected to pretty draconian punishments, the non-noble professional soldiers were now forced to top out at captain, and all the top spots were reserved for the well-bred and well-connected sword nobility.

After the revolution hit, two-thirds of the officer corps emigrated, including almost all of those old sword nobles, leaving an officer corps that had to be restocked by men who then naturally tied their career prospects to the success of the revolution. And the common soldiers were then augmented by passionate volunteers who made up in zeal what they lacked in formal training, this all being brought to full fruition by the levee en masse, which eventually brought nearly a million men under arms, and then also the amalgame, which tied veterans to new recruits to quickly introduce rookies to the rules, responsibilities, and values of men under arms. The levee en masse also mobilized the entire nation to keep those soldiers fed, clothed, and armed. Forged in the crisis of 1793, this new French national army became a tidal wave that slowly pushed all its enemies back.

But one of the major features of the army during this period of crisis, and one of the major innovations in the history of political warfare, was the representative on mission. Political commissars like Saint-Just had the final word in everything, ordering their generals when to fight and where to fight, and controlling all promotions and discharges. While the Committee of Public Safety reigned, the army worked for them. The aggressive zeal of the representatives probably had a lot to do with France not only surviving the great military tests of 1793 and 1794, but frankly emerging stronger than ever. After Termidor, though, a few subtle shifts began to take place. The all-powerful representatives on mission were recalled, and when the central government sent agents back to the armies, they no longer had the authority to treat the senior officers as subordinates.

Yes, the generals were appointed by, and took their orders from, Paris. But that was about big-picture grand strategy. Day-to-day minutiae, the actual deployment of divisions, the hiring and firing of officers, that became the purview of the generals. And so it didn't take long for fidelity to whatever happened to be the political ideology of the day to become a client-patron relationship between the junior and senior officers out in the field. This led to the growth of an insular system of mutual loyalty among the officer corps who protected and elevated each other.

This developing esprit de corps was cemented further by the deplorable state of the national finances in 1795 and 1796. It was getting harder and harder to keep the armies properly supplied. And so, for example, when the Army of the North invaded the United Provinces, they were basically told to live off the land, requisitioning whatever supplies they needed. This wound up working pretty well for the army occupying the Netherlands, not so much for the Dutch. And so in the offensive campaigns of 1795, and then especially in 1796, which we're about to get into, the government told the army to live off the lands they invaded. Basically, please open your hymnals to Livy, Book 34, Chapter 9. It happened to be the time of year when the Spaniards had the grain on their threshing floors. Cato, therefore, forbade the contractors to purchase any, and sent them back to Rome, saying, the war will feed itself.

This change in the underlying supply logic had a couple of major impacts. First, it meant that to eat, the army needed to go out a-conquering. They could not afford to simply construct a defensive line along the Rhine and just sit there. Second, it meant that the men were now relying on their generals, not on their government, to keep them fed, clothed, and paid. Victory now meant full bellies and some cash. Defeat meant hunger and poverty. So if a general started to deliver spectacular victory after spectacular victory in one of the richest and most bountiful regions in Europe, well, who do you think you would give your allegiance to?

After the run of treaties in the spring and summer of 1795 pulled the Prussians, the Dutch, and the Spanish out of the war, the French were left to face off against the British and the Austrians. But with the British mostly focused now on using their naval power to attack French colonial holdings on the other side of the world, or blockade French ports, that meant that the only land power that threatened France was the Austrians and their various little satellite principalities. So just as the Thermidorian Convention was handing power to the Directory in the fall of 1795, they ordered their two major armies along the Rhine to go on the offensive with the intention of seizing all the major strongholds along the great river, the central prize being the great fortress city of Mainz, the really central prize being all their money. The French armies in question were both a little over 60,000 strong and led respectively by Jean-Baptiste Jourdan in the north and Charles Pichegru in the south.

In between those two field armies, a third French force sat on the west bank of the Rhine next to Mainz, attempting to blockade the city, but unless it was enveloped from the east, Mainz would never fall. The overall strategy for the year was developed back in Paris by Lazare Carnot, who was convinced that a double pincer move, his absolutely favorite kind of move, could envelop and crush Austrian resistance. Now he might have been right about that. It might have worked, except that Pichegru seemed suspiciously uninterested in holding up his end of the pincer.

Opposing the French was, for the moment, the main Austrian army of the lower Rhine, though an army of the upper Rhine was being hastily cobbled together in the Black Forest to reinforce the southern Austrian defenses. So Jourdan crossed the Rhine first in early September, seizing Dusseldorf before turning south to approach Mainz from the north. The main Austrian army moved to deflect this advance, which allowed Pichegru to cross the Rhine south of Mainz without any difficulty. Pichegru then quickly captured Mannheim without firing a shot, but then he bungled away a golden opportunity to seize the main Austrian supply depot at Heidelberg, just a few miles to the southeast. Had he taken Heidelberg, the Austrian position along the Rhine might have become completely untenable and forced them to retreat deeper into their own territory. But instead of prioritizing the supply depot, Pichegru only sent two divisions to capture it. And those two divisions were defeated by the Heidelberg garrison on September 24. It was a crucial blunder that may very well have been a conscious decision by Pichegru to undermine his own campaign, because as would be discovered later, Pichegru had by now opened up a treasonous correspondence with agents of the self-proclaimed Louis XVIII to discuss the logistics of restoration.
""".strip()

DEFAULT_TSS_V2_LABEL = "Bundled Revolutions directory-army sample"

AFS_RAW_ANCHORS = (36.47, 42.47, 56.77)
AFS_CALIBRATED_ANCHORS = (62.0, 67.0, 74.0)
TSS_RAW_ANCHORS = (41.49, 47.11, 58.98)
TSS_CALIBRATED_ANCHORS = (31.0, 40.0, 51.0)

DIMENSION_SPECS = (
    ("host_presence", "Host Presence", 6, 14),
    ("conversational_cadence", "Conversational Cadence", 12, 12),
    ("listener_guidance", "Listener Guidance", 10, 10),
    ("scene_immediacy", "Scene Immediacy", 10, 8),
    ("structural_signposting", "Structural Signposting", 10, 4),
    ("rhetorical_intensity", "Rhetorical Intensity", 8, 12),
    ("associative_riffing", "Associative Riffing", 4, 12),
    ("analogy_density", "Analogy Density", 6, 10),
    ("oral_redundancy", "Oral Redundancy", 8, 8),
    ("spoken_looseness", "Spoken Looseness", 6, 10),
    ("clarity_under_load", "Clarity Under Load", 12, 4),
    ("memorable_lines", "Memorable Lines", 8, 6),
)

DIMENSION_HELP = {
    "host_presence": "Is there a felt narrator on-mic, not just polished prose?",
    "conversational_cadence": "Do the sentences sound spoken, beat-driven, and breathable?",
    "listener_guidance": "Does the script actively steer and orient the listener?",
    "scene_immediacy": "Does it show something visible before it starts interpreting?",
    "structural_signposting": "Can a listener track the section turns and stakes in one pass?",
    "rhetorical_intensity": "How much spoken emphasis, escalation, contrast, and charge does it carry?",
    "associative_riffing": "Does it think outward, compare freely, and circle back?",
    "analogy_density": "How often does it explain through vivid analogy or comparison?",
    "oral_redundancy": "Does it restate key ideas for the ear instead of trusting a single pass?",
    "spoken_looseness": "Does it feel page-tight or naturally talk-shaped?",
    "clarity_under_load": "Can listeners track names, dates, and politics on the first listen?",
    "memorable_lines": "Are there lines that feel built to stick in memory?",
}

GUIDANCE_PATTERNS = (
    r"\bhold that\b",
    r"\bnotice\b",
    r"\bhere(?:'s| is)\b",
    r"\bthe point is\b",
    r"\bthe point\b",
    r"\bto understand\b",
    r"\bthink about\b",
    r"\bremember\b",
    r"\byou can see\b",
    r"\bread that\b",
    r"\bkeep in mind\b",
    r"\bwhat matters\b",
    r"\bwhat this means\b",
    r"\bin plain english\b",
    r"\bwhich is to say\b",
    r"\bthat is to say\b",
    r"\bfor our purposes\b",
    r"\bslow down\b",
    r"\bpause on\b",
    r"\bpicture\b",
)

CALLBACK_PATTERNS = (
    r"\bhold that\b",
    r"\bremember\b",
    r"\bcomes back\b",
    r"\breturns here\b",
    r"\bwe left\b",
    r"\bas we saw\b",
    r"\bas we've seen\b",
    r"\bcarry that forward\b",
    r"\bkeep that\b",
    r"\bagain\b",
)

ANALOGY_PATTERNS = (
    r"\blike\b",
    r"\bas if\b",
    r"\bit's a bit like\b",
    r"\bthink of\b",
    r"\bimagine\b",
    r"\bpicture\b",
    r"\bif you were\b",
    r"\bit's the counterpoint to\b",
)

ASIDE_PATTERNS = (
    r"\bi mean\b",
    r"\byou know\b",
    r"\bif you will\b",
    r"\bwouldn't you think\b",
    r"\bit's interesting\b",
    r"\bwell\b",
    r"\bof course\b",
    r"\bnow\b",
)

CONTRACTION_PATTERNS = (
    r"\b(?:i'm|we're|you're|it's|that's|there's|don't|doesn't|didn't|can't|won't|isn't|aren't|wasn't|weren't|they're|we've|you've|i've|we'll|you'll|wouldn't|shouldn't|couldn't)\b",
)

TRANSLATION_PATTERNS = (
    r"\bin plain english\b",
    r"\bwhich is to say\b",
    r"\bthat is to say\b",
    r"\bwhat this means\b",
    r"\bin other words\b",
    r"\bplainly\b",
)

CONTRAST_PATTERNS = (
    r"\bnot\b[^.?!]{0,80}\bbut\b",
    r"\binstead\b",
    r"\brather than\b",
    r"\bhowever\b",
    r"\bthe trouble is\b",
    r"\bthe result is\b",
)

EMPHASIS_PATTERNS = (
    r"\bthis is\b",
    r"\bthat is\b",
    r"\bthe whole thing\b",
    r"\bthe point is\b",
    r"\bwhat matters\b",
    r"\bthat is the\b",
    r"\bthis is the\b",
)

META_PATTERNS = (
    r"\bthis episode\b",
    r"\bthis story\b",
    r"\bthis series\b",
    r"\bwe will see\b",
    r"\bwe are about to see\b",
    r"\bwe're about to see\b",
    r"\bfor the next\b",
)

ABSTRACT_OPENER_PATTERNS = (
    r"^(?:the point|what this means|to understand|the question|the problem|the argument|the result|the reason|this is what|that is what|in plain english)\b",
)

CONCRETE_NOUNS = {
    "bridge",
    "room",
    "road",
    "crowd",
    "river",
    "city",
    "village",
    "street",
    "station",
    "train",
    "house",
    "hall",
    "chamber",
    "bagh",
    "door",
    "truck",
    "car",
    "night",
    "morning",
    "afternoon",
    "beach",
    "shore",
    "gate",
    "lane",
    "roof",
    "rooftop",
    "police",
    "rifle",
    "flag",
    "line",
}

TIME_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "dawn",
    "midnight",
    "morning",
    "evening",
    "afternoon",
    "night",
}

ACTION_WORDS = {
    "stands",
    "stand",
    "stood",
    "waits",
    "wait",
    "walks",
    "walk",
    "pulls",
    "arrives",
    "fires",
    "opens",
    "closes",
    "holds",
    "looks",
    "rolls",
    "leans",
    "moves",
    "steps",
    "runs",
    "boards",
}


@dataclass(frozen=True)
class SampleWindow:
    label: str
    paragraph_start: int
    paragraph_end: int
    word_count: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutoMetrics:
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    short_sentence_share: float
    long_sentence_share: float
    fragment_share: float
    question_density: float
    first_person_density: float
    direct_address_density: float
    listener_guidance_density: float
    callback_density: float
    analogy_density: float
    contrast_density: float
    translation_density: float
    aside_density: float
    contraction_density: float
    meta_density: float
    quote_density: float
    dash_density: float
    emphasis_density: float
    concrete_opener_rate: float
    abstract_opener_rate: float
    opening_anchor_density: float
    entity_load_per_200_words: float
    verdict_parallel_density: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DimensionScore:
    key: str
    label: str
    afs_weight: int
    tss_weight: int
    afs_rating: float
    afs_points: float
    tss_rating: float
    tss_points: float
    tss_v2_rating: float | None
    tss_v2_points: float | None
    help_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeReport:
    episode_number: int
    title: str
    script_artifact: str
    source_path: str
    total_word_count: int
    sample_word_count: int
    raw_afs: float
    afs: float
    raw_tss: float
    tss: float
    tss_v2: float | None
    dimensions: list[DimensionScore]
    sample_windows: list[SampleWindow]
    metrics: AutoMetrics
    evidence: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimensions"] = [dimension.to_dict() for dimension in self.dimensions]
        payload["sample_windows"] = [window.to_dict() for window in self.sample_windows]
        payload["metrics"] = self.metrics.to_dict()
        return payload


@dataclass(frozen=True)
class RunReport:
    run_id: str
    run_dir: str
    episode_count: int
    total_word_count: int
    sampled_word_count: int
    raw_afs: float
    afs: float
    raw_tss: float
    tss: float
    tss_v2: float | None
    dimensions: list[DimensionScore]
    metrics: AutoMetrics
    episodes: list[EpisodeReport]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimensions"] = [dimension.to_dict() for dimension in self.dimensions]
        payload["metrics"] = self.metrics.to_dict()
        payload["episodes"] = [episode.to_dict() for episode in self.episodes]
        return payload


def _round_score(value: float) -> float:
    return round(value, 2)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_script_path(episode_dir: Path, script_artifacts: Sequence[str]) -> Path | None:
    for artifact in script_artifacts:
        candidate = episode_dir / artifact
        if candidate.exists():
            return candidate
    return None


def _episode_script_paths(run_dir: Path, script_artifacts: Sequence[str]) -> list[Path]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return []
    selected: list[Path] = []
    for episode_dir in sorted(
        (path for path in episodes_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    ):
        chosen = _choose_script_path(episode_dir, script_artifacts)
        if chosen is not None:
            selected.append(chosen)
    return selected


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _sentence_lengths(text: str) -> list[int]:
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text.strip()) if segment.strip()]
    return [len(_tokenize(sentence)) for sentence in sentences]


def _count_pattern_hits(text: str, patterns: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(pattern, lowered, flags=re.IGNORECASE)) for pattern in patterns)


def _per_thousand(count: float, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return (count * 1000.0) / word_count


# Antithetical-parallel verdict frame: the "It is not X. It is Y." /
# "X is real, Y is not." / "Not A. B." structure this kind of writing falls into
# as its near-exclusive way to land a claim. Tracked as a diagnostic so the tic
# is measurable across runs (see verdict_parallel_density); not wired into the
# weighted AFS/TSS composite, to avoid disturbing the calibrated anchors.
_VERDICT_NEGATED_PREDICATE_RE = re.compile(
    r"\b(?:is|are|was|were|it's|that's|isn't|aren't|wasn't|weren't)\s+not\b",
    re.IGNORECASE,
)
_VERDICT_NOT_OPENER_RE = re.compile(r"(?:^|[.?!]\s+)not\s+[a-z]", re.IGNORECASE)


def _verdict_parallel_count(text: str) -> int:
    """Count antithetical-parallel verdict landings.

    Two markers: a short sentence (<=12 words) built on a negated predicate
    ("the mandate is real, the state is not"), and a sentence opening on a bare
    "Not ..." ("Not force. Silence."). Both are signatures of the single verdict
    frame the prose overuses.
    """
    sentences = [
        segment.strip()
        for segment in SENTENCE_SPLIT_RE.split(text.strip())
        if segment.strip()
    ]
    negated_short = sum(
        1
        for sentence in sentences
        if len(_tokenize(sentence)) <= 12
        and _VERDICT_NEGATED_PREDICATE_RE.search(sentence)
    )
    not_openers = len(_VERDICT_NOT_OPENER_RE.findall(text))
    return negated_short + not_openers


def _paragraphs_from_script(script_payload: dict[str, object]) -> list[str]:
    sections = script_payload.get("prose_sections")
    if not isinstance(sections, list):
        sections = script_payload.get("sections")
    paragraphs: list[str] = []
    if not isinstance(sections, list):
        return paragraphs
    for section in sections:
        if not isinstance(section, dict):
            continue
        text = section.get("text")
        if not isinstance(text, str):
            continue
        paragraphs.extend(part.strip() for part in text.split("\n\n") if part.strip())
    return paragraphs


def _paragraph_word_counts(paragraphs: Sequence[str]) -> list[int]:
    return [len(_tokenize(paragraph)) for paragraph in paragraphs]


def _window_from_start(paragraphs: Sequence[str], paragraph_word_counts: Sequence[int], target_words: int) -> tuple[int, int]:
    total = 0
    end_index = 0
    while end_index < len(paragraphs) and total < target_words:
        total += paragraph_word_counts[end_index]
        end_index += 1
    return 0, max(1, end_index)


def _window_from_center(
    paragraphs: Sequence[str],
    paragraph_word_counts: Sequence[int],
    target_words: int,
    target_fraction: float,
) -> tuple[int, int]:
    if not paragraphs:
        return 0, 0
    total_words = sum(paragraph_word_counts)
    target_word = total_words * target_fraction
    running = 0
    center_index = 0
    for index, count in enumerate(paragraph_word_counts):
        running += count
        if running >= target_word:
            center_index = index
            break
    start = center_index
    end = center_index + 1
    total = paragraph_word_counts[center_index]
    left = center_index - 1
    right = center_index + 1
    while total < target_words and (left >= 0 or right < len(paragraphs)):
        left_distance = abs(left - center_index) if left >= 0 else math.inf
        right_distance = abs(right - center_index) if right < len(paragraphs) else math.inf
        if left_distance <= right_distance and left >= 0:
            start = left
            total += paragraph_word_counts[left]
            left -= 1
        elif right < len(paragraphs):
            end = right + 1
            total += paragraph_word_counts[right]
            right += 1
        else:
            break
    return start, end


def _build_sample_windows(
    paragraphs: Sequence[str],
    sample_word_targets: Sequence[int] = DEFAULT_SAMPLE_WORD_TARGETS,
) -> list[SampleWindow]:
    if not paragraphs:
        return []
    paragraph_word_counts = _paragraph_word_counts(paragraphs)
    labels = ("opening", "middle", "late")
    fractions = (0.15, 0.5, 0.82)
    windows: list[SampleWindow] = []
    for index, label in enumerate(labels):
        target_words = sample_word_targets[index] if index < len(sample_word_targets) else sample_word_targets[-1]
        if label == "opening":
            start, end = _window_from_start(paragraphs, paragraph_word_counts, target_words)
        else:
            start, end = _window_from_center(
                paragraphs,
                paragraph_word_counts,
                target_words,
                fractions[index],
            )
        selected = paragraphs[start:end]
        windows.append(
            SampleWindow(
                label=label,
                paragraph_start=start + 1,
                paragraph_end=end,
                word_count=sum(paragraph_word_counts[start:end]),
                text="\n\n".join(selected),
            )
        )
    return windows


def _merged_window_text(windows: Sequence[SampleWindow], paragraphs: Sequence[str]) -> tuple[str, int]:
    if not windows:
        return "", 0
    paragraph_indexes: list[int] = []
    for window in windows:
        paragraph_indexes.extend(range(window.paragraph_start - 1, window.paragraph_end))
    unique_indexes = sorted(set(paragraph_indexes))
    selected = [paragraphs[index] for index in unique_indexes]
    merged = "\n\n".join(selected)
    return merged, len(_tokenize(merged))


def _extract_opening_anchor_density(text: str) -> float:
    opening_tokens = _tokenize(" ".join(text.split()[:120]))
    if not opening_tokens:
        return 0.0
    anchor_hits = sum(
        1
        for token in opening_tokens
        if token in TIME_WORDS or token in CONCRETE_NOUNS or token in ACTION_WORDS
    )
    return anchor_hits * 100.0 / len(opening_tokens)


def _count_concrete_openers(paragraphs: Sequence[str]) -> tuple[int, int]:
    concrete = 0
    abstract = 0
    for paragraph in paragraphs:
        opener = " ".join(paragraph.strip().split()[:18])
        opener_lower = opener.lower()
        tokens = _tokenize(opener)
        concrete_hits = 0
        if re.search(r"\b\d{2,4}\b", opener):
            concrete_hits += 1
        if any(token in TIME_WORDS for token in tokens):
            concrete_hits += 1
        if any(token in CONCRETE_NOUNS for token in tokens):
            concrete_hits += 1
        if any(token in ACTION_WORDS for token in tokens):
            concrete_hits += 1
        if CAPITALIZED_ENTITY_RE.search(opener):
            concrete_hits += 1
        if concrete_hits >= 2:
            concrete += 1
        if any(re.search(pattern, opener_lower, flags=re.IGNORECASE) for pattern in ABSTRACT_OPENER_PATTERNS):
            abstract += 1
    return concrete, abstract


def extract_auto_metrics(text: str, *, paragraphs: Sequence[str], opening_text: str) -> AutoMetrics:
    tokens = _tokenize(text)
    word_count = len(tokens)
    sentence_lengths = _sentence_lengths(text)
    sentence_count = len(sentence_lengths)
    paragraph_count = len(paragraphs)
    avg_sentence_length = sum(sentence_lengths) / sentence_count if sentence_count else 0.0
    short_sentence_share = (
        100.0 * sum(1 for length in sentence_lengths if length <= 8) / sentence_count if sentence_count else 0.0
    )
    long_sentence_share = (
        100.0 * sum(1 for length in sentence_lengths if length >= 35) / sentence_count if sentence_count else 0.0
    )
    fragment_share = (
        100.0 * sum(1 for length in sentence_lengths if length <= 5) / sentence_count if sentence_count else 0.0
    )
    question_density = _per_thousand(text.count("?"), word_count)
    first_person_density = _per_thousand(
        len(re.findall(r"\b(?:i|me|my|mine|myself|we|us|our|ours|let's|let us)\b", text, flags=re.IGNORECASE)),
        word_count,
    )
    direct_address_density = _per_thousand(
        len(re.findall(r"\b(?:you|your|yours)\b", text, flags=re.IGNORECASE)),
        word_count,
    )
    listener_guidance_density = _per_thousand(_count_pattern_hits(text, GUIDANCE_PATTERNS), word_count)
    callback_density = _per_thousand(_count_pattern_hits(text, CALLBACK_PATTERNS), word_count)
    analogy_density = _per_thousand(_count_pattern_hits(text, ANALOGY_PATTERNS), word_count)
    contrast_density = _per_thousand(_count_pattern_hits(text, CONTRAST_PATTERNS), word_count)
    translation_density = _per_thousand(_count_pattern_hits(text, TRANSLATION_PATTERNS), word_count)
    aside_density = _per_thousand(_count_pattern_hits(text, ASIDE_PATTERNS), word_count)
    contraction_density = _per_thousand(_count_pattern_hits(text, CONTRACTION_PATTERNS), word_count)
    meta_density = _per_thousand(_count_pattern_hits(text, META_PATTERNS), word_count)
    quote_density = _per_thousand(text.count('"') / 2.0, word_count)
    dash_density = _per_thousand(text.count("-") + text.count("--"), word_count)
    emphasis_density = _per_thousand(_count_pattern_hits(text, EMPHASIS_PATTERNS), word_count)

    concrete_openers, abstract_openers = _count_concrete_openers(paragraphs)
    concrete_opener_rate = 100.0 * concrete_openers / paragraph_count if paragraph_count else 0.0
    abstract_opener_rate = 100.0 * abstract_openers / paragraph_count if paragraph_count else 0.0
    opening_anchor_density = _extract_opening_anchor_density(opening_text)

    entity_count = len(CAPITALIZED_ENTITY_RE.findall(text))
    entity_load_per_200_words = entity_count * 200.0 / word_count if word_count else 0.0

    verdict_parallel_density = _per_thousand(_verdict_parallel_count(text), word_count)

    return AutoMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_sentence_length=_round_score(avg_sentence_length),
        short_sentence_share=_round_score(short_sentence_share),
        long_sentence_share=_round_score(long_sentence_share),
        fragment_share=_round_score(fragment_share),
        question_density=_round_score(question_density),
        first_person_density=_round_score(first_person_density),
        direct_address_density=_round_score(direct_address_density),
        listener_guidance_density=_round_score(listener_guidance_density),
        callback_density=_round_score(callback_density),
        analogy_density=_round_score(analogy_density),
        contrast_density=_round_score(contrast_density),
        translation_density=_round_score(translation_density),
        aside_density=_round_score(aside_density),
        contraction_density=_round_score(contraction_density),
        meta_density=_round_score(meta_density),
        quote_density=_round_score(quote_density),
        dash_density=_round_score(dash_density),
        emphasis_density=_round_score(emphasis_density),
        concrete_opener_rate=_round_score(concrete_opener_rate),
        abstract_opener_rate=_round_score(abstract_opener_rate),
        opening_anchor_density=_round_score(opening_anchor_density),
        entity_load_per_200_words=_round_score(entity_load_per_200_words),
        verdict_parallel_density=_round_score(verdict_parallel_density),
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _band_score(value: float, *, ideal_low: float, ideal_high: float, hard_low: float, hard_high: float) -> float:
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return _clamp((value - hard_low) / (ideal_low - hard_low)) if ideal_low > hard_low else 0.0
    return _clamp((hard_high - value) / (hard_high - ideal_high)) if hard_high > ideal_high else 0.0


def _range_score(value: float, *, low: float, high: float) -> float:
    return _clamp((value - low) / (high - low)) if high > low else 0.0


def _inverse_range_score(value: float, *, good: float, bad: float) -> float:
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return _clamp((bad - value) / (bad - good))


def _build_target_metrics(target_snippet: str) -> AutoMetrics:
    target_paragraphs = [part.strip() for part in target_snippet.split("\n\n") if part.strip()]
    return extract_auto_metrics(
        target_snippet,
        paragraphs=target_paragraphs,
        opening_text=target_snippet,
    )


def _afs_dimension_rating(key: str, metrics: AutoMetrics) -> float:
    if key == "host_presence":
        score = (
            0.45
            * _range_score(
                metrics.first_person_density + metrics.direct_address_density + metrics.meta_density,
                low=0.5,
                high=8.0,
            )
            + 0.35 * _range_score(metrics.contraction_density + metrics.aside_density, low=0.1, high=6.0)
            + 0.20 * _range_score(metrics.direct_address_density, low=0.1, high=4.0)
        )
    elif key == "conversational_cadence":
        score = (
            0.45
            * _band_score(
                metrics.avg_sentence_length,
                ideal_low=13.0,
                ideal_high=16.0,
                hard_low=10.0,
                hard_high=22.0,
            )
            + 0.25
            * _band_score(
                metrics.short_sentence_share,
                ideal_low=28.0,
                ideal_high=42.0,
                hard_low=15.0,
                hard_high=55.0,
            )
            + 0.20 * _inverse_range_score(metrics.long_sentence_share, good=6.0, bad=14.0)
            + 0.10
            * _band_score(
                metrics.fragment_share,
                ideal_low=2.0,
                ideal_high=18.0,
                hard_low=0.0,
                hard_high=35.0,
            )
        )
    elif key == "listener_guidance":
        score = (
            0.55 * _range_score(metrics.listener_guidance_density, low=0.2, high=4.0)
            + 0.25 * _range_score(metrics.callback_density, low=0.0, high=2.0)
            + 0.20 * _range_score(metrics.translation_density, low=0.0, high=2.0)
        )
    elif key == "scene_immediacy":
        score = (
            0.50 * _range_score(metrics.concrete_opener_rate, low=40.0, high=95.0)
            + 0.25 * _range_score(metrics.opening_anchor_density, low=3.0, high=15.0)
            + 0.25 * _inverse_range_score(metrics.abstract_opener_rate, good=5.0, bad=30.0)
        )
    elif key == "structural_signposting":
        score = (
            0.35 * _range_score(metrics.listener_guidance_density, low=0.2, high=4.0)
            + 0.25 * _range_score(metrics.callback_density, low=0.0, high=2.0)
            + 0.20 * _range_score(metrics.translation_density, low=0.0, high=2.0)
            + 0.20 * _range_score(metrics.contrast_density, low=0.0, high=3.0)
        )
    elif key == "rhetorical_intensity":
        score = (
            0.30 * _range_score(metrics.question_density, low=0.0, high=1.2)
            + 0.30 * _range_score(metrics.contrast_density, low=0.0, high=3.0)
            + 0.20 * _range_score(metrics.emphasis_density, low=0.0, high=4.0)
            + 0.20 * _range_score(metrics.dash_density + metrics.quote_density, low=0.1, high=6.0)
        )
    elif key == "associative_riffing":
        score = (
            0.40 * _range_score(metrics.analogy_density, low=0.0, high=5.0)
            + 0.25 * _range_score(metrics.aside_density, low=0.0, high=5.0)
            + 0.20 * _range_score(metrics.first_person_density, low=0.0, high=5.0)
            + 0.15 * _range_score(metrics.question_density, low=0.0, high=1.5)
        )
    elif key == "analogy_density":
        score = _range_score(metrics.analogy_density, low=0.0, high=5.0)
    elif key == "oral_redundancy":
        score = (
            0.35 * _range_score(metrics.listener_guidance_density, low=0.0, high=4.0)
            + 0.25 * _range_score(metrics.callback_density, low=0.0, high=2.0)
            + 0.20 * _range_score(metrics.translation_density, low=0.0, high=2.0)
            + 0.20 * _range_score(metrics.emphasis_density, low=0.0, high=4.0)
        )
    elif key == "spoken_looseness":
        score = (
            0.30 * _range_score(metrics.contraction_density, low=0.0, high=10.0)
            + 0.25 * _range_score(metrics.aside_density, low=0.0, high=5.0)
            + 0.20
            * _band_score(
                metrics.fragment_share,
                ideal_low=2.0,
                ideal_high=18.0,
                hard_low=0.0,
                hard_high=35.0,
            )
            + 0.15 * _range_score(metrics.first_person_density, low=0.0, high=5.0)
            + 0.10 * _range_score(metrics.dash_density, low=0.0, high=4.0)
        )
    elif key == "clarity_under_load":
        score = (
            0.30
            * _band_score(
                metrics.avg_sentence_length,
                ideal_low=13.0,
                ideal_high=16.0,
                hard_low=10.0,
                hard_high=22.0,
            )
            + 0.25 * _inverse_range_score(metrics.long_sentence_share, good=5.0, bad=12.0)
            + 0.20 * _inverse_range_score(metrics.entity_load_per_200_words, good=2.0, bad=12.0)
            + 0.15 * _range_score(metrics.listener_guidance_density, low=0.0, high=4.0)
            + 0.10
            * _band_score(
                metrics.short_sentence_share,
                ideal_low=25.0,
                ideal_high=40.0,
                hard_low=15.0,
                hard_high=55.0,
            )
        )
    elif key == "memorable_lines":
        score = (
            0.25 * _range_score(metrics.quote_density, low=0.0, high=4.0)
            + 0.25 * _range_score(metrics.emphasis_density, low=0.0, high=4.0)
            + 0.20 * _range_score(metrics.concrete_opener_rate, low=40.0, high=95.0)
            + 0.15
            * _band_score(
                metrics.short_sentence_share,
                ideal_low=28.0,
                ideal_high=42.0,
                hard_low=15.0,
                hard_high=55.0,
            )
            + 0.15 * _range_score(metrics.contrast_density, low=0.0, high=3.0)
        )
    else:  # pragma: no cover - defensive fallback
        score = 0.0
    return _round_score(score * 5.0)


def _tss_dimension_rating(key: str, metrics: AutoMetrics, target: AutoMetrics) -> float:
    del key, metrics, target
    return 0.0


def _rating_similarity(rating: float, target_rating: float) -> float:
    return _clamp(1.0 - (abs(rating - target_rating) / 5.0))


def _piecewise_calibrate(value: float, raw_anchors: Sequence[float], calibrated_anchors: Sequence[float]) -> float:
    if len(raw_anchors) != len(calibrated_anchors):
        raise ValueError("raw_anchors and calibrated_anchors must have the same length")
    if len(raw_anchors) < 2:
        raise ValueError("at least two anchors are required for calibration")

    if value <= raw_anchors[0]:
        raw_span = raw_anchors[1] - raw_anchors[0]
        cal_span = calibrated_anchors[1] - calibrated_anchors[0]
        slope = cal_span / raw_span if raw_span else 0.0
        return calibrated_anchors[0] + (value - raw_anchors[0]) * slope

    for index in range(1, len(raw_anchors)):
        left_raw = raw_anchors[index - 1]
        right_raw = raw_anchors[index]
        if value <= right_raw:
            left_cal = calibrated_anchors[index - 1]
            right_cal = calibrated_anchors[index]
            raw_span = right_raw - left_raw
            if raw_span == 0:
                return right_cal
            fraction = (value - left_raw) / raw_span
            return left_cal + fraction * (right_cal - left_cal)

    raw_span = raw_anchors[-1] - raw_anchors[-2]
    cal_span = calibrated_anchors[-1] - calibrated_anchors[-2]
    slope = cal_span / raw_span if raw_span else 0.0
    return calibrated_anchors[-1] + (value - raw_anchors[-1]) * slope


def _dimension_results(
    metrics: AutoMetrics,
    target: AutoMetrics,
    comparison_target: AutoMetrics | None = None,
) -> list[DimensionScore]:
    results: list[DimensionScore] = []
    for key, label, afs_weight, tss_weight in DIMENSION_SPECS:
        afs_rating = _afs_dimension_rating(key, metrics)
        tss_rating = afs_rating
        tss_v2_rating: float | None = None
        tss_v2_points: float | None = None
        if comparison_target is not None:
            target_rating = _afs_dimension_rating(key, comparison_target)
            tss_v2_rating = _round_score(_rating_similarity(afs_rating, target_rating) * 5.0)
            tss_v2_points = _round_score((tss_v2_rating / 5.0) * tss_weight)
        results.append(
            DimensionScore(
                key=key,
                label=label,
                afs_weight=afs_weight,
                tss_weight=tss_weight,
                afs_rating=afs_rating,
                afs_points=_round_score((afs_rating / 5.0) * afs_weight),
                tss_rating=tss_rating,
                tss_points=_round_score((tss_rating / 5.0) * tss_weight),
                tss_v2_rating=tss_v2_rating,
                tss_v2_points=tss_v2_points,
                help_text=DIMENSION_HELP[key],
            )
        )
    return results


def _pick_best_sentence(text: str, patterns: Iterable[str]) -> str:
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text.strip()) if segment.strip()]
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        score = _count_pattern_hits(sentence, patterns)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def _pick_memorable_line(text: str) -> str:
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text.strip()) if segment.strip()]
    best_sentence = ""
    best_score = -1.0
    for sentence in sentences:
        words = len(_tokenize(sentence))
        if words == 0:
            continue
        score = (
            _count_pattern_hits(sentence, EMPHASIS_PATTERNS)
            + _count_pattern_hits(sentence, CONTRAST_PATTERNS)
            + _count_pattern_hits(sentence, ANALOGY_PATTERNS)
            + (1.0 if words <= 14 else 0.0)
            + (0.5 if '"' in sentence else 0.0)
        )
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def _pick_page_bound_paragraph(paragraphs: Sequence[str]) -> str:
    worst_paragraph = ""
    worst_score = -1.0
    for paragraph in paragraphs:
        sentence_lengths = _sentence_lengths(paragraph)
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        score = (
            avg_sentence_length
            + _count_pattern_hits(paragraph, ABSTRACT_OPENER_PATTERNS) * 4.0
            + len(CAPITALIZED_ENTITY_RE.findall(paragraph)) * 0.1
        )
        if score > worst_score:
            worst_score = score
            worst_paragraph = paragraph
    return worst_paragraph


def _build_episode_report(
    script_path: Path,
    *,
    target_metrics: AutoMetrics,
    comparison_target_metrics: AutoMetrics | None,
    sample_word_targets: Sequence[int],
) -> EpisodeReport:
    payload = _load_json(script_path)
    paragraphs = _paragraphs_from_script(payload)
    sample_windows = _build_sample_windows(paragraphs, sample_word_targets)
    sample_text, sample_word_count = _merged_window_text(sample_windows, paragraphs)
    opening_text = sample_windows[0].text if sample_windows else sample_text
    metrics = extract_auto_metrics(sample_text, paragraphs=paragraphs, opening_text=opening_text)
    dimensions = _dimension_results(metrics, target_metrics, comparison_target_metrics)
    raw_afs = _round_score(sum(dimension.afs_points for dimension in dimensions))
    raw_tss = _round_score(sum(dimension.tss_points for dimension in dimensions))
    afs = _round_score(_piecewise_calibrate(raw_afs, AFS_RAW_ANCHORS, AFS_CALIBRATED_ANCHORS))
    tss = _round_score(_piecewise_calibrate(raw_tss, TSS_RAW_ANCHORS, TSS_CALIBRATED_ANCHORS))
    tss_v2 = None
    if comparison_target_metrics is not None:
        tss_v2 = _round_score(
            sum(dimension.tss_v2_points or 0.0 for dimension in dimensions)
        )
    title = payload.get("title") if isinstance(payload.get("title"), str) else script_path.parent.name
    episode_number = payload.get("episode_number") if isinstance(payload.get("episode_number"), int) else int(script_path.parent.name)
    evidence = {
        "best_opening_paragraph": paragraphs[0] if paragraphs else "",
        "best_listener_guidance_line": _pick_best_sentence(sample_text, GUIDANCE_PATTERNS),
        "best_memorable_line": _pick_memorable_line(sample_text),
        "most_page_bound_paragraph": _pick_page_bound_paragraph(paragraphs),
    }
    return EpisodeReport(
        episode_number=episode_number,
        title=title,
        script_artifact=script_path.name,
        source_path=str(script_path),
        total_word_count=sum(_paragraph_word_counts(paragraphs)),
        sample_word_count=sample_word_count,
        raw_afs=raw_afs,
        afs=afs,
        raw_tss=raw_tss,
        tss=tss,
        tss_v2=tss_v2,
        dimensions=dimensions,
        sample_windows=sample_windows,
        metrics=metrics,
        evidence=evidence,
    )


def _weighted_average(values: Sequence[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def _combine_metrics(metrics_list: Sequence[AutoMetrics]) -> AutoMetrics:
    if not metrics_list:
        return extract_auto_metrics("", paragraphs=[], opening_text="")

    def weighted(attribute: str, weight_attr: str = "word_count") -> float:
        pairs = [(getattr(metric, attribute), getattr(metric, weight_attr)) for metric in metrics_list]
        return _weighted_average(pairs)

    word_count = sum(metric.word_count for metric in metrics_list)
    sentence_count = sum(metric.sentence_count for metric in metrics_list)
    paragraph_count = sum(metric.paragraph_count for metric in metrics_list)
    return AutoMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_sentence_length=_round_score(weighted("avg_sentence_length", "sentence_count")),
        short_sentence_share=_round_score(weighted("short_sentence_share", "sentence_count")),
        long_sentence_share=_round_score(weighted("long_sentence_share", "sentence_count")),
        fragment_share=_round_score(weighted("fragment_share", "sentence_count")),
        question_density=_round_score(weighted("question_density")),
        first_person_density=_round_score(weighted("first_person_density")),
        direct_address_density=_round_score(weighted("direct_address_density")),
        listener_guidance_density=_round_score(weighted("listener_guidance_density")),
        callback_density=_round_score(weighted("callback_density")),
        analogy_density=_round_score(weighted("analogy_density")),
        contrast_density=_round_score(weighted("contrast_density")),
        translation_density=_round_score(weighted("translation_density")),
        aside_density=_round_score(weighted("aside_density")),
        contraction_density=_round_score(weighted("contraction_density")),
        meta_density=_round_score(weighted("meta_density")),
        quote_density=_round_score(weighted("quote_density")),
        dash_density=_round_score(weighted("dash_density")),
        emphasis_density=_round_score(weighted("emphasis_density")),
        concrete_opener_rate=_round_score(weighted("concrete_opener_rate", "paragraph_count")),
        abstract_opener_rate=_round_score(weighted("abstract_opener_rate", "paragraph_count")),
        opening_anchor_density=_round_score(weighted("opening_anchor_density")),
        entity_load_per_200_words=_round_score(weighted("entity_load_per_200_words")),
        verdict_parallel_density=_round_score(weighted("verdict_parallel_density")),
    )


def _combine_dimensions(dimensions_list: Sequence[list[DimensionScore]]) -> list[DimensionScore]:
    if not dimensions_list:
        return []
    results: list[DimensionScore] = []
    for index, spec in enumerate(DIMENSION_SPECS):
        key, label, afs_weight, tss_weight = spec
        items = [dimensions[index] for dimensions in dimensions_list]
        afs_rating = _weighted_average([(item.afs_rating, item.afs_weight) for item in items])
        tss_rating = _weighted_average([(item.tss_rating, item.tss_weight) for item in items])
        tss_v2_items = [(item.tss_v2_rating, item.tss_weight) for item in items if item.tss_v2_rating is not None]
        tss_v2_rating = _weighted_average(tss_v2_items) if tss_v2_items else None
        results.append(
            DimensionScore(
                key=key,
                label=label,
                afs_weight=afs_weight,
                tss_weight=tss_weight,
                afs_rating=_round_score(afs_rating),
                afs_points=_round_score((afs_rating / 5.0) * afs_weight),
                tss_rating=_round_score(tss_rating),
                tss_points=_round_score((tss_rating / 5.0) * tss_weight),
                tss_v2_rating=_round_score(tss_v2_rating) if tss_v2_rating is not None else None,
                tss_v2_points=(
                    _round_score((tss_v2_rating / 5.0) * tss_weight) if tss_v2_rating is not None else None
                ),
                help_text=DIMENSION_HELP[key],
            )
        )
    return results


def build_audio_style_payload(
    run_dirs: Sequence[Path],
    *,
    title: str,
    target_snippet: str | None = None,
    comparison_target_snippet: str | None = None,
    comparison_target_label: str | None = None,
    script_artifacts: Sequence[str] = DEFAULT_SCRIPT_ARTIFACTS,
    sample_word_targets: Sequence[int] = DEFAULT_SAMPLE_WORD_TARGETS,
) -> dict[str, object]:
    target_text = (target_snippet or DEFAULT_TARGET_SNIPPET).strip()
    target_metrics = _build_target_metrics(target_text)
    comparison_target_text = (comparison_target_snippet or DEFAULT_TSS_V2_SNIPPET).strip()
    comparison_target_metrics = _build_target_metrics(comparison_target_text)
    run_reports: list[RunReport] = []

    for run_dir in run_dirs:
        resolved_run = run_dir.resolve()
        script_paths = _episode_script_paths(resolved_run, script_artifacts)
        if not script_paths:
            raise ValueError(
                f"No script artifacts matching {list(script_artifacts)} found under {resolved_run / 'episodes'}"
            )
        episodes = [
            _build_episode_report(
                script_path,
                target_metrics=target_metrics,
                comparison_target_metrics=comparison_target_metrics,
                sample_word_targets=sample_word_targets,
            )
            for script_path in script_paths
        ]
        run_dimensions = _combine_dimensions([episode.dimensions for episode in episodes])
        run_metrics = _combine_metrics([episode.metrics for episode in episodes])
        total_word_count = sum(episode.total_word_count for episode in episodes)
        sampled_word_count = sum(episode.sample_word_count for episode in episodes)
        raw_afs = _round_score(
            _weighted_average([(episode.raw_afs, max(episode.sample_word_count, 1)) for episode in episodes])
        )
        raw_tss = _round_score(
            _weighted_average([(episode.raw_tss, max(episode.sample_word_count, 1)) for episode in episodes])
        )
        afs = _round_score(_piecewise_calibrate(raw_afs, AFS_RAW_ANCHORS, AFS_CALIBRATED_ANCHORS))
        tss = _round_score(_piecewise_calibrate(raw_tss, TSS_RAW_ANCHORS, TSS_CALIBRATED_ANCHORS))
        tss_v2 = None
        tss_v2 = _round_score(
            _weighted_average([(episode.tss_v2 or 0.0, max(episode.sample_word_count, 1)) for episode in episodes])
        )
        run_reports.append(
            RunReport(
                run_id=resolved_run.name,
                run_dir=str(resolved_run),
                episode_count=len(episodes),
                total_word_count=total_word_count,
                sampled_word_count=sampled_word_count,
                raw_afs=raw_afs,
                afs=afs,
                raw_tss=raw_tss,
                tss=tss,
                tss_v2=tss_v2,
                dimensions=run_dimensions,
                metrics=run_metrics,
                episodes=episodes,
            )
        )

    ranked_runs = sorted(run_reports, key=lambda report: (-report.afs, -report.tss, report.run_id))
    summary = {
        "run_count": len(run_reports),
        "episode_count": sum(report.episode_count for report in run_reports),
        "mean_afs": _round_score(sum(report.afs for report in run_reports) / len(run_reports)),
        "mean_tss": _round_score(sum(report.tss for report in run_reports) / len(run_reports)),
        "mean_tss_v2": _round_score(sum(report.tss_v2 or 0.0 for report in run_reports) / len(run_reports)),
        "winner_by_afs": ranked_runs[0].run_id if ranked_runs else None,
        "winner_by_tss": sorted(run_reports, key=lambda report: (-report.tss, -report.afs, report.run_id))[0].run_id if run_reports else None,
        "winner_by_tss_v2": sorted(run_reports, key=lambda report: (-(report.tss_v2 or 0.0), -report.afs, report.run_id))[0].run_id if run_reports else None,
    }
    result = {
        "title": title,
        "script_artifacts": list(script_artifacts),
        "sample_word_targets": list(sample_word_targets),
        "summary": summary,
        "target": {
            "label": "Default spoken-host target snippet" if target_snippet is None else "Custom target snippet",
            "word_count": target_metrics.word_count,
            "metrics": target_metrics.to_dict(),
            "text": target_text,
        },
        "runs": [report.to_dict() for report in run_reports],
    }
    result["comparison_target"] = {
        "label": comparison_target_label or DEFAULT_TSS_V2_LABEL,
        "word_count": comparison_target_metrics.word_count,
        "metrics": comparison_target_metrics.to_dict(),
        "text": comparison_target_text,
    }
    return result


def render_audio_style_html(payload: dict[str, object]) -> str:
    title = html.escape(str(payload["title"]))
    summary = payload["summary"]
    runs = payload["runs"]
    comparison_target = payload.get("comparison_target")
    has_tss_v2 = isinstance(comparison_target, dict)

    def esc(value: object) -> str:
        return html.escape(str(value))

    def slugify(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "run"

    def score_badge(label: str, value: float | None, tone: str) -> str:
        if value is None:
            return ""
        return (
            f'<div class="score-badge {tone}"><span class="badge-label">{esc(label)}</span>'
            f'<strong>{value:.2f}</strong></div>'
        )

    def stat_chip(label: str, value: str) -> str:
        return f'<div class="stat-chip"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'

    def metric_chip(label: str, value: str) -> str:
        return f'<div class="metric-chip"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'

    def winner_tags(run_id: str) -> str:
        tags: list[str] = []
        if summary.get("winner_by_afs") == run_id:
            tags.append('<span class="winner-tag afs">AFS winner</span>')
        if summary.get("winner_by_tss") == run_id:
            tags.append('<span class="winner-tag tss">TSS winner</span>')
        if has_tss_v2 and summary.get("winner_by_tss_v2") == run_id:
            tags.append('<span class="winner-tag tssv2">TSSv2 winner</span>')
        return "".join(tags)

    def score_bar(label: str, value: float | None, tone: str) -> str:
        if value is None:
            return (
                f'<div class="score-bar {tone}"><span class="bar-label">{esc(label)}</span>'
                '<div class="track empty"></div><span class="bar-value">-</span></div>'
            )
        width = max(0.0, min(100.0, (value / 5.0) * 100.0))
        return (
            f'<div class="score-bar {tone}"><span class="bar-label">{esc(label)}</span>'
            f'<div class="track"><span style="width:{width:.1f}%"></span></div>'
            f'<span class="bar-value">{value:.2f}</span></div>'
        )

    sorted_runs = sorted(runs, key=lambda run: (-run["afs"], -run["tss"], str(run["run_id"])))
    run_rank = {str(run["run_id"]): index + 1 for index, run in enumerate(sorted_runs)}

    run_nav = "".join(
        f'<a class="run-nav-link" href="#run-{slugify(run["run_id"])}">'
        f'<span>{esc(run["run_id"])}</span><strong>{run["afs"]:.2f}</strong></a>'
        for run in sorted_runs
    )

    leaderboard_rows: list[str] = []
    for run in sorted_runs:
        run_id = str(run["run_id"])
        tss_v2_cell = f"<td>{run['tss_v2']:.2f}</td>" if has_tss_v2 and run["tss_v2"] is not None else ""
        leaderboard_rows.append(
            f"""
            <tr>
              <td class="rank-cell">{run_rank[run_id]}</td>
              <td>
                <div class="run-cell">
                  <a href="#run-{slugify(run_id)}">{esc(run_id)}</a>
                  <div class="winner-tag-row">{winner_tags(run_id)}</div>
                </div>
              </td>
              <td>{run['afs']:.2f}</td>
              <td>{run['tss']:.2f}</td>
              {tss_v2_cell}
              <td>{run['sampled_word_count']}</td>
            </tr>
            """
        )

    run_sections: list[str] = []
    for run in sorted_runs:
        run_id = str(run["run_id"])
        run_slug = slugify(run_id)
        score_badges = [
            score_badge("AFS", run["afs"], "afs"),
            score_badge("TSS", run["tss"], "tss"),
        ]
        if has_tss_v2 and run["tss_v2"] is not None:
            score_badges.append(score_badge("TSSv2", run["tss_v2"], "tssv2"))

        summary_chips = [
            stat_chip("Rank by AFS", f"#{run_rank[run_id]}"),
            stat_chip("Episodes", str(run["episode_count"])),
            stat_chip("Sampled words", str(run["sampled_word_count"])),
            stat_chip("Raw AFS/TSS", f"{run['raw_afs']:.2f} / {run['raw_tss']:.2f}"),
        ]
        if has_tss_v2 and run["tss_v2"] is not None:
            summary_chips.append(stat_chip("TSSv2", f"{run['tss_v2']:.2f}"))

        dimension_rows: list[str] = []
        technical_rows: list[str] = []
        for dimension in run["dimensions"]:
            dimension_rows.append(
                f"""
                <div class="dimension-row">
                  <div class="dimension-copy">
                    <h4>{esc(dimension['label'])}</h4>
                    <p>{esc(dimension['help_text'])}</p>
                  </div>
                  <div class="dimension-bars">
                    {score_bar("AFS", dimension['afs_rating'], "afs")}
                    {score_bar("TSS", dimension['tss_rating'], "tss")}
                    {score_bar("TSSv2", dimension.get('tss_v2_rating'), "tssv2") if has_tss_v2 else ""}
                  </div>
                </div>
                """
            )
            tss_v2_cells = (
                f"<td>{dimension['tss_v2_rating']:.2f} / 5</td><td>{dimension['tss_v2_points']:.2f}</td>"
                if has_tss_v2 and dimension.get("tss_v2_rating") is not None and dimension.get("tss_v2_points") is not None
                else ""
            )
            technical_rows.append(
                f"""
                <tr>
                  <th>{esc(dimension['label'])}</th>
                  <td>{dimension['afs_rating']:.2f} / 5</td>
                  <td>{dimension['afs_points']:.2f}</td>
                  <td>{dimension['tss_rating']:.2f} / 5</td>
                  <td>{dimension['tss_points']:.2f}</td>
                  {tss_v2_cells}
                </tr>
                """
            )

        metrics = run["metrics"]
        metric_chips_html = "".join(
            [
                metric_chip("Avg sentence", str(metrics["avg_sentence_length"])),
                metric_chip("Short sentences", f'{metrics["short_sentence_share"]}%'),
                metric_chip("Long sentences", f'{metrics["long_sentence_share"]}%'),
                metric_chip("Guidance density", f'{metrics["listener_guidance_density"]} / 1k'),
                metric_chip("Question density", f'{metrics["question_density"]} / 1k'),
                metric_chip("First-person density", f'{metrics["first_person_density"]} / 1k'),
                metric_chip("Analogy density", f'{metrics["analogy_density"]} / 1k'),
                metric_chip("Concrete openers", f'{metrics["concrete_opener_rate"]}%'),
                metric_chip("Abstract openers", f'{metrics["abstract_opener_rate"]}%'),
                metric_chip("Entity load", f'{metrics["entity_load_per_200_words"]} / 200w'),
            ]
        )

        episode_cards: list[str] = []
        for episode in run["episodes"]:
            episode_score_badges = [
                score_badge("AFS", episode["afs"], "afs"),
                score_badge("TSS", episode["tss"], "tss"),
            ]
            if has_tss_v2 and episode["tss_v2"] is not None:
                episode_score_badges.append(score_badge("TSSv2", episode["tss_v2"], "tssv2"))
            episode_meta = [
                stat_chip("Artifact", episode["script_artifact"]),
                stat_chip("Total words", str(episode["total_word_count"])),
                stat_chip("Sampled words", str(episode["sample_word_count"])),
                stat_chip("Raw AFS/TSS", f'{episode["raw_afs"]:.2f} / {episode["raw_tss"]:.2f}'),
            ]
            if has_tss_v2 and episode["tss_v2"] is not None:
                episode_meta.append(stat_chip("TSSv2", f'{episode["tss_v2"]:.2f}'))

            evidence_cards = [
                (
                    "Opening paragraph",
                    episode["evidence"].get("best_opening_paragraph", ""),
                ),
                (
                    "Best guidance line",
                    episode["evidence"].get("best_listener_guidance_line", ""),
                ),
                (
                    "Most memorable line",
                    episode["evidence"].get("best_memorable_line", ""),
                ),
            ]
            evidence_html = "".join(
                f"""
                <article class="quote-card">
                  <div class="card-eyebrow">{esc(label)}</div>
                  <blockquote>{esc(text)}</blockquote>
                </article>
                """
                for label, text in evidence_cards
                if text
            )
            window_cards = "".join(
                f"""
                <details class="window-card">
                  <summary>{esc(window['label'].title())} sample <span>{window['word_count']} words</span></summary>
                  <pre>{esc(window['text'])}</pre>
                </details>
                """
                for window in episode["sample_windows"]
            )
            episode_metric_strip = "".join(
                [
                    metric_chip("Avg sentence", str(episode["metrics"]["avg_sentence_length"])),
                    metric_chip("Short", f'{episode["metrics"]["short_sentence_share"]}%'),
                    metric_chip("Long", f'{episode["metrics"]["long_sentence_share"]}%'),
                    metric_chip("Guidance", f'{episode["metrics"]["listener_guidance_density"]} / 1k'),
                    metric_chip("Analogy", f'{episode["metrics"]["analogy_density"]} / 1k'),
                ]
            )
            episode_cards.append(
                f"""
                <details class="episode-card">
                  <summary>
                    <div>
                      <div class="episode-kicker">Episode {episode['episode_number']}</div>
                      <h3>{esc(episode['title'])}</h3>
                    </div>
                    <div class="episode-score-row">{''.join(episode_score_badges)}</div>
                  </summary>
                  <div class="episode-body">
                    <div class="chip-grid compact">{''.join(episode_meta)}</div>
                    <div class="episode-columns">
                      <div class="evidence-stack">{evidence_html}</div>
                      <div class="support-stack">
                        <div class="metric-chip-grid compact">{episode_metric_strip}</div>
                        <details class="window-card">
                          <summary>Most page-bound paragraph</summary>
                          <pre>{esc(episode['evidence']['most_page_bound_paragraph'])}</pre>
                        </details>
                      </div>
                    </div>
                    <div class="window-stack">{window_cards}</div>
                  </div>
                </details>
                """
            )

        technical_head = (
            "<th>TSSv2 rating</th><th>TSSv2 points</th>" if has_tss_v2 else ""
        )
        run_sections.append(
            f"""
            <section id="run-{run_slug}" class="run-section panel">
              <header class="run-header">
                <div class="run-title-block">
                  <div class="section-kicker">Run {run_rank[run_id]}</div>
                  <h2>{esc(run_id)}</h2>
                  <div class="winner-tag-row">{winner_tags(run_id)}</div>
                </div>
                <div class="score-badge-row">{''.join(score_badges)}</div>
              </header>
              <div class="chip-grid">{''.join(summary_chips)}</div>
              <div class="run-grid">
                <section class="subpanel">
                  <div class="subpanel-head">
                    <div>
                      <div class="card-eyebrow">Dimension Breakdown</div>
                      <h3>How the run scores by rubric dimension</h3>
                    </div>
                  </div>
                  <div class="dimension-list">{''.join(dimension_rows)}</div>
                  <details class="technical-card">
                    <summary>Show technical dimension table</summary>
                    <table>
                      <thead>
                        <tr>
                          <th>Dimension</th>
                          <th>AFS rating</th>
                          <th>AFS points</th>
                          <th>TSS rating</th>
                          <th>TSS points</th>
                          {technical_head}
                        </tr>
                      </thead>
                      <tbody>
                        {''.join(technical_rows)}
                      </tbody>
                    </table>
                  </details>
                </section>
                <section class="subpanel">
                  <div class="subpanel-head">
                    <div>
                      <div class="card-eyebrow">Auto Metrics</div>
                      <h3>Fast quantitative readout</h3>
                    </div>
                  </div>
                  <div class="metric-chip-grid">{metric_chips_html}</div>
                </section>
              </div>
              <section class="subpanel episode-panel">
                <div class="subpanel-head">
                  <div>
                    <div class="card-eyebrow">Episodes</div>
                    <h3>Evidence and sampled windows</h3>
                  </div>
                </div>
                <div class="episode-list">{''.join(episode_cards)}</div>
              </section>
            </section>
            """
        )

    rubric_rows: list[str] = []
    for key, label, afs_weight, tss_weight in DIMENSION_SPECS:
        rubric_rows.append(
            f"""
            <tr>
              <th>{esc(label)}</th>
              <td>{afs_weight}</td>
              <td>{tss_weight}</td>
              {"<td>" + str(tss_weight) + "</td>" if has_tss_v2 else ""}
              <td>{esc(DIMENSION_HELP[key])}</td>
            </tr>
            """
        )

    winner_mean_line = f"Mean AFS {summary['mean_afs']:.2f} | Mean TSS {summary['mean_tss']:.2f}"
    if has_tss_v2 and summary.get("mean_tss_v2") is not None:
        winner_mean_line += f" | Mean TSSv2 {summary['mean_tss_v2']:.2f}"

    comparison_sample_note = (
        "TSSv2 compares each script to the bundled comparison sample, or to your CLI override if provided."
        if has_tss_v2
        else ""
    )
    rubric_intro = (
        "AFS measures audio fitness. TSS measures similarity to the spoken-host anchor scale. "
        f"{comparison_sample_note}".strip()
    )
    calibration_note = (
        "The large headline scores are calibrated onto the editorial anchor scale. "
        "Raw rubric totals are shown alongside them for transparency."
    )
    if has_tss_v2:
        calibration_note += " TSSv2 is a direct 0-100 weighted similarity score and is not anchor-calibrated."

    comparison_section = ""
    if has_tss_v2 and isinstance(comparison_target, dict):
        comparison_section = (
            f"""
            <section class="sidebar-section">
              <div class="card-eyebrow">Comparison Sample</div>
              <h3>{esc(comparison_target['label'])}</h3>
              <p class="small">{comparison_target['word_count']} words</p>
              <details class="sidebar-detail">
                <summary>Show comparison sample</summary>
                <pre>{esc(comparison_target['text'])}</pre>
              </details>
            </section>
            """
        )

    artifact_tags = "".join(
        f'<span class="nav-tag">{esc(artifact)}</span>' for artifact in payload["script_artifacts"]
    )
    window_targets = " / ".join(str(target) for target in payload["sample_word_targets"])
    leaderboard_tssv2_header = "<th>TSSv2</th>" if has_tss_v2 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #f3efe8;
        --surface: #fffdfa;
        --panel: #fbf8f1;
        --panel-strong: #f5efe4;
        --line: #ddd4c7;
        --line-strong: #cdbfae;
        --ink: #191714;
        --muted: #6a635c;
        --navy: #153a59;
        --navy-soft: #e7eef5;
        --rust: #9a5a2f;
        --rust-soft: #f5e6db;
        --olive: #215d41;
        --olive-soft: #e6f0ea;
        --gold: #8f6b2a;
        --gold-soft: #f3ecd9;
        --shadow: 0 18px 40px rgba(36, 25, 14, 0.08);
      }}
      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: smooth; }}
      body {{
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(21, 58, 89, 0.08), transparent 28%),
          radial-gradient(circle at top right, rgba(154, 90, 47, 0.08), transparent 26%),
          linear-gradient(180deg, #f8f4ed 0%, var(--bg) 38%, #efe7da 100%);
        font-family: "Avenir Next", "Aptos", "Segoe UI Variable", "Helvetica Neue", sans-serif;
        line-height: 1.55;
      }}
      h1, h2, h3, h4 {{
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        letter-spacing: -0.02em;
      }}
      a {{ color: inherit; }}
      .shell {{
        width: min(1440px, calc(100vw - 32px));
        margin: 24px auto 48px;
      }}
      .panel {{
        background: rgba(255, 253, 248, 0.92);
        border: 1px solid var(--line);
        border-radius: 26px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(6px);
      }}
      .hero {{
        padding: 34px 34px 30px;
        background:
          linear-gradient(135deg, rgba(21, 58, 89, 0.08), rgba(21, 58, 89, 0.00) 48%),
          linear-gradient(180deg, rgba(154, 90, 47, 0.09), rgba(154, 90, 47, 0.02));
      }}
      .hero-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
        gap: 22px;
      }}
      h1 {{
        margin: 0 0 10px;
        font-size: clamp(2.6rem, 4.8vw, 4.2rem);
        line-height: 0.98;
      }}
      h2 {{
        margin: 0;
        font-size: clamp(1.6rem, 2vw, 2.2rem);
      }}
      h3 {{
        margin: 0;
        font-size: 1.18rem;
      }}
      h4 {{
        margin: 0;
        font-size: 1rem;
      }}
      p {{ margin: 0; }}
      .lede {{
        max-width: 64ch;
        color: var(--muted);
        font-size: 1.02rem;
      }}
      .section-kicker,
      .card-eyebrow {{
        color: var(--rust);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        font-weight: 700;
      }}
      .hero-meta {{
        display: flex;
        flex-direction: column;
        gap: 14px;
      }}
      .hero-card {{
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(255, 253, 248, 0.8);
        border: 1px solid var(--line);
      }}
      .hero-card p {{
        color: var(--muted);
      }}
      .run-nav {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }}
      .run-nav-link {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 253, 248, 0.9);
        text-decoration: none;
      }}
      .run-nav-link strong {{
        color: var(--navy);
        font-size: 0.95rem;
      }}
      .nav-tag-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
      }}
      .nav-tag {{
        display: inline-flex;
        align-items: center;
        padding: 7px 10px;
        border-radius: 999px;
        background: var(--panel-strong);
        border: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.86rem;
      }}
      .overview-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
        gap: 18px;
        margin-top: 18px;
      }}
      .overview-card,
      .sidebar,
      .subpanel {{
        padding: 24px;
      }}
      .workspace {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 340px;
        gap: 18px;
        margin-top: 18px;
      }}
      .main-stack {{
        display: flex;
        flex-direction: column;
        gap: 18px;
      }}
      .sidebar {{
        align-self: start;
        position: sticky;
        top: 18px;
        display: flex;
        flex-direction: column;
        gap: 18px;
      }}
      .sidebar-section {{
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}
      .small {{
        color: var(--muted);
        font-size: 0.93rem;
      }}
      .small + .small {{
        margin-top: 10px;
      }}
      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-top: 18px;
      }}
      .summary-card {{
        padding: 16px 18px;
        border-radius: 18px;
        background: var(--panel);
        border: 1px solid var(--line);
      }}
      .summary-card strong {{
        display: block;
        margin-top: 6px;
        font-size: 1.45rem;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
      }}
      thead th {{
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      th, td {{
        padding: 11px 10px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}
      tbody tr:last-child th,
      tbody tr:last-child td {{
        border-bottom: 0;
      }}
      .rank-cell {{
        width: 40px;
        color: var(--muted);
        font-variant-numeric: tabular-nums;
      }}
      .run-cell {{
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      .run-cell a {{
        font-weight: 700;
        text-decoration: none;
      }}
      .winner-tag-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }}
      .winner-tag {{
        display: inline-flex;
        align-items: center;
        padding: 5px 8px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
      }}
      .winner-tag.afs {{
        background: var(--olive-soft);
        color: var(--olive);
      }}
      .winner-tag.tss {{
        background: var(--navy-soft);
        color: var(--navy);
      }}
      .winner-tag.tssv2 {{
        background: var(--gold-soft);
        color: var(--gold);
      }}
      .run-section {{
        padding: 28px;
      }}
      .run-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        flex-wrap: wrap;
      }}
      .run-title-block {{
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}
      .score-badge-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }}
      .score-badge {{
        min-width: 118px;
        padding: 12px 14px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: var(--surface);
      }}
      .score-badge.afs {{
        border-color: rgba(33, 93, 65, 0.18);
        background: linear-gradient(180deg, rgba(33, 93, 65, 0.08), rgba(33, 93, 65, 0.02));
      }}
      .score-badge.tss {{
        border-color: rgba(21, 58, 89, 0.16);
        background: linear-gradient(180deg, rgba(21, 58, 89, 0.08), rgba(21, 58, 89, 0.02));
      }}
      .score-badge.tssv2 {{
        border-color: rgba(143, 107, 42, 0.18);
        background: linear-gradient(180deg, rgba(143, 107, 42, 0.10), rgba(143, 107, 42, 0.02));
      }}
      .badge-label {{
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .score-badge strong {{
        display: block;
        margin-top: 8px;
        font-size: 1.7rem;
        line-height: 1;
      }}
      .chip-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
        margin-top: 20px;
      }}
      .chip-grid.compact {{
        margin-top: 0;
      }}
      .stat-chip,
      .metric-chip {{
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel);
        border: 1px solid var(--line);
      }}
      .stat-chip span,
      .metric-chip span {{
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .stat-chip strong,
      .metric-chip strong {{
        font-size: 1rem;
      }}
      .run-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
        gap: 18px;
        margin-top: 18px;
      }}
      .subpanel {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
      }}
      .subpanel-head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 18px;
      }}
      .dimension-list {{
        display: flex;
        flex-direction: column;
        gap: 14px;
      }}
      .dimension-row {{
        display: grid;
        grid-template-columns: minmax(220px, 0.95fr) minmax(0, 1.05fr);
        gap: 18px;
        padding: 16px 0;
        border-top: 1px solid var(--line);
      }}
      .dimension-row:first-child {{
        border-top: 0;
        padding-top: 0;
      }}
      .dimension-copy {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .dimension-copy p {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .dimension-bars {{
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}
      .score-bar {{
        display: grid;
        grid-template-columns: 52px minmax(0, 1fr) 42px;
        align-items: center;
        gap: 10px;
      }}
      .bar-label {{
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .track {{
        height: 10px;
        border-radius: 999px;
        background: #ece6dd;
        overflow: hidden;
      }}
      .track.empty {{
        opacity: 0.55;
      }}
      .track span {{
        display: block;
        height: 100%;
        border-radius: inherit;
      }}
      .score-bar.afs .track span {{
        background: linear-gradient(90deg, #215d41, #5db890);
      }}
      .score-bar.tss .track span {{
        background: linear-gradient(90deg, #153a59, #5b8bb7);
      }}
      .score-bar.tssv2 .track span {{
        background: linear-gradient(90deg, #8f6b2a, #d1a24b);
      }}
      .bar-value {{
        text-align: right;
        font-variant-numeric: tabular-nums;
      }}
      .metric-chip-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
      }}
      .metric-chip-grid.compact {{
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      }}
      .technical-card,
      .sidebar-detail,
      .window-card,
      .episode-card {{
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
      }}
      details + details {{
        margin-top: 12px;
      }}
      summary {{
        cursor: pointer;
        list-style: none;
      }}
      .technical-card summary,
      .sidebar-detail summary,
      .window-card summary {{
        padding: 14px 16px;
        font-weight: 700;
      }}
      .episode-panel {{
        margin-top: 18px;
      }}
      .episode-list {{
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      .episode-card summary {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 18px 20px;
      }}
      .episode-kicker {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.74rem;
        margin-bottom: 6px;
      }}
      .episode-card h3 {{
        margin: 0;
        font-size: 1.18rem;
      }}
      .episode-score-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }}
      .episode-score-row .score-badge {{
        min-width: 92px;
        padding: 10px 12px;
      }}
      .episode-score-row .score-badge strong {{
        font-size: 1.15rem;
      }}
      .episode-body {{
        padding: 0 20px 20px;
      }}
      .episode-columns {{
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(260px, 0.85fr);
        gap: 14px;
        margin-top: 14px;
      }}
      .evidence-stack,
      .support-stack,
      .window-stack {{
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      .quote-card {{
        padding: 16px;
        border-radius: 18px;
        background: var(--surface);
        border: 1px solid var(--line);
      }}
      blockquote {{
        margin: 10px 0 0;
        padding-left: 14px;
        border-left: 3px solid var(--line-strong);
        color: var(--ink);
      }}
      pre {{
        margin: 0;
        padding: 0 16px 16px;
        white-space: pre-wrap;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        line-height: 1.62;
        color: #2a2521;
      }}
      code {{
        background: #eee7dc;
        border-radius: 6px;
        padding: 2px 6px;
      }}
      @media (max-width: 1180px) {{
        .hero-grid,
        .overview-grid,
        .workspace,
        .run-grid,
        .episode-columns,
        .dimension-row {{
          grid-template-columns: 1fr;
        }}
        .sidebar {{
          position: static;
        }}
      }}
      @media (max-width: 760px) {{
        .shell {{
          width: min(100vw - 18px, 100%);
          margin: 9px auto 28px;
        }}
        .hero,
        .overview-card,
        .sidebar,
        .run-section,
        .subpanel {{
          padding: 20px;
        }}
        .episode-card summary,
        .run-header {{
          flex-direction: column;
        }}
        .score-badge-row,
        .episode-score-row {{
          justify-content: flex-start;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="panel hero">
        <div class="hero-grid">
          <div>
            <div class="section-kicker">Audio Style Evaluation</div>
            <h1>{title}</h1>
            <p class="lede">A report for comparing completed podcast scripts as audio, as host-style narration, and as resemblance to the bundled Revolutions sample.</p>
            <div class="nav-tag-row">
              {artifact_tags}
              <span class="nav-tag">Sample windows: {esc(window_targets)} words</span>
            </div>
            <div class="run-nav">{run_nav}</div>
          </div>
          <div class="hero-meta">
            <article class="hero-card">
              <div class="card-eyebrow">Summary</div>
              <div class="summary-grid">
                <div class="summary-card"><span class="small">Runs</span><strong>{summary['run_count']}</strong></div>
                <div class="summary-card"><span class="small">Episodes</span><strong>{summary['episode_count']}</strong></div>
                <div class="summary-card"><span class="small">Winner by AFS</span><strong>{esc(summary['winner_by_afs'])}</strong></div>
                <div class="summary-card"><span class="small">Winner by TSS</span><strong>{esc(summary['winner_by_tss'])}</strong></div>
              </div>
            </article>
            <article class="hero-card">
              <div class="card-eyebrow">Score Notes</div>
              <p class="small">{winner_mean_line}</p>
              <p class="small">{calibration_note}</p>
            </article>
          </div>
        </div>
      </section>

      <section class="overview-grid">
        <article class="panel overview-card">
          <div class="card-eyebrow">Leaderboard</div>
          <h2>Run comparison at a glance</h2>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Run</th>
                <th>AFS</th>
                <th>TSS</th>
                {leaderboard_tssv2_header}
                <th>Sampled words</th>
              </tr>
            </thead>
            <tbody>
              {''.join(leaderboard_rows)}
            </tbody>
          </table>
        </article>
        <article class="panel overview-card">
          <div class="card-eyebrow">How To Read It</div>
          <h2>Three score families, three different jobs</h2>
          <p class="small">{rubric_intro}</p>
          <p class="small">Raw AFS/TSS stay visible inside each run so the calibration layer never hides the underlying rubric totals.</p>
          <p class="small">Use the run sections below for evidence. The pretty summary is only the top layer.</p>
        </article>
      </section>

      <section class="workspace">
        <main class="main-stack">
          {''.join(run_sections)}
        </main>
        <aside class="panel sidebar">
          <section class="sidebar-section">
            <div class="card-eyebrow">Rubric</div>
            <h2>Dimension weights</h2>
            <p class="small">The large headline scores are calibrated onto the editorial anchor scale. Raw rubric totals are shown alongside them for transparency.</p>
            <table>
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th>AFS</th>
                  <th>TSS</th>
                  {"<th>TSSv2</th>" if has_tss_v2 else ""}
                </tr>
              </thead>
              <tbody>
                {''.join(rubric_rows)}
              </tbody>
            </table>
          </section>
          <section class="sidebar-section">
            <div class="card-eyebrow">Target Snippet</div>
            <h3>{esc(payload['target']['label'])}</h3>
            <p class="small">{payload['target']['word_count']} words</p>
            <details class="sidebar-detail">
              <summary>Show target snippet</summary>
              <pre>{esc(payload['target']['text'])}</pre>
            </details>
          </section>
          {comparison_section}
        </aside>
      </section>
    </div>
  </body>
</html>
"""
