#!/usr/bin/env python3
"""Apply targeted chapter-opening text fixes for the May 2 cleaned batch."""

from __future__ import annotations

import re
from pathlib import Path


OUTPUT_DIR = Path("sample_books") / "temp_clean"
CHAPTER_RE = re.compile(r"^Chapter (?P<number>\d+)$", re.MULTILINE)


def _split_chapters(text: str) -> list[str]:
    matches = list(CHAPTER_RE.finditer(text))
    bodies: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        bodies.append(text[start:end].strip())
    return bodies


def _render_chapters(bodies: list[str]) -> str:
    return (
        "\n\n".join(
            f"Chapter {index}\n\n{body.strip()}" for index, body in enumerate(bodies, start=1)
        ).strip()
        + "\n"
    )


def _replace_prefix(body: str, old: str, new: str) -> str:
    if body.startswith(old):
        return new + body[len(old) :]
    return body


def _replace_before_anchor(body: str, anchor: str, new_prefix: str) -> str:
    index = body.find(anchor)
    if index == -1:
        return body
    return new_prefix + body[index + len(anchor) :]


def _replace_start_regex(body: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, body, count=1, flags=re.DOTALL)


def _split_body_at_markers(body: str, markers: list[tuple[str, str]]) -> list[str]:
    parts: list[str] = []
    remaining = body
    for marker, replacement in markers:
        index = remaining.find(marker)
        if index == -1:
            raise ValueError(f"missing split marker: {marker!r}")
        parts.append(remaining[:index].strip())
        remaining = replacement + remaining[index + len(marker) :]
    parts.append(remaining.strip())
    return parts


def _truncate_before_first(body: str, markers: list[str]) -> str:
    indexes = [body.find(marker) for marker in markers if body.find(marker) != -1]
    if not indexes:
        return body
    return body[: min(indexes)].rstrip()


def _normalize_opening_spacing(body: str) -> str:
    head = body[:600]
    tail = body[600:]
    replacements = [
        ("1970 s", "1970s"),
        ("1980 s", "1980s"),
        ("1990 s", "1990s"),
        ("1940 s", "1940s"),
        ("1880 s", "1880s"),
        ("4,000 feet", "4,000 feet"),
        ("20,000 a month", "20,000 a month"),
        ("150,000 American", "150,000 American"),
        ('" Abu', '" Abu'),
        ('" We', '" We'),
        ("IWALKED", "I WALKED"),
        ("IDON'TKNOW", "I DON'T KNOW"),
        ("APALATIAL", "A PALATIAL"),
        ("ASar", "As Sar"),
        ("ASafir", "As Safir"),
        ("airconditioning", "air-conditioning"),
        ("west Beirut who had suered", "west Beirut who had suffered"),
        ("4year-old", "four-year-old"),
        ("ACambridge", "A Cambridge"),
        ("BenGurion", "Ben-Gurion"),
        ("THELIKUD'SVICTORY", "THE LIKUD'S VICTORY"),
        ("EHUD BARAK'SVICTORY", "EHUD BARAK'S VICTORY"),
        ("EHUD OLMERT'SRISE", "EHUD OLMERT'S RISE"),
    ]
    for old, new in replacements:
        head = head.replace(old, new)
    head = re.sub(r'([A-Za-z0-9,.;!?])\s+(["”])', r"\1\2", head)
    head = re.sub(r'([.!?]["”])(?=[A-Z])', r"\1 ", head)
    head = re.sub(r'([,;:])(?=[A-Za-z"])', r"\1 ", head)
    head = re.sub(r"(?<=\d)(A\.M\.|P\.M\.)", r" \1", head)
    head = re.sub(r"\b(\d{4}) s\b", r"\1s", head)
    head = re.sub(r"\s{2,}", " ", head)
    return head + tail


def _fix_directorate(text: str) -> str:
    bodies = _split_chapters(text)
    if len(bodies) == 35:
        return text
    if len(bodies) != 23:
        raise ValueError(f"unexpected Directorate S chapter count: {len(bodies)}")
    bodies[6] = _replace_prefix(bodies[6], "RSmall Change ich Blee", "Rich Blee")
    bodies[18] = _replace_prefix(bodies[18], "OTough Love n July 7, 2008,", "On July 7, 2008,")
    bodies[20] = _replace_prefix(bodies[20], "Since the 1970 s,", "Since the 1970s,")

    tail_21_to_29 = _split_body_at_markers(
        bodies[20],
        [
            (
                "ates and Mullen decided they needed to replace General Dave McKiernan,",
                "Gates and Mullen decided they needed to replace General Dave McKiernan,",
            ),
            (
                "uring the first week of August 2009, while on vacation in the south of France,",
                "During the first week of August 2009, while on vacation in the south of France,",
            ),
            (
                "On April 20, 2010, Doug Lute convened the Conflict Resolution Cell,",
                "On April 20, 2010, Doug Lute convened the Conflict Resolution Cell,",
            ),
            (
                'On July 19, as she began to digest Kayani\'s "2.0"white paper,',
                'On July 19, as she began to digest Kayani\'s "2.0" white paper,',
            ),
            (
                "June 19, 2010:",
                "June 19, 2010:",
            ),
            (
                "Richard Holbrooke extended himself as if he were in his thirties.",
                "At sixty-nine, Richard Holbrooke extended himself as if he were in his thirties.",
            ),
            (
                "BTWENTY -EIGHT Hostages y late 2010,",
                "By late 2010,",
            ),
            (
                "Dragon's Breath atta Khel, a town in North Waziristan, lies about twenty-five miles southwest of Miranshah, toward the border with Afghanistan.",
                "Datta Khel, a town in North Waziristan, lies about twenty-five miles southwest of Miranshah, toward the border with Afghanistan.",
            ),
        ],
    )
    tail_30_to_35 = _split_body_at_markers(
        bodies[21],
        [
            (
                "oon after Clinton returned to Washington, Tayeb Agha delivered the biggest breakthrough yet:",
                "Soon after Clinton returned to Washington, Tayeb Agha delivered the biggest breakthrough yet:",
            ),
            (
                "Hand Sent: Sunday, April 10, 2011 7: AM To: Loftis Email Subject: RE: Arrived in Kabul...",
                "Hand Sent: Sunday, April 10, 2011 7: AM To: Loftis Email Subject: RE: Arrived in Kabul...",
            ),
            (
                "bdul Saboor's murder of Darin Loftis and Robert Marchanti moved the American war command to restudy the threat.",
                "Abdul Saboor's murder of Darin Loftis and Robert Marchanti moved the American war command to restudy the threat.",
            ),
            (
                "After Karzai's demands scuttled the talks with Tayeb Agha, the Qataris initiated a new attempt to get direct talks between the Taliban and the United States back on track.",
                "After Karzai's demands scuttled the talks with Tayeb Agha, the Qataris initiated a new attempt to get direct talks between the Taliban and the United States back on track.",
            ),
            (
                "RTHIRTY -FIVE Coups d'État ahmatullah Nabil first took charge of Afghan intelligence in 2010, after Hamid Karzai forced the resignation of Amrullah Saleh, amid Karzai's flirtations with I.S.I.",
                "Rahmatullah Nabil first took charge of Afghan intelligence in 2010, after Hamid Karzai forced the resignation of Amrullah Saleh, amid Karzai's flirtations with I.S.I.",
            ),
        ],
    )

    replacements = {
        "Eleak": "leak",
        "Pto recover": "to recover",
        "asha flew to Washington in April.": "Pasha flew to Washington in April.",
        "SClinton tried to clarify": "Clinton tried to clarify",
        "A t Forward Operating Base Lonestar": "At Forward Operating Base Lonestar",
        "Aarin Loftis": "Darin Loftis",
        "AH llen": "Allen",
        "olly Loftis": "Holly Loftis",
        "ahmatullah Nabil": "Rahmatullah Nabil",
        "Oconfidence-building": "confidence-building",
        "AfghanistanPakistan": "Afghanistan-Pakistan",
        "Talibanjan": "Taliban",
        "OObama": "Obama",
        "KKayani": "Kayani",
        "TKayani": "Kayani",
        "hammeredtogether": "hammered-together",
        "civilianpolitical": "civilian-political",
        "eTaiba": "-e-Taiba",
    }

    repaired = bodies[:20] + tail_21_to_29 + tail_30_to_35
    cleaned: list[str] = []
    for body in repaired:
        for old, new in replacements.items():
            body = body.replace(old, new)
        cleaned.append(_normalize_opening_spacing(body))
    return _render_chapters(cleaned)


def _fix_plan(text: str) -> str:
    bodies = _split_chapters(text)
    replacements = {
        0: (
            "IN EARLY JANUARY 2001, before George W. Bush was inaugurated, Vice President-elect Dick Cheney p President-elect Dick Cheney",
            "IN EARLY JANUARY 2001, before George W. Bush was inaugurated, Vice President-elect Dick Cheney",
        ),
        1: ("T HESEPTEMBER 11, 2001,terrorist", "THE SEPTEMBER 11, 2001, terrorist"),
        2: ("A FTERFRANKS'S MINI-EXPLOSIONon", "AFTER FRANKS'S MINI-EXPLOSION on"),
        3: ("I N LATENOVEMBER,the", "IN LATE NOVEMBER, the"),
        4: ("T HE MORNING OFFRIDAY,December", "THE MORNING OF FRIDAY, December"),
        5: ("B Y THE BEGINNING OF 2002,", "BY THE BEGINNING OF 2002,"),
        6: (
            "AFTER THE DECEMBER 28 Crawford briefing for the president, Rumsfeld ordered Franks to come back Franks to come back",
            "AFTER THE DECEMBER 28 Crawford briefing for the president, Rumsfeld ordered Franks to come back",
        ),
        7: (
            "SITTING IN HIS SMALL West Wing office with a structural pillar squarely in the middle, presidenti middle, presidential",
            "SITTING IN HIS SMALL West Wing office with a structural pillar squarely in the middle, presidential",
        ),
    }
    for index, (old, new) in replacements.items():
        bodies[index] = _replace_prefix(bodies[index], old, new)
    more_replacements = {
        8: ("R UMSFELD WASN'T WASTING TIME.On Friday,", "RUMSFELD WASN'T WASTING TIME. On Friday,"),
        9: ("1 0 T ERRORISM, ESPECIALLY ALQAEDA,", "TERRORISM, ESPECIALLY AL-QAEDA,"),
        10: (
            "IN MARCH , Tenet met secretly with two individuals who would be critical to covert action insid INMARCH, Tenet met secretly",
            "IN MARCH, Tenet met secretly with two individuals who would be critical to covert action inside Iraq:",
        ),
        11: (
            '1 2 "S TOP BOTHERING ME!"the president said',
            '"STOP BOTHERING ME!" the president said',
        ),
        12: (
            "1 3 W ITH THE PRESIDENTIALfinding authorizing",
            "WITH THE PRESIDENTIAL finding authorizing",
        ),
        13: (
            "AT 4:30 P.M. on Monday, August 5, Franks, carrying 110 slides of Top Secret/Polo Step war planni A T 4:30 P.M. on Monday, August 5,",
            "AT 4:30 P.M. on Monday, August 5, Franks, carrying 110 slides of Top Secret/Polo Step war planning,",
        ),
        14: ("1 5 O NWEDNESDAY, AUGUST 14,", "ON WEDNESDAY, AUGUST 14,"),
        15: (
            "THE PRESIDENT RETURNED from Crawford to the White House on Sunday, September 1. An unhappy Powel THE PRESIDENT RETURNED from Crawford to the White House on Sunday,",
            "THE PRESIDENT RETURNED from Crawford to the White House on Sunday, September 1.",
        ),
        16: (
            "SPEECHWRITER MIKE GERSON probed the president about precisely what he wanted to say to the U.N SPEECHWRITERMIKEGERSON probed the president about precisely what he wanted to say to the U.N",
            "SPEECHWRITER MIKE GERSON probed the president about precisely what he wanted to say to the U.N.",
        ),
        17: ("1 8 S IX MONTHS EARLIER,", "SIX MONTHS EARLIER,"),
        18: ("1 9 R UMSFELD KEPT HONINGthe", "RUMSFELD KEPT HONING the"),
        19: ("2 0 T HE TELEVISION MONITORhad", "THE TELEVISION MONITOR had"),
        20: (
            "POWELL REALIZED THAT HE , the president and perhaps the rest of the world were traveling a road POWELL REALIZED THAT HE, the president and perhaps the rest of the world were traveling a road",
            "POWELL REALIZED THAT HE, the president and perhaps the rest of the world were traveling a road",
        ),
        21: ("2 2 O NFRIDAY, NOVEMBER 15,", "ON FRIDAY, NOVEMBER 15,"),
        22: ("2 3 R UMSFELD'S STRATEGY OF DRIBBLINGout", "RUMSFELD'S STRATEGY OF DRIBBLING out"),
        23: ("2 4 R ICE WENT TO HER AUNT'Sfor", "RICE WENT TO HER AUNT'S for"),
        24: (
            "FROM THEIR ALMOST daily conversations, Cheney had come to realize that the president had made hi FROM THEIR ALMOST daily conversations, Cheney had come to realize that the president had made his decision.",
            "FROM THEIR ALMOST daily conversations, Cheney had come to realize that the president had made his decision.",
        ),
        25: ("2 6 B EFORE A MEETINGwith", "BEFORE A MEETING with"),
        26: ("2 7 A T THESTATEDEPARTMENT,Armitage", "AT THE STATE DEPARTMENT, Armitage"),
        27: ("2 8 A T A PRIVATE MEETINGwith", "AT A PRIVATE MEETING with"),
        28: ("2 9 O NWEDNESDAY,February 5,", "ON WEDNESDAY, February 5,"),
        29: ("3 0 F EBRUARY 15HAD BEEN Apotential", "FEBRUARY 15 HAD BEEN A potential"),
        30: (
            '3 1 "L OOKS REALLY GOOD. This is going to happen,"Saul',
            '"LOOKS REALLY GOOD. This is going to happen," Saul',
        ),
        31: ("3 2 W HEN THE PRESIDENTmet", "WHEN THE PRESIDENT met"),
        32: ("3 3 A NDYCARD HAD SUGGESTEDthat", "ANDY CARD HAD SUGGESTED that"),
        33: ("3 4 I NWASHINGTON THE NEXT DAY,Monday,", "IN WASHINGTON THE NEXT DAY, Monday,"),
        34: ("3 5 B USH BEGAN BUSINESSon", "BUSH BEGAN BUSINESS on"),
    }
    for index, (old, new) in more_replacements.items():
        bodies[index] = _replace_prefix(bodies[index], old, new)
    bodies[1] = bodies[1].replace("killed nearly 3, altered", "killed nearly 3,000, altered", 1)
    bodies[10] = _replace_prefix(
        bodies[10],
        "IN MARCH, Tenet met secretly with two individuals who would be critical to covert action inside Iraq: with two individuals who would be critical to covert action inside Iraq:",
        "IN MARCH, Tenet met secretly with two individuals who would be critical to covert action inside Iraq:",
    )
    bodies[13] = _replace_prefix(
        bodies[13],
        "AT 4:30 P.M. on Monday, August 5, Franks, carrying 110 slides of Top Secret/Polo Step war planning, Franks, carrying 110 slides of Top Secret/Polo Step war planning,",
        "AT 4:30 P.M. on Monday, August 5, Franks, carrying 110 slides of Top Secret/Polo Step war planning,",
    )
    bodies[15] = _replace_prefix(
        bodies[15],
        "THE PRESIDENT RETURNED from Crawford to the White House on Sunday, September 1. September 1.",
        "THE PRESIDENT RETURNED from Crawford to the White House on Sunday, September 1.",
    )
    bodies[16] = _replace_prefix(
        bodies[16],
        "SPEECHWRITER MIKE GERSON probed the president about precisely what he wanted to say to the U.N..",
        "SPEECHWRITER MIKE GERSON probed the president about precisely what he wanted to say to the U.N.",
    )
    bodies[30] = _replace_prefix(
        bodies[30],
        '"LOOKS REALLY GOOD. This is going to happen, " Saul',
        '"LOOKS REALLY GOOD. This is going to happen," Saul',
    )
    bodies[34] = _replace_prefix(
        bodies[34],
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote." Not only did you win, but public opinion has shifted because you\'re leading, "Bush said,',
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote. "Not only did you win, but public opinion has shifted because you\'re leading," Bush said,',
    )
    bodies[34] = _replace_prefix(
        bodies[34],
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote." Not only did you win, but public opinion has shifted because you\'re leading, " Bush said,',
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote. "Not only did you win, but public opinion has shifted because you\'re leading," Bush said,',
    )
    bodies[-1] = _truncate_before_first(
        bodies[-1],
        [
            "A c k n o w l e d g m e n t s",
            "A special thanks to John Wahler,",
            "Mark Malseed and I give special thanks",
            "The core of this book comes from more than 75 sources.",
            "GlobalSecurity.org is an invaluable resource",
        ],
    )
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    bodies[30] = _replace_prefix(
        bodies[30],
        '"LOOKS REALLY GOOD. This is going to happen, " Saul',
        '"LOOKS REALLY GOOD. This is going to happen," Saul',
    )
    bodies[34] = _replace_prefix(
        bodies[34],
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote." Not only did you win, but public opinion has shifted because you\'re leading, "Bush said,',
        'BUSH BEGAN BUSINESS on Wednesday, March 19, at 7:40 A.M. with a 20-minute call to Blair on the secure phone. Both leaders were in high spirits. Bush congratulated Blair on the vote. "Not only did you win, but public opinion has shifted because you\'re leading," Bush said,',
    )
    bodies[34] = bodies[34].replace(
        'vote." Not only did you win, but public opinion has shifted because you\'re leading, " Bush said,',
        'vote. "Not only did you win, but public opinion has shifted because you\'re leading," Bush said,',
        1,
    )
    return _render_chapters(bodies)


def _fix_pity(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[1] = re.sub(
        r"^It is a tragedy.*?When David Roberts toured",
        "When David Roberts toured",
        bodies[1],
        count=1,
        flags=re.DOTALL,
    )
    bodies[6] = _replace_start_regex(
        bodies[6], r"^.*?The dark brown wooden door", "The dark brown wooden door"
    )
    bodies[7] = _replace_start_regex(
        bodies[7], r"^.*?The young man wanted to help us\.", "The young man wanted to help us."
    )
    bodies[9] = _replace_prefix(bodies[9], "As Sar", "As Safir")
    bodies[10] = _replace_before_anchor(
        bodies[10],
        "It was the ies that told us.",
        "It was the ies that told us.",
    )
    bodies[11] = _replace_start_regex(
        bodies[11],
        r"^.*?The week after the massacre at Sabra and Chatila,",
        "The week after the massacre at Sabra and Chatila,",
    )
    bodies[12] = _replace_start_regex(
        bodies[12],
        r"^.*?The winter rains came early in 1982\.",
        "The winter rains came early in 1982.",
    )
    bodies[13] = _replace_start_regex(
        bodies[13],
        r"^.*?The two Phalangists were frightened\.",
        "The two Phalangists were frightened.",
    )
    bodies[14] = _replace_start_regex(
        bodies[14],
        r"^.*?The pariah of Lebanon became the honoured guest of Syria in just 24 hours\.",
        "The pariah of Lebanon became the honoured guest of Syria in just 24 hours.",
    )
    bodies[15] = _replace_before_anchor(
        bodies[15],
        "Our guards had seen the piece on the early news",
        "Our guards had seen the piece on the early news",
    )
    bodies[16] = _replace_start_regex(
        bodies[16], r"^.*?IN early December 1991", "In early December 1991"
    )
    bodies[4] = _replace_prefix(bodies[4], "May 1980 On 23 March 1978,", "On 23 March 1978,")
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_black_banners(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[0] = _replace_prefix(
        bodies[0],
        '"You can\'t stop the mujahideen, "Abu Jandal told me on September 17, 2001." We will be victorious."',
        '"You can\'t stop the mujahideen," Abu Jandal told me on September 17, 2001. "We will be victorious."',
    )
    bodies[5] = _replace_prefix(
        bodies[5],
        'July 1999." We\'ve got one more important order of business, "said Tom Donlon, the I-40 squad leader.',
        'July 1999. "We\'ve got one more important order of business," said Tom Donlon, the I-40 squad leader.',
    )
    bodies[10] = _replace_start_regex(
        bodies[10],
        r'^"We\'re Stubborn, but We\'re Not Crazy"\s+The initial leads',
        "The initial leads",
    )
    bodies[12] = _replace_start_regex(
        bodies[12],
        r'^"What Is al-Qaeda Doing in Malaysia\?"Having gotten',
        "Having gotten",
    )
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    bodies[0] = _replace_prefix(
        bodies[0],
        '"You can\'t stop the mujahideen, " Abu Jandal told me on September 17, 2001." We will be victorious."',
        '"You can\'t stop the mujahideen," Abu Jandal told me on September 17, 2001. "We will be victorious."',
    )
    bodies[5] = _replace_prefix(
        bodies[5],
        'July 1999." We\'ve got one more important order of business, " said Tom Donlon, the I-40 squad leader.',
        'July 1999. "We\'ve got one more important order of business," said Tom Donlon, the I-40 squad leader.',
    )
    return _render_chapters(bodies)


def _fix_imperial(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[6] = _replace_before_anchor(
        bodies[6],
        "while most CPA staffers were still eating breakfast.",
        "OUR MOTORCADE ROARED AWAY from the Republican Palace while most CPA staffers were still eating breakfast.",
    )
    bodies[2] = _replace_prefix(
        bodies[2], "IHAD MY FIRST LOOK INSIDE", "I HAD MY FIRST LOOK INSIDE"
    )
    bodies[4] = _replace_prefix(bodies[4], "AFEW HOURS AFTER", "A FEW HOURS AFTER")
    bodies[12] = _replace_prefix(bodies[12], "EVERY TIME IWALKED", "EVERY TIME I WALKED")
    bodies[14] = _replace_prefix(bodies[14], "\"IDON'TKNOW", "\"I DON'T KNOW")
    bodies[17] = _replace_prefix(bodies[17], "IN APALATIAL VILLA", "IN A PALATIAL VILLA")
    bodies[21] = _replace_before_anchor(
        bodies[21],
        "in southwestern Baghdad.",
        "EIGHT GUNMEN WAITED ON EITHER SIDE of the Sajjad Mosque in southwestern Baghdad.",
    )
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_one_palestine(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[11] = _replace_prefix(
        bodies[11], "1 1 1. Some of the immigrants", "Some of the immigrants"
    )
    bodies[18] = bodies[18].replace("ACambridge graduate", "A Cambridge graduate", 1)
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_india_wins(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[12] = _replace_before_anchor(
        bodies[12],
        "met in Delhi on the 17th.",
        "I have said that the Congress had entrusted the Parliamentary Committee with the task of forming the Interim Government. Accordingly Jawaharlal, Patel, Rajendra Prasad and I met in Delhi on the 17th.",
    )
    bodies[14] = bodies[14].replace("lingering hupe", "lingering hope", 1)
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_indian_mutiny(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[3] = _replace_start_regex(
        bodies[3], r"^.*?Historians have tended to agree\.", "Historians have tended to agree."
    )
    bodies[6] = _replace_prefix(bodies[6], "British Army led,", "Where the British Army led,")
    bodies[13] = _replace_before_anchor(
        bodies[13],
        "Hugh Massy Wheeler, commanding the Cawnpore Division,",
        "Major-General Sir Hugh Massy Wheeler, commanding the Cawnpore Division,",
    )
    bodies[17] = _replace_start_regex(
        bodies[17],
        r"^.*?Nicholson's arrival at Delhi raised British spirits immeasurably\.",
        "Nicholson's arrival at Delhi raised British spirits immeasurably.",
    )
    bodies[-1] = re.sub(r"\s+In 1893, for\s*$", "", bodies[-1])
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_iron_wall(text: str) -> str:
    bodies = _split_chapters(text)
    replacements = {
        0: ("I N 1907", "IN 1907"),
        1: ("T HE STATE OF ISRAEL", "THE STATE OF ISRAEL"),
        6: ("L EVI ESHKOL", "LEVI ESHKOL"),
        9: ("THE LIKUD'SVICTORY IN", "THE LIKUD'S VICTORY IN"),
        11: (
            "1 1 POLITICAL PARALYSIS 1984-1988 EMBROILMENT IN THE LEBANESE",
            "EMBROILMENT IN THE LEBANESE",
        ),
        13: ("W HEN THE LABOR PARTY", "WHEN THE LABOR PARTY"),
        18: ("I N THE QUARTER OF", "IN THE QUARTER OF"),
        19: ("L ESS THAN A MONTH", "LESS THAN A MONTH"),
        20: ("EHUD OLMERT'SRISE TO", "EHUD OLMERT'S RISE TO"),
    }
    for index, (old, new) in replacements.items():
        bodies[index] = _replace_prefix(bodies[index], old, new)
    bodies[9] = bodies[9].replace(
        "Equally vehement was its denial that the Palestinians had a Foreign Policy The Likud's ideology could be summed up in two words—Greater Israel. According to this ideology, Judea and Samaria, the biblical terms for the West Bank, were an integral part of Eretz Israel, the Land of Israel. The Likud categorically denied that Jordan had any claim to sovereignty over this area. Equally vehement was its denial that the Palestinians had a",
        "Equally vehement was its denial that the Palestinians had a",
        1,
    )
    bodies[-1] = re.sub(r"\s+AVI SHLAIM December 2013\s*$", "", bodies[-1])
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def _fix_arabs_history(text: str) -> str:
    bodies = _split_chapters(text)
    bodies[12] = _replace_before_anchor(
        bodies[12],
        "Growing global",
        "The Arab world was shaped by the power of oil in the eventful years of the 1970s. Nature spread oil unevenly among the Arab states. With the exception of Iraq, where the mighty Tigris and Euphrates rivers have supported large agrarian populations for millennia, the greatest oil reserves are to be found in the least densely populated Arab states: Saudi Arabia, Kuwait, and the other Persian Gulf states, Libya, and Algeria in North Africa. Token discoveries have been made in Egypt, Syria, and Jordan, though not enough to meet local demand. Oil was first discovered in the Arab world in the late 1920s and early 1930s. For four decades, Western oil companies enjoyed unfettered control over the production and marketing of Arab hydrocarbons. Rulers in oil-producing states grew wealthy and in the 1950s and 1960s initiated development schemes to bring the benefits of oil wealth to their impoverished populations. It was only in the 1970s, however, that a convergence of factors turned oil into a source of power for the Arab world. Growing global",
    )
    bodies = [_normalize_opening_spacing(body) for body in bodies]
    return _render_chapters(bodies)


def main() -> int:
    fixers = {
        "directorate_s_steve_coll.cleaned.txt": _fix_directorate,
        "imperial_life_in_the_emerald_city_rajiv_chandrasekaran.cleaned.txt": _fix_imperial,
        "india_wins_freedom_maulana_abul_kalam_azad.cleaned.txt": _fix_india_wins,
        "one_palestine_complete_tom_segev.cleaned.txt": _fix_one_palestine,
        "plan_of_attack_bob_woodward.cleaned.txt": _fix_plan,
        "pity_the_nation_lebanon_at_war_robert_fisk.cleaned.txt": _fix_pity,
        "the_arabs_a_history_eugene_rogan.cleaned.txt": _fix_arabs_history,
        "the_black_banners_declassified_ali_soufan.cleaned.txt": _fix_black_banners,
        "the_indian_mutiny_saul_david.cleaned.txt": _fix_indian_mutiny,
        "the_iron_wall_avi_shlaim.cleaned.txt": _fix_iron_wall,
    }
    for name, fixer in fixers.items():
        path = OUTPUT_DIR / name
        text = path.read_text(encoding="utf-8")
        path.write_text(fixer(text), encoding="utf-8")
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
