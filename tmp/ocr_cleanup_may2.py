#!/usr/bin/env python3
"""Apply a conservative OCR cleanup pass to the May 2 cleaned batch."""

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
    return "\n\n".join(f"Chapter {index}\n\n{body.strip()}" for index, body in enumerate(bodies, start=1)).strip() + "\n"


def _cleanup_common(body: str) -> str:
    replacements = {
        "alQaeda": "al-Qaeda",
        "al Qaeda–General": "al-Qaeda–General",
        "selfimmolation": "self-immolation",
        "highlevel": "high-level",
        "fulltime": "full-time",
        "twelvecylinder": "twelve-cylinder",
        "postconflict": "post-conflict",
        "thePost": "the Post",
        "theTimes": "the Times",
        "theLos": "the Los",
        "readTheodore": "read Theodore",
        "andWashington": "and Washington",
        "andNew": "and New",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    body = body.replace("bbby my stay", "by my stay")
    body = re.sub(r"\b(al|el)([A-Z][A-Za-z]+)\b", r"\1-\2", body)
    body = re.sub(r"\b(anti|pro|post|pre|mid|non|ultra)([A-Z][A-Za-z]+)\b", r"\1-\2", body)
    body = re.sub(r'(?<=[.!?])"(?=[A-Z])', '" ', body)
    body = re.sub(r'(?<=[,;:])"(?=[A-Za-z])', '" ', body)
    body = re.sub(r'([.?!]["\']?)\d+\s*$', r"\1", body)
    return body


def _cleanup_india_at_war(body: str) -> str:
    return body.replace(
        "'embarrassing'.1 Since November, 'embarrassing'. Since November,",
        "'embarrassing'. Since November,",
        1,
    )


def _cleanup_india_wins(body: str) -> str:
    body = re.sub(r"\b\d+\s+\{?NDIA WINS FREEDOM\b", "", body)
    body = re.sub(r"\b\d+\s+INDIA WINS FREEDOM\b", "", body)
    body = re.sub(r"\b(?:PRELUDE TO PARTITION|DIVIDED INDIA)\s+\d+\b", "", body)
    body = re.sub(r"\bINTERIM GOVERNMENT\s+\d+\s+", "", body)

    body = body.replace("It was only in the Punjab and Sind that the Congress did not achieve comparable success. .", "It was only in the Punjab and Sind that the Congress did not achieve comparable success.", 1)
    body = body.replace("The - Government of India Act 1935", "The Government of India Act 1935", 1)
    body = body.replace("pro- -vincial autonomy", "provincial autonomy", 1)
    replacements = {
        "which eft a bad impression": "which left a bad impression",
        "leadership to men‘ of different communities": "leadership to men of different communities",
        "Congress must prepare to capture the Legislatures in 1924 and 1 use them": "Congress must prepare to capture the Legislatures in 1924 and use them",
        "Hakim Ajmal] Khan": "Hakim Ajmal Khan",
        "The epoca. acted as an incentive": "The episode acted as an incentive",
        "before they could arrive, trouble ies the capital.": "before they could arrive, trouble reached the capital.",
        "pracucal": "practical",
        "Withjn": "Within",
        "Bnitish": "British",
        "Delbi": "Delhi",
        "adherance": "adherence",
        "affort": "effort",
        "litthke": "little",
        "Pp ce": "place",
        "Necide": "decide",
        "this.and": "this and",
        "mcet": "meet",
        "alse": "also",
        "call out.our": "called out our",
        "at.eight": "at eight",
        "to. us": "to us",
        "Oaid-i-Azam": "Qaid-i-Azam",
        "Qaid-iAzam": "Qaid-i-Azam",
        "and‘all": "and all",
        "the‘lines": "the lines",
        "with~Lord": "with Lord",
        "turn‘to": "turn to",
        "Mountbatten.-He_is": "Mountbatten. He is",
        "Arrerican_arms": "American arms",
        "rese1 ve": "reserve",
        "9a.m.": "9 a.m.",
        "9a.m": "9 a.m.",
        "offer,the": "offer, the",
        "influencé": "influence",
        "extrémely": "extremely",
        "placés": "places",
        "samé": "same",
        "thé": "the",
        "falso": "false",
        "up toa point": "up to a point",
        "penitant sinner": "penitent sinner",
        "y my stay": "by my stay",
        "bby my stay": "by my stay",
        "notin any sense": "not in any sense",
        "aher some time": "after some time",
        "Liagqat Ali": "Liaqat Ali",
        "pclicy": "policy",
        "one TP its own trusted men": "one of its own trusted men",
        "If on of the other hand": "If on the other hand",
        "Liaqat Alias the chief representative": "Liaqat Ali as the chief representative",
        "1.1. Chundrigar": "I.I. Chundrigar",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)

    body = re.sub(r"\b1t\b", "it", body)
    body = re.sub(r"(?<=\.\s)it\b", "It", body)
    body = body.replace("When 1 did not return", "When I did not return")
    body = body.replace("1 told them", "I told them")
    body = body.replace("[ remained President", "I remained President")
    body = body.replace("| was therefore", "I was therefore")
    body = body.replace("| said I would try", "I said I would try")
    body = body.replace("My other longings were sent-in", "My other belongings were sent in")
    body = body.replace("had peel orders", "had received orders")
    body = body.replace("freedom+s4ks attitude", "freedom; his attitude")
    body = body.replace("Either both would have to be taken o1 none", "Either both would have to be taken or none")
    body = body.replace("The statement will he found on nace 1f6.", "The statement will be found on page 116.")

    body = re.sub(
        r"The resolution runs as follows:.*?The Congress has always aimed at a constitution where the fullest freedom and opportunities of development are guaranteed to the group and the individual, and social injustice yields place to the juster social order\.",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"ELECTION MANIFESTO .*?Let all those who care and long for freedom and the independence of India meet this test with strength and confidence, and march together to the free India of our dreams\.",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )

    body = body.replace(
        "As the war crisis deepened, people expected that there would be a change in the British government’s attitude to Mission, it is necessary to refer to a previous occasion when, soon after the outbreak of the War, Sir Stafford Cripps had visited India.",
        "As the war crisis deepened, people expected that there would be a change in the British Government's attitude to the Indian problem. This actually happened and the outcome was the Cripps Mission of 1942. Before discussing this Mission, it is necessary to refer to a previous occasion when, soon after the outbreak of the war, Sir Stafford Cripps had visited India.",
        1,
    )
    body = body.replace(
        "When the resolution of the Working Committee was published, it created an electric atmosphere in the country. People did not pause to consider what the implications were, but felt that at last Congress was launching a mass Resolution by both the people and the Government. The masses like some of the members of the Working Committee, had an implicit faith in Gandhiji’s leadership and felt that he had some move in his mind which would paralyse the Government and force it to come to terms. I may here confess that many people thought that Gandhiji would bring freedom for India by some magic or superhuman method, and did not therefore think it necessary to make any special personal effort.",
        "When the resolution of the Working Committee was published, it created an electric atmosphere in the country. People did not pause to consider what the implications were, but felt that at last Congress was launching a mass movement to make the British quit India. In fact, very soon the resolution came to be described as the 'Quit India' resolution by both the people and the Government. The masses, like some of the members of the Working Committee, had an implicit faith in Gandhiji's leadership and felt that he had some move in his mind which would paralyse the Government and force it to come to terms. I may here confess that there were also people who thought that Gandhiji would bring freedom for India by some magic or superhuman method and did not therefore think it necessary to make any special personal effort.",
        1,
    )
    body = body.replace(
        "wo points arise out of the present situation. The first is that the attitude of the Muslim League has been responsible for the failure of the Conference, The second point which emerges from the refusal of the Muslim League is that it is for Lord Wavell to decide whether to go forward or not. His ne has decided not to proceed for the P peeias In this connection I must repeat what I said at the Conference. The British Government cannot absolve themselves of the responsibility for the communal problems here. Whether it is today or tomorrow, they must take up a firm stand on a just and fair basis. There is no other alternative but to doso. And once a decision is taken, we must move forward. Those who are prepared to ge forward must be allowed to go forward and those who wish to be left out should be left out. Without determination, nothing can be done. Wavering minds and faltering mers will never carry us forward in the path of progress. We must think before we take a step, but once we decide, hesitation is not a virtue but a sign of definite weakness.",
        "Two points arise out of the present situation. The first is that the attitude of the Muslim League has been responsible for the failure of the Conference. The second point which emerges from the refusal of the Muslim League is that it is for Lord Wavell to decide whether to go forward or not. His Excellency has decided not to proceed for the present. In this connection I must repeat what I said at the Conference. The British Government cannot absolve themselves of the responsibility for the communal problem here. Whether it is today or tomorrow, they must take up a firm stand on a just and fair basis. There is no other alternative but to do so. And once a decision is taken, we must move forward. Those who are prepared to go forward must be allowed to go forward and those who wish to be left out should be left out. Without determination, nothing can be done. Wavering minds and faltering steps will never carry us forward in the path of progress. We must think before we take a step, but once we decide, hesitation is not a virtue but a sign of definite weakness.",
        1,
    )
    body = body.replace("highly eran question", "highly controversial question", 1)
    body = body.replace("have seen toit", "have seen to it")
    body = body.replace("benefitted much\n\nbby my stay", "benefitted much by my stay")
    body = body.replace("called out.our names", "called out our names")
    body = body.replace("From the very beginning of the War.", "From the very beginning of the war,")
    body = body.replace("declarations about, non-violence", "declarations about non-violence")
    body = body.replace("The Working Committee met on 5 August and Prepared a draft resolution", "The Working Committee met on 5 August and prepared a draft resolution")
    body = body.replace("open rebellion even if the, rebellion was non-violent", "open rebellion even if the rebellion was non-violent")
    body = body.replace("first draft of the ‘Quit India’ Resolution’’", "first draft of the ‘Quit India’ Resolution'")
    body = body.replace("first draft of the ‘Quit India’ Resolution’'", "first draft of the 'Quit India' Resolution.", 1)
    body = body.replace(
        "It was while I was there that I learnt that the Labour Party had won an unprecedented a letter of congratulation to Attlee and Cripps.",
        "It was while I was there that I learnt that the Labour Party had won an unprecedented victory. I immediately sent a letter of congratulation to Attlee and Cripps.",
        1,
    )
    body = re.sub(
        r"Sir Stafford Cripps replied Ane he agreed with me and his view was that the Government would\s+adopt the same attitude\.",
        "Sir Stafford Cripps replied that he agreed with me and his view was that the Government would adopt the same attitude.",
        body,
        count=1,
    )
    body = body.replace(
        "My colleagues pressed hard that I the same view. It was for me a delicate question but after careful consideration I came to the conclusion that I should remain outside. I therefore advised that Asaf Ali should be taken into the Cabinet. When Asaf Ali heard this, he also ressed that I should join but I did not agree.",
        "My colleagues pressed hard that I should take the same view. It was for me a delicate question but after careful consideration I came to the conclusion that I should remain outside. I therefore advised that Asaf Ali should be taken into the Cabinet. When Asaf Ali heard this, he also urged me to join, but I did not agree.",
        1,
    )
    body = body.replace("When Asaf Ali heard this, he also pressed that I should join, but I did not agree.", "When Asaf Ali heard this, he also urged me to join, but I did not agree.", 1)
    body = body.replace("one GovernorGeneral", "one Governor-General")
    body = body.replace("press conference [In reply", "press conference. In reply")
    body = body.replace("I did not oppose the resolution\n\n urging direct action", "I did not oppose the resolution urging direct action")
    body = body.replace("Liagat Ali", "Liaqat Ali")
    body = body.replace("thoygh", "though")

    body = body.replace(
        "In international affairs, the Congress stands for the establishment of a world federation of free nations. Till such time as such a federation takes shape, India must apart Path relations with all nations, and particularly with her neighbours on the east and the west and north. In the Far East, in South-East Asia and in Western Asia, India has had trade and cultural relations for thousands of years and it is inevitable that with freedom she should renew and develop these relations. Reasons of security and future trends of trade also demand these closer contacts with these regions India, which has conducted her own cheb os for freedom on a non-violent basis, will ae throw her weight on the side of world peace and cooperation. She will also champion the freedom of all other subject nations and Peoples, for only on the basis of this freedom and the elimination of imperialism everywhere can world peace be established.",
        "In international affairs, the Congress stands for the establishment of a world federation of free nations. Till such time as such a federation takes shape, India must develop friendly relations with all nations, and particularly with her neighbours on the east and the west and north. In the Far East, in South-East Asia and in Western Asia, India has had trade and cultural relations for thousands of years and it is inevitable that with freedom she should renew and develop these relations. Reasons of security and future trends of trade also demand these closer contacts with these regions. India, which has conducted her own struggle for freedom on a non-violent basis, will always throw her weight on the side of world peace and cooperation. She will also champion the freedom of all other subject nations and peoples, for only on the basis of this freedom and the elimination of imperialism everywhere can world peace be established.",
        1,
    )

    body = re.sub(
        r"The Committee have noted that criticisms have been advanced on behalf of the Muslim League.*?By that decision of the AICC they must stand, and they \) piasines to proceed accordingly with their work in the Constituent me",
        (
            "The Committee have noted that criticisms have been advanced on behalf of the Muslim League to the effect that the Congress acceptance of the proposals contained in the Statement of May 16th was conditional. "
            "The Committee wish to make it clear that while they did not approve of all the proposals contained in this Statement, they accepted the scheme in its entirety. "
            "They interpreted it so as to resolve the inconsistencies contained in it and fill the omissions in accordance with the principles laid down in that Statement. "
            "They hold that provincial autonomy is a basic provision and each province has the right to decide whether to form or join a group or not. "
            "Questions of interpretation will be decided by the procedure laid down in the Statement itself, and the Congress will advise its representatives in the Constituent Assembly to function accordingly. "
            "The Committee have emphasized the sovereign character of the Constituent Assembly, that is its right to function and draw up a constitution for India without the interference of any external power or authority. "
            "But the Assembly will naturally function within the internal limitations which are inherent in its task, and will therefore seek the largest measure of cooperation in drawing up a constitution of free India allowing the greatest measure of freedom and protection for all just claims and interests. "
            "It was with this object and with the desire to function in the Constituent Assembly and make it a success, that the Working Committee passed their resolution on June 26, 1946 which was subsequently ratified by the All India Congress Committee on July 7, 1946. "
            "By that decision of the AICC they must stand, and they propose to proceed accordingly with their work in the Constituent Assembly."
        ),
        body,
        count=1,
        flags=re.DOTALL,
    )

    body = body.replace(
        "At first Gandhiji would not agree and kept insisting on his own conditions. Finally however he relented and said that if",
        "At first Gandhiji would not agree and kept insisting on his own conditions. Finally however he relented and said that if the conditions I had suggested satisfied me, he also would accept them. I thanked him for his consideration for my views and begged him to accept my suggestions.",
        1,
    )
    body = body.replace(
        "Finally however he relented and said that if the conditions I had suggested satisfied me, he also would accept them. I thanked him for his consideration for my views and begged him to accept my suggestions. the conditions I had suggested satisfied me, he also would accept them. I thanked him for his consideration for my views and begged him to accept my suggestions.",
        "Finally however he relented and said that if the conditions I had suggested satisfied me, he also would accept them. I thanked him for his consideration for my views and begged him to accept my suggestions.",
        1,
    )
    body = re.sub(
        r"I thanked him for his consideration for my views and begged him to accept my suggestions\. the conditions I had suggested satisfied me, he also would accept them\. I thanked him for his consideration for my views and begged him to accept my suggestions\.",
        "I thanked him for his consideration for my views and begged him to accept my suggestions.",
        body,
        count=1,
    )
    body = re.sub(
        r"Meeting of the Congress Working Committee, Wardha, February 1942\..*?\bDecember 1948\.[^\n]*",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"The Congress President, Maulana Azad, arriving at the Viceregal Lodge.*",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"(I need not quote it as it has become a part of Indian history as the first draft of the ‘Quit India’ Resolution’’)\s+.*",
        r"\1",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = body.replace(
        "I need not quote it as it has become a part of Indian history as the first draft of the ‘Quit India’ Resolution’'\n\n. a > at’ i F . 2) © ao i . rs fy f { io; se i | s - rs. \" ‘",
        "I need not quote it as it has become a part of Indian history as the first draft of the ‘Quit India’ Resolution’'",
        1,
    )
    body = re.sub(
        r"em © Y.*?Rajkuman Amrit Kaur, Lord and Lady Mountbatten, the Hon'ble Pamela Mountbatten, Maulana Azad, and the Chinese Ambassador to India, Dr Lo Chia Luen, during the cremation of Mahatma Gandhi The Education Minister and the Prime Minister of India, when the latter laid the foundation stone of the Central Institute of Education, Delhi\..*",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    return body


def _cleanup_pity(body: str) -> str:
    replacements = {
        "big, heavy akes": "big, heavy flakes",
        "the second oor": "the second floor",
        "we had diculty": "we had difficulty",
        "we heard rst": "we heard first",
        "a mued bark": "a muffled bark",
        "feet shuing": "feet shuffling",
        "on the oor": "on the floor",
        "the ies that told us": "the flies that told us",
        "the dierence between the living and the dead": "the difference between the living and the dead",
        "suering": "suffering",
        "ocers": "officers",
        "coee": "coffee",
        "indelity": "infidelity",
        "condence": "confidence",
        "camouage": "camouflage",
        "inltration": "infiltration",
        "y-crawling bodies": "fly-crawling bodies",
        "the green elds": "the green fields",
        "mass graves lled": "mass graves filled",
        "onto the oor": "onto the floor",
        "Szymon Datner slowly rose from his chair and shued": "Szymon Datner slowly rose from his chair and shuffled",
        "were axed some faded sepia photographs": "were fixed some faded sepia photographs",
        "were sitting stiy to attention": "were sitting stiffly to attention",
        "the rst picture labelled 1926": "the first picture labelled 1926",
        "the Bialystok Gymnasium sta": "the Bialystok Gymnasium staff",
        "his stubby nger": "his stubby finger",
        "the akes larger": "the flakes larger",
        "beside the elds": "beside the fields",
        "the trac was backed up": "the traffic was backed up",
        "wind-deectors": "wind-deflectors",
        "snowelds": "snowfields",
        "cattle wagons and atbed trucks": "cattle wagons and flatbed trucks",
        "moving softly o into the cold fog": "moving softly off into the cold fog",
        "the ocial Polish guide": "the official Polish guide",
        "the SS camp sta": "the SS camp staff",
        "his English awless": "his English flawless",
        "cannot be justied": "cannot be justified",
        "Then I learned that the lm existed": "Then I learned that the film existed",
        "he had seen the lm": "he had seen the film",
        "Surely it must be possible to see this lm. Surely someone must remember what was on it.": "Surely it must be possible to see this film. Surely someone must remember what was on it.",
        "I kept my les on southern Lebanon": "I kept my files on southern Lebanon",
        "the lm was taken": "the film was taken",
        "comes the 'drone', trailing smoke from its engines, ying low over the base": "comes the 'drone', trailing smoke from its engines, flying low over the base",
        "'Fijibatt headquarters is under re.'": "'Fijibatt headquarters is under fire.'",
        "with photos fom the tape": "with photos from the tape",
        "'Massacre lm puts Israel in dock.'": "'Massacre film puts Israel in dock.'",
        "and at no prot": "and at no profit",
        "of the lm –of which they had": "of the film –of which they had",
        "UN ocials": "UN officials",
        "They red at our car": "They fired at our car",
        "had red the shells at Qana": "had fired the shells at Qana",
        "continue ring like the great ghters": "continue firing like the great fighters",
        "the bastards re at you": "the bastards fire at you",
        "we were ring well": "we were firing well",
        "were ring Katyushas": "were firing Katyushas",
        "after the ceasere": "after the ceasefire",
        "opened re from the Qana cemetery": "opened fire from the Qana cemetery",
        "called for re support": "called for fire support",
        "is lled with blood, biblical ire": "is filled with blood, biblical fire",
        "bunches of owers": "bunches of flowers",
        "Hezbollah gunre": "Hezbollah gunfire",
        "had suered so much": "had suffered so much",
        "They would ght, they said, to the nish": "They would fight, they said, to the finish",
        "walked home to Israel over the elds": "walked home to Israel over the fields",
        "Israeli tanks red at the returning villagers": "Israeli tanks fired at the returning villagers",
        "a BBC reporter and lm crew": "a BBC reporter and film crew",
        "to nd the prisoners": "to find the prisoners",
        "their yellow banners oating": "their yellow banners floating",
        "lifted o the elds": "lifted off the fields",
        "the kind who nds a book": "the kind who finds a book",
        "the Palestinians nally realised": "the Palestinians finally realised",
        "the Israeli oer of '96 per cent'": "the Israeli offer of '96 per cent'",
        "the city were among areas not included in the oer": "the city were among areas not included in the offer",
        "the inuence of all that I had witnessed": "the influence of all that I had witnessed",
        "It urged the Palestinians to ght": "It urged the Palestinians to fight",
        "was cordoned o by the Israelis": "was cordoned off by the Israelis",
        "F-l 6s bombed Palestinian oces": "F-16s bombed Palestinian offices",
        "tanks red into the refugee camps of Gaza": "tanks fired into the refugee camps of Gaza",
        "their familar counterfeit role": "their familiar counterfeit role",
        "as young and t as I was when I rst came": "as young and fit as I was when I first came",
        "still trying to nd one more clue": "still trying to find one more clue",
        "through a eld": "through a field",
        "in scientic evil": "in scientific evil",
        "think–even briey –then": "think–even briefly –then",
        "in the very res of Auschwitz": "in the very fires of Auschwitz",
        "of inconsequential signicance": "of inconsequential significance",
        "Borowski identies himself": "Borowski identifies himself",
        "pull o their coats": "pull off their coats",
        "my ight through Vienna": "my flight through Vienna",
        "the UN's nal report": "the UN's final report",
        "asked for a ceasere": "asked for a ceasefire",
        "many SLA men ed for their lives": "many SLA men fled for their lives",
        "killed by Israeli tank re in southern Lebanon": "killed by Israeli tank fire in southern Lebanon",
        "Israel had its F-l 6s": "Israel had its F-16s",
        "ripped o as if": "ripped off as if",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)

    word_replacements = {
        "oor": "floor",
        "nger": "finger",
        "rst": "first",
        "lm": "film",
        "nally": "finally",
        "inuence": "influence",
        "elds": "fields",
        "oating": "floating",
        "oces": "offices",
        "ocials": "officials",
        "ght": "fight",
        "ghters": "fighters",
        "nish": "finish",
        "eld": "field",
        "scientic": "scientific",
        "briey": "briefly",
        "signicance": "significance",
        "identies": "identifies",
        "nal": "final",
        "ceasere": "ceasefire",
    }
    for old, new in word_replacements.items():
        body = re.sub(rf"\b{re.escape(old)}\b", new, body)
    return body


def _cleanup_iron_wall(body: str) -> str:
    replacements = {
        'entitled"A Hidden Question"in': 'entitled "A Hidden Question" in',
        'Palestine." Among': 'Palestine. "Among',
        '"Zionism"was': '"Zionism" was',
        'it."A great deal': 'it." A great deal',
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body


def _cleanup_black_banners(body: str) -> str:
    replacements = {
        'means"father of death"': 'means "father of death"',
        'box,"and': 'box," and',
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body


def _cleanup_imperial(body: str) -> str:
    replacements = {
        "asked for 130, classroom desks": "asked for 130,000 classroom desks",
        "He got 8,.": "He got 8,000.",
        "AUSAID": "A USAID",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body


def main() -> int:
    specific_cleaners = {
        "india_at_war_yasmin_khan.cleaned.txt": _cleanup_india_at_war,
        "india_wins_freedom_maulana_abul_kalam_azad.cleaned.txt": _cleanup_india_wins,
        "imperial_life_in_the_emerald_city_rajiv_chandrasekaran.cleaned.txt": _cleanup_imperial,
        "pity_the_nation_lebanon_at_war_robert_fisk.cleaned.txt": _cleanup_pity,
        "the_iron_wall_avi_shlaim.cleaned.txt": _cleanup_iron_wall,
        "the_black_banners_declassified_ali_soufan.cleaned.txt": _cleanup_black_banners,
    }
    for path in sorted(OUTPUT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        bodies = _split_chapters(text)
        bodies = [_cleanup_common(body) for body in bodies]
        cleaner = specific_cleaners.get(path.name)
        if cleaner is not None:
            bodies = [cleaner(body) for body in bodies]
        path.write_text(_render_chapters(bodies), encoding="utf-8")
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
