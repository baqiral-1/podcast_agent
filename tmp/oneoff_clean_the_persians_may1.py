from __future__ import annotations

import re
from pathlib import Path

from oneoff_iran_book_clean_common_may1 import BookSpec, Bookmark, clean_book


def keep_chapter(bookmark: Bookmark) -> bool:
    return bookmark.title[:1].isdigit() and "Appendix" not in bookmark.title


OPENING_REPLACEMENTS = (
    (
        r"^Judgement, and life everlasting for the united soul and body\. Mary Boyce WITHIN TWENTY YEARS OF\b",
        "Within twenty years of",
    ),
    (
        r"^(?:RANIAN HISTORY HAS BEEN|HAS BEEN)punctuated\b",
        "Iranian history has been punctuated",
    ),
    (r"^PERSIAN EMPIRE was restored\b", "The ancient Persian empire was restored"),
    (
        r"^At a turn of the deep blue sky Neither Nader remained nor a Naderite\.\* Popular poem circulating after the assassination of Nader Shah THE EIGHTEENTH CENTURY W ASa\b",
        "The eighteenth century was a",
    ),
    (
        r"^\(quoted by Mehdiqoli Hedayat in Khaterat va Khatarat, p\. 386\) N THE PERIOD\b",
        "In the period",
    ),
    (r"^FIRST YEARS\b", "In the first years"),
    (r"^PERSONALLY RULED Iran between 1963 and 1978\.", "The shah personally ruled Iran between 1963 and 1978."),
    (r"^\(November 1978\) THE SHAH BELIEVED THAT\b", "The shah believed that"),
    (
        r"^Revolution in Two Moves \(1984\) THE REVOLUTION THAT 'SHOULD NOT HAVE HAPPENED' NS O M EO FI T Sbasic characteristics,",
        "In some of its basic characteristics,",
    ),
    (r"^W AS still\b", "In 2008 Iran was still"),
    (r"^WERE A Central Asian\b", "The Turks were a Central Asian"),
    (
        r"^All the order and progress\.\.\..*?Naser al-Din Shah \(1889\) THE PREMATURE DEATH IN 1848\b",
        "The premature death in 1848",
    ),
    (
        r"^People! Nothing would develop.*?Seyyed Jamal al-Din Isfahani \(1906\)\s+NASER AL\s*-DIN SHAH WAS\b",
        "Naser al-Din Shah was",
    ),
)


def polish(text: str) -> str:
    for pattern, replacement in OPENING_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.S)
    text = re.sub(r"\b44R\b", "", text)
    text = re.sub(r"\bHE GREAT PERSIAN POET\b", "THE GREAT PERSIAN POET", text, count=1)
    text = re.sub(r"\bRAN IS MUCH OLDER\b", "IRAN IS MUCH OLDER", text)
    text = re.sub(r"\bText\s+[A-Z][A-Z ]+\s+\d+\s+", "", text)
    text = re.sub(r"\bCHAPTER\s+\d+\b(?:\s+[A-Z][A-Za-z,&-]+){0,8}\s*", "", text)
    text = re.sub(r"\bText\b", "", text)
    text = re.sub(r"\bTHE POLITICS OF ELIMINATION\s+The politics of elimination\b", "The politics of elimination", text)
    text = re.sub(r"\btreasurehouseof\b", "treasure house of ", text)
    text = re.sub(r"\bconflictingsentimentsandemotions\b", "conflicting sentiments and emotions", text)
    text = re.sub(r"\bunified cultural entity\b", "unified cultural entity", text)
    text = re.sub(r"\bY et\b", "Yet", text)
    text = re.sub(r"\bY az([a-z]+)\b", r"Yaz\1", text)
    text = re.sub(r"\bSa' d\b", "Sa'd", text)
    text = re.sub(r"\bSa' di\b", "Sa'di", text)
    text = re.sub(r"\bIslam ist\b", "Islamist", text)
    text = re.sub(r"\bMUCH OLDER than\b", "Iran is much older than", text, count=1)
    text = re.sub(r"\band 323 began\b", "and began", text)
    text = re.sub(r"\b([1-9])\. (?=(?:million|billion|percent)\b)", r"\1 ", text)
    text = text.replace(
        "People! Nothing would develop your country other than subjection to law, observation of law, preservation of law, respect for law, implementation of the law, and again law, and once again law. Seyyed Jamal al-Din Isfahani (1906) NASER AL -DIN SHAH WAS",
        "Naser al-Din Shah was",
    )
    text = text.replace(
        "p r a c t i s e da l o n gw i t ho t h e rc u l t sa n dr e l i g i o n s. G r e e k,n o wt h eo f fi c i a ll a n g u a g ea n d",
        "practised along with other cults and religions. Greek, now the official language and",
    )
    text = text.replace(
        "w h i c hw a su s e db ys u b s e q u e n tT u r k i s hr u l e r sa sw e l l–f o rt h er e l a t i o n s h i pb e t w e e n t h ec a l i p h a t ea n dt h es u l t a n a t ef r o mt h i st i m eo n w a r d s. S o m eh a v ee v e ni n t e r p r e t e d it",
        "which was used by subsequent Turkish rulers as well – for the relationship between the caliphate and the sultanate from this time onwards. Some have even interpreted it",
    )
    text = text.replace(
        "did n o th a v et h ed o c t r i n a la u t h o r i t yo ft h ep o p e;a n dt h en e ws u l t a n ' si n d e p e n d e n t",
        "did not have the doctrinal authority of the pope; and the new sultan's independent",
    )
    text = text.replace("qonstitusiyun', t h a t i s,", "qonstitusiyun', that is,")
    text = text.replace(
        "greater legal secu ri ty o f p ri v a t e p r o pe rty i n la n d a s w e ll a s ca p i tal a n d a g o v e rn m e n t l ed b y",
        "greater legal security of private property in land as well as capital and a government led by",
    )
    text = text.replace("Y a' qub", "Ya'qub")
    text = text.replace("Y emen", "Yemen")
    text = text.replace("Y oung", "Young")
    text = text.replace("Y okhari", "Yokhari")
    text = text.replace("Y alda", "Yalda")
    text = text.replace("Y ahweh", "Yahweh")
    text = text.replace("Y ears", "Years")
    text = text.replace("Y athrib", "Yathrib")
    text = text.replace("New Y ear", "New Year")
    text = text.replace("A 'lam", "A'lam")
    text = text.replace("H a nafi", "Hanafi")
    text = text.replace("so n's", "son's")
    text = text.replace("sp ares", "spares")
    text = text.replace("bu t", "but")
    text = text.replace("Y erevan", "Yerevan")
    text = text.replace("acti vely", "actively")
    text = text.replace("Moahammad", "Mohammad")
    text = text.replace("Reza Kahn", "Reza Khan")
    text = text.replace("parliame ntary", "parliamentary")
    text = text.replace("displayi ng", "displaying")
    text = text.replace("ten dencies", "tendencies")
    text = text.replace("Y adollah", "Yadollah")
    text = text.replace("Qa' em", "Qa'em")
    text = text.replace("Sae' d", "Sae'd")
    text = text.replace("Sa' ed", "Sa'ed")
    text = text.replace("Shahnamehand", "Shahnameh and")
    text = text.replace("Th ere", "There")
    text = text.replace("no t", "not")
    text = text.replace("law..", "law.")
    text = text.replace("on the basis of the law.. and", "on the basis of the law, and")
    text = text.replace("thirtyseven", "thirty-seven")
    text = text.replace("nonMuslims", "non-Muslims")
    text = text.replace("nonMuslim", "non-Muslim")
    text = text.replace("alSadr", "al-Sadr")
    text = text.replace("Kia-Rostami", "Kiarostami")
    text = text.replace("born in Mecca c. into", "born in Mecca c. 570 into")
    text = text.replace("dangerous from the start. first converts included", "dangerous from the start. The first converts included")
    text = text.replace("the line of Keyanian –'key' meaning chief or king –with whom Shahnameh's second cycle begins. KEYANIYAN Kavus is", "the line of Keyanian – 'key' meaning chief or king – with whom Shahnameh's second cycle begins. Kavus is")
    text = text.replace("The tragedy of Rostam and Sohrab The story of Rostam and Sohrab", "The story of Rostam and Sohrab")
    text = text.replace("governmentrunning", "government running")
    text = text.replace("fancydress", "fancy-dress")
    text = text.replace("opponentsand", "opponents and")
    text = text.replace("enqelabatand", "enqelabat and")
    text = text.replace("otherworldy", "otherworldly")
    text = text.replace("Ashraf 's", "Ashraf's")
    text = text.replace("Saoshiyant", "Saoshyant")
    text = text.replace("in in virtually every Iranian village", "in virtually every Iranian village")
    text = text.replace("emanated form Ahura Mazda", "emanated from Ahura Mazda")
    text = text.replace("Cosmic history All beneficent", "All beneficent")
    text = text.replace("The rise of New Persian and classical literature It used to be believed", "It used to be believed")
    text = text.replace("The last great Seljuk The glorious days of the Seljuk empire", "The glorious days of the Seljuk empire")
    text = text.replace("most wellknown", "most well-known")
    text = text.replace("Sa' eb-e Tabizi", "Sa' eb-e Tabrizi")
    text = text.replace("Hazrat-e Abodl'azim", "Hazrat-e Abdol'azim")
    text = text.replace("third world counties", "third world countries")
    text = text.replace("before his death, in April 1989", "Before his death, in April 1989")
    text = text.replace("discussio n", "discussion")
    text = text.replace("prime minster", "prime minister")
    text = text.replace("marja'or", "marja' or")
    text = text.replace("marja'in", "marja' in")
    text = text.replace("socialscientific", "social-scientific")
    text = text.replace("325 He was pictured", "He was pictured")
    text = text.replace("liberaldemocratic", "liberal-democratic")
    text = text.replace("SAV AK.", "SAVAK.")
    text = text.replace("courts for Law\n\nFOR LAW 171 in the centre and into the hands of", "courts in the centre and into the hands of")
    text = text.replace("city's 200, inhabitants", "city's inhabitants")
    text = text.replace("he had 12, concubines in his 'golden harem' (moshku-ye zarrin). Even if 1 per cent of that number is true,", "he had thousands of concubines in his 'golden harem' (moshku-ye zarrin). Even if that number is exaggerated,")
    text = text.replace("Malkam cashed the £40, worth of royalties and moved into open opposition", "Malkam cashed the royalties and moved into open opposition")
    text = re.sub(r"\b(\d+)\.\s+per cent\b", r"\1 per cent", text)
    text = text.replace(
        "In 1941 the wholesale price index was at 20.; by 1944 it had risen to 61.. In the same period, the general cost of living index rose from 16 to 67. and the index for food from 18. to 75..",
        "In 1941 the wholesale price index was at 20; by 1944 it had risen to 61. In the same period, the general cost of living index rose from 16 to 67 and the index for food from 18 to 75.",
    )
    inline_heading_patterns = (
        r"(?<=[.!?])\s+MYTHS AND LEGENDS\s+(?=[A-Z][a-z])",
        r"(?<=[.!?])\s+THE DAWN OF MAN\s+(?=Shahnameh's)",
        r"(?<=[.!?])\s+MOHAMMAD\s+(?=The Prophet)",
        r"(?<=[.!?])\s+THE SAFAVID CLIMAX\s+(?=In the twelve years)",
        r"(?<=[.!?])\s+THE FALL OF ISFAHAN\s+(?=Mirveis)",
        r"(?<=[.!?])\s+NASER AL-DIN SHAH:\s+PHASE I \(1848–58\)\s+(?=Between 1848 and 1852)",
        r"(?<=[.!?])\s+THE RISE OF QAJAR POWER\s+(?=The last of the Zands)",
        r"(?<=[.!?])\s+THE FOUR MAIN FACTIONS\s+(?=However, the forces behind Hezbollah)",
        r"(?<=[.!?])\s+WILD ASS BAHRAM\s+(?=[A-Z][a-z])",
    )
    for pattern in inline_heading_patterns:
        text = re.sub(pattern, " ", text)
    text = re.sub(
        r"\*\s+Sar-e shab beh del qasd-e taraj dasht\s*/\s*Sahargah nah tan sar nah sar taj dasht\s*/\s*Beh yek gardesh-e charkh-e nilufari\s*/\s*Nah Nader beh ja mand o nah Naderi\.\s*",
        " ",
        text,
    )
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    cleaned: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"^(?:[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){1,6})\s+(?=[A-Z][a-z])", "", paragraph)
        paragraph = re.sub(r"^[ \t]{2,}", " ", paragraph)
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph)
        cleaned.append(paragraph.strip())
    text = "\n\n".join(paragraph for paragraph in cleaned if paragraph)
    text = re.sub(r"(?<=[.!?])\s+'?[A-Z][A-Z'’.\-]+(?:\s+[A-Z][A-Z'’.\-]+){1,8}'?\s+(?=[A-Z][a-z])", " ", text)
    text = text.replace("law..", "law.")
    text = text.replace("following month 40 that", "following month that")
    text = text.replace("There were campaigns against him especially in Tehran, Tabriz and Isfahan The Belgian customs officials", "There were campaigns against him especially in Tehran, Tabriz and Isfahan. The Belgian customs officials")
    text = text.replace("the protector of law.. The legislative assembly", "the protector of law. The legislative assembly")
    text = text.replace("the law.. Observing religion", "the law. Observing religion")
    text = text.replace("the AIOC. The shah was anxious", "the AIOC, and the shah was anxious")
    text = text.replace("WINDS OF CHANGE On 5 June 1989", "On 5 June 1989")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    spec = BookSpec(
        pdf_path=Path("/Users/baqir/Downloads/_OceanofPDF.com_The_Persians_Ancient_Mediaeval_and_Modern_Iran_-_Homa_Katouzian.pdf"),
        output_path=Path("/Users/baqir/Python/podcast_agent/sample_books/iran/the_persians.cleaned.txt"),
        book_title="The Persians",
        chapter_selector=keep_chapter,
        last_page_inclusive=405,
        postprocess=polish,
    )
    rendered = clean_book(spec)
    chapter_count = len(re.findall(r"(?m)^Chapter \d+:\n", rendered))
    print(f"Wrote {chapter_count} chapters to {spec.output_path.name}")


if __name__ == "__main__":
    main()
