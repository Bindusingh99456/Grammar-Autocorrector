"""Synthesize realistic noisy student sentences and build Stage 1 & Stage 2 training sets.

Phase M2 requirements:
- Clean sentences pool: C4 outputs, W&I corrected text, CoEdIT targets
- Corruptions:
  * typos: adjacent-key substitution, letter swap, dropped letter, doubled letter
  * confusables: their/there/they're, your/you're, its/it's, then/than, to/too/two,
    affect/effect, lose/loose, whose/who's, a lot/alot, definitely/definately,
    plus 150+ other common student misspellings & homophones
  * grammar: wrong verb agreement & tense (using lemminflect), missing or extra articles,
    wrong prepositions, missing capital letters, doubled words, missing commas and full stops
- Rate: 5-20% words corrupted; 10% kept untouched (input == output)
- Outputs:
  * data/processed/noise_pairs.csv (300,000 pairs)
  * data/processed/stage1.csv (noise_pairs + C4 + coedit_gec, shuffled ~688k rows)
  * data/processed/stage1_colab.csv (150,000 subsampled rows for Colab sessions)
  * data/processed/stage2.csv (wi_train + fce_train + coedit_gec, oversampled x3, shuffled)
- Prints 20 random noise examples and category statistics.
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path
import lemminflect
import pandas as pd


PROCESSED_DIR = Path("data/processed")
NOISE_PAIRS_CSV = PROCESSED_DIR / "noise_pairs.csv"
STAGE1_CSV = PROCESSED_DIR / "stage1.csv"
STAGE1_COLAB_CSV = PROCESSED_DIR / "stage1_colab.csv"
STAGE2_CSV = PROCESSED_DIR / "stage2.csv"
TARGETED_CSV = PROCESSED_DIR / "targeted.csv"

TARGET_NOISE_ROWS = 300_000
TARGETED_TOTAL_ROWS = 30_000
MAX_WORDS = 64
UNTOUCHED_RATIO = 0.10

# QWERTY adjacent key map
QWERTY_NEIGHBORS = {
    "q": ["w", "a", "s"],
    "w": ["q", "e", "a", "s", "d"],
    "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"],
    "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"],
    "u": ["y", "i", "h", "j", "k"],
    "i": ["u", "o", "j", "k", "l"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z"],
    "s": ["a", "w", "e", "d", "z", "x"],
    "d": ["s", "e", "r", "f", "x", "c"],
    "f": ["d", "r", "t", "g", "c", "v"],
    "g": ["f", "t", "y", "h", "v", "b"],
    "h": ["g", "y", "u", "j", "b", "n"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
}

# Homophone sets
HOMOPHONE_GROUPS = [
    {"their", "there", "they're"},
    {"your", "you're"},
    {"its", "it's"},
    {"then", "than"},
    {"to", "too", "two"},
    {"affect", "effect"},
    {"lose", "loose"},
    {"whose", "who's"},
    {"accept", "except"},
    {"advice", "advise"},
    {"choose", "chose"},
    {"complement", "compliment"},
    {"principle", "principal"},
    {"stationary", "stationery"},
    {"weather", "whether"},
    {"quiet", "quite"},
    {"lead", "led"},
    {"passed", "past"},
    {"sight", "site"},
    {"where", "were"},
    {"here", "hear"},
    {"know", "no"},
    {"knew", "new"},
    {"buy", "by"},
    {"right", "write"},
    {"whole", "hole"},
    {"break", "brake"},
    {"peace", "piece"},
    {"meat", "meet"},
    {"plane", "plain"},
    {"stair", "stare"},
    {"waste", "waist"},
    {"weight", "wait"},
    {"allowed", "aloud"},
    {"bear", "bare"},
    {"course", "coarse"},
    {"desert", "dessert"},
    {"great", "grate"},
    {"lesson", "lessen"},
    {"mail", "male"},
    {"scene", "seen"},
    {"soul", "sole"},
    {"some", "sum"},
    {"sun", "son"},
    {"tale", "tail"},
    {"weak", "week"},
]

# 170+ common misspellings & student error mappings
COMMON_MISSPELLINGS = {
    "definitely": "definately",
    "accommodate": "accomodate",
    "accommodation": "accomodation",
    "achieve": "acheive",
    "across": "accross",
    "aggressive": "agressive",
    "almost": "allmost",
    "already": "allready",
    "although": "altho",
    "always": "allways",
    "amateur": "amatuer",
    "apparent": "apparant",
    "appearance": "appearence",
    "argument": "arguement",
    "basically": "basicly",
    "beginning": "begining",
    "believe": "beleive",
    "bizarre": "bizzare",
    "business": "buisness",
    "calendar": "calender",
    "category": "catagory",
    "cemetery": "cemetary",
    "colleague": "collegue",
    "coming": "comming",
    "committee": "commitee",
    "completely": "completly",
    "conscious": "concious",
    "curiosity": "curiousity",
    "disappear": "dissapear",
    "disappoint": "dissapoint",
    "embarrass": "embarass",
    "environment": "enviroment",
    "existence": "existance",
    "familiar": "familier",
    "finally": "finaly",
    "foreign": "foriegn",
    "forward": "foward",
    "further": "futher",
    "government": "goverment",
    "grammar": "grammer",
    "guarantee": "garantee",
    "guidance": "guidence",
    "happened": "happend",
    "harass": "harrass",
    "height": "heigth",
    "humorous": "humerous",
    "immediately": "immediatly",
    "incidentally": "incidently",
    "independent": "independant",
    "intelligence": "inteligence",
    "interesting": "intresting",
    "interrupt": "interupt",
    "knowledge": "knowlege",
    "leisure": "liesure",
    "library": "libary",
    "lightning": "lightening",
    "maintenance": "maintainance",
    "millennium": "millenium",
    "miniature": "minature",
    "mischievous": "mischevious",
    "misspell": "mispell",
    "necessary": "neccessary",
    "noticeable": "noticable",
    "occasion": "ocassion",
    "occurred": "occured",
    "occurrence": "occurance",
    "opinion": "oppinion",
    "original": "orignal",
    "parallel": "paralell",
    "particular": "particuler",
    "pastime": "passtime",
    "performance": "preformance",
    "persistent": "persistant",
    "piece": "peice",
    "politician": "politican",
    "possession": "posession",
    "possible": "posible",
    "potatoes": "potatos",
    "preferred": "prefered",
    "presence": "presense",
    "privilege": "privelege",
    "probably": "probly",
    "pronunciation": "pronounciation",
    "publicly": "publically",
    "really": "realy",
    "receive": "recieve",
    "receipt": "reciept",
    "recommend": "recommed",
    "reference": "referance",
    "relevant": "relevent",
    "religious": "religous",
    "remember": "rember",
    "restaurant": "restaraunt",
    "rhythm": "rythm",
    "schedule": "schedual",
    "secretary": "secratary",
    "separate": "seperate",
    "similar": "similiar",
    "sincerely": "sinserely",
    "speech": "speach",
    "successful": "succesful",
    "surprise": "suprise",
    "tendency": "tendancy",
    "therefore": "therefor",
    "tomatoes": "tomatos",
    "tomorrow": "tomorow",
    "truly": "truely",
    "twelfth": "twelth",
    "unfortunately": "unfortunatly",
    "until": "untill",
    "unusual": "unusal",
    "weird": "wierd",
    "which": "wich",
    "writing": "writting",
    "written": "writen",
    "yesterday": "yesturday",
    "friend": "freind",
    "because": "becuase",
    "beautiful": "beatiful",
    "different": "diferent",
    "experience": "experiance",
    "important": "importent",
    "language": "langauge",
    "people": "peaple",
    "through": "thru",
    "thought": "thaught",
    "though": "tho",
    "enough": "enuff",
    "alright": "all right",
    "everyday": "every day",
    "anymore": "any more",
    "anytime": "any time",
    "cannot": "can not",
    "into": "in to",
    "onto": "on to",
    "without": "with out",
    "outside": "out side",
    "inside": "in side",
    "someone": "some one",
    "anyone": "any one",
    "everyone": "every one",
    "something": "some thing",
    "anything": "any thing",
    "everything": "every thing",
    "somewhere": "some where",
    "anywhere": "any where",
    "everywhere": "every where",
    "nowhere": "no where",
}

PREPOSITIONS = ["in", "on", "at", "for", "with", "to", "from", "by", "about", "of", "into", "onto"]
ARTICLES = ["a", "an", "the"]

# Memoize verb inflection lookups
VERB_CACHE: dict[str, list[str] | None] = {}


def match_casing(original: str, new_word: str) -> str:
    """Apply original casing style to new word."""
    if not original:
        return new_word
    if original.isupper():
        return new_word.upper()
    if original[0].isupper():
        return new_word.capitalize()
    return new_word.lower()


def apply_adjacent_key_typo(word: str) -> str:
    """Substitute an internal letter with an adjacent QWERTY key."""
    letters = list(word)
    alpha_indices = [i for i, c in enumerate(letters) if c.lower() in QWERTY_NEIGHBORS]
    if not alpha_indices:
        return word
    idx = random.choice(alpha_indices)
    char = letters[idx]
    lower_char = char.lower()
    neighbor = random.choice(QWERTY_NEIGHBORS[lower_char])
    letters[idx] = neighbor.upper() if char.isupper() else neighbor
    return "".join(letters)


def apply_letter_swap(word: str) -> str:
    """Swap two adjacent letters inside the word."""
    letters = list(word)
    swappable = [i for i in range(len(letters) - 1) if letters[i].isalpha() and letters[i + 1].isalpha()]
    if not swappable:
        return word
    idx = random.choice(swappable)
    letters[idx], letters[idx + 1] = letters[idx + 1], letters[idx]
    return "".join(letters)


def apply_dropped_letter(word: str) -> str:
    """Drop a letter from the word (preserving first letter)."""
    if len(word) <= 3:
        return word
    letters = list(word)
    drop_idx = random.randint(1, len(letters) - 1)
    letters.pop(drop_idx)
    return "".join(letters)


def apply_doubled_letter(word: str) -> str:
    """Duplicate a letter in the word."""
    if not word:
        return word
    letters = list(word)
    alpha_indices = [i for i, c in enumerate(letters) if c.isalpha()]
    if not alpha_indices:
        return word
    idx = random.choice(alpha_indices)
    letters.insert(idx, letters[idx].lower())
    return "".join(letters)


def apply_confusable_or_misspelling(word: str) -> str | None:
    """Replace word with its confusable homophone or common misspelling."""
    lw = word.lower()

    # Check homophone groups
    for group in HOMOPHONE_GROUPS:
        if lw in group:
            choices = [w for w in group if w != lw]
            if choices:
                replacement = random.choice(choices)
                return match_casing(word, replacement)

    # Check common misspellings dict
    if lw in COMMON_MISSPELLINGS:
        return match_casing(word, COMMON_MISSPELLINGS[lw])

    return None


def get_verb_inflections(word: str) -> list[str] | None:
    """Lookup alternative verb inflections using lemminflect with caching."""
    lw = word.lower()
    if lw in VERB_CACHE:
        return VERB_CACHE[lw]

    lem_tup = lemminflect.getLemma(lw, upos="VERB")
    if not lem_tup:
        VERB_CACHE[lw] = None
        return None

    lemma = lem_tup[0]
    all_infs = lemminflect.getAllInflections(lemma)
    if not all_infs:
        VERB_CACHE[lw] = None
        return None

    alternatives = []
    for tag, forms in all_infs.items():
        if tag.startswith("V"):
            for f in forms:
                if f.lower() != lw:
                    alternatives.append(f)

    res = list(set(alternatives)) if alternatives else None
    VERB_CACHE[lw] = res
    return res


def corrupt_sentence(sentence: str) -> tuple[str, list[str]]:
    """Corrupt 5-20% of words in a sentence, returning (corrupted_sentence, categories_applied)."""
    working_text = sentence
    categories = []

    # 1. Check for phrase confusables like "a lot" -> "alot"
    if re.search(r"\ba lot\b", working_text, re.IGNORECASE) and random.random() < 0.65:
        working_text = re.sub(r"\ba lot\b", "alot", working_text, count=1, flags=re.IGNORECASE)
        categories.append("confusables")

    # Fast tokenization: split into words and punctuation tokens
    tokens = re.findall(r"\w+|[^\w\s]", working_text)
    if len(tokens) <= 2:
        return working_text, ["untouched"]

    word_indices = [i for i, t in enumerate(tokens) if t[0].isalnum()]
    n_words = len(word_indices)
    if n_words == 0:
        return working_text, ["untouched"]

    # Compute target corruptions (5% - 20% of words)
    rate = random.uniform(0.05, 0.20)
    target_corruptions = max(1, min(n_words, round(rate * n_words)))

    indices_to_try = random.sample(word_indices, len(word_indices))
    applied_count = 0

    for idx in indices_to_try:
        if applied_count >= target_corruptions:
            break

        tok = tokens[idx]
        lw = tok.lower()
        action_roll = random.random()

        # 1. Confusables & Misspellings (weight ~30%)
        if action_roll < 0.30:
            repl = apply_confusable_or_misspelling(tok)
            if repl and repl != tok:
                tokens[idx] = repl
                categories.append("confusables")
                applied_count += 1
                continue

        # 2. Verb Agreement & Tense via lemminflect (weight ~25%)
        if action_roll < 0.55:
            alt_verbs = get_verb_inflections(tok)
            if alt_verbs:
                repl = match_casing(tok, random.choice(alt_verbs))
                if repl != tok:
                    tokens[idx] = repl
                    categories.append("verb_grammar")
                    applied_count += 1
                    continue

        # 3. Articles (weight ~15%)
        if lw in ARTICLES:
            if random.random() < 0.5:
                tokens[idx] = ""  # drop article
            else:
                other_articles = [a for a in ARTICLES if a != lw]
                tokens[idx] = match_casing(tok, random.choice(other_articles))
            categories.append("articles")
            applied_count += 1
            continue

        # 4. Prepositions (weight ~15%)
        if lw in PREPOSITIONS:
            other_prep = [p for p in PREPOSITIONS if p != lw]
            tokens[idx] = match_casing(tok, random.choice(other_prep))
            categories.append("prepositions")
            applied_count += 1
            continue

        # 5. Typos (adjacent key, swap, drop, double) (~35% fallback)
        typo_type = random.choice(["adjacent", "swap", "drop", "double"])
        corrupted_word = tok
        if typo_type == "adjacent":
            corrupted_word = apply_adjacent_key_typo(tok)
        elif typo_type == "swap":
            corrupted_word = apply_letter_swap(tok)
        elif typo_type == "drop":
            corrupted_word = apply_dropped_letter(tok)
        elif typo_type == "double":
            corrupted_word = apply_doubled_letter(tok)

        if corrupted_word != tok:
            tokens[idx] = corrupted_word
            categories.append("typos")
            applied_count += 1
            continue

    # Occasional casing, punctuation, or doubled word operations (~15% chance)
    if random.random() < 0.15:
        sub_op = random.choice(["casing", "comma", "period", "double_word"])
        if sub_op == "casing" and tokens and tokens[0]:
            tokens[0] = tokens[0].lower()
            categories.append("casing_punct")
        elif sub_op == "comma":
            comma_indices = [i for i, t in enumerate(tokens) if t == ","]
            if comma_indices:
                tokens.pop(random.choice(comma_indices))
                categories.append("casing_punct")
        elif sub_op == "period":
            if tokens and tokens[-1] in [".", "!", "?"]:
                tokens.pop()
                categories.append("casing_punct")
        elif sub_op == "double_word" and word_indices:
            dup_idx = random.choice(word_indices)
            tokens.insert(dup_idx, tokens[dup_idx])
            categories.append("doubled_words")

    # Reassemble tokens with natural spacing around punctuation
    assembled = " ".join([t for t in tokens if t])
    assembled = re.sub(r"\s+([,.:;?!'’])", r"\1", assembled)
    assembled = re.sub(r"\(\s+", "(", assembled)
    assembled = re.sub(r"\s+\)", ")", assembled)
    assembled = re.sub(r"\s+", " ", assembled).strip()

    if not categories:
        categories.append("untouched")

    return assembled, categories


# =========================================================================
# PHASE M6: TARGETED CORRUPTION GENERATORS FOR TOP 10 WEAK CATEGORIES
# =========================================================================

PREP_COLLOCATIONS = [
    (r"\bdepend on\b", "depend of"),
    (r"\brely on\b", "rely in"),
    (r"\binterested in\b", "interested for"),
    (r"\bgood at\b", "good in"),
    (r"\bbad at\b", "bad in"),
    (r"\barrive at\b", "arrive to"),
    (r"\barrive in\b", "arrive to"),
    (r"\bon the internet\b", "in the internet"),
    (r"\blook forward to\b", "look forward for"),
    (r"\bpay for\b", "pay to"),
    (r"\bwait for\b", "wait to"),
    (r"\blisten to\b", "listen at"),
    (r"\bcongratulate on\b", "congratulate for"),
    (r"\bat night\b", "in night"),
    (r"\bin the morning\b", "on the morning"),
    (r"\bin the afternoon\b", "on the afternoon"),
    (r"\bin the evening\b", "on the evening"),
]

UNCOUNTABLE_OR_SPORT_WORDS = [
    "football", "basketball", "tennis", "cricket", "chess", "soccer",
    "breakfast", "lunch", "dinner",
    "nature", "music", "advice", "information", "furniture", "luggage",
    "equipment", "homework", "knowledge", "research", "feedback",
    "biology", "mathematics", "physics", "chemistry", "history", "society",
]

UNCOUNTABLE_NOUNS = [
    "information", "advice", "furniture", "luggage", "equipment",
    "homework", "knowledge", "research", "feedback", "traffic",
]

ORTH_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ORTH_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ORTH_NATIONALITIES = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Italian", "Russian", "Indian", "American", "British", "European"]

MORPH_PAIRS = [
    (r"\bfluently\b", "fluent"),
    (r"\bquickly\b", "quick"),
    (r"\bbadly\b", "bad"),
    (r"\bextremely\b", "extreme"),
    (r"\bcarefully\b", "careful"),
    (r"\beasily\b", "easy"),
    (r"\bclearly\b", "clear"),
    (r"\bproperly\b", "proper"),
    (r"\bslowly\b", "slow"),
    (r"\bperfectly\b", "perfect"),
    (r"\bimportance\b", "important"),
    (r"\bdifference\b", "different"),
    (r"\bsuccess\b", "successful"),
    (r"\bconfidence\b", "confident"),
]


def corrupt_targeted_prep(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:PREP (Preposition replacement)."""
    for pat, repl in PREP_COLLOCATIONS:
        if re.search(pat, sentence, re.IGNORECASE):
            corrupted = re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE)
            if corrupted != sentence:
                return corrupted, True

    tokens = sentence.split()
    prep_map = {
        "in": ["on", "at"],
        "on": ["in", "at"],
        "at": ["in", "on"],
        "for": ["to", "of"],
        "to": ["for", "at"],
        "of": ["for", "about"],
        "with": ["by"],
        "by": ["with"],
        "about": ["of", "for"],
    }
    prep_indices = [i for i, t in enumerate(tokens) if t.lower() in prep_map]
    if prep_indices:
        idx = random.choice(prep_indices)
        orig = tokens[idx]
        repl = random.choice(prep_map[orig.lower()])
        tokens[idx] = match_casing(orig, repl)
        return " ".join(tokens), True

    return sentence, False


def corrupt_targeted_det_missing(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for M:DET (Missing article before singular/definite noun)."""
    matches = list(re.finditer(r"\b(a|an|the)\s+([A-Za-z]+)\b", sentence, re.IGNORECASE))
    if matches:
        m = random.choice(matches)
        start, end = m.span()
        noun = m.group(2)
        if start == 0:
            noun = noun.capitalize()
        corrupted = sentence[:start] + noun + sentence[end:]
        return corrupted, True
    return sentence, False


def corrupt_targeted_det_unnecessary(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for U:DET (Unnecessary article before sports, meals, uncountables)."""
    for word in UNCOUNTABLE_OR_SPORT_WORDS:
        pat = rf"\b{word}\b"
        if re.search(pat, sentence, re.IGNORECASE):
            preceded = re.search(rf"\b(the|a|an)\s+{word}\b", sentence, re.IGNORECASE)
            if not preceded:
                def add_the(m):
                    orig = m.group(0)
                    if orig.isupper():
                        return f"THE {orig}"
                    if orig[0].isupper():
                        return f"The {orig.lower()}"
                    return f"the {orig}"
                corrupted = re.sub(pat, add_the, sentence, count=1, flags=re.IGNORECASE)
                if corrupted != sentence:
                    return corrupted, True

    m_plur = re.search(r"\b(people|children|students|cars|computers|books)\b", sentence, re.IGNORECASE)
    if m_plur and not re.search(r"\bthe\s+(people|children|students|cars|computers|books)\b", sentence, re.IGNORECASE):
        start, end = m_plur.span()
        word = m_plur.group(0)
        prefix = "The " if start == 0 else "the "
        return sentence[:start] + prefix + word.lower() + sentence[end:], True

    return sentence, False


def corrupt_targeted_verb_tense(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:VERB:TENSE (Temporal tense inconsistency)."""
    past_markers = ["yesterday", "last year", "last week", "last month", "ago", "in the past", "previously"]
    has_past_marker = any(m in sentence.lower() for m in past_markers)

    past_to_pres = {
        "went": "go", "saw": "see", "bought": "buy", "came": "come", "took": "take",
        "made": "make", "found": "find", "said": "say", "gave": "give", "became": "become",
        "visited": "visit", "received": "receive", "walked": "walk", "worked": "work",
        "studied": "study", "decided": "decide", "wanted": "want", "started": "start",
        "watched": "watch", "played": "play", "called": "call", "met": "meet",
    }
    pres_to_past = {v: k for k, v in past_to_pres.items()}

    if has_past_marker:
        for past, pres in past_to_pres.items():
            if re.search(rf"\b{past}\b", sentence, re.IGNORECASE):
                return re.sub(rf"\b{past}\b", lambda m: match_casing(m.group(0), pres), sentence, count=1, flags=re.IGNORECASE), True

    pres_markers = ["now", "today", "currently", "every day", "always", "usually"]
    has_pres_marker = any(m in sentence.lower() for m in pres_markers)
    if has_pres_marker:
        for pres, past in pres_to_past.items():
            if re.search(rf"\b{pres}\b", sentence, re.IGNORECASE):
                return re.sub(rf"\b{pres}\b", lambda m: match_casing(m.group(0), past), sentence, count=1, flags=re.IGNORECASE), True

    for past, pres in past_to_pres.items():
        if re.search(rf"\b{past}\b", sentence, re.IGNORECASE):
            return re.sub(rf"\b{past}\b", lambda m: match_casing(m.group(0), pres), sentence, count=1, flags=re.IGNORECASE), True

    return sentence, False


def corrupt_targeted_verb_form(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:VERB:FORM (Gerund vs Infinitive & modal form)."""
    m_prep_ing = re.search(r"\b(for|by|without|after|before|about)\s+([a-z]+)ing\b", sentence, re.IGNORECASE)
    if m_prep_ing:
        prep = m_prep_ing.group(1)
        base = m_prep_ing.group(2)
        repl = f"{prep} {base}"
        return re.sub(r"\b" + re.escape(m_prep_ing.group(0)) + r"\b", repl, sentence, count=1), True

    gerund_verbs = ["enjoy", "avoid", "finish", "keep", "suggest", "mind", "consider", "practice"]
    for gv in gerund_verbs:
        m = re.search(rf"\b{gv}\s+([a-z]+)ing\b", sentence, re.IGNORECASE)
        if m:
            verb_ing = m.group(1)
            repl = f"{gv} to {verb_ing}"
            return re.sub(r"\b" + re.escape(m.group(0)) + r"\b", repl, sentence, count=1, flags=re.IGNORECASE), True

    inf_verbs = ["decide", "decided", "want", "wants", "wanted", "hope", "hopes", "hoped", "plan", "plans", "planned"]
    for iv in inf_verbs:
        m = re.search(rf"\b{iv}\s+to\s+([a-z]+)\b", sentence, re.IGNORECASE)
        if m:
            base_v = m.group(1)
            repl = f"{iv} {base_v}"
            return re.sub(r"\b" + re.escape(m.group(0)) + r"\b", repl, sentence, count=1, flags=re.IGNORECASE), True

    modals = ["can", "could", "must", "should", "will", "would"]
    for mod in modals:
        m = re.search(rf"\b{mod}\s+([a-z]+)\b", sentence, re.IGNORECASE)
        if m and m.group(1).lower() not in ["be", "have", "not", "the", "a", "an"]:
            base_v = m.group(1)
            repl = f"{mod} to {base_v}"
            return re.sub(r"\b" + re.escape(m.group(0)) + r"\b", repl, sentence, count=1, flags=re.IGNORECASE), True

    return sentence, False


def corrupt_targeted_noun_num(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:NOUN:NUM (Singular vs Plural & uncountables)."""
    for unc in UNCOUNTABLE_NOUNS:
        if re.search(rf"\b{unc}\b", sentence, re.IGNORECASE):
            return re.sub(rf"\b{unc}\b", f"{unc}s", sentence, count=1, flags=re.IGNORECASE), True

    quantifiers = ["two", "three", "four", "five", "many", "several", "both", "various", "numerous", "few"]
    for q in quantifiers:
        m = re.search(rf"\b{q}\s+([a-z]+)s\b", sentence, re.IGNORECASE)
        if m and not m.group(1).endswith("s"):
            sing = m.group(1)
            repl = f"{q} {sing}"
            return re.sub(r"\b" + re.escape(m.group(0)) + r"\b", repl, sentence, count=1, flags=re.IGNORECASE), True

    sing_quants = ["every", "each", "one"]
    for sq in sing_quants:
        m = re.search(rf"\b{sq}\s+([a-z]+)\b", sentence, re.IGNORECASE)
        if m and not m.group(1).endswith("s"):
            plur = m.group(1) + "s"
            repl = f"{sq} {plur}"
            return re.sub(r"\b" + re.escape(m.group(0)) + r"\b", repl, sentence, count=1, flags=re.IGNORECASE), True

    return sentence, False


def corrupt_targeted_complex_sva(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:VERB:SVA (Intervening phrases causing SVA error)."""
    patterns_sing_head = [
        (r"\b(quality|cost|price|variety|box|group|list|number|one|cause)\s+(of|for|in)\s+([a-z]+s)\s+(is)\b", r"\1 \2 \3 are"),
        (r"\b(quality|cost|price|variety|box|group|list|number|one|cause)\s+(of|for|in)\s+([a-z]+s)\s+(was)\b", r"\1 \2 \3 were"),
        (r"\b(quality|cost|price|variety|box|group|list|number|one|cause)\s+(of|for|in)\s+([a-z]+s)\s+(has)\b", r"\1 \2 \3 have"),
    ]
    for pat, repl in patterns_sing_head:
        if re.search(pat, sentence, re.IGNORECASE):
            return re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE), True

    patterns_plur_head = [
        (r"\b(members|teachers|students|benefits|results|causes)\s+(of|in|at)\s+([a-z]+)\s+(are)\b", r"\1 \2 \3 is"),
        (r"\b(members|teachers|students|benefits|results|causes)\s+(of|in|at)\s+([a-z]+)\s+(were)\b", r"\1 \2 \3 was"),
        (r"\b(members|teachers|students|benefits|results|causes)\s+(of|in|at)\s+([a-z]+)\s+(have)\b", r"\1 \2 \3 has"),
    ]
    for pat, repl in patterns_plur_head:
        if re.search(pat, sentence, re.IGNORECASE):
            return re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE), True

    sva_pairs = [
        (r"\b(he|she|it|everyone|someone)\s+(is)\b", r"\1 are"),
        (r"\b(he|she|it|everyone|someone)\s+(was)\b", r"\1 were"),
        (r"\b(he|she|it|everyone|someone)\s+(has)\b", r"\1 have"),
        (r"\b(they|we|you)\s+(are)\b", r"\1 is"),
        (r"\b(they|we|you)\s+(were)\b", r"\1 was"),
        (r"\b(they|we|you)\s+(have)\b", r"\1 has"),
    ]
    for pat, repl in sva_pairs:
        if re.search(pat, sentence, re.IGNORECASE):
            return re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE), True

    return sentence, False


def corrupt_targeted_morphology(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:MORPH (Derivational morphology: adjective vs adverb)."""
    for pat, repl in MORPH_PAIRS:
        if re.search(pat, sentence, re.IGNORECASE):
            return re.sub(pat, lambda m: match_casing(m.group(0), repl), sentence, count=1, flags=re.IGNORECASE), True

    m_ly = re.search(r"\b([a-z]{4,})ly\b", sentence, re.IGNORECASE)
    if m_ly:
        base = m_ly.group(1)
        return re.sub(r"\b" + re.escape(m_ly.group(0)) + r"\b", base, sentence, count=1, flags=re.IGNORECASE), True

    return sentence, False


def corrupt_targeted_orthography(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:ORTH (Orthography & Casing)."""
    for day in ORTH_DAYS:
        if re.search(rf"\b{day}\b", sentence):
            return re.sub(rf"\b{day}\b", day.lower(), sentence, count=1), True

    for month in ORTH_MONTHS:
        if re.search(rf"\b{month}\b", sentence):
            return re.sub(rf"\b{month}\b", month.lower(), sentence, count=1), True

    for nat in ORTH_NATIONALITIES:
        if re.search(rf"\b{nat}\b", sentence):
            return re.sub(rf"\b{nat}\b", nat.lower(), sentence, count=1), True

    if sentence and sentence[0].isupper():
        return sentence[0].lower() + sentence[1:], True

    return sentence, False


def corrupt_targeted_word_order(sentence: str) -> tuple[str, bool]:
    """Targeted corruption for R:WO (Word order & adverb placement)."""
    adv_match = re.search(r"\b(he|she|they|we|i)\s+(always|usually|often|never|sometimes)\b", sentence, re.IGNORECASE)
    if adv_match:
        subj = adv_match.group(1)
        adv = adv_match.group(2)
        repl = f"{adv.capitalize()} {subj.lower()}"
        return re.sub(r"\b" + re.escape(adv_match.group(0)) + r"\b", repl, sentence, count=1), True

    ind_match = re.search(r"\b(where|what|how|why)\s+(he|she|it|they)\s+(is|are|was|were)\b", sentence, re.IGNORECASE)
    if ind_match:
        wh = ind_match.group(1)
        subj = ind_match.group(2)
        verb = ind_match.group(3)
        repl = f"{wh} {verb} {subj}"
        return re.sub(r"\b" + re.escape(ind_match.group(0)) + r"\b", repl, sentence, count=1), True

    adv_obj = re.search(r"\b([a-z]+s?)\s+([a-z]+)\s+(often|always|fluently|well|very much)\b", sentence, re.IGNORECASE)
    if adv_obj:
        verb = adv_obj.group(1)
        obj = adv_obj.group(2)
        adv = adv_obj.group(3)
        repl = f"{verb} {adv} {obj}"
        return re.sub(r"\b" + re.escape(adv_obj.group(0)) + r"\b", repl, sentence, count=1), True

    return sentence, False


TARGETED_GENERATORS = [
    ("R:PREP", corrupt_targeted_prep),
    ("M:DET", corrupt_targeted_det_missing),
    ("U:DET", corrupt_targeted_det_unnecessary),
    ("R:VERB:TENSE", corrupt_targeted_verb_tense),
    ("R:VERB:FORM", corrupt_targeted_verb_form),
    ("R:NOUN:NUM", corrupt_targeted_noun_num),
    ("R:VERB:SVA", corrupt_targeted_complex_sva),
    ("R:MORPH", corrupt_targeted_morphology),
    ("R:ORTH", corrupt_targeted_orthography),
    ("R:WO", corrupt_targeted_word_order),
]


def generate_targeted_pairs(clean_pool: list[str], total_rows: int = TARGETED_TOTAL_ROWS) -> list[dict]:
    """Generate ~30,000 targeted error pairs balanced equally across the 10 weak categories."""
    target_per_category = total_rows // len(TARGETED_GENERATORS)
    print("=" * 80, flush=True)
    print(f"PHASE M6: GENERATING {total_rows:,} TARGETED ERROR PAIRS ({target_per_category:,} per category)", flush=True)
    print("=" * 80, flush=True)

    targeted_pairs = []
    category_counts = Counter()
    random.seed(42)

    # Shuffle clean pool for diversity
    pool_indices = list(range(len(clean_pool)))
    random.shuffle(pool_indices)

    for cat_name, gen_func in TARGETED_GENERATORS:
        collected = 0
        pool_cursor = 0
        attempts = 0
        max_attempts = len(clean_pool) * 3

        while collected < target_per_category and attempts < max_attempts:
            clean_sent = clean_pool[pool_indices[pool_cursor % len(pool_indices)]]
            pool_cursor += 1
            attempts += 1

            corrupted, success = gen_func(clean_sent)
            if success and corrupted != clean_sent and len(corrupted.split()) <= MAX_WORDS:
                targeted_pairs.append({
                    "input": corrupted,
                    "output": clean_sent,
                    "category": cat_name,
                })
                collected += 1
                category_counts[cat_name] += 1

        print(f"  [{cat_name:14s}] Generated {collected:,} pairs (searched {attempts:,} candidates)", flush=True)

    random.shuffle(targeted_pairs)

    # Save to data/processed/targeted.csv
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(TARGETED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output"])
        for p in targeted_pairs:
            writer.writerow([p["input"], p["output"]])

    print(f"\n[SUCCESS] Saved {len(targeted_pairs):,} targeted pairs -> {TARGETED_CSV} ({TARGETED_CSV.stat().st_size:,} bytes)", flush=True)
    return targeted_pairs


def load_clean_sentence_pool() -> list[str]:
    """Load and deduplicate clean sentences from C4 outputs, W&I outputs, and CoEdIT targets."""
    print("Loading clean sentence pool...", flush=True)
    pool = set()

    # 1. C4 outputs
    c4_train_path = PROCESSED_DIR / "train.csv"
    if c4_train_path.exists():
        df_c4 = pd.read_csv(c4_train_path, encoding="utf-8")
        for s in df_c4["output"].dropna().astype(str):
            s_clean = s.strip()
            w_len = len(s_clean.split())
            if 4 <= w_len <= MAX_WORDS:
                pool.add(s_clean)
        print(f"  Loaded C4 clean targets: pool size now {len(pool):,}", flush=True)

    # 2. W&I corrected outputs
    wi_train_path = PROCESSED_DIR / "wi_train.csv"
    if wi_train_path.exists():
        df_wi = pd.read_csv(wi_train_path, encoding="utf-8")
        for s in df_wi["output"].dropna().astype(str):
            s_clean = s.strip()
            w_len = len(s_clean.split())
            if 4 <= w_len <= MAX_WORDS:
                pool.add(s_clean)
        print(f"  Loaded W&I clean targets: pool size now {len(pool):,}", flush=True)

    # 3. CoEdIT targets
    coedit_path = PROCESSED_DIR / "coedit_gec.csv"
    if coedit_path.exists():
        df_coedit = pd.read_csv(coedit_path, encoding="utf-8")
        for s in df_coedit["output"].dropna().astype(str):
            s_clean = s.strip()
            w_len = len(s_clean.split())
            if 4 <= w_len <= MAX_WORDS:
                pool.add(s_clean)
        print(f"  Loaded CoEdIT clean targets: pool size now {len(pool):,}", flush=True)

    clean_list = list(pool)
    random.seed(42)
    random.shuffle(clean_list)
    print(f"Final deduplicated clean sentence pool size: {len(clean_list):,} sentences.\n", flush=True)
    return clean_list


def generate_noise_pairs(clean_pool: list[str]) -> list[dict]:
    """Generate exactly TARGET_NOISE_ROWS noisy pairs (wrong -> right)."""
    print(f"Generating {TARGET_NOISE_ROWS:,} synthetic noise pairs...", flush=True)
    noise_pairs = []
    category_counter = Counter()

    n_pool = len(clean_pool)
    random.seed(42)

    for i in range(TARGET_NOISE_ROWS):
        clean_sent = clean_pool[i % n_pool]

        # 10% kept untouched
        if random.random() < UNTOUCHED_RATIO:
            corrupted = clean_sent
            cats = ["untouched"]
        else:
            corrupted, cats = corrupt_sentence(clean_sent)
            if not corrupted or len(corrupted.split()) > MAX_WORDS:
                corrupted = clean_sent
                cats = ["untouched"]

        for c in set(cats):
            category_counter[c] += 1

        noise_pairs.append({
            "input": corrupted,
            "output": clean_sent,
            "categories": "+".join(cats),
        })

        if (i + 1) % 50_000 == 0:
            print(f"  Generated {i + 1:,} / {TARGET_NOISE_ROWS:,} pairs ({(i + 1)/TARGET_NOISE_ROWS*100:.0f}%)...", flush=True)

    print("\nCategory Distribution in Synthetic Noise Pairs:", flush=True)
    for cat, cnt in category_counter.most_common():
        pct = (cnt / TARGET_NOISE_ROWS) * 100
        print(f"  - {cat:20s}: {cnt:>8,} ({pct:>5.1f}%)", flush=True)

    # Save to noise_pairs.csv (input, output)
    with open(NOISE_PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output"])
        for p in noise_pairs:
            writer.writerow([p["input"], p["output"]])

    print(f"\nSaved synthetic noise pairs -> {NOISE_PAIRS_CSV} ({len(noise_pairs):,} rows, {NOISE_PAIRS_CSV.stat().st_size:,} bytes)\n", flush=True)
    return noise_pairs


def build_stage_files():
    """Build stage1.csv, stage1_colab.csv, and stage2.csv."""
    print("=" * 80, flush=True)
    print("BUILDING MULTI-STAGE TRAINING FILES", flush=True)
    print("=" * 80, flush=True)

    # 1. Build Stage 1
    # stage1.csv = noise_pairs + C4 pairs + coedit_gec (shuffled, ~688k rows)
    print("\nAssembling Stage 1 dataset...", flush=True)
    df_noise = pd.read_csv(NOISE_PAIRS_CSV, encoding="utf-8")
    df_c4 = pd.read_csv(PROCESSED_DIR / "train.csv", encoding="utf-8")
    df_coedit = pd.read_csv(PROCESSED_DIR / "coedit_gec.csv", encoding="utf-8")

    print(f"  * Noise pairs:   {len(df_noise):,} rows", flush=True)
    print(f"  * C4 pairs:      {len(df_c4):,} rows", flush=True)
    print(f"  * CoEdIT GEC:    {len(df_coedit):,} rows", flush=True)

    stage1_df = pd.concat([df_noise[["input", "output"]], df_c4[["input", "output"]], df_coedit[["input", "output"]]], ignore_index=True)
    stage1_df = stage1_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    stage1_df.to_csv(STAGE1_CSV, index=False, encoding="utf-8")
    print(f"Saved Stage 1 dataset -> {STAGE1_CSV} ({len(stage1_df):,} rows, {STAGE1_CSV.stat().st_size:,} bytes)", flush=True)

    # stage1_colab.csv (subsample to 150k for Colab session)
    colab_subsample_size = min(150_000, len(stage1_df))
    stage1_colab_df = stage1_df.sample(n=colab_subsample_size, random_state=42).reset_index(drop=True)
    stage1_colab_df.to_csv(STAGE1_COLAB_CSV, index=False, encoding="utf-8")
    print(f"Saved Stage 1 Colab subsample -> {STAGE1_COLAB_CSV} ({len(stage1_colab_df):,} rows, {STAGE1_COLAB_CSV.stat().st_size:,} bytes)", flush=True)

    # 2. Build Stage 2
    # stage2.csv = wi_train + fce_train + coedit_gec, each real pair oversampled x3
    print("\nAssembling Stage 2 dataset (x3 oversampling of gold pairs)...", flush=True)
    df_wi = pd.read_csv(PROCESSED_DIR / "wi_train.csv", encoding="utf-8")
    df_fce = pd.read_csv(PROCESSED_DIR / "fce_train.csv", encoding="utf-8")
    df_coedit = pd.read_csv(PROCESSED_DIR / "coedit_gec.csv", encoding="utf-8")

    print(f"  * W&I Train (raw):   {len(df_wi):,} rows", flush=True)
    print(f"  * FCE Train (raw):   {len(df_fce):,} rows", flush=True)
    print(f"  * CoEdIT GEC (raw):  {len(df_coedit):,} rows", flush=True)

    gold_combined = pd.concat([df_wi[["input", "output"]], df_fce[["input", "output"]], df_coedit[["input", "output"]]], ignore_index=True)
    stage2_df = pd.concat([gold_combined, gold_combined, gold_combined], ignore_index=True)
    stage2_df = stage2_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    stage2_df.to_csv(STAGE2_CSV, index=False, encoding="utf-8")
    print(f"Saved Stage 2 dataset -> {STAGE2_CSV} ({len(stage2_df):,} rows, {STAGE2_CSV.stat().st_size:,} bytes)\n", flush=True)


def print_random_examples(noise_pairs: list[dict], count: int = 20):
    """Print count random noise examples showing corrupted input, clean target, and categories."""
    print("=" * 80, flush=True)
    print("20 RANDOM NOISE CORRUPTION EXAMPLES", flush=True)
    print("=" * 80, flush=True)

    random.seed(42)
    sample_indices = random.sample(range(len(noise_pairs)), count)

    for rank, idx in enumerate(sample_indices, 1):
        pair = noise_pairs[idx]
        cats = pair.get("categories", "unknown")
        print(f"[{rank:02d}] Categories: [{cats}]", flush=True)
        print(f"     Input (wrong):  {pair['input']}", flush=True)
        print(f"     Output (right): {pair['output']}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Noise & Targeted GEC Dataset Generator")
    parser.add_argument("--targeted", action="store_true", help="Generate targeted.csv (~30k pairs for top 10 weak categories)")
    parser.add_argument("--full", action="store_true", help="Generate full noise_pairs.csv (300k) and stage files")
    args = parser.parse_args()

    clean_pool = load_clean_sentence_pool()

    if args.targeted:
        print("=" * 80, flush=True)
        print("PHASE M6: TARGETED DATASET GENERATION", flush=True)
        print("=" * 80, flush=True)
        targeted_pairs = generate_targeted_pairs(clean_pool, total_rows=TARGETED_TOTAL_ROWS)
        print_random_examples(targeted_pairs, count=15)
        print("[COMPLETE] Phase M6 targeted dataset generated successfully.", flush=True)
        return

    print("=" * 80, flush=True)
    print("PHASE M2: SYNTHETIC NOISE & STAGED DATASET GENERATION", flush=True)
    print("=" * 80, flush=True)

    noise_pairs = generate_noise_pairs(clean_pool)
    build_stage_files()
    print_random_examples(noise_pairs, count=20)

    # Also generate targeted dataset
    targeted_pairs = generate_targeted_pairs(clean_pool, total_rows=TARGETED_TOTAL_ROWS)
    print("[COMPLETE] Phase M2 & Phase M6 data pipelines finished successfully.", flush=True)


if __name__ == "__main__":
    main()
