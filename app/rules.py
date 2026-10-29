import re
from dataclasses import dataclass
from typing import List, Tuple, Set
from app.nlp import analyze


@dataclass
class Edit:
    position: int
    old: str
    new: str
    rule: str


IRREGULAR_THIRD_PERSON = {
    "have": "has",
    "do": "does",
    "go": "goes",
    "be": "is",
    "am": "is",
    "are": "is",
}

IRREGULAR_BASE_FORM = {
    "has": "have",
    "does": "do",
    "goes": "go",
    "is": "are",
}

IRREGULAR_PAST_VERBS = {
    "was", "were", "had", "did", "said", "ate", "went", "saw", "ran", "came",
    "bought", "brought", "thought", "wrote", "spoke", "drove", "took", "gave",
    "found", "told", "became", "felt", "left", "kept", "began", "knew", "stood",
    "lost", "met", "paid", "sat", "sent", "built", "spent", "fell", "hit", "led",
    "heard", "held", "understood", "chose", "swam", "sang", "drank", "grew",
    "drew", "flew", "threw", "wore", "rode", "rose", "won", "sold", "cost",
    "hurt", "shut"
}

MODAL_VERBS = {
    "can", "could", "will", "would", "shall", "should", "may", "might", "must"
}

COMMON_ADVERBS = {
    "always", "never", "often", "usually", "sometimes", "rarely", "seldom",
    "frequently", "definitely", "certainly", "probably", "actually", "really",
    "simply", "just", "already", "still", "also", "currently", "now", "slowly",
    "quickly", "happily", "easily", "hardly", "barely"
}

SILENT_H_PREFIXES = ("hour", "honest", "honor", "honour", "heir")
U_YOO_PREFIXES = (
    "univers", "uniform", "union", "unit", "uniqu", "unanim",
    "usage", "user", "useful", "useless", "usual", "utensil",
    "utilit", "utiliz", "utopi", "unicorn"
)
EU_PREFIXES = ("europ", "euphem", "eulog", "euphor", "euro", "ewe")
O_WA_PREFIXES = ("one", "once")
ACRONYM_VOWEL_LETTERS = set("AEFHILMNORSX")


def match_casing(original: str, target: str) -> str:
    """Preserve uppercase, capitalized, or lowercase format."""
    if original.isupper():
        return target.upper()
    if original and original[0].isupper():
        return target.capitalize()
    return target.lower()


def is_already_third_person(verb: str) -> bool:
    """Check if a verb token is already in third-person singular form."""
    w = verb.lower()
    if w in {"is", "has", "does", "goes"}:
        return True
    if w.endswith("ies") and w not in {"ties", "dies", "lies"}:
        return True
    if w.endswith(("sses", "zzes", "xes", "ches", "shes", "oes")):
        return True
    if w.endswith("s") and not w.endswith(("ss", "zz", "us", "is", "as")):
        return True
    return False


def to_third_person(verb: str) -> str:
    """
    Inflect a base verb to its third-person singular form.
    Handles -es after s/x/z/ch/sh/o, consonant+y -> -ies, and irregulars.
    """
    v = verb.lower()
    if v in IRREGULAR_THIRD_PERSON:
        res = IRREGULAR_THIRD_PERSON[v]
    elif len(v) > 1 and v.endswith("y") and v[-2] not in "aeiou":
        res = v[:-1] + "ies"
    elif v.endswith(("s", "x", "z", "ch", "sh", "o")):
        res = v + "es"
    else:
        res = v + "s"
    return match_casing(verb, res)


def to_base_form(verb: str) -> str:
    """
    Reduce a third-person singular verb to its base form.
    Handles -ies -> -y, -es after s/x/z/ch/sh/o, and irregulars.
    """
    v = verb.lower()
    if v in IRREGULAR_BASE_FORM:
        res = IRREGULAR_BASE_FORM[v]
    elif v.endswith("ies"):
        if v in ("ties", "dies", "lies"):
            res = v[:-1]
        else:
            res = v[:-3] + "y"
    elif v.endswith(("sses", "zzes", "xes", "ches", "shes", "oes")):
        res = v[:-2]
    elif v.endswith("s") and not v.endswith("ss"):
        res = v[:-1]
    else:
        res = v
    return match_casing(verb, res)


def starts_with_vowel_sound(word: str) -> bool:
    """
    Determine if a word begins with a vowel sound phonetically.
    Properly handles silent 'h' (an hour), 'yoo' sounds (a university),
    'w' sounds (a one-time), and initialisms (an MBA).
    """
    clean = re.sub(r"^[^\w]+", "", word.strip().lower())
    if not clean:
        return False

    orig_clean = re.sub(r"^[^\w]+", "", word.strip())
    # Acronyms pronounced letter-by-letter
    if len(orig_clean) >= 2 and orig_clean[:2].isupper():
        return orig_clean[0].upper() in ACRONYM_VOWEL_LETTERS

    # Silent H -> vowel sound
    if clean.startswith(SILENT_H_PREFIXES):
        return True

    # Voiced H -> consonant sound
    if clean.startswith("h"):
        return False

    # European, Euro -> /j/ consonant sound
    if clean.startswith(EU_PREFIXES):
        return False

    # One, Once -> /w/ consonant sound
    if clean.startswith(O_WA_PREFIXES):
        return False

    # University, Uniform, Unique -> /juː/ consonant sound
    if clean.startswith(U_YOO_PREFIXES):
        return False

    # Standard vowel check
    if clean[0] in "aeiou":
        return True

    return False


def find_quoted_intervals(text: str) -> List[Tuple[int, int]]:
    """
    Find all character intervals in text that are enclosed in quotation marks.
    """
    intervals = []
    # Double quotes (standard and curly)
    for m in re.finditer(r'"([^"]*)"|“([^”]*)”', text):
        intervals.append((m.start(), m.end()))
    # Single quotes (standard and curly), excluding apostrophes in words
    for m in re.finditer(r"(?<!\w)'([^']*)'(?!\w)|‘([^’]*)’", text):
        intervals.append((m.start(), m.end()))
    return intervals


def is_token_in_quotes(token_span: Tuple[int, int], quote_intervals: List[Tuple[int, int]]) -> bool:
    """Check if a token falls within any quoted section."""
    start, end = token_span
    for q_start, q_end in quote_intervals:
        if start >= q_start and end <= q_end:
            return True
    return False


def is_past_tense(token: str, tag: str) -> bool:
    """Check if token is a genuine past tense verb."""
    t_lower = token.lower()
    if t_lower in IRREGULAR_PAST_VERBS:
        return True
    if tag == "VBD" and t_lower.endswith("ed") and t_lower not in {"need", "feed", "bleed", "speed"}:
        return True
    return False


def apply_rules(text: str) -> Tuple[str, List[Edit]]:
    """
    Apply deterministic linguistic rules to text.
    Returns:
        corrected_text: the corrected string
        edits: list of Edit objects detailing what was changed
    """
    clean_text, tokens, tags, spans = analyze(text)
    if not tokens:
        return clean_text, []

    quote_intervals = find_quoted_intervals(clean_text)
    edits: List[Edit] = []
    edited_positions: Set[int] = set()

    n = len(tokens)

    # 1. Subject-Verb Agreement, is/are, has/have, does/do
    for i in range(n):
        tok_lower = tokens[i].lower()
        tag = tags[i]

        # Protected tokens (quotes or proper nouns) cannot initiate rules or be modified
        if is_token_in_quotes(spans[i], quote_intervals) or tag in {"NNP", "NNPS"}:
            continue

        is_he_she_it = tok_lower in {"he", "she", "it"}
        is_i_you_we_they = tok_lower in {"i", "you", "we", "they"}
        is_singular_noun = (tag == "NN" and tok_lower not in {"i", "you", "we", "they"})
        is_plural_noun = (tag == "NNS")

        # Skip tokens that are not valid subjects
        if not (is_he_she_it or is_i_you_we_they or is_singular_noun or is_plural_noun):
            continue

        # Look ahead for verb, skipping genuine frequency/manner adverbs
        j = i + 1
        while j < n and (tokens[j].lower() in COMMON_ADVERBS or (tags[j] in {"RB", "RBR", "RBS"} and tokens[j].lower() in COMMON_ADVERBS)):
            j += 1

        if j >= n or j in edited_positions:
            continue

        # Never touch quoted tokens or proper nouns
        if is_token_in_quotes(spans[j], quote_intervals) or tags[j] in {"NNP", "NNPS"}:
            continue

        v_tok = tokens[j]
        v_lower = v_tok.lower()
        v_tag = tags[j]

        # Do not alter if preceded by modal or infinitive marker
        if v_lower in MODAL_VERBS or v_tag == "MD" or v_lower == "to" or v_tag == "TO":
            continue

        if j > 0 and (tokens[j - 1].lower() in MODAL_VERBS or tags[j - 1] in {"MD", "TO"} or tokens[j - 1].lower() == "to"):
            continue

        # Auxiliary negative check: "does not / did not / do not"
        if j > 1 and tokens[j - 1].lower() in {"not", "n't"} and tokens[j - 2].lower() in {"do", "does", "did"}:
            continue

        # Specific auxiliary checks (is/are, has/have, does/do) apply to pronouns and nouns:
        # A. Singular subjects (he/she/it or singular noun)
        if is_he_she_it or is_singular_noun:
            # is / are mismatch: "She are" -> "She is"
            if v_lower == "are":
                new_verb = match_casing(v_tok, "is")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="is_are"))
                edited_positions.add(j)
                continue

            # has / have mismatch: "He have" -> "He has"
            if v_lower == "have":
                new_verb = match_casing(v_tok, "has")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="has_have"))
                edited_positions.add(j)
                continue

            # does / do mismatch: "He do" -> "He does"
            if v_lower == "do":
                new_verb = match_casing(v_tok, "does")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="does_do"))
                edited_positions.add(j)
                continue

        # B. Plural subjects or 'I' (they/we/you/I or plural noun)
        if is_i_you_we_they or is_plural_noun:
            # is / are mismatch: "They is" -> "They are", "I is" -> "I am"
            if v_lower == "is":
                target = "am" if tok_lower == "i" else "are"
                new_verb = match_casing(v_tok, target)
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="is_are"))
                edited_positions.add(j)
                continue

            if tok_lower == "i" and v_lower == "are":
                new_verb = match_casing(v_tok, "am")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="is_are"))
                edited_positions.add(j)
                continue

            # has / have mismatch: "I has" -> "I have", "They has" -> "They have"
            if v_lower == "has":
                new_verb = match_casing(v_tok, "have")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="has_have"))
                edited_positions.add(j)
                continue

            # does / do mismatch: "They does" -> "They do"
            if v_lower == "does":
                new_verb = match_casing(v_tok, "do")
                edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="does_do"))
                edited_positions.add(j)
                continue

        # C. General verb inflection for pronouns:
        # he / she / it + base verb -> third person
        if is_he_she_it:
            if not is_already_third_person(v_tok) and not is_past_tense(v_tok, v_tag) and v_tag not in {"VBG", "VBN"}:
                # Ensure it's not a non-verb POS
                if v_tag not in {"PRP", "PRP$", "IN", "DT", "CC", "CD", ".", ",", ":", "POS"}:
                    new_verb = to_third_person(v_tok)
                    if new_verb.lower() != v_lower:
                        edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="subject_verb_agreement"))
                        edited_positions.add(j)
                        continue

        # I / you / we / they + third person -> base form
        elif is_i_you_we_they:
            if is_already_third_person(v_tok) and not is_past_tense(v_tok, v_tag) and v_tag not in {"VBG", "VBN"}:
                if v_tag not in {"PRP", "PRP$", "IN", "DT", "CC", "CD", ".", ",", ":", "POS"}:
                    new_verb = to_base_form(v_tok)
                    if new_verb.lower() != v_lower:
                        edits.append(Edit(position=j, old=v_tok, new=new_verb, rule="subject_verb_agreement"))
                        edited_positions.add(j)
                        continue

    # 2. Indefinite article rule (a / an)
    for i in range(n):
        if i in edited_positions:
            continue

        tok = tokens[i]
        tok_lower = tok.lower()

        if tok_lower not in {"a", "an"}:
            continue

        # Never touch quoted text or proper nouns
        if is_token_in_quotes(spans[i], quote_intervals) or tags[i] in {"NNP", "NNPS"}:
            continue

        # Find the next non-punctuation word
        next_idx = i + 1
        while next_idx < n and re.match(r"^[^\w]+$", tokens[next_idx]):
            next_idx += 1

        if next_idx >= n:
            continue

        next_word = tokens[next_idx]
        vowel_sound = starts_with_vowel_sound(next_word)

        if tok_lower == "a" and vowel_sound:
            new_art = match_casing(tok, "an")
            edits.append(Edit(position=i, old=tok, new=new_art, rule="a_an"))
            edited_positions.add(i)
        elif tok_lower == "an" and not vowel_sound:
            new_art = match_casing(tok, "a")
            edits.append(Edit(position=i, old=tok, new=new_art, rule="a_an"))
            edited_positions.add(i)

    # Reconstruct text using exact character spans from right to left
    span_sorted_edits = sorted(edits, key=lambda e: spans[e.position][0], reverse=True)

    result_text = clean_text
    for edit in span_sorted_edits:
        start, end = spans[edit.position]
        result_text = result_text[:start] + edit.new + result_text[end:]

    return result_text, edits
