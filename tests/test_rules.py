import pytest
from app.rules import (
    apply_rules,
    to_third_person,
    to_base_form,
    starts_with_vowel_sound,
    Edit,
)


# ---------------------------------------------------------------------------
# 1. Primary Report Test Cases (8 specified cases)
# ---------------------------------------------------------------------------
REPORT_TEST_CASES = [
    ("She go to college every day.", "She goes to college every day."),
    ("He eat an apple.", "He eats an apple."),
    ("They goes to school.", "They go to school."),
    ("She are my friend.", "She is my friend."),
    ("They is playing football.", "They are playing football."),
    ("He have a car.", "He has a car."),
    ("I has a book.", "I have a book."),
    ("She is a engineer.", "She is an engineer."),
]


@pytest.mark.parametrize("input_text, expected_output", REPORT_TEST_CASES)
def test_report_cases(input_text, expected_output):
    corrected, edits = apply_rules(input_text)
    assert corrected == expected_output
    assert len(edits) >= 1
    for edit in edits:
        assert isinstance(edit, Edit)
        assert edit.rule in {"subject_verb_agreement", "is_are", "has_have", "does_do", "a_an"}


# ---------------------------------------------------------------------------
# 2. Verb Inflection Routines (General coverage for ANY verb)
# ---------------------------------------------------------------------------
def test_verb_inflection_third_person():
    # Irregulars
    assert to_third_person("go") == "goes"
    assert to_third_person("have") == "has"
    assert to_third_person("do") == "does"
    assert to_third_person("be") == "is"
    assert to_third_person("am") == "is"
    assert to_third_person("are") == "is"

    # Consonant + y -> -ies
    assert to_third_person("study") == "studies"
    assert to_third_person("carry") == "carries"
    assert to_third_person("fly") == "flies"
    assert to_third_person("try") == "tries"
    assert to_third_person("cry") == "cries"

    # Vowel + y -> adds s
    assert to_third_person("play") == "plays"
    assert to_third_person("stay") == "stays"
    assert to_third_person("enjoy") == "enjoys"
    assert to_third_person("buy") == "buys"

    # Ends with s, x, z, ch, sh, o -> adds es
    assert to_third_person("watch") == "watches"
    assert to_third_person("pass") == "passes"
    assert to_third_person("buzz") == "buzzes"
    assert to_third_person("fix") == "fixes"
    assert to_third_person("wash") == "washes"
    assert to_third_person("teach") == "teaches"
    assert to_third_person("catch") == "catches"

    # Standard verbs -> adds s
    assert to_third_person("eat") == "eats"
    assert to_third_person("work") == "works"
    assert to_third_person("write") == "writes"
    assert to_third_person("dance") == "dances"
    assert to_third_person("sleep") == "sleeps"
    assert to_third_person("run") == "runs"
    assert to_third_person("walk") == "walks"


def test_verb_inflection_base_form():
    # Irregulars
    assert to_base_form("goes") == "go"
    assert to_base_form("has") == "have"
    assert to_base_form("does") == "do"
    assert to_base_form("is") == "are"

    # -ies -> y
    assert to_base_form("studies") == "study"
    assert to_base_form("carries") == "carry"
    assert to_base_form("flies") == "fly"
    assert to_base_form("tries") == "try"
    assert to_base_form("cries") == "cry"

    # -es after s, x, z, ch, sh
    assert to_base_form("watches") == "watch"
    assert to_base_form("passes") == "pass"
    assert to_base_form("buzzes") == "buzz"
    assert to_base_form("fixes") == "fix"
    assert to_base_form("washes") == "wash"
    assert to_base_form("teaches") == "teach"
    assert to_base_form("catches") == "catch"

    # Standard -s
    assert to_base_form("eats") == "eat"
    assert to_base_form("plays") == "play"
    assert to_base_form("works") == "work"
    assert to_base_form("writes") == "write"
    assert to_base_form("dances") == "dance"
    assert to_base_form("runs") == "run"
    assert to_base_form("walks") == "walk"


# ---------------------------------------------------------------------------
# 3. Phonetic Sound Detection for Articles
# ---------------------------------------------------------------------------
def test_phonetic_sound_detection():
    # Silent H (starts with vowel sound)
    assert starts_with_vowel_sound("hour") is True
    assert starts_with_vowel_sound("honest") is True
    assert starts_with_vowel_sound("honor") is True
    assert starts_with_vowel_sound("heir") is True

    # Voiced H (starts with consonant sound)
    assert starts_with_vowel_sound("house") is False
    assert starts_with_vowel_sound("horse") is False
    assert starts_with_vowel_sound("hotel") is False
    assert starts_with_vowel_sound("history") is False

    # 'u' sounding like /juː/ (consonant sound)
    assert starts_with_vowel_sound("university") is False
    assert starts_with_vowel_sound("uniform") is False
    assert starts_with_vowel_sound("unique") is False
    assert starts_with_vowel_sound("user") is False
    assert starts_with_vowel_sound("useful") is False

    # 'u' sounding like short vowel
    assert starts_with_vowel_sound("umbrella") is True
    assert starts_with_vowel_sound("uncle") is True
    assert starts_with_vowel_sound("ugly") is True

    # 'eu' sounding like /juː/ (consonant sound)
    assert starts_with_vowel_sound("European") is False
    assert starts_with_vowel_sound("euro") is False

    # 'one' sounding like /w/ (consonant sound)
    assert starts_with_vowel_sound("one") is False
    assert starts_with_vowel_sound("one-time") is False

    # Standard vowels
    assert starts_with_vowel_sound("apple") is True
    assert starts_with_vowel_sound("engineer") is True
    assert starts_with_vowel_sound("elephant") is True
    assert starts_with_vowel_sound("orange") is True
    assert starts_with_vowel_sound("idea") is True

    # Standard consonants
    assert starts_with_vowel_sound("dog") is False
    assert starts_with_vowel_sound("car") is False
    assert starts_with_vowel_sound("book") is False
    assert starts_with_vowel_sound("friend") is False

    # Acronyms pronounced letter-by-letter
    assert starts_with_vowel_sound("MBA") is True
    assert starts_with_vowel_sound("FBI") is True
    assert starts_with_vowel_sound("HTML") is True
    assert starts_with_vowel_sound("PhD") is False
    assert starts_with_vowel_sound("CEO") is False


# ---------------------------------------------------------------------------
# 4. Hand-Checked Grammar Correction Cases (50+ cases)
# ---------------------------------------------------------------------------
ADDITIONAL_CORRECTION_CASES = [
    # he/she/it + base verb -> third-person form
    ("He play football every Sunday.", "He plays football every Sunday.", "subject_verb_agreement"),
    ("She study hard for exams.", "She studies hard for exams.", "subject_verb_agreement"),
    ("It work well.", "It works well.", "subject_verb_agreement"),
    ("He watch movies on weekends.", "He watches movies on weekends.", "subject_verb_agreement"),
    ("She carry the groceries.", "She carries the groceries.", "subject_verb_agreement"),
    ("He fix the car.", "He fixes the car.", "subject_verb_agreement"),
    ("It buzz loudly.", "It buzzes loudly.", "subject_verb_agreement"),
    ("She pass the exam.", "She passes the exam.", "subject_verb_agreement"),
    ("He dance beautifully.", "He dances beautifully.", "subject_verb_agreement"),
    ("She write poems.", "She writes poems.", "subject_verb_agreement"),

    # I/you/we/they + third-person -> base form
    ("We carries the luggage.", "We carry the luggage.", "subject_verb_agreement"),
    ("They plays outside.", "They play outside.", "subject_verb_agreement"),
    ("You watches television.", "You watch television.", "subject_verb_agreement"),
    ("I reads books.", "I read books.", "subject_verb_agreement"),
    ("We eats dinner together.", "We eat dinner together.", "subject_verb_agreement"),
    ("They runs fast.", "They run fast.", "subject_verb_agreement"),
    ("You walks to the store.", "You walk to the store.", "subject_verb_agreement"),

    # is / are mismatches
    ("He are very happy.", "He is very happy.", "is_are"),
    ("It are a sunny day.", "It is a sunny day.", "is_are"),
    ("We is going to the market.", "We are going to the market.", "is_are"),
    ("You is invited to the party.", "You are invited to the party.", "is_are"),
    ("The student are diligent.", "The student is diligent.", "is_are"),
    ("The cars is moving fast.", "The cars are moving fast.", "is_are"),

    # has / have mismatches
    ("She have two sisters.", "She has two sisters.", "has_have"),
    ("It have great potential.", "It has great potential.", "has_have"),
    ("They has completed their work.", "They have completed their work.", "has_have"),
    ("We has an announcement.", "We have an announcement.", "has_have"),
    ("You has a nice smile.", "You have a nice smile.", "has_have"),

    # does / do mismatches
    ("He do not understand.", "He does not understand.", "does_do"),
    ("She do her best.", "She does her best.", "does_do"),
    ("It do not make sense.", "It does not make sense.", "does_do"),
    ("They does not like spicy food.", "They do not like spicy food.", "does_do"),
    ("We does not want to be late.", "We do not want to be late.", "does_do"),
    ("You does not need to worry.", "You do not need to worry.", "does_do"),

    # a / an phonetic corrections
    ("Give me a apple.", "Give me an apple.", "a_an"),
    ("He waited for a hour.", "He waited for an hour.", "a_an"),
    ("She is a honest student.", "She is an honest student.", "a_an"),
    ("He saw a elephant at the zoo.", "He saw an elephant at the zoo.", "a_an"),
    ("He is a MBA student.", "He is an MBA student.", "a_an"),
    ("He saw an dog.", "He saw a dog.", "a_an"),
    ("She is an university professor.", "She is a university professor.", "a_an"),
    ("He wears an uniform.", "He wears a uniform.", "a_an"),
    ("It was an European tour.", "It was a European tour.", "a_an"),
    ("This is an one-time opportunity.", "This is a one-time opportunity.", "a_an"),
    ("She found an useful tool.", "She found a useful tool.", "a_an"),

    # Adverbs between subject and verb
    ("She always go to college.", "She always goes to college.", "subject_verb_agreement"),
    ("They usually goes to the library.", "They usually go to the library.", "subject_verb_agreement"),
    ("He never eat breakfast.", "He never eats breakfast.", "subject_verb_agreement"),
    ("She often read novel books.", "She often reads novel books.", "subject_verb_agreement"),
]


@pytest.mark.parametrize("input_text, expected_output, expected_rule", ADDITIONAL_CORRECTION_CASES)
def test_additional_corrections(input_text, expected_output, expected_rule):
    corrected, edits = apply_rules(input_text)
    assert corrected == expected_output
    assert any(e.rule == expected_rule for e in edits)


# ---------------------------------------------------------------------------
# 5. Tricky Cases That Must NOT Change (No False Positives)
# ---------------------------------------------------------------------------
TRICKY_NO_CHANGE_CASES = [
    # Already grammatically correct present tense
    "She goes home.",
    "I read books.",
    "They play football.",
    "He has a car.",
    "She is my friend.",
    "He eats an apple.",
    "They go to school.",
    "She speaks fluent English.",
    "We live in Bengaluru.",

    # Articles that are already correct phonetically
    "It takes an hour.",
    "She goes to a university.",
    "He is an honest man.",
    "It is a European city.",
    "She bought a uniform.",
    "He has an umbrella.",
    "This is a one-way street.",
    "He earned an MBA degree.",

    # Modal verbs (verb must remain base form)
    "He can go home.",
    "She will eat lunch.",
    "They must study for the test.",
    "We should help them.",
    "He might come tomorrow.",

    # Infinitives with 'to' (verb must remain base form)
    "He wants to go.",
    "They decided to eat.",
    "She hopes to win the prize.",
    "We need to leave now.",

    # Auxiliary negatives (verb after does not / did not must remain base form)
    "She does not eat meat.",
    "They did not go to the market.",
    "He does not like cold weather.",

    # Genuine past tense verbs (must never be forced into present third person)
    "She went home yesterday.",
    "He ate an apple for lunch.",
    "They played tennis last evening.",
    "She loved him deeply.",
    "He saw a movie yesterday.",
    "They built a large house.",
    "I bought a new laptop.",
    "She wrote a long essay.",
    "He took the morning train.",
    "They ran five kilometers.",
]


@pytest.mark.parametrize("sentence", TRICKY_NO_CHANGE_CASES)
def test_tricky_cases_do_not_change(sentence):
    corrected, edits = apply_rules(sentence)
    assert corrected == sentence
    assert len(edits) == 0


# ---------------------------------------------------------------------------
# 6. Quoted Text and Proper Nouns Protection
# ---------------------------------------------------------------------------
PROTECTED_TEXT_CASES = [
    ('She said, "they goes to school."', 'She said, "they goes to school."'),
    ('He shouted, "she are my friend!"', 'He shouted, "she are my friend!"'),
    ('The sign said "a elephant" on the wall.', 'The sign said "a elephant" on the wall.'),
    ("Microsoft is a large company.", "Microsoft is a large company."),
    ("John goes to school.", "John goes to school."),
    ("London is the capital of England.", "London is the capital of England."),
]


@pytest.mark.parametrize("sentence, expected", PROTECTED_TEXT_CASES)
def test_quoted_text_and_proper_nouns_protected(sentence, expected):
    corrected, edits = apply_rules(sentence)
    assert corrected == expected
    assert len(edits) == 0


# ---------------------------------------------------------------------------
# 7. Exact Edit Details Verification
# ---------------------------------------------------------------------------
def test_exact_edit_metadata():
    text = "She go to college every day."
    corrected, edits = apply_rules(text)
    assert len(edits) == 1
    edit = edits[0]
    assert edit.position == 1
    assert edit.old == "go"
    assert edit.new == "goes"
    assert edit.rule == "subject_verb_agreement"

    text2 = "She is a engineer."
    corrected2, edits2 = apply_rules(text2)
    assert len(edits2) == 1
    edit2 = edits2[0]
    assert edit2.position == 2
    assert edit2.old == "a"
    assert edit2.new == "an"
    assert edit2.rule == "a_an"
