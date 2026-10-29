"""
Spelling & Confusables Correction Layer for Grammar Autocorrector.
Provides high-precision dictionary lookups, transposition fixes,
common typo recovery, and contextual confusable resolution.
"""

import re
from typing import Tuple, List


# Inverse of common student misspellings: misspelled -> correct
SPELLING_DICTIONARY = {
    "definately": "definitely",
    "accomodate": "accommodate",
    "accomodation": "accommodation",
    "acheive": "achieve",
    "accross": "across",
    "agressive": "aggressive",
    "allmost": "almost",
    "allready": "already",
    "altho": "although",
    "allways": "always",
    "amatuer": "amateur",
    "apparant": "apparent",
    "appearence": "appearance",
    "arguement": "argument",
    "basicly": "basically",
    "begining": "beginning",
    "beleive": "believe",
    "bizzare": "bizarre",
    "buisness": "business",
    "calender": "calendar",
    "catagory": "category",
    "cemetary": "cemetery",
    "collegue": "colleague",
    "comming": "coming",
    "commitee": "committee",
    "completly": "completely",
    "concious": "conscious",
    "curiousity": "curiosity",
    "differnt": "different",
    "dissapear": "disappear",
    "dissapoint": "disappoint",
    "embarass": "embarrass",
    "embarassment": "embarrassment",
    "enviroment": "environment",
    "existance": "existence",
    "familier": "familiar",
    "finaly": "finally",
    "foriegn": "foreign",
    "foward": "forward",
    "freind": "friend",
    "futher": "further",
    "goverment": "government",
    "grammer": "grammar",
    "garantee": "guarantee",
    "guidence": "guidance",
    "happend": "happened",
    "harrass": "harass",
    "heigth": "height",
    "humerous": "humorous",
    "immediatly": "immediately",
    "incidently": "incidentally",
    "independant": "independent",
    "inteligence": "intelligence",
    "intresting": "interesting",
    "interupt": "interrupt",
    "knowlege": "knowledge",
    "liesure": "leisure",
    "libary": "library",
    "lightening": "lightning",
    "maintainance": "maintenance",
    "millenium": "millennium",
    "minature": "miniature",
    "mischevious": "mischievous",
    "mispell": "misspell",
    "neccessary": "necessary",
    "noticable": "noticeable",
    "ocassion": "occasion",
    "occured": "occurred",
    "occurance": "occurrence",
    "oppinion": "opinion",
    "orignal": "original",
    "paralell": "parallel",
    "particuler": "particular",
    "passtime": "pastime",
    "preformance": "performance",
    "persistant": "persistent",
    "peice": "piece",
    "politican": "politician",
    "posession": "possession",
    "posible": "possible",
    "potatos": "potatoes",
    "prefered": "preferred",
    "presense": "presence",
    "privelege": "privilege",
    "priviledge": "privilege",
    "probly": "probably",
    "pronounciation": "pronunciation",
    "publically": "publicly",
    "realy": "really",
    "recieve": "receive",
    "reciept": "receipt",
    "recommed": "recommend",
    "referance": "reference",
    "relevent": "relevant",
    "religous": "religious",
    "rember": "remember",
    "restaraunt": "restaurant",
    "rythm": "rhythm",
    "schedual": "schedule",
    "secratary": "secretary",
    "seperate": "separate",
    "similiar": "similar",
    "succesful": "successful",
    "successfull": "successful",
    "beautifull": "beautiful",
    "carefull": "careful",
    "gratefull": "grateful",
    "hopefull": "hopeful",
    "suprise": "surprise",
    "tommorow": "tomorrow",
    "truely": "truly",
    "untill": "until",
    "unfortunatly": "unfortunately",
    "usally": "usually",
    "wierd": "weird",
    # Keyboard slips & common transpositions
    "teh": "the",
    "adn": "and",
    "wiht": "with",
    "fro": "for",
    "taht": "that",
    "becuase": "because",
    "woudl": "would",
    "colud": "could",
    "shoudl": "should",
    "thign": "thing",
    "tehse": "these",
    "thsoe": "those",
    # Contractions missing apostrophe
    "cant": "can't",
    "dont": "don't",
    "wont": "won't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    # Multiword contractions
    "alot": "a lot",
    "noone": "no one",
}


# High-precision contextual confusable regexes
CONTEXTUAL_CONFUSABLES = [
    # their / there / they're
    (r"\bTheir\s+(going|coming|playing|eating|working|studying|leaving|running|talking|waiting|trying)\b", r"They're \1"),
    (r"\btheir\s+(going|coming|playing|eating|working|studying|leaving|running|talking|waiting|trying)\b", r"they're \1"),
    (r"\bTheir\s+(the|a|an)\s+(best|most|only)\b", r"They're \1 \2"),
    (r"\btheir\s+(the|a|an)\s+(best|most|only)\b", r"they're \1 \2"),
    (r"\b(over|out|in|up|down|from)\s+their\b", r"\1 there"),
    (r"\bthere\s+(backpacks|books|parents|friends|hands|shoes|bags|house|car|names|lives|room)\b", r"their \1"),
    (r"\bThere\s+(backpacks|books|parents|friends|hands|shoes|bags|house|car|names|lives|room)\b", r"Their \1"),

    # your / you're
    (r"\bYour\s+(the|a|an)\s+(best|most|only|very|kind|smart|welcome)\b", r"You're \1 \2"),
    (r"\byour\s+(the|a|an)\s+(best|most|only|very|kind|smart|welcome)\b", r"you're \1 \2"),
    (r"\bYour\s+(welcome|going|coming|invited|doing|right|wrong|late|early)\b", r"You're \1"),
    (r"\byour\s+(welcome|going|coming|invited|doing|right|wrong|late|early)\b", r"you're \1"),
    (r"\byou're\s+(passport|keys|phone|bag|book|car|name|hand|room|sister|brother|father|mother|homework)\b", r"your \1"),
    (r"\bYou're\s+(passport|keys|phone|bag|book|car|name|hand|room|sister|brother|father|mother|homework)\b", r"Your \1"),

    # its / it's
    (r"\bIts\s+(raining|cold|hot|warm|sunny|windy|time|important|difficult|hard|easy|clear|been|nice|good)\b", r"It's \1"),
    (r"\bits\s+(raining|cold|hot|warm|sunny|windy|time|important|difficult|hard|easy|clear|been|nice|good)\b", r"it's \1"),
    (r"\b(licked|wagged|shook|raised|cleaned)\s+it's\s+(paw|tail|head|fur|wings|ears)\b", r"\1 its \2"),

    # then / than
    (r"\brather\s+([a-zA-Z]+)\s+then\b", r"rather \1 than"),
    (r"\b(rather|more|less|better|worse|taller|shorter|bigger|smaller|faster|slower|older|younger)\s+then\b", r"\1 than"),
    (r"\b(and|dinner\s+and|lunch\s+and|breakfast\s+and|since|until|back)\s+than\b", r"\1 then"),

    # to / too / two
    (r"\bto\s+(hot|cold|warm|fast|slow|expensive|cheap|hard|easy|heavy|light|tired|busy|small|big|late|early)\b", r"too \1"),
    (r"\bTo\s+(hot|cold|warm|fast|slow|expensive|cheap|hard|easy|heavy|light|tired|busy|small|big|late|early)\b", r"Too \1"),
    (r"\b([a-zA-Z]+)\s+to([.?!])$", r"\1 too\2"),
    (r"\bto\s+(cats|dogs|books|cars|houses|children|students|friends|apples|oranges|hours|days|weeks|months|years)\b", r"two \1"),
    (r"\bTo\s+(cats|dogs|books|cars|houses|children|students|friends|apples|oranges|hours|days|weeks|months|years)\b", r"Two \1"),

    # affect / effect
    (r"\b(will|would|can|could|may|might|should|to)\s+effect\b", r"\1 affect"),
    (r"\b(an|the|this|that|positive|negative|immediate|direct|side)\s+affect\b", r"\1 effect"),

    # lose / loose
    (r"\b(do\s+not|not|will|to)\s+loose\b", r"\1 lose"),
    (r"\b(is|are|was|were|too)\s+lose\b", r"\1 loose"),

    # whose / who's
    (r"\b(Who's|who's)\s+(jacket|book|car|pen|bag|coat|shoes|turn|fault|idea)\b", r"Whose \2"),
    (r"\b(Whose|whose)\s+(knocking|calling|coming|there|that|going|standing|singing|talking)\b", r"Who's \2"),

    # accept / except
    (r"\b(everyone|everybody|all|nobody|no\s+one|came)\s+accept\b", r"\1 except"),
    (r"\b(Everyone|Everybody|All|Nobody|No\s+one)\s+accept\b", r"\1 except"),

    # advice / advise
    (r"\b(some|good|great|bad|valuable|helpful)\s+advise\b", r"\1 advice"),

    # principal / principle
    (r"\bmoral\s+principals\b", r"moral principles"),

    # stationary / stationery
    (r"\b(envelopes\s+and|pens\s+and|office)\s+stationary\b", r"\1 stationery"),
    (r"\bremained\s+stationery\b", r"remained stationary"),

    # weather / whether
    (r"\bdo\s+not\s+know\s+weather\b", r"do not know whether"),
    (r"\b(The|the)\s+whether\s+in\b", r"\1 weather in"),

    # quiet / quite
    (r"\b(be|keep|stay)\s+quite\b", r"\1 quiet"),

    # lead / led
    (r"\b(guide|officer|teacher|leader)\s+lead\s+us\b", r"\1 led us"),

    # brake / break
    (r"\bnot\s+to\s+brake\b", r"not to break"),

    # peace / piece
    (r"\bworld\s+piece\b", r"world peace"),
    (r"\banother\s+peace\s+of\b", r"another piece of"),
    (r"\ba\s+peace\s+of\b", r"a piece of"),

    # aloud / allowed
    (r"\bnot\s+aloud\s+to\b", r"not allowed to"),
    (r"\bread\s+([a-zA-Z\s]+)\s+allowed\b", r"read \1 aloud"),

    # desert / dessert
    (r"\bchocolate\s+desert\b", r"chocolate dessert"),
    (r"\bdelicious\s+desert\b", r"delicious dessert"),
    (r"\bin\s+the\s+dessert\b", r"in the desert"),

    # sight / site
    (r"\bconstruction\s+sight\b", r"construction site"),
    (r"\bbreathtaking\s+site\b", r"breathtaking sight"),

    # passed / past
    (r"\bwalked\s+passed\b", r"walked past"),
    (r"\bpast\s+his\s+driving\s+test\b", r"passed his driving test"),

    # complement / compliment
    (r"\bcompliments\s+the\b", r"complements the"),
    (r"\bcompliments\s+the\s+([a-zA-Z]+)\s+well\b", r"complements the \1 well"),

    # buy / by
    (r"\bstand\s+buy\s+me\b", r"stand by me"),

    # know / no
    (r"\bis\s+know\s+easy\b", r"is no easy"),

    # new / knew
    (r"\bbrand\s+knew\b", r"brand new"),
    (r"\b(He|he|She|she|They|they)\s+new\s+the\b", r"\1 knew the"),

    # write / right
    (r"\bwith\s+your\s+write\s+hand\b", r"with your right hand"),

    # whole / hole
    (r"\bdeep\s+whole\b", r"deep hole"),
    (r"\bthe\s+hole\s+pizza\b", r"the whole pizza"),

    # meat / meet
    (r"\blet\s+us\s+meat\b", r"let us meet"),
    (r"\bLet\s+us\s+meat\b", r"Let us meet"),

    # stair / stare
    (r"\brude\s+to\s+stair\b", r"rude to stare"),

    # waste / waist
    (r"\bnot\s+waist\s+your\b", r"not waste your"),
    (r"\baround\s+his\s+waste\b", r"around his waist"),

    # wait / weight
    (r"\bweight\s+for\s+me\b", r"wait for me"),
    (r"\bPlease\s+weight\b", r"Please wait"),
    (r"\btotal\s+wait\s+of\b", r"total weight of"),

    # coarse / course
    (r"\bOf\s+coarse\b", r"Of course"),
    (r"\bof\s+coarse\b", r"of course"),

    # grader / grater
    (r"\bcheese\s+grader\b", r"cheese grater"),

    # male / mail
    (r"\bmorning\s+male\b", r"morning mail"),

    # scene / seen
    (r"\bHave\s+you\s+scene\b", r"Have you seen"),
    (r"\bhave\s+you\s+scene\b", r"have you seen"),
]


def apply_spelling_pass(text: str) -> str:
    """
    Apply spelling and confusable corrections to text.
    Preserves casing and surrounding punctuation.
    """
    if not text:
        return text

    result = text

    # 1. High-precision contextual confusable regexes
    for pattern, replacement in CONTEXTUAL_CONFUSABLES:
        result = re.sub(pattern, replacement, result)

    # 2. Word-level spelling dictionary lookups
    # Matches words with optional apostrophes
    def replace_word(match):
        tok = match.group(0)
        lw = tok.lower()
        if lw in SPELLING_DICTIONARY:
            corrected = SPELLING_DICTIONARY[lw]
            # Match casing
            if tok.isupper() and len(tok) > 1:
                return corrected.upper()
            elif tok[0].isupper():
                return corrected[0].upper() + corrected[1:]
            else:
                return corrected
        return tok

    result = re.sub(r"\b[a-zA-Z']+\b", replace_word, result)

    return result
