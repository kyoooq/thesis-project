import re


GENDER_NEUTRAL_MAP = {
    # -man / -men professions
    "fireman":        "firefighter",
    "firemen":        "firefighters",
    "policeman":      "police officer",
    "policemen":      "police officers",
    "policewoman":    "police officer",
    "policewomen":    "police officers",
    "mailman":        "mail carrier",
    "mailmen":        "mail carriers",
    "postman":        "mail carrier",
    "postmen":        "mail carriers",
    "salesman":       "salesperson",
    "salesmen":       "salespeople",
    "saleswoman":     "salesperson",
    "saleswomen":     "salespeople",
    "businessman":    "businessperson",
    "businessmen":    "businesspeople",
    "businesswoman":  "businessperson",
    "businesswomen":  "businesspeople",
    "chairman":       "chairperson",
    "chairmen":       "chairpersons",
    "chairwoman":     "chairperson",
    "chairwomen":     "chairpersons",
    "spokesman":      "spokesperson",
    "spokesmen":      "spokespeople",
    "spokeswoman":    "spokesperson",
    "spokeswomen":    "spokespeople",
    "congressman":    "member of Congress",
    "congressmen":    "members of Congress",
    "congresswoman":  "member of Congress",
    "congresswomen":  "members of Congress",
    "weatherman":     "meteorologist",
    "weathermen":     "meteorologists",
    "cameraman":      "camera operator",
    "cameramen":      "camera operators",
    "foreman":        "supervisor",
    "foremen":        "supervisors",
    "forewoman":      "supervisor",
    "craftsman":      "artisan",
    "craftsmen":      "artisans",
    "craftswoman":    "artisan",
    "draftsman":      "drafter",
    "draftsmen":      "drafters",
    "repairman":      "technician",
    "repairmen":      "technicians",
    "handyman":       "handyperson",
    "fisherman":      "fisher",
    "fishermen":      "fishers",
    "newsman":        "reporter",
    "newsmen":        "reporters",
    "anchorman":      "news anchor",
    "anchormen":      "news anchors",
    "anchorwoman":    "news anchor",
    "clergyman":      "clergy member",
    "clergymen":      "clergy members",
    "clergywoman":    "clergy member",
    "alderman":       "council member",
    "aldermen":       "council members",
    "ombudsman":      "ombudsperson",
    "freshman":       "first-year student",
    "freshmen":       "first-year students",
    "best man":       "honor attendant",
    "headmaster":     "principal",
    "headmistress":   "principal",
    "milkman":        "milk deliverer",
    "garbageman":     "sanitation worker",
    "delivery boy":   "delivery driver",
    "paperboy":       "newspaper carrier",
    "stuntman":       "stunt performer",
    "gunman":         "shooter",
    "juryman":        "juror",
    "jurymen":        "jurors",
    "swordsman":      "fencer",
    "yachtsman":      "sailor",

    # -ess feminine suffixes
    "stewardess":     "flight attendant",
    "stewardesses":   "flight attendants",
    "waitress":       "server",
    "waitresses":     "servers",
    "actress":        "actor",
    "actresses":      "actors",
    "authoress":      "author",
    "poetess":        "poet",
    "usherette":      "usher",
    "seamstress":     "tailor",
    "editress":       "editor",
    "governess":      "tutor",
    "prophetess":     "prophet",
    "hostess":        "host",
    "heiress":        "heir",
    "mistress":       "head",
    "barmaid":        "bartender",
    "landlady":       "landlord",
    "landlord":       "property owner",

    # Generic "man" as humanity
    "mankind":        "humanity",
    "man-made":       "artificial",
    "manmade":        "artificial",
    "manpower":       "workforce",
    "manhunt":        "search",
    "manhole":        "utility hole",
    "man-hours":      "work hours",
    "manned":         "crewed",
    "unmanned":       "uncrewed",
    "cavemen":        "cave dwellers",

    # Other gendered terms
    "housewife":      "homemaker",
    "househusband":   "homemaker",
    "housewives":     "homemakers",
    "gentleman":      "person",
    "gentlemen":      "people",
    "lady":           "person",
    "ladies":         "people",
    "girl friday":    "assistant",
    "gal friday":     "assistant",
    "old boys":       "exclusive network",
    "bachelorette":   "unmarried person",
    "master bedroom": "primary bedroom",
    "maiden name":    "birth name",
    "maiden voyage":  "inaugural voyage",
}


def lookup_neutral(phrase: str) -> str | None:
    if not phrase:
        return None
    key = phrase.strip().lower()
    if key in GENDER_NEUTRAL_MAP:
        return GENDER_NEUTRAL_MAP[key]
    for word in re.findall(r"[A-Za-z][A-Za-z\-]+", key):
        if word in GENDER_NEUTRAL_MAP:
            return GENDER_NEUTRAL_MAP[word]
    return None


def find_gendered_word(sentence: str) -> str | None:
    if not sentence:
        return None
    lower = sentence.lower()
    for key in sorted(GENDER_NEUTRAL_MAP.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        match = re.search(pattern, lower)
        if match:
            start, end = match.span()
            return sentence[start:end]
    return None


# Overall score formula 

SEMANTIC_WEIGHT        = 0.6 
AUTHORSHIP_WEIGHT      = 0.2 
REPRESENTATION_WEIGHT  = 0.2 

SEVERITY_WEIGHTS = {
    "gender_sensitive": 1.0,
    "stereotyping":     2.5,
}

DISAGGREGATION_KEYWORDS = [
    r"\brespondents?\b",
    r"\bparticipants?\b",
    r"\bbeneficiar(?:y|ies)\b",
    r"\bsample size\b",
    r"\btarget population\b",
    r"\bstudy population\b",
    r"\bsurveyed\b",
    r"\binterviewed\b",
    r"\benrolled\b",
    r"\bn\s*=\s*\d+",
    r"\bN\s*=\s*\d+",
    r"\bsample of\b",
    r"\bpopulation of\b",
]

_NUM = r"(\d+)"
_MALE_WORDS   = r"(?:male|males|men|man)"
_FEMALE_WORDS = r"(?:female|females|women|woman)"

DISAGGREGATION_PATTERNS = [
    rf"{_NUM}\s*{_MALE_WORDS}[^.]*?{_NUM}\s*{_FEMALE_WORDS}",
    rf"{_NUM}\s*{_FEMALE_WORDS}[^.]*?{_NUM}\s*{_MALE_WORDS}",
    rf"(?:n\s*=\s*)?{_NUM}\s*\)\s*{_MALE_WORDS}",
    rf"{_MALE_WORDS}\s*\(\s*n\s*=\s*{_NUM}\s*\)",
    rf"(?:n\s*=\s*)?{_NUM}\s*\)\s*{_FEMALE_WORDS}",
    rf"{_FEMALE_WORDS}\s*\(\s*n\s*=\s*{_NUM}\s*\)",
]


def check_disaggregation_needed(full_text: str) -> bool:
    text_lower = full_text.lower()
    for pattern in DISAGGREGATION_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def extract_disaggregated_counts(full_text: str) -> tuple[int, int] | tuple[None, None]:
    WINDOW = 400

    keyword_positions = []
    for kw in DISAGGREGATION_KEYWORDS:
        for m in re.finditer(kw, full_text, re.IGNORECASE):
            keyword_positions.append(m.start())

    if not keyword_positions:
        return None, None

    def near_keyword(pos):
        return any(abs(pos - kpos) <= WINDOW for kpos in keyword_positions)

    def get_num(m):
        return int(next(x for x in m.groups() if x is not None))

    lv_male   = re.compile(r'(?i)\b(?:male|men|man)\b(?!\w)[ \t:]+(\d+)')
    lv_female = re.compile(r'(?i)\b(?:female|women|woman)\b(?!\w)[ \t:]+(\d+)')

    male_val = female_val = None
    for m in lv_male.finditer(full_text):
        if near_keyword(m.start()):
            male_val = get_num(m); break
    for m in lv_female.finditer(full_text):
        if near_keyword(m.start()):
            female_val = get_num(m); break
    if male_val is not None and female_val is not None:
        return male_val, female_val

    # Strategy 2: value-label  N male  (same line only — no newline)
    vl_male   = re.compile(r'(?i)(\d+)[ \t]+(?:male|men|man)\b')
    vl_female = re.compile(r'(?i)(\d+)[ \t]+(?:female|women|woman)\b')

    male_val = female_val = None
    for m in vl_male.finditer(full_text):
        if near_keyword(m.start()):
            male_val = get_num(m); break
    for m in vl_female.finditer(full_text):
        if near_keyword(m.start()):
            female_val = get_num(m); break
    if male_val is not None and female_val is not None:
        return male_val, female_val

    return None, None

def compute_representation_penalty(
    needs_disaggregation: bool,
    male_count: int | None,
    female_count: int | None,
) -> dict:

    if not needs_disaggregation:
        return {
            "representation_penalty": 0.0,
            "representation_status":  "not_applicable",
            "representation_flagged": False,
            "rep_male_count":         None,
            "rep_female_count":       None,
            "rep_female_ratio":       None,
        }

    if male_count is None or female_count is None:
        return {
            "representation_penalty": 20.0,
            "representation_status":  "missing",
            "representation_flagged": True,
            "rep_male_count":         None,
            "rep_female_count":       None,
            "rep_female_ratio":       None,
        }

    known = male_count + female_count
    if known == 0:
        return {
            "representation_penalty": 20.0,
            "representation_status":  "missing",
            "representation_flagged": True,
            "rep_male_count":         0,
            "rep_female_count":       0,
            "rep_female_ratio":       None,
        }

    female_ratio = female_count / known
    imbalance    = abs(0.5 - female_ratio) * 2 
    penalty      = imbalance * 100 * REPRESENTATION_WEIGHT 

    flagged = abs(0.5 - female_ratio) > 0.10 

    return {
        "representation_penalty": round(penalty, 2),
        "representation_status":  "balanced" if not flagged else "skewed",
        "representation_flagged": flagged,
        "rep_male_count":         male_count,
        "rep_female_count":       female_count,
        "rep_female_ratio":       round(female_ratio, 3),
    }


def compute_overall_score(
    flagged_count: int,
    total_sentences: int,
    male_count: int,
    female_count: int,
    label_flag_counts: dict | None = None,
    representation_info: dict | None = None,
) -> dict:
    
    # Semantic component 
    total         = max(total_sentences, 1)
    flagged_ratio = flagged_count / total
    ratio_penalty = flagged_ratio * 100 * SEMANTIC_WEIGHT

    floor_penalty = 0.0
    if label_flag_counts:
        for label, count in label_flag_counts.items():
            weight = SEVERITY_WEIGHTS.get(label, 1.0)
            floor_penalty += count * weight

    semantic_penalty = max(ratio_penalty, floor_penalty)

    # Authorship component
    known = male_count + female_count
    if known == 0:
        female_ratio       = None
        authorship_penalty = 0.0
        authorship_flagged = False
    else:
        female_ratio       = female_count / known
        imbalance          = abs(0.5 - female_ratio) * 2
        authorship_penalty = imbalance * 100 * AUTHORSHIP_WEIGHT
        authorship_flagged = abs(0.5 - female_ratio) > 0.10

    # Representation component 
    rep = representation_info or {}
    representation_penalty = rep.get("representation_penalty", 0.0)

    # Final score 
    score = 100 - semantic_penalty - authorship_penalty - representation_penalty
    score = max(0, min(100, round(score)))

    return {
        "score":                 score,
        "semantic_penalty":      round(semantic_penalty, 2),
        "ratio_penalty":         round(ratio_penalty, 2),
        "floor_penalty":         round(floor_penalty, 2),
        "authorship_penalty":    round(authorship_penalty, 2),
        "female_ratio":          round(female_ratio, 3) if female_ratio is not None else None,
        "authorship_flagged":    authorship_flagged,
        "representation_penalty": representation_penalty,
        "representation_status":  rep.get("representation_status", "not_applicable"),
        "representation_flagged": rep.get("representation_flagged", False),
        "rep_male_count":         rep.get("rep_male_count"),
        "rep_female_count":       rep.get("rep_female_count"),
        "rep_female_ratio":       rep.get("rep_female_ratio"),
    }