"""
Scoring formulas and gender-neutral term lookup.

Gender-neutral replacements are drawn from:
  - American Psychological Association (APA) Publication Manual,
    7th ed., "Bias-Free Language" section.
  - Linguistic Society of America, "Guidelines for Nonsexist Usage."
  - Conscious Style Guide, gender-inclusive language entries.

Cite these sources in your thesis methodology section.
"""
import re


# ── Gender-neutral replacements ───────────────────────────────────────────────
# Keys are lowercase. Lookup is case-insensitive; matching preserves the
# original capitalization pattern on output.
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
    """Return a gender-neutral alternative if known, else None."""
    if not phrase:
        return None
    key = phrase.strip().lower()
    if key in GENDER_NEUTRAL_MAP:
        return GENDER_NEUTRAL_MAP[key]
    # Try single-word match within a longer phrase
    for word in re.findall(r"[A-Za-z][A-Za-z\-]+", key):
        if word in GENDER_NEUTRAL_MAP:
            return GENDER_NEUTRAL_MAP[word]
    return None


def find_gendered_word(sentence: str) -> str | None:
    """
    Find the first gendered word in the sentence that has a known
    neutral replacement. Used for the 'phrase' field on gender_sensitive rows.
    """
    if not sentence:
        return None
    lower = sentence.lower()
    # Longest keys first so "master bedroom" beats "master"
    for key in sorted(GENDER_NEUTRAL_MAP.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        match = re.search(pattern, lower)
        if match:
            # Return the original-case substring
            start, end = match.span()
            return sentence[start:end]
    return None


# ── Overall score formula ─────────────────────────────────────────────────────

SEMANTIC_WEIGHT   = 0.7
AUTHORSHIP_WEIGHT = 0.3

# Severity-weighted floor penalty per flagged row (label × sentence pair).
# Used when the ratio-based penalty is too small to reflect the real issue
# (e.g. 2 stereotyping sentences in a 500-sentence paper should not score 99%).
# Values are tunable design choices — document them in the thesis methodology.
SEVERITY_WEIGHTS = {
    "gender_sensitive": 1.0,
    "stereotyping":     2.5,
    "representation":   2.5,
}


def compute_overall_score(
    flagged_count: int,
    total_sentences: int,
    male_count: int,
    female_count: int,
    label_flag_counts: dict | None = None,
) -> dict:
    """
    Compute the overall responsiveness score.

    Formula:
      ratio_penalty       = (flagged / total) × 100 × 0.7
      floor_penalty       = Σ (count_per_label × severity_weight)
      semantic_penalty    = max(ratio_penalty, floor_penalty)
      authorship_penalty  = |0.5 − female_ratio| × 2 × 100 × 0.3
      score               = 100 − semantic_penalty − authorship_penalty
    """
    # Semantic component
    total = max(total_sentences, 1)
    flagged_ratio = flagged_count / total
    ratio_penalty = flagged_ratio * 100 * SEMANTIC_WEIGHT

    # Severity-weighted floor: ensures a few clear violations aren't drowned
    # out by a large document's overall sentence count.
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

    score = 100 - semantic_penalty - authorship_penalty
    score = max(0, min(100, round(score)))

    return {
        "score":              score,
        "semantic_penalty":   round(semantic_penalty, 2),
        "ratio_penalty":      round(ratio_penalty, 2),
        "floor_penalty":      round(floor_penalty, 2),
        "authorship_penalty": round(authorship_penalty, 2),
        "female_ratio":       round(female_ratio, 3) if female_ratio is not None else None,
        "authorship_flagged": authorship_flagged,
    }