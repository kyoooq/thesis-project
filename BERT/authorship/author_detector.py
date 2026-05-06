"""
authorship/author_detector.py

Strict authorship detection with references-as-ground-truth.

Pipeline:
    1. Split full text into (body, references) using heading heuristics.
    2. Parse the references section into canonical Reference objects
       (surname, initials, year, raw line).
    3. Scan the body for in-text citations (APA / MLA style).
    4. Cross-check: only count references that appear in both sections.
    5. Resolve each matched reference to a gender using names-dataset,
       with gender_guesser as fallback.
    6. Deduplicate on (surname_lower, year) so "Palma (2004)" and
       "Palma, S. (2004)" collapse to one author.

Public API (unchanged shape where possible):
    analyze_authorship(full_text) -> {
        "authors": [
            {"surname": "Palma", "initials": "S.",
             "year": "2004", "gender": "male",
             "confidence": 0.92, "source": "names-dataset",
             "cited_in_body": True},
            ...
        ],
        "male_count":    int,
        "female_count":  int,
        "unknown_count": int,
        "diagnostics":   {...}   # useful for debugging / thesis writeup
    }

Legacy per-sentence API kept as `detect_authorship(sentence)` so existing
callers don't break, but it's now strongly discouraged — it can't do
cross-checking by design.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from names_dataset import NameDataset
import gender_guesser.detector as gender_guesser


# ══════════════════════════════════════════════════════════════════════════════
# Lazy singletons — avoid loading the ~500MB names-dataset on import
# ══════════════════════════════════════════════════════════════════════════════
_name_ds: Optional[NameDataset] = None
_gg: Optional[gender_guesser.Detector] = None


def _get_name_ds() -> NameDataset:
    global _name_ds
    if _name_ds is None:
        _name_ds = NameDataset()
    return _name_ds


def _get_gg() -> gender_guesser.Detector:
    global _gg
    if _gg is None:
        _gg = gender_guesser.Detector(case_sensitive=False)
    return _gg


# ══════════════════════════════════════════════════════════════════════════════
# Section splitting
# ══════════════════════════════════════════════════════════════════════════════

# Headings that mark the start of the references section.
# Ordered most-specific to least so we match the strongest signal.
REFERENCES_HEADINGS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works?\s+cited\s*$",
    r"^\s*literature\s+cited\s*$",
    r"^\s*reference\s+list\s*$",
]

# Headings that mark the END of the references section (appendices come after)
POST_REFERENCES_HEADINGS = [
    r"^\s*appendix(?:\s+[A-Z0-9])?\s*$",
    r"^\s*appendices\s*$",
    r"^\s*acknowledgements?\s*$",
    r"^\s*author\s+biographies?\s*$",
    r"^\s*about\s+the\s+authors?\s*$",
]


def split_body_and_references(text: str) -> tuple[str, str]:
    """
    Return (body_text, references_text).
    If no references heading is found, references_text is ''.
    """
    lines = text.split("\n")

    # Find the references heading — scan from the back since it's near the end
    ref_start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line or len(line) > 80:
            # headings are short; skip paragraphs
            continue
        for pat in REFERENCES_HEADINGS:
            if re.match(pat, line, re.IGNORECASE):
                ref_start_idx = i
                break
        if ref_start_idx is not None:
            break

    if ref_start_idx is None:
        return text, ""

    # Find where references end (appendix, acknowledgements, etc.)
    ref_end_idx = len(lines)
    for i in range(ref_start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or len(line) > 80:
            continue
        for pat in POST_REFERENCES_HEADINGS:
            if re.match(pat, line, re.IGNORECASE):
                ref_end_idx = i
                break
        if ref_end_idx != len(lines):
            break

    body       = "\n".join(lines[:ref_start_idx])
    references = "\n".join(lines[ref_start_idx + 1:ref_end_idx])
    return body, references


# ══════════════════════════════════════════════════════════════════════════════
# Reference parsing (APA / MLA)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Reference:
    surname: str          # "Palma"
    initials: str         # "S." or "" if MLA without initials
    year: Optional[str]   # "2004" or None (MLA often omits year in entry start)
    first_name: str       # "" if only initial available; "Silvio" if full
    raw: str              # the full reference line for audit
    entry_id: int = 0     # co-authors from the same entry share an entry_id

    @property
    def canonical_key(self) -> str:
        """Key used for deduplication — per author, not per entry."""
        return f"{self.surname.lower()}|{self.year or ''}"


# APA reference entry:
#   Palma, S. (2004). Title of work. Journal.
#   Palma, S., & Cruz, J. M. (2004). Title.
#   Palma, S., Cruz, J. M., & Reyes, A. (2004). Title.
#
# MLA works-cited entry:
#   Palma, Silvio. "Title." Journal, vol. 5, 2004.
#   Palma, Silvio, and John Cruz. Title. Publisher, 2004.
APA_AUTHOR_BLOCK = re.compile(
    r"""
    ^                                                # start of the reference line
    (?P<authors>
        [A-Z][a-zA-ZÀ-ÿ'\-]+                        # first surname
        ,\s*
        (?:[A-Z]\.(?:\s*[A-Z]\.)*|[A-Z][a-zA-ZÀ-ÿ'\-]+)   # APA initials OR MLA full first name
        (?:                                          # optional: additional authors
            (?:\s*,\s*(?:&|and)\s*|\s*,\s+)
            [A-Z][a-zA-ZÀ-ÿ'\-]+
            ,\s*
            (?:[A-Z]\.(?:\s*[A-Z]\.)*|[A-Z][a-zA-ZÀ-ÿ'\-]+)
        )*
    )
    \s*
    (?:\(|\.\s*)                                     # then "(year)" or ". Year"
    """,
    re.VERBOSE,
)

YEAR_IN_PAREN = re.compile(r"\((\d{4})[a-z]?\)")
YEAR_BARE     = re.compile(r"\b(19|20)\d{2}\b")

# One author inside an author block:
#   "Palma, S."  "Palma, S. M."  "Palma, Silvio"
AUTHOR_PIECE = re.compile(
    r"""
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    \s*,\s*
    (?P<given>
        (?:[A-Z]\.(?:\s*[A-Z]\.)*)          # initials like "S." or "S. M."
        |
        (?:[A-Z][a-zA-ZÀ-ÿ'\-]+(?:\s+[A-Z][a-zA-ZÀ-ÿ'\-]+)?)   # full name "Silvio" or "Silvio M"
    )
    """,
    re.VERBOSE,
)

# MLA secondary authors appear uninverted: "and Tom Cole" (not "Cole, Tom").
# Used as a supplementary pass after AUTHOR_PIECE.
MLA_COAUTHOR = re.compile(
    r"""
    (?:,\s*)?\band\s+
    (?P<first>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    \s+
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    (?=\s*[.,]|\s+and\s|\s*$)
    """,
    re.VERBOSE,
)


def _normalize_ref_lines(references_text: str) -> list[str]:
    """
    Reference entries often wrap across multiple lines. Join them by detecting
    that a new entry starts with a recognizable entry-start pattern — but only
    if the current buffer already contains a year (which means the previous
    entry is complete).

    Without the year check, lines like "Chang, W. K., Cross, Z. R., ..." inside
    a multi-author entry get misdetected as new entries, causing phantom
    reference splits.
    """
    lines = [l.rstrip() for l in references_text.split("\n")]
    entries, buf = [], []

    # Patterns that signal a plausible new reference entry start:
    entry_start_patterns = [
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+,\s+[A-Z]"),               # APA: "Palma, E."
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+\s+[A-Z][a-zA-ZÀ-ÿ'\-]+,"),# MLA-style: "Kiran GL Lee,"
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+\.\s*\("),                 # "AIBS. (2023" / "Pastidja. (2022"
        re.compile(r"^\s*[A-Z][A-ZÀ-ÿ\s]{10,}"),                        # ALL-CAPS title: "HARMONIZED GENDER..."
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]+\s+){1,6}\|"),           # "Gender Mainstreaming |"
        re.compile(r"^\s*Gender[- ]\w"),                                 # "Gender-Responsive ..."
        re.compile(r"^\s*(?:Exploration|Optimization|From Detection|How to|Submission)"),
        re.compile(r"^\s*de\s+[A-Z][a-zA-ZÀ-ÿ'\-]+"),                    # "de Vassimon Manela"
        re.compile(r"^\s*Sans\s+Auteur"),                                # Sans Auteur.
        # Multi-word name starts ("Huy Quoc To, ...", "Kiran GL Lee, ...")
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]*\s+){2,4}[A-Z][a-zA-ZÀ-ÿ'\-]+,"),
        # Single surname + ampersand ("Ali, & Nasrawi, D. A.")
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+,\s*&\s+"),
        # Title-style entries that end with a year like "Title Report (2024)."
        # Caught by looking for 3+ capitalized words near start, followed by text that ends with (YYYY)
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]+[\s\-]){2,}.*?\(\d{4}\)"),
    ]

    # A completed entry has a 4-digit year OR "(n.d.)" somewhere
    year_anywhere = re.compile(r"\b(19|20)\d{2}[a-z]?\b|\(n\.d\.\)", re.IGNORECASE)

    def _is_entry_start(line: str) -> bool:
        return any(p.match(line) for p in entry_start_patterns)

    def _buf_has_year() -> bool:
        """True if the current buffer already contains a year — i.e. the
        previous entry is likely complete and anything that follows is a
        new entry."""
        return bool(year_anywhere.search(" ".join(buf)))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line always flushes — it's a strong signal
            if buf:
                entries.append(" ".join(buf).strip())
                buf = []
            continue

        # A line that LOOKS like an entry start only actually starts a new
        # entry if the buffered previous entry already has a year in it.
        # Otherwise it's a wrapped continuation of an ongoing author list.
        if _is_entry_start(stripped) and buf and _buf_has_year():
            entries.append(" ".join(buf).strip())
            buf = [stripped]
        else:
            buf.append(stripped)

    if buf:
        entries.append(" ".join(buf).strip())

    # Drop entries that are obviously too short to be a real reference
    return [e for e in entries if len(e) > 20]


def parse_references(references_text: str) -> list[Reference]:
    """Parse the references section into Reference records."""
    if not references_text.strip():
        return []

    refs: list[Reference] = []

    for entry_id, entry in enumerate(_normalize_ref_lines(references_text)):
        # Try to capture the author block up to the year
        m = APA_AUTHOR_BLOCK.match(entry)
        if not m:
            # fall back: try to find a year and take everything before it as authors
            ym = YEAR_IN_PAREN.search(entry) or YEAR_BARE.search(entry)
            if not ym:
                continue
            author_block = entry[:ym.start()].rstrip(" .,")
        else:
            author_block = m.group("authors")

        # Extract year (prefer "(YYYY)", fall back to bare year)
        ym = YEAR_IN_PAREN.search(entry)
        year = ym.group(1) if ym else None
        if year is None:
            ym_bare = YEAR_BARE.search(entry)
            year = ym_bare.group(0) if ym_bare else None

        # Pull every "Surname, Given" piece out of the author block
        seen_in_entry: set[str] = set()
        for piece in AUTHOR_PIECE.finditer(author_block):
            surname = piece.group("surname")
            given   = piece.group("given").strip()

            # Determine whether `given` is initials or a full first name
            if re.fullmatch(r"(?:[A-Z]\.\s*)+", given):
                initials   = re.sub(r"\s+", " ", given).strip()
                first_name = ""
            else:
                # Full name — store the first token as first_name, keep
                # an initial form too for consistency
                first_name = given.split()[0]
                initials   = first_name[0] + "."

            seen_in_entry.add(surname.lower())
            refs.append(Reference(
                surname    = surname,
                initials   = initials,
                year       = year,
                first_name = first_name,
                raw        = entry,
                entry_id   = entry_id,
            ))

        # MLA co-author pass: catch "and Tom Cole" / "and Jane Smith" —
        # uninverted second+ authors that AUTHOR_PIECE cannot match.
        for piece in MLA_COAUTHOR.finditer(author_block):
            surname = piece.group("surname")
            first   = piece.group("first")

            # Skip if we already captured this surname from this entry
            if surname.lower() in seen_in_entry:
                continue
            # Basic sanity: avoid common words that might accidentally match
            if first.lower() in {"the", "an", "this", "that", "it", "a"}:
                continue

            seen_in_entry.add(surname.lower())
            refs.append(Reference(
                surname    = surname,
                initials   = first[0] + ".",
                year       = year,
                first_name = first,
                raw        = entry,
                entry_id   = entry_id,
            ))

    # Deduplicate within the references list itself
    seen, unique = set(), []
    for r in refs:
        if r.canonical_key in seen:
            continue
        seen.add(r.canonical_key)
        unique.append(r)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# In-text citation scanning
# ══════════════════════════════════════════════════════════════════════════════

# Matches:
#   Palma (2004)
#   Palma and Cruz (2004)
#   Palma et al. (2004)
#   (Palma, 2004)
#   (Palma & Cruz, 2004)
#   (Palma et al., 2004)
IN_TEXT_NARRATIVE = re.compile(
    r"""
    \b
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    (?:\s+(?:and|&)\s+(?P<surname2>[A-Z][a-zA-ZÀ-ÿ'\-]+))?    # optional "and Cruz"
    (?:\s+et\s+al\.?)?                                        # optional "et al."
    \s*\((?P<year>\d{4})[a-z]?\)
    """,
    re.VERBOSE,
)

IN_TEXT_PARENTHETICAL = re.compile(
    r"""
    \(\s*
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    (?:\s+(?:and|&)\s+(?P<surname2>[A-Z][a-zA-ZÀ-ÿ'\-]+))?
    (?:\s+et\s+al\.?)?
    \s*,\s*(?P<year>\d{4})[a-z]?
    \s*\)
    """,
    re.VERBOSE,
)


def find_body_citations(body_text: str) -> set[tuple[str, str]]:
    """
    Return a set of (surname_lower, year) pairs that are cited in the body.
    Captures both surnames in "Palma and Cruz (2004)" style citations so
    co-authors can be cross-checked correctly.
    """
    found: set[tuple[str, str]] = set()
    for pat in (IN_TEXT_NARRATIVE, IN_TEXT_PARENTHETICAL):
        for m in pat.finditer(body_text):
            year = m.group("year")
            found.add((m.group("surname").lower(), year))
            s2 = m.groupdict().get("surname2")
            if s2:
                found.add((s2.lower(), year))
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Gender resolution
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenderResult:
    gender: str      # "male" | "female" | "unknown"
    confidence: float
    source: str


def resolve_gender(first_name: str, initials: str) -> GenderResult:
    """
    Prefer a full first name. If only initials are available, return unknown
    (initials cannot be reliably sexed — "S." could be Silvio or Sofia).
    """
    if first_name:
        return _gender_from_name(first_name)
    # Initials-only: we can't classify reliably
    return GenderResult(gender="unknown", confidence=0.0, source="initials-only")


def _gender_from_name(first_name: str) -> GenderResult:
    # Strip any trailing period / punctuation
    clean = first_name.strip(" .,").split("-")[0]
    if not clean:
        return GenderResult("unknown", 0.0, "none")

    # 1) names-dataset (global coverage)
    try:
        info = _get_name_ds().search(clean)
        fn = (info or {}).get("first_name") if info else None
        if fn and fn.get("gender"):
            g = fn["gender"]
            male, female = g.get("Male", 0.0), g.get("Female", 0.0)
            top = max(male, female)
            if top >= 0.6:
                return GenderResult(
                    gender     = "male" if male > female else "female",
                    confidence = round(top, 2),
                    source     = "names-dataset",
                )
    except Exception:
        pass

    # 2) gender_guesser fallback
    try:
        gg_result = _get_gg().get_gender(clean)
        mapping = {
            "male":          ("male",    0.9),
            "mostly_male":   ("male",    0.7),
            "female":        ("female",  0.9),
            "mostly_female": ("female",  0.7),
            "andy":          ("unknown", 0.5),
            "unknown":       ("unknown", 0.0),
        }
        g, conf = mapping.get(gg_result, ("unknown", 0.0))
        return GenderResult(gender=g, confidence=conf, source="gender-guesser")
    except Exception:
        return GenderResult("unknown", 0.0, "none")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def analyze_authorship(full_text: str) -> dict:
    """
    Strict authorship analysis: references are ground truth, body citations
    confirm usage, duplicates are collapsed.
    """
    body, references = split_body_and_references(full_text)
    refs             = parse_references(references)
    body_citations   = find_body_citations(body)

    # Entry-level cross-check: if ANY author from an entry is matched in
    # the body (by surname+year or just surname), the entire entry counts
    # as cited. This handles "Trinkenreich et al. (2022)" → all co-authors
    # and "Cruz and Reyes (2018)" → both Cruz and Reyes.
    cited_surnames_by_year: dict[str, set[str]] = {}
    cited_surnames_any: set[str] = set()
    for surname, year in body_citations:
        cited_surnames_by_year.setdefault(year, set()).add(surname)
        cited_surnames_any.add(surname)

    # Group references by entry_id so co-authors can vouch for each other
    entries_cited: set[int] = set()
    for ref in refs:
        year_key = ref.year or ""
        surname_key = ref.surname.lower()
        if surname_key in cited_surnames_by_year.get(year_key, set()):
            entries_cited.add(ref.entry_id)
        elif surname_key in cited_surnames_any:
            # A surname-only match (rare edge case) — still counts
            entries_cited.add(ref.entry_id)

    # Strict: only keep references belonging to entries that were cited
    authors = []
    for ref in refs:
        if ref.entry_id not in entries_cited:
            continue

        g = resolve_gender(ref.first_name, ref.initials)
        authors.append({
            "surname":       ref.surname,
            "initials":      ref.initials,
            "first_name":    ref.first_name,
            "year":          ref.year,
            "gender":        g.gender,
            "confidence":    g.confidence,
            "source":        g.source,
            "cited_in_body": True,
        })

    # Tallies
    male    = sum(1 for a in authors if a["gender"] == "male")
    female  = sum(1 for a in authors if a["gender"] == "female")
    unknown = sum(1 for a in authors if a["gender"] == "unknown")

    return {
        "authors":       authors,
        "male_count":    male,
        "female_count":  female,
        "unknown_count": unknown,
        "diagnostics": {
            "references_parsed":      len(refs),
            "body_citations_found":   len(body_citations),
            "authors_after_crosscheck": len(authors),
            "references_section_present": bool(references.strip()),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Legacy per-sentence API — kept for backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

def detect_authorship(sentence: str) -> list:
    """
    Legacy API. Operates on a single sentence without references context, so
    it cannot cross-check or deduplicate properly. Prefer analyze_authorship().
    """
    import warnings
    warnings.warn(
        "detect_authorship(sentence) is a legacy API and cannot cross-check "
        "names against a references section. Use analyze_authorship(full_text) "
        "for correct results.",
        DeprecationWarning,
        stacklevel=2,
    )
    body_citations = find_body_citations(sentence)
    return [
        {"name": s.title(), "year": y, "gender": "unknown",
         "confidence": 0.0, "source": "legacy-no-crosscheck"}
        for s, y in body_citations
    ]