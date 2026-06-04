
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from names_dataset import NameDataset
import gender_guesser.detector as gender_guesser


# Lazy singletons

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


# Section splitting

REFERENCES_HEADINGS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works?\s+cited\s*$",
    r"^\s*literature\s+cited\s*$",
    r"^\s*reference\s+list\s*$",
]

POST_REFERENCES_HEADINGS = [
    r"^\s*appendix(?:\s+[A-Z0-9])?\s*$",
    r"^\s*appendices\s*$",
    r"^\s*acknowledgements?\s*$",
    r"^\s*author\s+biographies?\s*$",
    r"^\s*about\s+the\s+authors?\s*$",
]


def split_body_and_references(text: str) -> tuple[str, str]:
    lines = text.split("\n")

    ref_start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line or len(line) > 80:
            continue
        for pat in REFERENCES_HEADINGS:
            if re.match(pat, line, re.IGNORECASE):
                ref_start_idx = i
                break
        if ref_start_idx is not None:
            break

    if ref_start_idx is None:
        return text, ""

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


# Reference parsing (APA / MLA)

@dataclass(frozen=True)
class Reference:
    surname: str
    initials: str
    year: Optional[str]
    first_name: str
    raw: str
    entry_id: int = 0

    @property
    def canonical_key(self) -> str:
        """Key used for deduplication — per author, not per entry."""
        return f"{self.surname.lower()}|{self.year or ''}"


APA_AUTHOR_BLOCK = re.compile(
    r"""
    ^ 
    (?P<authors> 
        [A-Z][a-zA-ZÀ-ÿ'\-]+ 
        ,\s* 
        (?:[A-Z]\.(?:\s*[A-Z]\.)*|[A-Z][a-zA-ZÀ-ÿ'\-]+) 
        (?: 
            (?:\s*,\s*(?:&|and)\s*|\s*,\s+)
            [A-Z][a-zA-ZÀ-ÿ'\-]+
            ,\s*
            (?:[A-Z]\.(?:\s*[A-Z]\.)*|[A-Z][a-zA-ZÀ-ÿ'\-]+)
        )*
    )
    \s*
    (?:\(|\.\s*) 
    """,
    re.VERBOSE,
)

YEAR_IN_PAREN = re.compile(r"\((\d{4})[a-z]?\)")
YEAR_BARE     = re.compile(r"\b(19|20)\d{2}\b")


AUTHOR_PIECE = re.compile(
    r"""
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    \s*,\s*
    (?P<given>
        (?:[A-Z]\.(?:\s*[A-Z]\.)*) 
        |
        (?:[A-Z][a-zA-ZÀ-ÿ'\-]+(?:\s+[A-Z][a-zA-ZÀ-ÿ'\-]+)?) 
    )
    """,
    re.VERBOSE,
)


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

    lines = [l.rstrip() for l in references_text.split("\n")]
    entries, buf = [], []

    entry_start_patterns = [
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+,\s+[A-Z]"), 
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+\s+[A-Z][a-zA-ZÀ-ÿ'\-]+,"),
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+\.\s*\("), 
        re.compile(r"^\s*[A-Z][A-ZÀ-ÿ\s]{10,}"), 
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]+\s+){1,6}\|"), 
        re.compile(r"^\s*Gender[- ]\w"), 
        re.compile(r"^\s*(?:Exploration|Optimization|From Detection|How to|Submission)"),
        re.compile(r"^\s*de\s+[A-Z][a-zA-ZÀ-ÿ'\-]+"), 
        re.compile(r"^\s*Sans\s+Auteur"), 
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]*\s+){2,4}[A-Z][a-zA-ZÀ-ÿ'\-]+,"),
        re.compile(r"^\s*[A-Z][a-zA-ZÀ-ÿ'\-]+,\s*&\s+"),
        re.compile(r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ'\-]+[\s\-]){2,}.*?\(\d{4}\)"),
    ]


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
            if buf:
                entries.append(" ".join(buf).strip())
                buf = []
            continue


        if _is_entry_start(stripped) and buf and _buf_has_year():
            entries.append(" ".join(buf).strip())
            buf = [stripped]
        else:
            buf.append(stripped)

    if buf:
        entries.append(" ".join(buf).strip())

    return [e for e in entries if len(e) > 20]


def parse_references(references_text: str) -> list[Reference]:
    if not references_text.strip():
        return []

    refs: list[Reference] = []

    for entry_id, entry in enumerate(_normalize_ref_lines(references_text)):
        m = APA_AUTHOR_BLOCK.match(entry)
        if not m:
            ym = YEAR_IN_PAREN.search(entry) or YEAR_BARE.search(entry)
            if not ym:
                continue
            author_block = entry[:ym.start()].rstrip(" .,")
        else:
            author_block = m.group("authors")

        ym = YEAR_IN_PAREN.search(entry)
        year = ym.group(1) if ym else None
        if year is None:
            ym_bare = YEAR_BARE.search(entry)
            year = ym_bare.group(0) if ym_bare else None

        seen_in_entry: set[str] = set()
        for piece in AUTHOR_PIECE.finditer(author_block):
            surname = piece.group("surname")
            given   = piece.group("given").strip()
            if re.fullmatch(r"(?:[A-Z]\.\s*)+", given):
                initials   = re.sub(r"\s+", " ", given).strip()
                first_name = ""
            else:
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

        for piece in MLA_COAUTHOR.finditer(author_block):
            surname = piece.group("surname")
            first   = piece.group("first")

            if surname.lower() in seen_in_entry:
                continue
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

    seen, unique = set(), []
    for r in refs:
        if r.canonical_key in seen:
            continue
        seen.add(r.canonical_key)
        unique.append(r)
    return unique


# In-text citation scanning

IN_TEXT_NARRATIVE = re.compile(
    r"""
    \b
    (?P<surname>[A-Z][a-zA-ZÀ-ÿ'\-]+)
    (?:\s+(?:and|&)\s+(?P<surname2>[A-Z][a-zA-ZÀ-ÿ'\-]+))? 
    (?:\s+et\s+al\.?)? 
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
    found: set[tuple[str, str]] = set()
    for pat in (IN_TEXT_NARRATIVE, IN_TEXT_PARENTHETICAL):
        for m in pat.finditer(body_text):
            year = m.group("year")
            found.add((m.group("surname").lower(), year))
            s2 = m.groupdict().get("surname2")
            if s2:
                found.add((s2.lower(), year))
    return found


# Gender resolution

@dataclass
class GenderResult:
    gender: str      # "male", "female", "unknown"
    confidence: float
    source: str


def resolve_gender(first_name: str, initials: str) -> GenderResult:
    if first_name:
        return _gender_from_name(first_name)
    # Initials-only
    return GenderResult(gender="unknown", confidence=0.0, source="initials-only")


def _gender_from_name(first_name: str) -> GenderResult:
    clean = first_name.strip(" .,").split("-")[0]
    if not clean:
        return GenderResult("unknown", 0.0, "none")

    # names-dataset 
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

    # gender guesser fallback
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


# Main entry

def analyze_authorship(full_text: str) -> dict:

    body, references = split_body_and_references(full_text)
    refs             = parse_references(references)
    body_citations   = find_body_citations(body)


    cited_surnames_by_year: dict[str, set[str]] = {}
    cited_surnames_any: set[str] = set()
    for surname, year in body_citations:
        cited_surnames_by_year.setdefault(year, set()).add(surname)
        cited_surnames_any.add(surname)

    entries_cited: set[int] = set()
    for ref in refs:
        year_key = ref.year or ""
        surname_key = ref.surname.lower()
        if surname_key in cited_surnames_by_year.get(year_key, set()):
            entries_cited.add(ref.entry_id)
        elif surname_key in cited_surnames_any:
            entries_cited.add(ref.entry_id)

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


# Legacy per-sentence API (kept for backward compatibility)

def detect_authorship(sentence: str) -> list:
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