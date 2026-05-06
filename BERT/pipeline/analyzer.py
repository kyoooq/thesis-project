from collections import defaultdict
from html import escape

import pandas as pd

from extractor.router import extract_text_and_sentences
from classifier.bert_classifier import predict, LABELS, MODEL_NAME
from authorship.author_detector import analyze_authorship
from math import gcd

from pipeline.scoring import (
    compute_overall_score,
    find_gendered_word,
    lookup_neutral,
)


# ── Display metadata per label ────────────────────────────────────────────────
LABEL_META = {
    "gender_sensitive": {
        "aspect":   "Gender-Sensitive",
        "subtitle": "Gender-sensitive issue",
        "rec_note": "Use a gender-neutral alternative.",
    },
    "stereotyping": {
        "aspect":   "Stereotyping",
        "subtitle": "Stereotyping issue",
        "rec_note": "Rephrase to avoid gender-based generalization.",
    },
    "representation": {
        "aspect":   "Representation",
        "subtitle": "Representation issue",
        "rec_note": "Balance perspectives and give equal framing to women and men.",
    },
}


def _build_issue_html(sentence: str, phrase: str | None) -> str:
    """Build the 'issue' HTML shown in the detail modal."""
    safe_sentence = escape(sentence)
    if phrase:
        safe_phrase = escape(phrase)
        safe_sentence = safe_sentence.replace(
            safe_phrase, f"<strong>{safe_phrase}</strong>", 1
        )
    return f'<p>Found in: <em>"{safe_sentence}"</em></p>'


def _build_row(label: str, sentence: str, probability: float,
               triggers: list) -> dict | None:
    meta = LABEL_META[label]
    score_str = f"{probability:.2f}"

    if label == "gender_sensitive":
        phrase = find_gendered_word(sentence)
        if phrase is None:
            # BERT flagged but no canonical gendered word found.
            # Under hybrid logic this can happen when BERT detects
            # semantic gender content (e.g., "she/he" patterns) without
            # a lexicon-matchable word. Fall back to the whole sentence.
            phrase = sentence
            neutral = None
        else:
            neutral = lookup_neutral(phrase)

        rec_to   = neutral if neutral else "Consider a neutral alternative"
        rec_note = meta["rec_note"]

    else:
        # Stereotyping / representation: show the whole sentence
        phrase   = sentence
        rec_to   = "Rephrase"
        rec_note = meta["rec_note"]

    return {
        "aspect":         meta["aspect"],
        "phrase":         phrase,
        "recommendation": rec_to,
        "model":          MODEL_NAME,
        "score":          score_str,
        "detail": {
            "title":    phrase if (label == "gender_sensitive" and phrase != sentence) else meta["aspect"],
            "subtitle": meta["subtitle"],
            "issue":    _build_issue_html(
                sentence,
                phrase if (label == "gender_sensitive" and phrase != sentence) else None
            ),
            "recFrom":  phrase if (label == "gender_sensitive" and phrase != sentence) else "current wording",
            "recTo":    rec_to,
            "recNote":  rec_note,
            "model":    MODEL_NAME,
            "score":    score_str,
        },
    }


def _build_authorship_row(male_count: int, female_count: int,
                          unknown_count: int, female_ratio: float | None) -> dict:
    """Build the single summary row for authorship when imbalanced."""
    total_known = male_count + female_count
    if total_known == 0:
        phrase = "No author names detected"
        rec_note = "No authors could be identified for gender analysis."
    else:
        # Reduce to simplest integer ratio, e.g. 20:2 → 10:1
        divisor    = gcd(male_count, female_count) or 1
        male_ratio   = male_count // divisor
        female_ratio = female_count // divisor
        phrase = f"{male_ratio}:{female_ratio} male-to-female author ratio"
        rec_note = (
            "Aim for balanced citation — approximately 50% women and 50% men "
            "among cited authors."
        )

    score_str = (
        f"{round((1 - abs(0.5 - female_ratio) * 2), 2):.2f}"
        if female_ratio is not None else "0.00"
    )

    issue_html = (
        f"<p>Author gender distribution in citations: "
        f"<strong>{male_count}</strong> male, "
        f"<strong>{female_count}</strong> female, "
        f"<strong>{unknown_count}</strong> unknown.</p>"
    )

    return {
        "aspect":         "Authorship",
        "phrase":         phrase,
        "recommendation": "Balance citations 50/50",
        "model":          "Rule-based (spaCy NER + names-dataset)",
        "score":          score_str,
        "detail": {
            "title":    "Authorship Balance",
            "subtitle": "Authorship issue",
            "issue":    issue_html,
            "recFrom":  "imbalanced",
            "recTo":    "50/50 balance",
            "recNote":  rec_note,
            "model":    "Rule-based (spaCy NER + names-dataset)",
            "score":    score_str,
        },
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_paper(file_path: str, threshold: float = 0.7,
                  save_csv: bool = True) -> dict:
    """
    Analyze a .pdf or .docx file and return a React-ready result dict.
    """
    # ── Extraction ───────────────────────────────────────────────────────────
    full_text, sentences = extract_text_and_sentences(file_path)

    # ── Authorship (once, on the full paper) ─────────────────────────────────
    authorship_result = analyze_authorship(full_text)
    male_count    = authorship_result["male_count"]
    female_count  = authorship_result["female_count"]
    unknown_count = authorship_result["unknown_count"]

    # ── Per-sentence analysis (HYBRID for gender_sensitive) ──────────────────
    rows               = []
    flagged_sentences  = set()
    label_flag_counts  = defaultdict(int)
    csv_flagged_rows   = []

    for sentence in sentences:
        bert_result = predict(sentence)

        sentence_was_flagged = False
        csv_row = {"sentence": sentence}

        # ── gender_sensitive: HYBRID (BERT OR lexicon) ───────────────────
        gs_result     = bert_result["gender_sensitive"]
        gendered_word = find_gendered_word(sentence)

        bert_fires    = gs_result["predicted"]
        lexicon_fires = gendered_word is not None

        if bert_fires or lexicon_fires:
            # If lexicon fires, use rule-based confidence (1.0).
            # If only BERT fires, use BERT's probability.
            probability = 1.0 if lexicon_fires else gs_result["probability"]

            row = _build_row(
                label       = "gender_sensitive",
                sentence    = sentence,
                probability = probability,
                triggers    = gs_result["triggers"],
            )
            if row is not None:
                sentence_was_flagged = True
                label_flag_counts["gender_sensitive"] += 1
                rows.append(row)

                # Annotate the audit CSV with which signal(s) fired
                if bert_fires and lexicon_fires:
                    csv_row["gender_sensitive"] = (
                        f"BERT={gs_result['probability']:.4f} + LEX={gendered_word}"
                    )
                elif lexicon_fires:
                    csv_row["gender_sensitive"] = f"LEX → {gendered_word}"
                else:
                    csv_row["gender_sensitive"] = (
                        f"BERT={gs_result['probability']:.4f} → "
                        f"{', '.join(gs_result['triggers'])}"
                    )
        else:
            csv_row["gender_sensitive"] = "No"

        # ── stereotyping & representation: BERT-only ─────────────────────
        for label in ("stereotyping", "representation"):
            r = bert_result[label]
            if r["predicted"]:
                row = _build_row(
                    label       = label,
                    sentence    = sentence,
                    probability = r["probability"],
                    triggers    = r["triggers"],
                )
                if row is not None:
                    sentence_was_flagged = True
                    label_flag_counts[label] += 1
                    rows.append(row)
                    csv_row[label] = (
                        f"{r['probability']:.4f} → {', '.join(r['triggers'])}"
                    )
            else:
                csv_row[label] = "No"

        if sentence_was_flagged:
            flagged_sentences.add(sentence)
            csv_flagged_rows.append(csv_row)

    # ── Scoring ──────────────────────────────────────────────────────────────
    score_info = compute_overall_score(
        flagged_count     = len(flagged_sentences),
        total_sentences   = len(sentences),
        male_count        = male_count,
        female_count      = female_count,
        label_flag_counts = dict(label_flag_counts),
    )

    # Append authorship row if imbalanced
    if score_info["authorship_flagged"]:
        rows.append(_build_authorship_row(
            male_count    = male_count,
            female_count  = female_count,
            unknown_count = unknown_count,
            female_ratio  = score_info["female_ratio"],
        ))

    # ── Audit CSVs ───────────────────────────────────────────────────────────
    if save_csv:
        import os
        os.makedirs("output", exist_ok=True)
        if csv_flagged_rows:
            pd.DataFrame(csv_flagged_rows).to_csv(
                "output/flagged_sentences.csv", index=False
            )
        if authorship_result["authors"]:
            pd.DataFrame(authorship_result["authors"]).to_csv(
                "output/authorship_names.csv", index=False
            )

    # ── Final shape ──────────────────────────────────────────────────────────
    overall_score = score_info["score"]
    return {
        "overallScore": overall_score,
        "overallLabel": f"{overall_score}% RESPONSIVE",
        "stats": {
            "totalSentences":     len(sentences),
            "flaggedSentences":   len(flagged_sentences),
            "flagsByLabel":       dict(label_flag_counts),
            "maleNames":          male_count,
            "femaleNames":        female_count,
            "unknownNames":       unknown_count,
            "femaleRatio":        score_info["female_ratio"],
            "semanticPenalty":    score_info["semantic_penalty"],
            "ratioPenalty":       score_info["ratio_penalty"],
            "floorPenalty":       score_info["floor_penalty"],
            "authorshipPenalty":  score_info["authorship_penalty"],
            "authorshipDiagnostics": authorship_result["diagnostics"],
        },
        "rows": rows,
    }