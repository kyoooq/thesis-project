from collections import defaultdict
from html import escape
from math import gcd

import pandas as pd

from extractor.router import extract_text_and_sentences
from classifier.bert_classifier import predict_batch, LABELS, MODEL_NAME
from authorship.author_detector import analyze_authorship

from pipeline.scoring import (
    compute_overall_score,
    compute_representation_penalty,
    check_disaggregation_needed,
    extract_disaggregated_counts,
    find_gendered_word,
    lookup_neutral,
    SEMANTIC_WEIGHT,
)


# Display metadata per label — representation removed
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
}


def _build_issue_html(sentence: str, phrase: str | None) -> str:
    safe_sentence = escape(sentence)
    if phrase:
        safe_phrase   = escape(phrase)
        safe_sentence = safe_sentence.replace(
            safe_phrase, f"<strong>{safe_phrase}</strong>", 1
        )
    return f'<p>Found in: <em>"{safe_sentence}"</em></p>'


def _build_row(label: str, sentence: str, probability: float,
               triggers: list, total_sentences: int = 1) -> dict | None:
    meta = LABEL_META[label]
    score_str = f"{probability:.2f}"

    # Contribution = ratio-based share of semantic penalty
    contribution_str = f"-{(1 / max(total_sentences, 1)) * 100 * SEMANTIC_WEIGHT:.2f}%"

    if label == "gender_sensitive":
        phrase  = find_gendered_word(sentence)
        if phrase is None:
            phrase  = sentence
            neutral = None
        else:
            neutral = lookup_neutral(phrase)
        rec_to   = neutral if neutral else "Consider a neutral alternative"
        rec_note = meta["rec_note"]
    else:
        # Stereotyping: show the whole sentence
        phrase   = sentence
        rec_to   = "Rephrase"
        rec_note = meta["rec_note"]

    return {
        "aspect":         meta["aspect"],
        "phrase":         phrase,
        "recommendation": rec_to,
        "model":          MODEL_NAME,
        "score":          score_str,
        "contribution":   contribution_str,
        "detail": {
            "title":    phrase if (label == "gender_sensitive" and phrase != sentence) else meta["aspect"],
            "subtitle": meta["subtitle"],
            "issue":    _build_issue_html(
                sentence,
                phrase if (label == "gender_sensitive" and phrase != sentence) else None,
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
    total_known = male_count + female_count
    if total_known == 0:
        phrase   = "No author names detected"
        rec_note = "No authors could be identified for gender analysis."
    else:
        divisor      = gcd(male_count, female_count) or 1
        male_r       = male_count   // divisor
        female_r     = female_count // divisor
        phrase       = f"{male_r}:{female_r} male-to-female author ratio"
        rec_note     = (
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
        "contribution":   f"-{round(abs(0.5 - female_ratio) * 2 * 100 * 0.2, 2) if female_ratio is not None else 0:.2f}%",
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


def _build_representation_row(status: str, male_count: int | None,
                               female_count: int | None,
                               female_ratio: float | None,
                               penalty: float) -> dict:
    """Build the single summary row for representation when flagged."""
    score_str = f"{round(1 - (penalty / 20), 2):.2f}"  # normalise to 0-1

    if status == "missing":
        phrase     = "No sex-disaggregated data detected"
        issue_html = (
            "<p>This paper appears to involve human participants or target "
            "beneficiaries but does not report sex-disaggregated data. "
            "Consider adding a breakdown of male and female counts.</p>"
        )
        rec_to   = "Add sex-disaggregated data (e.g. number of male and female respondents)"
        rec_note = "Sex-disaggregated reporting is required for GAD-responsive research."

    else:  # skewed
        from math import gcd as _gcd
        divisor  = _gcd(male_count, female_count) or 1
        male_r   = male_count   // divisor
        female_r = female_count // divisor
        phrase   = f"{male_r}:{female_r} male-to-female participant ratio"
        issue_html = (
            f"<p>Disaggregated data detected: "
            f"<strong>{male_count}</strong> male, "
            f"<strong>{female_count}</strong> female participants. "
            f"The ratio is skewed — aim for a more balanced distribution.</p>"
        )
        rec_to   = "Balance participant representation closer to 50/50"
        rec_note = (
            "A skewed sex ratio in the study sample may limit the "
            "generalizability of findings across genders."
        )

    return {
        "aspect":         "Representation",
        "phrase":         phrase,
        "recommendation": rec_to,
        "model":          "Rule-based (regex pattern matching)",
        "score":          score_str,
        "contribution":   f"-{penalty:.2f}%",
        "detail": {
            "title":    "Sex-Disaggregated Data",
            "subtitle": "Representation issue",
            "issue":    issue_html,
            "recFrom":  "current reporting",
            "recTo":    rec_to,
            "recNote":  rec_note,
            "model":    "Rule-based (regex pattern matching)",
            "score":    score_str,
        },
    }


def analyze_paper(file_path: str, threshold: float = 0.7,
                  save_csv: bool = True) -> dict:
    full_text, sentences = extract_text_and_sentences(file_path)

    # ── Authorship ────────────────────────────────────────────────────────────
    authorship_result = analyze_authorship(full_text)
    male_count    = authorship_result["male_count"]
    female_count  = authorship_result["female_count"]
    unknown_count = authorship_result["unknown_count"]

    # ── Representation: check before sentence loop ────────────────────────────
    needs_disaggregation = check_disaggregation_needed(full_text)
    if needs_disaggregation:
        rep_male, rep_female = extract_disaggregated_counts(full_text)
    else:
        rep_male, rep_female = None, None

    representation_info = compute_representation_penalty(
        needs_disaggregation = needs_disaggregation,
        male_count           = rep_male,
        female_count         = rep_female,
    )

    # ── BERT sentence loop (gender_sensitive + stereotyping only) ─────────────
    rows              = []
    flagged_sentences = set()
    label_flag_counts = defaultdict(int)
    csv_flagged_rows  = []

    bert_results = predict_batch(sentences, batch_size=16)

    for sentence, bert_result in zip(sentences, bert_results):
        sentence_was_flagged = False
        csv_row = {"sentence": sentence}

        # gender_sensitive: BERT or lexicon
        gs_result     = bert_result["gender_sensitive"]
        gendered_word = find_gendered_word(sentence)

        bert_fires    = gs_result["predicted"]
        lexicon_fires = gendered_word is not None

        if bert_fires or lexicon_fires:
            probability = 1.0 if lexicon_fires else gs_result["probability"]
            row = _build_row(
                label           = "gender_sensitive",
                sentence        = sentence,
                probability     = probability,
                triggers        = gs_result["triggers"],
                total_sentences = len(sentences),
            )
            if row is not None:
                sentence_was_flagged = True
                label_flag_counts["gender_sensitive"] += 1
                rows.append(row)

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

        # stereotyping: BERT only
        r = bert_result["stereotyping"]
        if r["predicted"]:
            row = _build_row(
                label           = "stereotyping",
                sentence        = sentence,
                probability     = r["probability"],
                triggers        = r["triggers"],
                total_sentences = len(sentences),
            )
            if row is not None:
                sentence_was_flagged = True
                label_flag_counts["stereotyping"] += 1
                rows.append(row)
                csv_row["stereotyping"] = (
                    f"{r['probability']:.4f} → {', '.join(r['triggers'])}"
                )
        else:
            csv_row["stereotyping"] = "No"

        if sentence_was_flagged:
            flagged_sentences.add(sentence)
            csv_flagged_rows.append(csv_row)

    # ── Scoring ───────────────────────────────────────────────────────────────
    score_info = compute_overall_score(
        flagged_count       = len(flagged_sentences),
        total_sentences     = len(sentences),
        male_count          = male_count,
        female_count        = female_count,
        label_flag_counts   = dict(label_flag_counts),
        representation_info = representation_info,
    )

    # ── Append summary rows for flagged dimensions ────────────────────────────
    if score_info["representation_flagged"]:
        rows.append(_build_representation_row(
            status       = score_info["representation_status"],
            male_count   = score_info["rep_male_count"],
            female_count = score_info["rep_female_count"],
            female_ratio = score_info["rep_female_ratio"],
            penalty      = score_info["representation_penalty"],
        ))

    if score_info["authorship_flagged"]:
        rows.append(_build_authorship_row(
            male_count    = male_count,
            female_count  = female_count,
            unknown_count = unknown_count,
            female_ratio  = score_info["female_ratio"],
        ))

    # ── Audit CSVs ────────────────────────────────────────────────────────────
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

    # ── Final shape ───────────────────────────────────────────────────────────
    overall_score = score_info["score"]
    return {
        "overallScore": overall_score,
        "overallLabel": f"{overall_score}% RESPONSIVE",
        "stats": {
            "totalSentences":          len(sentences),
            "flaggedSentences":        len(flagged_sentences),
            "flagsByLabel":            dict(label_flag_counts),
            "maleNames":               male_count,
            "femaleNames":             female_count,
            "unknownNames":            unknown_count,
            "femaleRatio":             score_info["female_ratio"],
            "semanticPenalty":         score_info["semantic_penalty"],
            "ratioPenalty":            score_info["ratio_penalty"],
            "floorPenalty":            score_info["floor_penalty"],
            "authorshipPenalty":       score_info["authorship_penalty"],
            "authorshipDiagnostics":   authorship_result["diagnostics"],
            # representation
            "representationStatus":    score_info["representation_status"],
            "representationPenalty":   score_info["representation_penalty"],
            "repMaleCount":            score_info["rep_male_count"],
            "repFemaleCount":          score_info["rep_female_count"],
            "repFemaleRatio":          score_info["rep_female_ratio"],
        },
        "rows": rows,
    }