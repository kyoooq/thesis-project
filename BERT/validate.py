"""
External validation of the GAD classifier against a labeled test set.

Tests the FULL pipeline behavior using HYBRID logic for gender_sensitive,
matching pipeline/analyzer.py:

    gender_sensitive fires if:
        BERT predicts gender_sensitive (above threshold)
        OR
        find_gendered_word returns a lexicon match

    stereotyping / representation fire if BERT predicts them (above threshold).

Each run appends its accuracy summary to output/validation_history.json so
that subsequent runs show a tabular history of how the model has performed
on the same test set across retrainings / threshold changes.

Usage (from your thesis project root):
    python validate.py test_sets/trix_psenka_2003.csv
    python validate.py test_sets/moss_racusin_2012.csv

Optional:
    python validate.py test_sets/trix_psenka_2003.csv --label "run_2_seed_123"
        (custom label for this run; defaults to a timestamp)

    python validate.py test_sets/trix_psenka_2003.csv --reset-history
        (clears the saved history for this test set before recording)

CSV format:
    Three columns required (header row required):
        section, sentence, expected_labels
    expected_labels is comma-separated; use "none" for sentences
    that should produce no flags.

Output:
    - Console: human-readable accuracy report + run-history table
    - File:    output/<csv_name>_validation.csv (per-sentence breakdown)
    - File:    output/validation_history.json (cross-run history)
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from classifier.bert_classifier import predict, LABELS
from pipeline.scoring import find_gendered_word


# ─── Pipeline-faithful labeling (HYBRID) ───────────────────────────────────────

def get_final_labels(sentence: str):
    bert_result = predict(sentence)
    final = {}
    for label in LABELS:
        bert_predicted = bert_result[label]["predicted"]
        if label == "gender_sensitive":
            lexicon_match = find_gendered_word(sentence) is not None
            final[label] = bool(bert_predicted or lexicon_match)
        else:
            final[label] = bool(bert_predicted)
    return final, bert_result


# ─── Test set loading ──────────────────────────────────────────────────────────

def load_test_set(csv_path: Path):
    test_data = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"section", "sentence", "expected_labels"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must have columns: {sorted(required)}. "
                f"Got: {reader.fieldnames}"
            )
        for row_num, row in enumerate(reader, start=2):
            section = row["section"].strip()
            sentence = row["sentence"].strip()
            raw_labels = row["expected_labels"].strip().lower()
            if raw_labels in ("none", ""):
                expected = set()
            else:
                expected = {lbl.strip() for lbl in raw_labels.split(",") if lbl.strip()}
                unknown = expected - set(LABELS)
                if unknown:
                    raise ValueError(
                        f"Row {row_num}: unknown label(s) {unknown}. "
                        f"Valid labels are: {LABELS}"
                    )
            test_data.append((section, sentence, expected))
    return test_data


# ─── History persistence ──────────────────────────────────────────────────────

HISTORY_FILE = Path(__file__).resolve().parent / "output" / "validation_history.json"


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_history(history: dict) -> None:
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def labels_to_vector(label_set: set) -> np.ndarray:
    return np.array([1 if lbl in label_set else 0 for lbl in LABELS], dtype=int)


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def quality_label(f1: float) -> str:
    if f1 >= 0.90: return "EXCELLENT"
    if f1 >= 0.80: return "GOOD"
    if f1 >= 0.70: return "FAIR"
    if f1 >= 0.50: return "MODERATE"
    if f1 > 0:     return "WEAK"
    return "N/A"


def hbar(width=78, char="="):
    return char * width


# ─── Main reporting ────────────────────────────────────────────────────────────

def run_validation(csv_path: Path, run_label: str | None,
                   reset_history: bool) -> None:
    test_data = load_test_set(csv_path)

    print(hbar())
    print(f"  ACCURACY REPORT — {csv_path.name}")
    print(hbar())
    print(f"  Test sentences:  {len(test_data)}")
    print(f"  Pipeline mode:   HYBRID (BERT + lexicon for gender_sensitive)")
    print(hbar())

    y_true_rows, y_pred_rows, csv_rows = [], [], []

    for section, sentence, expected in test_data:
        final_labels, bert_result = get_final_labels(sentence)
        gendered_word = find_gendered_word(sentence)

        true_vec = labels_to_vector(expected)
        pred_vec = np.array(
            [1 if final_labels[lbl] else 0 for lbl in LABELS], dtype=int
        )
        y_true_rows.append(true_vec)
        y_pred_rows.append(pred_vec)

        if final_labels["gender_sensitive"]:
            bert_fires = bert_result["gender_sensitive"]["predicted"]
            lex_fires  = gendered_word is not None
            gs_signal = ("BERT+LEX" if (bert_fires and lex_fires)
                         else "LEX" if lex_fires else "BERT")
        else:
            gs_signal = ""

        row = {
            "section": section,
            "sentence": sentence,
            "expected": ",".join(sorted(expected)) if expected else "(none)",
            "gendered_word_match": gendered_word or "",
            "gs_signal": gs_signal,
        }
        for lbl in LABELS:
            row[f"prob_{lbl}"]    = bert_result[lbl]["probability"]
            row[f"bert_{lbl}"]    = "Y" if bert_result[lbl]["predicted"] else "N"
            row[f"final_{lbl}"]   = "Y" if final_labels[lbl] else "N"
            row[f"correct_{lbl}"] = (
                "Y" if (lbl in expected) == final_labels[lbl] else "N"
            )
        csv_rows.append(row)

    y_true = np.array(y_true_rows)
    y_pred = np.array(y_pred_rows)
    cms = multilabel_confusion_matrix(y_true, y_pred)

    # ── 1) HEADLINE ACCURACY (table form) ─────────────────────────────────────
    exact_match  = float(np.all(y_true == y_pred, axis=1).mean())
    hamming_acc  = float((y_true == y_pred).mean())
    total_labels = y_true.size
    correct_labels = int((y_true == y_pred).sum())

    print()
    print("  HEADLINE ACCURACY")
    print("  " + "─" * 76)
    print(f"  {'Metric':<25} {'Value':>10} {'Detail':>40}")
    print("  " + "─" * 76)
    print(f"  {'Hamming accuracy':<25} {hamming_acc * 100:>9.1f}% "
          f"{f'({correct_labels} of {total_labels} labels correct)':>40}")
    print(f"  {'Exact-match accuracy':<25} {exact_match * 100:>9.1f}% "
          f"{'(all 3 labels correct on same sentence)':>40}")
    print()

    # ── 2) PER-LABEL ACCURACY ──────────────────────────────────────────────────
    print("  PER-LABEL ACCURACY")
    print("  " + "─" * 76)
    print(f"  {'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>8}  "
          f"{'Quality':>11}  {'Support':>8}")
    print("  " + "─" * 76)

    label_summary = []
    for i, lbl in enumerate(LABELS):
        tn, fp = cms[i][0]
        fn, tp = cms[i][1]
        support = int(tp + fn)
        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1        = safe_div(2 * precision * recall, precision + recall)
        if support == 0:
            print(f"  {lbl:<20} {'—':>10} {'—':>10} {'—':>8}  "
                  f"{'N/A':>11}  {support:>8}   "
                  f"(no expected examples)")
        else:
            print(f"  {lbl:<20} {precision*100:>9.1f}% {recall*100:>9.1f}% "
                  f"{f1*100:>7.1f}%  {quality_label(f1):>11}  {support:>8}")
        label_summary.append((lbl, tp, fp, fn, tn, support, precision, recall, f1))
    print()

    # ── 3) WHAT IT GOT RIGHT / WRONG ──────────────────────────────────────────
    print("  FLAGGING ACCURACY")
    print("  " + "─" * 76)
    print(f"  {'Label':<20} {'Correct flags':>14} {'False alarms':>14} "
          f"{'Missed':>10}")
    print("  " + "─" * 76)
    for lbl, tp, fp, fn, tn, *_ in label_summary:
        print(f"  {lbl:<20} {tp:>14} {fp:>14} {fn:>10}")
    print()
    print("    Correct flags  = the model correctly identified the issue")
    print("    False alarms   = the model flagged something that wasn't an issue")
    print("    Missed         = the model failed to flag a real issue")
    print()

    # ── 4) FALSE POSITIVES ON NEUTRAL TEXT ────────────────────────────────────
    neutral_idxs = [i for i, (_, _, exp) in enumerate(test_data) if not exp]
    neutral_correct = neutral_total = 0
    if neutral_idxs:
        neutral_pred = y_pred[neutral_idxs]
        neutral_correct = int((neutral_pred.sum(axis=1) == 0).sum())
        neutral_total = len(neutral_idxs)
        print("  NEUTRAL TEXT SAFETY CHECK")
        print("  " + "─" * 76)
        print(f"  {neutral_correct} of {neutral_total} neutral methodological "
              f"sentences correctly NOT flagged "
              f"({safe_div(neutral_correct, neutral_total) * 100:.0f}%)")
        if neutral_correct == neutral_total:
            print("  → No false positives on neutral text. ✓")
        else:
            print("  → WARNING: model produced false positives on neutral text.")
        print()

    # ── 5) SECTION BREAKDOWN ──────────────────────────────────────────────────
    print("  PER-SECTION BREAKDOWN")
    print("  " + "─" * 76)
    print(f"  {'Section':<32} {'n':>4} {'Accuracy':>12} {'All-correct':>14}")
    print("  " + "─" * 76)
    sections = sorted({s for s, _, _ in test_data})
    for sect in sections:
        idxs = [i for i, (s, _, _) in enumerate(test_data) if s == sect]
        sect_true = y_true[idxs]
        sect_pred = y_pred[idxs]
        sect_exact   = float(np.all(sect_true == sect_pred, axis=1).mean())
        sect_hamming = float((sect_true == sect_pred).mean())
        print(f"  {sect:<32} {len(idxs):>4} "
              f"{sect_hamming * 100:>11.1f}% {sect_exact * 100:>13.1f}%")
    print()
    print("    Accuracy    = % of label predictions correct in this section")
    print("    All-correct = % of sentences where ALL 3 labels were correct")
    print()

    # ── 6) PLAIN-ENGLISH SUMMARY ──────────────────────────────────────────────
    print("  PLAIN-ENGLISH SUMMARY")
    print("  " + "─" * 76)

    valid_f1 = [f1 for *_, support, _, _, f1 in label_summary if support > 0]
    avg_f1 = float(np.mean(valid_f1)) if valid_f1 else 0.0
    avg_precision = float(np.mean(
        [p for *_, support, p, _, _ in label_summary if support > 0]
    )) if valid_f1 else 0.0
    avg_recall = float(np.mean(
        [r for *_, support, _, r, _ in label_summary if support > 0]
    )) if valid_f1 else 0.0

    print(f"  • Across labels with data, the model achieves an F1 of "
          f"{avg_f1:.2f} on average.")
    print(f"  • Average precision: {avg_precision:.2f}  "
          f"(when model flags, how often it's right)")
    print(f"  • Average recall:    {avg_recall:.2f}  "
          f"(of real issues, how many it catches)")
    if avg_precision >= 0.90:
        print("  • The model rarely produces false positives.")
    if avg_recall < 0.80:
        print("  • The model is conservative — it under-flags rather than over-flags.")
    if neutral_idxs and (neutral_correct == neutral_total):
        print("  • Zero false positives on neutral methodological text.")
    print()

    # ── 7) RUN HISTORY TABLE ──────────────────────────────────────────────────
    history = {} if reset_history else load_history()
    test_set_key = csv_path.stem

    if reset_history and test_set_key in history:
        history.pop(test_set_key, None)
        print(f"  (run history for {test_set_key} cleared)")
        print()

    if test_set_key not in history:
        history[test_set_key] = []

    if run_label is None:
        run_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    this_run = {
        "label":        run_label,
        "hamming":      round(hamming_acc, 4),
        "exact":        round(exact_match, 4),
        "macro_f1":     round(avg_f1, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall":    round(avg_recall, 4),
        "per_label_f1": {
            lbl: round(f1, 4)
            for lbl, _, _, _, _, support, _, _, f1 in label_summary
            if support > 0
        },
    }
    history[test_set_key].append(this_run)
    save_history(history)

    runs = history[test_set_key]
    print(hbar(char="="))
    print(f"  RUN HISTORY — {csv_path.name}  "
          f"(this run is #{len(runs)})")
    print("  " + "─" * 76)
    print(f"  {'#':>3}  {'Label':<26} {'Hamming':>9} {'Exact':>8} "
          f"{'Macro F1':>10} {'Prec':>6} {'Recall':>7}")
    print("  " + "─" * 76)
    for idx, r in enumerate(runs, start=1):
        marker = "← this run" if idx == len(runs) else ""
        print(f"  {idx:>3}  {r['label'][:26]:<26} "
              f"{r['hamming']*100:>8.1f}% {r['exact']*100:>7.1f}% "
              f"{r['macro_f1']:>9.3f} {r['avg_precision']:>6.2f} "
              f"{r['avg_recall']:>7.2f}  {marker}")
    print(hbar(char="="))
    print(f"  Run history saved to: {HISTORY_FILE}")
    print(f"    (use --reset-history to clear, or --label NAME to tag a run)")
    print(hbar(char="="))

    # ── 8) Per-sentence audit CSV ──────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_validation.csv"

    fieldnames = [
        "section", "sentence", "expected", "gendered_word_match", "gs_signal"
    ]
    for lbl in LABELS:
        fieldnames += [
            f"prob_{lbl}", f"bert_{lbl}", f"final_{lbl}", f"correct_{lbl}"
        ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  Per-sentence results saved to: {out_path}")
    print(hbar())


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate the full GAD pipeline against a labeled CSV test set."
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label for this run (e.g. 'run_3_seed_2025'). "
             "Defaults to a timestamp."
    )
    parser.add_argument(
        "--reset-history", action="store_true",
        help="Clear saved run history for this test set before recording."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    run_validation(csv_path, args.label, args.reset_history)


if __name__ == "__main__":
    main()