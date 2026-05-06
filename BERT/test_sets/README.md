# External Validation Framework

This folder contains a generic validation script and labeled test sets drawn from peer-reviewed papers on gender bias. Use it to demonstrate that your BERT classifier generalizes beyond your own training data.

## Folder layout

Place these files in your thesis project root (where `main.py` lives):

```
thesis/
├── main.py
├── validate.py                   ← the generic script
├── test_sets/
│   ├── trix_psenka_2003.csv     ← test set 1
│   └── moss_racusin_2012.csv    ← test set 2
├── classifier/
│   └── bert_classifier.py
└── output/                       ← validation CSVs land here
```

## Running

From the project root:

```bash
python validate.py test_sets/trix_psenka_2003.csv
python validate.py test_sets/moss_racusin_2012.csv
```

Each run prints to console and saves a per-sentence CSV to `output/<csv_name>_validation.csv`.

## What you get

Per run:
- **Per-label classification report** — precision, recall, F1, support for each of `gender_sensitive`, `stereotyping`, `representation`.
- **Confusion matrix** per label — TP / FP / FN / TN counts.
- **Aggregate accuracy** — exact match (all labels correct on a sentence) and Hamming accuracy (per-label correct rate).
- **Per-section breakdown** — accuracy split by the section labels in your CSV (e.g., `S1_finding` vs. `S3_representation` vs. `S4_neutral`). Useful for separating the cleanest signal (true bias quotes) from documented limitation zones (meta-commentary).
- **Per-sentence CSV** for your appendix — every sentence with predicted probabilities, predictions, and correctness flags.

## Test set 1 — Trix & Psenka (2003)

**Source:** Trix, F., & Psenka, C. (2003). Exploring the color of glass: Letters of recommendation for female and male medical faculty. *Discourse & Society, 14*(2), 191–220.

**PDF:** https://faculty.usc.edu/wp-content/uploads/sites/13/2019/03/Trix-and-Psenka-2003.pdf

**32 sentences in 3 sections:**
- **Section 1** (n=18) — verbatim biased excerpts from the original recommendation letters (gender marking, grindstone framing, communal framing, doubt raisers, diminutive language)
- **Section 2** (n=9) — authors' analytical commentary about the patterns
- **Section 3** (n=5) — neutral methodological text (false-positive sanity check)

**What this paper tests:** discourse-level bias and stereotyping in academic recommendation letters.

## Test set 2 — Moss-Racusin et al. (2012)

**Source:** Moss-Racusin, C. A., Dovidio, J. F., Brescoll, V. L., Graham, M. J., & Handelsman, J. (2012). Science faculty's subtle gender biases favor male students. *Proceedings of the National Academy of Sciences, 109*(41), 16474–16479.

**24 sentences in 4 sections:**
- **Section 1** (n=7) — experimental findings showing differential treatment of male and female applicants
- **Section 2** (n=5) — discussion of cultural stereotypes about women in science
- **Section 3** (n=5) — systemic representation of women in academic science (pipeline statistics)
- **Section 4** (n=7) — neutral methodological text (false-positive sanity check)

**What this paper tests:** systemic representation patterns and stereotype discussion in a different academic domain (STEM rather than medicine), and a different methodology (experimental rather than discourse analysis). This complements Trix & Psenka.

## CSV format

Three columns, header row required:

```
section,sentence,expected_labels
```

- **section** — short label grouping the sentence (`S1_finding`, `S3_neutral`, etc.). Free-form, but used for the per-section breakdown.
- **sentence** — the sentence text. Quote it if it contains commas.
- **expected_labels** — comma-separated label names, no spaces. Use `none` for sentences expected to receive no flags.

Valid label names match `LABELS` in `classifier/bert_classifier.py`. Currently:

```
gender_sensitive
stereotyping
representation
```

## Adding more test sets

Build a new CSV in the same format and run `python validate.py test_sets/your_paper.csv`. No code changes needed.

Suggested additional papers if you want a third benchmark:
- Madera, Hebl & Martin (2009) — agentic vs. communal language in recommendation letters (similar domain to Trix & Psenka, different corpus)
- Schmader, Whitehead & Wysocki (2007) — gendered language in postdoc letters (replication of Trix & Psenka)
- Caliskan, Bryson & Narayanan (2017) — Science paper documenting bias in word embeddings

## Methodology framing for your thesis

> "We validated the classifier against two external benchmarks of documented gender bias: Trix & Psenka (2003), a discourse-analytic study of recommendation letters in academic medicine, and Moss-Racusin et al. (2012), an experimental study of subtle gender bias in science faculty. Each test set was hand-labeled against our three classifier labels (`gender_sensitive`, `stereotyping`, `representation`) and partitioned into sections corresponding to (a) verbatim biased content, (b) authors' analytical commentary, and (c) neutral methodological text. Per-label F1 and per-section accuracy were computed for each benchmark."

## Caveat to disclose in your write-up

The "authors' analytical commentary" sections (S2 in Trix & Psenka, parts of S2 in Moss-Racusin) describe bias rather than commit it. The classifier may flag these because it operates at the sentence level without modeling authorial stance. Document this as a known limitation: a flag indicates the sentence engages with gender content, not that the author's intent is biased.
