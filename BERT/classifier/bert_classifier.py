import os
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from classifier.highlighter import get_trigger_words


LABELS = ["gender_sensitive", "stereotyping", "representation"]

# ── Per-label thresholds ──────────────────────────────────────────────────────
# Tune each label independently based on observed false-positive/negative rates.
# Starting points from testing:
#   gender_sensitive: keyword-driven, reliable → can afford a lower threshold
#   stereotyping:     prone to false positives on contrastive prose → stricter
#   representation:   prone to false positives on citation-heavy text → stricter
THRESHOLDS = {
    "gender_sensitive": 0.65,
    "stereotyping":     0.65,
    "representation":   0.65,
}

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "BERT-base-uncased"


# ── Resolve model path relative to the project root ──────────────────────────
# Layout assumed:
#   <project_root>/
#       classifier/bert_classifier.py   ← this file
#       model/                          ← where the fine-tuned weights live
#
# Override with the BERT_MODEL_PATH environment variable if you need to.
_ENV_PATH = os.environ.get("BERT_MODEL_PATH")
if _ENV_PATH:
    MODEL_PATH = _ENV_PATH
else:
    MODEL_PATH = str(Path(__file__).resolve().parent.parent / "model")


# ── Load once at import time ──────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    local_files_only=True,
)
tokenizer = BertTokenizerFast.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
model.to(DEVICE)
model.eval()


def predict(sentence: str, thresholds: dict | None = None) -> dict:
    """
    Run BERT classification on one sentence.

    Args:
        sentence:   Raw input text.
        thresholds: Optional per-label threshold override, e.g.
                    {"gender_sensitive": 0.7, "stereotyping": 0.9, "representation": 0.8}
                    If None, uses the module-level THRESHOLDS dict.

    Returns:
        {label: {probability, predicted, triggers}, ...}
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    encoding = tokenizer(
        sentence,
        max_length     = 128,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    tokens         = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        output = model(
            input_ids         = input_ids,
            attention_mask    = attention_mask,
            output_attentions = True,
        )
        probs = torch.sigmoid(output.logits).squeeze(0).cpu().numpy()

        attentions  = torch.stack(output.attentions)
        avg_attn    = attentions.mean(dim=0).mean(dim=1)
        attn_scores = avg_attn.squeeze(0).mean(dim=0).cpu().numpy()

    triggers = get_trigger_words(tokens, attn_scores, top_k=3)

    results = {}
    for i, label in enumerate(LABELS):
        prob = float(probs[i])
        threshold = thresholds.get(label, 0.7)   # fallback if label missing from dict
        predicted = prob >= threshold
        results[label] = {
            "probability": round(prob, 4),
            "predicted":   bool(predicted),
            "triggers":    triggers if predicted else [],
        }
    return results