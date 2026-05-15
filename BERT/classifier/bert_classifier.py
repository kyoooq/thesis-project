import os
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from classifier.highlighter import get_trigger_words


LABELS = ["gender_sensitive", "stereotyping", "representation"]

THRESHOLDS = {
    "gender_sensitive": 0.65,
    "stereotyping":     0.65,
    "representation":   0.65,
}


DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "BERT-base-uncased"


_ENV_PATH = os.environ.get("BERT_MODEL_PATH")
if _ENV_PATH:
    MODEL_PATH = _ENV_PATH
else:
    MODEL_PATH = str(Path(__file__).resolve().parent.parent / "model")


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


def predict_batch(sentences: list[str], thresholds: dict | None = None,
                  batch_size: int = 16) -> list[dict]:

    if thresholds is None:
        thresholds = THRESHOLDS

    all_results = []

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]

        # token whole chunk
        encoding = tokenizer(
            chunk,
            max_length     = 128,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            output = model(
                input_ids         = input_ids,
                attention_mask    = attention_mask,
                output_attentions = True,
            )

            probs = torch.sigmoid(output.logits).cpu().numpy()

            attentions = torch.stack(output.attentions)
            avg_attn   = attentions.mean(dim=0).mean(dim=1)
            attn_per_token = avg_attn.mean(dim=1).cpu().numpy()


        for i in range(len(chunk)):
            tokens      = tokenizer.convert_ids_to_tokens(input_ids[i])
            attn_scores = attn_per_token[i]
            triggers    = get_trigger_words(tokens, attn_scores, top_k=3)

            sentence_probs = probs[i]
            sentence_result = {}
            for j, label in enumerate(LABELS):
                prob      = float(sentence_probs[j])
                threshold = thresholds.get(label, 0.7)
                predicted = prob >= threshold
                sentence_result[label] = {
                    "probability": round(prob, 4),
                    "predicted":   bool(predicted),
                    "triggers":    triggers if predicted else [],
                }
            all_results.append(sentence_result)

    return all_results