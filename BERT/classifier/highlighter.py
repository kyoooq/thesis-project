IGNORE_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]",
    ".", ",", "!", "?", "'", '"',
    "''", "``", "--", "-", ":",
    ";", "(", ")", "a", "the",
    "is", "are", "was", "were",
    "it", "its", "be", "been",
    "an", "of", "in", "to",
    "and", "or", "not", "for",
    "on", "at", "by", "with",
}


def get_trigger_words(tokens: list, attn_scores: list, top_k: int = 3) -> list:
    """
    Reconstructs subword tokens into full words, scores them by attention,
    and returns the top_k highest-attention words (excluding noise tokens).
    """
    words       = []
    word_scores = []
    curr_word   = ""
    curr_score  = 0

    for token, score in zip(tokens, attn_scores):
        if token in IGNORE_TOKENS:
            continue
        if token.startswith("##"):
            curr_word  += token[2:]
            curr_score  = max(curr_score, float(score))
        else:
            if curr_word:
                words.append(curr_word)
                word_scores.append(curr_score)
            curr_word  = token
            curr_score = float(score)

    if curr_word:
        words.append(curr_word)
        word_scores.append(curr_score)

    ranked = sorted(zip(words, word_scores), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:top_k]]
