import re
import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def split_into_sentences(text: str) -> list:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" +", " ", text)

    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)

    text = re.sub(
        r"\([A-Za-z][A-Za-z\-\.' &,]*?\d{4}[a-z]?(?:\s*;\s*[A-Za-z][A-Za-z\-\.' &,]*?\d{4}[a-z]?)*\)",
        "",
        text,
    )

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) < 5:
            continue
        if s.isupper():
            continue
        if not re.search(r"[a-zA-Z]{3,}", s):
            continue
        cleaned.append(s)

    return cleaned
