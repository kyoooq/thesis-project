import os

from extractor.pdf_extractor import extract_text_from_pdf, split_into_sentences
from extractor.docx_extractor import extract_text_from_docx


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported file extension: {ext}")


def extract_sentences(file_path: str) -> list:
    return split_into_sentences(extract_text(file_path))


def extract_text_and_sentences(file_path: str) -> tuple[str, list]:
    full_text = extract_text(file_path)
    sentences = split_into_sentences(full_text)
    return full_text, sentences
