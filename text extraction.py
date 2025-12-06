import fitz
from docx import Document

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


#PyMuPDF
def pdf_extract(path_pdf, output_pdf):
    pdf = fitz.open(path_pdf)
    pdftext = ""

    for page in pdf:
        pdftext += page.get_text()

    pdf.close()

    with open(output_pdf, "w", encoding="utf-8") as f:
        f.write(pdftext)

#python-docx
def docx_extract(path_docx, output_docx):
    doc = Document(path_docx)
    docxtext = ""

    for para in doc.paragraphs:
        docxtext += para.text + "\n"
        
    with open(output_docx, "w", encoding="utf-8") as f:
        f.write(docxtext)

#NLTK
def nltk_module(text_file_path, preprocess):
    with open(text_file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = word_tokenize(text)                            #tokenization
    tokens = [w.lower() for w in tokens]                    #normalization
    tokens = [w for w in tokens if w.isalpha()]             #punctuation removal

    stop_words = set(stopwords.words('english'))            #stop word removal
    tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()                        #lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    print(tokens[:50])                                      #tokenization outputs (limit to 50 first for less overloading)

    with open(preprocess, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))


pdf_extract(r'THESIS\test.pdf', r'THESIS\output_pdf.txt')
docx_extract(r'THESIS\test.docx', r'THESIS\output_docx.txt')

nltk_module(r'THESIS\output_pdf.txt', r'THESIS\preprocess_pdf.txt')
nltk_module(r'THESIS\output_docx.txt', r'THESIS\preprocess_docx.txt')




