# BackEnd/pdf_utils.py
import pdfplumber
from langchain_core.documents import Document

def extract_docs_and_text_from_pdf(file):
    docs = []
    full_text = ""
    for fl in file:
        with pdfplumber.open(fl) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    docs.append(Document(page_content=page_text, metadata={"page": i + 1}))

    return docs, full_text
