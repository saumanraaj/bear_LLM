from pathlib import Path
import fitz  # PyMuPDF
from llama_index.core.schema import Document

def load_clean_text_from_pdf(filepath, max_pages=30):
    doc = fitz.open(filepath)
    texts = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if len(text.strip()) > 100:  # skip blank or image-heavy pages
            texts.append(text)
    joined_text = "\n\n".join(texts)
    return Document(text=joined_text, metadata={"filename": Path(filepath).name})
