import pdfplumber
import os
import json

PDF_DIR = "data/pdfs"
OUTPUT_FILE = "data/extracted_text.json"


def extract_pdfs():
    extracted_pages = []
    doc_counter = 1

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        doc_id = f"doc_{doc_counter}"

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    extracted_pages.append({
                        "doc_id": doc_id,
                        "pdf_name": filename,
                        "page": page_num,
                        "text": text.strip()
                    })

        doc_counter += 1

    return extracted_pages


if __name__ == "__main__":
    pages = extract_pdfs()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(pages)} pages from PDFs.")
