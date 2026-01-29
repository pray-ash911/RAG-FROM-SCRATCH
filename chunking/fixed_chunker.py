import json
import os
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


INPUT_FILE = "data/extracted_text.json"
OUTPUT_FILE = "data/fixed_chunks.json"

CHUNK_SIZE = 400
OVERLAP = 80

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer


def chunk_page_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_len = len(tokens)

        if current_len + sent_len > CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))

            # overlap handling
            overlap_tokens = []
            for s in reversed(current_chunk):
                t = tokenizer.encode(s, add_special_tokens=False)
                overlap_tokens.extend(t)
                if len(overlap_tokens) >= OVERLAP:
                    break

            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_fixed_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_page_text(page["text"])

        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id": f"{page['doc_id']}_p{page['page']}_fixed_{idx}",
                "doc_id": page["doc_id"],
                "pdf_name": page["pdf_name"],
                "page": page["page"],
                "chunk_type": "fixed",
                "text": chunk_text.strip()
            })
            chunk_counter += 1

    return all_chunks


if __name__ == "__main__":
    chunks = create_fixed_chunks()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} fixed-size chunks.")
