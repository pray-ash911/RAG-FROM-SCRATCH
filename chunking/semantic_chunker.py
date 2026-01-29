import json
import re
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

INPUT_FILE = "data/extracted_text.json"
OUTPUT_FILE = "data/semantic_chunks.json"

SIM_THRESHOLD = 0.75

model = SentenceTransformer("all-MiniLM-L6-v2")


def is_reference_section(sentence):
    return bool(re.match(r"^\s*(references|bibliography|works cited)\s*$",
                          sentence.strip(), re.IGNORECASE))


def group_pages_by_doc(pages):
    docs = {}
    for page in pages:
        docs.setdefault(page["doc_id"], []).append(page)
    return docs


def semantic_chunk_document(doc_pages):
    sentences = []
    sentence_pages = []

    stop = False
    for page in doc_pages:
        page_sents = sent_tokenize(page["text"])

        for s in page_sents:
            if is_reference_section(s):
                stop = True
                break
            sentences.append(s)
            sentence_pages.append(page["page"])

        if stop:
            break

    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current_chunk = [sentences[0]]
    current_pages = {sentence_pages[0]}

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            embeddings[i - 1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if sim < SIM_THRESHOLD:
            chunks.append({
                "text": " ".join(current_chunk),
                "pages": sorted(current_pages)
            })
            current_chunk = [sentences[i]]
            current_pages = {sentence_pages[i]}
        else:
            current_chunk.append(sentences[i])
            current_pages.add(sentence_pages[i])

    chunks.append({
        "text": " ".join(current_chunk),
        "pages": sorted(current_pages)
    })

    return chunks


def create_semantic_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    docs = group_pages_by_doc(pages)
    all_chunks = []

    for doc_id, doc_pages in docs.items():
        doc_pages = sorted(doc_pages, key=lambda x: x["page"])
        pdf_name = doc_pages[0]["pdf_name"]

        sem_chunks = semantic_chunk_document(doc_pages)

        for idx, chunk in enumerate(sem_chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}_sem_{idx}",
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "pages": chunk["pages"],
                "chunk_type": "semantic",
                "text": chunk["text"].strip()
            })

    return all_chunks


if __name__ == "__main__":
    chunks = create_semantic_chunks()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} semantic chunks.")
