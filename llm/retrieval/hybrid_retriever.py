import json
import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------
TOP_K = 5
FAISS_CANDIDATES = 20
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNKS_FILE = "data/all_chunks.json"
BM25_FILE = "data/bm25_corpus.json"
FAISS_INDEX_FILE = "data/faiss_hnsw.index"
ID_MAP_FILE = "data/faiss_id_to_chunk.json"


# ---------- UTIL ----------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ---------- LOAD ----------
print("Loading resources...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(BM25_FILE, "r", encoding="utf-8") as f:
    bm25_corpus = json.load(f)

with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    faiss_id_to_chunk = json.load(f)

chunk_lookup = {c["chunk_id"]: c for c in chunks}

bm25 = BM25Okapi(bm25_corpus)

model = SentenceTransformer(MODEL_NAME)

index = faiss.read_index(FAISS_INDEX_FILE)


# ---------- RETRIEVER ----------
def retrieve(query, top_k=TOP_K):
    # ---- Dense search ----
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    dense_scores, dense_ids = index.search(q_vec, FAISS_CANDIDATES)

    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    # Normalize dense scores to [0,1]
    if dense_scores.max() != 0:
        dense_scores = dense_scores / dense_scores.max()

    # ---- Sparse search ----
    query_tokens = tokenize(query)
    sparse_scores_all = bm25.get_scores(query_tokens)

    # ---- Fusion ----
    fused = []

    for faiss_id, dense_score in zip(dense_ids, dense_scores):
        chunk_id = faiss_id_to_chunk[str(faiss_id)]
        sparse_score = sparse_scores_all[faiss_id]

        # Normalize sparse score
        if sparse_scores_all.max() != 0:
            sparse_score = sparse_score / sparse_scores_all.max()

        final_score = (
            DENSE_WEIGHT * dense_score +
            SPARSE_WEIGHT * sparse_score
        )

        fused.append((final_score, chunk_id))

    # ---- Rank ----
    fused.sort(reverse=True, key=lambda x: x[0])
    top_chunks = []

    for score, chunk_id in fused[:top_k]:
        chunk = chunk_lookup[chunk_id]
        chunk["score"] = round(float(score), 4)
        top_chunks.append(chunk)

    return top_chunks


# ---------- TEST ----------
if __name__ == "__main__":
    q = "Describe the experimental setup and evaluation methodology"
    results = retrieve(q)

    for r in results:
        print("=" * 60)
        print(f"Score: {r['score']}")
        print(f"Doc: {r['pdf_name']} | Pages: {r['pages']}")
        print(r["text"][:500])
