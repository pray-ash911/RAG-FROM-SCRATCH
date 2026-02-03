import faiss
import numpy as np
import json
import os

EMBED_FILE = "data/dense_embeddings.npy"
ID_MAP_FILE = "data/faiss_id_to_chunk.json"
INDEX_FILE = "data/faiss_hnsw.index"


def build_faiss_index():
    print("Loading embeddings...")
    embeddings = np.load(EMBED_FILE).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding count: {embeddings.shape[0]}, dim: {dim}")

    print("Creating HNSW index...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50

    print("Adding vectors to index...")
    index.add(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_FILE)

    print(f"FAISS index saved to {INDEX_FILE}")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    build_faiss_index()
