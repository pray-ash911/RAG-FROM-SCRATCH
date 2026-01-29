import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "data/all_chunks.json"
EMBED_FILE = "data/dense_embeddings.npy"
ID_MAP_FILE = "data/faiss_id_to_chunk.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading chunks...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print("Loading sentence-transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating dense embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # VERY IMPORTANT
    )

    embeddings = embeddings.astype("float32")

    print("Saving embeddings...")
    np.save(EMBED_FILE, embeddings)

    id_map = {i: chunks[i]["chunk_id"] for i in range(len(chunks))}
    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    print(f"Done. Embedded {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
