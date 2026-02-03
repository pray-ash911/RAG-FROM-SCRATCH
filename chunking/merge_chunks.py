import json

FIXED_FILE = "data/fixed_chunks.json"
SEMANTIC_FILE = "data/semantic_chunks.json"
OUTPUT_FILE = "data/all_chunks.json"


def normalize_fixed_chunks(fixed_chunks):
    normalized = []
    for c in fixed_chunks:
        normalized.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "pdf_name": c["pdf_name"],
            "pages": [c["page"]],
            "chunk_type": "fixed",
            "text": c["text"]
        })
    return normalized


def load_and_merge():
    with open(FIXED_FILE, "r", encoding="utf-8") as f:
        fixed_chunks = json.load(f)

    with open(SEMANTIC_FILE, "r", encoding="utf-8") as f:
        semantic_chunks = json.load(f)

    fixed_chunks = normalize_fixed_chunks(fixed_chunks)

    all_chunks = fixed_chunks + semantic_chunks
    return all_chunks


if __name__ == "__main__":
    all_chunks = load_and_merge()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total merged chunks: {len(all_chunks)}")
