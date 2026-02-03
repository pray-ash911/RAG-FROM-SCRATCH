import json
import re
from rank_bm25 import BM25Okapi

INPUT_FILE = "data/all_chunks.json"
OUTPUT_FILE = "data/bm25_corpus.json"


def tokenize(text):
    # basic research-safe tokenizer
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens


def build_bm25_corpus():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenized_corpus = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        tokenized_corpus.append(tokens)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f)

    print(f"BM25 corpus built with {len(tokenized_corpus)} documents.")


if __name__ == "__main__":
    build_bm25_corpus()
