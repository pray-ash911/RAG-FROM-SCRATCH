def build_rag_prompt(query: str, retrieved_chunks: list):
    context = "\n\n".join(
        [f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are an expert research assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()
