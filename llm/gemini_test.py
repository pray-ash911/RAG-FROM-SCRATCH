import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard local imports
from gemini_client import get_gemini_model
from prompt_builder import build_rag_prompt

# Import from the retrieval folder
from retrieval.hybrid_retriever import retrieve

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 1. Setup - use your API key correctly
model = get_gemini_model(os.getenv("GEMINI_API_KEY"))


def run_full_rag(query):
    # Retrieve
    chunks = retrieve(query, top_k=5)
    print(f"retrived{chunks}")

    # Build Prompt
    final_prompt = build_rag_prompt(query, chunks)

    # Generate
    response = model.generate_content(final_prompt)
    return response.text


if __name__ == "__main__":
    print("--- RAG SYSTEM READY ---")
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break

        print("Thinking...")
        answer = run_full_rag(user_query)
        print(f"\nGEMINI: {answer}")