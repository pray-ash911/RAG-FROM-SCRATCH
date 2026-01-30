from llm.gemini_client import get_gemini_model
from llm.prompt_builder import build_rag_prompt
import os

model = get_gemini_model(os.getenv("GEMINI_API_KEY"))

def generate_answer(query, retrieved_chunks):
    prompt = build_rag_prompt(query, retrieved_chunks)
    response = model.generate_content(prompt)
    return response.text
