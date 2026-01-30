import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai

def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)

    # Wrap your dictionary in the actual GenerationConfig class
    config = genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=1024,
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Use the stable ID
        generation_config=config,      # Now the types match!
    )
    return model
