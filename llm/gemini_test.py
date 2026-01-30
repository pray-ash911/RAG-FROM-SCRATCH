from gemini_client import get_gemini_model
import os

model = get_gemini_model(os.getenv("GEMINI_API_KEY"))

response = model.generate_content(
    "Explain gan in one paragraph for research papers"
)

print(response.text)
