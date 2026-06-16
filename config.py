import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError("GROQ_API_KEY is missing. Copy .env.example to .env and set your key.")
