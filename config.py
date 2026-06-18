import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

# Optional fallback when OpenRouter free-tier limits are hit
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Copy .env.example to .env and set your key."
    )
