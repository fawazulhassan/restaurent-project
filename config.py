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

# Speech-to-text (faster-whisper) — Phase 3
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# Default "ur" avoids Hindi (Devanagari) mis-detection for Pakistani Roman Urdu speech
_lang = os.getenv("WHISPER_LANGUAGE", "ur").strip().lower()
WHISPER_LANGUAGE = None if _lang in ("", "none", "auto") else _lang
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    "mujhe pizza chahiye, chicken karahi, biryani, naan, Kasur, delivery address bata dein.",
).strip()
WHISPER_ROMAN_PROMPT = os.getenv(
    "WHISPER_ROMAN_PROMPT",
    "Assalam o alaikum. mujhe ek chicken tikka pizza chahiye small size. "
    "mint margarita regular. delivery address bata dein. confirm. Kot Kasur.",
).strip()
SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = float(os.getenv("STT_RECORD_SECONDS", "5"))

# Text-to-speech (MMS-TTS) — Phase 4
TTS_MODEL = os.getenv("TTS_MODEL", "facebook/mms-tts-urd-script_latin").strip()
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu").strip()
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "16000"))

# Agent / push-to-talk — Phase 5
PTT_MAX_SECONDS = float(os.getenv("PTT_MAX_SECONDS", "30"))
PTT_MIN_SECONDS = float(os.getenv("PTT_MIN_SECONDS", "0.5"))
AGENT_GREETING = os.getenv(
    "AGENT_GREETING", "Assalam o alaikum, aap ka kya order hai?"
).strip()
AGENT_LOG_LATENCY = os.getenv("AGENT_LOG_LATENCY", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
