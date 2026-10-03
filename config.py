import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}.")


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _env_non_negative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be zero or greater.")
    return value


LLM_PROVIDER = "openai"

# Hybrid deterministic order parser with one-call OpenAI fallback
HYBRID_INTENT_ENABLED = _env_bool("HYBRID_INTENT_ENABLED", True)
HYBRID_LLM_FALLBACK_ENABLED = _env_bool("HYBRID_LLM_FALLBACK_ENABLED", True)
HYBRID_FUZZY_MATCH_ENABLED = _env_bool("HYBRID_FUZZY_MATCH_ENABLED", True)
HYBRID_FUZZY_MIN_SCORE = _env_non_negative_float(
    "HYBRID_FUZZY_MIN_SCORE", 90.0
)
HYBRID_FUZZY_MIN_GAP = _env_non_negative_float("HYBRID_FUZZY_MIN_GAP", 10.0)
HYBRID_SIZE_LLM_AFTER_MISSES = _env_positive_int(
    "HYBRID_SIZE_LLM_AFTER_MISSES", 1
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
OPENAI_REASONING_EFFORT = (
    os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
)
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_COMPLETION_TOKENS = _env_positive_int(
    "OPENAI_MAX_COMPLETION_TOKENS", 1024
)
OPENAI_FALLBACK_MAX_COMPLETION_TOKENS = _env_positive_int(
    "OPENAI_FALLBACK_MAX_COMPLETION_TOKENS", 512
)

_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
if OPENAI_REASONING_EFFORT not in _VALID_REASONING_EFFORTS:
    raise ValueError(
        f"OPENAI_REASONING_EFFORT must be one of "
        f"{sorted(_VALID_REASONING_EFFORTS)}, got {OPENAI_REASONING_EFFORT!r}."
    )
if OPENAI_TIMEOUT_SECONDS <= 0:
    raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero.")
if HYBRID_FUZZY_MIN_SCORE > 100:
    raise ValueError("HYBRID_FUZZY_MIN_SCORE must not exceed 100.")
if HYBRID_FUZZY_MIN_GAP > 100:
    raise ValueError("HYBRID_FUZZY_MIN_GAP must not exceed 100.")


def _placeholder_or_missing(value: str, placeholder: str) -> bool:
    return not value or value == placeholder


if _placeholder_or_missing(OPENAI_API_KEY, "your_openai_api_key_here"):
    raise ValueError("OPENAI_API_KEY is required when OpenAI is configured.")

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
    "mujhe pizza chahiye, chicken karahi, biryani, naan, Kasur, "
    "aadhi karahi, poori karahi, aadhi, poori, chota, bara, family pack, "
    "delivery address bata dein.",
).strip()
WHISPER_ROMAN_PROMPT = os.getenv(
    "WHISPER_ROMAN_PROMPT",
    "قصور کچن۔ چکن کڑاہی، مٹن کڑاہی، چکن بریانی، بیف بریانی، "
    "چکن تکہ پیزا، مارگریٹا پیزا، سادہ نان، گارلک نان، لہسن والا نان، "
    "ڈیڑھ لیٹر کوک، پودینے والی مارگریٹا۔ "
    "آدھی کڑاہی، پوری کڑاہی، آدھی، پوری، چھوٹا، بڑا، فیملی پیک۔ "
    "Chicken karahi, mutton karahi, biryani, garlic naan, pizza.",
).strip()
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").strip().lower()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1").strip()
_stt_lang = os.getenv("ELEVENLABS_STT_LANGUAGE", "").strip().lower()
ELEVENLABS_STT_LANGUAGE = None if _stt_lang in ("", "none", "auto") else _stt_lang
ELEVENLABS_STT_TIMEOUT_SECONDS = float(
    os.getenv("ELEVENLABS_STT_TIMEOUT_SECONDS", "30")
)
SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = float(os.getenv("STT_RECORD_SECONDS", "5"))

# Text-to-speech (Piper HTTP) — Phase 4
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "piper_http").strip().lower()
PIPER_TTS_URL = os.getenv("PIPER_TTS_URL", "http://127.0.0.1:5000").strip()
PIPER_TTS_VOICE = os.getenv(
    "PIPER_TTS_VOICE", "ur_PK-aegis_female-medium"
).strip()
PIPER_TTS_TIMEOUT_SECONDS = float(
    os.getenv("PIPER_TTS_TIMEOUT_SECONDS", "30")
)
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "16000"))

# Agent / push-to-talk — Phase 5
PTT_MAX_SECONDS = float(os.getenv("PTT_MAX_SECONDS", "30"))
PTT_MIN_SECONDS = float(os.getenv("PTT_MIN_SECONDS", "0.5"))
AGENT_GREETING = os.getenv(
    "AGENT_GREETING", "السلام علیکم، آپ کیا آرڈر کرنا چاہیں گے؟"
).strip()
AGENT_LOG_LATENCY = os.getenv("AGENT_LOG_LATENCY", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
