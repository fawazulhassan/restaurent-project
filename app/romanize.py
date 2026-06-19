"""Convert Urdu/Hindi script transcripts to Roman Urdu (Latin letters only)."""

import re

from openai import OpenAI

import config

_NON_LATIN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0900-\u097F]"
)

_openrouter = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Kasur Kitchen Voice Agent",
    },
)


def needs_latinize(text: str) -> bool:
    return bool(_NON_LATIN_RE.search(text))


def to_roman_latin(text: str) -> str:
    """Return Roman Urdu in Latin script. Pass-through if already Latin."""
    text = text.strip()
    if not text or not needs_latinize(text):
        return text

    response = _openrouter.chat.completions.create(
        model=config.OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Convert the following to Roman Urdu using ONLY Latin letters (a-z). "
                    "Mixed English is OK (pizza, chicken, karahi, Total, Rs). "
                    "Examples: 'mujhe ek chicken tikka pizza chahiye small size', "
                    "'delivery address bata dein'. "
                    "Output ONLY the converted text — no explanation.\n\n"
                    f"{text}"
                ),
            }
        ],
        temperature=0,
    )
    out = (response.choices[0].message.content or "").strip()
    if out and not needs_latinize(out):
        return out
    return text
