"""Shared folding/token helpers for Urdu/Roman Urdu comparisons."""

from __future__ import annotations

import re
import unicodedata

COMPARE_PUNCT = "۔.،,؟?!"
TRAILING_PUNCT = COMPARE_PUNCT
SPACE_RE = re.compile(r"\s+")

FOLD_TRANSLATION = str.maketrans(
    {
        "\u06be": "\u06c1",
        "\u0647": "\u06c1",
        "\u0643": "\u06a9",
        "\u064a": "\u06cc",
        "\u0649": "\u06cc",
        "\u0622": "\u0627",
        "\u0640": None,
        "\u200c": None,
        **{chr(code): None for code in range(0x064B, 0x0660)},
    }
)


def fold_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(FOLD_TRANSLATION).casefold()


def original_tokens(text: str) -> list[str]:
    return SPACE_RE.sub(" ", text.strip()).split()


def compare_tokens(text: str) -> list[str]:
    return [
        token.strip(COMPARE_PUNCT)
        for token in fold_text(text).split()
        if token.strip(COMPARE_PUNCT)
    ]


def phrase_tokens(text: str) -> tuple[str, ...]:
    return tuple(compare_tokens(text))


def normalize_candidate_text(text: str) -> str:
    cleaned = text.replace("،", " ").replace(",", " ")
    cleaned = SPACE_RE.sub(" ", cleaned.strip()).strip()
    return cleaned.rstrip(TRAILING_PUNCT).strip()


def is_digit_token(token: str) -> bool:
    if not token:
        return False
    return all(
        ("0" <= char <= "9")
        or ("\u0660" <= char <= "\u0669")
        or ("\u06f0" <= char <= "\u06f9")
        for char in token
    )


def find_phrase_windows(
    tokens: list[str],
    phrases: set[tuple[str, ...]],
) -> list[tuple[int, int, tuple[str, ...]]]:
    matches: list[tuple[int, int, tuple[str, ...]]] = []
    ordered = sorted(phrases, key=len, reverse=True)
    for index in range(len(tokens)):
        for phrase in ordered:
            width = len(phrase)
            if width and tuple(tokens[index : index + width]) == phrase:
                matches.append((index, index + width, phrase))
                break
    return matches
