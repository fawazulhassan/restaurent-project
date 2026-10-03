"""Fast deterministic intent parsing for routine restaurant order turns."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import config
from app.dialog_state import DialogStage, DialogState
from app.order import Order


class IntentType(str, Enum):
    LIST_MENU = "list_menu"
    ADD_ITEM = "add_item"
    REMOVE_ITEM = "remove_item"
    SET_QUANTITY = "set_quantity"
    CHANGE_SIZE = "change_size"
    SET_ADDRESS = "set_address"
    SET_INSTRUCTIONS = "set_instructions"
    SHOW_ORDER = "show_order"
    CONFIRM_ORDER = "confirm_order"
    CANCEL = "cancel"
    UNKNOWN_ITEM = "unknown_item"
    AMBIGUOUS = "ambiguous"


class ParsedIntent(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    item_id: str | None = None
    size_id: str | None = None
    quantity: int | None = Field(default=None, ge=0)
    address: str | None = None
    instructions: str | None = None
    matched_text: str | None = None
    clarification_reason: str | None = None
    available: bool | None = None
    clear_address: bool = False


@dataclass(frozen=True)
class ItemMatch:
    item: dict[str, Any]
    alias: str
    start: int
    end: int
    score: float = 100.0


_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_CHAR_TRANSLATION = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ہ"})
_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06ff.]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_customer_text(text: str) -> tuple[str, str]:
    """Return (normalized, original); addresses must always use original."""
    original = text
    normalized = unicodedata.normalize("NFKC", text).translate(_DIGIT_TRANSLATION)
    normalized = normalized.translate(_CHAR_TRANSLATION).casefold()
    normalized = _PUNCT_RE.sub(" ", normalized).replace(".", " ")
    return _SPACE_RE.sub(" ", normalized).strip(), original


def _normalize_alias(text: str) -> str:
    return normalize_customer_text(text)[0]


_NUMBER_WORDS = {
    "zero": 0,
    "sifar": 0,
    "صفر": 0,
    "one": 1,
    "aik": 1,
    "ek": 1,
    "ایک": 1,
    "two": 2,
    "do": 2,
    "دو": 2,
    "three": 3,
    "teen": 3,
    "تین": 3,
    "four": 4,
    "char": 4,
    "چار": 4,
    "five": 5,
    "panch": 5,
    "پانچ": 5,
    "six": 6,
    "chay": 6,
    "چھ": 6,
    "seven": 7,
    "saat": 7,
    "سات": 7,
    "eight": 8,
    "aath": 8,
    "آٹھ": 8,
    "nine": 9,
    "nau": 9,
    "نو": 9,
    "ten": 10,
    "das": 10,
    "دس": 10,
}

_SIZE_ALIASES = {
    "half": (
        "half",
        "aadhi",
        "adhi",
        "aadi",
        "adi",
        "aadha",
        "adha",
        "آدھی",
        "آدھا",
        "نصف",
    ),
    "full": ("full", "poori", "puri", "pura", "poora", "پوری", "پورا"),
    "small": ("small", "chota", "choti", "چھوٹا", "چھوٹی", "سمال"),
    "large": ("large", "bara", "bari", "بڑا", "بڑی", "لارج"),
    "single": (
        "single",
        "plate",
        "ek plate",
        "ایک پلیٹ",
        "پلیٹ",
        "ایک لیٹ",
        "ایک بلیٹ",
        "ایک پلیڈ",
        "سنگل",
    ),
    "family": (
        "family",
        "family pack",
        "family wali",
        "فیملی",
        "فیملی پیک",
    ),
    "regular": ("regular", "normal", "عام سائز", "ریگولر"),
}

# Size-only folding: drop heh letters and flatten alef-with-madda.
# Must not be applied to addresses or general customer text.
_SIZE_FOLD = str.maketrans(
    {"\u06BE": None, "\u06C1": None, "\u0622": "\u0627"}
)


def _fold_size_text(text: str) -> str:
    return text.translate(_SIZE_FOLD)

_REMOVE_WORDS = (
    "remove",
    "hata",
    "hata do",
    "nikal do",
    "ہٹا",
    "ہٹا دیں",
    "نکال دیں",
)
_QUANTITY_WORDS = ("quantity", "qty", "tadaad", "تعداد", "make it", "اس کو")
_CHANGE_WORDS = ("change", "badal", "kar do", "کر دو", "کر دیں", "بنا دو")
_ADD_WORDS = (
    "add",
    "shamil",
    "chahiye",
    "order",
    "چاہیے",
    "شامل",
    "آرڈر",
    "لینا",
)
_CONFIRM_PHRASES = (
    "yes confirm",
    "haan confirm",
    "han confirm",
    "confirm kar do",
    "order confirm",
    "ji theek hai",
    "haan theek hai",
    "ہاں کنفرم",
    "کنفرم کر دیں",
    "جی ٹھیک ہے",
    "ہاں ٹھیک ہے",
    "yes thats all",
    "yes that s all",
    "thats all",
    "that s all",
    "bas itna",
    "بس اتنا",
)
_YES_ONLY = {"yes", "haan", "han", "ji", "ہاں", "جی", "theek", "ٹھیک"}
_ADDRESS_PREFIXES = (
    r"\bmera address(?: is)?\b",
    r"\bmy address(?: is)?\b",
    r"\bdelivery address(?: is)?\b",
    r"\bmera pata(?: hai| is)?\b",
    r"میرا پتہ(?: ہے)?",
    r"ڈیلیوری کا پتہ(?: ہے)?",
)
_ADDRESS_DENIAL_PHRASES = (
    "not my address",
    "this is not my address",
    "address nahi",
    "pata nahi",
    "میرا پتہ یہ نہیں",
    "یہ میرا پتہ نہیں",
    "پتہ یہ نہیں",
    "ایڈریس نہیں",
)
_ORDER_REQUEST_MARKERS = (
    "add",
    "add kar",
    "order",
    "item",
    "chahiye",
    "shamil",
    "ایڈ",
    "آرڈر",
    "آڈر",
    "اوٹر",
    "آئٹم",
    "چاہیے",
    "شامل",
)
_INSTRUCTION_WORDS = (
    "extra cheese",
    "no cheese",
    "extra sauce",
    "extra spicy",
    "less spicy",
    "no salad",
    "instruction",
    "instructions",
    "note",
    "تیز مصالحہ",
    "کم مصالحہ",
    "بغیر سلاد",
    "ہدایات",
)
_MENU_WORDS = ("menu", "مینو", "منیو", "مینی", "نانیوں", "منیومی")
_MENU_QUESTION_WORDS = (
    "what",
    "kya",
    "bata",
    "available",
    "کیا",
    "بتا",
    "دستیاب",
    "جید",
    "چیز",
)
_SHOW_ORDER_PHRASES = (
    "show order",
    "my order",
    "order summary",
    "mera order",
    "میرا آرڈر",
    "آرڈر بتائیں",
    "total",
    "کل کتنے",
)
_UNKNOWN_FOOD_WORDS = ("burger", "zinger", "shawarma", "sandwich", "roll", "برگر", "شوارما")


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_normalize_alias(phrase) in text for phrase in phrases)


@lru_cache(maxsize=1)
def load_intent_menu() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "data" / "menu.json"
    with open(path, encoding="utf-8") as menu_file:
        return json.load(menu_file)


def _all_menu_items(menu_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for category in menu_data.get("categories", [])
        for item in category.get("items", [])
    ]


def _item_aliases(item: dict[str, Any]) -> list[str]:
    raw = [
        item.get("id", ""),
        item.get("name", ""),
        item.get("name_urdu", ""),
        item.get("spoken_name_urdu", ""),
        *item.get("aliases", []),
    ]
    aliases = {_normalize_alias(alias) for alias in raw if alias}
    return sorted((alias for alias in aliases if alias), key=len, reverse=True)


def compact_menu_for_fallback(
    menu_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the exact compact menu fields used by the ambiguity resolver."""
    data = menu_data or load_intent_menu()
    compact: list[dict[str, Any]] = []
    for item in _all_menu_items(data):
        if not item.get("available", True):
            continue
        compact.append(
            {
                "id": item["id"],
                "size_ids": [size["id"] for size in item.get("sizes", [])],
                "aliases": _item_aliases(item),
            }
        )
    return compact


def _boundary_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.UNICODE)


def _exact_item_matches(
    normalized: str, menu_data: dict[str, Any]
) -> list[ItemMatch]:
    best_by_id: dict[str, ItemMatch] = {}
    for item in _all_menu_items(menu_data):
        for alias in _item_aliases(item):
            match = _boundary_pattern(alias).search(normalized)
            if match:
                candidate = ItemMatch(item, alias, match.start(), match.end())
                previous = best_by_id.get(item["id"])
                if previous is None or len(alias) > len(previous.alias):
                    best_by_id[item["id"]] = candidate
                break
    return sorted(best_by_id.values(), key=lambda match: len(match.alias), reverse=True)


def _fuzzy_item_match(
    normalized: str, menu_data: dict[str, Any]
) -> ItemMatch | None:
    if not config.HYBRID_FUZZY_MATCH_ENABLED or not normalized:
        return None
    words = normalized.split()
    best_by_id: dict[str, tuple[float, str]] = {}
    for item in _all_menu_items(menu_data):
        for alias in _item_aliases(item):
            alias_words = alias.split()
            width = len(alias_words)
            if not width or width > len(words):
                continue
            for start in range(len(words) - width + 1):
                phrase = " ".join(words[start : start + width])
                score = float(fuzz.ratio(phrase, alias))
                previous = best_by_id.get(item["id"])
                if previous is None or score > previous[0]:
                    best_by_id[item["id"]] = (score, phrase)
    ranked = sorted(best_by_id.items(), key=lambda entry: entry[1][0], reverse=True)
    if not ranked:
        return None
    best_id, (best_score, phrase) = ranked[0]
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
    if best_score < config.HYBRID_FUZZY_MIN_SCORE:
        return None
    if best_score - second_score < config.HYBRID_FUZZY_MIN_GAP:
        return None
    item = next(item for item in _all_menu_items(menu_data) if item["id"] == best_id)
    start = normalized.find(phrase)
    return ItemMatch(item, phrase, start, start + len(phrase), best_score)


def _parse_size(normalized: str, item: dict[str, Any] | None) -> str | None:
    valid_sizes = {
        size["id"] for size in (item or {}).get("sizes", [])
    } if item else set(_SIZE_ALIASES)
    folded_text = _fold_size_text(normalized)
    for size_id, aliases in _SIZE_ALIASES.items():
        if size_id not in valid_sizes:
            continue
        if any(
            _boundary_pattern(_fold_size_text(_normalize_alias(alias))).search(
                folded_text
            )
            for alias in aliases
        ):
            return size_id
    return None


def _parse_quantity(normalized: str, item_match: ItemMatch | None) -> int | None:
    text = normalized
    if item_match is not None and item_match.start >= 0:
        text = f"{text[:item_match.start]} {text[item_match.end:]}"
    tokens = text.split()
    for index, token in enumerate(tokens):
        if token.isdigit():
            value = int(token)
            if 0 <= value <= 99:
                return value
        if (
            token == "do"
            and index > 0
            and tokens[index - 1] in {"kar", "hata", "nikal", "confirm"}
        ):
            continue
        if token in _NUMBER_WORDS:
            return _NUMBER_WORDS[token]
    return None


def _extract_prefixed_address(original: str) -> str | None:
    for pattern in _ADDRESS_PREFIXES:
        match = re.search(pattern, original, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = original[match.end() :].strip(" ,.-")
        candidate = re.sub(r"\s+(?:hai|ہے)\s*[.؟?]*$", "", candidate, flags=re.IGNORECASE)
        return candidate.strip(" ,.-") or None
    residence = re.search(r"(?:میں|main)\s+(.+?)\s+(?:رہتا ہوں|رہتی ہوں|rehta hun|rehti hun)\s*$", original, re.IGNORECASE)
    if residence:
        return residence.group(1).strip(" ,.-") or None
    return None


def extract_address_from_original(original: str, *, allow_entire: bool) -> str | None:
    """Extract an address only from the untouched original transcription."""
    normalized_original, _ = normalize_customer_text(original)
    if _contains_phrase(normalized_original, _ADDRESS_DENIAL_PHRASES):
        return None
    candidate = _extract_prefixed_address(original)
    if candidate is None and allow_entire:
        candidate = original.strip(" ,.-")
        candidate = re.sub(r"\s+(?:hai|ہے)\s*[.؟?]*$", "", candidate, flags=re.IGNORECASE)
    if not candidate or len(candidate.strip()) < 4:
        return None
    normalized, _ = normalize_customer_text(candidate)
    if normalized in _YES_ONLY or _contains_phrase(normalized, _CONFIRM_PHRASES):
        return None
    return candidate.strip()


def _extract_instructions(original: str) -> str:
    normalized, _ = normalize_customer_text(original)
    matched = [
        phrase
        for phrase in _INSTRUCTION_WORDS
        if _normalize_alias(phrase) in normalized
        and _normalize_alias(phrase) not in {"instruction", "instructions", "note", "ہدایات"}
    ]
    if matched:
        return ", ".join(dict.fromkeys(matched))
    candidate = re.sub(
        r"^(?:note|instructions?|ہدایات)\s*[:,-]?\s*",
        "",
        original.strip(),
        flags=re.IGNORECASE,
    )
    return candidate.strip()


def parse_customer_intent(
    text: str,
    state: DialogState,
    order: Order,
    menu_data: dict[str, Any] | None = None,
) -> ParsedIntent:
    normalized, original = normalize_customer_text(text)
    data = menu_data or load_intent_menu()
    exact_matches = _exact_item_matches(normalized, data)
    address_denied = _contains_phrase(normalized, _ADDRESS_DENIAL_PHRASES)

    # Resolve a size-only response against the unfinished item from the
    # immediately preceding "which size?" question.
    if state.pending_item_id and not exact_matches:
        pending_item = next(
            (
                item
                for item in _all_menu_items(data)
                if item["id"] == state.pending_item_id
            ),
            None,
        )
        pending_size = _parse_size(normalized, pending_item)
        if pending_size is not None:
            return ParsedIntent(
                intent=IntentType.ADD_ITEM,
                confidence=1,
                item_id=state.pending_item_id,
                size_id=pending_size,
                quantity=state.pending_quantity,
                available=(pending_item or {}).get("available", True),
                instructions=state.pending_instructions,
            )

        # First short miss: cheap reprompt. Later misses: hand to LLM.
        if len(normalized.split()) <= 4:
            if state.size_attempts < config.HYBRID_SIZE_LLM_AFTER_MISSES:
                return ParsedIntent(
                    intent=IntentType.ADD_ITEM,
                    confidence=0.4,
                    item_id=state.pending_item_id,
                    quantity=state.pending_quantity,
                    available=(pending_item or {}).get("available", True),
                    instructions=state.pending_instructions,
                    clarification_reason="missing_size",
                )
            return ParsedIntent(
                intent=IntentType.AMBIGUOUS,
                confidence=0.2,
                clarification_reason="missing_size",
            )

    if _contains_phrase(normalized, _REMOVE_WORDS):
        if len(exact_matches) == 1:
            match = exact_matches[0]
            return ParsedIntent(
                intent=IntentType.REMOVE_ITEM,
                confidence=1,
                item_id=match.item["id"],
                size_id=_parse_size(normalized, match.item),
                matched_text=match.alias,
                available=match.item.get("available", True),
            )
        if not exact_matches and state.last_item_id:
            return ParsedIntent(
                intent=IntentType.REMOVE_ITEM,
                confidence=0.95,
                item_id=state.last_item_id,
                size_id=state.last_item_size,
            )
        return ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="remove_item")

    if not exact_matches and _contains_phrase(normalized, _CONFIRM_PHRASES):
        return ParsedIntent(intent=IntentType.CONFIRM_ORDER, confidence=1)

    quantity = _parse_quantity(normalized, exact_matches[0] if len(exact_matches) == 1 else None)
    has_quantity_change = _contains_phrase(normalized, _QUANTITY_WORDS) or (
        quantity is not None
        and _contains_phrase(normalized, _CHANGE_WORDS)
        and not _contains_phrase(normalized, _ADD_WORDS)
    )
    if has_quantity_change:
        match = exact_matches[0] if len(exact_matches) == 1 else None
        return ParsedIntent(
            intent=IntentType.SET_QUANTITY,
            confidence=1 if quantity is not None else 0.4,
            item_id=match.item["id"] if match else state.last_item_id,
            size_id=_parse_size(normalized, match.item) if match else state.last_item_size,
            quantity=quantity,
            clarification_reason=None if quantity is not None else "missing_quantity",
        )

    if len(exact_matches) == 1:
        match = exact_matches[0]
        new_size = _parse_size(normalized, match.item)
        existing = [line for line in order.items if line.id == match.item["id"]]
        if new_size and existing and _contains_phrase(normalized, _CHANGE_WORDS):
            return ParsedIntent(
                intent=IntentType.CHANGE_SIZE,
                confidence=1,
                item_id=match.item["id"],
                size_id=new_size,
            )

    explicit_address = extract_address_from_original(original, allow_entire=False)
    if explicit_address is not None:
        return ParsedIntent(
            intent=IntentType.SET_ADDRESS,
            confidence=1,
            address=explicit_address,
        )

    if not exact_matches and _contains_phrase(normalized, _INSTRUCTION_WORDS):
        return ParsedIntent(
            intent=IntentType.SET_INSTRUCTIONS,
            confidence=1,
            instructions=_extract_instructions(original),
        )

    if _contains_phrase(normalized, _CONFIRM_PHRASES):
        return ParsedIntent(intent=IntentType.CONFIRM_ORDER, confidence=1)

    if _contains_phrase(normalized, _MENU_WORDS) and _contains_phrase(
        normalized, _MENU_QUESTION_WORDS
    ):
        return ParsedIntent(intent=IntentType.LIST_MENU, confidence=1)

    if _contains_phrase(normalized, _SHOW_ORDER_PHRASES):
        return ParsedIntent(intent=IntentType.SHOW_ORDER, confidence=1)

    if len(exact_matches) > 1:
        return ParsedIntent(
            intent=IntentType.AMBIGUOUS,
            confidence=0,
            clarification_reason="multiple_items",
        )

    match = exact_matches[0] if exact_matches else _fuzzy_item_match(normalized, data)
    if match is not None:
        instructions = (
            _extract_instructions(original)
            if _contains_phrase(normalized, _INSTRUCTION_WORDS)
            else None
        )
        return ParsedIntent(
            intent=IntentType.ADD_ITEM,
            confidence=match.score / 100,
            item_id=match.item["id"],
            size_id=_parse_size(normalized, match.item),
            quantity=quantity if quantity is not None else 1,
            matched_text=match.alias,
            available=match.item.get("available", True),
            instructions=instructions,
            clear_address=address_denied,
        )

    if state.stage == DialogStage.AWAITING_ADDRESS:
        if _contains_phrase(normalized, _ORDER_REQUEST_MARKERS):
            return ParsedIntent(
                intent=IntentType.AMBIGUOUS,
                confidence=1,
                clarification_reason="order_while_awaiting_address",
            )
        address = extract_address_from_original(original, allow_entire=True)
        if address:
            return ParsedIntent(
                intent=IntentType.SET_ADDRESS,
                confidence=0.95,
                address=address,
            )

    if address_denied:
        return ParsedIntent(
            intent=IntentType.AMBIGUOUS,
            confidence=1,
            clarification_reason="address_denied",
            clear_address=True,
        )

    if any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in _UNKNOWN_FOOD_WORDS):
        return ParsedIntent(
            intent=IntentType.UNKNOWN_ITEM,
            confidence=1,
            matched_text=original.strip(),
        )

    return ParsedIntent(
        intent=IntentType.AMBIGUOUS,
        confidence=0,
        clarification_reason="no_safe_rule",
    )
