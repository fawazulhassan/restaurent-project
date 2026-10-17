"""Fast deterministic intent parsing for routine restaurant order turns."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import config
from app.confirmation_urdu import (
    FILLER_WINDOWS,
    NO_WINDOWS,
    ORDER_FILLER,
    _ORDER_EDIT_VERBS,
    classify_confirmation_reply,
    classify_tokens,
    is_address_like_token_class,
    load_menu_noise,
)
from app.dialog_state import DialogStage, DialogState, ItemRequest, PendingChoice
from app.order import Order
from app.text_fold import (
    COMPARE_PUNCT,
    compare_tokens,
    find_phrase_windows,
    fold_text,
    is_digit_token,
    normalize_candidate_text,
    original_tokens,
    phrase_tokens,
)


class IntentType(str, Enum):
    LIST_MENU = "list_menu"
    ADD_ITEM = "add_item"
    ADD_ITEMS = "add_items"
    REMOVE_ITEM = "remove_item"
    SET_QUANTITY = "set_quantity"
    REDUCE_QUANTITY = "reduce_quantity"
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
    add_items: list[ItemRequest] = Field(default_factory=list)
    choice_groups: list[PendingChoice] = Field(default_factory=list)
    edit_action_hint: Literal["reduce", "remove"] | None = None
    dropped_unknown: bool = False


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


def _address_compare_tokens(text: str) -> list[str]:
    return compare_tokens(text)


_ADDRESS_NEGATION_WINDOWS = NO_WINDOWS
_ADDRESS_LEAD_INS = sorted(
    (
        phrase_tokens("میرا ایڈریس ہے"),
        phrase_tokens("میرا ادریس ہے"),
        phrase_tokens("میرا پتہ ہے"),
        phrase_tokens("پتہ ہے"),
        phrase_tokens("ایڈریس ہے"),
        phrase_tokens("mera address hai"),
        phrase_tokens("my address is"),
        phrase_tokens("delivery address is"),
        phrase_tokens("address hai"),
        *(
            phrase_tokens(f"{prefix}{noun} {ending}")
            for prefix in ("", "میرا ")
            for noun in ("پتہ", "ایڈریس", "ادریس", "اڈریس", "ڈریس")
            for ending in ("ہے", "یہ ہے", "یہ نہیں ہے")
        ),
    ),
    key=len,
    reverse=True,
)
_UNAMBIGUOUS_YES_WINDOWS = {
    phrase_tokens(item)
    for item in (
        "ہاں جی",
        "جی ہاں",
        "haan ji",
        "han ji",
        "han g",
        "haan g",
        "ہاں",
        "haan",
        "han",
        "yes",
        "حنجی",
        "ہنجی",
    )
}
_TRAILING_ADDRESS_FILLER_WINDOWS = {
    phrase_tokens(item)
    for item in ("ہے", "میرا", "میری", "بتا", "بتائیں", "hai", "mera")
}


def _normalize_address_candidate(text: str) -> str:
    return normalize_candidate_text(text)


def _match_prefix(
    compare: list[str],
    windows: set[tuple[str, ...]],
) -> tuple[int, tuple[str, ...] | None]:
    for _start, end, phrase in find_phrase_windows(compare, windows):
        if _start == 0:
            return end, phrase
    return 0, None


def _strip_one_leading_negation(
    original: list[str],
    compare: list[str],
) -> tuple[list[str], list[str], bool]:
    width, _phrase = _match_prefix(compare, _ADDRESS_NEGATION_WINDOWS)
    if width == 1 and compare[:1] == ["no"] and len(compare) > 1 and is_digit_token(compare[1]):
        return original, compare, False
    if width:
        return original[width:], compare[width:], True
    return original, compare, False


def _strip_one_address_lead_in(
    original_tokens: list[str],
    compare_tokens: list[str],
) -> tuple[list[str], list[str], bool]:
    for lead_in in _ADDRESS_LEAD_INS:
        width = len(lead_in)
        if width and tuple(compare_tokens[:width]) == lead_in:
            return original_tokens[width:], compare_tokens[width:], True
    return original_tokens, compare_tokens, False


def _strip_leading_unambiguous_yes(
    original: list[str],
    compare: list[str],
) -> tuple[list[str], list[str]]:
    while True:
        width, _phrase = _match_prefix(compare, _UNAMBIGUOUS_YES_WINDOWS)
        if not width:
            return original, compare
        original = original[width:]
        compare = compare[width:]


def _strip_leading_fillers(
    original: list[str],
    compare: list[str],
) -> tuple[list[str], list[str]]:
    while True:
        width, _phrase = _match_prefix(compare, FILLER_WINDOWS)
        if not width:
            return original, compare
        original = original[width:]
        compare = compare[width:]


def _strip_trailing_fillers(candidate: str) -> str:
    original = original_tokens(candidate)
    compare = compare_tokens(candidate)
    while original and compare:
        removed = False
        for filler in sorted(_TRAILING_ADDRESS_FILLER_WINDOWS, key=len, reverse=True):
            width = len(filler)
            if width and tuple(compare[-width:]) == filler:
                original = original[:-width]
                compare = compare[:-width]
                removed = True
                break
        if not removed:
            break
    return _normalize_address_candidate(" ".join(original))


def _address_like_count(text: str) -> int:
    _tokens, classes = classify_tokens(text)
    return sum(1 for class_name in classes if is_address_like_token_class(class_name))


def _is_valid_address_candidate(candidate: str | None) -> bool:
    if not candidate:
        return False
    tokens, classes = classify_tokens(candidate)
    # A remaining rejection is not a replacement address. House "no 5"
    # remains address-like through the shared digit rule.
    return len(tokens) >= 2 and "no" not in classes and "address" in classes


def extract_address_candidate(
    text: str,
    *,
    confirmation_correction: bool = False,
) -> str | None:
    source_tokens = original_tokens(text)
    compare = compare_tokens(text)
    if not source_tokens or not compare:
        return None
    source_tokens, compare, _had_negation = _strip_one_leading_negation(
        source_tokens, compare
    )
    source_tokens, compare, _had_lead_in = _strip_one_address_lead_in(
        source_tokens, compare
    )
    if confirmation_correction:
        source_tokens, compare = _strip_leading_unambiguous_yes(source_tokens, compare)
    source_tokens, compare = _strip_leading_fillers(source_tokens, compare)
    candidate = _strip_trailing_fillers(_normalize_address_candidate(" ".join(source_tokens)))
    return candidate or None


def classify_address_reply(
    text: str,
) -> Literal["yes", "no", "new_address", "unclear"]:
    tokens = _address_compare_tokens(text)
    if not tokens:
        return "unclear"

    # Recognize pure rejection before attempting candidate extraction,
    # including negation after the address descriptor ("میرا پتہ یہ نہیں ہے").
    confirmation = classify_confirmation_reply(text)
    if confirmation == "no":
        return "no"

    source_tokens = original_tokens(text)
    stripped_original, stripped_compare, had_negation = _strip_one_leading_negation(
        source_tokens, tokens
    )
    stripped_original, stripped_compare, had_lead_in = _strip_one_address_lead_in(
        stripped_original, stripped_compare
    )
    candidate = extract_address_candidate(text, confirmation_correction=True)
    if (had_negation or had_lead_in) and _is_valid_address_candidate(candidate):
        return "new_address"
    if (
        not had_negation
        and not had_lead_in
        and sum(
            1
            for class_name in classify_tokens(text)[1]
            if is_address_like_token_class(class_name)
        )
        >= 2
        and _is_valid_address_candidate(candidate)
    ):
        return "new_address"
    if had_negation:
        rest_text = " ".join(stripped_original)
        if _address_like_count(rest_text) <= 1:
            return "no"

    return classify_confirmation_reply(text)


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

_SIZE_LEMMAS = {
    "half": (
        "half",
        "ہاف",
        "آدھی",
        "آدھا",
        "ادھی",
        "ادھا",
        "adhi",
        "aadhi",
        "آدی",
        "ادی",
        "aadi",
        "adi",
        "aadha",
        "adha",
        "نصف",
    ),
    "full": (
        "full",
        "فول",
        "فل",
        "پوری",
        "پورا",
        "poori",
        "puri",
        "pori",
        "pura",
        "poora",
    ),
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
_OLD_BEFORE_MARKERS = ("کی جگہ", "کے بجائے", "ki jagah", "ke bajaye")
_OLD_AFTER_MARKERS = ("instead of",)

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
        menu = json.load(menu_file)
    load_menu_noise(menu)
    return menu


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
                if item["id"] == "plain-naan" and alias in {"nan", "naan", "نان"} and normalized[:match.start()].rstrip().endswith(("garlic", "لہسن", "گارلک", "کالک")):
                    continue
                candidate = ItemMatch(item, alias, match.start(), match.end())
                previous = best_by_id.get(item["id"])
                if previous is None or len(alias) > len(previous.alias):
                    best_by_id[item["id"]] = candidate
                break
    matches = list(best_by_id.values())
    # Generic "naan" must not also match inside the more specific garlic naan.
    matches = [match for match in matches if not any(
        other.start <= match.start and other.end >= match.end
        and len(other.alias) > len(match.alias)
        for other in matches
    )]
    return sorted(matches, key=lambda match: len(match.alias), reverse=True)


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
                if item["id"] == "plain-naan" and alias in {"nan", "naan", "نان"} and start and words[start - 1] in {"garlic", "لہسن", "گارلک", "کالک"}:
                    continue
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
    found = sizes_in_text(normalized, item)
    if len(found) == 1:
        return found[0]
    return None


@lru_cache(maxsize=1)
def _size_phrase_index() -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    for size_id, lemmas in _SIZE_LEMMAS.items():
        for lemma in lemmas:
            phrase = phrase_tokens(lemma)
            if not phrase:
                continue
            previous = index.get(phrase)
            if previous is not None and previous != size_id:
                raise RuntimeError(f"Size lemma {lemma!r} maps to {previous} and {size_id}.")
            index[phrase] = size_id
    return index


def sizes_in_text(text: str, item: dict[str, Any] | None = None) -> list[str]:
    """Size ids whose lemmas occur in text, restricted to the item's menu sizes."""
    valid = None if item is None else {size["id"] for size in item.get("sizes", [])}
    if valid is not None and not valid:
        return []
    index = _size_phrase_index()
    found: list[str] = []
    for _start, _end, phrase in find_phrase_windows(compare_tokens(text), set(index)):
        size_id = index[phrase]
        if valid is not None and size_id not in valid:
            continue
        if size_id not in found:
            found.append(size_id)
    return found


def _marker_span(tokens: list[str]) -> tuple[str, int, int] | None:
    old_before = {phrase_tokens(marker) for marker in _OLD_BEFORE_MARKERS}
    old_after = {phrase_tokens(marker) for marker in _OLD_AFTER_MARKERS}
    for start, end, phrase in find_phrase_windows(tokens, old_before | old_after):
        kind = "old_before" if phrase in old_before else "old_after"
        return kind, start, end
    return None


def size_direction(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (old size, new size, error). One lemma is the new size."""
    tokens = compare_tokens(text)
    index = _size_phrase_index()
    spans: list[tuple[str, int, int]] = []
    for start, end, phrase in find_phrase_windows(tokens, set(index)):
        size_id = index[phrase]
        if not spans or spans[-1][0] != size_id:
            spans.append((size_id, start, end))
    unique: list[tuple[str, int, int]] = []
    for span in spans:
        if span[0] not in [item[0] for item in unique]:
            unique.append(span)
    if not unique:
        return None, None, None
    if len(unique) == 1:
        return None, unique[0][0], None
    marker = _marker_span(tokens)
    if marker is None:
        return None, None, "both_lemmas_no_marker"
    kind, marker_start, marker_end = marker
    before = [span for span in unique if span[2] <= marker_start]
    after = [span for span in unique if span[1] >= marker_end]
    if not before or not after:
        return None, None, "both_lemmas_no_marker"
    if kind == "old_before":
        return before[-1][0], after[0][0], None
    return after[0][0], before[-1][0], None


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
        return _strip_trailing_fillers(_normalize_address_candidate(candidate)) or None
    residence = re.search(r"(?:میں|main)\s+(.+?)\s+(?:رہتا ہوں|رہتی ہوں|rehta hun|rehti hun)\s*$", original, re.IGNORECASE)
    if residence:
        return _strip_trailing_fillers(_normalize_address_candidate(residence.group(1))) or None
    return None


def extract_address_from_original(original: str, *, allow_entire: bool) -> str | None:
    """Extract an address only from the untouched original transcription."""
    normalized_original, _ = normalize_customer_text(original)
    candidate = _extract_prefixed_address(original)
    if candidate is None and _contains_phrase(normalized_original, _ADDRESS_DENIAL_PHRASES):
        return None
    if candidate is None and allow_entire:
        candidate = extract_address_candidate(original) or original.strip(" ,.-")
        candidate = re.sub(r"\s+(?:hai|ہے)\s*[.؟?]*$", "", candidate, flags=re.IGNORECASE)
    if not candidate or len(candidate.strip()) < 4:
        return None
    normalized, _ = normalize_customer_text(candidate)
    if _contains_phrase(normalized, _ADDRESS_DENIAL_PHRASES):
        return None
    if classify_address_reply(candidate) == "no":
        return None
    if normalized in _YES_ONLY or _contains_phrase(normalized, _CONFIRM_PHRASES):
        return None
    return candidate.strip()


def is_explicit_order_command(text: str) -> bool:
    """Menu words alone must not turn an address reply into an order mutation."""
    normalized, _ = normalize_customer_text(text)
    return (
        _contains_phrase(normalized, _ORDER_REQUEST_MARKERS + _REMOVE_WORDS + _QUANTITY_WORDS + _CHANGE_WORDS + _CONFIRM_PHRASES + _SHOW_ORDER_PHRASES)
        or _contains_phrase(normalized, _MENU_WORDS)
        or normalized in {"cancel", "quit", "exit", "بند"}
    )


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


def quantity_evidence(text: str) -> tuple[set[int], set[int]]:
    """Known spoken numbers and explicitly rejected numbers; never guess STT typos."""
    normalized, _ = normalize_customer_text(text)
    for match in _exact_item_matches(normalized, load_intent_menu()):
        normalized = _boundary_pattern(match.alias).sub(" ", normalized)
    tokens = [token.strip("۔.،,؟?!") for token in normalized.split()]
    values: set[int] = set()
    rejected: set[int] = set()
    for index, token in enumerate(tokens):
        if token in {"do", "دو"} and index and tokens[index - 1] in {"kar", "hata", "nikal", "confirm", "کر", "ہٹا", "نکال", "ختم"}:
            continue
        value = int(token) if token.isdigit() else _NUMBER_WORDS.get(token)
        if value is None:
            continue
        values.add(value)
        after = tokens[index + 1:index + 2]
        before = tokens[index - 1:index] if index else []
        if (after and tuple(compare_tokens(after[0])) in NO_WINDOWS) or before == ["not"]:
            rejected.add(value)
    return values, rejected


def mentioned_item_ids(text: str) -> set[str]:
    normalized, _ = normalize_customer_text(text)
    return {match.item["id"] for match in _exact_item_matches(normalized, load_intent_menu())}


def _quantity_correction_intent(
    normalized: str, matches: list[ItemMatch], state: DialogState, order: Order,
) -> ParsedIntent | None:
    if not order.items or state.pending_item_id or len(matches) > 1:
        return None
    if _contains_phrase(normalized, _REMOVE_WORDS):
        return None
    item_id = matches[0].item["id"] if matches else state.quantity_correction_item_id or state.last_item_id
    if not item_id or not any(line.id == item_id for line in order.items):
        return None
    if extract_address_from_original(normalized, allow_entire=False):
        return None
    absolute = any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in (
        "total", "ٹوٹل", "کل", "صرف", "only", "set", "make", "quantity", "qty", "تعداد",
    ))
    changing = any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in (
        "کرنا", "کر دو", "کر دیں", "کردیں", "بنا دو", "بنا دیں", "ہو جائیں",
        "karna", "kar do", "kar dein", "kardo", "kardein", "change", "badal",
    ))
    explicit_add = any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in (
        "add", "more", "extra", "order", "آرڈر", "اوڈر", "مزید", "اور", "بھی", "شامل", "aur", "our",
    ))
    has_no = "no" in classify_tokens(normalized)[1]
    waiting = bool(state.quantity_correction_item_id and not explicit_add and not matches)
    if not absolute and not waiting and not ((changing or has_no) and matches and not explicit_add):
        return None
    # Keep the existing fast path for unambiguous explicit quantity commands.
    numbers, rejected = quantity_evidence(normalized)
    _old_size, directed_new, direction_error = size_direction(normalized)
    if matches and (directed_new or direction_error or _parse_size(normalized, matches[0].item)) and not numbers and not absolute and not has_no:
        return None
    if waiting and len(normalized.split()) == 1 and len(numbers) == 1 and not rejected:
        return ParsedIntent(
            intent=(IntentType.REDUCE_QUANTITY if state.quantity_correction_action == "reduce" else IntentType.SET_QUANTITY),
            confidence=1, item_id=item_id,
            size_id=state.quantity_correction_size_id, quantity=next(iter(numbers)),
        )
    if _contains_phrase(normalized, _QUANTITY_WORDS) and len(numbers) == 1 and not rejected and not has_no:
        return None
    return ParsedIntent(
        intent=IntentType.AMBIGUOUS, confidence=0, item_id=item_id,
        size_id=_parse_size(normalized, matches[0].item) if matches else (
            state.quantity_correction_size_id if waiting else state.last_item_size
        ),
        clarification_reason="quantity_correction",
    )


def _basket_edit_intent(normalized: str, matches: list[ItemMatch], state: DialogState, order: Order) -> ParsedIntent | None:
    if not order.items or state.pending_item_id or extract_address_from_original(normalized, allow_entire=False):
        return None
    decreases = ("کم", "kam", "reduce", "decrease", "subtract", "minus", "less", "گھٹا", "ghata")
    removals = (*_REMOVE_WORDS, "remove", "delete", "cancel", "hatao", "nikalo", "ہٹاؤ", "نکالو", "نکال", "ختم")
    reducing = any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in decreases)
    removing = any(_boundary_pattern(_normalize_alias(word)).search(normalized) for word in removals)
    if not reducing and not removing:
        return None
    numbers, _ = quantity_evidence(normalized)
    # A single named line with no count stays on the deterministic remove path.
    if contains_removal_verb(normalized) and not reducing and not numbers and len(matches) == 1:
        return None
    return ParsedIntent(
        intent=IntentType.AMBIGUOUS, confidence=0,
        item_id=matches[0].item["id"] if len(matches) == 1 else None,
        size_id=_parse_size(normalized, matches[0].item) if len(matches) == 1 else None,
        clarification_reason="basket_edit",
        edit_action_hint=(None if len(matches) > 1 else "reduce" if reducing or numbers else "remove"),
    )


_REMOVAL_VERB_RE = re.compile(
    r"(?<!\w)(?:ہٹا\w*|نکال\w*|hata\w*|nikal\w*|ختم|remove|delete|cancel)(?!\w)"
)


def structured_addition(text: str, item_ids: list[str]) -> bool:
    """True when every token is an item, one size, a quantity, or an add word."""
    normalized, _ = normalize_customer_text(text)
    if len(sizes_in_text(normalized)) > 1:
        return False
    data = load_intent_menu()
    from app.confirmation_urdu import ORDER_FILLER

    allowed = set(_NUMBER_WORDS)
    allowed.update(_normalize_alias(word) for word in (*_ADD_WORDS, *ORDER_FILLER))
    allowed.update(_normalize_alias(word) for word in (
        "aur", "and", "our", "اور", "more", "extra", "please", "بھی",
        "kar", "karo", "do", "دو", "کر", "کرو", "دیں", "کرنا",
    ))
    for lemmas in _SIZE_LEMMAS.values():
        allowed.update(_normalize_alias(lemma) for lemma in lemmas)
    wanted = set(item_ids)
    for item in _all_menu_items(data):
        if item["id"] not in wanted:
            continue
        for alias in _item_aliases(item):
            allowed.update(alias.split())
    return all(token.isdigit() or token in allowed for token in normalized.split())


def contains_removal_verb(normalized: str) -> bool:
    return _REMOVAL_VERB_RE.search(normalized) is not None


def removal_is_prohibited(normalized: str) -> bool:
    """Verb negation, including نہیں within two tokens before a removal verb."""
    if re.search(
        r"(?:do not|don't|don t|dont|not)\s+(?:remove|delete|reduce|subtract)\b|"
        r"(?:مت|mat)\s+(?:ہٹا\w*|نکال\w*|hata\w*|nikal\w*|kam)",
        normalized,
    ):
        return True
    tokens = normalized.split()
    negations = {"نہیں", "nahi", "nahin"}
    for index, token in enumerate(tokens):
        if _REMOVAL_VERB_RE.fullmatch(token) and any(word in negations for word in tokens[max(0, index - 2):index]):
            return True
    return False


def contains_add_word(normalized: str) -> bool:
    return _contains_phrase(normalized, _ADD_WORDS)


def addition_is_prohibited(normalized: str) -> bool:
    return re.search(
        r"(?:do not|don't|don t|dont|not)\s+add\b|(?:مت|mat)\s+(?:add|شامل|ڈال\w*|dal\w*)",
        normalized,
    ) is not None


@lru_cache(maxsize=1)
def item_group_index() -> dict[str, tuple[str, ...]]:
    """Tokens shared by exactly two items. Ingredients shared more widely are excluded."""
    from app.confirmation_urdu import ORDER_FILLER, _ORDER_QUANTITY_WORDS

    items = [item for item in _all_menu_items(load_intent_menu()) if item.get("available", True)]
    closed = {_normalize_alias(word) for word in (*_ORDER_QUANTITY_WORDS, *ORDER_FILLER, *tuple(_NUMBER_WORDS))}
    for lemmas in _SIZE_LEMMAS.values():
        closed.update(_normalize_alias(lemma) for lemma in lemmas)
    alias_owners: dict[str, set[str]] = {}
    token_owners: dict[str, set[str]] = {}
    for item in items:
        for alias in _item_aliases(item):
            alias_owners.setdefault(alias, set()).add(item["id"])
            for token in alias.split():
                token_owners.setdefault(token, set()).add(item["id"])
    unique_alias = {alias for alias, owners in alias_owners.items() if len(owners) == 1}
    return {
        token: tuple(sorted(owners))
        for token, owners in token_owners.items()
        if len(owners) == 2 and token not in unique_alias and token not in closed
    }


def _group_key(token: str) -> str:
    """Shared-dish key. Built on fold_text; fold_text itself stays unchanged."""
    folded = fold_text(token).translate({
        ord("ڑ"): "ر",
        ord("ژ"): "ز",
        ord("ئ"): "ی",
        ord("ے"): "ی",
    })
    chars = [
        char for index, char in enumerate(folded)
        if char != "ہ" or index == 0 or index == len(folded) - 1
    ]
    collapsed: list[str] = []
    for char in chars:
        if collapsed and collapsed[-1] == char:
            continue
        collapsed.append(char)
    return "".join(collapsed)


_CLAUSE_AUXILIARIES = (
    "ہوں", "ہو", "گا", "گی", "گے", "تھا", "تھی", "تھے",
    "hoon", "hun", "ho", "ga", "gi", "ge", "tha", "thi", "the",
)
_INFLECTION_ENDINGS = (
    "oon", "وں", "تا", "تی", "تے", "یں", "نا", "un", "ta", "ti", "te", "na",
)
_CHOICE_NEGATIONS = ("نہیں", "nahi", "nahin", "مت", "mat", "not", "don't")


def _choice_tokens(clause: str) -> list[str]:
    return [
        token for raw in clause.split()
        if (token := raw.strip(COMPARE_PUNCT))
    ]


def _folded_words(words: tuple[str, ...] | list[str]) -> set[str]:
    return {fold_text(word) for word in words}


def _alias_tokens(menu_data: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for item in _all_menu_items(menu_data):
        for alias in _item_aliases(item):
            tokens.update(fold_text(part) for part in alias.split())
    return tokens


def _is_edit_or_removal(token: str) -> bool:
    if _REMOVAL_VERB_RE.fullmatch(token):
        return True
    folded = fold_text(token)
    return any(
        " " not in verb and fold_text(verb) == folded
        for verb in _ORDER_EDIT_VERBS
    )


def _is_inflection(token: str, alias_tokens: set[str]) -> bool:
    """A filler or add word plus one verbal ending. Alias, negation, and edit tokens never match."""
    folded = fold_text(token)
    if folded in alias_tokens or folded in _folded_words(_CHOICE_NEGATIONS) or _is_edit_or_removal(token):
        return False
    sources = [
        fold_text(word) for word in (*ORDER_FILLER, *_ADD_WORDS)
        if len(fold_text(word)) >= 4
    ]
    endings = sorted((fold_text(ending) for ending in _INFLECTION_ENDINGS), key=len, reverse=True)
    for ending in endings:
        if not ending or not folded.endswith(ending) or len(folded) <= len(ending):
            continue
        remainder = folded[:len(folded) - len(ending)]
        if len(remainder) < 3:
            continue
        if any(source[:len(remainder)] == remainder for source in sources):
            return True
    return False


def _is_framing(token: str, alias_tokens: set[str]) -> bool:
    folded = fold_text(token)
    if folded in _folded_words((*ORDER_FILLER, *_ADD_WORDS, *_CLAUSE_AUXILIARIES)):
        return True
    return _is_inflection(token, alias_tokens)


def _is_quantity_token(token: str, previous: str | None) -> bool:
    if token.isdigit() or is_digit_token(token):
        return True
    if token == "do" and previous in {"kar", "hata", "nikal", "confirm"}:
        return False
    return token in _NUMBER_WORDS


def _near_group(token: str, groups: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    if len(fold_text(token)) < 4:
        return None
    key = _group_key(token)
    owners = [item_ids for group_token, item_ids in groups.items() if _group_key(group_token) == key]
    if len(owners) != 1:
        return None
    return owners[0]


_CLAUSE_JOINERS = {"اور", "and", "aur", "our"}


def _extends_framing(token: str, alias_tokens: set[str]) -> bool:
    """کرنی-style ending on an order word. Not used for dish or group matching."""
    folded = fold_text(token)
    if (
        not folded
        or folded in alias_tokens
        or folded in _folded_words(_CHOICE_NEGATIONS)
        or _is_edit_or_removal(token)
    ):
        return False
    for source in _folded_words((*ORDER_FILLER, *_ADD_WORDS, *_CLAUSE_AUXILIARIES)):
        extra = len(folded) - len(source)
        if len(source) >= 2 and 0 < extra <= 2 and folded.startswith(source):
            return True
    return False


def _is_size_token(token: str) -> bool:
    return bool(sizes_in_text(token))


def _token_near_alias(token: str, menu_data: dict[str, Any]) -> bool:
    """A token within HYBRID_FUZZY_MIN_SCORE of any alias is explained."""
    if not config.HYBRID_FUZZY_MATCH_ENABLED:
        return False
    folded = fold_text(token)
    if not folded:
        return False
    threshold = config.HYBRID_FUZZY_MIN_SCORE
    for item in _all_menu_items(menu_data):
        for alias in _item_aliases(item):
            if fuzz.ratio(folded, fold_text(alias)) >= threshold:
                return True
    return False


def _has_order_cue(normalized: str, tokens: list[str], alias_tokens: set[str]) -> bool:
    if any(
        _boundary_pattern(_normalize_alias(word)).search(normalized)
        for word in (*_UNKNOWN_FOOD_WORDS, "اوڈر", "order")
    ):
        return True
    if _contains_phrase(normalized, _ADD_WORDS) or _contains_phrase(normalized, _ORDER_REQUEST_MARKERS):
        return True
    return any(_is_inflection(token, alias_tokens) for token in tokens)


def _unknown_menu_order(normalized: str, state: DialogState, menu_data: dict[str, Any]) -> bool:
    """Order-shaped leftovers that match no item. Size, choice, and menu turns are not this."""
    if state.pending_choices or state.stage == DialogStage.AWAITING_SIZE:
        return False
    if _contains_phrase(normalized, _MENU_WORDS):
        return False
    alias_tokens = _alias_tokens(menu_data)
    tokens = _choice_tokens(normalized)
    unexplained = False
    previous: str | None = None
    for token in tokens:
        folded = fold_text(token)
        if (
            token in _CLAUSE_JOINERS
            or _is_quantity_token(token, previous)
            or _is_framing(token, alias_tokens)
            or _is_size_token(token)
            or _extends_framing(token, alias_tokens)
            or folded in alias_tokens
            or _token_near_alias(token, menu_data)
        ):
            previous = token
            continue
        unexplained = True
        previous = token
    if not unexplained:
        return False
    return _has_order_cue(normalized, tokens, alias_tokens)


def _pure_group_choice(clause: str, menu_data: dict[str, Any]) -> tuple[tuple[str, ...], int] | None:
    """One group hit, at most one quantity, and only framing besides those."""
    if sizes_in_text(clause) or _exact_item_matches(clause, menu_data):
        return None
    groups = item_group_index()
    alias_tokens = _alias_tokens(menu_data)
    tokens = _choice_tokens(clause)
    group_ids: tuple[str, ...] | None = None
    quantities = 0
    previous: str | None = None
    for token in tokens:
        if _is_quantity_token(token, previous):
            quantities += 1
        elif _is_framing(token, alias_tokens):
            pass
        elif token in groups:
            if group_ids is not None:
                return None
            group_ids = groups[token]
        elif (near := _near_group(token, groups)) is not None:
            if group_ids is not None:
                return None
            group_ids = near
        else:
            return None
        previous = token
    if group_ids is None or quantities > 1:
        return None
    return group_ids, _parse_quantity(clause, None) or 1


def _candidate_from_choice(normalized: str, candidate_ids: list[str], menu_data: dict[str, Any]) -> str | None:
    """The one candidate named by an exact alias or a token that belongs to it alone."""
    allowed = set(candidate_ids)
    named = [match.item["id"] for match in _exact_item_matches(normalized, menu_data) if match.item["id"] in allowed]
    groups = item_group_index()
    for token in normalized.split():
        owners = set(groups.get(token, ())) & allowed
        if len(owners) == 1:
            named.append(next(iter(owners)))
    unique = list(dict.fromkeys(named))
    if len(unique) == 1:
        return unique[0]
    return None


def _combine_choice_clauses(parts: list[ParsedIntent]) -> ParsedIntent | None:
    groups: list[PendingChoice] = []
    adds: list[ItemRequest] = []
    if not any(part.clarification_reason == "item_choice" for part in parts):
        return None
    for part in parts:
        if part.clarification_reason == "item_choice":
            groups.extend(part.choice_groups)
            continue
        if part.intent == IntentType.ADD_ITEM and part.item_id and not part.clarification_reason:
            adds.append(ItemRequest(
                item_id=part.item_id, size_id=part.size_id, quantity=part.quantity or 1,
                instructions=part.instructions,
            ))
            continue
        return None
    if not groups:
        return None
    return ParsedIntent(
        intent=IntentType.AMBIGUOUS,
        confidence=1,
        clarification_reason="item_choice",
        add_items=adds,
        choice_groups=groups,
    )


def _pending_menu_item(state: DialogState, menu_data: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (item for item in _all_menu_items(menu_data) if item["id"] == state.pending_item_id),
        None,
    )


def _missing_pending_size(state: DialogState, pending_item: dict[str, Any] | None) -> ParsedIntent:
    return ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=0.4,
        item_id=state.pending_item_id,
        quantity=state.pending_quantity,
        available=(pending_item or {}).get("available", True),
        instructions=state.pending_instructions,
        clarification_reason="missing_size",
    )


def _pending_size_intent(
    normalized: str,
    exact_matches: list[ItemMatch],
    state: DialogState,
    menu_data: dict[str, Any],
) -> ParsedIntent | None:
    """Resolve a size answer for the item just asked, or for one named pending item."""
    if not state.pending_item_id or not sizes_in_text(normalized):
        return None
    pending_item = _pending_menu_item(state, menu_data)
    unsized_ids = [request.item_id for request in state.pending_additions if not request.size_id]
    if state.pending_item_id not in unsized_ids:
        unsized_ids.insert(0, state.pending_item_id)
    if len(exact_matches) >= 2:
        return _missing_pending_size(state, pending_item)
    if len(exact_matches) == 1:
        named = exact_matches[0].item
        if named["id"] in unsized_ids:
            size_id = _parse_size(normalized, named)
            if size_id:
                request = next(
                    (part for part in state.pending_additions if part.item_id == named["id"]),
                    None,
                )
                return ParsedIntent(
                    intent=IntentType.ADD_ITEM,
                    confidence=1,
                    item_id=named["id"],
                    size_id=size_id,
                    quantity=request.quantity if request else state.pending_quantity,
                    available=named.get("available", True),
                    instructions=request.instructions if request else state.pending_instructions,
                )
        return _missing_pending_size(state, pending_item)
    size_id = _parse_size(normalized, pending_item)
    if size_id is None:
        return _missing_pending_size(state, pending_item)
    return ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=1,
        item_id=state.pending_item_id,
        size_id=size_id,
        quantity=state.pending_quantity,
        available=(pending_item or {}).get("available", True),
        instructions=state.pending_instructions,
    )


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

    if state.pending_choices:
        chosen = _candidate_from_choice(normalized, state.pending_choices, data)
        if chosen:
            return ParsedIntent(
                intent=IntentType.ADD_ITEM,
                confidence=1,
                item_id=chosen,
                quantity=state.pending_choice_quantity or 1,
                clarification_reason="choice_selected",
            )
        return ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="choice_retry")

    pending_size = _pending_size_intent(normalized, exact_matches, state, data)
    if pending_size is not None:
        return pending_size

    edit = _basket_edit_intent(normalized, exact_matches, state, order)
    if edit is not None:
        return edit

    correction = _quantity_correction_intent(normalized, exact_matches, state, order)
    if correction is not None:
        return correction

    pure_choice = _pure_group_choice(normalized, data)
    if pure_choice is not None:
        item_ids, quantity = pure_choice
        return ParsedIntent(
            intent=IntentType.AMBIGUOUS,
            confidence=1,
            clarification_reason="item_choice",
            choice_groups=[PendingChoice(item_ids=list(item_ids), quantity=quantity)],
        )

    # Parse each spoken item clause independently so quantities/sizes cannot
    # leak from one item to another. Address and mutation requests stay separate.
    clauses = [clause.strip() for clause in re.split(
        r"(?<!\w)(?:اور|and|aur|our)(?!\w)|[,،]", normalized
    ) if clause.strip()]
    if (
        len(clauses) > 1
        and state.stage != DialogStage.AWAITING_ADDRESS_CONFIRMATION
        and not _contains_phrase(normalized, _REMOVE_WORDS + _QUANTITY_WORDS)
        and extract_address_from_original(original, allow_entire=False) is None
    ):
        clause_intents = [parse_customer_intent(clause, state, order, data) for clause in clauses]
        if (
            state.stage != DialogStage.AWAITING_SIZE
            and not state.pending_choices
            and not _contains_phrase(normalized, _MENU_WORDS)
            and any(part.intent == IntentType.UNKNOWN_ITEM for part in clause_intents)
        ):
            has_real_item = any(
                part.clarification_reason == "item_choice"
                or (
                    part.intent == IntentType.ADD_ITEM
                    and part.item_id
                    and not part.clarification_reason
                )
                for part in clause_intents
            )
            if has_real_item:
                return ParsedIntent(
                    intent=IntentType.AMBIGUOUS,
                    confidence=0,
                    clarification_reason="no_safe_rule",
                )
            return ParsedIntent(
                intent=IntentType.UNKNOWN_ITEM,
                confidence=1,
                matched_text=original.strip(),
            )
        combined_choice = _combine_choice_clauses(clause_intents)
        if combined_choice is not None:
            return combined_choice
        if all(part.intent == IntentType.ADD_ITEM for part in clause_intents):
            return ParsedIntent(
                intent=IntentType.ADD_ITEMS,
                confidence=min(part.confidence for part in clause_intents),
                add_items=[ItemRequest(
                    item_id=part.item_id, size_id=part.size_id,
                    quantity=part.quantity or 1, instructions=part.instructions,
                ) for part in clause_intents],
            )
        if (
            any(part.intent == IntentType.ADD_ITEM for part in clause_intents)
            and re.search(r"(?<!\w)(?:اور|and|aur|our)(?!\w)", normalized)
            and not all(
                part.intent in {IntentType.ADD_ITEM, IntentType.CONFIRM_ORDER}
                or classify_confirmation_reply(clause) == "yes"
                or phrase_tokens(clause) in {
                    phrase_tokens(phrase) for phrase in (
                        "شامل کریں", "شامل کر دیں", "بھی شامل کریں", "ایڈ کریں",
                        "add kar do", "add please", "بھی", "please",
                    )
                }
                for clause, part in zip(clauses, clause_intents)
            )
        ):
            return ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0,
                                clarification_reason="multiple_items")

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

    if contains_removal_verb(normalized) and len(exact_matches) == 1:
        match = exact_matches[0]
        return ParsedIntent(
            intent=IntentType.REMOVE_ITEM,
            confidence=1,
            item_id=match.item["id"],
            size_id=_parse_size(normalized, match.item),
            matched_text=match.alias,
            available=match.item.get("available", True),
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
        existing = [line for line in order.items if line.id == match.item["id"]]
        if existing and _contains_phrase(normalized, _CHANGE_WORDS):
            _old_size, directed_new, direction_error = size_direction(normalized)
            offered = {size["id"] for size in match.item.get("sizes", [])}
            if directed_new and offered and directed_new not in offered:
                directed_new = None
            new_size = directed_new or _parse_size(normalized, match.item)
            if new_size or direction_error:
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

    if _unknown_menu_order(normalized, state, data):
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
