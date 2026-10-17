"""Generic Urdu/Roman Urdu confirmation classification."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from rapidfuzz import fuzz

from app.text_fold import compare_tokens, find_phrase_windows, fold_text, is_digit_token, phrase_tokens

ConfirmationLabel = Literal["yes", "no", "unclear"]

YES_PHRASES = {
    "ہاں",
    "جی",
    "جی ہاں",
    "ہاں جی",
    # Observed Whisper rendering of the short acknowledgement "ہاں جی".
    # Match the whole phrase; bare "ڈی" is not an acknowledgement.
    "ہاں ڈی",
    "ہان جی",
    "جی ہان",
    "ٹھیک",
    "ٹھیک ہے",
    "اوکے",
    "کنفرم",
    "بالکل",
    "yes",
    "ok",
    "okay",
    "confirm",
    "theek",
    "theek hai",
    "haan",
    "han",
    "haan ji",
    "han ji",
    "han g",
    "haan g",
    "hn g",
    "g hn",
    "g han",
    "g haan",
    "han gee",
    "ji",
    "ji han",
    "hanji",
    "haanji",
    "حنجی",
    "ہنجی",
    "ھانجی",
    "ہاںجی",
    "انجی",
    "آنڈی",
}
NO_PHRASES = {
    "نہیں",
    "نہی",
    "نہیم",
    "نی",
    "نای",
    "نہ",
    "غلط",
    "نہیں جی",
    "no",
    "nahi",
    "nhi",
    "nahin",
    "nai",
    "wrong",
    "galat",
}

# Compose common spellings, both word orders, and joined forms instead of
# enumerating whole customer sentences. Matching remains exact after folding.
_ROMAN_YES = {"haan", "han", "hn", "hanh"}
_ROMAN_NO = {"nahi", "nahin", "nhi", "nhin", "naheen", "nahein", "nai", "ni", "na", "no", "nope"}
_ROMAN_JI = {"ji", "jee", "gee", "g"}
YES_PHRASES.update(_ROMAN_YES | _ROMAN_JI)
NO_PHRASES.update(_ROMAN_NO | {"نھیں", "نہں", "نا", "نئیں"})
for words, target in ((_ROMAN_YES, YES_PHRASES), (_ROMAN_NO, NO_PHRASES)):
    for word in words:
        for ji in _ROMAN_JI:
            target.update({f"{word} {ji}", f"{ji} {word}", f"{word}{ji}", f"{ji}{word}"})
YES_PHRASES.update({"ہان", "ڈی ہاں", "ڈی ہان", "جی ہن", "ہن جی"})

# This transcript is also a food word. Recognize it only in the address
# confirmation context, never in the general food/order parser.
_ADDRESS_SHORT_YES = {"ہنڈی"}
FILLER_PHRASES = {
    "یہ",
    "یہی",
    "ye",
    "ہے",
    "hai",
    "ہی",
    "hi",
    "میرا",
    "میری",
    "mera",
    "please",
    "پلیز",
    "پتہ",
    "ایڈریس",
    "ادریس",
    "اڈریس",
    "ڈریس",
    "address",
    "رہا",
}
STATIC_MENU_NOISE_PHRASES = {
    "ڈیڑھ",
    "لیٹر",
    "ڈیڑھ لیٹر",
    "آدھا",
    "آدھی",
    "پوری",
    "پورا",
    "half",
    "full",
    "liter",
    "litre",
}

YES_WINDOWS = {phrase_tokens(item) for item in YES_PHRASES}
NO_WINDOWS = {phrase_tokens(item) for item in NO_PHRASES}
FILLER_WINDOWS = {phrase_tokens(item) for item in FILLER_PHRASES}
STATIC_MENU_NOISE_WINDOWS = {phrase_tokens(item) for item in STATIC_MENU_NOISE_PHRASES}
_MENU_NOISE_WINDOWS = set(STATIC_MENU_NOISE_WINDOWS)


def reset_menu_noise() -> None:
    global _MENU_NOISE_WINDOWS
    _MENU_NOISE_WINDOWS = set(STATIC_MENU_NOISE_WINDOWS)
    _menu_noise_version.cache_clear()


def load_menu_noise(menu: dict) -> None:
    global _MENU_NOISE_WINDOWS
    windows = set(STATIC_MENU_NOISE_WINDOWS)
    for category in menu.get("categories", []):
        for item in category.get("items", []):
            raw = [
                item.get("name", ""),
                item.get("name_urdu", ""),
                item.get("spoken_name_urdu", ""),
                *item.get("aliases", []),
            ]
            for value in raw:
                if value:
                    tokens = phrase_tokens(str(value))
                    if tokens:
                        windows.add(tokens)
    _MENU_NOISE_WINDOWS = windows
    _menu_noise_version.cache_clear()


@lru_cache(maxsize=1)
def _menu_noise_version() -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(_MENU_NOISE_WINDOWS))


def _window_indexes(tokens: list[str], phrases: set[tuple[str, ...]]) -> set[int]:
    indexes: set[int] = set()
    for start, end, _phrase in find_phrase_windows(tokens, phrases):
        indexes.update(range(start, end))
    return indexes


def _token_classes(tokens: list[str]) -> list[str]:
    no_indexes = _window_indexes(tokens, NO_WINDOWS)
    yes_indexes = _window_indexes(tokens, YES_WINDOWS)
    filler_indexes = _window_indexes(tokens, FILLER_WINDOWS)
    menu_indexes = _window_indexes(tokens, set(_menu_noise_version()))
    classes: list[str] = []
    for index, token in enumerate(tokens):
        if (
            token == "no"
            and index + 1 < len(tokens)
            and is_digit_token(tokens[index + 1])
        ):
            classes.append("address")
        elif index in no_indexes:
            classes.append("no")
        elif index in yes_indexes:
            classes.append("yes")
        elif index in filler_indexes:
            classes.append("filler")
        elif index in menu_indexes:
            classes.append("menu")
        elif is_digit_token(token):
            classes.append("address")
        else:
            classes.append("address")
    return classes


def classify_tokens(text: str) -> tuple[list[str], list[str]]:
    tokens = compare_tokens(text)
    return tokens, _token_classes(tokens)


def is_address_like_token_class(class_name: str) -> bool:
    return class_name == "address"


def classify_confirmation_reply(text: str) -> ConfirmationLabel:
    tokens, classes = classify_tokens(text)
    if not tokens:
        return "unclear"
    has_no = "no" in classes
    has_yes = "yes" in classes
    address_like_count = classes.count("address")
    if has_no and address_like_count == 0:
        return "no"
    if has_yes and not has_no and address_like_count == 0:
        return "yes"
    return "unclear"


def classify_address_confirmation(text: str) -> ConfirmationLabel:
    """Local yes/no plus observed short STT aliases scoped to a read-back."""
    label = classify_confirmation_reply(text)
    if label != "unclear":
        return label
    tokens, classes = classify_tokens(text)
    aliases = {phrase_tokens(word)[0] for word in _ADDRESS_SHORT_YES}
    # Require the rest to be confirmation vocabulary. A street/place token
    # keeps the reply ambiguous so it can go through address extraction.
    if not tokens or not any(token in aliases for token in tokens):
        return "unclear"
    if any(kind not in {"yes", "no", "filler"} and token not in aliases
           for token, kind in zip(tokens, classes)):
        return "unclear"
    return "no" if "no" in classes else "yes"


# Vocabulary for answering "anything else?" only. Keeping it scoped avoids
# stripping these words out of addresses or treating item changes as bare no.
_MORE_ITEMS_WORDS = {
    token
    for phrase in (
        "میں کہہ رہا رہی رہے ہوں ہو ہیں تھا تھی آپ کو بس اتنا شکریہ",
        "کچھ اور مزید شامل کرنا کرنی کرنے کرتا کرتی کرتے چاہتا چاہتی چاہتے جاتا جاتی لوں گا گی",
        "main mein mai me keh ke raha rahi rahe hoon hun hon hu bas itna shukriya",
        "kuch aur our or mazeed shamil add karna karni karne karta karti chahta chahti chahte chahtay",
        "i do want to add anything else more thanks thank you",
        "ڈی",
    )
    for token in phrase_tokens(phrase)
}
_MORE_ITEMS_NEGATION = {phrase_tokens(word)[0] for word in ("not", "don't", "dont")}
_DONE_ADDING = {
    phrase_tokens(phrase) for phrase in (
        "بس", "بس اتنا", "اور کچھ نہیں", "نہیں بس", "نہیں شکریہ",
        "no thanks", "no more", "nothing else", "that's all",
        "bas", "bas itna", "nahi bas", "nhi bas", "aur kuch nahi",
    )
}


def classify_more_items_reply(text: str) -> ConfirmationLabel:
    """Handle short and sentence-length yes/no replies to the more-items prompt."""
    tokens, classes = classify_tokens(text)
    if tuple(tokens) in _DONE_ADDING:
        return "no"
    local = classify_address_confirmation(text)
    if local != "unclear":
        return local
    if not tokens or any(
        kind not in {"yes", "no", "filler"}
        and token not in _MORE_ITEMS_WORDS | _MORE_ITEMS_NEGATION
        for token, kind in zip(tokens, classes)
    ):
        return "unclear"
    if "no" in classes or any(token in _MORE_ITEMS_NEGATION for token in tokens):
        return "no"
    if "yes" in classes:
        return "yes"
    return "unclear"


CONFIRM_ACT = ("تصدیق", "tasdeeq", "tasdiq", "tasdik", "کنفرم", "confirm")
ORDER_FILLER = (
    "آرڈر", "اوڈر", "سوڈر", "order", "اس", "یہ", "is", "ye", "کی", "کا", "کو", "میں",
    "ہے", "ہیں", "hai", "hain", "کر", "کرو", "کرتے", "کریں", "کرنا", "kar", "karo",
    "karte", "karen", "karna", "دیں", "دو", "دیجیے", "dein", "do", "dijiye", "please", "پلیز",
)
_ORDER_QUANTITY_WORDS = (
    "zero", "sifar", "صفر", "one", "aik", "ek", "ایک", "two", "three", "teen", "تین",
    "four", "char", "چار", "five", "panch", "پانچ", "six", "chay", "چھ", "seven", "saat",
    "سات", "eight", "aath", "آٹھ", "nine", "nau", "نو", "ten", "das", "دس",
)
_ORDER_EDIT_VERBS = (
    "remove", "delete", "reduce", "subtract", "ہٹا", "نکال", "hata", "nikal", "kam", "کم",
    "add", "شامل", "replace", "change", "badal", "بدل", "کی جگہ", "کے بجائے", "ki jagah",
    "ke bajaye", "instead of",
)
OrderDecision = Literal["yes", "no", "mutation", "unclear"]


def confirm_token_is_fuzzy(token: str) -> bool:
    """Near-miss of one confirm word. Fillers and short tokens never match."""
    folded = fold_text(token)
    if len(folded) < 4:
        return False
    scores = [(fuzz.ratio(folded, fold_text(phrase)), fold_text(phrase)) for phrase in CONFIRM_ACT]
    scores.sort(reverse=True)
    best_score, best_word = scores[0]
    second = next((score for score, word in scores if word != best_word), 0)
    return best_score >= 80 and best_score - second >= 10


def classify_order_confirmation(text: str) -> OrderDecision:
    """Yes/no for the final order question. Menu, size, quantity, and edit words are mutations."""
    from app.intent import mentioned_item_ids, sizes_in_text

    tokens = compare_tokens(text)
    if not tokens:
        return "unclear"
    confirm_windows = {phrase_tokens(phrase) for phrase in CONFIRM_ACT}
    filler_windows = {phrase_tokens(phrase) for phrase in ORDER_FILLER}
    edit_windows = {phrase_tokens(phrase) for phrase in _ORDER_EDIT_VERBS}
    quantity_tokens = {phrase_tokens(word)[0] for word in _ORDER_QUANTITY_WORDS if phrase_tokens(word)}
    quantity_windows = {phrase_tokens(word) for word in _ORDER_QUANTITY_WORDS if phrase_tokens(word)}
    covered: set[int] = set()
    covered |= _window_indexes(tokens, YES_WINDOWS)
    covered |= _window_indexes(tokens, NO_WINDOWS)
    covered |= _window_indexes(tokens, confirm_windows)
    covered |= _window_indexes(tokens, filler_windows)
    covered |= _window_indexes(tokens, set(_menu_noise_version()))
    covered |= _window_indexes(tokens, quantity_windows)
    has_no = bool(_window_indexes(tokens, NO_WINDOWS))
    has_yes = bool(_window_indexes(tokens, YES_WINDOWS))
    has_confirm = bool(_window_indexes(tokens, confirm_windows))
    fuzzy_indexes = {
        index for index, token in enumerate(tokens)
        if index not in covered and confirm_token_is_fuzzy(token)
    }
    has_confirm = has_confirm or bool(fuzzy_indexes)
    has_quantity = any(is_digit_token(token) or token in quantity_tokens for token in tokens)
    has_mutation = bool(
        has_quantity
        or _window_indexes(tokens, edit_windows)
        or sizes_in_text(text)
        or mentioned_item_ids(text)
    )
    all_covered = all(index in covered or index in fuzzy_indexes for index in range(len(tokens)))
    if has_no and all_covered:
        return "no"
    if has_mutation:
        return "mutation"
    if not has_no and (has_yes or has_confirm) and (all_covered or has_confirm):
        return "yes"
    return "unclear"
