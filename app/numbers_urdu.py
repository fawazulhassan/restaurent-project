"""Convert integer quantities and prices to words for Urdu speech output."""

from __future__ import annotations


_UNDER_100 = (
    "صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو",
    "دس", "گیارہ", "بارہ", "تیرہ", "چودہ", "پندرہ", "سولہ", "سترہ", "اٹھارہ", "انیس",
    "بیس", "اکیس", "بائیس", "تئیس", "چوبیس", "پچیس", "چھبیس", "ستائیس", "اٹھائیس", "انتیس",
    "تیس", "اکتیس", "بتیس", "تینتیس", "چونتیس", "پینتیس", "چھتیس", "سینتیس", "اڑتیس", "انتالیس",
    "چالیس", "اکتالیس", "بیالیس", "تینتالیس", "چوالیس", "پینتالیس", "چھیالیس", "سینتالیس", "اڑتالیس", "انچاس",
    "پچاس", "اکیاون", "باون", "ترپن", "چَوّن", "پچپن", "چھپن", "ستاون", "اٹھاون", "انسٹھ",
    "ساٹھ", "اکسٹھ", "باسٹھ", "تریسٹھ", "چونسٹھ", "پینسٹھ", "چھیاسٹھ", "سڑسٹھ", "اڑسٹھ", "انہتر",
    "ستر", "اکہتر", "بہتر", "تہتر", "چوہتر", "پچہتر", "چھہتر", "ستتر", "اٹھہتر", "اناسی",
    "اسی", "اکیاسی", "بیاسی", "تراسی", "چوراسی", "پچاسی", "چھیاسی", "ستاسی", "اٹھاسی", "نواسی",
    "نوے", "اکانوے", "بانوے", "ترانوے", "چورانوے", "پچانوے", "چھیانوے", "ستانوے", "اٹھانوے", "ننانوے",
)


def number_to_urdu_words(value: int) -> str:
    """Return a non-negative integer as Urdu words for clearer TTS."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value < 0:
        raise ValueError("value must not be negative")
    if value < 100:
        return _UNDER_100[value]

    for unit, label in (
        (10_000_000, "کروڑ"),
        (100_000, "لاکھ"),
        (1_000, "ہزار"),
        (100, "سو"),
    ):
        if value >= unit:
            leading, remainder = divmod(value, unit)
            result = f"{number_to_urdu_words(leading)} {label}"
            if remainder:
                result += f" {number_to_urdu_words(remainder)}"
            return result

    raise AssertionError("unreachable")
