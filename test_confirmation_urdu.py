from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.confirmation_urdu import (
    classify_address_confirmation,
    classify_more_items_reply,
    classify_confirmation_reply,
    load_menu_noise,
    reset_menu_noise,
)


@pytest.fixture(autouse=True)
def _reset_menu_noise():
    reset_menu_noise()
    yield
    reset_menu_noise()


def _load_test_menu(extra_aliases: list[str] | None = None) -> None:
    aliases = ["naan", "نان", "coke", "کوک", *(extra_aliases or [])]
    load_menu_noise(
        {
            "categories": [
                {
                    "items": [
                        {
                            "name": "Chicken Karahi",
                            "name_urdu": "چکن کڑاہی",
                            "spoken_name_urdu": "چکن کڑاہی",
                            "aliases": ["karahi", "کڑاہی"],
                        },
                        {
                            "name": "Chicken Biryani",
                            "name_urdu": "چکن بریانی",
                            "spoken_name_urdu": "چکن بریانی",
                            "aliases": ["biryani", "بریانی"],
                        },
                        {
                            "name": "Naan",
                            "name_urdu": "نان",
                            "spoken_name_urdu": "نان",
                            "aliases": aliases,
                        },
                    ]
                }
            ]
        }
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ہاں جی", "yes"),
        ("ہاں ڈی", "yes"),
        ("ہاں، ڈی۔", "yes"),
        ("ہان جی", "yes"),
        ("جی ہان", "yes"),
        ("g hn", "yes"),
        ("hn g", "yes"),
        ("g han", "yes"),
        ("g haan", "yes"),
        ("ڈی", "unclear"),
        ("ہاں ڈی نہیں", "no"),
        ("ہاں ڈی بلاک پانچ قصور", "unclear"),
        ("میں ہان جی، میرا پتہ یہ ہی ہے کوچ آمی شہید۔", "unclear"),
        ("han g", "yes"),
        ("haan ji", "yes"),
        ("حنجی", "yes"),
        ("ہنجی", "yes"),
        ("ہنجی یہی ہے۔", "yes"),
        ("ہاں جی، یہی ہے", "yes"),
        ("ہنڈی۔", "unclear"),
        ("ڈیڑھ یہی ہے۔", "unclear"),
        ("آنڈی", "yes"),
        ("ڈیڑھ لیٹر، حنجی", "yes"),
        ("نان، ہاں جی", "yes"),
        ("جی نہیں", "no"),
        ("نہیں", "no"),
        ("No", "no"),
        ("NAHI", "no"),
        ("NHI", "no"),
        ("نہیم", "no"),
        ("نی.", "no"),
        ("نہیں، میرا ڈریس یہ نہیں ہے۔", "no"),
        ("میرا ادریس یہ نہیں ہے", "no"),
        ("نہیں جی غلط ہے", "no"),
        ("نہیں نہیں", "no"),
        ("یہ میرا پتہ نہیں", "no"),
        ("نہیں، ڈیڑھ لیٹر", "no"),
        ("جی جی", "yes"),
        ("ہاں ہاں جی", "yes"),
        ("کنفرم", "yes"),
        ("بالکل", "yes"),
        ("ڈیڑھ، ڈیڑھ، ڈیڑھ", "unclear"),
        ("نان", "unclear"),
        ("کوک", "unclear"),
        ("جی ٹی روڈ پرانا لاری اڈہ", "unclear"),
    ],
)
def test_confirmation_classifier(text: str, expected: str) -> None:
    _load_test_menu()
    assert classify_confirmation_reply(text) == expected


def test_default_menu_noise_uses_static_terms_only() -> None:
    assert classify_confirmation_reply("ڈیڑھ لیٹر، حنجی") == "yes"
    assert classify_confirmation_reply("نان، ہاں جی") == "unclear"


def test_token_class_precedence_keeps_hi_as_filler() -> None:
    _load_test_menu(extra_aliases=["hi"])
    assert classify_confirmation_reply("hi haan") == "yes"


def test_real_menu_has_required_noise_keys() -> None:
    menu = json.loads(Path("data/menu.json").read_text(encoding="utf-8"))
    for category in menu.get("categories", []):
        for item in category.get("items", []):
            for key in ("name", "name_urdu", "spoken_name_urdu", "aliases"):
                assert key in item


def test_common_roman_spellings_in_both_orders_and_joined_forms() -> None:
    for words, expected in (
        ({"haan", "han", "hn", "hanh"}, "yes"),
        ({"nahi", "nahin", "nhi", "nhin", "naheen", "nahein", "nai", "ni", "na", "no", "nope"}, "no"),
    ):
        for word in words:
            assert classify_address_confirmation(word) == expected
            for ji in ("ji", "jee", "gee", "g"):
                for text in (f"{word} {ji}", f"{ji} {word}", f"{word}{ji}", f"{ji}{word}"):
                    assert classify_address_confirmation(text) == expected, text
                    assert classify_address_confirmation(text.upper() + ".") == expected, text


@pytest.mark.parametrize("text,expected", [
    ("ہنڈی۔", "yes"),
    ("ہنڈی، ہنڈی", "yes"),
    ("ہنڈی یہی ہے", "yes"),
    ("ڈی، ہاں۔", "yes"),
    ("ڈی ہان", "yes"),
    ("جی ہن", "yes"),
    ("نھیں جی", "no"),
    ("نئیں جی", "no"),
    ("نہں", "no"),
    ("ہنڈی نہیں", "no"),
    ("haanji nhi", "no"),
    ("ڈیڑھ", "unclear"),
    ("ڈی", "unclear"),
    ("ہنڈی روڈ قصور", "unclear"),
    ("g hn mera address House 5", "unclear"),
    ("nhi mera address House 5", "unclear"),
    ("House no 5 Lari Adda", "unclear"),
    ("ہنڈی چکن", "unclear"),
])
def test_address_confirmation_aliases_preserve_rejection_and_location_words(text, expected):
    assert classify_address_confirmation(text) == expected


def test_handi_alias_does_not_change_general_order_confirmation():
    assert classify_confirmation_reply("ہنڈی") == "unclear"


@pytest.mark.parametrize("text", [
    "جی نہیں میں کچھ اور شامل نہیں کرنا چاہتا ہوں",
    "میں کہہ رہا ہوں میں کچھ اور شامل نہیں کرنا جاتا",
    "ڈی نہیں",
])
def test_long_more_items_refusals_are_scoped_to_that_question(text):
    assert classify_more_items_reply(text) == "no"
    assert classify_address_confirmation(text) == "unclear"


@pytest.mark.parametrize("text", [
    "نہیں، دو نان بھی شامل کریں",
    "جی ہاں دو نان شامل کریں",
    "نہیں میرا پتہ ہے House 5 Kasur",
    "I don't want garlic naan",
])
def test_more_items_classifier_leaves_item_and_address_details_to_parser(text):
    assert classify_more_items_reply(text) == "unclear"
