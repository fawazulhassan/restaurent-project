"""Fast offline tests for deterministic restaurant intent parsing."""

from __future__ import annotations

import json
from pathlib import Path

import config
from app.dialog_state import DialogStage, DialogState
from app.intent import (
    IntentType,
    _SIZE_ALIASES,
    _fold_size_text,
    _normalize_alias,
    normalize_customer_text,
    parse_customer_intent,
)
from app.order import Order, add_item_to_order, save_order


def test_normalization_returns_normalized_and_untouched_original() -> None:
    original = "  میرا پتہ House 45، Model Town ہے  "
    normalized, returned_original = normalize_customer_text(original)
    assert returned_original == original
    assert "house 45" in normalized


def test_add_item_quantity_and_size() -> None:
    parsed = parse_customer_intent(
        "mujhe 2 full chicken karahi chahiye", DialogState(), Order()
    )
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "chicken-karahi"
    assert parsed.size_id == "full"
    assert parsed.quantity == 2


def test_remove_outranks_add_words() -> None:
    order = Order()
    add_item_to_order(order, "garlic-naan", None, 1)
    parsed = parse_customer_intent(
        "garlic naan remove kar do", DialogState(last_item_id="garlic-naan"), order
    )
    assert parsed.intent == IntentType.REMOVE_ITEM
    assert parsed.item_id == "garlic-naan"


def test_quantity_change_uses_last_item() -> None:
    order = Order()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    state = DialogState(last_item_id="chicken-karahi", last_item_size="full")
    parsed = parse_customer_intent("quantity 1 kar do", state, order)
    assert parsed.intent == IntentType.SET_QUANTITY
    assert parsed.item_id == "chicken-karahi"
    assert parsed.size_id == "full"
    assert parsed.quantity == 1


def test_confirm_phrase_does_not_parse_do_as_quantity_two() -> None:
    parsed = parse_customer_intent("haan confirm kar do", DialogState(), Order())
    assert parsed.intent == IntentType.CONFIRM_ORDER
    assert parsed.quantity is None


def test_prefixed_address_uses_original_spelling() -> None:
    original = "mera address House 45 Model Town Kasur hai"
    parsed = parse_customer_intent(original, DialogState(), Order())
    assert parsed.intent == IntentType.SET_ADDRESS
    assert parsed.address == "House 45 Model Town Kasur"


def test_house_number_is_address_while_awaiting_address() -> None:
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    parsed = parse_customer_intent("House 45 Model Town", state, Order())
    assert parsed.intent == IntentType.SET_ADDRESS
    assert parsed.address == "House 45 Model Town"


def test_yes_is_not_saved_as_address() -> None:
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    parsed = parse_customer_intent("yes", state, Order())
    assert parsed.intent == IntentType.AMBIGUOUS


def test_unavailable_item_is_recognized_without_llm() -> None:
    parsed = parse_customer_intent(
        "test unavailable item order", DialogState(), Order()
    )
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "test-unavailable-item"
    assert parsed.available is False


def test_unknown_known_food_word_is_deterministic() -> None:
    parsed = parse_customer_intent("3 zinger burger", DialogState(), Order())
    assert parsed.intent == IntentType.UNKNOWN_ITEM


def test_observed_whisper_menu_mistake_is_deterministic() -> None:
    parsed = parse_customer_intent(
        "آپ کے نانیوں میں کیا جید ہے؟", DialogState(), Order()
    )
    assert parsed.intent == IntentType.LIST_MENU


def test_fuzzy_matching_uses_90_and_10_rule(monkeypatch) -> None:
    monkeypatch.setattr(config, "HYBRID_FUZZY_MATCH_ENABLED", True)
    monkeypatch.setattr(config, "HYBRID_FUZZY_MIN_SCORE", 90.0)
    monkeypatch.setattr(config, "HYBRID_FUZZY_MIN_GAP", 10.0)
    parsed = parse_customer_intent("garlic nan add", DialogState(), Order())
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "garlic-naan"


def test_fuzzy_disabled_falls_through_to_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(config, "HYBRID_FUZZY_MATCH_ENABLED", False)
    parsed = parse_customer_intent("garlic nan add", DialogState(), Order())
    assert parsed.intent == IntentType.AMBIGUOUS


def test_size_only_answer_uses_pending_item() -> None:
    state = DialogState(
        stage=DialogStage.AWAITING_SIZE,
        pending_item_id="chicken-tikka-pizza",
        pending_quantity=1,
    )
    parsed = parse_customer_intent("چھوٹا", state, Order())
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "chicken-tikka-pizza"
    assert parsed.size_id == "small"
    assert parsed.quantity == 1


def test_whisper_plate_variant_uses_pending_item() -> None:
    state = DialogState(
        stage=DialogStage.AWAITING_SIZE,
        pending_item_id="chicken-biryani",
        pending_quantity=2,
    )
    parsed = parse_customer_intent("ایک پلیڈ", state, Order())
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "chicken-biryani"
    assert parsed.size_id == "single"
    assert parsed.quantity == 2


def test_misheard_cold_drink_is_item_not_address() -> None:
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    parsed = parse_customer_intent("ایک کولگرنگ بھی کرتے ہیں", state, Order())
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "coke-1-5l"
    assert parsed.address is None


def test_address_denial_is_never_extracted_as_address() -> None:
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    parsed = parse_customer_intent("میرا پتہ یہ نہیں", state, Order())
    assert parsed.intent == IntentType.AMBIGUOUS
    assert parsed.clarification_reason == "address_denied"
    assert parsed.clear_address


def test_item_with_address_denial_adds_item_and_marks_address_for_clear() -> None:
    parsed = parse_customer_intent(
        "ایک آرڈر میں کالڈرنگ ایڈ کریں، میرا پتہ یہ نہیں",
        DialogState(stage=DialogStage.AWAITING_CONFIRMATION),
        Order(delivery_address="غلط پتہ"),
    )
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "coke-1-5l"
    assert parsed.clear_address


def test_unrecognized_order_request_is_not_saved_as_address() -> None:
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    parsed = parse_customer_intent(
        "ایک فو بار بھی ایڈ کریں", state, Order()
    )
    assert parsed.intent == IntentType.AMBIGUOUS
    assert parsed.clarification_reason == "order_while_awaiting_address"
    assert parsed.address is None


def test_fold_size_text_collapses_whisper_variants() -> None:
    assert _fold_size_text("آدھی") == "ادی"
    assert _fold_size_text("آدی") == "ادی"
    assert _fold_size_text("ادھی") == "ادی"
    assert _fold_size_text("adi") == "adi"
    assert _fold_size_text("half") == "half"


def test_no_two_size_ids_share_a_folded_alias() -> None:
    seen: dict[str, str] = {}
    for size_id, aliases in _SIZE_ALIASES.items():
        for alias in aliases:
            folded = _fold_size_text(_normalize_alias(alias))
            if folded in seen:
                assert seen[folded] == size_id, (
                    f"folded alias {folded!r} maps to both "
                    f"{seen[folded]!r} and {size_id!r}"
                )
            else:
                seen[folded] = size_id


def test_whisper_aadi_variants_parse_as_half() -> None:
    state = DialogState(
        stage=DialogStage.AWAITING_SIZE,
        pending_item_id="chicken-karahi",
        pending_quantity=1,
    )
    for text in ("آدی", "ادھی", "aadi", "آدھی", "آدھا"):
        parsed = parse_customer_intent(text, state, Order())
        assert parsed.intent == IntentType.ADD_ITEM, text
        assert parsed.item_id == "chicken-karahi", text
        assert parsed.size_id == "half", text


def test_poora_parses_as_full() -> None:
    state = DialogState(
        stage=DialogStage.AWAITING_SIZE,
        pending_item_id="chicken-karahi",
        pending_quantity=1,
    )
    parsed = parse_customer_intent("پورا", state, Order())
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.size_id == "full"


def test_full_sentence_with_whisper_aadi_adds_half() -> None:
    parsed = parse_customer_intent(
        "میں چکن کڑاہی، آدی آرڈر کرنا چاہوں گا",
        DialogState(),
        Order(),
    )
    assert parsed.intent == IntentType.ADD_ITEM
    assert parsed.item_id == "chicken-karahi"
    assert parsed.size_id == "half"


def test_address_with_heh_keeps_original_spelling() -> None:
    original = "میرا پتہ شہرِ لاہور، گلی نمبر ھائی ہے"
    parsed = parse_customer_intent(original, DialogState(), Order())
    assert parsed.intent == IntentType.SET_ADDRESS
    assert parsed.address is not None
    assert "ھائی" in parsed.address


def test_address_with_alef_madda_keeps_original_spelling(tmp_path: Path) -> None:
    from app.dialog import build_system_prompt, chat_turn

    order = Order()
    add_item_to_order(order, "garlic-naan", None, 1)
    state = DialogState(stage=DialogStage.AWAITING_ADDRESS)
    state.sync_stage(order)
    messages = [{"role": "system", "content": build_system_prompt()}]

    reply, order, messages, complete = chat_turn(
        "میرا پتہ آرام باغ، قصور ہے",
        order,
        messages,
        dialog_state=state,
    )
    assert order.delivery_address == "آرام باغ، قصور"
    assert "آرام" in order.delivery_address
    assert "ارام" not in order.delivery_address.replace("آرام", "")
    path = save_order(order, orders_dir=tmp_path)
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert saved["delivery_address"] == "آرام باغ، قصور"
    assert not complete
    assert reply
