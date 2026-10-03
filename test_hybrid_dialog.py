"""Offline integration tests for the deterministic-first hybrid dialog."""

from __future__ import annotations

from unittest.mock import patch

import config
from app.dialog import build_system_prompt, chat_turn
from app.dialog_state import DialogStage, DialogState
from app.intent import IntentType, ParsedIntent
from app.llm import (
    RESOLVE_ORDER_INTENT_TOOL,
    build_compact_fallback_messages,
    estimate_fallback_tokens,
)
from app.numbers_urdu import number_to_urdu_words
from app.order import Order, add_item_to_order, calculate_total, change_item_size


def _session():
    return (
        Order(),
        [{"role": "system", "content": build_system_prompt()}],
        DialogState(),
    )


def test_manual_acceptance_conversation_never_calls_ollama() -> None:
    order, messages, state = _session()
    turns = [
        "mujhe 2 full chicken karahi chahiye",
        "quantity 1 kar do",
        "ek garlic naan add kar do",
        "garlic naan remove kar do",
        "mera address House 45 Model Town Kasur hai",
        "haan confirm kar do",
    ]
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        for turn in turns:
            reply, order, messages, complete = chat_turn(
                turn, order, messages, dialog_state=state
            )
            assert reply
            assert state.last_path == "deterministic"
            assert state.last_llm_seconds == 0

    fallback.assert_not_called()
    assert complete
    assert order.delivery_address == "House 45 Model Town Kasur"
    assert [(item.id, item.size, item.qty) for item in order.items] == [
        ("chicken-karahi", "full", 1)
    ]
    assert calculate_total(order) == 1400


def test_address_is_stored_before_confirmation_is_requested() -> None:
    order, messages, state = _session()
    _, order, messages, _ = chat_turn(
        "ek garlic naan add kar do", order, messages, dialog_state=state
    )
    reply, order, messages, complete = chat_turn(
        "mera address House 45 Model Town Kasur hai",
        order,
        messages,
        dialog_state=state,
    )
    assert not complete
    assert order.delivery_address == "House 45 Model Town Kasur"
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented
    assert "House 45 Model Town Kasur" in reply


def test_add_and_confirm_same_turn_requires_new_confirmation() -> None:
    order, messages, state = _session()
    reply, order, messages, complete = chat_turn(
        "ek garlic naan add kar do aur confirm",
        order,
        messages,
        dialog_state=state,
    )
    assert not complete
    assert len(order.items) == 1
    assert order.delivery_address is None
    assert not state.summary_presented
    assert reply


def test_ambiguous_turn_calls_local_fallback_exactly_once() -> None:
    order, messages, state = _session()
    resolved = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=0.95,
        item_id="garlic-naan",
        quantity=1,
        available=True,
    )
    with patch("app.dialog.resolve_ambiguous_intent", return_value=resolved) as fallback:
        reply, order, messages, complete = chat_turn(
            "give me the bread one", order, messages, dialog_state=state
        )
    fallback.assert_called_once()
    assert state.last_path == f"{config.LLM_PROVIDER}_fallback"
    assert len(order.items) == 1
    assert order.items[0].id == "garlic-naan"
    assert not complete
    assert reply


def test_invalid_fallback_mutation_preserves_order() -> None:
    order, messages, state = _session()
    resolved = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=0.95,
        item_id="invented-burger",
        quantity=1,
    )
    with patch("app.dialog.resolve_ambiguous_intent", return_value=resolved):
        chat_turn("something ambiguous", order, messages, dialog_state=state)
    assert order.items == []


def test_fallback_mutation_rebuilds_summary_with_verified_state() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "Model Town Kasur"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    resolved = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=0.9,
        item_id="garlic-naan",
        quantity=1,
        available=True,
    )
    with patch("app.dialog.resolve_ambiguous_intent", return_value=resolved):
        reply, order, messages, complete = chat_turn(
            "another bread please", order, messages, dialog_state=state
        )
    assert not complete
    assert len(order.items) == 2
    assert state.summary_presented
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert number_to_urdu_words(calculate_total(order)) in reply


def test_atomic_size_change_leaves_original_on_failure() -> None:
    order = Order()
    add_item_to_order(order, "chicken-karahi", "half", 2)
    before = order.model_dump(mode="json")
    try:
        change_item_size(order, "chicken-karahi", "half", "invalid")
    except Exception:
        pass
    assert order.model_dump(mode="json") == before


def test_compact_prompt_has_256_token_headroom() -> None:
    messages = build_compact_fallback_messages(
        "please resolve this ambiguous restaurant request",
        Order(),
        DialogState(),
    )
    assert estimate_fallback_tokens(messages) <= 2048 - 256


def test_compact_prompt_contract_and_distinct_tool_schema() -> None:
    original = "میرا پتہ House 45 Model Town ہے"
    messages = build_compact_fallback_messages(original, Order(), DialogState())
    payload = messages[1]["content"]
    assert "STATE:" in payload
    assert "MENU:" in payload
    assert "CUSTOMER_NORMALIZED:" in payload
    assert f"CUSTOMER_ORIGINAL:\n{original}" in payload
    tool = RESOLVE_ORDER_INTENT_TOOL["function"]
    assert tool["name"] == "resolve_order_intent"
    assert tool["name"] != "update_order"
    properties = tool["parameters"]["properties"]
    assert properties["intent"]["type"] == "string"
    assert properties["quantity"]["type"] == "integer"
    assert properties["confidence"]["type"] == "number"


def test_size_follow_up_adds_pending_item_without_ollama() -> None:
    order, messages, state = _session()

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک چکن تکہ پیزا", order, messages, dialog_state=state
        )
        assert "سائز" in reply
        assert state.stage == DialogStage.AWAITING_SIZE
        assert state.pending_item_id == "chicken-tikka-pizza"

        reply, order, messages, complete = chat_turn(
            "چھوٹا", order, messages, dialog_state=state
        )

    fallback.assert_not_called()
    assert not complete
    assert len(order.items) == 1
    assert order.items[0].id == "chicken-tikka-pizza"
    assert order.items[0].size == "small"
    assert state.pending_item_id is None
    assert state.stage == DialogStage.AWAITING_ADDRESS


def test_unrecognized_short_size_reply_reprompts_without_ollama() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن بریانی", order, messages, dialog_state=state)

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک نامعلوم لفظ", order, messages, dialog_state=state
        )

    fallback.assert_not_called()
    assert "سائز" in reply
    assert not order.items
    assert not complete
    assert state.size_attempts == 1


def test_second_unrecognized_size_reply_calls_llm_once() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    chat_turn("نامعلوم", order, messages, dialog_state=state)
    assert state.size_attempts == 1

    resolved = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=0.95,
        item_id="chicken-karahi",
        size_id="half",
        quantity=1,
        available=True,
    )
    with patch(
        "app.dialog.resolve_ambiguous_intent", return_value=resolved
    ) as fallback:
        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )

    fallback.assert_called_once()
    assert not complete
    assert len(order.items) == 1
    assert order.items[0].id == "chicken-karahi"
    assert order.items[0].size == "half"
    assert state.pending_item_id is None
    assert state.size_attempts == 0
    assert "آئٹم" in reply or "آرڈر" in reply or "پتہ" in reply


def test_three_size_misses_produce_help_reply() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    assert state.size_attempts == 0

    with patch(
        "app.dialog.resolve_ambiguous_intent",
        return_value=ParsedIntent(
            intent=IntentType.AMBIGUOUS, confidence=0.1
        ),
    ) as fallback:
        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
        assert state.size_attempts == 1
        assert fallback.call_count == 0

        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
        assert state.size_attempts == 2
        assert fallback.call_count == 1
        assert "سائز" in reply

        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
        assert state.size_attempts == 3
        assert fallback.call_count == 2

    assert "صرف ایک لفظ" in reply
    assert "آدھی" in reply
    assert "پوری" in reply
    assert "چھوٹا" not in reply
    assert "فیملی پیک" not in reply
    assert not order.items
    assert not complete


def test_same_item_preserves_then_increments_new_item_resets() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    chat_turn("نامعلوم", order, messages, dialog_state=state)
    assert state.size_attempts == 1
    assert state.pending_item_id == "chicken-karahi"

    # Same item again: await_size does not reset; _reask_size increments once.
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    assert state.size_attempts == 2
    assert state.pending_item_id == "chicken-karahi"

    # Different sized item: await_size resets the counter.
    chat_turn("ایک چکن بریانی", order, messages, dialog_state=state)
    assert state.size_attempts == 0
    assert state.pending_item_id == "chicken-biryani"


def test_whisper_aadi_adds_half_karahi_deterministically() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "میں چکن کڑاہی، آدی آرڈر کرنا چاہوں گا",
            order,
            messages,
            dialog_state=state,
        )
    fallback.assert_not_called()
    assert state.last_path == "deterministic"
    assert len(order.items) == 1
    assert order.items[0].id == "chicken-karahi"
    assert order.items[0].size == "half"
    assert not complete
    assert reply


def test_cold_drink_follow_up_is_not_saved_as_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-tikka-pizza", "small", 1)
    state.sync_stage(order)

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک کولگرنگ بھی کرتے ہیں", order, messages, dialog_state=state
        )

    fallback.assert_not_called()
    assert not complete
    assert [item.id for item in order.items] == [
        "chicken-tikka-pizza",
        "coke-1-5l",
    ]
    assert order.delivery_address is None
    assert state.stage == DialogStage.AWAITING_ADDRESS


def test_correction_adds_drink_and_clears_mistaken_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-tikka-pizza", "small", 1)
    order.delivery_address = "ایک کولگرنگ بھی کرتے ہیں"
    state.sync_stage(order)
    state.summary_presented = True

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک آرڈر میں کالڈرنگ ایڈ کریں، میرا پتہ یہ نہیں",
            order,
            messages,
            dialog_state=state,
        )

    fallback.assert_not_called()
    assert not complete
    assert any(item.id == "coke-1-5l" for item in order.items)
    assert order.delivery_address is None
    assert state.stage == DialogStage.AWAITING_ADDRESS


def test_unknown_add_request_during_address_stage_is_not_an_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.sync_stage(order)

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک فو بار بھی ایڈ کریں", order, messages, dialog_state=state
        )

    fallback.assert_not_called()
    assert "پتہ محفوظ نہیں" in reply
    assert order.delivery_address is None
    assert not complete
