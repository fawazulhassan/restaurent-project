"""Offline integration tests for the deterministic-first hybrid dialog."""

from __future__ import annotations

from unittest.mock import patch

import config
from app.confirmation_urdu import classify_order_confirmation, confirm_token_is_fuzzy
from app.dialog import _menu_item, build_system_prompt, chat_turn
from app.dialog_state import DialogStage, DialogState, ItemRequest
from app.intent import IntentType, ParsedIntent, _group_key, _near_group, _pure_group_choice, item_group_index, load_intent_menu
from app.llm import (
    RESOLVE_ORDER_INTENT_TOOL,
    UnhandledTurn,
    build_compact_fallback_messages,
    estimate_fallback_tokens,
)
from app.numbers_urdu import number_to_urdu_words
from app.order import Order, OrderStatus, add_item_to_order, calculate_total, change_item_size
from app.replies import (
    ASK_ADDRESS_URDU,
    ASK_CLARIFY_URDU,
    ASK_CONFIRM_URDU,
    ASK_MORE_ITEMS_URDU,
    ASK_ORDER_URDU,
    ASK_WHAT_TO_CHANGE_URDU,
    HUMAN_HANDOFF_URDU,
    UNAVAILABLE_ITEM_URDU,
    build_address_saved_urdu,
    build_confirm_address_urdu,
    build_item_choice_urdu,
    build_missing_size_urdu,
    build_size_help_urdu,
)


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
        "haan",
        "haan confirm kar do",
    ]
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        for turn in turns:
            reply, order, messages, complete = chat_turn(
                turn, order, messages, dialog_state=state
            )
            assert reply
            assert state.last_path in {"deterministic", "local_add", "local_remove"}
            assert state.last_llm_seconds == 0

    fallback.assert_not_called()
    assert complete
    assert order.delivery_address == "House 45 Model Town Kasur"
    assert [(item.id, item.size, item.qty) for item in order.items] == [
        ("chicken-karahi", "full", 1)
    ]
    assert calculate_total(order) == 1400


def test_address_is_pending_before_confirmation_is_requested() -> None:
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
    assert order.delivery_address is None
    assert state.pending_address == "House 45 Model Town Kasur"
    assert state.stage == DialogStage.AWAITING_ADDRESS_CONFIRMATION
    assert not state.summary_presented
    assert reply == build_confirm_address_urdu("House 45 Model Town Kasur")

    reply, order, messages, complete = chat_turn(
        "جی",
        order,
        messages,
        dialog_state=state,
    )
    assert not complete
    assert order.delivery_address == "House 45 Model Town Kasur"
    assert state.pending_address is None
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented


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


def test_ambiguous_turn_calls_turn_handler_exactly_once() -> None:
    order, messages, state = _session()
    decision = UnhandledTurn(
        action="add_items",
        items=[{"item_id": "garlic-naan", "size_id": None, "quantity": 1}],
        validation="ok",
    )
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision) as handler:
        reply, order, messages, complete = chat_turn(
            "give me the bread one", order, messages, dialog_state=state
        )
    handler.assert_called_once()
    assert state.last_path == "deterministic"
    assert len(order.items) == 1
    assert order.items[0].id == "garlic-naan"
    assert not complete
    assert reply


def test_fallback_address_routes_through_pending_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.sync_stage(order)
    resolved = ParsedIntent(
        intent=IntentType.SET_ADDRESS,
        confidence=0.95,
        address="House 45 Model Town",
    )
    with patch("app.dialog.resolve_ambiguous_intent", return_value=resolved):
        reply, order, messages, complete = chat_turn(
            "???", order, messages, dialog_state=state
        )
    assert not complete
    assert order.delivery_address is None
    assert state.pending_address == "House 45 Model Town"
    assert reply == build_confirm_address_urdu("House 45 Model Town")


def test_invalid_fallback_mutation_preserves_order(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
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


def test_fallback_mutation_rebuilds_summary_with_verified_state(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "Model Town Kasur"
    state.stage = DialogStage.AWAITING_MORE_ITEMS
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
    assert not state.summary_presented
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
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
        assert reply == build_missing_size_urdu(_menu_item("chicken-tikka-pizza"))
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
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS


def test_unrecognized_short_size_reply_reprompts_without_ollama() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن بریانی", order, messages, dialog_state=state)

    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(
            "ایک نامعلوم لفظ", order, messages, dialog_state=state
        )

    fallback.assert_not_called()
    assert reply == build_missing_size_urdu(_menu_item("chicken-biryani"))
    assert not order.items
    assert not complete
    assert state.size_attempts == 1


def test_second_unrecognized_size_reply_stays_on_the_script() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    chat_turn("نامعلوم", order, messages, dialog_state=state)
    assert state.size_attempts == 1

    with (
        patch("app.dialog.resolve_ambiguous_intent") as fallback,
        patch("app.dialog.resolve_unhandled_turn", return_value=UnhandledTurn(action="use_script")) as handler,
    ):
        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )

    handler.assert_called_once()
    fallback.assert_not_called()
    assert not complete
    assert order.items == []
    assert state.pending_item_id == "chicken-karahi"
    assert state.size_attempts == 2
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))


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
        assert fallback.call_count == 0
        assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))

        reply, order, messages, complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
        assert state.size_attempts == 3
        assert fallback.call_count == 0

    assert reply == build_size_help_urdu(_menu_item("chicken-karahi"))
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
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS


def test_correction_adds_drink_and_clears_mistaken_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-tikka-pizza", "small", 1)
    order.delivery_address = "ایک کولگرنگ بھی کرتے ہیں"
    state.sync_stage(order)
    state.stage = DialogStage.AWAITING_MORE_ITEMS
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
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS


def test_pending_address_blocks_final_confirmation_and_sets_stage() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.pending_address = "House 45"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    reply, order, messages, complete = chat_turn(
        "confirm", order, messages, dialog_state=state
    )
    assert not complete
    assert order.status.value != "confirmed"
    assert state.stage == DialogStage.AWAITING_ADDRESS_CONFIRMATION
    assert reply == build_confirm_address_urdu("House 45")


def test_ji_nahi_rejects_pending_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("Wrong address")
    reply, order, messages, complete = chat_turn(
        "جی نہیں", order, messages, dialog_state=state
    )
    assert not complete
    assert state.pending_address is None
    assert order.delivery_address is None
    assert state.stage == DialogStage.AWAITING_ADDRESS
    assert reply == ASK_ADDRESS_URDU


def test_three_rejections_trigger_handoff() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    for _ in range(3):
        state.await_address_confirmation("Wrong address")
        reply, order, messages, complete = chat_turn(
            "no", order, messages, dialog_state=state
        )
    assert reply == HUMAN_HANDOFF_URDU
    assert state.pending_address is None
    assert state.address_rejections == 0
    assert state.address_confirmation_attempts == 0


def test_address_confirmation_noise_does_not_replace_pending() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.sync_stage(order)

    reply, order, messages, complete = chat_turn(
        "پرانا، لاری، آدھا، قصور۔", order, messages, dialog_state=state
    )
    assert state.pending_address == "پرانا لاری آدھا قصور"
    assert order.delivery_address is None
    assert reply == build_confirm_address_urdu("پرانا لاری آدھا قصور")

    reply, order, messages, complete = chat_turn(
        "ڈیڑھ، ڈیڑھ، ڈیڑھ، ڈیڑھ۔", order, messages, dialog_state=state
    )
    assert state.pending_address == "پرانا لاری آدھا قصور"
    assert order.delivery_address is None
    assert reply == build_confirm_address_urdu("پرانا لاری آدھا قصور")


def test_noisy_yes_confirms_existing_pending_address() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("پرانا لاری آدھا قصور")

    reply, order, messages, complete = chat_turn(
        "ڈیڑھ لیٹر، حنجی۔", order, messages, dialog_state=state
    )
    assert order.delivery_address == "پرانا لاری آدھا قصور"
    assert state.pending_address is None
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented is True
    assert reply == build_address_saved_urdu(order, state)


def test_address_confirmation_correction_cleans_candidate() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("wrong address")

    reply, order, messages, complete = chat_turn(
        "نہیں، پرانا، لاری، آدھا، قصور ہے میرا، بتا۔",
        order,
        messages,
        dialog_state=state,
    )
    assert state.pending_address == "پرانا لاری آدھا قصور"
    assert order.delivery_address is None
    assert state.address_rejections == 0
    assert reply == build_confirm_address_urdu("پرانا لاری آدھا قصور")


def test_address_confirmation_bare_addresses_replace_pending() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("wrong address")

    reply, order, messages, complete = chat_turn(
        "جی ٹی روڈ پرانا لاری اڈہ", order, messages, dialog_state=state
    )
    assert state.pending_address == "جی ٹی روڈ پرانا لاری اڈہ"
    assert order.delivery_address is None
    assert reply == build_confirm_address_urdu("جی ٹی روڈ پرانا لاری اڈہ")

    reply, order, messages, complete = chat_turn(
        "House no 5 Lari Adda", order, messages, dialog_state=state
    )
    assert state.pending_address == "House no 5 Lari Adda"
    assert order.delivery_address is None
    assert reply == build_confirm_address_urdu("House no 5 Lari Adda")


def test_no_my_address_replaces_pending_without_rejection() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("wrong address")

    reply, order, messages, complete = chat_turn(
        "no my address is Purana Lari Adda Kasur",
        order,
        messages,
        dialog_state=state,
    )
    assert state.pending_address == "Purana Lari Adda Kasur"
    assert state.address_rejections == 0
    assert order.delivery_address is None


def test_reported_urdu_address_corrections_and_rejections() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("پورانا لاری گیڑہ قصور")
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        for text, expected in (
            ("نہیم پرانا لاری اڈڑا قصول۔", "پرانا لاری اڈڑا قصول"),
            ("نہیں، میرا ادریس یہ ہے پرانا لاری اڑھا قصور۔", "پرانا لاری اڑھا قصور"),
        ):
            reply, order, messages, complete = chat_turn(
                text, order, messages, dialog_state=state
            )
            assert state.pending_address == expected
            assert state.address_rejections == 0
            assert order.delivery_address is None
            assert not complete
            assert reply == build_confirm_address_urdu(expected)

        for rejection in ("نی.", "نہیں، میرا ڈریس یہ نہیں ہے۔"):
            reply, order, messages, complete = chat_turn(
                rejection, order, messages, dialog_state=state
            )
            assert state.pending_address is None
            assert state.stage == DialogStage.AWAITING_ADDRESS
            assert order.delivery_address is None
            assert not complete
            assert reply == ASK_ADDRESS_URDU
            # Start a fresh read-back before testing the next rejection.
            state.await_address_confirmation("پرانا لاری اڈہ قصور")
        fallback.assert_not_called()


def test_three_no_rejections_with_new_candidates_trigger_handoff() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)

    for candidate in ("wrong one", "wrong two", "wrong three"):
        state.await_address_confirmation(candidate)
        reply, order, messages, complete = chat_turn(
            "no", order, messages, dialog_state=state
        )
    assert reply == HUMAN_HANDOFF_URDU
    assert state.pending_address is None
    assert state.address_rejections == 0
    assert state.address_confirmation_attempts == 0


def test_three_unclear_address_replies_trigger_handoff() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("House 45")
    with patch("app.dialog.extract_address_with_llm") as resolver:
        for _ in range(3):
            reply, order, messages, complete = chat_turn(
                "?", order, messages, dialog_state=state
            )
        resolver.assert_not_called()
    assert reply == HUMAN_HANDOFF_URDU
    assert state.pending_address is None
    assert state.address_rejections == 0
    assert state.address_confirmation_attempts == 0


def test_disabled_address_fallback_and_handoff_do_not_mutate_order(monkeypatch) -> None:
    monkeypatch.setattr(config, "HYBRID_LLM_FALLBACK_ENABLED", False)
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("House 45")
    with patch("app.dialog.extract_address_with_llm") as resolver:
        for text in ("?", "?", "?", "میرا اوڈر کنفرم کرتے ہیں کلیڈ"):
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        resolver.assert_not_called()
    assert reply == HUMAN_HANDOFF_URDU
    assert state.stage == DialogStage.HANDOFF
    assert state.pending_address is None
    assert order.delivery_address is None
    assert not complete


def test_hanji_readback_confirmation_does_not_call_llm() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.await_address_confirmation("پورانا لاری اڈھا قصور")
    with patch("app.dialog.extract_address_with_llm") as resolver:
        _, order, messages, complete = chat_turn("ہنجی یہی ہے۔", order, messages, dialog_state=state)
        resolver.assert_not_called()
    assert order.delivery_address == "پورانا لاری اڈھا قصور"
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert not complete


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


def test_size_word_for_another_item_reasks_the_pending_item() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    assert state.size_attempts == 0
    reply, order, messages, complete = chat_turn("چھوٹا پیزا", order, messages, dialog_state=state)
    assert not complete
    assert not order.items
    assert state.pending_item_id == "chicken-karahi"
    assert state.size_attempts == 1
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))


def test_single_size_item_is_stored_without_a_question(monkeypatch) -> None:
    import json
    from pathlib import Path

    menu = json.loads(Path("data/menu.json").read_text(encoding="utf-8"))
    menu["categories"].append({
        "name": "Fixture",
        "items": [{
            "id": "solo-lassi",
            "name": "Solo Lassi",
            "name_urdu": "سولو لسی",
            "spoken_name_urdu": "سولو لسی",
            "aliases": ["سولو لسی", "solo lassi"],
            "sizes": [{
                "id": "regular",
                "label": "Regular",
                "label_urdu": "عام سائز",
                "price_pkr": 100,
            }],
            "available": True,
        }],
    })
    index = {
        item["id"]: item
        for category in menu["categories"]
        for item in category["items"]
    }
    monkeypatch.setattr("app.dialog.load_menu_data", lambda path=None: menu)
    monkeypatch.setattr("app.intent.load_intent_menu", lambda: menu)
    monkeypatch.setattr("app.order.load_menu", lambda path=None: index)
    order, messages, state = _session()
    reply, order, messages, complete = chat_turn("ایک سولو لسی", order, messages, dialog_state=state)
    assert not complete
    assert state.stage != DialogStage.AWAITING_SIZE
    assert state.pending_item_id is None
    line = next(item for item in order.items if item.id == "solo-lassi")
    assert line.size == "regular"
    assert line.qty == 1
    assert "چاہیے؟" not in reply


def test_marked_size_change_stays_on_the_deterministic_path() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    with patch("app.dialog.resolve_basket_edits") as editor:
        chat_turn(
            "change chicken karahi poori ki jagah adhi",
            order, messages, dialog_state=state,
        )
    editor.assert_not_called()
    line = next(item for item in order.items if item.id == "chicken-karahi")
    assert line.size == "half"
    assert line.qty == 2
    assert line.unit_price_pkr == 800


def test_confirmation_mutation_without_the_editor_does_not_confirm() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    with patch("app.dialog.resolve_basket_edits") as editor:
        reply, order, messages, complete = chat_turn(
            "ہاں اور ایک نان", order, messages, dialog_state=state,
        )
    editor.assert_not_called()
    assert not complete
    assert order.status.value != "confirmed"
    assert state.order_confirmation_attempts == 1
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert order.items[0].qty == 1


def test_group_tokens_are_the_step0_set() -> None:
    assert set(item_group_index()) == {
        "biryani", "karahi", "pizza", "بریانی", "مارگریٹا", "پیزا", "کڑاہی",
    }


def test_choice_and_size_builders_for_two_and_three_labels() -> None:
    two = {"spoken_name_urdu": "آئٹم", "sizes": [
        {"id": "a", "label_urdu": "پہلا"}, {"id": "b", "label_urdu": "دوسرا"},
    ]}
    three = {"spoken_name_urdu": "آئٹم", "sizes": [
        {"id": "a", "label_urdu": "پہلا"}, {"id": "b", "label_urdu": "دوسرا"},
        {"id": "c", "label_urdu": "تیسرا"},
    ]}
    assert build_missing_size_urdu(two) == "آئٹم پہلا چاہیے یا دوسرا؟"
    assert build_missing_size_urdu(three) == "آئٹم پہلا، دوسرا یا تیسرا چاہیے؟"
    assert build_item_choice_urdu([
        {"spoken_name_urdu": "پہلا"}, {"spoken_name_urdu": "دوسرا"},
    ]) == "پہلا چاہیے یا دوسرا؟"
    assert build_item_choice_urdu([
        {"spoken_name_urdu": "پہلا"}, {"spoken_name_urdu": "دوسرا"},
        {"spoken_name_urdu": "تیسرا"},
    ]) == "پہلا، دوسرا یا تیسرا چاہیے؟"


def test_pizza_group_asks_by_spoken_name_then_stores_the_pick() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_basket_edits") as editor:
        reply, order, messages, complete = chat_turn("ایک پیزا", order, messages, dialog_state=state)
    editor.assert_not_called()
    names = [_menu_item(item_id)["spoken_name_urdu"] for item_id in state.pending_choices]
    assert state.pending_choice_quantity == 1
    assert reply == f"{names[0]} چاہیے یا {names[1]}؟"
    reply, order, messages, complete = chat_turn("مارگریٹا", order, messages, dialog_state=state)
    assert not complete
    assert state.choice_attempts == 0
    assert state.pending_item_id == "margherita-pizza"
    assert reply == build_missing_size_urdu(_menu_item("margherita-pizza"))
    fresh, fresh_messages, fresh_state = _session()
    chat_turn("دو پیزا", fresh, fresh_messages, dialog_state=fresh_state)
    assert fresh_state.pending_choice_quantity == 2
    assert not fresh.items


def test_two_groups_are_asked_before_any_size() -> None:
    order, messages, state = _session()
    chat_turn("ایک پیزا اور ایک بریانی", order, messages, dialog_state=state)
    assert state.pending_choices
    assert state.pending_choice_queue[0].quantity == 1
    assert state.pending_item_id is None
    chat_turn("مارگریٹا", order, messages, dialog_state=state)
    assert state.pending_item_id is None
    assert any("بریانی" in (_menu_item(item_id)["spoken_name_urdu"] or "") for item_id in state.pending_choices)


def test_wrong_choice_replies_share_one_attempt_counter() -> None:
    order, messages, state = _session()
    chat_turn("ایک پیزا", order, messages, dialog_state=state)
    question = build_item_choice_urdu([_menu_item(item_id) for item_id in state.pending_choices])
    reply, order, messages, complete = chat_turn("چھوٹا", order, messages, dialog_state=state)
    assert reply == question
    assert state.choice_attempts == 1
    assert not order.items
    reply, order, messages, complete = chat_turn("نان", order, messages, dialog_state=state)
    assert reply == question
    assert state.choice_attempts == 2
    chat_turn("مارگریٹا", order, messages, dialog_state=state)
    assert state.choice_attempts == 0


def test_third_wrong_choice_drops_only_the_staged_karahi() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.pending_additions = [ItemRequest(item_id="plain-naan", quantity=1)]
    chat_turn("ایک پیزا اور ایک چکن کڑاہی", order, messages, dialog_state=state)
    assert state.staged_start_index == 1
    assert any(part.item_id == "chicken-karahi" for part in state.pending_additions)
    question = build_item_choice_urdu([_menu_item(item_id) for item_id in state.pending_choices])
    for _ in range(2):
        reply, order, messages, complete = chat_turn("نان", order, messages, dialog_state=state)
        assert reply == question
    reply, order, messages, complete = chat_turn("نان", order, messages, dialog_state=state)
    assert reply == ASK_MORE_ITEMS_URDU
    assert [part.item_id for part in state.pending_additions] == ["plain-naan"]
    assert [item.id for item in order.items] == ["plain-naan"]
    assert not state.pending_choices


def test_small_pizza_is_not_a_group_choice() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    with patch("app.dialog.resolve_basket_edits") as editor:
        chat_turn("چھوٹا پیزا", order, messages, dialog_state=state)
    editor.assert_not_called()
    assert not state.pending_choices


def test_biryani_group_and_plain_naan() -> None:
    order, messages, state = _session()
    reply, order, messages, complete = chat_turn("ایک بریانی", order, messages, dialog_state=state)
    names = {_menu_item(item_id)["spoken_name_urdu"] for item_id in state.pending_choices}
    assert names == {"چکن بریانی", "بیف بریانی"}
    assert reply == build_item_choice_urdu([_menu_item(item_id) for item_id in state.pending_choices])
    fresh, messages, fresh_state = _session()
    chat_turn("دو نان", fresh, messages, dialog_state=fresh_state)
    assert [item.id for item in fresh.items] == ["plain-naan"]
    assert fresh_state.pending_choices == []


def test_existing_size_is_reused_unless_the_customer_names_one() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-tikka-pizza", "small", 1)
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    chat_turn("ایک چکن تکہ پیزا", order, messages, dialog_state=state)
    lines = [item for item in order.items if item.id == "chicken-tikka-pizza"]
    assert len(lines) == 1 and lines[0].size == "small" and lines[0].qty == 2
    chat_turn("ایک بڑا چکن تکہ پیزا", order, messages, dialog_state=state)
    small = next(item for item in order.items if item.id == "chicken-tikka-pizza" and item.size == "small")
    spoken = next(item for item in order.items if item.id == "chicken-tikka-pizza" and item.size == "large")
    assert small.qty == 2 and spoken.qty == 1
    chat_turn("ایک چکن تکہ پیزا", order, messages, dialog_state=state)
    assert state.pending_item_id == "chicken-tikka-pizza"
    chat_turn("بڑا", order, messages, dialog_state=state)
    large = next(item for item in order.items if item.id == "chicken-tikka-pizza" and item.size == "large")
    assert large.qty == 2


def test_named_pair_does_not_call_the_basket_model() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    with patch("app.dialog.resolve_basket_edits") as editor:
        reply, order, messages, complete = chat_turn(
            "ایک چکن بریانی اور ایک مارگریٹا پیزا", order, messages, dialog_state=state,
        )
    editor.assert_not_called()
    assert state.stage == DialogStage.AWAITING_SIZE
    assert reply == build_missing_size_urdu(_menu_item(state.pending_item_id))


def test_confirmation_extra_words_need_a_confirm_word() -> None:
    assert classify_order_confirmation("ہاں جی میں نے پہلے بھی کہا ہے تصدیق کر دیں") == "yes"
    assert classify_order_confirmation("انجیس اوڈر کو کنفرم کر دیں پلیز") == "yes"
    assert classify_order_confirmation("تصدی کر دیں") == "yes"
    assert classify_order_confirmation("ہاں پتہ غلط ہے") == "unclear"
    assert classify_order_confirmation("ہاں ڈلیوری کتنی دیر میں ہوگی") == "unclear"
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    reply, order, messages, complete = chat_turn("ہاں پتہ غلط ہے", order, messages, dialog_state=state)
    assert ASK_CONFIRM_URDU in reply
    assert state.order_confirmation_attempts == 1
    assert order.status != OrderStatus.CONFIRMED
    reply, order, messages, complete = chat_turn("نہیں", order, messages, dialog_state=state)
    assert reply == ASK_WHAT_TO_CHANGE_URDU
    assert state.stage == DialogStage.AWAITING_ORDER_CHANGE


def test_confirm_fuzzy_boundaries() -> None:
    from app.text_fold import fold_text

    assert confirm_token_is_fuzzy("تصد") is False
    assert classify_order_confirmation("ہاں پلیز") == "yes"

    def almost(left: str, right: str) -> int:
        return 79 if right == fold_text("تصدیق") else 0

    with patch("app.confirmation_urdu.fuzz.ratio", side_effect=almost):
        assert confirm_token_is_fuzzy("تصدیقx") is False

    def exact_filler(left: str, right: str) -> int:
        return 100 if right == fold_text("تصدیق") else 0

    with patch("app.confirmation_urdu.fuzz.ratio", side_effect=exact_filler):
        assert confirm_token_is_fuzzy("پلیز") is True
        assert classify_order_confirmation("پلیز") == "unclear"


def test_negated_remove_and_add_leave_the_basket() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-biryani", "single", 1)
    add_item_to_order(order, "plain-naan", None, 2)
    before = [(item.id, item.qty) for item in order.items]
    for text in (
        "نہیں میں چکن بریانی نہیں کرنا چاہتا، اوڈر سے نکال دی",
        "نان نہیں ہٹانا",
        "naan mat hatao",
        "naan mat dalo",
        "don't add naan",
    ):
        basket, messages, turn_state = _session()
        add_item_to_order(basket, "chicken-biryani", "single", 1)
        add_item_to_order(basket, "plain-naan", None, 2)
        turn_state.stage = DialogStage.AWAITING_MORE_ITEMS
        with patch("app.dialog.resolve_basket_edits") as editor:
            chat_turn(text, basket, messages, dialog_state=turn_state)
        editor.assert_not_called()
        if "نکال" in text:
            assert not any(item.id == "chicken-biryani" for item in basket.items)
            assert turn_state.last_path == "local_remove"
        else:
            assert [(item.id, item.qty) for item in basket.items] == before


def test_local_change_during_confirmation_rereads_the_summary() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 2)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    state.order_confirmation_attempts = 2
    reply, order, messages, complete = chat_turn("نان ہٹا دو", order, messages, dialog_state=state)
    assert not order.items
    assert state.stage == DialogStage.ORDERING
    assert reply == ASK_ORDER_URDU
    assert state.order_confirmation_attempts == 0
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 2)
    add_item_to_order(order, "chicken-karahi", "full", 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_ORDER_CHANGE
    state.order_change_failures = 2
    reply, order, messages, complete = chat_turn("نان ہٹا دو", order, messages, dialog_state=state)
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented is True
    assert state.order_confirmation_attempts == 0
    assert state.order_change_failures == 0
    assert ASK_CONFIRM_URDU in reply
    assert not complete


def test_two_sizes_and_a_missing_item_do_not_change_stage() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-karahi", "half", 1)
    add_item_to_order(order, "chicken-karahi", "full", 1)
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    reply, order, messages, complete = chat_turn("چکن کڑاہی ختم", order, messages, dialog_state=state)
    assert reply == "براہ کرم بتائیں کون سا سائز نکالنا ہے۔"
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
    assert len([item for item in order.items if item.id == "chicken-karahi"]) == 2
    reply, order, messages, complete = chat_turn("مٹن کڑاہی ختم", order, messages, dialog_state=state)
    assert reply == "یہ آئٹم آپ کے موجودہ آرڈر میں شامل نہیں ہے۔"
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS


def test_confirmed_order_clears_pending_choice_state() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.pending_choices = ["chicken-tikka-pizza", "margherita-pizza"]
    state.choice_attempts = 2
    state.staged_start_index = 1
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    chat_turn("تصدیق کر دیں", order, messages, dialog_state=state)
    assert state.stage == DialogStage.COMPLETE
    assert state.pending_choices == []
    assert state.choice_attempts == 0
    assert state.staged_start_index == 0


def test_group_key_equates_karahi_spellings_only() -> None:
    assert _group_key("کڑاہی") == _group_key("کرائی") == "کرای"
    assert _group_key("کرایہ") != _group_key("کڑاہی")


def test_framed_pizza_and_karahi_asks_before_any_size() -> None:
    for text in (
        "میں ایک پیزا اور ایک کرائی اوڈر کرنا چاہوں گا۔",
        "میں ایک پیزا اور ایک karahi اوڈر کرنا چاہتا ہوں۔",
    ):
        order, messages, state = _session()
        with (
            patch("app.dialog.resolve_basket_edits") as editor,
            patch("app.dialog.resolve_ambiguous_intent") as fallback,
        ):
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        editor.assert_not_called()
        fallback.assert_not_called()
        assert state.last_path == "deterministic"
        assert reply == build_item_choice_urdu([_menu_item(item_id) for item_id in state.pending_choices])
        assert state.pending_choice_queue[0].quantity == 1
        assert set(state.pending_choice_queue[0].item_ids) == {"chicken-karahi", "mutton-karahi"}
        reply, order, messages, complete = chat_turn("مارگریٹا", order, messages, dialog_state=state)
        assert state.pending_item_id is None
        assert set(state.pending_choices) == {"chicken-karahi", "mutton-karahi"}
        assert reply == build_item_choice_urdu([_menu_item(item_id) for item_id in state.pending_choices])


def test_framed_size_word_is_not_a_choice(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = _session()
    unclear = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0)
    with patch("app.dialog.resolve_ambiguous_intent", return_value=unclear):
        reply, order, messages, complete = chat_turn(
            "میں ایک چھوٹا پیزا", order, messages, dialog_state=state,
        )
    assert not state.pending_choices
    assert reply != build_item_choice_urdu([
        _menu_item("chicken-tikka-pizza"), _menu_item("margherita-pizza"),
    ])


def test_negated_and_removal_sentences_ask_no_choice() -> None:
    unclear = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0)
    for text in ("میں ایک پیزا نہیں چاہتا", "میں پیزا ہٹانا چاہتا ہوں"):
        order, messages, state = _session()
        with patch("app.dialog.resolve_ambiguous_intent", return_value=unclear):
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        assert not state.pending_choices
        assert reply != build_item_choice_urdu([
            _menu_item("chicken-tikka-pizza"), _menu_item("margherita-pizza"),
        ])


def test_three_character_token_is_not_a_near_hit() -> None:
    groups = {"کرہا": ("chicken-karahi", "mutton-karahi")}
    menu = load_intent_menu()
    assert len("کرا") == 3
    assert _group_key("کرہا") == _group_key("کرا")
    assert _near_group("کرا", groups) is None
    with patch("app.intent.item_group_index", return_value=groups):
        assert _pure_group_choice("کرا", menu) is None
        assert _pure_group_choice("کرہا", menu) == (("chicken-karahi", "mutton-karahi"), 1)


def test_colliding_group_keys_block_near_hits_only() -> None:
    groups = {
        "کڑاہی": ("chicken-karahi", "mutton-karahi"),
        "کراہی": ("chicken-tikka-pizza", "margherita-pizza"),
    }
    menu = load_intent_menu()
    assert _group_key("کڑاہی") == _group_key("کراہی") == _group_key("کرائی")
    with patch("app.intent.item_group_index", return_value=groups):
        assert _pure_group_choice("کرائی", menu) is None
        assert _pure_group_choice("کڑاہی", menu) == (groups["کڑاہی"], 1)
        assert _pure_group_choice("کراہی", menu) == (groups["کراہی"], 1)


def test_whisper_unknown_items_stay_local_and_print_no_basket_edit() -> None:
    for text in (
        "میں ایک زنگر پرکر اوڈر کرنا چاہوں گا۔",
        "جنگر بڑگر اوڈر کرنا چاہتا ہوں۔",
        "3 zinger burger",
    ):
        order, messages, state = _session()
        with (
            patch("app.dialog.resolve_ambiguous_intent") as fallback,
            patch("app.dialog.resolve_basket_edits") as editor,
        ):
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        fallback.assert_not_called()
        editor.assert_not_called()
        assert reply == UNAVAILABLE_ITEM_URDU
        assert order.items == []
        assert not complete
        assert state.last_path == "deterministic"
        assert state.last_basket_edit_detail == ""


def test_catalog_unavailable_item_uses_the_same_reply() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, _complete = chat_turn(
            "test unavailable item order", order, messages, dialog_state=state
        )
    fallback.assert_not_called()
    assert reply == UNAVAILABLE_ITEM_URDU
    assert order.items == []


def test_unknown_order_does_not_interrupt_size_or_choice() -> None:
    order, messages, state = _session()
    state.stage = DialogStage.AWAITING_SIZE
    state.pending_item_id = "chicken-karahi"
    state.pending_quantity = 2
    unclear = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0)
    with patch("app.dialog.resolve_ambiguous_intent", return_value=unclear):
        reply, _order, _messages, _complete = chat_turn(
            "میں ایک زنگر پرکر اوڈر کرنا چاہوں گا۔",
            order,
            messages,
            dialog_state=state,
        )
    assert reply != UNAVAILABLE_ITEM_URDU
    assert state.pending_item_id == "chicken-karahi"
    assert state.pending_quantity == 2
    assert state.pending_choices == []

    order, messages, state = _session()
    state.pending_choices = ["chicken-karahi", "mutton-karahi"]
    state.pending_choice_quantity = 1
    reply, _order, _messages, _complete = chat_turn(
        "جنگر بڑگر اوڈر کرنا چاہتا ہوں۔",
        order,
        messages,
        dialog_state=state,
    )
    assert reply != UNAVAILABLE_ITEM_URDU
    assert state.pending_choices == ["chicken-karahi", "mutton-karahi"]
    assert state.pending_choice_quantity == 1


def test_mixed_unknown_item_asks_the_turn_handler() -> None:
    order, messages, state = _session()
    decision = UnhandledTurn(action="clarify", validation="stub")
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision) as handler:
        reply, order, messages, _complete = chat_turn(
            "ایک نان اور جنگر بڑگر اوڈر کرنا چاہتا ہوں",
            order,
            messages,
            dialog_state=state,
        )
    handler.assert_called_once()
    assert reply == ASK_CLARIFY_URDU
    assert order.items == []
    assert state.pending_additions == []
    assert state.staged_start_index == 0
    assert state.last_basket_edit_detail == ""


def test_unknown_item_during_confirmation_does_not_spend_an_attempt() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "Model Town Kasur"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    state.order_confirmation_attempts = 1
    state.order_change_failures = 1
    with (
        patch("app.dialog.resolve_ambiguous_intent") as fallback,
        patch("app.dialog.resolve_basket_edits") as editor,
    ):
        reply, order, messages, _complete = chat_turn(
            "میں ایک زنگر پرکر اوڈر کرنا چاہوں گا۔",
            order,
            messages,
            dialog_state=state,
        )
    fallback.assert_not_called()
    editor.assert_not_called()
    assert reply == UNAVAILABLE_ITEM_URDU
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.order_confirmation_attempts == 1
    assert state.order_change_failures == 1
    assert state.last_basket_edit_detail == ""
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1)]

    state.stage = DialogStage.AWAITING_ORDER_CHANGE
    with (
        patch("app.dialog.resolve_ambiguous_intent") as fallback,
        patch("app.dialog.resolve_basket_edits") as editor,
    ):
        reply, order, messages, _complete = chat_turn(
            "جنگر بڑگر اوڈر کرنا چاہتا ہوں۔",
            order,
            messages,
            dialog_state=state,
        )
    fallback.assert_not_called()
    editor.assert_not_called()
    assert reply == UNAVAILABLE_ITEM_URDU
    assert state.stage == DialogStage.AWAITING_ORDER_CHANGE
    assert state.order_confirmation_attempts == 1
    assert state.order_change_failures == 1
    assert state.last_basket_edit_detail == ""
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1)]

