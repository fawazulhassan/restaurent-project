"""Multi-item requests and the anything-else stage, without network/audio."""

from unittest.mock import patch

import pytest

from app.dialog import _menu_item, chat_turn, build_system_prompt
from app.dialog_state import DialogStage, DialogState, ItemRequest
from app.intent import IntentType, ParsedIntent, parse_customer_intent
from app.order import Order, calculate_total, OrderStatus
from app.replies import ASK_ADDRESS_URDU, ASK_MORE_ITEMS_URDU, build_missing_size_urdu
import config
from app import llm


def session():
    return Order(), [{"role": "system", "content": build_system_prompt()}], DialogState()


def lines(order):
    return { (item.id, item.size): item.qty for item in order.items }


@pytest.mark.parametrize("text,expected", [
    ("میں ایک نان اور ایک پوری چکن کڑاہی اوڈر کرنا چاہوں گا۔", {("plain-naan", None):1, ("chicken-karahi", "full"):1}),
    ("aik nan aur aik pori chicken karahi", {("plain-naan", None):1, ("chicken-karahi", "full"):1}),
    ("aik nan aur aik pori chicken karhi", {("plain-naan", None):1, ("chicken-karahi", "full"):1}),
    ("2 naan and 3 full chicken karahi", {("plain-naan", None):2, ("chicken-karahi", "full"):3}),
    ("ایک پوری چکن کڑاہی اور دو نان", {("plain-naan", None):2, ("chicken-karahi", "full"):1}),
    ("2 garlic naan and 1 full chicken karahi", {("garlic-naan", None):2, ("chicken-karahi", "full"):1}),
    ("ایک آدھی چکن کڑاہی اور دو پوری چکن کڑاہی", {("chicken-karahi", "half"):1, ("chicken-karahi", "full"):2}),
])
def test_multi_item_requests_preserve_each_quantity_and_size(text, expected):
    order, messages, state = session()
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    fallback.assert_not_called()
    assert lines(order) == expected
    assert ASK_MORE_ITEMS_URDU in reply
    assert ASK_ADDRESS_URDU not in reply
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
    assert order.delivery_address is None
    assert not complete


def test_anything_else_loop_then_address_and_final_confirmation(monkeypatch):
    import config
    monkeypatch.setattr(config, "ADDRESS_LLM_EXTRACTION_ENABLED", True)
    order, messages, state = session()
    with (
        patch("app.dialog.resolve_ambiguous_intent") as fallback,
        patch("app.dialog.extract_address_with_llm", return_value="House 5 Kasur") as extract,
    ):
        reply, order, messages, _ = chat_turn("ایک نان اور ایک پوری چکن کڑاہی", order, messages, dialog_state=state)
        assert "سادہ نان" in reply and "چکن کڑاہی" in reply
        assert calculate_total(order) == 1450
        reply, order, messages, _ = chat_turn("haan ji", order, messages, dialog_state=state)
        assert state.stage == DialogStage.AWAITING_MORE_ITEMS
        assert "مزید" in reply
        reply, order, messages, _ = chat_turn("دو نان", order, messages, dialog_state=state)
        assert lines(order)[("plain-naan", None)] == 3
        assert ASK_MORE_ITEMS_URDU in reply
        extract.assert_not_called()
        reply, order, messages, _ = chat_turn("nhi g", order, messages, dialog_state=state)
        assert reply == ASK_ADDRESS_URDU
        assert state.stage == DialogStage.AWAITING_ADDRESS
        chat_turn("my address is House 5 Kasur", order, messages, dialog_state=state)
        extract.assert_called_once()
        assert order.delivery_address is None
        _, order, messages, complete = chat_turn("ہاں جی", order, messages, dialog_state=state)
        assert order.delivery_address == "House 5 Kasur" and not complete
        _, order, messages, complete = chat_turn("haan confirm kar do", order, messages, dialog_state=state)
        assert complete and order.status == OrderStatus.CONFIRMED
        fallback.assert_not_called()


def test_customer_can_name_another_item_without_a_separate_yes():
    order, messages, state = session()
    chat_turn("ایک نان", order, messages, dialog_state=state)
    reply, order, messages, _ = chat_turn("ایک پوری چکن کڑاہی بھی شامل کریں", order, messages, dialog_state=state)
    assert len(order.items) == 2
    assert ASK_MORE_ITEMS_URDU in reply


def test_missing_size_keeps_all_requested_items_until_answered():
    order, messages, state = session()
    reply, order, messages, _ = chat_turn("دو نان اور تین چکن کڑاہی", order, messages, dialog_state=state)
    assert state.stage == DialogStage.AWAITING_SIZE
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))
    assert not order.items  # staged; no partial/duplicate additions
    assert len(state.pending_additions) == 2
    reply, order, messages, _ = chat_turn("آدھی", order, messages, dialog_state=state)
    assert lines(order) == {("plain-naan", None):2, ("chicken-karahi", "half"):3}
    assert not state.pending_additions
    assert ASK_MORE_ITEMS_URDU in reply


def test_two_missing_sizes_are_requested_without_losing_either_item():
    order, messages, state = session()
    chat_turn("دو چکن کڑاہی اور تین مٹن کڑاہی", order, messages, dialog_state=state)
    reply, order, messages, _ = chat_turn("آدھی", order, messages, dialog_state=state)
    assert state.stage == DialogStage.AWAITING_SIZE
    assert state.pending_item_id == "mutton-karahi"
    assert reply == build_missing_size_urdu(_menu_item("mutton-karahi"))
    chat_turn("پوری", order, messages, dialog_state=state)
    assert lines(order) == {("chicken-karahi", "half"):2, ("mutton-karahi", "full"):3}


def test_invalid_llm_batch_cannot_partially_mutate_order(monkeypatch):
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = session()
    resolved = ParsedIntent(intent=IntentType.ADD_ITEMS, confidence=0.99, add_items=[
        ItemRequest(item_id="plain-naan", quantity=2),
        ItemRequest(item_id="invented-item", quantity=1),
    ])
    with patch("app.dialog.resolve_ambiguous_intent", return_value=resolved):
        chat_turn("unclear food request", order, messages, dialog_state=state)
    assert not order.items


def test_generic_naan_alias_does_not_duplicate_garlic_naan():
    parsed = parse_customer_intent("ایک لہسن والا نان", DialogState(), Order())
    assert parsed.intent == IntentType.ADD_ITEM and parsed.item_id == "garlic-naan"


@pytest.mark.parametrize("done", [
    "nhi", "جی نہیں", "نہیں بس", "بس اتنا", "اور کچھ نہیں", "no thanks", "nothing else", "nhi bas",
    "ڈی نہیں۔",
    "جی نہیں میں کچھ اور شامل نہیں کرنا جاتا ہوں۔",
    "میں کہہ رہا ہوں میں کچھ اور شامل نہیں کرنا جاتا۔",
    "جی نہیں میں کچھ اور شامل نہیں کرنا جاتا۔",
    "نہیں میں کچھ اور شامل نہیں کرنا جاتا۔",
    "جی نہیں، میں کچھ اور شامل نہیں کرنا چاہتا ہوں۔",
    "nahi main kuch aur shamil nahi karna chahta hun",
    "I don't want to add anything else",
])
def test_done_adding_goes_to_address_without_llm(done):
    order, messages, state = session()
    chat_turn("ایک نان", order, messages, dialog_state=state)
    with patch("app.dialog.resolve_ambiguous_intent") as fallback:
        reply, order, messages, complete = chat_turn(done, order, messages, dialog_state=state)
    fallback.assert_not_called()
    assert reply == ASK_ADDRESS_URDU
    assert state.stage == DialogStage.AWAITING_ADDRESS
    assert not complete


def test_yes_then_changed_mind_no_still_goes_to_address():
    order, messages, state = session()
    chat_turn("ایک نان", order, messages, dialog_state=state)
    chat_turn("yes", order, messages, dialog_state=state)
    reply, order, messages, _ = chat_turn("no", order, messages, dialog_state=state)
    assert reply == ASK_ADDRESS_URDU
    assert state.stage == DialogStage.AWAITING_ADDRESS


def test_partial_llm_answer_cannot_silently_drop_other_requested_items(monkeypatch):
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = session()
    result = ParsedIntent(intent=IntentType.ADD_ITEM, item_id="plain-naan", quantity=1, confidence=0.99)
    with patch("app.dialog.resolve_ambiguous_intent", return_value=result):
        reply, order, messages, complete = chat_turn("ایک نان اور کوئی دوسری چیز", order, messages, dialog_state=state)
    assert not order.items
    assert "تمام آئٹمز" in reply
    assert not complete


def test_done_after_adding_to_addressed_order_repeats_final_summary():
    order, messages, state = session()
    order.delivery_address = "House 5 Kasur"
    chat_turn("ایک نان", order, messages, dialog_state=state)
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
    reply, order, messages, complete = chat_turn("no thanks", order, messages, dialog_state=state)
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented
    assert "House 5 Kasur" in reply
    assert not complete


def test_llm_resolver_preserves_a_batch_and_normalizes_missing_sizes():
    result = llm.AssistantResult(tool_calls=[llm.ToolCall("1", "resolve_order_intent", {
        "intent": "add_items", "confidence": 0.98,
        "add_items": [
            {"item_id": "plain-naan", "quantity": 2, "size_id": ""},
            {"item_id": "chicken-karahi", "quantity": 1, "size_id": "full"},
        ],
    })])
    with patch("app.llm._call_openai", return_value=result):
        parsed = llm.resolve_ambiguous_intent("bread and karahi", Order(), DialogState())
    assert parsed.intent == IntentType.ADD_ITEMS
    assert [(item.item_id, item.quantity, item.size_id) for item in parsed.add_items] == [
        ("plain-naan", 2, None), ("chicken-karahi", 1, "full"),
    ]


def test_malformed_llm_batch_is_a_service_error():
    result = llm.AssistantResult(tool_calls=[llm.ToolCall("1", "resolve_order_intent", {
        "intent": "add_items", "confidence": 0.98, "add_items": 3,
    })])
    with patch("app.llm._call_openai", return_value=result):
        with pytest.raises(llm.LLMMalformedResponseError):
            llm.resolve_ambiguous_intent("bread and karahi", Order(), DialogState())


def test_basket_readback_uses_order_is_wording():
    order, messages, state = session()
    reply, order, messages, complete = chat_turn("ایک نان اور ایک پوری چکن کڑاہی", order, messages, dialog_state=state)
    assert reply.startswith("آپ کا آرڈر یہ ہے۔")
    assert "اپ ڈیٹ" not in reply
    assert "سادہ نان" in reply and "چکن کڑاہی" in reply
    assert ASK_MORE_ITEMS_URDU in reply
    assert not complete


def test_long_refusal_after_combined_order_moves_to_address_locally():
    order, messages, state = session()
    chat_turn("ایک نان اور ایک پوری چکن کڑاہی", order, messages, dialog_state=state)
    before = order.model_dump()
    with (
        patch("app.dialog.resolve_ambiguous_intent") as general,
        patch("app.dialog.extract_address_with_llm") as address,
    ):
        reply, order, messages, complete = chat_turn(
            "جی نہیں میں کچھ اور شامل نہیں کرنا جاتا ہوں۔", order, messages, dialog_state=state
        )
    general.assert_not_called()
    address.assert_not_called()
    assert order.model_dump() == before
    assert state.last_path == "deterministic"
    assert state.stage == DialogStage.AWAITING_ADDRESS
    assert reply == ASK_ADDRESS_URDU
    assert not complete


def test_no_with_new_item_does_not_mean_finished_adding():
    order, messages, state = session()
    chat_turn("ایک پوری چکن کڑاہی", order, messages, dialog_state=state)
    with patch("app.dialog.resolve_ambiguous_intent") as general:
        reply, order, messages, _ = chat_turn("نہیں، دو نان بھی شامل کریں", order, messages, dialog_state=state)
    general.assert_not_called()
    assert lines(order)[("plain-naan", None)] == 2
    assert ASK_MORE_ITEMS_URDU in reply
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
