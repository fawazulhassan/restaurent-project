"""Absolute quantity corrections versus incremental additions."""

from unittest.mock import patch

import pytest

from app.dialog import chat_turn, build_system_prompt
from app.dialog_state import DialogState, DialogStage
from app.intent import ParsedIntent, IntentType
from app.llm import LLMServiceError
from app.order import Order, add_item_to_order
from app.replies import ASK_QUANTITY_URDU, ASK_MORE_ITEMS_URDU


def session(qty=2):
    order = Order()
    add_item_to_order(order, "plain-naan", None, qty)
    add_item_to_order(order, "chicken-karahi", "full", 1)
    state = DialogState(stage=DialogStage.AWAITING_MORE_ITEMS,
                        last_item_id="chicken-karahi", last_item_size="full")
    return order, [{"role": "system", "content": build_system_prompt()}], state


def set_four(**overrides):
    fields = dict(intent=IntentType.SET_QUANTITY, item_id="plain-naan", quantity=4, confidence=0.99)
    fields.update(overrides)
    return ParsedIntent(**fields)


@pytest.mark.parametrize("text", [
    "ہاں جی میں نان چار کرنا چاہتا ہوں۔",
    "نہیں میں ٹوٹل نان چار کرنا چاہتا ہوں، پانچ نہیں۔",
    "ہن جی میں نان چار کرنا چاہتا ہوں، ٹوٹل میرے نان ہو جائیں گے چار۔",
    "main naan char karna chahta hun",
    "make naan total four not five",
])
def test_quantity_correction_sets_final_total_and_leaves_karahi_alone(text):
    order, messages, state = session()
    with patch("app.dialog.resolve_ambiguous_intent", return_value=set_four()) as resolve:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        assert order.items[0].qty == 4
        assert order.items[1].qty == 1
        assert not complete
        assert state.last_path == "openai_fallback"
        assert ASK_MORE_ITEMS_URDU in reply
        # Retrying the same correction must not add another four.
        chat_turn(text, order, messages, dialog_state=state)
        assert order.items[0].qty == 4
        assert resolve.call_count == 2


@pytest.mark.parametrize("resolved", [
    set_four(intent=IntentType.ADD_ITEM),
    set_four(confidence=0.5),
    set_four(item_id="chicken-karahi", size_id="full"),
    set_four(quantity=5),  # explicitly rejected number
    set_four(quantity=8),  # invented total / sum
    ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0),
    LLMServiceError("timeout"),
])
def test_bad_quantity_resolution_cannot_mutate_order(resolved):
    order, messages, state = session()
    before = order.model_dump()
    kwargs = {"side_effect": resolved} if isinstance(resolved, Exception) else {"return_value": resolved}
    with patch("app.dialog.resolve_ambiguous_intent", **kwargs):
        reply, order, messages, complete = chat_turn(
            "نہیں میں ٹوٹل نان چار کرنا چاہتا ہوں، پانچ نہیں۔", order, messages, dialog_state=state
        )
    assert order.model_dump() == before
    assert reply == ASK_QUANTITY_URDU
    assert state.quantity_correction_item_id == "plain-naan"
    assert not complete
    # The clarification is for naan, even though the last item was karahi.
    with patch("app.dialog.resolve_ambiguous_intent") as resolve:
        reply, order, messages, _ = chat_turn("چار", order, messages, dialog_state=state)
    resolve.assert_not_called()
    assert order.items[0].qty == 4 and order.items[1].qty == 1
    assert state.quantity_correction_item_id is None


def test_garbled_number_requires_clarification_instead_of_guessing_four():
    order, messages, state = session()
    before = order.model_dump()
    with patch("app.dialog.resolve_ambiguous_intent", return_value=set_four()):
        reply, order, messages, _ = chat_turn(
            "نہیں میں سادھ نان نہیں لینا چاہتا میں صرف چادھ نان لینا چاہتا ہوں۔",
            order, messages, dialog_state=state,
        )
    assert reply == ASK_QUANTITY_URDU
    assert order.model_dump() == before


@pytest.mark.parametrize("text,final", [
    ("دو نان اور شامل کریں", 4),
    ("add two more naan", 4),
    ("چار نان بھی شامل کریں", 6),
])
def test_explicit_additions_still_increment(text, final):
    order, messages, state = session()
    with patch("app.dialog.resolve_ambiguous_intent") as resolve:
        chat_turn(text, order, messages, dialog_state=state)
    resolve.assert_not_called()
    assert order.items[0].qty == final


def test_disabled_llm_asks_for_total_and_keeps_order_unchanged(monkeypatch):
    import config
    monkeypatch.setattr(config, "HYBRID_LLM_FALLBACK_ENABLED", False)
    order, messages, state = session()
    with patch("app.dialog.resolve_ambiguous_intent") as resolve:
        reply, order, messages, _ = chat_turn("ٹوٹل نان چار کرنا چاہتا ہوں", order, messages, dialog_state=state)
    resolve.assert_not_called()
    assert order.items[0].qty == 2
    assert reply == ASK_QUANTITY_URDU
