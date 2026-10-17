"""Generic semantic basket editing: model transport mocked, engine exercised."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.basket_edit import BasketEdit, BasketEditPlan, apply_basket_edits
from app.dialog import chat_turn, build_system_prompt
from app.dialog_state import DialogState, DialogStage
from app.llm import AssistantResult, ToolCall, LLMServiceError
from app.order import Order, OrderError, add_item_to_order, calculate_total
from app.replies import ASK_MORE_ITEMS_URDU, ASK_ADDRESS_URDU


def session(qty=5):
    order = Order(delivery_address=None, special_instructions="less spicy")
    add_item_to_order(order, "plain-naan", None, qty)
    add_item_to_order(order, "chicken-karahi", "full", 1)
    return order, [{"role": "system", "content": build_system_prompt()}], DialogState(
        stage=DialogStage.AWAITING_MORE_ITEMS, last_item_id="chicken-karahi", last_item_size="full"
    )


def result(edits=(), *, confidence=0.99, status="apply"):
    return AssistantResult(tool_calls=[ToolCall("1", "edit_basket", {
        "status": status, "confidence": confidence, "edits": list(edits),
    })])


def edit(action, quantity=None, item_id="plain-naan", size_id=None):
    return dict(action=action, item_id=item_id, size_id=size_id, quantity=quantity)


def naan_qty(order):
    return next((line.qty for line in order.items if line.id == "plain-naan"), 0)


@pytest.mark.parametrize("text,operation,expected", [
    ("do naan kam kar do", edit("reduce", 2), 3),
    ("دو نان کم کر دو", edit("reduce", 2), 3),
    ("remove two naan", edit("reduce", 2), 3),
    ("دو نان ہٹا دو", edit("reduce", 2), 3),
    ("subtract two from the naan count", edit("reduce", 2), 3),
    ("bring the naan count down by two", edit("reduce", 2), 3),
    ("total naan four not five", edit("set", 4), 4),
    ("I only need three naan now", edit("set", 3), 3),
    ("پانچ نان کم کر دو", edit("reduce", 5), 0),
])
def test_semantic_edit_operations_preserve_unrelated_items(text, operation, expected):
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([operation])) as api:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    api.assert_called_once()
    assert state.last_path == "openai_basket_edit"
    assert naan_qty(order) == expected
    assert next(line.qty for line in order.items if line.id == "chicken-karahi") == 1
    assert order.special_instructions == "less spicy"
    assert calculate_total(order) == 1400 + 50 * expected
    assert ASK_MORE_ITEMS_URDU in reply
    assert not complete


@pytest.mark.parametrize("text,expected,kind", [
    ("naan hata do", 0, "local_remove"),
    ("نان ہٹا دو", 0, "local_remove"),
    ("naan nikal do", 0, "local_remove"),
    ("cancel the naan", 0, "local_remove"),
    ("دو نان اور شامل کریں", 7, "local_add"),
    ("add two more naan", 7, "local_add"),
])
def test_fully_named_add_or_remove_stays_local(text, expected, kind):
    order, messages, state = session()
    with patch("app.dialog.resolve_basket_edits") as editor:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    editor.assert_not_called()
    assert state.last_path == kind
    assert state.last_basket_edit_detail == kind
    assert naan_qty(order) == expected
    assert next(line.qty for line in order.items if line.id == "chicken-karahi") == 1
    assert ASK_MORE_ITEMS_URDU in reply
    assert not complete


def test_mixed_reduce_and_add_is_applied_as_one_plan():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([
        edit("reduce", 2), edit("add", 1, "coke-1-5l"),
    ])):
        chat_turn("دو نان کم کر دو اور ایک کوک شامل کرو", order, messages, dialog_state=state)
    assert naan_qty(order) == 3
    assert next(line.qty for line in order.items if line.id == "coke-1-5l") == 1


def test_mixed_set_and_add_uses_generic_editor():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([
        edit("set", 4), edit("add", 2, "garlic-naan"),
    ])):
        chat_turn("ٹوٹل نان چار کر دو اور دو لہسن والے نان شامل کرو", order, messages, dialog_state=state)
    assert naan_qty(order) == 4
    assert next(line.qty for line in order.items if line.id == "garlic-naan") == 2


@pytest.mark.parametrize("operations", [
    [edit("reduce", 6)],
    [edit("reduce", 0)],
    [edit("remove", 2)],
    [edit("set", None)],
    [edit("remove", item_id="not-on-menu")],
    [edit("add", 1, "garlic-naan"), edit("reduce", 6)],
    [edit("reduce", 1), edit("remove")],
    [edit("add", 1, "chicken-karahi", "invented-size")],
])
def test_invalid_plan_has_no_partial_mutation(operations):
    order, _, _ = session()
    before = order.model_dump()
    with pytest.raises(OrderError):
        apply_basket_edits(order, [BasketEdit(**operation) for operation in operations])
    assert order.model_dump() == before


def test_ambiguous_size_cannot_remove_one_arbitrary_line():
    order, _, _ = session()
    add_item_to_order(order, "chicken-karahi", "half", 1)
    before = order.model_dump()
    with pytest.raises(OrderError):
        apply_basket_edits(order, [BasketEdit(**edit("remove", item_id="chicken-karahi"))])
    assert order.model_dump() == before
    apply_basket_edits(order, [BasketEdit(**edit("remove", item_id="chicken-karahi", size_id="half"))])
    assert any(line.id == "chicken-karahi" and line.size == "full" for line in order.items)


@pytest.mark.parametrize("quantity", [-1, True, 1.5, "2"])
def test_model_quantity_must_be_a_nonnegative_integer(quantity):
    with pytest.raises(ValidationError):
        BasketEdit(**edit("reduce", quantity))


@pytest.mark.parametrize("response", [
    result([edit("remove")]),  # counted reduction cannot become whole-line removal
    result([edit("reduce", 9)]),
    result([edit("reduce", 2)], confidence=0.4),
    result([], status="clarify"),
    LLMServiceError("timeout"),
    AssistantResult(content="unstructured response"),
])
def test_failed_or_misclassified_reduction_keeps_basket_unchanged(response):
    order, messages, state = session()
    before = order.model_dump()
    kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
    with patch("app.llm._call_openai", **kwargs):
        _, order, messages, complete = chat_turn("دو نان کم کر دو", order, messages, dialog_state=state)
    assert order.model_dump() == before
    assert not complete


def test_missing_reduction_amount_can_be_answered_with_a_number():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([], status="clarify")) as api:
        chat_turn("نان کم کر دو", order, messages, dialog_state=state)
        assert naan_qty(order) == 5
        chat_turn("دو", order, messages, dialog_state=state)
    api.assert_called_once()
    assert naan_qty(order) == 3


def test_unresolved_request_is_available_to_semantic_followup():
    order, messages, state = session()
    with patch("app.llm._call_openai", side_effect=[
        result([], status="clarify"), result([edit("reduce", 2)]),
    ]) as api:
        chat_turn("please make the naan count smaller", order, messages, dialog_state=state)
        chat_turn("two", order, messages, dialog_state=state)
    assert naan_qty(order) == 3
    assert "please make the naan count smaller" in api.call_args.args[0][1]["content"]
    assert not state.basket_edit_context


@pytest.mark.parametrize("text", ["do not remove naan", "don't remove naan", "naan mat hatao"])
def test_explicit_prohibition_cannot_remove_items_even_if_model_is_wrong(text):
    order, messages, state = session()
    before = order.model_dump()
    with patch("app.llm._call_openai", return_value=result([edit("remove")])):
        chat_turn(text, order, messages, dialog_state=state)
    assert order.model_dump() == before


def test_local_yes_no_and_address_stage_do_not_call_basket_editor():
    order, messages, state = session()
    with patch("app.llm._call_openai") as api:
        chat_turn("جی ہاں", order, messages, dialog_state=state)
        reply, order, messages, complete = chat_turn("نہیں بس", order, messages, dialog_state=state)
    api.assert_not_called()
    assert reply == ASK_ADDRESS_URDU
    assert state.stage == DialogStage.AWAITING_ADDRESS
    assert not complete


def test_removing_last_item_returns_to_item_taking():
    order = Order()
    add_item_to_order(order, "plain-naan", None, 2)
    apply_basket_edits(order, [BasketEdit(**edit("remove"))])
    assert not order.items


def test_misidentified_named_item_cannot_remove_a_different_line():
    order, messages, state = session()
    with patch("app.dialog.resolve_basket_edits") as editor:
        chat_turn("naan hata do", order, messages, dialog_state=state)
    editor.assert_not_called()
    assert naan_qty(order) == 0
    assert any(line.id == "chicken-karahi" and line.size == "full" for line in order.items)


def test_negated_removal_does_not_block_a_separate_requested_addition():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([edit("add", 2, "coke-1-5l")])):
        chat_turn("do not remove naan, add two coke", order, messages, dialog_state=state)
    assert naan_qty(order) == 5
    assert next(line.qty for line in order.items if line.id == "coke-1-5l") == 2


def test_repeated_final_total_is_idempotent():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([edit("set", 4)])):
        for _ in range(2):
            chat_turn("total naan four", order, messages, dialog_state=state)
    assert naan_qty(order) == 4


def test_editor_is_independent_of_general_fallback_toggle(monkeypatch):
    import config
    monkeypatch.setattr(config, "HYBRID_LLM_FALLBACK_ENABLED", False)
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result([edit("reduce", 2)])) as api:
        chat_turn("دو نان کم کر دو", order, messages, dialog_state=state)
    api.assert_called_once()
    assert naan_qty(order) == 3


def _ready():
    order, _, _ = session()
    order.delivery_address = "House 1 Kasur"
    state = DialogState(
        stage=DialogStage.AWAITING_CONFIRMATION,
        summary_presented=True,
        last_item_id="chicken-karahi",
        last_item_size="full",
    )
    messages = [{"role": "system", "content": build_system_prompt()}]
    return order, messages, state


def test_poori_ki_jagah_adhi_replaces_size_and_keeps_quantity():
    order, messages, state = session()
    replacement = BasketEdit(
        action="replace_size", item_id="chicken-karahi", size_id="full",
        new_size_id="half", quantity=None,
    )
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=[replacement],
    )):
        chat_turn("چکن کڑاہی پوری کی جگہ آدھی", order, messages, dialog_state=state)
    line = next(item for item in order.items if item.id == "chicken-karahi")
    assert line.size == "half"
    assert line.qty == 1
    assert line.unit_price_pkr == 800


def test_instead_of_marker_uses_the_size_before_it_as_new():
    order, messages, state = session()
    for line in order.items:
        if line.id == "chicken-karahi":
            line.size = "half"
            line.unit_price_pkr = 800
            line.qty = 2
            line.size_label = "آدھی"
    replacement = BasketEdit(
        action="replace_size", item_id="chicken-karahi", size_id="half",
        new_size_id="full", quantity=None,
    )
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=[replacement],
    )):
        chat_turn("chicken karahi full instead of half", order, messages, dialog_state=state)
    line = next(item for item in order.items if item.id == "chicken-karahi")
    assert line.size == "full"
    assert line.qty == 2
    assert line.unit_price_pkr == 1400


def test_both_size_lemmas_without_a_marker_are_rejected():
    order, messages, state = session()
    before = order.model_dump()
    replacement = BasketEdit(
        action="replace_size", item_id="chicken-karahi", size_id="full",
        new_size_id="half", quantity=None,
    )
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=[replacement],
    )):
        chat_turn("چکن کڑاہی آدھی پوری", order, messages, dialog_state=state)
    assert order.model_dump() == before
    assert state.last_basket_edit_detail == "rejected=both_lemmas_no_marker"


def test_two_replace_size_ops_that_need_a_marker_are_rejected():
    order, messages, state = session()
    add_item_to_order(order, "mutton-karahi", "full", 1)
    before = order.model_dump()
    edits = [
        BasketEdit(action="replace_size", item_id="chicken-karahi", size_id="full", new_size_id="half", quantity=None),
        BasketEdit(action="replace_size", item_id="mutton-karahi", size_id="full", new_size_id="half", quantity=None),
    ]
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=edits,
    )):
        chat_turn("پوری کی جگہ آدھی", order, messages, dialog_state=state)
    assert order.model_dump() == before
    assert state.last_basket_edit_detail == "rejected=both_lemmas_no_marker"


def test_replace_size_merges_into_existing_target_line():
    order, _, _ = session()
    add_item_to_order(order, "chicken-karahi", "half", 1)
    apply_basket_edits(order, [BasketEdit(
        action="replace_size", item_id="chicken-karahi", size_id="full",
        new_size_id="half", quantity=None,
    )])
    lines = [line for line in order.items if line.id == "chicken-karahi"]
    assert len(lines) == 1
    assert lines[0].size == "half"
    assert lines[0].qty == 2
    assert lines[0].unit_price_pkr == 800


def test_replace_size_without_a_size_is_rejected_when_two_lines_exist():
    order, _, _ = session()
    add_item_to_order(order, "chicken-karahi", "half", 1)
    before = order.model_dump()
    with pytest.raises(OrderError, match="ambiguous_size_line"):
        apply_basket_edits(order, [BasketEdit(
            action="replace_size", item_id="chicken-karahi", new_size_id="half", quantity=None,
        )])
    assert order.model_dump() == before


def test_remove_half_and_add_full_apply_together():
    order, messages, state = session()
    for line in order.items:
        if line.id == "chicken-karahi":
            line.size = "half"
            line.unit_price_pkr = 800
            line.size_label = "آدھی"
    edits = [
        BasketEdit(action="remove", item_id="chicken-karahi", size_id="half", quantity=None),
        BasketEdit(action="add", item_id="chicken-karahi", size_id="full", quantity=1),
    ]
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=edits,
    )):
        chat_turn("چکن کڑاہی آدھی ختم کرنی ہے اور پوری شامل کرنی ہے", order, messages, dialog_state=state)
    line = next(item for item in order.items if item.id == "chicken-karahi")
    assert line.size == "full"
    assert line.qty == 1


def test_failed_replace_size_leaves_the_basket_unchanged():
    order, _, _ = session()
    before = order.model_dump()
    with pytest.raises(OrderError):
        apply_basket_edits(order, [BasketEdit(
            action="replace_size", item_id="chicken-karahi", size_id="full",
            new_size_id="no-such-size", quantity=None,
        )])
    assert order.model_dump() == before


def test_confirmation_phrases_confirm_without_a_basket_edit():
    from app.order import OrderStatus
    from app.replies import build_confirmation_urdu
    for text in (
        "تصدیق کرتے ہیں",
        "ہاں جی تصدیق کرتے ہیں سوڈر کی",
        "کنفرم کر دیں",
        "haan confirm kar do",
        "yes confirm order",
    ):
        order, messages, state = _ready()
        with patch("app.dialog.resolve_basket_edits") as editor:
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
        editor.assert_not_called()
        assert complete, text
        assert order.status == OrderStatus.CONFIRMED
        assert reply == build_confirmation_urdu(order)


def test_yes_plus_an_item_goes_to_the_editor_and_does_not_confirm():
    from app.order import OrderStatus
    from app.replies import ASK_CONFIRM_URDU
    order, messages, state = _ready()
    addition = BasketEdit(action="add", item_id="plain-naan", quantity=1)
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=[addition],
    )) as editor:
        reply, order, messages, complete = chat_turn("ہاں اور ایک نان", order, messages, dialog_state=state)
    editor.assert_called_once()
    assert not complete
    assert order.status != OrderStatus.CONFIRMED
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert ASK_CONFIRM_URDU in reply
    assert naan_qty(order) == 6


def test_no_asks_what_to_change_and_a_later_edit_returns_to_confirmation():
    from app.replies import ASK_CONFIRM_URDU, ASK_WHAT_TO_CHANGE_URDU
    order, messages, state = _ready()
    state.order_change_failures = 2
    reply, order, messages, complete = chat_turn("نہیں", order, messages, dialog_state=state)
    assert reply == ASK_WHAT_TO_CHANGE_URDU
    assert state.stage == DialogStage.AWAITING_ORDER_CHANGE
    assert state.order_change_failures == 0
    assert not complete
    replacement = BasketEdit(
        action="replace_size", item_id="chicken-karahi", size_id="full",
        new_size_id="half", quantity=None,
    )
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="apply", confidence=0.99, edits=[replacement],
    )) as editor:
        reply, order, messages, complete = chat_turn(
            "چکن کڑاہی پوری کی جگہ آدھی", order, messages, dialog_state=state,
        )
    editor.assert_called_once()
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.order_change_failures == 0
    assert ASK_CONFIRM_URDU in reply
    assert not complete


def test_three_order_change_failures_return_to_confirmation():
    from app.replies import ASK_CONFIRM_URDU, ASK_WHAT_TO_CHANGE_URDU
    order, messages, state = _ready()
    chat_turn("نہیں", order, messages, dialog_state=state)
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="clarify", confidence=1, edits=[],
    )):
        first, _, _, _ = chat_turn("نامعلوم", order, messages, dialog_state=state)
        second, _, _, _ = chat_turn("نامعلوم", order, messages, dialog_state=state)
        third, _, _, _ = chat_turn("نامعلوم", order, messages, dialog_state=state)
    assert first == ASK_WHAT_TO_CHANGE_URDU
    assert second == ASK_WHAT_TO_CHANGE_URDU
    assert state.order_confirmation_attempts == 0
    assert state.order_change_failures == 0
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert ASK_CONFIRM_URDU in third


def test_three_unclear_confirmations_hand_off():
    from app.replies import ASK_CONFIRM_URDU, HUMAN_HANDOFF_CONFIRM_URDU
    order, messages, state = _ready()
    replies = []
    for _ in range(3):
        reply, order, messages, complete = chat_turn("کیا مطلب", order, messages, dialog_state=state)
        replies.append(reply)
        assert not complete
    assert ASK_CONFIRM_URDU in replies[0]
    assert ASK_CONFIRM_URDU in replies[1]
    assert replies[2] == HUMAN_HANDOFF_CONFIRM_URDU
    assert state.stage == DialogStage.HANDOFF
    assert order.delivery_address == "House 1 Kasur"


def test_failing_editor_on_a_mutation_signal_does_not_confirm():
    from app.order import OrderStatus
    order, messages, state = _ready()
    with patch("app.dialog.resolve_basket_edits", side_effect=LLMServiceError("down")):
        chat_turn("ہاں اور ایک نان", order, messages, dialog_state=state)
    assert order.status != OrderStatus.CONFIRMED
    assert state.order_confirmation_attempts == 1
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert naan_qty(order) == 5


def test_editor_clarify_during_confirmation_counts_as_unclear():
    from app.replies import ASK_CONFIRM_URDU
    order, messages, state = _ready()
    with patch("app.dialog.resolve_basket_edits", return_value=BasketEditPlan(
        status="clarify", confidence=1, edits=[],
    )):
        reply, order, messages, complete = chat_turn(
            "ہاں اور ایک نان", order, messages, dialog_state=state,
        )
    assert not complete
    assert state.order_confirmation_attempts == 1
    assert ASK_CONFIRM_URDU in reply
    assert naan_qty(order) == 5


def test_size_word_with_a_group_token_calls_the_basket_editor():
    order, messages, state = session()
    with patch("app.llm._call_openai", return_value=result(status="clarify")) as api:
        chat_turn("چھوٹا پیزا", order, messages, dialog_state=state)
    api.assert_called_once()
    assert state.last_path == "openai_basket_edit"
    assert not state.pending_choices
