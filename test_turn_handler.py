"""Stubbed tests for the validated model turn handler. No network calls."""

from __future__ import annotations

import json
from unittest.mock import patch

import config
from app.dialog import _apply_unhandled_turn, _menu_item, build_spoken_menu_urdu, chat_turn, build_system_prompt
from app.dialog_state import DialogStage, DialogState, ItemRequest
from app.intent import IntentType, ParsedIntent
from app.llm import (
    LLMServiceError,
    UnhandledTurn,
    _call_turn_model,
    validate_unhandled_turn,
)
from app.order import Order, add_item_to_order
from app.replies import (
    ASK_CLARIFY_URDU,
    ASK_CONFIRM_URDU,
    UNAVAILABLE_ITEM_URDU,
    build_item_choice_urdu,
    build_missing_size_urdu,
)


def _session():
    return Order(), [{"role": "system", "content": build_system_prompt()}], DialogState()


def _payload(**overrides) -> str:
    body = {
        "action": "add_items",
        "items": [{"item_id": "plain-naan", "size_id": None, "quantity": 1}],
        "unknown_names": [],
        "clarify_text": None,
    }
    body.update(overrides)
    return json.dumps(body)


def test_valid_add_items_output_adds_the_item() -> None:
    decision = validate_unhandled_turn(_payload())
    order, _messages, state = _session()
    reply, _complete = _apply_unhandled_turn(decision, order, state)
    assert decision.validation == "ok"
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1)]
    assert state.last_path == "deterministic"
    assert UNAVAILABLE_ITEM_URDU not in reply


def test_bad_items_and_quantities_are_clarify_with_no_staging() -> None:
    order, _messages, state = _session()
    state.pending_additions = [ItemRequest(item_id="plain-naan", quantity=1)]
    state.staged_start_index = 1
    for raw in (
        _payload(items=[{"item_id": "not-a-dish", "size_id": None, "quantity": 1}]),
        _payload(items=[{"item_id": "chicken-karahi", "size_id": "huge", "quantity": 1}]),
        _payload(items=[{"item_id": "plain-naan", "size_id": None, "quantity": 0}]),
        _payload(items=[{"item_id": "plain-naan", "size_id": None, "quantity": 99}]),
    ):
        decision = validate_unhandled_turn(raw)
        reply, _complete = _apply_unhandled_turn(decision, order, state)
        assert reply == ASK_CLARIFY_URDU
        assert order.items == []
        assert len(state.pending_additions) == 1
        assert state.staged_start_index == 1


def test_too_many_items_or_quantity_is_clarify() -> None:
    items = [{"item_id": "plain-naan", "size_id": None, "quantity": 1} for _ in range(6)]
    assert validate_unhandled_turn(_payload(items=items)).validation == "too_many_items"
    items = [{"item_id": "plain-naan", "size_id": None, "quantity": 11} for _ in range(2)]
    assert validate_unhandled_turn(_payload(items=items)).validation == "too_much_quantity"


def test_unknown_name_near_an_alias_is_clarify() -> None:
    decision = validate_unhandled_turn(_payload(unknown_names=["plain naan"]))
    assert decision.action == "clarify"
    assert decision.validation == "unknown_name_matches_menu"


def test_choice_without_a_shared_group_is_clarify() -> None:
    decision = validate_unhandled_turn(_payload(
        action="choice",
        items=[
            {"item_id": "plain-naan", "size_id": None, "quantity": 1},
            {"item_id": "coke-1-5l", "size_id": None, "quantity": 1},
        ],
    ))
    assert decision.validation == "bad_choice"


def test_price_or_confirmed_is_ignored() -> None:
    priced = _payload().replace("null", "null", 1)
    body = json.loads(_payload())
    body["price"] = 100
    assert validate_unhandled_turn(json.dumps(body)).validation == "forbidden_field"
    body = json.loads(_payload())
    body["note"] = "confirmed"
    assert validate_unhandled_turn(json.dumps(body)).validation == "forbidden_field"
    assert priced


def test_invalid_json_is_clarify() -> None:
    decision = validate_unhandled_turn("not json")
    order, _messages, state = _session()
    reply, _complete = _apply_unhandled_turn(decision, order, state)
    assert reply == ASK_CLARIFY_URDU
    assert order.items == []
    assert state.pending_choices == []


def test_timeout_falls_back_without_changing_state() -> None:
    order, messages, state = _session()
    failed = UnhandledTurn(action="clarify", validation="timeout")
    with patch("app.dialog.resolve_unhandled_turn", return_value=failed):
        reply, order, messages, _complete = chat_turn(
            "ایک چکن کڑاہی", order, messages, dialog_state=state
        )
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))
    assert order.items == []
    assert state.order_confirmation_attempts == 0
    assert state.order_change_failures == 0
    assert state.pending_item_id == "chicken-karahi"


def test_unavailable_plus_choice_is_one_string_and_is_not_repeated() -> None:
    decision = validate_unhandled_turn(_payload(
        action="unavailable",
        unknown_names=["سنگر برگر"],
        items=[
            {"item_id": "chicken-tikka-pizza", "size_id": None, "quantity": 1},
            {"item_id": "margherita-pizza", "size_id": None, "quantity": 1},
        ],
    ))
    order, messages, state = _session()
    reply, _complete = _apply_unhandled_turn(decision, order, state)
    question = build_item_choice_urdu([
        _menu_item("chicken-tikka-pizza"),
        _menu_item("margherita-pizza"),
    ])
    assert reply == f"{UNAVAILABLE_ITEM_URDU} {question}"
    assert order.items == []
    follow, _order, _messages, _complete = chat_turn(
        "چکن تکہ پیزا", order, messages, dialog_state=state
    )
    assert not follow.startswith(UNAVAILABLE_ITEM_URDU)


def test_pure_unknown_asks_the_model() -> None:
    order, messages, state = _session()
    decision = UnhandledTurn(action="unavailable", validation="ok")
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision) as handler:
        reply, order, messages, _complete = chat_turn(
            "جنگر بڑگر اوڈر کرنا چاہتا ہوں۔", order, messages, dialog_state=state
        )
    handler.assert_called_once()
    assert reply == UNAVAILABLE_ITEM_URDU
    assert order.items == []


def test_dropped_unknown_does_not_call_the_model() -> None:
    order, messages, state = _session()
    intent = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        confidence=1,
        item_id="plain-naan",
        quantity=1,
        dropped_unknown=True,
    )
    with (
        patch("app.dialog.parse_customer_intent", return_value=intent),
        patch("app.dialog.resolve_unhandled_turn") as handler,
    ):
        reply, order, messages, _complete = chat_turn("ignored", order, messages, dialog_state=state)
    handler.assert_not_called()
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1)]
    assert UNAVAILABLE_ITEM_URDU not in reply


def test_model_runs_for_size_and_choice_but_not_yes_no() -> None:
    order, messages, state = _session()
    script = UnhandledTurn(action="use_script")
    with patch("app.dialog.resolve_unhandled_turn", return_value=script) as handler:
        chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
        chat_turn("نامعلوم", order, messages, dialog_state=state)
        assert state.stage == DialogStage.AWAITING_SIZE
        assert handler.call_count == 2
        state.pending_choices = ["chicken-karahi", "mutton-karahi"]
        state.pending_choice_quantity = 1
        state.stage = DialogStage.ORDERING
        chat_turn("جنگر بڑگر اوڈر کرنا چاہتا ہوں۔", order, messages, dialog_state=state)
        assert handler.call_count == 3
        state.pending_choices = []
        add_item_to_order(order, "plain-naan", None, 1)
        order.delivery_address = "House 1"
        state.stage = DialogStage.AWAITING_CONFIRMATION
        state.summary_presented = True
        reply, order, messages, complete = chat_turn(
            "ہاں", order, messages, dialog_state=state
        )
    assert handler.call_count == 3
    assert complete
    assert reply


def test_menu_does_not_change_stage_and_other_clarifies() -> None:
    order, _messages, state = _session()
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    reply, _complete = _apply_unhandled_turn(
        UnhandledTurn(action="menu", validation="ok"), order, state
    )
    assert reply == build_spoken_menu_urdu()
    assert state.stage == DialogStage.AWAITING_MORE_ITEMS
    reply, _complete = _apply_unhandled_turn(
        UnhandledTurn(action="other", validation="ok"), order, state
    )
    assert reply == ASK_CLARIFY_URDU


def test_clarify_during_confirmation_counts_as_unclear() -> None:
    order, _messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    reply, _complete = _apply_unhandled_turn(
        UnhandledTurn(action="clarify", validation="ok", clarify_text="براہ کرم واضح کریں"),
        order,
        state,
    )
    assert state.order_confirmation_attempts == 1
    assert ASK_CONFIRM_URDU in reply
    assert order.items[0].qty == 1


def test_flag_off_uses_resolve_ambiguous_intent(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_TURN_HANDLER_ENABLED", False)
    order, messages, state = _session()
    resolved = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="no_safe_rule")
    with (
        patch("app.dialog.resolve_ambiguous_intent", return_value=resolved) as old,
        patch("app.dialog.resolve_unhandled_turn") as handler,
    ):
        reply, order, messages, _complete = chat_turn(
            "give me the bread one", order, messages, dialog_state=state
        )
    old.assert_called_once()
    handler.assert_not_called()
    assert reply == ASK_CLARIFY_URDU
    assert order.items == []


def test_turn_handler_payload_uses_temperature_zero() -> None:
    import app.llm as llm

    llm._temperature_omitted = False
    captured = {}

    def accept(messages, tools, **kwargs):
        captured.update(kwargs)
        from app.llm import AssistantResult

        return AssistantResult(content=_payload())

    with patch("app.llm._call_openai", side_effect=accept):
        text = _call_turn_model([{"role": "user", "content": "hi"}])
    assert json.loads(text)["action"] == "add_items"
    assert captured["temperature"] == 0
    assert captured["json_mode"] is True
    assert captured["timeout_seconds"] == config.LLM_TURN_TIMEOUT_SECONDS
    assert captured["max_completion_tokens"] == config.OPENAI_FALLBACK_MAX_COMPLETION_TOKENS


def test_temperature_rejection_is_omitted_and_logged_once(caplog) -> None:
    import app.llm as llm

    llm._temperature_omitted = False
    calls = []

    def reject_once(messages, tools, **kwargs):
        calls.append(kwargs.get("temperature", "absent"))
        from app.llm import AssistantResult

        if kwargs.get("temperature") is not None:
            raise LLMServiceError("OpenAI rejected the request: temperature")
        return AssistantResult(content=_payload())

    try:
        with caplog.at_level("WARNING"), patch("app.llm._call_openai", side_effect=reject_once):
            _call_turn_model([{"role": "user", "content": "hi"}])
            _call_turn_model([{"role": "user", "content": "again"}])
        assert calls == [0, "absent", "absent"]
    finally:
        llm._temperature_omitted = False
    assert sum("omitting it" in record.message for record in caplog.records) == 1


def _script():
    return UnhandledTurn(action="use_script")


def test_use_script_for_karahi_asks_size_once() -> None:
    order, messages, state = _session()
    with (
        patch("app.dialog.resolve_unhandled_turn", return_value=_script()) as handler,
        patch("app.dialog.resolve_ambiguous_intent") as old,
    ):
        reply, order, messages, _complete = chat_turn(
            "ایک چکن کڑاہی", order, messages, dialog_state=state
        )
    handler.assert_called_once()
    old.assert_not_called()
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))


def test_plain_yes_does_not_call_the_model() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    with patch("app.dialog.resolve_unhandled_turn") as handler:
        reply, order, messages, complete = chat_turn(
            "ہاں", order, messages, dialog_state=state
        )
    handler.assert_not_called()
    assert complete
    assert "کنفرم" in reply


def test_drink_add_from_the_model_adds_the_item() -> None:
    decision = UnhandledTurn(
        action="add_items",
        validation="ok",
        items=[{"item_id": "coke-1-5l", "size_id": None, "quantity": 1}],
    )
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision):
        _reply, order, _messages, _complete = chat_turn(
            "ڈرنگس اوڈر کرنا چاہتا ہوں۔", order, messages, dialog_state=state
        )
    assert [(item.id, item.qty) for item in order.items] == [("coke-1-5l", 1)]


def test_exact_add_ignores_a_different_model_item() -> None:
    decision = UnhandledTurn(
        action="add_items",
        validation="ok",
        items=[{"item_id": "garlic-naan", "size_id": None, "quantity": 1}],
    )
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision):
        reply, order, messages, _complete = chat_turn(
            "ایک چکن کڑاہی", order, messages, dialog_state=state
        )
    assert order.items == []
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))


def test_spoken_size_is_not_taken_from_the_model() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن تکہ پیزا", order, messages, dialog_state=state)
    decision = UnhandledTurn(
        action="add_items",
        validation="ok",
        items=[{"item_id": "chicken-tikka-pizza", "size_id": "large", "quantity": 1}],
    )
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision):
        _reply, order, _messages, _complete = chat_turn(
            "چھوٹا", order, messages, dialog_state=state
        )
    assert [(item.id, item.size) for item in order.items] == [("chicken-tikka-pizza", "small")]


def test_restricted_stages_keep_only_unavailable_and_menu() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    added = UnhandledTurn(
        action="add_items",
        validation="ok",
        items=[{"item_id": "plain-naan", "size_id": None, "quantity": 1}],
    )
    with patch("app.dialog.resolve_unhandled_turn", return_value=added):
        reply, order, messages, _complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
    assert order.items == []
    assert state.pending_item_id == "chicken-karahi"
    assert reply == build_missing_size_urdu(_menu_item("chicken-karahi"))

    menu = UnhandledTurn(action="menu", validation="ok")
    with patch("app.dialog.resolve_unhandled_turn", return_value=menu):
        reply, order, messages, _complete = chat_turn(
            "مینو سنائیں", order, messages, dialog_state=state
        )
    assert reply.startswith(build_spoken_menu_urdu())
    assert reply.endswith(build_missing_size_urdu(_menu_item("chicken-karahi")))
    assert state.pending_item_id == "chicken-karahi"
    assert state.stage == DialogStage.AWAITING_SIZE

    state.pending_choices = ["chicken-karahi", "mutton-karahi"]
    state.pending_choice_quantity = 1
    state.pending_item_id = None
    state.stage = DialogStage.ORDERING
    with patch("app.dialog.resolve_unhandled_turn", return_value=added):
        _reply, order, _messages, _complete = chat_turn(
            "نامعلوم", order, messages, dialog_state=state
        )
    assert order.items == []
    assert state.pending_choices == ["chicken-karahi", "mutton-karahi"]

    state.pending_choices = []
    state.stage = DialogStage.AWAITING_ADDRESS
    with patch("app.dialog.resolve_unhandled_turn", return_value=added):
        _reply, order, _messages, _complete = chat_turn(
            "???", order, messages, dialog_state=state
        )
    assert order.items == []


def test_confirmation_add_rereads_the_summary() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    state.order_confirmation_attempts = 2
    state.order_change_failures = 2
    decision = UnhandledTurn(
        action="add_items",
        validation="ok",
        items=[{"item_id": "garlic-naan", "size_id": None, "quantity": 1}],
    )
    with patch("app.dialog.resolve_unhandled_turn", return_value=decision):
        reply, order, _messages, _complete = chat_turn(
            "کچھ اور شامل کر دیں", order, messages, dialog_state=state
        )
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1), ("garlic-naan", 1)]
    assert state.order_confirmation_attempts == 0
    assert state.order_change_failures == 0
    assert ASK_CONFIRM_URDU in reply
    assert state.stage == DialogStage.AWAITING_CONFIRMATION


def test_address_outside_address_stage_is_redacted(tmp_path, monkeypatch) -> None:
    from app import llm

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(llm, "_append_model_turn", lambda record: log_dir.joinpath("hit.json").write_text(
        __import__("json").dumps(record, ensure_ascii=False), encoding="utf-8"
    ))
    order, messages, state = _session()
    chat_turn(
        "میرا پتہ کوٹ شامی شہید نیر موڈل بزار قصور ہے",
        order,
        messages,
        dialog_state=state,
    )
    saved = __import__("json").loads((log_dir / "hit.json").read_text(encoding="utf-8"))
    assert saved["utterance"] == "[redacted]"
    assert saved["final_path"] != "llm_turn"


_GREETING = "وعلیکم السلام۔ بتائیں کیا آرڈر کروں؟"


def _reply(text: str = _GREETING) -> UnhandledTurn:
    return UnhandledTurn(action="reply", validation="ok", reply_text=text)


def test_greeting_reply_is_spoken() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        reply, order, _messages, _complete = chat_turn(
            "والیکم اسلام", order, messages, dialog_state=state
        )
    assert reply == _GREETING
    assert ASK_CLARIFY_URDU not in reply
    assert order.items == []
    assert state.last_path == "llm_turn"
    assert state.reply_streak == 1


def test_quantity_question_reply_does_not_edit_the_basket() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "chicken-karahi", "half", 1)
    with (
        patch("app.dialog.resolve_unhandled_turn", return_value=_reply()),
        patch("app.dialog.resolve_basket_edits") as editor,
    ):
        reply, order, _messages, _complete = chat_turn(
            "کیا میں تعداد بدل سکتا ہوں؟", order, messages, dialog_state=state
        )
    editor.assert_not_called()
    assert reply == _GREETING
    assert order.items[0].qty == 1


def test_unsafe_reply_text_falls_back_to_the_script() -> None:
    samples = [
        "کل آٹھ سو روپے بنتے ہیں",
        "آپ کا آرڈر کنفرم ہو گیا",
        "Hello, I can help you order food",
        "ا" * 201,
    ]
    for text in samples:
        decision = validate_unhandled_turn(json.dumps({
            "action": "reply",
            "items": [],
            "unknown_names": [],
            "clarify_text": None,
            "reply_text": text,
        }))
        assert decision.validation != "ok"
        order, messages, state = _session()
        with patch("app.dialog.resolve_unhandled_turn", return_value=decision):
            reply, order, _messages, _complete = chat_turn(
                "والیکم اسلام", order, messages, dialog_state=state
            )
        assert order.items == []
        assert reply == ASK_CLARIFY_URDU
        assert state.last_path != "llm_turn"


def test_ordinary_menu_word_is_allowed_and_unknown_food_is_not() -> None:
    allowed = validate_unhandled_turn(json.dumps({
        "action": "reply",
        "reply_text": "کیا آپ مینو سنا سکتے ہیں؟",
    }))
    blocked = validate_unhandled_turn(json.dumps({
        "action": "reply",
        "reply_text": "کیا آپ برگر بھی دیتے ہیں؟",
    }))
    assert allowed.validation == "ok"
    assert blocked.validation != "ok"


def test_reply_during_confirmation_appends_the_question() -> None:
    order, messages, state = _session()
    add_item_to_order(order, "plain-naan", None, 1)
    order.delivery_address = "House 1"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    state.order_confirmation_attempts = 2
    state.order_change_failures = 1
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        reply, order, _messages, _complete = chat_turn(
            "والیکم اسلام", order, messages, dialog_state=state
        )
    assert reply == f"{_GREETING} {ASK_CONFIRM_URDU}"
    assert state.order_confirmation_attempts == 2
    assert state.order_change_failures == 1
    assert state.summary_presented
    assert [(item.id, item.qty) for item in order.items] == [("plain-naan", 1)]


def test_reply_on_a_mixed_order_is_ignored() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        reply, order, _messages, _complete = chat_turn(
            "ایک نان اور ایک چکن کڑاہی", order, messages, dialog_state=state
        )
    assert _GREETING not in reply
    assert state.last_path != "llm_turn"
    assert state.pending_item_id == "chicken-karahi"


def test_fourth_reply_uses_the_script() -> None:
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        for expected in (1, 2, 3):
            reply, order, messages, _complete = chat_turn(
                "والیکم اسلام", order, messages, dialog_state=state
            )
            assert reply == _GREETING
            assert state.reply_streak == expected
        reply, order, messages, _complete = chat_turn(
            "والیکم اسلام", order, messages, dialog_state=state
        )
    assert reply == ASK_CLARIFY_URDU
    assert state.reply_streak == 0
    assert state.last_path != "llm_turn"


def test_reply_text_appears_in_the_log(tmp_path, monkeypatch) -> None:
    from app import llm

    captured: dict = {}
    monkeypatch.setattr(llm, "_append_model_turn", lambda record: captured.update(record))
    order, messages, state = _session()
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        chat_turn("والیکم اسلام", order, messages, dialog_state=state)
    assert captured["reply_text"] == _GREETING


def test_pending_size_ignores_a_reply() -> None:
    order, messages, state = _session()
    chat_turn("ایک چکن کڑاہی", order, messages, dialog_state=state)
    with patch("app.dialog.resolve_unhandled_turn", return_value=_reply()):
        reply, order, _messages, _complete = chat_turn(
            "آدھی", order, messages, dialog_state=state
        )
    assert _GREETING not in reply
    assert [(item.id, item.size) for item in order.items] == [("chicken-karahi", "half")]
