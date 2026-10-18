"""Conversation memory recorded for today's handler. No network calls."""

from __future__ import annotations

import json

from app.dialog import HISTORY_MAX_PAIRS, HISTORY_TEXT_LIMIT, _finish_hybrid_turn, chat_turn
from app.dialog_state import DialogStage, DialogState, HistoryTurn
from app.intent import IntentType, ParsedIntent
from app.llm import resolve_unhandled_turn
from app.order import Order


def _session():
    return Order(), [{"role": "system", "content": "test"}], DialogState()


def test_dialog_state_deep_copy_does_not_alias_history() -> None:
    state = DialogState(history=[HistoryTurn(role="customer", text="سلام")])
    copied = state.model_copy(deep=True)
    copied.history.append(HistoryTurn(role="assistant", text="جی"))
    assert len(state.history) == 1
    assert state.history[0].text == "سلام"


def test_history_is_capped_and_clipped() -> None:
    order, messages, state = _session()
    long_text = "ا" * (HISTORY_TEXT_LIMIT + 50)
    chat_turn(long_text, order, messages, dialog_state=state)
    assert len(state.history[0].text) == HISTORY_TEXT_LIMIT
    for index in range(HISTORY_MAX_PAIRS + 2):
        chat_turn(f"نامعلوم {index}", order, messages, dialog_state=state)
    assert len(state.history) == HISTORY_MAX_PAIRS * 2
    assert state.agent_turns == HISTORY_MAX_PAIRS + 3


def test_address_utterance_is_redacted_in_history() -> None:
    order, messages, state = _session()
    state.stage = DialogStage.AWAITING_ADDRESS
    chat_turn("جی ٹی روڈ پرانا لاری اڈہ", order, messages, dialog_state=state)
    assert state.history[0].text == "[redacted]"
    assert state.history[0].role == "customer"


def test_complete_and_handoff_append_then_clear() -> None:
    class Spy(list):
        def __init__(self) -> None:
            super().__init__()
            self.appended: list = []

        def append(self, item):
            self.appended.append(item)
            super().append(item)

    for stage, reply in (
        (DialogStage.COMPLETE, "آرڈر مکمل ہو گیا"),
        (DialogStage.HANDOFF, "معذرت"),
    ):
        state = DialogState(stage=stage)
        spy = Spy()
        state.history = spy
        intent = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0)
        _finish_hybrid_turn("سلام", Order(), [], state, intent, None, reply, stage == DialogStage.COMPLETE)
        assert len(spy.appended) == 2
        assert spy.appended[0].role == "customer"
        assert spy.appended[1].text == reply
        assert state.history == []
        assert state.last_reply == ""
        assert state.agent_turns == 1


def test_new_dialog_state_starts_empty() -> None:
    state = DialogState()
    assert state.history == []
    assert state.last_reply == ""


def test_follow_up_reaches_the_model_with_history(monkeypatch) -> None:
    captured: dict = {}

    def fake_call(messages):
        captured["content"] = messages[1]["content"]
        return '{"action":"use_script"}'

    monkeypatch.setattr("app.llm._call_turn_model", fake_call)
    monkeypatch.setattr("app.dialog.resolve_unhandled_turn", resolve_unhandled_turn)
    order, messages, state = _session()
    chat_turn("ایک نان", order, messages, dialog_state=state)
    chat_turn("الگا والا بڑا", order, messages, dialog_state=state)
    payload = json.loads(captured["content"])
    history = payload["untrusted"]["history"]
    assert payload["untrusted"]["utterance"] == "الگا والا بڑا"
    assert history[0]["role"] == "customer"
    assert "نان" in history[0]["text"] or history[0]["text"]
    assert history[1]["role"] == "assistant"
    assert history[1]["text"] == state.history[1].text
    assert "menu" in payload
    assert payload["basket"][0]["spoken_name"]
