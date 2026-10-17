"""Offline contract tests for the focused address ambiguity resolver."""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import pytest

from app import llm


def test_address_transport_uses_short_timeout_without_retry():
    client = MagicMock()
    client.with_options.return_value.responses.create.return_value = SimpleNamespace(
        status="completed", output=[SimpleNamespace(
            type="message", content=[SimpleNamespace(type="output_text", text="ok")]
        )]
    )
    with patch("app.llm._openai_client", return_value=client):
        llm._call_openai([], [], allow_tools=False, timeout_seconds=4)
    client.with_options.assert_called_once_with(timeout=4, max_retries=0)


def test_voice_session_stops_on_handoff_without_saving():
    from app.agent import run_conversation
    from app.dialog_state import DialogStage

    record = MagicMock(return_value="unclear speech")

    def handoff(text, order, messages, *, dialog_state, log_latency):
        dialog_state.stage = DialogStage.HANDOFF
        return "Contact staff", None, order, messages, False

    with (
        patch("app.agent.preload_models"),
        patch("app.agent.process_text_turn", side_effect=handoff),
        patch("app.agent.save_order") as save,
    ):
        assert run_conversation(record, play_greeting_audio=False) is None
    record.assert_called_once()
    save.assert_not_called()
