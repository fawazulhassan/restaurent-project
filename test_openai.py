"""Tests for the OpenAI (GPT-6 Luna) provider."""

from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

import config
from app import llm
from app.dialog_state import DialogState
from app.intent import IntentType
from app.llm import (
    RESOLVE_ORDER_INTENT_TOOL,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.order import Order, add_item_to_order

SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_order",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _response(
    *,
    content: str = "",
    tool_calls: list[tuple[str, dict]] | None = None,
    status: str = "completed",
) -> SimpleNamespace:
    output: list[SimpleNamespace] = []
    if content:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=content)],
            )
        )
    for index, (name, arguments) in enumerate(tool_calls or []):
        output.append(
            SimpleNamespace(
                type="function_call",
                call_id=f"call_{index}",
                name=name,
                arguments=json.dumps(arguments),
            )
        )
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=output,
    )


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", "https://api.openai.com/v1")
    )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    monkeypatch.setattr(llm, "_openai_client_instance", client)
    monkeypatch.setattr(config, "OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setattr(config, "OPENAI_REASONING_EFFORT", "low")
    return client


class TestOpenAIProvider:
    def test_request_uses_luna_with_low_reasoning(
        self, fake_client: MagicMock
    ) -> None:
        fake_client.responses.create.return_value = _response(content="جی")
        llm._call_openai(
            [{"role": "user", "content": "hi"}], [SIMPLE_TOOL], allow_tools=True
        )
        kwargs = fake_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-6-luna"
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == config.OPENAI_MAX_COMPLETION_TOKENS
        assert kwargs["input"] == [{"role": "user", "content": "hi"}]
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "update_order",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert kwargs["tool_choice"] == "auto"
        assert "temperature" not in kwargs

    def test_tool_history_is_serialized(self, fake_client: MagicMock) -> None:
        fake_client.responses.create.return_value = _response(content="جی")
        llm._call_openai(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "update_order", "arguments": {"a": 1}}
                    ],
                },
                {"role": "tool", "_tool_call_id": "c1", "content": "ok"},
            ],
            [SIMPLE_TOOL],
            allow_tools=True,
        )
        assert fake_client.responses.create.call_args.kwargs["input"] == [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "update_order",
                "arguments": '{"a": 1}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]

    def test_tools_omitted_when_disabled(self, fake_client: MagicMock) -> None:
        fake_client.responses.create.return_value = _response(content="جی")
        llm._call_openai([], [SIMPLE_TOOL], allow_tools=False)
        kwargs = fake_client.responses.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_tool_call_is_parsed(self, fake_client: MagicMock) -> None:
        fake_client.responses.create.return_value = _response(
            tool_calls=[("update_order", {"action": "add"})]
        )
        result = llm._call_openai([], [SIMPLE_TOOL], allow_tools=True)
        assert result.tool_calls[0].name == "update_order"
        assert result.tool_calls[0].arguments == {"action": "add"}

    def test_truncated_response_is_rejected(self, fake_client: MagicMock) -> None:
        fake_client.responses.create.return_value = _response(status="incomplete")
        with pytest.raises(LLMMalformedResponseError, match="truncated"):
            llm._call_openai([], [], allow_tools=False)

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (
                RateLimitError(
                    "limit", response=_http_response(429), body=None
                ),
                LLMRateLimitError,
            ),
            (
                APITimeoutError(request=httpx.Request("POST", "https://x")),
                LLMTimeoutError,
            ),
            (
                AuthenticationError(
                    "bad key", response=_http_response(401), body=None
                ),
                LLMServiceError,
            ),
        ],
    )
    def test_errors_map_to_llm_errors(
        self, fake_client: MagicMock, error: Exception, expected: type
    ) -> None:
        fake_client.responses.create.side_effect = error
        with pytest.raises(expected):
            llm._call_openai([], [], allow_tools=False)

    def test_call_llm_uses_openai(self, fake_client: MagicMock) -> None:
        fake_client.responses.create.return_value = _response(content="جی")
        result = llm.call_llm([], [], allow_tools=False)
        assert result.content == "جی"

    def test_client_is_created_lazily_and_reused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm, "_openai_client_instance", None)
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
        with patch("app.llm.OpenAI") as factory:
            first = llm._openai_client()
            second = llm._openai_client()
        factory.assert_called_once()
        assert first is second

    def test_check_service_retrieves_model(
        self, fake_client: MagicMock
    ) -> None:
        llm.check_llm_service()
        fake_client.models.retrieve.assert_called_once_with("gpt-6-luna")


class TestOpenAIResolver:
    def test_fallback_prompt_includes_pending_item(self) -> None:
        state = DialogState(
            pending_item_id="chicken-karahi",
            pending_quantity=2,
        )
        messages = llm.build_compact_fallback_messages("آدی", Order(), state)
        user_content = messages[1]["content"]
        assert '"pending_item"' in user_content
        assert '"id":"chicken-karahi"' in user_content
        assert '"quantity":2' in user_content
        assert "pending_item.id" in messages[0]["content"]

    def test_resolver_forces_tool_and_uses_fallback_budget(
        self, fake_client: MagicMock
    ) -> None:
        fake_client.responses.create.return_value = _response(
            tool_calls=[
                (
                    "resolve_order_intent",
                    {
                        "intent": "add_item",
                        "item_id": "garlic-naan",
                        "size_id": "",
                        "quantity": 1,
                        "confidence": 0.9,
                    },
                )
            ]
        )
        parsed = llm.resolve_ambiguous_intent(
            "give me the bread one", Order(), DialogState()
        )
        kwargs = fake_client.responses.create.call_args.kwargs
        assert kwargs["tool_choice"] == {
            "type": "function",
            "name": "resolve_order_intent",
        }
        assert [tool["name"] for tool in kwargs["tools"]] == [
            RESOLVE_ORDER_INTENT_TOOL["function"]["name"]
        ]
        assert (
            kwargs["max_output_tokens"]
            == config.OPENAI_FALLBACK_MAX_COMPLETION_TOKENS
        )
        assert parsed.intent == IntentType.ADD_ITEM
        assert parsed.item_id == "garlic-naan"
        assert not parsed.size_id

    def test_resolver_rejects_missing_tool_call(
        self, fake_client: MagicMock
    ) -> None:
        fake_client.responses.create.return_value = _response(content="hmm")
        with pytest.raises(LLMMalformedResponseError, match="exactly once"):
            llm.resolve_ambiguous_intent("???", Order(), DialogState())


def test_config_requires_key_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OPENAI_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_LIVE_OPENAI_TESTS=1 and OPENAI_API_KEY to run live OpenAI tests",
)
class TestOpenAILive:
    def test_service_and_model_are_ready(
        self
    ) -> None:
        llm.check_llm_service()

    def test_ambiguous_order_is_resolved(
        self
    ) -> None:
        order = Order()
        add_item_to_order(order, "garlic-naan", None, 1)
        state = DialogState(last_item_id="garlic-naan", last_item_size=None)
        parsed = llm.resolve_ambiguous_intent(
            "ek aur wala bhi de dein", order, state
        )
        assert parsed.intent in set(IntentType)
