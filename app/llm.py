"""OpenAI LLM transport for the restaurant order agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
)

import config


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM service cannot complete a request."""


class LLMTimeoutError(LLMServiceError):
    """Raised when the configured LLM service times out."""


class LLMMalformedResponseError(LLMServiceError):
    """Raised when an LLM response does not follow the expected schema."""


class LLMRateLimitError(LLMServiceError):
    """Raised when a cloud LLM provider is rate-limited."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


RESOLVE_ORDER_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "resolve_order_intent",
        "description": (
            "Resolve one ambiguous restaurant-order intent. "
            "This does not mutate the order."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "add_item",
                        "remove_item",
                        "set_quantity",
                        "change_size",
                        "set_address",
                        "set_instructions",
                        "show_order",
                        "confirm_order",
                        "clarify",
                    ],
                },
                "item_id": {"type": "string"},
                "size_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 0},
                "address": {"type": "string"},
                "instructions": {"type": "string"},
                "clarification_question": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["intent", "confidence"],
        },
    },
}

_INTENT_SYSTEM_PROMPT = (
    "You extract exactly one restaurant-order intent. Call resolve_order_intent "
    "exactly once and return no prose. Use only item ids and size ids present in "
    "MENU. Never calculate prices. Never claim an update succeeded. If "
    "STATE.pending_item.id is set, the customer is most likely answering which "
    "size they want for that item; return add_item with that item_id and the "
    "matching size_id. If the request cannot be resolved safely, use "
    'intent="clarify".'
)


def check_llm_service() -> None:
    """Check the configured OpenAI model."""
    _check_openai_service()
    return None


def preload_llm() -> None:
    """Validate the configured OpenAI model before the first turn."""
    if (
        config.HYBRID_INTENT_ENABLED
        and not config.HYBRID_LLM_FALLBACK_ENABLED
    ):
        return
    check_llm_service()
    return


_openai_client_instance: OpenAI | None = None


def _openai_client() -> OpenAI:
    global _openai_client_instance
    if _openai_client_instance is None:
        _openai_client_instance = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT_SECONDS,
            max_retries=1,
        )
    return _openai_client_instance


def _openai_error(exc: Exception, operation: str) -> LLMServiceError:
    if isinstance(exc, OpenAIRateLimitError):
        return LLMRateLimitError("OpenAI is rate-limited or out of credit.")
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError(
            f"OpenAI {operation} timed out after "
            f"{config.OPENAI_TIMEOUT_SECONDS:g} seconds."
        )
    if isinstance(exc, AuthenticationError):
        return LLMServiceError("OpenAI rejected OPENAI_API_KEY.")
    if isinstance(exc, NotFoundError):
        return LLMServiceError(
            f"OpenAI model {config.OPENAI_MODEL!r} was not found for this key."
        )
    if isinstance(exc, APIConnectionError):
        return LLMServiceError("OpenAI is not reachable. Check the connection.")
    if isinstance(exc, BadRequestError):
        return LLMServiceError(f"OpenAI rejected the request: {exc}")
    return LLMServiceError(f"OpenAI {operation} failed: {exc}")


def _check_openai_service() -> None:
    try:
        _openai_client().models.retrieve(config.OPENAI_MODEL)
    except APIError as exc:
        raise _openai_error(exc, "model check") from exc
    print(
        f"OpenAI ready (model {config.OPENAI_MODEL}, "
        f"reasoning_effort={config.OPENAI_REASONING_EFFORT})."
    )


def _serialize_responses_input(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["_tool_call_id"],
                    "output": str(message.get("content", "")),
                }
            )
            continue
        content = message.get("content") or ""
        if content or not message.get("tool_calls"):
            items.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                }
            )
    return items


def _responses_tools(tools: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for tool in tools:
        function = tool["function"]
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    return converted


def _parse_responses_result(response: Any) -> AssistantResult:
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or "unknown"
        raise LLMMalformedResponseError(
            f"OpenAI response was truncated ({reason}). Raise the "
            "OPENAI_*MAX_COMPLETION_TOKENS setting."
        )
    content_parts: list[str] = []
    calls: list[ToolCall] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text":
                    content_parts.append(part.text)
        elif item_type == "function_call":
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise LLMMalformedResponseError(
                    "OpenAI returned invalid tool arguments."
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMMalformedResponseError(
                    "OpenAI tool arguments must be an object."
                )
            calls.append(
                ToolCall(
                    id=item.call_id or str(uuid.uuid4()),
                    name=item.name,
                    arguments=arguments,
                )
            )
    content = "".join(content_parts)
    if not content.strip() and not calls:
        raise LLMMalformedResponseError("OpenAI returned an empty response.")
    return AssistantResult(content=content, tool_calls=calls)


def _call_openai(
    messages: list[dict],
    tools: list[dict],
    *,
    allow_tools: bool,
    tool_choice: str | dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
) -> AssistantResult:
    """Call OpenAI through the Responses API.

    GPT-6 models reject function tools combined with reasoning_effort on
    /v1/chat/completions, so the Responses API is required here.
    """
    kwargs: dict[str, Any] = {
        "model": config.OPENAI_MODEL,
        "input": _serialize_responses_input(messages),
        "reasoning": {"effort": config.OPENAI_REASONING_EFFORT},
        "max_output_tokens": (
            max_completion_tokens or config.OPENAI_MAX_COMPLETION_TOKENS
        ),
        "store": False,
    }
    if allow_tools:
        kwargs["tools"] = _responses_tools(tools)
        kwargs["tool_choice"] = tool_choice or "auto"
    try:
        response = _openai_client().responses.create(**kwargs)
    except APIError as exc:
        raise _openai_error(exc, "request") from exc
    return _parse_responses_result(response)


def call_llm(
    messages: list[dict],
    tools: list[dict],
    *,
    allow_tools: bool = True,
) -> AssistantResult:
    """Call OpenAI."""
    return _call_openai(messages, tools, allow_tools=allow_tools)


def build_compact_fallback_messages(
    user_text: str,
    order: Any,
    state: Any,
) -> list[dict[str, str]]:
    """Build the bounded ambiguity prompt without conversation history."""
    from app.intent import compact_menu_for_fallback, normalize_customer_text

    normalized, original = normalize_customer_text(user_text)
    state_payload = {
        "stage": state.stage.value,
        "summary_presented": state.summary_presented,
        "last_item": {
            "id": state.last_item_id or "",
            "size_id": state.last_item_size or "",
        },
        "pending_item": {
            "id": state.pending_item_id or "",
            "quantity": state.pending_quantity,
        },
        "address_set": bool(order.delivery_address),
        "instructions_set": bool(order.special_instructions),
        "items": [
            {
                "id": item.id,
                "size_id": item.size or "",
                "quantity": item.qty,
            }
            for item in order.items
        ],
    }
    user_payload = (
        "STATE:\n"
        + json.dumps(state_payload, ensure_ascii=False, separators=(",", ":"))
        + "\nMENU:\n"
        + json.dumps(
            compact_menu_for_fallback(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nCUSTOMER_NORMALIZED:\n"
        + normalized
        + "\nCUSTOMER_ORIGINAL:\n"
        + original
    )
    return [
        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def estimate_fallback_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative dependency-free estimate used to enforce prompt headroom."""
    encoded_bytes = sum(
        len(message["content"].encode("utf-8")) for message in messages
    )
    tool_bytes = len(
        json.dumps(
            RESOLVE_ORDER_INTENT_TOOL,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return (encoded_bytes + tool_bytes + 1) // 2


def resolve_ambiguous_intent(user_text: str, order: Any, state: Any) -> Any:
    """Resolve one ambiguous turn with exactly one LLM request."""
    from app.intent import (
        IntentType,
        ParsedIntent,
        extract_address_from_original,
        normalize_customer_text,
    )

    messages = build_compact_fallback_messages(user_text, order, state)
    tools = [RESOLVE_ORDER_INTENT_TOOL]
    result = _call_openai(
        messages,
        tools,
        allow_tools=True,
        tool_choice={"type": "function", "name": "resolve_order_intent"},
        max_completion_tokens=config.OPENAI_FALLBACK_MAX_COMPLETION_TOKENS,
    )

    if len(result.tool_calls) != 1:
        raise LLMMalformedResponseError(
            "OpenAI ambiguity resolver must call resolve_order_intent "
            "exactly once."
        )
    tool_call = result.tool_calls[0]
    if tool_call.name != "resolve_order_intent":
        raise LLMMalformedResponseError(
            f"OpenAI called unexpected ambiguity tool {tool_call.name!r}."
        )

    arguments = {
        key: value for key, value in tool_call.arguments.items() if value != ""
    }
    raw_intent = arguments.pop("intent", "clarify")
    if raw_intent == "clarify":
        raw_intent = IntentType.AMBIGUOUS.value
    arguments.pop("clarification_question", None)
    try:
        parsed = ParsedIntent(intent=raw_intent, **arguments)
    except (TypeError, ValueError) as exc:
        raise LLMMalformedResponseError(
            "OpenAI returned an invalid resolve_order_intent payload."
        ) from exc

    if parsed.intent == IntentType.SET_ADDRESS:
        _, original = normalize_customer_text(user_text)
        address = extract_address_from_original(original, allow_entire=True)
        if not address:
            return ParsedIntent(
                intent=IntentType.AMBIGUOUS,
                confidence=0,
                clarification_reason="invalid_address",
            )
        parsed.address = address
    return parsed
