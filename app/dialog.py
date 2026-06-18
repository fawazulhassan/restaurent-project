import json
import time
from pathlib import Path

from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion

import config
from app.order import (
    Order,
    OrderError,
    OrderStatus,
    apply_order_update,
    build_confirmation_english,
    build_order_status_reply,
    calculate_total,
)


class LLMRateLimitError(Exception):
    """Raised when all LLM providers are rate-limited."""


openrouter_client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Kasur Kitchen Voice Agent",
    },
)
OPENROUTER_MODEL = config.OPENROUTER_MODEL

_groq_client: Groq | None = None
if config.GROQ_API_KEY and config.GROQ_API_KEY != "your_groq_api_key_here":
    _groq_client = Groq(api_key=config.GROQ_API_KEY)
GROQ_MODEL = config.GROQ_MODEL
UPDATE_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "update_order",
        "description": "Update the in-progress food order. Use exact menu item ids and size ids.",
        "parameters": {
            "type": "object",
            "properties": {
                "add_items": {
                    "type": "array",
                    "description": "Items to add (or increment if same id+size already on order).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Menu item id, e.g. chicken-karahi",
                            },
                            "size": {
                                "type": "string",
                                "description": "Size id for sized items; omit or null for fixed-price items",
                            },
                            "qty": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Quantity to add",
                            },
                        },
                        "required": ["id", "qty"],
                    },
                },
                "set_qty": {
                    "type": "array",
                    "description": "Set absolute quantity for an existing line item.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "size": {
                                "type": "string",
                                "description": "Size id; omit or null for fixed-price items",
                            },
                            "qty": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "New quantity; 0 removes the line",
                            },
                        },
                        "required": ["id", "qty"],
                    },
                },
                "remove_items": {
                    "type": "array",
                    "description": "Remove line items from the order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "size": {
                                "type": "string",
                                "description": "Size id; omit or null for fixed-price items",
                            },
                        },
                        "required": ["id"],
                    },
                },
                "delivery_address": {
                    "type": ["string", "null"],
                    "description": "Full delivery address",
                },
                "special_instructions": {
                    "type": ["string", "null"],
                    "description": "Notes like extra spicy, no salad, extra cheese",
                },
                "customer_phone": {
                    "type": ["string", "null"],
                    "description": "Optional caller phone number",
                },
                "confirm": {
                    "type": ["boolean", "null"],
                    "description": "Set true only after user explicitly confirms the full order",
                },
            },
        },
    },
}


def load_menu_data(path: Path | None = None) -> dict:
    menu_path = path or Path(__file__).parent.parent / "data" / "menu.json"
    with open(menu_path, encoding="utf-8") as f:
        return json.load(f)


def format_menu_for_prompt(menu_data: dict) -> str:
    lines: list[str] = []
    for category in menu_data.get("categories", []):
        for item in category.get("items", []):
            availability = "available" if item.get("available", True) else "unavailable"
            sizes = item.get("sizes")
            if sizes:
                size_part = ", ".join(f"{s['id']}={s['price_pkr']}" for s in sizes)
                price_part = f"sizes: {size_part}"
            else:
                price_part = f"price: {item['price_pkr']}"
            lines.append(
                f"- {item['id']} | {item['name']} / {item['name_urdu']} | {price_part} | {availability}"
            )
    return "\n".join(lines)


def build_system_prompt() -> str:
    menu_data = load_menu_data()
    menu_text = format_menu_for_prompt(menu_data)
    restaurant_name = menu_data.get("restaurant_name", "Restaurant")

    return f"""You are a friendly order-taker for {restaurant_name} in Kasur, Pakistan.

Customers speak Roman Urdu mixed with English.
Always reply in Roman English only (Latin script). Example: "2 chicken karahi add ho gaye. Delivery address bata dein."

MENU (use exact id and size values):
{menu_text}

RULES:
- Only order items from the menu above using exact ids and size ids.
- Customizations not on the menu (e.g. extra cheese, extra spicy) go in special_instructions, not as new items.
- Ask for delivery address before confirming.
- Summarize the order and PKR total before asking for final confirmation.
- If an item is not on the menu or unavailable, say so and suggest real alternatives.
- Use the update_order tool to change the order. Set confirm=true only when the customer explicitly confirms.
- Keep replies short and natural in Roman English. Do not use Urdu or Arabic script.
- Do not state item quantities or Rs totals in your message; the system shows the verified order summary."""


def _order_snapshot(order: Order) -> str:
    return json.dumps(order.model_dump(mode="json"), ensure_ascii=False)


def _assistant_to_message(assistant_message) -> dict:
    msg: dict = {"role": "assistant", "content": assistant_message.content}
    reasoning_details = getattr(assistant_message, "reasoning_details", None)
    if reasoning_details is None and hasattr(assistant_message, "model_extra"):
        reasoning_details = (assistant_message.model_extra or {}).get("reasoning_details")
    if reasoning_details:
        msg["reasoning_details"] = reasoning_details
    if assistant_message.tool_calls:        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_message.tool_calls
        ]
    return msg


def _api_messages_with_context(messages: list[dict], order: Order) -> list[dict]:
    api_messages = messages.copy()
    order_context = (
        f"\n\nCurrent order state:\n{_order_snapshot(order)}\n"
        f"Running total: Rs {calculate_total(order)}"
    )
    if api_messages and api_messages[0]["role"] == "system":
        api_messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + order_context,
        }
    return api_messages


def _rate_limit_wait_seconds(err: RateLimitError) -> float | None:
    """Return seconds to wait before retry, or None if daily limit (use fallback)."""
    msg = str(err).lower()
    if "per-day" in msg:
        return None

    body = getattr(err, "body", None) or {}
    if isinstance(body, dict):
        meta = body.get("error", {}).get("metadata", {})
        headers = meta.get("headers", {})
        reset_ms = headers.get("X-RateLimit-Reset")
        if reset_ms:
            wait = int(reset_ms) / 1000 - time.time() + 1
            return min(max(wait, 1), 120)
    return 5


def _call_openrouter(
    messages: list[dict], *, tool_choice: str = "auto"
) -> ChatCompletion:
    kwargs: dict = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "extra_body": {"reasoning": {"enabled": True}},
    }
    if tool_choice != "none":
        kwargs["tools"] = [UPDATE_ORDER_TOOL]
        kwargs["tool_choice"] = tool_choice
    return openrouter_client.chat.completions.create(**kwargs)


def _call_groq_fallback(
    messages: list[dict], *, tool_choice: str = "auto"
) -> ChatCompletion:
    if _groq_client is None:
        raise LLMRateLimitError(
            "OpenRouter rate limit reached and no GROQ_API_KEY fallback is configured. "
            "Wait for the daily reset, add OpenRouter credits, or set GROQ_API_KEY in .env."
        )
    kwargs: dict = {"model": GROQ_MODEL, "messages": messages}
    if tool_choice != "none":
        kwargs["tools"] = [UPDATE_ORDER_TOOL]
        kwargs["tool_choice"] = tool_choice
    return _groq_client.chat.completions.create(**kwargs)


def _extract_assistant_message(response: ChatCompletion):
    if not response.choices:
        raise LLMRateLimitError("LLM returned an empty response. Try again in a moment.")
    return response.choices[0].message


def _call_llm(messages: list[dict], *, tool_choice: str = "auto") -> ChatCompletion:
    last_err: Exception | None = None
    for provider_attempt in range(2):
        for attempt in range(3):
            try:
                response = _call_openrouter(messages, tool_choice=tool_choice)
                if response.choices:
                    return response
                last_err = LLMRateLimitError("OpenRouter returned empty choices.")
            except RateLimitError as err:
                last_err = err
                wait = _rate_limit_wait_seconds(err)
                if wait is None:
                    break
                if attempt < 2:
                    time.sleep(wait)
                    continue
                break
            except BadRequestError:
                raise

        try:
            response = _call_groq_fallback(messages, tool_choice=tool_choice)
            if response.choices:
                return response
            last_err = LLMRateLimitError("Groq returned empty choices.")
        except GroqRateLimitError as err:
            last_err = err

        if provider_attempt == 0:
            time.sleep(3)

    if isinstance(last_err, GroqRateLimitError):
        raise LLMRateLimitError(
            "Both OpenRouter and Groq are rate-limited. "
            "Wait and try again later, or add credits on OpenRouter."
        ) from last_err
    raise LLMRateLimitError(
        "LLM returned an empty response after retries. Try again in a moment."
    ) from last_err

def _process_tool_calls(
    order: Order, assistant_message, messages: list[dict]
) -> tuple[Order, list[dict]]:
    messages.append(_assistant_to_message(assistant_message))

    for tool_call in assistant_message.tool_calls or []:
        if tool_call.function.name != "update_order":
            continue

        args = json.loads(tool_call.function.arguments)
        args = {k: v for k, v in args.items() if v is not None}
        try:
            order = apply_order_update(order, args)
            result = {"success": True, "order": order.model_dump(mode="json")}
        except OrderError as e:
            result = {"success": False, "error": str(e)}

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    return order, messages


def _order_fingerprint(order: Order) -> str:
    return json.dumps(order.model_dump(mode="json"), sort_keys=True)


def chat_turn(
    user_message: str,
    order: Order,
    messages: list[dict],
) -> tuple[str, Order, list[dict], bool]:
    messages.append({"role": "user", "content": user_message})
    order_before = _order_fingerprint(order)

    reply_text = ""
    for _ in range(5):
        api_messages = _api_messages_with_context(messages, order)
        response = _call_llm(api_messages)
        assistant_message = _extract_assistant_message(response)

        if assistant_message.tool_calls:
            order, messages = _process_tool_calls(order, assistant_message, messages)
            continue

        messages.append(_assistant_to_message(assistant_message))
        reply_text = assistant_message.content or ""
        break

    if not reply_text.strip():
        api_messages = _api_messages_with_context(messages, order)
        response = _call_llm(api_messages, tool_choice="none")
        assistant_message = _extract_assistant_message(response)
        messages.append(_assistant_to_message(assistant_message))
        reply_text = assistant_message.content or ""

    is_complete = order.status == OrderStatus.CONFIRMED
    order_changed = _order_fingerprint(order) != order_before

    if is_complete:
        reply_text = build_confirmation_english(order)
    elif order_changed:
        reply_text = build_order_status_reply(order)

    return reply_text, order, messages, is_complete
