import json
import re
import time
from pathlib import Path

import config
from app.dialog_state import DialogStage, DialogState, infer_dialog_state
from app.intent import IntentType, ParsedIntent, parse_customer_intent
from app.llm import (
    AssistantResult,
    LLMRateLimitError,
    LLMServiceError,
    call_llm,
    resolve_ambiguous_intent,
)
from app.numbers_urdu import number_to_urdu_words
from app.order import (
    Order,
    OrderError,
    OrderStatus,
    add_item_to_order,
    apply_order_update,
    build_confirmation_urdu,
    calculate_total,
    change_item_size,
    confirm_order,
    remove_item_from_order,
    update_item_qty,
)
from app.replies import (
    ASK_ADDRESS_URDU,
    ASK_CLARIFY_URDU,
    EMPTY_ORDER_URDU,
    UNAVAILABLE_ITEM_URDU,
    build_address_saved_urdu,
    build_missing_item_urdu,
    build_missing_size_urdu,
    build_size_help_urdu,
    build_update_reply_urdu,
    build_verified_summary_urdu,
)


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
            spoken_name = item.get("spoken_name_urdu", item["name_urdu"])
            aliases = ", ".join(item.get("aliases", []))
            sizes = item.get("sizes")
            if sizes:
                size_part = ", ".join(
                    f"{s['id']} ({s.get('label_urdu', s['label'])})={s['price_pkr']}"
                    for s in sizes
                )
                price_part = f"sizes: {size_part}"
            else:
                price_part = f"price: {item['price_pkr']}"
            lines.append(
                f"- {item['id']} | English: {item['name']} | "
                f"say: {spoken_name} | aliases: {aliases} | "
                f"{price_part} | {availability}"
            )
    return "\n".join(lines)


def build_spoken_menu_urdu(menu_data: dict | None = None) -> str:
    """Build a deterministic, Urdu-only menu response for clear Piper speech."""
    data = menu_data or load_menu_data()
    parts = ["جی، ہمارے مینو میں یہ چیزیں دستیاب ہیں۔"]

    for category in data.get("categories", []):
        for item in category.get("items", []):
            if not item.get("available", True):
                continue

            name = item.get("spoken_name_urdu", item["name_urdu"])
            sizes = item.get("sizes")
            if sizes:
                choices = [
                    f"{size.get('label_urdu', size['label'])} "
                    f"{number_to_urdu_words(size['price_pkr'])} روپے"
                    for size in sizes
                ]
                parts.append(f"{name}، {'، '.join(choices)}۔")
            else:
                price_words = number_to_urdu_words(item["price_pkr"])
                parts.append(f"{name}، {price_words} روپے۔")

    parts.append("آپ کیا آرڈر کرنا چاہیں گے؟")
    return " ".join(parts)


def _is_menu_inquiry(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    menu_words = ("مینو", "منیو", "مینی", "menu")
    question_words = (
        "کیا",
        "چیز",
        "بتا",
        "دستیاب",
        "what",
        "list",
        "available",
        "have",
    )
    return any(word in normalized for word in menu_words) and any(
        word in normalized for word in question_words
    )


def normalize_menu_aliases(
    text: str, menu_data: dict | None = None
) -> str:
    """Replace known STT/menu aliases with the canonical spoken Urdu name."""
    data = menu_data or load_menu_data()
    replacements: list[tuple[str, str]] = []

    for category in data.get("categories", []):
        for item in category.get("items", []):
            if not item.get("available", True):
                continue
            canonical = item.get("spoken_name_urdu", item["name_urdu"])
            for alias in item.get("aliases", []):
                if alias and alias.casefold() != canonical.casefold():
                    replacements.append((alias, canonical))

    normalized = text
    for alias, canonical in sorted(
        replacements, key=lambda pair: len(pair[0]), reverse=True
    ):
        normalized = re.sub(
            re.escape(alias),
            canonical,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def build_system_prompt() -> str:
    menu_data = load_menu_data()
    menu_text = format_menu_for_prompt(menu_data)
    restaurant_name = menu_data.get("restaurant_name", "Restaurant")

    return f"""You are a friendly order-taker for {restaurant_name} in Kasur, Pakistan.

Customers speak Roman Urdu mixed with English (sometimes Urdu script from voice).
Always reply in simple, natural Urdu script suitable for text-to-speech.
This is simple order taking, not a general reasoning task. Act directly and keep every spoken reply brief.
Example: "آپ کا آرڈر شامل ہو گیا ہے۔ براہ کرم ڈیلیوری کا پتہ بتائیں۔"

MENU (use exact id and size values):
{menu_text}

RULES:
- Only order items from the menu above using exact ids and size ids.
- Customizations not on the menu (e.g. extra cheese, extra spicy) go in special_instructions, not as new items.
- Ask for delivery address before confirming.
- Summarize the order and PKR total before asking for final confirmation.
- If an item is not on the menu or unavailable, say so and suggest real alternatives.
- Use the update_order tool to change the order. Set confirm=true only when the customer explicitly confirms.
- Use each item's "say" value whenever mentioning it aloud. Recognize its aliases as the same item.
- Use Urdu size labels shown in parentheses; never speak internal ids or English size labels.
- Keep replies short and natural in Urdu script. Never insert Korean, Chinese, Cyrillic, or English words.
- Do not use Markdown bullets, parentheses, slashes, or decorative symbols because the reply is spoken by TTS.
- Do not state item quantities or Rs totals in your message; the system shows the verified order summary."""


def _order_snapshot(order: Order) -> str:
    return json.dumps(order.model_dump(mode="json"), ensure_ascii=False)


def _assistant_to_message(assistant_message: AssistantResult) -> dict:
    msg: dict = {"role": "assistant", "content": assistant_message.content}
    if assistant_message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
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


def _call_llm(
    messages: list[dict], *, tool_choice: str = "auto"
) -> AssistantResult:
    return call_llm(
        messages,
        [UPDATE_ORDER_TOOL],
        allow_tools=tool_choice != "none",
    )

def _process_tool_calls(
    order: Order, assistant_message: AssistantResult, messages: list[dict]
) -> tuple[Order, list[dict]]:
    messages.append(_assistant_to_message(assistant_message))

    for tool_call in assistant_message.tool_calls or []:
        if tool_call.name != "update_order":
            continue

        args = tool_call.arguments
        args = {k: v for k, v in args.items() if v is not None}
        try:
            order = apply_order_update(order, args)
            result = {"success": True, "order": order.model_dump(mode="json")}
        except OrderError as e:
            result = {"success": False, "error": str(e)}

        messages.append(
            {
                "role": "tool",
                "_tool_call_id": tool_call.id,
                "_tool_name": tool_call.name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    return order, messages


def _chat_turn_legacy(
    user_message: str,
    order: Order,
    messages: list[dict],
) -> tuple[str, Order, list[dict], bool]:
    normalized_user_message = normalize_menu_aliases(user_message)
    messages.append({"role": "user", "content": normalized_user_message})

    if _is_menu_inquiry(normalized_user_message):
        reply_text = build_spoken_menu_urdu()
        messages.append({"role": "assistant", "content": reply_text})
        return reply_text, order, messages, False

    reply_text = ""
    for _ in range(5):
        api_messages = _api_messages_with_context(messages, order)
        assistant_message = _call_llm(api_messages)

        if assistant_message.tool_calls:
            order, messages = _process_tool_calls(order, assistant_message, messages)
            continue

        messages.append(_assistant_to_message(assistant_message))
        reply_text = assistant_message.content or ""
        break

    if not reply_text.strip():
        api_messages = _api_messages_with_context(messages, order)
        assistant_message = _call_llm(api_messages, tool_choice="none")
        messages.append(_assistant_to_message(assistant_message))
        reply_text = assistant_message.content or ""

    is_complete = order.status == OrderStatus.CONFIRMED

    if is_complete:
        reply_text = build_confirmation_urdu(order)

    return reply_text, order, messages, is_complete


def _menu_item(item_id: str) -> dict | None:
    for category in load_menu_data().get("categories", []):
        for item in category.get("items", []):
            if item.get("id") == item_id:
                return item
    return None


def _matching_order_lines(order: Order, item_id: str, size_id: str | None):
    lines = [line for line in order.items if line.id == item_id]
    if size_id is not None:
        lines = [line for line in lines if line.size == size_id]
    return lines


def _size_labels(item: dict) -> list[str]:
    return [
        size.get("label_urdu", size.get("label", size["id"]))
        for size in item.get("sizes", [])
    ]


def _reask_size(
    item: dict,
    quantity: int,
    instructions: str | None,
    state: DialogState,
) -> str:
    """Re-ask for size; sole site that increments size_attempts (once per turn)."""
    repeat = state.pending_item_id == item["id"]
    state.await_size(item["id"], quantity, instructions)
    if repeat:
        state.size_attempts += 1
    labels = _size_labels(item)
    if state.size_attempts >= 3:
        return build_size_help_urdu(labels)
    return build_missing_size_urdu(labels)


def _dispatch_deterministic_intent(
    intent: ParsedIntent,
    user_message: str,
    order: Order,
    state: DialogState,
    *,
    allow_fallback: bool,
) -> tuple[str, bool]:
    """Validate/apply one intent and return (verified reply, complete)."""
    if intent.intent == IntentType.LIST_MENU:
        return build_spoken_menu_urdu(), False

    if intent.intent == IntentType.SHOW_ORDER:
        ask_confirm = bool(order.items and order.delivery_address)
        if ask_confirm:
            state.stage = DialogStage.AWAITING_CONFIRMATION
            state.summary_presented = True
        return build_verified_summary_urdu(
            order, ask_for_confirmation=ask_confirm
        ), False

    if intent.intent == IntentType.UNKNOWN_ITEM:
        return UNAVAILABLE_ITEM_URDU, False

    if intent.intent == IntentType.ADD_ITEM:
        if not intent.item_id:
            return build_missing_item_urdu(), False
        item = _menu_item(intent.item_id)
        if item is None or not item.get("available", True) or intent.available is False:
            return UNAVAILABLE_ITEM_URDU, False
        if item.get("sizes") and not intent.size_id:
            return _reask_size(
                item,
                intent.quantity or 1,
                intent.instructions,
                state,
            ), False
        try:
            add_item_to_order(
                order,
                intent.item_id,
                intent.size_id,
                intent.quantity or 1,
            )
        except OrderError:
            return ASK_CLARIFY_URDU, False
        if intent.clear_address:
            order.delivery_address = None
        if intent.instructions:
            if order.special_instructions:
                order.special_instructions = (
                    f"{order.special_instructions}, {intent.instructions}"
                )
            else:
                order.special_instructions = intent.instructions
        state.mark_mutation()
        state.clear_pending_size()
        state.remember_item(intent.item_id, intent.size_id)
        return build_update_reply_urdu(order, state, "add"), False

    if intent.intent == IntentType.REMOVE_ITEM:
        if not intent.item_id:
            return build_missing_item_urdu(), False
        lines = _matching_order_lines(order, intent.item_id, intent.size_id)
        if not lines:
            return "یہ آئٹم آپ کے موجودہ آرڈر میں شامل نہیں ہے۔", False
        if len(lines) > 1:
            return "براہ کرم بتائیں کون سا سائز نکالنا ہے۔", False
        line = lines[0]
        try:
            remove_item_from_order(order, line.id, line.size)
        except OrderError:
            return ASK_CLARIFY_URDU, False
        state.mark_mutation()
        state.clear_removed_item(line.id, line.size)
        return build_update_reply_urdu(order, state, "remove"), False

    if intent.intent == IntentType.SET_QUANTITY:
        if intent.quantity is None:
            return "براہ کرم نئی تعداد بتائیں۔", False
        item_id = intent.item_id or state.last_item_id
        size_id = intent.size_id if intent.item_id else state.last_item_size
        if not item_id:
            return build_missing_item_urdu(), False
        lines = _matching_order_lines(order, item_id, size_id)
        if len(lines) != 1:
            return build_missing_item_urdu(), False
        line = lines[0]
        try:
            update_item_qty(order, line.id, line.size, intent.quantity)
        except OrderError:
            return ASK_CLARIFY_URDU, False
        state.mark_mutation()
        if intent.quantity == 0:
            state.clear_removed_item(line.id, line.size)
        else:
            state.remember_item(line.id, line.size)
        return build_update_reply_urdu(order, state, "quantity"), False

    if intent.intent == IntentType.CHANGE_SIZE:
        if not intent.item_id or not intent.size_id:
            return ASK_CLARIFY_URDU, False
        lines = _matching_order_lines(order, intent.item_id, None)
        if state.last_item_id == intent.item_id and state.last_item_size is not None:
            selected = [line for line in lines if line.size == state.last_item_size]
            if selected:
                lines = selected
        if len(lines) != 1:
            return "براہ کرم موجودہ آئٹم کا سائز واضح کریں۔", False
        old_line = lines[0]
        try:
            change_item_size(order, old_line.id, old_line.size, intent.size_id)
        except OrderError:
            return ASK_CLARIFY_URDU, False
        state.mark_mutation()
        state.remember_item(old_line.id, intent.size_id)
        return build_update_reply_urdu(order, state, "size"), False

    if intent.intent == IntentType.SET_ADDRESS:
        if not intent.address:
            return ASK_ADDRESS_URDU, False
        order.delivery_address = intent.address
        state.mark_mutation()
        return build_address_saved_urdu(order, state), False

    if intent.intent == IntentType.SET_INSTRUCTIONS:
        if not intent.instructions:
            return "براہ کرم ہدایات دوبارہ بتائیں۔", False
        order.special_instructions = intent.instructions
        state.mark_mutation()
        return build_update_reply_urdu(order, state, "instructions"), False

    if intent.intent == IntentType.CONFIRM_ORDER:
        if not order.items:
            return EMPTY_ORDER_URDU, False
        if not order.delivery_address:
            state.stage = DialogStage.AWAITING_ADDRESS
            state.summary_presented = False
            return ASK_ADDRESS_URDU, False
        if state.stage != DialogStage.AWAITING_CONFIRMATION or not state.summary_presented:
            state.stage = DialogStage.AWAITING_CONFIRMATION
            state.summary_presented = True
            return build_verified_summary_urdu(
                order, ask_for_confirmation=True
            ), False
        confirm_order(order)
        state.stage = DialogStage.COMPLETE
        return build_confirmation_urdu(order), True

    if intent.intent == IntentType.CANCEL:
        return "ٹھیک ہے، گفتگو ختم کر دی گئی ہے۔", False

    if (
        intent.intent == IntentType.AMBIGUOUS
        and intent.clarification_reason
        in {"address_denied", "order_while_awaiting_address"}
    ):
        if intent.clear_address:
            order.delivery_address = None
            state.mark_mutation()
            state.sync_stage(order)
            return "ٹھیک ہے، پچھلا پتہ ہٹا دیا گیا ہے۔ درست پتہ بتائیں۔", False
        return (
            "میں اسے پتہ محفوظ نہیں کروں گی۔ براہ کرم آئٹم کا نام دوبارہ بتائیں۔",
            False,
        )

    if intent.intent == IntentType.AMBIGUOUS and allow_fallback:
        if not config.HYBRID_LLM_FALLBACK_ENABLED:
            if state.pending_item_id:
                pending = _menu_item(state.pending_item_id)
                if pending is not None:
                    return _reask_size(
                        pending,
                        state.pending_quantity,
                        state.pending_instructions,
                        state,
                    ), False
            return ASK_CLARIFY_URDU, False
        state.last_path = f"{config.LLM_PROVIDER}_fallback"
        fallback_start = time.perf_counter()
        try:
            resolved = resolve_ambiguous_intent(user_message, order, state)
        except LLMServiceError:
            state.last_llm_seconds = time.perf_counter() - fallback_start
            if state.pending_item_id:
                pending = _menu_item(state.pending_item_id)
                if pending is not None:
                    return _reask_size(
                        pending,
                        state.pending_quantity,
                        state.pending_instructions,
                        state,
                    ), False
            return ASK_CLARIFY_URDU, False
        state.last_llm_seconds = time.perf_counter() - fallback_start
        if resolved.intent == IntentType.AMBIGUOUS:
            if state.pending_item_id:
                pending = _menu_item(state.pending_item_id)
                if pending is not None:
                    return _reask_size(
                        pending,
                        state.pending_quantity,
                        state.pending_instructions,
                        state,
                    ), False
            return ASK_CLARIFY_URDU, False
        return _dispatch_deterministic_intent(
            resolved,
            user_message,
            order,
            state,
            allow_fallback=False,
        )

    return ASK_CLARIFY_URDU, False


def _chat_turn_hybrid(
    user_message: str,
    order: Order,
    messages: list[dict],
    state: DialogState,
) -> tuple[str, Order, list[dict], bool]:
    state.last_path = "deterministic"
    state.last_llm_seconds = 0.0
    parser_start = time.perf_counter()
    intent = parse_customer_intent(user_message, state, order)
    state.last_parser_seconds = time.perf_counter() - parser_start

    messages.append({"role": "user", "content": user_message})
    reply, complete = _dispatch_deterministic_intent(
        intent,
        user_message,
        order,
        state,
        allow_fallback=True,
    )
    messages.append({"role": "assistant", "content": reply})
    return reply, order, messages, complete


def chat_turn(
    user_message: str,
    order: Order,
    messages: list[dict],
    dialog_state: DialogState | None = None,
) -> tuple[str, Order, list[dict], bool]:
    """Process one turn through hybrid routing or the rollback legacy path."""
    if not config.HYBRID_INTENT_ENABLED:
        return _chat_turn_legacy(user_message, order, messages)
    state = dialog_state or infer_dialog_state(order)
    return _chat_turn_hybrid(user_message, order, messages, state)
