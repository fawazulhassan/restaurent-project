import contextvars
import json
import re
import time
from pathlib import Path

import config

_AMBIGUOUS_LLM_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ambiguous_llm_allowed",
    default=True,
)
from app.basket_edit import BasketEdit, apply_basket_edits
from app.confirmation_urdu import classify_address_confirmation, classify_confirmation_reply, classify_more_items_reply, classify_order_confirmation, load_menu_noise
from app.dialog_state import DialogStage, DialogState, ItemRequest, PendingChoice, infer_dialog_state
from app.intent import (
    IntentType,
    ParsedIntent,
    classify_address_reply,
    extract_address_candidate,
    is_explicit_order_command,
    parse_customer_intent,
    quantity_evidence,
    normalize_customer_text,
    mentioned_item_ids,
    addition_is_prohibited,
    contains_add_word,
    removal_is_prohibited,
    structured_addition,
    size_direction,
    sizes_in_text,
)
from app.llm import (
    AssistantResult,
    LLMRateLimitError,
    LLMServiceError,
    call_llm,
    record_model_turn,
    resolve_ambiguous_intent,
    resolve_unhandled_turn,
    extract_address_with_llm,
    resolve_basket_edits,
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
    ASK_CONFIRM_URDU,
    ASK_MORE_ITEMS_URDU,
    ASK_ORDER_URDU,
    ASK_QUANTITY_URDU,
    ASK_WHAT_TO_CHANGE_URDU,
    EMPTY_ORDER_URDU,
    HUMAN_HANDOFF_CONFIRM_URDU,
    HUMAN_HANDOFF_URDU,
    UNAVAILABLE_ITEM_URDU,
    build_address_saved_urdu,
    build_more_items_reply_urdu,
    build_confirm_address_urdu,
    build_item_choice_urdu,
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
        menu = json.load(f)
    load_menu_noise(menu)
    return menu


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

    parts.append(ASK_ORDER_URDU)
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

    # One pass: do not rewrite the "naan" inside a replacement for garlic naan.
    for category in data.get("categories", []):
        for item in category.get("items", []):
            canonical = item.get("spoken_name_urdu", item["name_urdu"])
            replacements.append((canonical, canonical))
    mapping = {alias.casefold(): canonical for alias, canonical in replacements}
    if not mapping:
        return text
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in sorted(mapping, key=len, reverse=True)) + r")(?!\w)"
    return re.sub(pattern, lambda match: mapping[match.group().casefold()], text, flags=re.IGNORECASE)


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
    if state.size_attempts >= 3:
        return build_size_help_urdu(item)
    return build_missing_size_urdu(item)


def _address_handoff(state: DialogState) -> str:
    state.clear_pending_address()
    state.address_rejections = 0
    state.reply_streak = 0
    state.stage = DialogStage.HANDOFF
    return HUMAN_HANDOFF_URDU


def _save_pending_address(order: Order, state: DialogState) -> str:
    address = state.confirm_pending_address(order)
    if address is None:
        state.stage = DialogStage.AWAITING_ADDRESS
        return ASK_ADDRESS_URDU
    state.mark_mutation()
    if order.items:
        state.enter_confirmation()
        state.summary_presented = True
        return build_address_saved_urdu(order, state)
    state.stage = DialogStage.ORDERING
    state.summary_presented = False
    return build_address_saved_urdu(order, state)


def _capture_address(user_message: str, state: DialogState) -> tuple[str, bool]:
    """Extract once, then read back; only local confirmation can save it."""
    state.last_path = "openai_address_extraction"
    started = time.perf_counter()
    diagnostics = {}
    try:
        candidate = extract_address_with_llm(user_message, diagnostics=diagnostics)
    except LLMServiceError as exc:
        candidate = None
        diagnostics.update(status=type(exc).__name__, reason=str(exc))
    finally:
        state.last_llm_seconds += time.perf_counter() - started
    state.last_address_extraction_detail = " ".join(
        f"{key}={value}" for key, value in diagnostics.items()
    )
    if candidate:
        state.await_address_confirmation(candidate)
        return build_confirm_address_urdu(candidate), False
    state.address_confirmation_attempts += 1
    if state.address_confirmation_attempts >= 3:
        return _address_handoff(state), False
    if state.pending_address:
        return (
            "معذرت، نیا پتہ واضح نہیں سمجھ سکی۔ "
            + build_confirm_address_urdu(state.pending_address), False
        )
    if "reason" in diagnostics:
        return "معذرت، پتہ سمجھنے کی سروس میں مسئلہ ہے۔ براہ کرم دوبارہ کوشش کریں۔", False
    return ASK_ADDRESS_URDU, False


def _handle_address_confirmation_turn(
    user_message: str,
    order: Order,
    state: DialogState,
) -> tuple[str, bool] | None:
    if state.stage != DialogStage.AWAITING_ADDRESS_CONFIRMATION and not state.pending_address:
        return None
    if not state.pending_address:
        state.stage = DialogStage.AWAITING_ADDRESS
        return ASK_ADDRESS_URDU, False
    if state.stage != DialogStage.AWAITING_ADDRESS_CONFIRMATION:
        state.stage = DialogStage.AWAITING_ADDRESS_CONFIRMATION
        return build_confirm_address_urdu(state.pending_address), False

    # Local confirmation is authoritative. LLM extraction has no yes/no label.
    classification = (
        classify_address_confirmation(user_message)
        if config.ADDRESS_LLM_EXTRACTION_ENABLED
        else classify_address_reply(user_message)
    )
    if classification == "yes":
        return _save_pending_address(order, state), False
    if classification == "no":
        state.address_rejections += 1
        state.clear_pending_address()
        if state.address_rejections >= 3:
            return _address_handoff(state), False
        if order.delivery_address:
            state.enter_confirmation()
            state.summary_presented = True
            return build_verified_summary_urdu(order, ask_for_confirmation=True), False
        state.stage = DialogStage.AWAITING_ADDRESS
        state.summary_presented = False
        return ASK_ADDRESS_URDU, False

    if config.ADDRESS_LLM_EXTRACTION_ENABLED:
        return _capture_address(user_message, state)
    if classification == "new_address":
        candidate = extract_address_candidate(user_message, confirmation_correction=True)
        if candidate:
            state.await_address_confirmation(candidate)
            return build_confirm_address_urdu(candidate), False
    state.address_confirmation_attempts += 1
    if state.address_confirmation_attempts >= 3:
        return _address_handoff(state), False
    return build_confirm_address_urdu(state.pending_address), False


def _apply_additions(
    requests: list[ItemRequest], clear_address: bool, order: Order, state: DialogState,
) -> tuple[str, bool]:
    # Stage the entire request so an unavailable item or missing size cannot
    # silently leave a partial order behind.
    for request in requests:
        item = _menu_item(request.item_id)
        if item is None or not item.get("available", True):
            return UNAVAILABLE_ITEM_URDU, False
        if request.size_id and not any(size["id"] == request.size_id for size in item.get("sizes", [])):
            return ASK_CLARIFY_URDU, False
    for request in requests:
        item = _menu_item(request.item_id)
        sizes = item.get("sizes") or []
        if len(sizes) == 1 and not request.size_id:
            request.size_id = sizes[0]["id"]
        if request.size_id:
            continue
        existing_sizes = {
            line.size for line in order.items if line.id == request.item_id and line.size
        }
        if len(existing_sizes) == 1:
            request.size_id = next(iter(existing_sizes))
    for request in requests:
        item = _menu_item(request.item_id)
        if item.get("sizes") and not request.size_id:
            state.pending_additions = [part.model_copy(deep=True) for part in requests]
            return _reask_size(item, request.quantity, request.instructions, state), False
    trial = order.model_copy(deep=True)
    try:
        for request in requests:
            add_item_to_order(trial, request.item_id, request.size_id, request.quantity)
            if request.instructions:
                trial.special_instructions = ", ".join(filter(None, [trial.special_instructions, request.instructions]))
    except OrderError:
        return ASK_CLARIFY_URDU, False
    order.items = trial.items
    order.special_instructions = trial.special_instructions
    if clear_address:
        order.delivery_address = None
    state.mark_mutation()
    state.clear_pending_size()
    state.pending_additions = []
    state.quantity_correction_item_id = None
    state.quantity_correction_size_id = None
    state.quantity_correction_action = "set"
    state.remember_item(requests[-1].item_id, requests[-1].size_id)
    return build_update_reply_urdu(order, state, "add"), False


def _handle_more_items_turn(user_message: str, order: Order, state: DialogState) -> tuple[str, bool] | None:
    if state.stage != DialogStage.AWAITING_MORE_ITEMS:
        return None
    answer = classify_more_items_reply(user_message)
    if answer == "no":
        state.basket_edit_context = []
        state.quantity_correction_item_id = None
        state.quantity_correction_size_id = None
        state.quantity_correction_action = "set"
        if order.delivery_address:
            state.enter_confirmation()
            state.summary_presented = True
            return build_verified_summary_urdu(order, ask_for_confirmation=True), False
        state.stage = DialogStage.AWAITING_ADDRESS
        return ASK_ADDRESS_URDU, False
    if answer == "yes":
        return "آپ مزید کیا شامل کرنا چاہیں گے؟", False
    # A new item named directly goes to the normal parser without a separate yes.
    return None


def _finish_basket_edits(edits: list[BasketEdit], order: Order, state: DialogState) -> tuple[str, bool]:
    try:
        apply_basket_edits(order, edits)
    except OrderError as exc:
        state.last_basket_edit_detail = f"rejected={exc}"
        return "براہ کرم بتائیں کون سا آئٹم شامل، کم یا ختم کرنا ہے اور کتنی تعداد؟", False
    state.mark_mutation()
    state.quantity_correction_item_id = None
    state.quantity_correction_size_id = None
    state.quantity_correction_action = "set"
    state.basket_edit_context = []
    for edit in edits:
        if any(line.id == edit.item_id and (edit.size_id is None or line.size == edit.size_id) for line in order.items):
            state.remember_item(edit.item_id, edit.size_id)
        else:
            state.clear_removed_item(edit.item_id, edit.size_id)
    return build_update_reply_urdu(order, state, "add"), False


def _basket_snapshot(state: DialogState) -> dict:
    return {
        "context": list(state.basket_edit_context),
        "item": state.quantity_correction_item_id,
        "size": state.quantity_correction_size_id,
        "action": state.quantity_correction_action,
    }


def _reject_basket(state: DialogState, reason: str, saved: dict) -> tuple[str, bool]:
    state.basket_edit_context = saved["context"]
    state.quantity_correction_item_id = saved["item"]
    state.quantity_correction_size_id = saved["size"]
    state.quantity_correction_action = saved["action"]
    state.last_basket_edit_detail = f"rejected={reason}"
    return "براہ کرم بتائیں کون سا آئٹم شامل، کم یا ختم کرنا ہے اور کتنی تعداد؟", False


def _prohibited_edits(normalized: str, edits: list[BasketEdit]) -> list[BasketEdit] | None:
    """Drop ops a negated verb targets. None means the whole plan is prohibited."""
    tokens = normalized.split()
    rules = (
        (
            re.compile(
                r"(?:do not|don't|don t|dont|not)\s+(?:remove|delete|reduce|subtract)\b|"
                r"(?:مت|mat)\s+(?:ہٹا\w*|نکال\w*|hata\w*|nikal\w*|kam)"
            ),
            {"remove", "reduce"},
        ),
        (
            re.compile(r"(?:do not|don't|don t|dont|not)\s+add\b|(?:مت|mat)\s+(?:add|شامل)"),
            {"add"},
        ),
    )
    current = list(edits)
    for pattern, actions in rules:
        match = pattern.search(normalized)
        if not match:
            continue
        start = normalized[:match.start()].count(" ")
        window = tokens[max(0, start - config.NEGATION_ITEM_WINDOW):start + config.NEGATION_ITEM_WINDOW]
        targets = mentioned_item_ids(" ".join(window))
        if not targets:
            if any(edit.action in actions for edit in current):
                return None
            continue
        window_sizes = set(sizes_in_text(" ".join(window)))
        kept = []
        for edit in current:
            targeted = edit.action in actions and edit.item_id in targets and (
                not window_sizes or edit.size_id is None or edit.size_id in window_sizes
            )
            if not targeted:
                kept.append(edit)
        current = kept
        if not current:
            return None
    return current


def _replace_size_error(text: str, edits: list[BasketEdit]) -> str | None:
    replaces = [edit for edit in edits if edit.action == "replace_size"]
    if not replaces:
        return None
    if len(replaces) >= 2 and len(sizes_in_text(text)) >= 2:
        return "both_lemmas_no_marker"
    old_size, new_size, error = size_direction(text)
    if error:
        return error
    for edit in replaces:
        if new_size is None or edit.new_size_id != new_size:
            return "unknown_new_size"
        if old_size and edit.size_id not in (None, old_size):
            return "unknown_new_size"
    return None


def _resolve_basket_change(intent: ParsedIntent, text: str, order: Order, state: DialogState) -> tuple[str, bool]:
    clarify = "براہ کرم بتائیں کون سا آئٹم شامل، کم یا ختم کرنا ہے اور کتنی تعداد؟"
    saved = _basket_snapshot(state)
    state.basket_edit_context = [*state.basket_edit_context, text[:1000]][-2:]
    if intent.edit_action_hint == "reduce" and intent.item_id:
        state.quantity_correction_item_id = intent.item_id
        state.quantity_correction_size_id = intent.size_id
        state.quantity_correction_action = "reduce"
    if not config.HYBRID_LLM_FALLBACK_ENABLED and not config.BASKET_EDIT_LLM_ENABLED:
        state.last_basket_edit_detail = "status=disabled"
        return clarify, False
    state.last_path = "openai_basket_edit"
    started = time.perf_counter()
    try:
        plan = resolve_basket_edits(text, order, state)
    except LLMServiceError as exc:
        state.last_basket_edit_detail = f"status={type(exc).__name__} reason={exc}"
        return clarify, False
    finally:
        state.last_llm_seconds = time.perf_counter() - started
    state.last_basket_edit_detail = f"status={plan.status} confidence={plan.confidence:g} operations={len(plan.edits)}"
    if plan.status != "apply" or plan.confidence < 0.9 or not plan.edits:
        return clarify, False
    normalized, _ = normalize_customer_text(text)
    edits = _prohibited_edits(normalized, list(plan.edits))
    if edits is None:
        return _reject_basket(state, "prohibited_verb", saved)
    direction_error = _replace_size_error(normalized, edits)
    if direction_error:
        return _reject_basket(state, direction_error, saved)
    numbers, rejected = quantity_evidence(text)
    named_items = mentioned_item_ids(text)
    if named_items and any(edit.item_id not in named_items and edit.action != "add" for edit in edits):
        return _reject_basket(state, "item_mismatch", saved)
    for edit in edits:
        if edit.action in {"add", "set", "reduce"}:
            default_addition = edit.action == "add" and edit.quantity == 1 and not numbers
            if not default_addition and (edit.quantity not in numbers or edit.quantity in rejected):
                return _reject_basket(state, "quantity_evidence", saved)
        elif (
            edit.action == "remove"
            and len(plan.edits) == 1
            and numbers
            and not re.search(r"(?<!\w)(?:all|تمام|سارے|سب)(?!\w)", normalized)
        ):
            # A counted removal is a subtraction, even when a novel phrasing
            # did not supply a local action hint.
            return _reject_basket(state, "counted_removal_requires_reduce", saved)
    if all(edit.action == "add" for edit in edits):
        reply, complete = _apply_additions([
            ItemRequest(item_id=edit.item_id, size_id=edit.size_id, quantity=edit.quantity)
            for edit in edits
        ], False, order, state)
        state.basket_edit_context = []
        return reply, complete
    try:
        reply, complete = _finish_basket_edits(edits, order, state)
    except OrderError as exc:
        return _reject_basket(state, str(exc), saved)
    if state.last_basket_edit_detail.startswith("rejected="):
        return _reject_basket(state, state.last_basket_edit_detail.removeprefix("rejected="), saved)
    return reply, complete


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
            state.enter_confirmation()
            state.summary_presented = True
        return build_verified_summary_urdu(
            order, ask_for_confirmation=ask_confirm
        ), False

    if intent.intent == IntentType.UNKNOWN_ITEM:
        return UNAVAILABLE_ITEM_URDU, False

    if intent.intent == IntentType.AMBIGUOUS and intent.clarification_reason == "basket_edit":
        return _resolve_basket_change(intent, user_message, order, state)

    if intent.intent == IntentType.REDUCE_QUANTITY:
        if not intent.item_id or intent.quantity is None:
            return ASK_QUANTITY_URDU, False
        return _finish_basket_edits([
            BasketEdit(action="reduce", item_id=intent.item_id, size_id=intent.size_id, quantity=intent.quantity)
        ], order, state)

    if intent.intent in {IntentType.ADD_ITEM, IntentType.ADD_ITEMS}:
        if intent.clarification_reason == "missing_size" and state.pending_item_id:
            pending = _menu_item(state.pending_item_id)
            if pending is not None:
                return _reask_size(
                    pending, state.pending_quantity, state.pending_instructions, state
                ), False
        if intent.intent == IntentType.ADD_ITEMS:
            requests = [*state.pending_additions, *intent.add_items]
            if not requests:
                return ASK_CLARIFY_URDU, False
        else:
            if not intent.item_id:
                return build_missing_item_urdu(), False
            if state.pending_additions:
                requests = [request.model_copy(deep=True) for request in state.pending_additions]
                unresolved = next((request for request in requests
                                   if request.item_id == intent.item_id and not request.size_id), None)
                if unresolved and intent.size_id:
                    unresolved.size_id = intent.size_id
                elif state.pending_item_id == intent.item_id and unresolved:
                    unresolved.size_id = intent.size_id
                else:
                    requests = [ItemRequest(
                        item_id=intent.item_id, size_id=intent.size_id,
                        quantity=intent.quantity or 1, instructions=intent.instructions,
                    ), *requests]
            else:
                requests = [ItemRequest(
                    item_id=intent.item_id, size_id=intent.size_id,
                    quantity=intent.quantity or 1, instructions=intent.instructions,
                )]
        return _apply_additions(requests, intent.clear_address, order, state)

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
        state.quantity_correction_item_id = None
        state.quantity_correction_size_id = None
        state.quantity_correction_action = "set"
        if intent.quantity == 0:
            state.clear_removed_item(line.id, line.size)
        else:
            state.remember_item(line.id, line.size)
        return build_update_reply_urdu(order, state, "quantity"), False

    if intent.intent == IntentType.CHANGE_SIZE:
        if not intent.item_id:
            return ASK_CLARIFY_URDU, False
        _old_size, directed_new, direction_error = size_direction(user_message)
        new_size = directed_new or intent.size_id
        if direction_error or not new_size:
            return ASK_CLARIFY_URDU, False
        lines = _matching_order_lines(order, intent.item_id, None)
        if _old_size:
            selected = [line for line in lines if line.size == _old_size]
            if selected:
                lines = selected
        elif state.last_item_id == intent.item_id and state.last_item_size is not None:
            selected = [line for line in lines if line.size == state.last_item_size]
            if selected:
                lines = selected
        if len(lines) != 1:
            return "براہ کرم موجودہ آئٹم کا سائز واضح کریں۔", False
        old_line = lines[0]
        try:
            change_item_size(order, old_line.id, old_line.size, new_size)
        except OrderError:
            return ASK_CLARIFY_URDU, False
        state.mark_mutation()
        state.remember_item(old_line.id, new_size)
        return build_update_reply_urdu(order, state, "size"), False

    if intent.intent == IntentType.SET_ADDRESS:
        if config.ADDRESS_LLM_EXTRACTION_ENABLED:
            return _capture_address(user_message, state)
        if not intent.address:
            return ASK_ADDRESS_URDU, False
        state.await_address_confirmation(intent.address)
        return build_confirm_address_urdu(intent.address), False

    if intent.intent == IntentType.SET_INSTRUCTIONS:
        if not intent.instructions:
            return "براہ کرم ہدایات دوبارہ بتائیں۔", False
        order.special_instructions = intent.instructions
        state.mark_mutation()
        return build_update_reply_urdu(order, state, "instructions"), False

    if intent.intent == IntentType.CONFIRM_ORDER:
        if not order.items:
            return EMPTY_ORDER_URDU, False
        if state.pending_address:
            state.stage = DialogStage.AWAITING_ADDRESS_CONFIRMATION
            return build_confirm_address_urdu(state.pending_address), False
        if not order.delivery_address:
            state.stage = DialogStage.AWAITING_ADDRESS
            state.summary_presented = False
            return ASK_ADDRESS_URDU, False
        if state.stage != DialogStage.AWAITING_CONFIRMATION or not state.summary_presented:
            state.enter_confirmation()
            state.summary_presented = True
            return build_verified_summary_urdu(
                order, ask_for_confirmation=True
            ), False
        confirm_order(order)
        state.stage = DialogStage.COMPLETE
        state.clear_pending_choice()
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

    if (
        intent.intent == IntentType.AMBIGUOUS
        and allow_fallback
        and intent.clarification_reason != "quantity_correction"
        and not _AMBIGUOUS_LLM_ALLOWED.get()
    ):
        if intent.clarification_reason == "missing_size" and state.pending_item_id:
            pending = _menu_item(state.pending_item_id)
            if pending is not None:
                return _reask_size(
                    pending, state.pending_quantity, state.pending_instructions, state
                ), False
        return ASK_CLARIFY_URDU, False

    if intent.intent == IntentType.AMBIGUOUS and allow_fallback:
        if intent.clarification_reason == "quantity_correction":
            state.quantity_correction_item_id = intent.item_id
            state.quantity_correction_size_id = intent.size_id
            state.quantity_correction_action = "set"
        if not config.HYBRID_LLM_FALLBACK_ENABLED:
            if intent.clarification_reason == "quantity_correction":
                return ASK_QUANTITY_URDU, False
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
            if intent.clarification_reason == "quantity_correction":
                return ASK_QUANTITY_URDU, False
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
        if intent.clarification_reason == "quantity_correction":
            numbers, rejected = quantity_evidence(user_message)
            if (
                resolved.intent != IntentType.SET_QUANTITY
                or resolved.confidence < 0.9
                or resolved.quantity not in numbers
                or resolved.quantity in rejected
                or resolved.item_id != intent.item_id
                or (intent.size_id is not None and resolved.size_id != intent.size_id)
            ):
                return ASK_QUANTITY_URDU, False
        if intent.clarification_reason == "multiple_items" and resolved.intent == IntentType.ADD_ITEM:
            return "براہ کرم تمام آئٹمز کے نام، تعداد اور سائز دوبارہ بتائیں۔", False
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


def _hint_item_ids(intent: ParsedIntent) -> list[str] | None:
    """Item ids of a high-confidence exact add. None for every other hint."""
    if intent.dropped_unknown:
        return None
    if intent.intent == IntentType.ADD_ITEM and intent.item_id:
        return [intent.item_id]
    if (
        intent.intent == IntentType.ADD_ITEMS
        and intent.add_items
        and all(part.item_id for part in intent.add_items)
    ):
        return [part.item_id for part in intent.add_items]
    return None


def _script_hint(intent: ParsedIntent, state: DialogState) -> dict:
    return {
        "intent": intent.intent.value,
        "reason": intent.clarification_reason,
        "item_ids": _hint_item_ids(intent) or [],
        "pending_size_item": state.pending_item_id,
        "pending_choices": list(state.pending_choices),
    }


def _restricted_model_stage(state: DialogState) -> bool:
    return (
        state.stage in {
            DialogStage.AWAITING_SIZE,
            DialogStage.AWAITING_ADDRESS,
            DialogStage.AWAITING_ADDRESS_CONFIRMATION,
        }
        or bool(state.pending_choices)
    )


_REPLY_STAGES = {
    DialogStage.ORDERING,
    DialogStage.AWAITING_MORE_ITEMS,
    DialogStage.AWAITING_CONFIRMATION,
    DialogStage.AWAITING_ORDER_CHANGE,
}


def _reply_allowed(state: DialogState, intent: ParsedIntent) -> bool:
    """A reply is spoken only when the parser hint did not already win."""
    if state.reply_streak >= 3:
        return False
    if intent.dropped_unknown or intent.clarification_reason == "quantity_correction":
        return False
    if intent.clarification_reason in {"item_choice", "choice_selected", "choice_retry"}:
        return False
    if intent.choice_groups or state.pending_choices or _hint_item_ids(intent):
        return False
    return state.stage in _REPLY_STAGES


def _accept_own_action(decision, state: DialogState, intent: ParsedIntent) -> bool:
    """True when the model's own action may replace the script."""
    if decision.validation != "ok" or decision.action == "use_script":
        return False
    if decision.action == "reply":
        return _reply_allowed(state, intent)
    if state.pending_item_id:
        for item in decision.items:
            item["size_id"] = None
    if _restricted_model_stage(state) and decision.action not in {"unavailable", "menu"}:
        return False
    hint = _hint_item_ids(intent)
    model_ids = [item["item_id"] for item in decision.items]
    disagrees = decision.action in {"clarify", "other"} or (
        decision.action == "add_items" and hint is not None and model_ids != hint
    )
    if disagrees and (hint is not None or state.stage == DialogStage.AWAITING_SIZE):
        return False
    return True


def _redacted_utterance(user_message: str, state: DialogState, intent: ParsedIntent) -> str:
    if (
        state.stage in {DialogStage.AWAITING_ADDRESS, DialogStage.AWAITING_ADDRESS_CONFIRMATION}
        or intent.intent == IntentType.SET_ADDRESS
        or classify_address_reply(user_message) == "new_address"
    ):
        return "[redacted]"
    return user_message


def _reread_confirmation(
    order: Order,
    state: DialogState,
    stage_before: DialogStage,
    reply: str,
    complete: bool,
) -> tuple[str, bool]:
    if stage_before not in {DialogStage.AWAITING_CONFIRMATION, DialogStage.AWAITING_ORDER_CHANGE}:
        return reply, complete
    state.order_confirmation_attempts = 0
    state.order_change_failures = 0
    if state.pending_item_id or state.pending_choices:
        return reply, complete
    state.enter_confirmation()
    return _confirmation_summary(order, state), False


def _apply_unhandled_turn(decision, order: Order, state: DialogState) -> tuple[str, bool]:
    """Apply one validated model decision. The unavailable prefix is not stored."""
    if decision.validation != "ok":
        return ASK_CLARIFY_URDU, False
    if decision.action == "reply":
        state.last_path = "llm_turn"
        text = decision.reply_text
        if state.stage == DialogStage.AWAITING_MORE_ITEMS:
            text = f"{text} {ASK_MORE_ITEMS_URDU}"
        elif state.stage == DialogStage.AWAITING_CONFIRMATION:
            text = f"{text} {ASK_CONFIRM_URDU}"
        elif state.stage == DialogStage.AWAITING_ORDER_CHANGE:
            text = f"{text} {ASK_WHAT_TO_CHANGE_URDU}"
        return text, False
    if decision.action in {"clarify", "other"}:
        if state.stage in {DialogStage.AWAITING_CONFIRMATION, DialogStage.AWAITING_ORDER_CHANGE}:
            return _count_unclear_confirmation(order, state)
        if decision.action == "clarify" and decision.clarify_text:
            return decision.clarify_text, False
        return ASK_CLARIFY_URDU, False
    if decision.action == "menu":
        reply = build_spoken_menu_urdu()
        if state.pending_item_id and (pending := _menu_item(state.pending_item_id)):
            reply = f"{reply} {build_missing_size_urdu(pending)}"
        elif state.pending_choices:
            reply = f"{reply} {_choice_question(state)}"
        return reply, False
    if decision.action == "unavailable" and not decision.items:
        return UNAVAILABLE_ITEM_URDU, False
    prefix = f"{UNAVAILABLE_ITEM_URDU} " if decision.action == "unavailable" else ""
    choice = decision.action == "choice" or (
        decision.action == "unavailable" and len(decision.items) >= 2
    )
    stage_before = state.stage
    if choice:
        reply, complete = _open_item_choice(ParsedIntent(
            intent=IntentType.AMBIGUOUS,
            confidence=1,
            clarification_reason="item_choice",
            choice_groups=[PendingChoice(
                item_ids=[item["item_id"] for item in decision.items],
                quantity=decision.items[0]["quantity"],
            )],
        ), state)
        return _reread_confirmation(order, state, stage_before, f"{prefix}{reply}", complete)
    reply, complete = _apply_additions([
        ItemRequest(item_id=item["item_id"], size_id=item["size_id"], quantity=item["quantity"])
        for item in decision.items
    ], False, order, state)
    return _reread_confirmation(order, state, stage_before, f"{prefix}{reply}", complete)


def _confirmation_summary(order: Order, state: DialogState) -> str:
    state.summary_presented = True
    return build_verified_summary_urdu(order, ask_for_confirmation=True)


def _count_unclear_confirmation(order: Order, state: DialogState) -> tuple[str, bool]:
    state.order_confirmation_attempts += 1
    if state.order_confirmation_attempts >= config.MAX_ORDER_CONFIRMATION_ATTEMPTS:
        state.order_confirmation_attempts = 0
        state.reply_streak = 0
        state.stage = DialogStage.HANDOFF
        return HUMAN_HANDOFF_CONFIRM_URDU, False
    return _confirmation_summary(order, state), False


def _accept_order_confirmation(order: Order, state: DialogState) -> tuple[str, bool]:
    state.order_confirmation_attempts = 0
    if not order.items:
        return EMPTY_ORDER_URDU, False
    if state.pending_address:
        state.stage = DialogStage.AWAITING_ADDRESS_CONFIRMATION
        return build_confirm_address_urdu(state.pending_address), False
    if not order.delivery_address:
        state.stage = DialogStage.AWAITING_ADDRESS
        state.summary_presented = False
        return ASK_ADDRESS_URDU, False
    if not state.summary_presented:
        state.enter_confirmation()
        return _confirmation_summary(order, state), False
    confirm_order(order)
    state.stage = DialogStage.COMPLETE
    state.clear_pending_choice()
    return build_confirmation_urdu(order), True


def _editor_enabled() -> bool:
    return bool(config.BASKET_EDIT_LLM_ENABLED)


def _mutation_applied(state: DialogState) -> bool:
    detail = state.last_basket_edit_detail
    if detail.startswith(("rejected=", "status=disabled", "status=LLM")):
        return False
    return detail.startswith("status=apply")


def _after_successful_mutation(order: Order, state: DialogState, reply: str, complete: bool) -> tuple[str, bool]:
    state.order_confirmation_attempts = 0
    state.order_change_failures = 0
    if state.stage == DialogStage.AWAITING_SIZE:
        return reply, complete
    if not order.items:
        state.stage = DialogStage.ORDERING
        state.summary_presented = False
        return ASK_ORDER_URDU, False
    if order.delivery_address:
        state.enter_confirmation()
        return _confirmation_summary(order, state), False
    return reply, complete


def _apply_local_edit(
    user_message: str, order: Order, state: DialogState,
) -> tuple[str, bool] | None:
    """Apply one fully named add or one uncounted removal. None leaves the turn to the editor."""
    if state.pending_item_id or state.pending_choices:
        return None
    intent = parse_customer_intent(user_message, state, order)
    if intent.intent == IntentType.UNKNOWN_ITEM:
        return UNAVAILABLE_ITEM_URDU, False
    if intent.clarification_reason == "item_choice":
        return None
    if _local_edit_blocked(intent, user_message):
        return ASK_CLARIFY_URDU, False
    if not _resolved_local_edit(intent, user_message):
        return None
    before_order = order.model_dump()
    stage_before = state.stage
    reply, complete = _dispatch_deterministic_intent(
        intent, user_message, order, state, allow_fallback=True,
    )
    if order.model_dump() != before_order and bool(before_order["items"]):
        kind = "local_remove" if intent.intent == IntentType.REMOVE_ITEM else "local_add"
        state.last_path = kind
        state.last_basket_edit_detail = kind
        if stage_before in {DialogStage.AWAITING_CONFIRMATION, DialogStage.AWAITING_ORDER_CHANGE}:
            reply, complete = _local_confirmation_reply(order, state)
    return reply, complete


def _handle_order_confirmation_turn(
    user_message: str, order: Order, state: DialogState,
) -> tuple[str, bool] | None:
    if state.stage != DialogStage.AWAITING_CONFIRMATION or state.pending_address:
        return None
    if parse_customer_intent(user_message, state, order).intent == IntentType.UNKNOWN_ITEM:
        return UNAVAILABLE_ITEM_URDU, False
    decision = classify_order_confirmation(user_message)
    if decision == "no":
        state.order_confirmation_attempts = 0
        state.enter_order_change()
        return ASK_WHAT_TO_CHANGE_URDU, False
    if decision == "yes":
        return _accept_order_confirmation(order, state)
    if decision == "mutation":
        local = _apply_local_edit(user_message, order, state)
        if local is not None:
            return local
        if parse_customer_intent(user_message, state, order).clarification_reason == "item_choice":
            return None
        if not _editor_enabled():
            return _count_unclear_confirmation(order, state)
        before = order.model_dump()
        reply, complete = _resolve_basket_change(
            ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="basket_edit"),
            user_message, order, state,
        )
        if state.last_basket_edit_detail.startswith(("status=disabled", "status=LLM")) or (
            not _mutation_applied(state) and state.stage != DialogStage.AWAITING_SIZE
        ):
            return _count_unclear_confirmation(order, state)
        return _after_successful_mutation(order, state, reply, complete)
    return _count_unclear_confirmation(order, state)


def _handle_order_change_turn(
    user_message: str, order: Order, state: DialogState,
) -> tuple[str, bool] | None:
    if state.stage != DialogStage.AWAITING_ORDER_CHANGE:
        return None
    if parse_customer_intent(user_message, state, order).intent == IntentType.UNKNOWN_ITEM:
        return UNAVAILABLE_ITEM_URDU, False
    local = _apply_local_edit(user_message, order, state)
    if local is not None:
        return local
    if parse_customer_intent(user_message, state, order).clarification_reason == "item_choice":
        return None
    if not _editor_enabled():
        applied = False
        reply, complete = ASK_WHAT_TO_CHANGE_URDU, False
    else:
        before = order.model_dump()
        reply, complete = _resolve_basket_change(
            ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="basket_edit"),
            user_message, order, state,
        )
        applied = state.stage == DialogStage.AWAITING_SIZE or _mutation_applied(state)
    if applied:
        return _after_successful_mutation(order, state, reply, complete)
    state.order_change_failures += 1
    if state.order_change_failures >= config.MAX_ORDER_CONFIRMATION_ATTEMPTS:
        state.order_change_failures = 0
        state.enter_confirmation()
        return _confirmation_summary(order, state), False
    return ASK_WHAT_TO_CHANGE_URDU, False


def _choice_question(state: DialogState) -> str:
    items = [item for item_id in state.pending_choices if (item := _menu_item(item_id))]
    return build_item_choice_urdu(items)


def _open_item_choice(intent: ParsedIntent, state: DialogState) -> tuple[str, bool]:
    state.staged_start_index = len(state.pending_additions)
    state.pending_additions.extend(intent.add_items)
    first, *rest = intent.choice_groups
    state.pending_choices = list(first.item_ids)
    state.pending_choice_quantity = first.quantity
    state.pending_choice_queue = list(rest)
    state.choice_attempts = 0
    return _choice_question(state), False


def _reask_item_choice(order: Order, state: DialogState) -> tuple[str, bool]:
    """Wrong choice replies increment choice_attempts in this one place."""
    state.choice_attempts += 1
    if state.choice_attempts >= 3:
        del state.pending_additions[state.staged_start_index:]
        state.clear_pending_choice()
        if order.items:
            return ASK_MORE_ITEMS_URDU, False
        return ASK_ORDER_URDU, False
    return _choice_question(state), False


def _accept_item_choice(intent: ParsedIntent, order: Order, state: DialogState) -> tuple[str, bool]:
    state.pending_additions.append(ItemRequest(
        item_id=intent.item_id or "",
        size_id=intent.size_id,
        quantity=intent.quantity or 1,
    ))
    state.choice_attempts = 0
    if state.pending_choice_queue:
        nxt = state.pending_choice_queue.pop(0)
        state.pending_choices = list(nxt.item_ids)
        state.pending_choice_quantity = nxt.quantity
        return _choice_question(state), False
    requests = [part.model_copy(deep=True) for part in state.pending_additions]
    state.clear_pending_choice()
    return _apply_additions(requests, False, order, state)


def _resolved_local_edit(intent: ParsedIntent, text: str) -> bool:
    if intent.clarification_reason in {"missing_size", "choice_selected", "choice_retry", "item_choice"}:
        return False
    if intent.intent == IntentType.ADD_ITEM and intent.item_id:
        return structured_addition(text, [intent.item_id])
    if intent.intent == IntentType.ADD_ITEMS and intent.add_items and all(part.item_id for part in intent.add_items):
        return structured_addition(text, [part.item_id for part in intent.add_items])
    if intent.intent == IntentType.REMOVE_ITEM and intent.item_id:
        normalized, _ = normalize_customer_text(text)
        numbers, _ = quantity_evidence(text)
        if numbers or contains_add_word(normalized):
            return False
        if len(sizes_in_text(normalized)) > 1:
            return False
        return True
    return False


def _local_edit_blocked(intent: ParsedIntent, text: str) -> bool:
    normalized, _ = normalize_customer_text(text)
    if intent.intent in {IntentType.ADD_ITEM, IntentType.ADD_ITEMS} and addition_is_prohibited(normalized):
        return True
    if intent.intent == IntentType.REMOVE_ITEM and removal_is_prohibited(normalized):
        return True
    return False


def _local_confirmation_reply(order: Order, state: DialogState) -> tuple[str, bool]:
    state.order_confirmation_attempts = 0
    state.order_change_failures = 0
    if not order.items:
        state.stage = DialogStage.ORDERING
        state.summary_presented = False
        state.clear_pending_choice()
        return ASK_ORDER_URDU, False
    state.enter_confirmation()
    state.summary_presented = True
    return _confirmation_summary(order, state), False


def _chat_turn_hybrid(
    user_message: str,
    order: Order,
    messages: list[dict],
    state: DialogState,
) -> tuple[str, Order, list[dict], bool]:
    state.last_path = "deterministic"
    state.last_llm_seconds = 0.0
    state.last_address_extraction_detail = ""
    state.last_basket_edit_detail = ""
    messages.append({"role": "user", "content": user_message})

    if state.stage == DialogStage.HANDOFF:
        state.reply_streak = 0
        state.last_parser_seconds = 0.0
        messages.append({"role": "assistant", "content": HUMAN_HANDOFF_URDU})
        return HUMAN_HANDOFF_URDU, order, messages, False

    _AMBIGUOUS_LLM_ALLOWED.set(True)
    parser_start = time.perf_counter()
    intent = parse_customer_intent(user_message, state, order)
    state.last_parser_seconds = time.perf_counter() - parser_start
    confirmation = classify_order_confirmation(user_message)
    skip_model = (
        not config.LLM_TURN_HANDLER_ENABLED
        or confirmation in {"yes", "no"}
        or intent.clarification_reason == "quantity_correction"
        or intent.dropped_unknown
    )
    model_decision = None
    if not skip_model:
        model_decision = resolve_unhandled_turn(
            user_message, order, state, _script_hint(intent, state),
        )
        _AMBIGUOUS_LLM_ALLOWED.set(False)
        if _accept_own_action(model_decision, state, intent):
            reply, complete = _apply_unhandled_turn(model_decision, order, state)
            return _finish_hybrid_turn(
                user_message, order, messages, state, intent, model_decision, reply, complete,
            )

    more_result = _handle_more_items_turn(user_message, order, state)
    if more_result is not None:
        state.last_parser_seconds = 0.0
        reply, complete = more_result
        return _finish_hybrid_turn(
            user_message, order, messages, state, intent, model_decision, reply, complete,
        )

    address_result = _handle_address_confirmation_turn(user_message, order, state)
    if address_result is not None:
        state.last_parser_seconds = 0.0
        reply, complete = address_result
        return _finish_hybrid_turn(
            user_message, order, messages, state, intent, model_decision, reply, complete,
        )

    for handler in (_handle_order_change_turn, _handle_order_confirmation_turn):
        handled = handler(user_message, order, state)
        if handled is not None:
            state.last_parser_seconds = 0.0
            reply, complete = handled
            return _finish_hybrid_turn(
                user_message, order, messages, state, intent, model_decision, reply, complete,
            )

    if (
        config.ADDRESS_LLM_EXTRACTION_ENABLED
        and state.stage == DialogStage.AWAITING_ADDRESS
        and not is_explicit_order_command(user_message)
    ):
        if classify_confirmation_reply(user_message) in {"yes", "no"}:
            reply, complete = ASK_ADDRESS_URDU, False
        else:
            reply, complete = _capture_address(user_message, state)
    else:
        local_quantity_answer = bool(
            state.quantity_correction_item_id
            and intent.intent in {IntentType.SET_QUANTITY, IntentType.REDUCE_QUANTITY}
        )
        if intent.clarification_reason == "item_choice":
            reply, complete = _open_item_choice(intent, state)
        elif intent.clarification_reason == "choice_retry":
            reply, complete = _reask_item_choice(order, state)
        elif intent.clarification_reason == "choice_selected":
            reply, complete = _accept_item_choice(intent, order, state)
        elif intent.dropped_unknown:
            if intent.clarification_reason == "item_choice":
                reply, complete = _open_item_choice(intent, state)
            elif intent.add_items or (intent.intent == IntentType.ADD_ITEM and intent.item_id):
                requests = intent.add_items or [ItemRequest(
                    item_id=intent.item_id or "", size_id=intent.size_id, quantity=intent.quantity or 1,
                )]
                reply, complete = _apply_additions(requests, False, order, state)
            else:
                reply, complete = ASK_CLARIFY_URDU, False
        elif (local := _apply_local_edit(user_message, order, state)) is not None:
            reply, complete = local
        elif (
            config.BASKET_EDIT_LLM_ENABLED and order.items and not state.pending_item_id
            and not state.pending_choices
            and not local_quantity_answer
            and intent.intent not in {
                IntentType.LIST_MENU, IntentType.SHOW_ORDER, IntentType.CONFIRM_ORDER,
                IntentType.SET_ADDRESS, IntentType.SET_INSTRUCTIONS, IntentType.CANCEL,
                IntentType.CHANGE_SIZE,
            }
        ):
            # Semantic editing covers every basket-content turn, not just
            # requests containing a particular remove/reduce keyword.
            if intent.clarification_reason != "basket_edit":
                intent = ParsedIntent(intent=IntentType.AMBIGUOUS, confidence=0, clarification_reason="basket_edit")
            reply, complete = _resolve_basket_change(intent, user_message, order, state)
        else:
            reply, complete = _dispatch_deterministic_intent(
                intent, user_message, order, state, allow_fallback=True,
            )
    return _finish_hybrid_turn(
        user_message, order, messages, state, intent, model_decision, reply, complete,
    )


def _finish_hybrid_turn(
    user_message: str,
    order: Order,
    messages: list[dict],
    state: DialogState,
    intent: ParsedIntent,
    model_decision,
    reply: str,
    complete: bool,
) -> tuple[str, Order, list[dict], bool]:
    spoke_reply = (
        model_decision is not None
        and model_decision.action == "reply"
        and model_decision.validation == "ok"
        and state.last_path == "llm_turn"
    )
    if spoke_reply:
        state.reply_streak += 1
    else:
        state.reply_streak = 0
    if model_decision is not None:
        record_model_turn({
            "utterance": _redacted_utterance(user_message, state, intent),
            "stage": state.stage.value,
            "model_output": model_decision.raw,
            "reply_text": model_decision.reply_text,
            "validation": model_decision.validation,
            "final_path": state.last_path,
        })
    if state.last_path == "openai_address_extraction":
        state.last_parser_seconds = 0.0
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
