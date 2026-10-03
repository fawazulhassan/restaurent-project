"""Deterministic Urdu replies built only from verified application state."""

from __future__ import annotations

from app.dialog_state import DialogStage, DialogState
from app.intent import IntentType, ParsedIntent
from app.numbers_urdu import number_to_urdu_words
from app.order import Order, build_confirmation_urdu, calculate_total


ASK_ADDRESS_URDU = "براہ کرم ڈیلیوری کا پتہ بتائیں۔"
ASK_CLARIFY_URDU = "معذرت، براہ کرم اپنا آرڈر دوبارہ واضح الفاظ میں بتائیں۔"
EMPTY_ORDER_URDU = "آپ کے آرڈر میں ابھی کوئی چیز شامل نہیں ہے۔"
UNAVAILABLE_ITEM_URDU = "معذرت، یہ چیز ہمارے مینو میں دستیاب نہیں ہے۔"
ASK_CONFIRM_URDU = "کیا آپ اس آرڈر کی تصدیق کرنا چاہتے ہیں؟"


def _line_spoken(item) -> str:
    quantity_words = number_to_urdu_words(item.qty)
    if item.size_label:
        return f"{quantity_words} {item.name_urdu}، {item.size_label}"
    return f"{quantity_words} {item.name_urdu}"


def build_verified_summary_urdu(
    order: Order,
    *,
    ask_for_confirmation: bool,
) -> str:
    """Build an Urdu summary from validated order fields and calculated total."""
    if not order.items:
        return EMPTY_ORDER_URDU

    item_text = "، ".join(_line_spoken(item) for item in order.items)
    total_words = number_to_urdu_words(calculate_total(order))
    parts = [f"آپ کا آرڈر یہ ہے۔ {item_text}۔", f"کل {total_words} روپے۔"]
    if order.delivery_address:
        parts.append(f"ڈیلیوری کا پتہ {order.delivery_address}۔")
    else:
        parts.append(ASK_ADDRESS_URDU)

    if order.special_instructions:
        parts.append(f"ہدایات {order.special_instructions}۔")

    if ask_for_confirmation and order.items and order.delivery_address:
        parts.append(ASK_CONFIRM_URDU)
    return " ".join(parts)


def build_update_reply_urdu(order: Order, state: DialogState, action: str) -> str:
    """Build a short update reply after a verified mutation."""
    state.sync_stage(order)
    if not order.items:
        state.summary_presented = False
        return EMPTY_ORDER_URDU
    if not order.delivery_address:
        action_text = {
            "add": "آپ کا آئٹم آرڈر میں شامل ہو گیا ہے۔",
            "remove": "آئٹم آرڈر سے نکال دیا گیا ہے۔",
            "quantity": "آئٹم کی تعداد تبدیل کر دی گئی ہے۔",
            "size": "آئٹم کا سائز تبدیل کر دیا گیا ہے۔",
            "instructions": "آپ کی ہدایات محفوظ کر لی گئی ہیں۔",
        }.get(action, "آپ کا آرڈر اپ ڈیٹ ہو گیا ہے۔")
        return f"{action_text} {ASK_ADDRESS_URDU}"

    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    return build_verified_summary_urdu(order, ask_for_confirmation=True)


def build_intent_reply_urdu(
    intent: ParsedIntent,
    order: Order,
    state: DialogState,
) -> str:
    """Build a safe generic reply for a parsed intent."""
    if intent.intent == IntentType.LIST_MENU:
        raise ValueError("Menu replies are built from menu data in app.dialog.")
    if intent.intent == IntentType.SHOW_ORDER:
        return build_verified_summary_urdu(
            order,
            ask_for_confirmation=bool(order.items and order.delivery_address),
        )
    if intent.intent == IntentType.CONFIRM_ORDER and order.status.value == "confirmed":
        return build_confirmation_urdu(order)
    if intent.intent in (IntentType.AMBIGUOUS, IntentType.UNKNOWN_ITEM):
        return ASK_CLARIFY_URDU
    return build_verified_summary_urdu(order, ask_for_confirmation=False)


def build_missing_size_urdu(size_labels: list[str]) -> str:
    choices = " یا ".join(size_labels)
    return f"براہ کرم سائز بتائیں۔ {choices}؟"


def build_size_help_urdu(size_labels: list[str]) -> str:
    choices = " یا ".join(size_labels)
    return f"سائز کے لیے صرف ایک لفظ بولیں: {choices}۔"


def build_missing_item_urdu() -> str:
    return "براہ کرم بتائیں کس آئٹم کو تبدیل کرنا ہے۔"


def build_address_saved_urdu(order: Order, state: DialogState) -> str:
    state.sync_stage(order)
    if not order.items:
        state.summary_presented = False
        return "ڈیلیوری کا پتہ محفوظ ہو گیا ہے۔ اب اپنا آرڈر بتائیں۔"
    state.stage = DialogStage.AWAITING_CONFIRMATION
    state.summary_presented = True
    return build_verified_summary_urdu(order, ask_for_confirmation=True)
