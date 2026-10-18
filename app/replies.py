"""Deterministic Urdu replies built only from verified application state."""

from __future__ import annotations

from app.dialog_state import DialogStage, DialogState
from app.intent import IntentType, ParsedIntent
from app.numbers_urdu import number_to_urdu_words
from app.order import Order, build_confirmation_urdu, calculate_total


ASK_ADDRESS_URDU = "براہ کرم ڈیلیوری کا پتہ بتائیں۔"
ASK_CLARIFY_URDU = "معذرت، براہ کرم اپنا آرڈر دوبارہ واضح الفاظ میں بتائیں۔"
MENU_OFFER_URDU = "اگر آپ مینو چاہتے ہیں تو بھی بتا دیں۔"
ASK_QUANTITY_URDU = "براہ کرم بتائیں، اس آئٹم کی کل تعداد کتنی رکھنی ہے؟"
EMPTY_ORDER_URDU = "آپ کے آرڈر میں ابھی کوئی چیز شامل نہیں ہے۔"
UNAVAILABLE_ITEM_URDU = "یہ آئٹم ہمارے مینو میں اس وقت دستیاب نہیں ہے۔ اگر آپ کچھ اور آرڈر کرنا چاہتے ہیں تو بتا دیں، اور اگر آپ مینو چاہتے ہیں تو بھی بتا دیں۔"
ASK_CONFIRM_URDU = "کیا آپ اس آرڈر کی تصدیق کرنا چاہتے ہیں؟"
ASK_MORE_ITEMS_URDU = "کیا آپ کچھ اور شامل کرنا چاہتے ہیں؟"
ASK_ORDER_URDU = "آپ کیا آرڈر کرنا چاہیں گے؟"
ASK_WHAT_TO_CHANGE_URDU = "ٹھیک ہے، آپ آرڈر میں کیا تبدیل کرنا چاہتے ہیں؟"
ASK_CANCEL_CONFIRM_URDU = "کیا آپ یہ آرڈر ختم کرنا چاہتے ہیں؟"
FACT_UNKNOWN_URDU = "یہ معلومات موجود نہیں"
HUMAN_HANDOFF_URDU = "معذرت، میں پتہ واضح نہیں سمجھ سکی۔ براہ کرم عملے سے رابطہ کریں۔"
HUMAN_HANDOFF_CONFIRM_URDU = "معذرت، آرڈر کی تصدیق نہیں ہو سکی۔ آپ کو ہمارے نمائندے سے ملایا جا رہا ہے۔"


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


def build_more_items_reply_urdu(order: Order, state: DialogState) -> str:
    state.stage = DialogStage.AWAITING_MORE_ITEMS
    state.summary_presented = False
    items = "، ".join(_line_spoken(item) for item in order.items)
    total = number_to_urdu_words(calculate_total(order))
    return f"آپ کا آرڈر یہ ہے۔ {items}۔ کل {total} روپے۔ {ASK_MORE_ITEMS_URDU}"


def build_update_reply_urdu(order: Order, state: DialogState, action: str) -> str:
    """Build a short update reply after a verified mutation."""
    state.sync_stage(order)
    if not order.items:
        state.summary_presented = False
        return EMPTY_ORDER_URDU
    if action == "add" or not order.delivery_address:
        return build_more_items_reply_urdu(order, state)
    state.enter_confirmation()
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


def _item_spoken_name(item: dict) -> str:
    return item.get("spoken_name_urdu") or item.get("name_urdu") or item.get("name") or ""


def _or_clause(labels: list[str]) -> str:
    """Two labels put چاہیے in the middle. Three or more join with ، and یا."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} چاہیے یا {labels[1]}"
    return "، ".join(labels[:-1]) + f" یا {labels[-1]}"


def _size_label_clause(labels: list[str]) -> str:
    return _or_clause(labels)


def build_missing_size_urdu(item: dict) -> str:
    name = _item_spoken_name(item)
    labels = [
        size.get("label_urdu", size.get("label", size["id"]))
        for size in item.get("sizes", [])
    ]
    clause = _or_clause(labels)
    if len(labels) == 2:
        return f"{name} {clause}؟"
    return f"{name} {clause} چاہیے؟"


def build_item_choice_urdu(items: list[dict]) -> str:
    names = [_item_spoken_name(item) for item in items]
    clause = _or_clause(names)
    if len(names) == 2:
        return f"{clause}؟"
    return f"{clause} چاہیے؟"


def build_size_help_urdu(item: dict) -> str:
    return build_missing_size_urdu(item)


def build_missing_item_urdu() -> str:
    return "براہ کرم بتائیں کس آئٹم کو تبدیل کرنا ہے۔"


def build_confirm_address_urdu(address: str) -> str:
    return f"کیا آپ کا پتہ یہ ہے: {address}؟"


def build_address_saved_urdu(order: Order, state: DialogState) -> str:
    state.sync_stage(order)
    if not order.items:
        state.summary_presented = False
        return f"پتہ محفوظ ہو گیا ہے۔ {ASK_ORDER_URDU}"
    state.enter_confirmation()
    state.summary_presented = True
    return (
        "پتہ محفوظ ہو گیا ہے۔ "
        + build_verified_summary_urdu(order, ask_for_confirmation=True)
    )
