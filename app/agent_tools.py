"""Tool schemas and executors. Every fact and mutation stays in code."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import config
from app.agent_prompt import FACT_KEYS
from app.basket_edit import BasketEdit, apply_basket_edits
from app.confirmation_urdu import ORDER_FILLER, _ORDER_QUANTITY_WORDS
from app.dialog_state import DialogStage, DialogState, ItemRequest
from app.order import Order, OrderError, calculate_total
from app.text_fold import compare_tokens, fold_text

_FACTS_PATH = Path(__file__).resolve().parent.parent / "data" / "facts.json"
_MAX_ADDITIONS = 5
_MAX_ADDED_QUANTITY = 20

READ_ONLY_TOOLS = ("get_menu", "get_item_info", "get_basket", "get_fact")
ALWAYS_SAFE_TOOLS = ("transfer_to_human", "request_cancel", "repeat_last", "record_complaint")
MUTATION_TOOLS = ("add_item", "apply_basket_edits")
FLOW_TOOLS = ("finish_items",)
SLOT_TOOLS = (
    "set_size", "choose_option", "set_address", "confirm_address",
    "confirm_order", "confirm_cancel", "decline_cancel",
)
KNOWN_TOOLS = READ_ONLY_TOOLS + ALWAYS_SAFE_TOOLS + MUTATION_TOOLS + FLOW_TOOLS + SLOT_TOOLS
_UTTERANCE: ContextVar[str] = ContextVar("agent_utterance", default="")
_PERSON_WORDS = frozenset(compare_tokens(
    "انسان عملہ شخص نمائندہ بندہ آدمی human person manager agent operator someone"
))
_COMPLAINT_CAP = 300


class AgentAbort(Exception):
    """The whole turn is discarded. The live order is not changed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class AdditionBudget:
    lines: set[tuple[str, str | None]] = field(default_factory=set)
    quantity: int = 0


def _fail(code: str) -> dict[str, Any]:
    return {"ok": False, "error": code}


def _closed_tokens(*groups: tuple[str, ...]) -> set[str]:
    words: set[str] = set()
    for group in groups:
        for phrase in group:
            words.update(compare_tokens(phrase))
    return words


def unmatched_words(text: str) -> list[str]:
    """Tokens that are not a quantity, framing word, size, alias, or group token."""
    from app.intent import _SIZE_LEMMAS, _all_menu_items, _item_aliases, item_group_index, load_intent_menu

    skip = _closed_tokens(ORDER_FILLER, _ORDER_QUANTITY_WORDS)
    for lemmas in _SIZE_LEMMAS.values():
        skip.update(_closed_tokens(tuple(lemmas)))
    for item in _all_menu_items(load_intent_menu()):
        skip.update(_closed_tokens(tuple(_item_aliases(item))))
    skip.update(item_group_index())
    skip.update(compare_tokens("اور بھی مجھے چاہتا چاہتے ہوں ہو تو کے پاس نے کہا ہی نہیں"))
    found: list[str] = []
    for token in compare_tokens(text):
        if token in skip or token in found:
            continue
        found.append(token)
        if len(found) == 5:
            break
    return found


def _negates_item(text: str, item_id: str) -> bool:
    from app.intent import _item_aliases, addition_is_prohibited

    if addition_is_prohibited(fold_text(text)):
        return True
    item = _find_item(item_id)
    if item is None:
        return False
    alias_tokens: set[str] = set()
    for alias in _item_aliases(item):
        alias_tokens.update(compare_tokens(alias))
    tokens = compare_tokens(text)
    negation = _closed_tokens(("نہیں", "نہی", "no", "not", "مت", "mat"))
    fillers = _closed_tokens(ORDER_FILLER)
    window = config.NEGATION_ITEM_WINDOW
    alias_at = [index for index, token in enumerate(tokens) if token in alias_tokens]
    negated_at = [index for index, token in enumerate(tokens) if token in negation]
    for index in alias_at:
        for neg_at in negated_at:
            if abs(neg_at - index) > window:
                continue
            between = tokens[min(neg_at, index) + 1:max(neg_at, index)]
            if any(part not in fillers and part not in alias_tokens for part in between):
                continue
            return True
    return False


def ground_item(text: str, item_id: str, state: DialogState) -> str:
    """ok, choice, or not_in_utterance. A negation never grounds the dish."""
    from app.intent import (
        _fuzzy_item_match,
        item_group_index,
        load_intent_menu,
        mentioned_item_ids,
        normalize_customer_text,
        sizes_in_text,
    )

    if _negates_item(text, item_id):
        return "not_in_utterance"
    if item_id in mentioned_item_ids(text):
        return "ok"
    tokens = set(compare_tokens(text))
    for token, owners in item_group_index().items():
        if token in tokens and item_id in owners:
            return "choice"
    if state.pending_item_id == item_id and compare_tokens(text) and all(
        token in _closed_tokens(ORDER_FILLER) or bool(sizes_in_text(token))
        for token in compare_tokens(text)
    ):
        return "ok"
    normalized, _noise = normalize_customer_text(text)
    fuzzy = _fuzzy_item_match(normalized, load_intent_menu())
    if fuzzy is not None and fuzzy.item["id"] == item_id and fuzzy.score >= config.HYBRID_FUZZY_MIN_SCORE:
        return "ok"
    return "not_in_utterance"


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        },
    }


def bind_utterance(text: str):
    """Tools read this utterance. The caller resets the token."""
    return _UTTERANCE.set(text)


def current_utterance() -> str:
    return _UTTERANCE.get()


def tool_schemas() -> list[dict]:
    return [
        _schema("get_menu", "List available menu items and their sizes. No prices.", {}, []),
        _schema(
            "get_item_info",
            "Read sizes, prices, notes, and availability for one item id.",
            {"item_id": {"type": "string"}},
            ["item_id"],
        ),
        _schema("get_basket", "Read the current basket lines and total.", {}, []),
        _schema(
            "get_fact",
            "Read one restaurant fact. Unknown or blank keys are unknown_fact.",
            {"key": {"type": "string"}},
            ["key"],
        ),
        _schema(
            "add_item",
            "Add one menu item. Omit size_id when the customer has not chosen a size.",
            {
                "item_id": {"type": "string"},
                "size_id": {"type": ["string", "null"]},
                "quantity": {"type": "integer"},
            },
            ["item_id", "quantity"],
        ),
        _schema(
            "apply_basket_edits",
            "Apply replace_size, set, reduce, or remove through the basket validator.",
            {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string"},
                            "item_id": {"type": "string"},
                            "size_id": {"type": ["string", "null"]},
                            "new_size_id": {"type": ["string", "null"]},
                            "quantity": {"type": ["integer", "null"]},
                        },
                        "required": ["action", "item_id"],
                    },
                }
            },
            ["edits"],
        ),
        _schema("request_cancel", "Ask the customer to confirm cancelling the basket.", {}, []),
        _schema(
            "transfer_to_human",
            "Hand the call to a person. Does not save the order.",
            {"reason": {"type": "string"}},
            ["reason"],
        ),
        _schema("repeat_last", "Repeat the previous spoken reply.", {}, []),
        _schema(
            "finish_items",
            "Customer is done adding items. Does not save or confirm the order.",
            {},
            [],
        ),
        _schema(
            "set_size",
            "Set the pending item's size from a size word in this utterance.",
            {"size_id": {"type": "string"}},
            ["size_id"],
        ),
        _schema(
            "choose_option",
            "Choose one pending item the customer named in this utterance.",
            {"item_id": {"type": "string"}},
            ["item_id"],
        ),
        _schema("set_address", "Read the delivery address from this utterance.", {}, []),
        _schema("confirm_address", "Save the address when this utterance is yes.", {}, []),
        _schema("confirm_order", "Confirm the order when this utterance is yes.", {}, []),
        _schema("confirm_cancel", "Cancel the basket when this utterance is yes.", {}, []),
        _schema("decline_cancel", "Keep the basket when this utterance is no.", {}, []),
        _schema(
            "record_complaint",
            "Store a complaint for staff. Does not change the basket.",
            {"text": {"type": "string"}},
            ["text"],
        ),
    ]


def tools_for_stage(state: DialogState) -> list[dict]:
    """Each stage offers the tools that can commit that stage."""
    read = set(READ_ONLY_TOOLS)
    common = read | {"record_complaint", "transfer_to_human", "repeat_last"}
    if state.pending_choices:
        names = common | {"choose_option", "add_item"}
    elif state.stage == DialogStage.AWAITING_SIZE:
        names = common | {"set_size", "add_item", "apply_basket_edits", "request_cancel"}
    elif state.stage in {DialogStage.ORDERING, DialogStage.AWAITING_MORE_ITEMS}:
        names = common | {"add_item", "apply_basket_edits", "finish_items", "request_cancel"}
    elif state.stage == DialogStage.AWAITING_ADDRESS:
        names = common | {"set_address"}
    elif state.stage == DialogStage.AWAITING_ADDRESS_CONFIRMATION:
        names = common | {"confirm_address"}
    elif state.stage == DialogStage.AWAITING_CONFIRMATION:
        names = common | {"confirm_order", "add_item", "apply_basket_edits", "request_cancel"}
    elif state.stage == DialogStage.AWAITING_ORDER_CHANGE:
        names = common | {"add_item", "apply_basket_edits"}
    elif state.stage == DialogStage.AWAITING_CANCEL_CONFIRM:
        names = read | {"confirm_cancel", "decline_cancel", "repeat_last"}
    else:
        names = common
    return [tool for tool in tool_schemas() if tool["function"]["name"] in names]


def load_facts() -> dict[str, str]:
    with _FACTS_PATH.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {key: "" if raw.get(key) is None else str(raw.get(key)) for key in FACT_KEYS}


def _menu_items() -> list[dict]:
    from app.dialog import load_menu_data

    items: list[dict] = []
    for category in load_menu_data().get("categories", []):
        items.extend(category.get("items", []))
    return items


def _find_item(item_id: str) -> dict | None:
    return next((item for item in _menu_items() if item.get("id") == item_id), None)


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    order: Order,
    state: DialogState,
    budget: AdditionBudget,
) -> dict[str, Any]:
    if name not in KNOWN_TOOLS:
        return _fail("unknown_tool")
    if not isinstance(arguments, dict):
        raise AgentAbort("invalid_arguments")
    if name == "get_menu":
        return _get_menu()
    if name == "get_item_info":
        return _get_item_info(arguments)
    if name == "get_basket":
        return _get_basket(order)
    if name == "get_fact":
        return _get_fact(arguments)
    if name == "add_item":
        return _add_item(arguments, order, state, budget)
    if name == "apply_basket_edits":
        return _apply_edits(arguments, order, state)
    if name == "request_cancel":
        return _request_cancel(order, state)
    if name == "transfer_to_human":
        return _transfer(arguments, state)
    if name == "repeat_last":
        return _repeat(state)
    if name == "finish_items":
        return _finish_items(arguments, order, state, budget)
    if name == "set_size":
        return _set_size(arguments, order, state)
    if name == "choose_option":
        return _choose_option(arguments, order, state)
    if name == "set_address":
        return _set_address(arguments, state)
    if name == "confirm_address":
        return _confirm_address(order, state)
    if name == "confirm_order":
        return _confirm_order(order, state)
    if name == "confirm_cancel":
        return _confirm_cancel(order, state, yes=True)
    if name == "decline_cancel":
        return _confirm_cancel(order, state, yes=False)
    if name == "record_complaint":
        return _record_complaint(arguments, order)
    return _fail("unknown_tool")


def _get_menu() -> dict[str, Any]:
    data = []
    for item in _menu_items():
        if not item.get("available", True):
            continue
        data.append({
            "item_id": item["id"],
            "spoken_name_urdu": item.get("spoken_name_urdu") or item.get("name_urdu") or "",
            "sizes": [
                {"id": size["id"], "label": size.get("label_urdu", size.get("label", size["id"]))}
                for size in item.get("sizes", [])
            ],
        })
    return _ok(data)


def _get_item_info(arguments: dict) -> dict[str, Any]:
    item_id = arguments.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise AgentAbort("invalid_arguments")
    item = _find_item(item_id)
    if item is None:
        return _fail("unknown_item")
    sizes = [
        {
            "id": size["id"],
            "label": size.get("label_urdu", size.get("label", size["id"])),
            "price_pkr": size["price_pkr"],
        }
        for size in item.get("sizes", [])
    ]
    return _ok({
        "item_id": item_id,
        "available": bool(item.get("available", True)),
        "notes": item.get("notes") or "",
        "sizes": sizes,
        "price_pkr": None if sizes else item.get("price_pkr"),
    })


def _get_basket(order: Order) -> dict[str, Any]:
    lines = []
    for line in order.items:
        lines.append({
            "item_id": line.id,
            "spoken_name": line.name_urdu,
            "size_id": line.size,
            "quantity": line.qty,
            "line_price_pkr": line.unit_price_pkr * line.qty,
        })
    return _ok({"lines": lines, "total_pkr": calculate_total(order)})


def _get_fact(arguments: dict) -> dict[str, Any]:
    key = arguments.get("key")
    if not isinstance(key, str):
        raise AgentAbort("invalid_arguments")
    facts = load_facts()
    if key not in facts or not str(facts[key]).strip():
        return _fail("unknown_fact")
    return _ok({"key": key, "value": facts[key]})


def _add_item(
    arguments: dict,
    order: Order,
    state: DialogState,
    budget: AdditionBudget,
) -> dict[str, Any]:
    item_id = arguments.get("item_id")
    quantity = arguments.get("quantity")
    if not isinstance(item_id, str) or not item_id:
        raise AgentAbort("invalid_arguments")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= 20:
        return _fail("bad_quantity")
    size_id = arguments.get("size_id", None)
    if size_id is not None and not isinstance(size_id, str):
        raise AgentAbort("invalid_arguments")
    item = _find_item(item_id)
    if item is None:
        return _fail("unknown_item")
    if not item.get("available", True):
        return _fail("item_unavailable")
    text = current_utterance()
    grounded = ground_item(text, item_id, state)
    if grounded == "not_in_utterance":
        return _fail("not_in_utterance")
    if grounded == "choice":
        from app.intent import item_group_index

        owners: list[str] = []
        for token, pair in item_group_index().items():
            if token in set(compare_tokens(text)) and item_id in pair:
                owners = list(pair)
                break
        state.pending_choices = owners or [item_id]
        state.pending_choice_quantity = quantity
        state.choice_attempts = 0
        return {"ok": True, "status": "awaiting_choice", "data": {"item_ids": state.pending_choices}}
    if state.pending_item_id and state.pending_item_id != item_id:
        from app.intent import mentioned_item_ids

        pending_id = state.pending_item_id
        if pending_id not in mentioned_item_ids(text):
            state.clear_pending_size()
            state.pending_additions = [
                part for part in state.pending_additions if part.item_id != pending_id
            ]
    sizes = item.get("sizes") or []
    if size_id and not any(size["id"] == size_id for size in sizes):
        return _fail("unknown_size")
    line_key = (item_id, size_id)
    new_line = line_key not in budget.lines and not any(
        existing.id == item_id and existing.size == size_id for existing in order.items
    )
    if new_line and len(budget.lines) >= _MAX_ADDITIONS:
        raise AgentAbort("too_many_items")
    if budget.quantity + quantity > _MAX_ADDED_QUANTITY:
        raise AgentAbort("too_many_items")
    needs_size = bool(sizes) and not size_id
    size_open = state.stage == DialogStage.AWAITING_SIZE or bool(state.pending_item_id)
    if needs_size or size_open:
        from app.dialog import _reask_size, _upsert_pending

        _upsert_pending(state, ItemRequest(item_id=item_id, size_id=size_id, quantity=quantity))
        budget.lines.add(line_key)
        budget.quantity += quantity
        if needs_size:
            reply = _reask_size(item, quantity, None, state)
            return {"ok": True, "status": "awaiting_size", "data": {"reply": reply}}
        return {"ok": True, "status": "staged", "data": {}}
    from app.dialog import _apply_additions

    reply, _complete = _apply_additions(
        [ItemRequest(item_id=item_id, size_id=size_id, quantity=quantity)],
        False,
        order,
        state,
    )
    if state.stage == DialogStage.AWAITING_SIZE and state.pending_item_id == item_id:
        return {"ok": True, "status": "awaiting_size", "data": {"reply": reply}}
    if reply and not any(line.id == item_id for line in order.items) and state.stage != DialogStage.AWAITING_SIZE:
        if item is None:
            return _fail("unknown_item")
        return _fail("item_unavailable")
    budget.lines.add(line_key)
    budget.quantity += quantity
    state.unmatched_words = []
    return {"ok": True, "status": "added", "data": {"reply": reply}}


def _apply_edits(arguments: dict, order: Order, state: DialogState) -> dict[str, Any]:
    raw = arguments.get("edits")
    if not isinstance(raw, list):
        raise AgentAbort("invalid_arguments")
    try:
        edits = [BasketEdit.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise AgentAbort("invalid_arguments") from exc
    try:
        apply_basket_edits(order, edits)
    except OrderError as exc:
        state.last_basket_edit_detail = f"rejected={exc}"
        return _fail("plan_rejected")
    return {"ok": True, "status": "edited", "data": {"count": len(edits)}}


def _request_cancel(order: Order, state: DialogState) -> dict[str, Any]:
    from app.replies import ASK_CANCEL_CONFIRM_URDU, ASK_CLARIFY_URDU

    if not order.items:
        return {"ok": False, "error": "not_allowed_in_stage", "speak": ASK_CLARIFY_URDU}
    state.cancel_resume_stage = state.stage
    state.stage = DialogStage.AWAITING_CANCEL_CONFIRM
    return {"ok": True, "status": "awaiting_cancel", "speak": ASK_CANCEL_CONFIRM_URDU}


def _transfer(arguments: dict, state: DialogState) -> dict[str, Any]:
    from app.replies import HUMAN_HANDOFF_URDU

    reason = arguments.get("reason")
    if not isinstance(reason, str):
        raise AgentAbort("invalid_arguments")
    if not any(token in _PERSON_WORDS for token in compare_tokens(current_utterance())):
        return _fail("not_allowed")
    state.stage = DialogStage.HANDOFF
    state.reply_streak = 0
    state.unmatched_words = []
    return {
        "ok": True,
        "status": "handoff",
        "speak": HUMAN_HANDOFF_URDU,
        "code_line": True,
        "data": {"reason": reason},
    }


def _code(reply: str, status: str) -> dict[str, Any]:
    return {"ok": True, "status": status, "speak": reply, "code_line": True}


def _set_size(arguments: dict, order: Order, state: DialogState) -> dict[str, Any]:
    from app.intent import sizes_in_text

    size_id = arguments.get("size_id")
    if not isinstance(size_id, str) or not size_id:
        raise AgentAbort("invalid_arguments")
    if state.stage != DialogStage.AWAITING_SIZE or not state.pending_item_id:
        return _fail("not_allowed_in_stage")
    item = _find_item(state.pending_item_id)
    if item is None or size_id not in sizes_in_text(current_utterance(), item):
        return _fail("not_in_utterance")
    from app.dialog import _apply_additions

    requests = [part.model_copy(deep=True) for part in state.pending_additions]
    unresolved = next((part for part in requests if part.item_id == state.pending_item_id and not part.size_id), None)
    if unresolved is None:
        requests.insert(0, ItemRequest(item_id=state.pending_item_id, size_id=size_id, quantity=state.pending_quantity))
    else:
        unresolved.size_id = size_id
    reply, _complete = _apply_additions(requests, False, order, state)
    state.unmatched_words = []
    return {"ok": True, "status": "added", "data": {"reply": reply}}


def _choose_option(arguments: dict, order: Order, state: DialogState) -> dict[str, Any]:
    from app.intent import IntentType, ParsedIntent, mentioned_item_ids

    item_id = arguments.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise AgentAbort("invalid_arguments")
    if item_id not in state.pending_choices or item_id not in mentioned_item_ids(current_utterance()):
        return _fail("not_in_utterance")
    from app.dialog import _accept_item_choice

    reply, _complete = _accept_item_choice(
        ParsedIntent(intent=IntentType.ADD_ITEM, confidence=1, item_id=item_id, quantity=state.pending_choice_quantity or 1),
        order,
        state,
    )
    state.unmatched_words = []
    return {"ok": True, "status": "added", "data": {"reply": reply}}


def _set_address(arguments: dict, state: DialogState) -> dict[str, Any]:
    if arguments:
        raise AgentAbort("invalid_arguments")
    from app.dialog import _capture_address

    reply, _complete = _capture_address(current_utterance(), state)
    return _code(reply, "addressed")


def _confirm_address(order: Order, state: DialogState) -> dict[str, Any]:
    from app.confirmation_urdu import classify_address_confirmation
    from app.dialog import _save_pending_address

    if state.stage != DialogStage.AWAITING_ADDRESS_CONFIRMATION:
        return _fail("not_allowed_in_stage")
    if classify_address_confirmation(current_utterance()) != "yes":
        return _fail("not_in_utterance")
    return _code(_save_pending_address(order, state), "addressed")


def _confirm_order(order: Order, state: DialogState) -> dict[str, Any]:
    from app.confirmation_urdu import classify_order_confirmation
    from app.dialog import _accept_order_confirmation

    if classify_order_confirmation(current_utterance()) != "yes" or not state.summary_presented:
        return _fail("not_in_utterance")
    reply, _complete = _accept_order_confirmation(order, state)
    return _code(reply, "confirmed")


def _confirm_cancel(order: Order, state: DialogState, *, yes: bool) -> dict[str, Any]:
    from app.confirmation_urdu import classify_order_confirmation
    from app.dialog import _cancel_confirmation_reply

    if state.stage != DialogStage.AWAITING_CANCEL_CONFIRM:
        return _fail("not_allowed_in_stage")
    decision = classify_order_confirmation(current_utterance())
    if yes and decision != "yes":
        return _fail("not_in_utterance")
    if not yes and decision != "no":
        return _fail("not_in_utterance")
    reply, _complete = _cancel_confirmation_reply(current_utterance(), order, state)
    return {"ok": True, "status": "cancelled", "speak": reply, "code_line": True}


def _record_complaint(arguments: dict, order: Order) -> dict[str, Any]:
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise AgentAbort("invalid_arguments")
    combined = f"{order.complaint} {text.strip()}".strip() if order.complaint else text.strip()
    order.complaint = combined[:_COMPLAINT_CAP]
    return {"ok": True, "status": "noted", "data": {"length": len(order.complaint)}}


def _repeat(state: DialogState) -> dict[str, Any]:
    return _ok({"last_reply": state.last_reply})


def _finish_items(
    arguments: dict,
    order: Order,
    state: DialogState,
    budget: AdditionBudget,
) -> dict[str, Any]:
    """Reuse the more-items no path. An add already made this turn blocks it."""
    if arguments:
        raise AgentAbort("invalid_arguments")
    if state.stage not in {DialogStage.ORDERING, DialogStage.AWAITING_MORE_ITEMS}:
        return _fail("not_allowed_in_stage")
    if state.pending_item_id or state.pending_choices:
        return _fail("not_allowed_in_stage")
    if budget.lines:
        return _fail("not_allowed_in_stage")
    from app.dialog import _close_more_items

    reply, _complete = _close_more_items(order, state)
    return {"ok": True, "status": "finished", "speak": reply}
