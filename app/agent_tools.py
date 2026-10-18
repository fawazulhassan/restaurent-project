"""Tool schemas and executors. Every fact and mutation stays in code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.agent_prompt import FACT_KEYS
from app.basket_edit import BasketEdit, apply_basket_edits
from app.dialog_state import DialogStage, DialogState, ItemRequest
from app.order import Order, OrderError, calculate_total

_FACTS_PATH = Path(__file__).resolve().parent.parent / "data" / "facts.json"
_MAX_ADDITIONS = 5
_MAX_ADDED_QUANTITY = 20

READ_ONLY_TOOLS = ("get_menu", "get_item_info", "get_basket", "get_fact")
ALWAYS_SAFE_TOOLS = ("transfer_to_human", "request_cancel", "repeat_last")
MUTATION_TOOLS = ("add_item", "apply_basket_edits")
FLOW_TOOLS = ("finish_items",)
KNOWN_TOOLS = READ_ONLY_TOOLS + ALWAYS_SAFE_TOOLS + MUTATION_TOOLS + FLOW_TOOLS

_RESTRICTED_STAGES = {
    DialogStage.AWAITING_SIZE,
    DialogStage.AWAITING_ADDRESS,
    DialogStage.AWAITING_ADDRESS_CONFIRMATION,
    DialogStage.AWAITING_CONFIRMATION,
}


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
    ]


def tools_for_stage(state: DialogState) -> list[dict]:
    """Mutation tools are omitted while a size, choice, address, or confirmation is open."""
    restricted = state.stage in _RESTRICTED_STAGES or bool(state.pending_choices)
    allowed = READ_ONLY_TOOLS + ALWAYS_SAFE_TOOLS
    if not restricted and state.stage in {
        DialogStage.ORDERING,
        DialogStage.AWAITING_MORE_ITEMS,
    }:
        allowed = KNOWN_TOOLS
    names = set(allowed)
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
    state.stage = DialogStage.HANDOFF
    state.reply_streak = 0
    return {"ok": True, "status": "handoff", "speak": HUMAN_HANDOFF_URDU, "data": {"reason": reason}}


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
