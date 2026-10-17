"""Validated basket operations; language interpretation never mutates an order."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.order import Order, OrderError, add_item_to_order, change_item_size, update_item_qty, validate_item


class BasketEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["add", "set", "reduce", "remove", "replace_size"]
    item_id: str
    size_id: str | None = None
    new_size_id: str | None = None
    quantity: StrictInt | None = Field(default=None, ge=0)


class BasketEditPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["apply", "clarify"]
    confidence: float = Field(ge=0, le=1)
    edits: list[BasketEdit] = Field(default_factory=list, max_length=20)


def _lines_for(order: Order, item_id: str, size_id: str | None) -> list:
    return [
        line for line in order.items
        if line.id == item_id and (size_id is None or line.size == size_id)
    ]


def _source_key(order: Order, edit: BasketEdit) -> tuple:
    if edit.action == "replace_size":
        if edit.quantity is not None:
            raise OrderError("quantity_not_null")
        lines = _lines_for(order, edit.item_id, None)
        if edit.size_id is None:
            if len(lines) != 1:
                raise OrderError("ambiguous_size_line")
            current = lines[0].size
        else:
            matched = [line for line in lines if line.size == edit.size_id]
            if len(matched) != 1:
                raise OrderError("ambiguous_size_line")
            current = edit.size_id
        if not edit.new_size_id or edit.new_size_id == current:
            raise OrderError("same_size" if edit.new_size_id == current else "unknown_new_size")
        try:
            validate_item(edit.item_id, edit.new_size_id)
        except OrderError as exc:
            raise OrderError("unknown_new_size") from exc
        return ("line", edit.item_id, current)
    if edit.action == "add":
        return ("add", edit.item_id, edit.size_id)
    lines = _lines_for(order, edit.item_id, edit.size_id)
    if len(lines) == 1:
        return ("line", edit.item_id, lines[0].size)
    return ("line", edit.item_id, edit.size_id)


def apply_basket_edits(order: Order, edits: list[BasketEdit]) -> None:
    """Apply the whole plan or none. replace_size runs before quantity edits."""
    if not edits:
        raise OrderError("No basket operations supplied.")
    trial = order.model_copy(deep=True)
    replaces = [edit for edit in edits if edit.action == "replace_size"]
    rest = [edit for edit in edits if edit.action != "replace_size"]
    sequence = [*replaces, *rest]
    seen: set[tuple] = set()
    for edit in sequence:
        key = _source_key(trial, edit)
        if key in seen:
            raise OrderError("same_source_line")
        seen.add(key)
    for edit in replaces:
        current = _source_key(trial, edit)[2]
        change_item_size(trial, edit.item_id, current, edit.new_size_id)
    for edit in rest:
        if edit.action == "add":
            if edit.quantity is None or edit.quantity <= 0:
                raise OrderError("An addition needs a positive quantity.")
            add_item_to_order(trial, edit.item_id, edit.size_id, edit.quantity)
            continue
        lines = _lines_for(trial, edit.item_id, edit.size_id)
        if len(lines) != 1:
            raise OrderError("Specify the existing item and size to change.")
        line = lines[0]
        if edit.action == "remove":
            if edit.quantity is not None:
                raise OrderError("Use reduce for a counted removal.")
            quantity = 0
        elif edit.action == "set":
            if edit.quantity is None:
                raise OrderError("Specify the final quantity.")
            quantity = edit.quantity
        else:
            if edit.quantity is None or edit.quantity <= 0 or edit.quantity > line.qty:
                raise OrderError("Reduction must be positive and cannot exceed the current quantity.")
            quantity = line.qty - edit.quantity
        update_item_qty(trial, line.id, line.size, quantity)
    order.items = trial.items
