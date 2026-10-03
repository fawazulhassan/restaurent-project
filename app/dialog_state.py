"""Typed conversation-control state for the hybrid order dialog."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from app.order import Order, OrderStatus


class DialogStage(str, Enum):
    ORDERING = "ordering"
    AWAITING_SIZE = "awaiting_size"
    AWAITING_ADDRESS = "awaiting_address"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETE = "complete"


class DialogState(BaseModel):
    stage: DialogStage = DialogStage.ORDERING
    last_item_id: str | None = None
    last_item_size: str | None = None
    pending_item_id: str | None = None
    pending_quantity: int = 1
    pending_instructions: str | None = None
    size_attempts: int = 0
    summary_presented: bool = False
    last_path: str = "deterministic"
    last_parser_seconds: float = 0.0
    last_llm_seconds: float = 0.0

    def mark_mutation(self) -> None:
        self.summary_presented = False

    def remember_item(self, item_id: str, size_id: str | None) -> None:
        self.last_item_id = item_id
        self.last_item_size = size_id

    def await_size(
        self,
        item_id: str,
        quantity: int = 1,
        instructions: str | None = None,
    ) -> None:
        if item_id != self.pending_item_id:
            self.size_attempts = 0
        self.stage = DialogStage.AWAITING_SIZE
        self.pending_item_id = item_id
        self.pending_quantity = quantity
        self.pending_instructions = instructions

    def clear_pending_size(self) -> None:
        self.pending_item_id = None
        self.pending_quantity = 1
        self.pending_instructions = None
        self.size_attempts = 0

    def clear_removed_item(self, item_id: str, size_id: str | None) -> None:
        if self.last_item_id == item_id and self.last_item_size == size_id:
            self.last_item_id = None
            self.last_item_size = None

    def sync_stage(self, order: Order) -> None:
        if order.status == OrderStatus.CONFIRMED:
            self.stage = DialogStage.COMPLETE
        elif order.items and order.delivery_address:
            self.stage = DialogStage.AWAITING_CONFIRMATION
        elif order.items:
            self.stage = DialogStage.AWAITING_ADDRESS
        else:
            self.stage = DialogStage.ORDERING


def infer_dialog_state(order: Order) -> DialogState:
    """Compatibility state for callers that have not yet retained a session state."""
    state = DialogState()
    state.sync_stage(order)
    if order.items:
        last = order.items[-1]
        state.remember_item(last.id, last.size)
    if state.stage == DialogStage.AWAITING_CONFIRMATION:
        state.summary_presented = True
    return state
