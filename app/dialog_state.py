"""Typed conversation-control state for the hybrid order dialog."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.order import Order, OrderStatus


class DialogStage(str, Enum):
    ORDERING = "ordering"
    AWAITING_SIZE = "awaiting_size"
    AWAITING_MORE_ITEMS = "awaiting_more_items"
    AWAITING_ADDRESS = "awaiting_address"
    AWAITING_ADDRESS_CONFIRMATION = "awaiting_address_confirmation"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_ORDER_CHANGE = "awaiting_order_change"
    AWAITING_CANCEL_CONFIRM = "awaiting_cancel_confirm"
    COMPLETE = "complete"
    HANDOFF = "handoff"


class HistoryTurn(BaseModel):
    role: Literal["customer", "assistant"]
    text: str


class ItemRequest(BaseModel):
    item_id: str
    size_id: str | None = None
    quantity: int = Field(default=1, ge=1)
    instructions: str | None = None


class PendingChoice(BaseModel):
    item_ids: list[str]   # candidate IDs sharing the group token
    quantity: int = 1


class DialogState(BaseModel):
    stage: DialogStage = DialogStage.ORDERING
    last_item_id: str | None = None
    last_item_size: str | None = None
    pending_item_id: str | None = None
    pending_quantity: int = 1
    pending_instructions: str | None = None
    pending_additions: list[ItemRequest] = Field(default_factory=list)
    quantity_correction_item_id: str | None = None
    quantity_correction_size_id: str | None = None
    quantity_correction_action: str = "set"
    basket_edit_context: list[str] = Field(default_factory=list)
    pending_address: str | None = None
    address_confirmation_attempts: int = 0
    address_rejections: int = 0
    order_confirmation_attempts: int = 0
    order_change_failures: int = 0
    reply_streak: int = 0
    history: list[HistoryTurn] = Field(default_factory=list)
    last_reply: str = ""
    agent_failure_streak: int = 0
    food_repeat_count: int = 0
    food_repeat_question: str = ""
    agent_turns: int = 0
    cancel_resume_stage: DialogStage | None = None
    size_attempts: int = 0
    pending_choices: list[str] = Field(default_factory=list)
    pending_choice_quantity: int = 0
    pending_choice_queue: list[PendingChoice] = Field(default_factory=list)
    choice_attempts: int = 0
    staged_start_index: int = 0
    summary_presented: bool = False
    last_path: str = "deterministic"
    last_parser_seconds: float = 0.0
    last_llm_seconds: float = 0.0
    last_address_extraction_detail: str = ""
    last_basket_edit_detail: str = ""

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

    def clear_pending_choice(self) -> None:
        self.pending_choices = []
        self.pending_choice_quantity = 0
        self.pending_choice_queue = []
        self.choice_attempts = 0
        self.staged_start_index = 0

    def clear_pending_size(self) -> None:
        self.pending_item_id = None
        self.pending_quantity = 1
        self.pending_instructions = None
        self.size_attempts = 0

    def await_address_confirmation(self, address: str) -> None:
        self.pending_address = address
        self.address_confirmation_attempts = 0
        self.stage = DialogStage.AWAITING_ADDRESS_CONFIRMATION

    def confirm_pending_address(self, order: Order) -> str | None:
        if self.pending_address is None:
            return None
        address = self.pending_address
        order.delivery_address = address
        self.pending_address = None
        self.address_confirmation_attempts = 0
        self.address_rejections = 0
        return address

    def clear_pending_address(self) -> None:
        self.pending_address = None
        self.address_confirmation_attempts = 0

    def clear_removed_item(self, item_id: str, size_id: str | None) -> None:
        if self.last_item_id == item_id and self.last_item_size == size_id:
            self.last_item_id = None
            self.last_item_size = None

    def enter_confirmation(self) -> None:
        if self.stage != DialogStage.AWAITING_CONFIRMATION:
            self.order_confirmation_attempts = 0
        self.stage = DialogStage.AWAITING_CONFIRMATION

    def enter_order_change(self) -> None:
        self.order_change_failures = 0
        self.stage = DialogStage.AWAITING_ORDER_CHANGE

    def sync_stage(self, order: Order) -> None:
        if self.stage in {
            DialogStage.HANDOFF,
            DialogStage.AWAITING_ORDER_CHANGE,
            DialogStage.AWAITING_CANCEL_CONFIRM,
        }:
            return
        if self.pending_address:
            self.stage = DialogStage.AWAITING_ADDRESS_CONFIRMATION
        elif order.status == OrderStatus.CONFIRMED:
            self.stage = DialogStage.COMPLETE
            self.clear_pending_choice()
        elif order.items and order.delivery_address:
            self.enter_confirmation()
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
