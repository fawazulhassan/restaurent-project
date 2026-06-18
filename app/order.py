import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

_MENU_CACHE: dict | None = None


class OrderError(Exception):
    pass


class OrderStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    CONFIRMED = "confirmed"


class OrderItem(BaseModel):
    id: str
    size: str | None
    qty: int = Field(ge=1)
    name: str
    name_urdu: str
    size_label: str | None
    unit_price_pkr: int


class Order(BaseModel):
    items: list[OrderItem] = []
    delivery_address: str | None = None
    special_instructions: str | None = None
    customer_phone: str | None = None
    status: OrderStatus = OrderStatus.IN_PROGRESS


def load_menu(path: Path | None = None) -> dict:
    global _MENU_CACHE
    if _MENU_CACHE is not None:
        return _MENU_CACHE

    menu_path = path or Path(__file__).parent.parent / "data" / "menu.json"
    with open(menu_path, encoding="utf-8") as f:
        menu = json.load(f)

    index: dict = {}
    for category in menu.get("categories", []):
        for item in category.get("items", []):
            index[item["id"]] = item

    _MENU_CACHE = index
    return _MENU_CACHE


def validate_item(item_id: str, size_id: str | None) -> dict:
    menu = load_menu()
    item = menu.get(item_id)
    if item is None:
        raise OrderError(f"Item '{item_id}' not found on menu.")

    if not item.get("available", True):
        raise OrderError(f"Item '{item_id}' is not available.")

    sizes = item.get("sizes")
    if sizes:
        if size_id is None:
            raise OrderError(f"Item '{item_id}' requires a size.")
        size = next((s for s in sizes if s["id"] == size_id), None)
        if size is None:
            raise OrderError(f"Size '{size_id}' is not valid for '{item_id}'.")
        return {"item": item, "size": size, "unit_price_pkr": size["price_pkr"]}

    if size_id is not None:
        raise OrderError(f"Item '{item_id}' has no sizes; do not pass a size.")

    return {"item": item, "size": None, "unit_price_pkr": item["price_pkr"]}


def add_item_to_order(
    order: Order, item_id: str, size_id: str | None, qty: int
) -> Order:
    result = validate_item(item_id, size_id)
    size = result["size"]
    size_label = size["label"] if size is not None else None

    for line in order.items:
        if line.id == item_id and line.size == size_id:
            line.qty += qty
            return order

    order.items.append(
        OrderItem(
            id=item_id,
            size=size_id,
            qty=qty,
            name=result["item"]["name"],
            name_urdu=result["item"]["name_urdu"],
            size_label=size_label,
            unit_price_pkr=result["unit_price_pkr"],
        )
    )
    return order


def calculate_total(order: Order) -> int:
    return sum(item.unit_price_pkr * item.qty for item in order.items)


def build_confirmation_urdu(order: Order) -> str:
    """Urdu-script confirmation (for TTS phase later)."""
    parts: list[str] = ["آپ کا آرڈر کنفرم ہو گیا ہے۔"]

    item_parts: list[str] = []
    for item in order.items:
        if item.size_label:
            item_parts.append(f"{item.qty} {item.name_urdu} ({item.size_label})")
        else:
            item_parts.append(f"{item.qty} {item.name_urdu}")

    if item_parts:
        parts.append("، ".join(item_parts) + "،")

    total = calculate_total(order)
    parts.append(f"کل Rs {total}۔")

    if order.delivery_address:
        parts.append(f"پتہ: {order.delivery_address}۔")
    else:
        parts.append("براہ کرم ڈیلیوری کا پتہ بتائیں۔")

    if order.special_instructions:
        parts.append(f"ہدایات: {order.special_instructions}۔")

    return " ".join(parts)


def build_confirmation_english(order: Order) -> str:
    parts: list[str] = ["Your order is confirmed."]

    item_parts: list[str] = []
    for item in order.items:
        if item.size_label:
            item_parts.append(f"{item.qty}x {item.name} ({item.size_label})")
        else:
            item_parts.append(f"{item.qty}x {item.name}")

    if item_parts:
        parts.append(", ".join(item_parts) + ".")

    total = calculate_total(order)
    parts.append(f"Total Rs {total}.")

    if order.delivery_address:
        parts.append(f"Address: {order.delivery_address}.")
    else:
        parts.append("Please share your delivery address.")

    if order.special_instructions:
        parts.append(f"Notes: {order.special_instructions}.")

    return " ".join(parts)


def build_order_status_reply(order: Order) -> str:
    """Factual in-progress order summary — qty and totals come from order state, not the LLM."""
    lines: list[str] = []

    if order.items:
        lines.append("Your order:")
        for item in order.items:
            size = f" ({item.size_label})" if item.size_label else ""
            lines.append(
                f"- {item.qty}x {item.name}{size} — Rs {item.unit_price_pkr * item.qty}"
            )
        lines.append(f"Total: Rs {calculate_total(order)}")
    else:
        lines.append("No items in your order yet.")

    if order.special_instructions:
        lines.append(f"Notes: {order.special_instructions}")

    if not order.delivery_address:
        lines.append("Delivery address bata dein?")
    elif order.status != OrderStatus.CONFIRMED:
        lines.append("Bol dein 'confirm' jab order theek ho.")

    return "\n".join(lines)


def save_order(order: Order, orders_dir: Path | None = None) -> Path:
    out_dir = orders_dir or Path(__file__).parent.parent / "data" / "orders"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"order-{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(order.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    return out_path


def confirm_order(order: Order) -> Order:
    order.status = OrderStatus.CONFIRMED
    return order


def _find_line_index(order: Order, item_id: str, size_id: str | None) -> int | None:
    for i, line in enumerate(order.items):
        if line.id == item_id and line.size == size_id:
            return i
    return None


def update_item_qty(
    order: Order, item_id: str, size_id: str | None, qty: int
) -> Order:
    idx = _find_line_index(order, item_id, size_id)
    if idx is None:
        raise OrderError("Item not in current order.")

    if qty <= 0:
        order.items.pop(idx)
    else:
        order.items[idx].qty = qty
    return order


def remove_item_from_order(
    order: Order, item_id: str, size_id: str | None
) -> Order:
    idx = _find_line_index(order, item_id, size_id)
    if idx is None:
        raise OrderError("Item not in current order.")
    order.items.pop(idx)
    return order


def apply_order_update(order: Order, update: dict) -> Order:
    for entry in update.get("add_items") or []:
        size_id = entry.get("size")
        add_item_to_order(order, entry["id"], size_id, entry["qty"])

    for entry in update.get("set_qty") or []:
        size_id = entry.get("size")
        update_item_qty(order, entry["id"], size_id, entry["qty"])

    for entry in update.get("remove_items") or []:
        size_id = entry.get("size")
        remove_item_from_order(order, entry["id"], size_id)

    if "delivery_address" in update and update["delivery_address"] is not None:
        order.delivery_address = update["delivery_address"]

    if "special_instructions" in update and update["special_instructions"] is not None:
        order.special_instructions = update["special_instructions"]

    if "customer_phone" in update and update["customer_phone"] is not None:
        order.customer_phone = update["customer_phone"]

    if update.get("confirm"):
        if not order.items:
            raise OrderError("Cannot confirm an empty order.")
        if not order.delivery_address:
            raise OrderError("Please provide a delivery address before confirming.")
        confirm_order(order)

    return order


if __name__ == "__main__":
    import sys

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")

    print(
        """Phase 1 — test yourself in Python REPL

  cd restaurant-voice-agent
  .\\venv\\Scripts\\activate
  python

Then paste commands one at a time:

  from app.order import Order, OrderError, add_item_to_order, calculate_total, build_confirmation_urdu, confirm_order, save_order

  order = Order()

  # 1.1 — valid item (expect qty 2, total 2800)
  add_item_to_order(order, "chicken-karahi", "full", 2)
  order.items
  calculate_total(order)

  # 1.2 — invalid item (expect OrderError)
  add_item_to_order(order, "xyz-burger", None, 1)

  # 1.3 — unavailable item (expect OrderError)
  add_item_to_order(order, "test-unavailable-item", None, 1)

  # 1.4 — total check (after 1.1: should be 2800)
  calculate_total(order)

  # 1.5 — Urdu confirmation with address
  order.delivery_address = "Model Town, Kasur, house 45"
  print(build_confirmation_urdu(order))

  # 1.7 — missing address warning
  order2 = Order()
  add_item_to_order(order2, "chicken-karahi", "full", 1)
  print(build_confirmation_urdu(order2))

  # 1.6 — save order
  confirm_order(order)
  save_order(order)
"""
    )
