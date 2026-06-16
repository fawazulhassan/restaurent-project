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


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
