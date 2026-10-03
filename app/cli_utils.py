import sys

from app.order import Order, calculate_total


def configure_stdout() -> None:
    """Configure Python's console output for Urdu and other Unicode text."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_order_summary(order: Order) -> None:
    """Print the verified application order state, not model-generated totals."""
    print("\n--- Order summary ---")
    if not order.items:
        print("  (no items yet)")
    else:
        for item in order.items:
            size = f" ({item.size_label})" if item.size_label else ""
            line_total = item.unit_price_pkr * item.qty
            print(f"  {item.qty}x {item.name}{size} — Rs {line_total}")
        print(f"  Total: Rs {calculate_total(order)}")

    if order.delivery_address:
        print(f"  Address: {order.delivery_address}")
    else:
        print("  Address: (not set)")
    if order.special_instructions:
        print(f"  Instructions: {order.special_instructions}")
    print("---------------------\n")
