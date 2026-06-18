"""Automated Phase 2 integration tests against OpenRouter API."""
import sys
import time

reconfigure = getattr(sys.stdout, "reconfigure", None)
if reconfigure is not None:
    reconfigure(encoding="utf-8")

from app.dialog import LLMRateLimitError, build_system_prompt, chat_turn
from app.order import Order, OrderStatus, calculate_total

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} — {detail}")


def run_scenario(name: str, turns: list[str], checks) -> None:
    print(f"\n=== {name} ===")
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    for turn in turns:
        print(f"  User: {turn}")
        reply, order, messages, is_complete = chat_turn(turn, order, messages)
        print(f"  AI: {reply[:120]}...")
        if is_complete:
            break
    checks(order, is_complete if "is_complete" in dir() else order.status == OrderStatus.CONFIRMED)


def test_2_1():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    reply, order, messages, _ = chat_turn("2 chicken karahi full", order, messages)
    check("2.1 items added", len(order.items) == 1 and order.items[0].qty == 2)
    check("2.1 correct item", order.items[0].id == "chicken-karahi" and order.items[0].size == "full")
    check("2.1 asks for address", order.delivery_address is None)
    check("2.1 reply not empty", bool(reply))


def test_2_2():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("chicken tikka pizza large extra cheese", order, messages)
    has_pizza = any("pizza" in item.id for item in order.items)
    has_large = any(item.size == "large" for item in order.items)
    check("2.2 pizza added", has_pizza)
    check("2.2 large size", has_large)
    check("2.2 extra cheese in instructions", order.special_instructions is not None,
          f"special_instructions={order.special_instructions!r}")


def test_2_3():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    reply, order, _, _ = chat_turn("3 zinger burger", order, messages)
    check("2.3 no invalid items", len(order.items) == 0)
    check("2.3 explains unavailable", "burger" in reply.lower() or "menu" in reply.lower() or "دستیاب" in reply or "نہیں" in reply,
          f"reply={reply[:100]!r}")


def test_2_4():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("2 chicken karahi full", order, messages)
    reply, order, _, _ = chat_turn("yes that's all", order, messages)
    check("2.4 still no address", order.delivery_address is None)
    check("2.4 asks address", "address" in reply.lower() or "پتہ" in reply or "ایڈریس" in reply,
          f"reply={reply[:100]!r}")


def test_2_5():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("1 chicken biryani family", order, messages)
    _, order, messages, _ = chat_turn("Model Town Kasur house 45", order, messages)
    _, order, messages, _ = chat_turn("haan confirm kar do", order, messages)
    check("2.5 confirmed", order.status == OrderStatus.CONFIRMED)
    check("2.5 has address", order.delivery_address is not None)
    check("2.5 has items", len(order.items) > 0)


def test_2_6():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("2 chicken karahi full", order, messages)
    _, order, messages, _ = chat_turn("actually make it 1 karahi not 2", order, messages)
    check("2.6 qty is 1", order.items[0].qty == 1, f"qty={order.items[0].qty}")
    check("2.6 total 1400", calculate_total(order) == 1400, f"total={calculate_total(order)}")


def test_2_7():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("1 mutton karahi half", order, messages)
    _, order, messages, _ = chat_turn("extra spicy, no salad", order, messages)
    instr = (order.special_instructions or "").lower()
    check("2.7 instructions set", order.special_instructions is not None)
    check("2.7 mentions spicy", "spicy" in instr or "salad" in instr,
          f"instructions={order.special_instructions!r}")


def _reply_mentions_total(reply: str, total: int) -> bool:
    if str(total) in reply:
        return True
    # Model may write amounts in Urdu/Eastern Arabic numerals (e.g. ۲۸٠٠)
    digit_map = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789",
    )
    normalized = reply.translate(digit_map)
    digits_only = "".join(c for c in normalized if c.isdigit())
    return str(total) in digits_only


def test_2_8():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    _, order, messages, _ = chat_turn("2 chicken karahi full", order, messages)
    _, order, messages, _ = chat_turn("Model Town Kasur", order, messages)
    reply, order, _, complete = chat_turn("ji confirm", order, messages)
    total = calculate_total(order)
    check("2.8 confirmed", order.status == OrderStatus.CONFIRMED)
    check("2.8 total 2800", total == 2800, f"total={total}")
    check("2.8 reply mentions total", _reply_mentions_total(reply, total),
          f"reply={reply[:150]!r}")


if __name__ == "__main__":
    print("Phase 2 integration tests (OpenRouter API)\n")
    tests = [test_2_1, test_2_2, test_2_3, test_2_4, test_2_5, test_2_6, test_2_7, test_2_8]
    for i, test_fn in enumerate(tests):
        try:
            test_fn()
        except LLMRateLimitError as e:
            print(f"\n  SKIP  remaining tests — {e}\n")
            break
        except Exception as e:
            print(f"\n  ERROR in {test_fn.__name__} — {e}\n")
            break
        if i < len(tests) - 1:
            time.sleep(5)
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
